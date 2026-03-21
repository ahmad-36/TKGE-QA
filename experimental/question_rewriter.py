import re
from typing import Dict, List, Optional, Tuple


TEMPORAL_PATTERNS_ORDERED = [
    ("at_the_time", r"\b[Aa]t\s+the\s+time\s+(?:when\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("following",   r"\b[Ff]ollowing\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("after",       r"\b[Aa]fter\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("before",      r"\b[Bb]efore\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("during",      r"\b[Dd]uring\s+(?:the\s+)?(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("once",        r"\b[Oo]nce\s+(.+?)(?=,|\s+who|\s+which|\s+what)"),
    ("when",        r"\b[Ww]hen\s+(.+?)(?=,|\s+who|\s+which|\s+what)"),
]

TEMPORAL_PATTERNS = {k: v for k, v in TEMPORAL_PATTERNS_ORDERED}


class QuestionRewriter:

    def __init__(self, retriever):
        self.retriever = retriever

    def detect_temporal_signal(self, question: str) -> Tuple[Optional[str], Optional[str]]:

        for signal_type, pattern in TEMPORAL_PATTERNS_ORDERED:
            match = re.search(pattern, question)
            if match:
                anchor = match.group(1).strip() #only what’s inside (.+?), group(0) full matched span
                return signal_type, anchor
        return None, None

    def extract_entities_from_anchor(self, anchor: str) -> List[str]:
        entities: List[str] = []

        role_pattern = r"([A-Z][a-zA-Z\s]+?)\s*\(([^)]+)\)"
        role_matches = re.findall(role_pattern, anchor)

        role_spans = []
        has_full_name = False

        for m in re.finditer(role_pattern, anchor):
            role_spans.append((m.start(), m.end()))

        for role, country in role_matches:
            role_ent = f"{role.strip()} ({country})"
            if role_ent not in entities:
                entities.append(role_ent)
            if country not in entities:
                entities.append(country)

        role_stop = {"Member", "Judiciary", "Government", "President", "Minister", "Police", "Army"}


        name_pattern = r"\b([A-Z][a-z]+(?:\s+(?:[A-Z]\.)?[-']?[A-Za-z]+)*(?:\s+(?:al|el|bin|ibn|van|von|de|da|di|le|la)\s+[A-Z][a-z]+)*)\b"

        for m in re.finditer(name_pattern, anchor):
            name = m.group(1)
            if name in role_stop:
                continue
            inside_role = any(s <= m.start() < e for s, e in role_spans)
            if inside_role:
                continue
            # prefer multi-token person names; avoid single-token fragments
            if len(name) <= 2:
                continue

            if " " in name or "-" in name:  # full name / hyphenated
                if name not in entities:
                    entities.append(name)
                has_full_name = True
            else:
                # only keep single-token names if we didn't find any full names
                if not has_full_name and name not in entities:
                    entities.append(name)

        #simple normalization variants
        norm = []
        for e in entities:
            e2 = e.replace("'s", "").strip()
            if e2 != e:
                norm.append(e2)
        for x in norm:
            if x not in entities:
                entities.append(x)

        return entities

    def _apply_template(self, question: str, signal_type: str, timestamp: str) -> str:
        year = (timestamp or "")[:4]
        if not year.isdigit():
            return question
        return question.rstrip("?") + f" (Time: {year})?"

    def find_anchor_timestamp(
            self,
            anchor_entities: List[str],
            anchor_phrase: str,
    ) -> Tuple[Optional[str], float]:

        #return timestamp, confidence (based on best sore and second best)

        if not anchor_entities:
            return None, 0.0

        candidates = self.retriever.retrieve(anchor_entities) # extract entities from that anchor, anchor_entities ["Gov (France)", "France", "China"]
        if not candidates:
            return None, 0.0

        ctx = (anchor_phrase or "").lower()

        SYN = {
            "praise": ["praise", "praised", "offer praise", "offered praise", "commend", "laud", "hail"],
            "endorse": ["endorse", "endorsed", "back", "support"],
            "consult": ["consult", "consulted", "consultation"],
            "appeal": ["appeal", "appealed", "request", "requested", "call for", "urge"],
            "threaten": ["threaten", "threatened", "threat", "warn", "warning"],
            "reject": ["reject", "rejected", "deny", "denied", "refuse", "refused"],
            "visit": ["visit", "visited", "travel", "traveled", "trip"],
            "meet": ["meet", "met", "meeting", "talk", "talks"],
            "criticize": ["criticize", "criticised", "criticized", "condemn", "condemned"],
            "negotiate": ["negotiate", "negotiated", "negotiations", "bargain"],
            "host": ["host", "hosted"],
        }

        trigger_to_canon = []
        for canon, triggers in SYN.items():
            for t in triggers:
                trigger_to_canon.append((t, canon))

        best_fact = None
        best_score = -1
        second_best_score = -1

        for fact in candidates[:50]:
            score = 0
            relation = (fact.get("relation", "") or "").lower()
            head = (fact.get("head", "") or "").lower()
            tail = (fact.get("tail", "") or "").lower()

            # (A) anchor-text ⇄ relation match (with synonyms)
            for trig, canon in trigger_to_canon:
                if trig in ctx:
                    if canon in relation or trig in relation:
                        score += 3

            # (B) entity alignment: anchor entities appearing in head/tail
            for ent in anchor_entities:
                ent_l = ent.lower()
                if ent_l in head:
                    score += 1
                if ent_l in tail:
                    score += 1

            # Maintain best and second best
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_fact = fact
            elif score > second_best_score:
                second_best_score = score

        # Hard minimum evidence: relation match(3) + >=2 entity matches => >=5
        if not best_fact or best_score < 5:
            return None, 0.0

        # Ambiguity gate: if runner-up is too close, abstain
        margin = best_score - max(second_best_score, 0)

        # Confidence: combine absolute score and margin.
        # - score_conf: how far above threshold 5 we are (cap at +4 => 1.0)
        # - margin_conf: margin >=3 => 1.0, margin 0 => 0.0
        score_conf = min(max((best_score - 5) / 4.0, 0.0), 1.0)
        margin_conf = min(max(margin / 3.0, 0.0), 1.0)
        conf = 0.5 * score_conf + 0.5 * margin_conf

        # Conservative abstention: require margin >= 2 OR very strong absolute evidence
        if margin < 2 and best_score < 7:
            return None, conf

        return best_fact.get("date"), conf

    def rewrite(self, question: str) -> Dict:
        result = {
            "original": question,
            "rewritten": question,
            "signal_type": None,
            "anchor_phrase": None,
            "anchor_entities": [],
            "anchor_timestamp": None,
            "was_rewritten": False,
            "constraints": {
                "temporal_operator": "NONE",  # AFTER / BEFORE / DURING / NONE
                "anchor_phrase": None,
                "anchor_timestamp": None,
                "inferred_year": None,
                "confidence": 0.0,
            },
        }

        signal_type, anchor_phrase = self.detect_temporal_signal(question)
        if not signal_type or not anchor_phrase:
            return result

        result["signal_type"] = signal_type
        result["anchor_phrase"] = anchor_phrase

        op_map = {
            "after": "AFTER",
            "following": "AFTER",
            "once": "AFTER",
            "before": "BEFORE",
            "during": "DURING",
            "when": "DURING",
            "at_the_time": "DURING",
        }
        temporal_operator = op_map.get(signal_type, "NONE")
        result["constraints"]["temporal_operator"] = temporal_operator
        result["constraints"]["anchor_phrase"] = anchor_phrase

        anchor_entities = self.extract_entities_from_anchor(anchor_phrase)
        result["anchor_entities"] = anchor_entities
        if not anchor_entities:
            return result

        # UPDATED: find_anchor_timestamp now returns (timestamp, confidence)
        timestamp, conf = self.find_anchor_timestamp(anchor_entities, anchor_phrase)
        result["constraints"]["confidence"] = float(conf)

        if not timestamp:
            # Abstain: keep anchor_timestamp None so pipeline falls back safely
            return result

        result["anchor_timestamp"] = timestamp
        result["constraints"]["anchor_timestamp"] = timestamp

        year = (timestamp or "")[:4]
        result["constraints"]["inferred_year"] = int(year) if year.isdigit() else None

        # Keep your existing rewrite behavior unchanged
        rewritten = self._apply_template(question, signal_type, timestamp)
        result["rewritten"] = rewritten
        result["was_rewritten"] = (rewritten != question)
        return result