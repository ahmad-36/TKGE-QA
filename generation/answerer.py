
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set


_YEAR_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm(s: str) -> str:
    return (s or "").strip()


def _is_date_like(s: str) -> bool:
    ss = _norm(s)
    return bool(_YEAR_RE.match(ss) or _MONTH_RE.match(ss) or _ISO_RE.match(ss))


def _question_intent(q: str) -> str:
    """Coarse intent: when / who / which / other."""
    ql = _norm(q).lower()

    # Temporal questions (allowed to return a date)
    if ql.startswith("when "):
        return "when"
    if "what year" in ql or "which year" in ql or "in what year" in ql or "in which year" in ql:
        return "when"
    if "what month" in ql or "which month" in ql or "in what month" in ql or "in which month" in ql:
        return "when"
    if "what date" in ql or "which date" in ql or "on what date" in ql or "on which date" in ql:
        return "when"

    # Entity questions
    if ql.startswith("who "):
        return "who"
    if ql.startswith("which "):
        return "which"

    return "other"


def _date_granularity(q: str) -> str:
    ql = _norm(q).lower()
    if "year" in ql:
        return "year"
    if "month" in ql:
        return "month"
    return "date"


def _format_date(date_str: str, granularity: str) -> str:
    """Format YYYY-MM-DD to year/month/date """
    ds = _norm(date_str)
    if not ds:
        return ""

    if granularity == "year":
        if _ISO_RE.match(ds) or _MONTH_RE.match(ds):
            return ds[:4]
        if _YEAR_RE.match(ds):
            return ds
        return ds[:4]

    if granularity == "month":
        if _ISO_RE.match(ds):
            return ds[:7]
        if _MONTH_RE.match(ds):
            return ds
        if _YEAR_RE.match(ds):
            return ds
        return ds[:7]

    return ds


def _allowed_entities(top_triples: Sequence[Dict[str, Any]]) -> Set[str]:
    allowed: Set[str] = set()
    for t in top_triples:
        h = _norm(str(t.get("head", "")))
        ta = _norm(str(t.get("tail", "")))
        if h and not _is_date_like(h):
            allowed.add(h)
        if ta and not _is_date_like(ta):
            allowed.add(ta)
    return allowed


def _pick_entity_from_triple(best: Dict[str, Any], mentioned: Sequence[str], prefer_head: bool) -> str:

    head = _norm(str(best.get("head", "")))
    tail = _norm(str(best.get("tail", "")))

    # Never return date-like strings as entities
    if _is_date_like(head):
        head = ""
    if _is_date_like(tail):
        tail = ""

    mentioned_set = {_norm(x) for x in (mentioned or []) if _norm(x)}

    # Prefer the "other" side relative to what the question already mentions
    if head and head not in mentioned_set and tail in mentioned_set:
        return head
    if tail and tail not in mentioned_set and head in mentioned_set:
        return tail

    return (head or tail) if prefer_head else (tail or head)


@dataclass
class Answerer:
    """Rule-compliant answerer.

    Rules:
    - Only temporal questions ("when/what year/month/date") may return dates.
    - All other questions must return an entity from {head, tail} of top_triples.
    - Dates must never be returned as a fallback for non-temporal questions.
    """

    prefer_head_for_who: bool = True

    def answer(
        self,
        question: str,
        top_triples: List[Dict[str, Any]],
        extracted_entities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not top_triples:
            return {"answer": None, "answer_type": "none", "evidence": None}

        best = top_triples[0]
        intent = _question_intent(question)
        mentioned = extracted_entities or []

        # Temporal answers are only permitted for temporal questions.
        if intent == "when":
            gran = _date_granularity(question)
            date_str = _format_date(str(best.get("date", "")), gran)
            return {"answer": date_str or None, "answer_type": f"date_{gran}", "evidence": best}

        # Non-temporal: must be an entity from head/tail of provided triples.
        allowed = _allowed_entities(top_triples)

        if intent == "who":
            candidate = _pick_entity_from_triple(best, mentioned, prefer_head=self.prefer_head_for_who)
        elif intent == "which":
            candidate = _pick_entity_from_triple(best, mentioned, prefer_head=True)
        else:
            candidate = _pick_entity_from_triple(best, mentioned, prefer_head=False)

        candidate = _norm(candidate)
        if candidate and candidate in allowed:
            return {"answer": candidate, "answer_type": "entity", "evidence": best}

        # Fallback 1: deterministic safe choice from best triple (tail then head)
        for fb in (_norm(str(best.get("tail", ""))), _norm(str(best.get("head", "")))):
            if fb and not _is_date_like(fb) and fb in allowed:
                return {"answer": fb, "answer_type": "entity", "evidence": best}

        # Fallback 2: pick any allowed entity (deterministic)
        if allowed:
            fb = sorted(allowed)[0]
            return {"answer": fb, "answer_type": "entity", "evidence": best}
        return {"answer": None, "answer_type": "none", "evidence": best}

