import json
import re
from typing import Dict, List, Set, Tuple
import spacy

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_MONTHS = {m.title() for m in _MONTH_MAP.keys()}

_STOP = {"Who", "Whom", "What", "Which", "Where", "When", "To", "For", "In", "On"}

_PROPER_NOUN_SPAN = re.compile(
    r"\b(?:[A-Z][\w’'\-]+|[A-Z]\.)"
    r"(?:\s+(?:[A-Z][\w’'\-]+|[A-Z]\.))+"
)
_SINGLE_CAP = re.compile(r"\b[A-Z][a-z][\w’'\-]+\b")

_ROLE_COUNTRY = re.compile(r"((?:[A-Z][a-z]+(?:\s*/\s*|\s+))+)\s*\(([^)]+)\)")
_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_YEAR_IN = re.compile(r"in\s+(\d{4})")
_MONTH_YEAR = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    re.IGNORECASE,
)

class Extractor:
    def __init__(self, spacy_model: str = "en_core_web_sm", allowed_role_heads: Set[str] | None = None):
        self.nlp = spacy.load(spacy_model)
        self.allowed_role_heads = allowed_role_heads

    def extract(self, question: str) -> Dict:
        role_entities, countries_in_roles = self._extract_role_entities(question)
        ner_entities = self._extract_ner(question, countries_in_roles)
        dates = self._extract_dates(question)

        entities = self._dedup(role_entities + ner_entities)

        # non-empty (avoid retrieval collapse)
        if not entities:
            entities = self._fallback_proper_nouns(question)

        return {"entities": entities, "dates": dates}

    def _extract_role_entities(self, text: str) -> Tuple[List[Dict], Set[str]]:
        entities: List[Dict] = []
        countries_in_roles: Set[str] = set()
        for m in _ROLE_COUNTRY.finditer(text):
            role_head = m.group(1).strip()
            country = m.group(2).strip()
            if self.allowed_role_heads and role_head not in self.allowed_role_heads:
                continue
            full = f"{role_head} ({country})"
            entities.append({"name": full, "type": "ICEWS_ROLE", "role": role_head, "country": country})
            countries_in_roles.add(country.lower())
        return entities, countries_in_roles

    def _extract_ner(self, text: str, exclude_countries: Set[str]) -> List[Dict]:
        doc = self.nlp(text)
        out: List[Dict] = []
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text.lower() in exclude_countries:
                continue
            if ent.label_ == "PERSON":
                out.append({"name": ent.text, "type": "LEADER"})
            elif ent.label_ == "GPE":
                out.append({"name": ent.text, "type": "COUNTRY"})
            elif ent.label_ == "ORG":
                out.append({"name": ent.text, "type": "ORG"})
        return out

    def _extract_dates(self, text: str) -> List[Dict]:
        dates: List[Dict] = []
        seen = set()

        for d in _ISO_DATE.findall(text):
            if d not in seen:
                seen.add(d)
                dates.append({"date": d, "format": "iso"})

        for month, year in _MONTH_YEAR.findall(text):
            d = f"{year}-{_MONTH_MAP[month.lower()]}"
            if d not in seen:
                seen.add(d)
                dates.append({"date": d, "format": "month_year"})

        for y in _YEAR_IN.findall(text):
            if y not in seen:
                seen.add(y)
                dates.append({"date": y, "format": "year"})
        return dates

    def _fallback_proper_nouns(self, text: str) -> List[Dict]:
        out: List[Dict] = []
        seen = set()

        for m in _PROPER_NOUN_SPAN.finditer(text):
            span = m.group(0).strip()
            if span in _STOP or span in _MONTHS:
                continue
            key = span.lower()
            if key not in seen:
                seen.add(key)
                out.append({"name": span, "type": "HEURISTIC_PROPER_NOUN"})

        if not out:
            for m in _SINGLE_CAP.finditer(text):
                tok = m.group(0).strip()
                if tok in _STOP or tok in _MONTHS:
                    continue
                key = tok.lower()
                if key not in seen:
                    seen.add(key)
                    out.append({"name": tok, "type": "HEURISTIC_SINGLE_CAP"})
                    break
        return out

    @staticmethod
    def _dedup(entities: List[Dict]) -> List[Dict]:
        out, seen = [], set()
        for e in entities:
            key = e.get("name", "").lower()
            if key and key not in seen:
                seen.add(key)
                out.append(e)
        return out

_extractor = Extractor()

def extract(question: str) -> Dict:
    return _extractor.extract(question)

