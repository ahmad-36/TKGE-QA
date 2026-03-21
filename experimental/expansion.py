from __future__ import annotations

import json
import re
from typing import Dict, List, Any


_ROLE_COUNTRY_PATTERN = re.compile(r"^.+\(([^)]+)\)$")


def load_implicit_graph(path: str) -> Dict[str, str]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup: Dict[str, str] = {}
    for rel in data.get("relations", []):
        if rel.get("type") == "affiliated_with" and "src" in rel and "dst" in rel:
            lookup[rel["src"]] = rel["dst"]
    return lookup



def expand_entities_pattern_based(entities: List[Dict[str, Any]], static_lookup: Dict[str, str]) -> List[str]:
    expanded: List[str] = []
    seen = set()

    for ent in entities:
        name = ent.get("name")
        if not name:
            continue

        if name not in seen:
            expanded.append(name)
            seen.add(name)

        # FIX: Only expand via static_lookup
        affiliated = static_lookup.get(name)
        if affiliated and affiliated not in seen:
            expanded.append(affiliated)
            seen.add(affiliated)
        
    return expanded

def expand_entities_static_only(entities: List[Dict[str, Any]], static_lookup: Dict[str, str]) -> List[str]:

    expanded: List[str] = []
    seen = set()

    for ent in entities:
        name = ent.get("name")
        if not name:
            continue

        if name not in seen:
            expanded.append(name)
            seen.add(name)

        affiliated = static_lookup.get(name)
        if affiliated and affiliated not in seen:
            expanded.append(affiliated)
            seen.add(affiliated)

    return expanded

