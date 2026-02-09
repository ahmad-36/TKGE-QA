
import csv
import json
from typing import Dict, List, Optional, Tuple


def compute_hit_mrr(ranks: List[Optional[int]], k: int) -> Dict[str, float]:

    total = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    rr_sum = sum(1.0 / r for r in ranks if r is not None and r > 0)

    return {
        f"hit@{k}": hits / total if total else 0.0,
        "mrr": rr_sum / total if total else 0.0,
        "hits": hits,
        "total": total,
    }


def write_table(rows: List[Dict], csv_path: str, json_path: str) -> None:
    if not rows:
        raise ValueError("rows is empty")


    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def format_table(rows: List[Dict], float_cols: Tuple[str, ...] = ("Hit@10", "MRR")) -> str:
    if not rows:
        return ""

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    lines = []
    header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        row_strs = []
        for c in cols:
            v = r.get(c, "")
            if c in float_cols and isinstance(v, (float, int)):
                row_strs.append(f"{v:<{widths[c]}.3f}")
            else:
                row_strs.append(f"{str(v):<{widths[c]}}")
        lines.append("  ".join(row_strs))

    return "\n".join(lines)
