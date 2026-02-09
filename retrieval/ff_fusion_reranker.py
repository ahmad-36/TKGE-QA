
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _parse_date_loose(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y":
                return dt.replace(month=1, day=1)
            if fmt == "%Y-%m":
                return dt.replace(day=1)
            return dt
        except ValueError:
            continue
    return None

def _zscore(xs: List[float], eps: float = 1e-8) -> List[float]:
    if not xs:
        return xs
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs), 1)
    s = (v ** 0.5) + eps
    return [(x - m) / s for x in xs]


@dataclass
class FFFusionConfig:
    hidden_dim: int = 16
    use_time_features: bool = False
    device: str = "cuda"


class FFFusionMLP(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim] -> [N]
        return self.net(x).squeeze(-1)


class FFFusionReranker:

    def __init__(
        self,
        encoder: Any,
        tkge: Any,
        cfg: Optional[FFFusionConfig] = None,
        model_state_path: Optional[str] = None,
    ):
        self.encoder = encoder
        self.tkge = tkge
        self.cfg = cfg or FFFusionConfig()


        self.in_dim = 2 + (2 if self.cfg.use_time_features else 0)

        self.device = torch.device(self.cfg.device)
        self.mlp = FFFusionMLP(in_dim=self.in_dim, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.mlp.eval()

        if model_state_path:
            self.load(model_state_path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device)
        self.mlp.load_state_dict(sd)
        self.mlp.eval()

    def save(self, path: str) -> None:
        torch.save(self.mlp.state_dict(), path)

    @torch.no_grad()
    def rerank(
            self,
            query: str,
            candidates: List[Dict[str, Any]],
            top_k: int = 10,
            pool_k: Optional[int] = None,
            anchor_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not candidates:
            return []

        pool_k = int(pool_k or max(top_k, 50))
        pool_k = min(pool_k, len(candidates))

        # 1) semantic pool
        ranked = self.encoder.rerank(query, candidates, top_k=pool_k)

        sem_scores: List[float] = []
        for r in ranked:
            s = r.get("score")
            if s is None:
                # safety fallback
                return ranked[:top_k]
            sem_scores.append(float(s))

        # 2) TKGE scores on the same pool
        tkge_scores = [float(x) for x in self.tkge.score_batch(ranked)]

        # 3) per query norm
        sem_scores_n = _zscore(sem_scores)
        tkge_scores_n = _zscore(tkge_scores)

        # 4) optional time features
        anchor_dt = _parse_date_loose(anchor_timestamp) if anchor_timestamp else None
        has_anchor = 1.0 if anchor_dt else 0.0

        time_diff_days: List[float] = []
        if self.cfg.use_time_features:
            for r in ranked:
                if not anchor_dt:
                    time_diff_days.append(0.0)
                    continue
                c_dt = _parse_date_loose(r.get("date", ""))
                time_diff_days.append(
                    float((c_dt - anchor_dt).days) if c_dt else 0.0
                )

        # 5) build feature matrix (same order as training!)
        feats: List[List[float]] = []
        for i in range(len(ranked)):
            row = [sem_scores_n[i], tkge_scores_n[i]]
            if self.cfg.use_time_features:
                row += [time_diff_days[i], has_anchor]
            feats.append(row)

        x = torch.tensor(feats, dtype=torch.float32, device=self.device)
        fused = self.mlp(x).detach().cpu().tolist()

        # 6) attach fused score and sort
        out = []
        for i, r in enumerate(ranked):
            rr = dict(r)
            rr["score_semantic"] = sem_scores[i]  # raw (for debugging)
            rr["score_tkge"] = tkge_scores[i]  # raw (for debugging)
            rr["score_fused"] = float(fused[i])
            rr["score"] = float(fused[i])  # used for ranking
            out.append(rr)

        out.sort(key=lambda z: z["score"], reverse=True)
        return out[:top_k]

