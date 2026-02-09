
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _load_map(path: str | Path) -> Dict[str, int]:
    m: Dict[str, int] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item, idx = line.rstrip("\n").split("\t")
            m[item] = int(idx)
    return m


class TTransE(nn.Module):

    def __init__(self, nrRelations: int, nrEntities: int, nrTimes: int, dimEmbedding: int):
        super().__init__()
        self.entities = nn.Embedding(nrEntities, dimEmbedding)
        self.relations = nn.Embedding(nrRelations, dimEmbedding)
        self.times = nn.Embedding(nrTimes, dimEmbedding)

    def forward(self, s_id: torch.Tensor, r_id: torch.Tensor, o_id: torch.Tensor, t_id: torch.Tensor) -> torch.Tensor:
        s = self.entities(s_id)
        r = self.relations(r_id)
        o = self.entities(o_id)
        tt = self.times(t_id)
        # higher is better
        return -torch.norm(s + r + tt - o, p=2, dim=1)


@dataclass
class TKGEConfig:
    ckpt_path: str
    entity2id_path: str
    relation2id_path: str
    time2id_path: str
    device: str = "cuda"


class TKGEScorer:

    def __init__(self, cfg: TKGEConfig):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.entity2id = _load_map(cfg.entity2id_path)
        self.relation2id = _load_map(cfg.relation2id_path)
        self.time2id = _load_map(cfg.time2id_path)

        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        nrE = int(ckpt["nrEntities"])
        nrR = int(ckpt["nrRelations"])
        nrT = int(ckpt["nrTimes"])
        dim = int(ckpt["dimEmbedding"])

        self.model = TTransE(nrR, nrE, nrT, dim)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _to_ids(self, cand: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        h = str(cand.get("head", ""))
        r = str(cand.get("relation", ""))
        o = str(cand.get("tail", ""))
        t = str(cand.get("date", ""))

        if h not in self.entity2id or o not in self.entity2id or r not in self.relation2id or t not in self.time2id:
            return None
        return (self.entity2id[h], self.relation2id[r], self.entity2id[o], self.time2id[t])

    @torch.no_grad()
    def score_batch(self, candidates: List[Dict[str, Any]], default_oov: float = -1e9) -> List[float]:
        # Gather valid ids
        ids: List[Optional[Tuple[int, int, int, int]]] = [self._to_ids(c) for c in candidates]
        valid_idx = [i for i, x in enumerate(ids) if x is not None]

        # Default all scores to very low for OOV
        out = [default_oov] * len(candidates)
        if not valid_idx:
            return out

        s_ids = torch.tensor([ids[i][0] for i in valid_idx], dtype=torch.long, device=self.device)
        r_ids = torch.tensor([ids[i][1] for i in valid_idx], dtype=torch.long, device=self.device)
        o_ids = torch.tensor([ids[i][2] for i in valid_idx], dtype=torch.long, device=self.device)
        t_ids = torch.tensor([ids[i][3] for i in valid_idx], dtype=torch.long, device=self.device)

        scores = self.model(s_ids, r_ids, o_ids, t_ids).detach().cpu().tolist()
        for j, i in enumerate(valid_idx):
            out[i] = float(scores[j])
        return out
