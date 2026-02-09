
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from pipeline import TKGQAPipeline

from retrieval.ff_fusion_reranker import _parse_date_loose


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list, got {type(data)}")
    return data


def get_question(ex: Dict[str, Any], mode: str) -> str:
    key = "question_implicit" if mode == "implicit" else "question_explicit"
    q = ex.get(key)
    if not q or not isinstance(q, str):
        raise KeyError(f"Missing '{key}' in example. Keys: {list(ex.keys())}")
    return q.strip()


def get_gold(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    quad = ex.get("quadruple")
    if not isinstance(quad, dict):
        return None
    if not all(k in quad for k in ("s", "p", "o", "t")):
        return None
    return {
        "head": str(quad["s"]),
        "relation": str(quad["p"]),
        "tail": str(quad["o"]),
        "date": str(quad["t"]),
    }


def match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    return (
        str(pred.get("head", "")) == gold["head"]
        and str(pred.get("relation", "")) == gold["relation"]
        and str(pred.get("tail", "")) == gold["tail"]
        and str(pred.get("date", "")) == gold["date"]
    )


def build_features(
    sem_scores: List[float],
    tkge_scores: List[float],
    cand_dates: List[str],              # kept for API compatibility
    anchor_timestamp: Optional[str],    # kept for API compatibility
) -> torch.Tensor:
    # 2D feature design for single-event fusion: [sem_z, tkge_z]
    # (time features removed to avoid noisy shortcuts + dimension mismatch)
    feats = [[float(s), float(g)] for s, g in zip(sem_scores, tkge_scores)]
    return torch.tensor(feats, dtype=torch.float32)


def _zscore(xs: List[float], eps: float = 1e-8) -> List[float]:
    if not xs:
        return xs
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs), 1)
    s = (v ** 0.5) + eps
    return [(x - m) / s for x in xs]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="official_QA_dev_split.json")
    ap.add_argument("--question_mode", default="implicit", choices=["implicit", "explicit"])
    ap.add_argument("--implicit_graph", default="implicit_relation_graph.json")
    ap.add_argument("--icews", default="icews_2014_train.txt")
    ap.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--retriever_cap", type=int, default=1000)
    ap.add_argument("--time_tolerance", type=int, default=30)

    ap.add_argument("--tkge_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--pool_k", type=int, default=50)   # encoder pool for training
    ap.add_argument("--max_train", type=int, default=0) # 0 = all

    ap.add_argument("--save_path", default="checkpoints/ff_fusion_mlp_2d.pt")
    args = ap.parse_args()


    def abs_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    data = load_dataset(abs_path(args.dataset))
    implicit_path = abs_path(args.implicit_graph)
    icews_path = abs_path(args.icews)


    pipeline = TKGQAPipeline(
        implicit_graph_path=implicit_path,
        icews_path=icews_path,
        encoder_model_name=args.encoder,
        time_tolerance_days=args.time_tolerance,
        device=args.device,
        retriever_cap=args.retriever_cap,
        tkge_dir=args.tkge_dir,
    )

    if pipeline.fusion_reranker is None:
        raise RuntimeError("fusion_reranker is None. Ensure tkge_dir is set and fusion reranker is constructed.")

    mlp = pipeline.fusion_reranker.mlp
    print("fusion in_dim:", pipeline.fusion_reranker.in_dim)
    print("mlp first layer weight shape:", tuple(mlp.net[0].weight.shape))

    mlp.train()
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    n_used = 0
    for ep in range(args.epochs):
        total_loss = 0.0
        steps = 0

        for ex in data:
            if args.max_train and n_used >= args.max_train:
                break

            q = get_question(ex, args.question_mode)
            gold = get_gold(ex)
            if gold is None:
                continue


            proc = pipeline.process(
                question=q,
                encoder_top_k=10,
                rerank_cap=200,
                use_rewriter=False,
                use_implicit=True,
                use_time_filter=True,
                use_reranker=False,
                single_event = True,
                gold_time_anchor=ex["temporalSignal"]["time_anchor"],
                single_event_time_mode="around",

            )

            anchor_ts = ex["temporalSignal"]["time_anchor"]  # not proc["anchor_timestamp"]

            filtered = proc["filtered_candidates"]
            if not filtered:
                continue

            # 3) Encoder pool
            pool = pipeline.encoder.rerank(q, filtered, top_k=args.pool_k)
            sem_scores = [float(c["score"]) for c in pool]

            # 4) TKGE scores on same pool
            tkge_scores = [float(x) for x in pipeline.tkge.score_batch(pool)]
            dates = [str(c.get("date", "")) for c in pool]

            # per-query normalization (pool-level)
            sem_scores = _zscore(sem_scores)
            tkge_scores = _zscore(tkge_scores)

            X = build_features(sem_scores, tkge_scores, dates, anchor_ts).to(args.device)
            scores = mlp(X)  # [N]

            # find gold index in pool
            pos_idx = None
            for i, cand in enumerate(pool):
                if match(cand, gold):
                    pos_idx = i
                    break
            if pos_idx is None:
                continue  # gold not in pool; skip

            # Pairwise logistic loss: pos should beat every neg
            pos = scores[pos_idx]
            neg = torch.cat([scores[:pos_idx], scores[pos_idx + 1 :]], dim=0)
            if neg.numel() == 0:
                continue

            loss = -F.logsigmoid(pos - neg).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1
            n_used += 1

        print(f"epoch={ep+1} steps={steps} avg_loss={(total_loss/max(steps,1)):.4f}")

    # Save learned MLP weights for pipeline loading
    pipeline.fusion_reranker.save(args.save_path)
    print("Saved fusion MLP checkpoint to:", args.save_path)


if __name__ == "__main__":
    main()
