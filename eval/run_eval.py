import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch.cuda


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from pipeline import TKGQAPipeline


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


def compute_metrics(ranks: List[Optional[int]], k: int) -> Dict[str, float]:
    n = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    rr = sum(1.0 / r for r in ranks if r is not None and r > 0)
    return {
        "hit@k": hits / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
        "n": n,
        "hits": hits,
    }


def get_rank(results: Dict[str, Any], gold: Dict[str, Any], top_k: int) -> Optional[int]:
    found = None
    for rank, cand in enumerate(results.get("final_triples", [])[:top_k], start=1):
        if match(cand, gold):
            found = rank
            break
    return found

def evaluate(
    pipeline: TKGQAPipeline,
    data: List[Dict[str, Any]],
    question_mode: str,
    top_k: int = 10,
    pool_k: Optional[int] = None,
    rerank_cap: int = 200,
    use_rewriter: bool = True,
    use_implicit: bool = True,
    use_time_filter: bool = True,
    use_reranker: bool = True,
    print_answers: bool = True,
    max_print: int = 20,
    debug_first_n: int = 0,
    use_gold_fallback: bool = False,
) -> Dict[str, float]:

    ranks: List[Optional[int]] = []

    # pool size for reranker (fusion/encoder pooling), separate from final top_k
    rerank_pool = int(pool_k) if pool_k is not None else max(50, int(top_k))

    def relation_to_mode(rel: str) -> str:
        r = (rel or "").strip().lower()
        if r in {"after", "following", "once"}:
            return "after"
        if r in {"before"}:
            return "before"
        if r in {"when", "during", "at_the_time", "at the time"}:
            return "around"
        return "around"

    for i, ex in enumerate(data, start=1):
        question = get_question(ex, question_mode)
        gold = get_gold(ex)

        ts = ex.get("temporalSignal") or {}
        time_anchor = ts.get("time_anchor")
        rel = ts.get("relation", "")
        single_event_mode = relation_to_mode(rel)

        gold_entities: List[str] = []
        if gold is not None:
            gold_entities = [gold["head"], gold["tail"]]


        results = pipeline.process(
            question=question,
            encoder_top_k=top_k,                 # final list size = 10
            rerank_cap=rerank_cap,               # candidate cap after time filter
            rerank_pool_k=rerank_pool,           # fusion/encoder pool size
            use_rewriter=False,
            use_implicit=True,
            use_time_filter=use_time_filter,
            use_reranker=use_reranker,
            single_event=True,
            gold_time_anchor=time_anchor,
            single_event_time_mode=single_event_mode,
        )

        # 2) debug (gold fallback: only if extractor returns none)
        extracted = results.get("extracted_entities") or []
        if use_gold_fallback and (not extracted) and gold_entities:
            results = pipeline.process(
                question=question,
                encoder_top_k=top_k,
                rerank_cap=rerank_cap,
                rerank_pool_k=rerank_pool,
                use_rewriter=False,
                use_implicit=True,
                use_time_filter=use_time_filter,
                use_reranker=use_reranker,
                single_event=True,
                gold_time_anchor=time_anchor,
                single_event_time_mode=single_event_mode,
                override_entities=gold_entities,
            )

        # Diagnostics
        if debug_first_n and i <= debug_first_n:
            filtered_cands = results.get("filtered_candidates") or []
            final_triples = results.get("final_triples") or []
            retrieved_n = int(results.get("retrieved_candidates") or 0)
            after_time_n = int(results.get("after_time_filter") or 0)
            gold_in_filtered = any(match(c, gold) for c in filtered_cands) if gold else False
            gold_in_final = any(match(c, gold) for c in final_triples) if gold else False

            print("[dbg ents]", results.get("extracted_entities"))
            print(
                f"[dbg {i}] rel={rel} mode={single_event_mode} retrieved={retrieved_n} after_time={after_time_n} "
                f"filtered={len(filtered_cands)} gold_in_filtered={gold_in_filtered} gold_in_final={gold_in_final} "
                f"time_anchor={time_anchor}"
            )

        
        if print_answers and i <= max_print:
            print(f"\n[{i}] Q: {question}")

            gold_answer = ex.get("answer")
            if gold_answer is not None:
                print(f"    Gold answer (dataset): {gold_answer}")

            if gold is not None:
                print(f"    Gold quadruple: {gold['head']} | {gold['relation']} | {gold['tail']} | {gold['date']}")

            print(f"    Pred answer (top-1): {results.get('answer')}")
            ev = results.get("answer_evidence") or {}
            if ev:
                print(
                    f"    Pred evidence: {ev.get('head')} | {ev.get('relation')} | {ev.get('tail')} | {ev.get('date')}")

            # top k-list and gold rank
            final_triples = results.get("final_triples") or []
            if gold is not None:
                gold_rank = get_rank(results, gold, top_k)
                print(f"    Gold rank@{top_k}: {gold_rank if gold_rank is not None else 'NOT IN TOP-K'}")

            print(f"    Top-{top_k}:")
            for r, cand in enumerate(final_triples[:top_k], start=1):
                h = cand.get("head");
                rel = cand.get("relation");
                t = cand.get("tail");
                d = cand.get("date")
                marker = ""
                if gold is not None and match(cand, gold):
                    marker = "  <== GOLD"
                print(f"      {r:>2}. {h} | {rel} | {t} | {d}{marker}")

        if gold is None:
            ranks.append(None)
            continue

        found = get_rank(results, gold, top_k)
        ranks.append(found)

    return compute_metrics(ranks, k=top_k)


def run_ablation(
    pipeline: TKGQAPipeline,
    data: List[Dict[str, Any]],
    question_mode: str,
    top_k: int = 10,
    pool_k: Optional[int] = None,
    rerank_cap: int = 200,
    use_gold_fallback: bool = False,
) -> None:

    configs = [
        ("TimeFilter=ON, Rerank=ON", True, True),
        ("TimeFilter=OFF, Rerank=ON", False, True),
        ("TimeFilter=ON, Rerank=OFF", True, False),
    ]

    print(f"\nAblation (n={len(data)}, k={top_k})")
    print(f"{'Config':<28} {'Hit@k':>8} {'MRR':>8} {'Hits':>6}")

    for name, time_on, rerank_on in configs:
        m = evaluate(
            pipeline,
            data,
            question_mode,
            top_k=top_k,
            pool_k=pool_k,
            rerank_cap=rerank_cap,
            use_time_filter=time_on,
            use_reranker=rerank_on,
            print_answers=False,
            use_gold_fallback=use_gold_fallback,  # <-- THIS is the only needed addition
        )
        print(f"{name:<28} {m['hit@k']:>8.3f} {m['mrr']:>8.3f} {m['hits']:>6}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="official_QA_dev_split.json")
    parser.add_argument("--question_mode", default="implicit", choices=["implicit", "explicit"])

    parser.add_argument("--implicit_graph", default="implicit_relation_graph.json")
    parser.add_argument("--icews", default="icews_2014_train.txt")

    parser.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--retriever_cap", type=int, default=1000)
    parser.add_argument("--time_tolerance", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--rerank_cap", type=int, default=200)
    parser.add_argument("--print_answers", action="store_true")
    parser.add_argument("--max_print", type=int, default=20)
    parser.add_argument("--pool_k", type=int, default=None,
                        help="Reranker pool size (encoder_top_k).  Example: --pool_k 200. If unset, uses --top_k.")

    # TKGE
    parser.add_argument(
        "--tkge_dir",
        default=None,
        help="Path to TKGE artifact folder containing checkpoint + entity2id/relation2id/time2id.",
    )

    parser.add_argument(
        "--run_both",
        action="store_true",
        help="Run encoder-only and hybrid (encoder+TKGE) sequentially.",
    )

    parser.add_argument(
        "--use_gold_fallback",
        action="store_true",
        help="If set, inject gold entities when extractor returns none (oracle fallback).",
    )

    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(REPO_ROOT, dataset_path)

    implicit_path = args.implicit_graph
    if not os.path.isabs(implicit_path):
        implicit_path = os.path.join(REPO_ROOT, implicit_path)

    icews_path = args.icews
    if not os.path.isabs(icews_path):
        icews_path = os.path.join(REPO_ROOT, icews_path)

    data = load_dataset(dataset_path)

    def make_pipeline(tkge_dir: Optional[str]):
        return TKGQAPipeline(
            implicit_graph_path=implicit_path,
            icews_path=icews_path,
            encoder_model_name=args.encoder,
            time_tolerance_days=args.time_tolerance,
            device=args.device,
            retriever_cap=args.retriever_cap,
            tkge_dir=tkge_dir,

        )

    def run_all(pipeline: TKGQAPipeline, title: str):
        print(title)

        if args.print_answers:
            print("\nSample Outputs")
            evaluate(
                pipeline,
                data,
                args.question_mode,
                top_k=args.top_k,
                pool_k=args.pool_k,
                rerank_cap=args.rerank_cap,
                print_answers=True,
                max_print=args.max_print,
                use_gold_fallback=args.use_gold_fallback,
            )

        run_ablation(
            pipeline,
            data,
            args.question_mode,
            top_k=args.top_k,
            pool_k=args.pool_k,
            rerank_cap=args.rerank_cap,
            use_gold_fallback=args.use_gold_fallback,
        )

    if args.run_both:
        p1 = make_pipeline(tkge_dir=None)
        run_all(p1, "RUN: Encoder-only (no TKGE)")

        p2 = make_pipeline(tkge_dir=args.tkge_dir)
        run_all(p2, f"RUN: FF Fusion (Encoder + TKGE) | tkge_dir={args.tkge_dir}")
    else:
        p = make_pipeline(tkge_dir=args.tkge_dir)
        title = (
            f"RUN: FF Fusion (Encoder + TKGE) | tkge_dir={args.tkge_dir}"
            if args.tkge_dir
            else "RUN: Encoder-only (no TKGE)"
        )
        run_all(p, title)


if __name__ == "__main__":
    main()