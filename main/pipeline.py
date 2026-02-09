from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import re
from retrieval.baseline_retriever import BaselineRetriever
from retrieval.time_filter import TimeFilter
from tkge_scorer import TKGEScorer, TKGEConfig
from retrieval.encoder_reranker import EncoderReranker
from retrieval.ff_fusion_reranker import FFFusionReranker, FFFusionConfig, _parse_date_loose
from datetime import timedelta

from preprocess.extractor import extract
from preprocess.expansion import load_implicit_graph, expand_entities_pattern_based
from question_rewriter import QuestionRewriter
from generation.answerer import Answerer



# remove prefix only
_TEMPORAL = r"(following|after|once|during|when|at the time when|at the time|in response to)"

_TEMPORAL_FRAME_PREFIX = re.compile(
    rf"^\s*{_TEMPORAL}\b[:\s,]*",
    re.IGNORECASE,
)

def _deframe_question(q: str) -> str:

    q2 = q.strip()
    q2 = _TEMPORAL_FRAME_PREFIX.sub("", q2).strip()
    return q2 if q2 else q


#time filter mode selection
def _operator_to_time_mode(temporal_operator: Optional[str]) -> str:
    if not temporal_operator:
        return "around"
    op = temporal_operator.strip().upper()
    if op == "AFTER":
        return "after"
    if op == "BEFORE":
        return "before"
    if op == "DURING":
        return "around"
    return "around"


def _signal_to_time_mode(signal_type: Optional[str]) -> str:
    if not signal_type:
        return "around"

    st = signal_type.lower().strip()

    # Order signals
    if st in {"after", "following", "once"}:
        return "after"
    if st in {"before"}:
        return "before"

    # Coincidence/overlap signals
    if st in {"during", "when", "at_the_time"}:
        return "around"

    return "around"


class TKGQAPipeline:
    def __init__(
        self,
        implicit_graph_path: str,
        icews_path: str,
        encoder_model_name: str = "BAAI/bge-large-en-v1.5",
        time_tolerance_days: int = 1,
        device: str = "cuda",
        retriever_cap: int = 1000,
        tkge_dir: Optional[str] = None,


    ):

        self.time_tolerance_days = int(time_tolerance_days)

        self.implicit_lookup = load_implicit_graph(implicit_graph_path)
        self.retriever = BaselineRetriever(events_path=icews_path, cap=retriever_cap)
        self.time_filter = TimeFilter(tolerance_days=time_tolerance_days)  # used for "around"
        self.encoder = EncoderReranker(model_name=encoder_model_name, device=device)
        self.fusion_reranker = None

        self.tkge = None


        if tkge_dir:
            td = tkge_dir.rstrip("/")
            self.tkge = TKGEScorer(
                TKGEConfig(
                    ckpt_path=f"{td}/ttranse_icews14_best.pt",
                    entity2id_path=f"{td}/entity2id.txt",
                    relation2id_path=f"{td}/relation2id.txt",
                    time2id_path=f"{td}/time2id.txt",
                    device=device,
                )
            )
            fusion_ckpt = "checkpoints/ff_fusion_mlp_2d.pt"
            self.fusion_reranker = FFFusionReranker(
                encoder=self.encoder,
                tkge=self.tkge,
                cfg=FFFusionConfig(hidden_dim=16, use_time_features=False, device=device),
                model_state_path=fusion_ckpt if os.path.exists(fusion_ckpt) else None,
            )

        self.answerer = Answerer()
        self.rewriter = QuestionRewriter(self.retriever)

    def _directional_time_filter(
            self,
            candidates: List[Dict[str, Any]],
            anchor_timestamp: str,
            signal_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        mode = _signal_to_time_mode(signal_type)
        return self._directional_time_filter_mode(candidates, anchor_timestamp, mode)

    def _directional_time_filter_mode(
            self,
            candidates: List[Dict[str, Any]],
            anchor_timestamp: str,
            mode: str,
    ) -> List[Dict[str, Any]]:
        anchor_dt = _parse_date_loose(anchor_timestamp)
        if not anchor_dt:
            return candidates

        tol = int(getattr(self, "time_tolerance_days", 0) or 0)
        delta = timedelta(days=tol)

        if mode == "after":
            lo, hi = anchor_dt, anchor_dt + delta
        elif mode == "before":
            lo, hi = anchor_dt - delta, anchor_dt
        else:
            # for "around", your code path already uses self.time_filter.filter(...)
            return candidates

        out: List[Dict[str, Any]] = []
        for c in candidates:
            c_dt = _parse_date_loose(c.get("date", ""))
            if not c_dt:
                # For directional constraints, unknown date can't satisfy direction.
                continue
            if lo <= c_dt <= hi:
                out.append(c)

        return out


    def process(
        self,
        question: str,
        encoder_top_k: int = 10,
        rerank_cap: int = 200,
        rerank_pool_k: Optional[int] = None,
        use_rewriter: bool = False,
        use_implicit: bool = True,
        use_time_filter: bool = True,
        use_reranker: bool = True,
        single_event: bool = False,
        override_entities: Optional[List[str]] = None,
        gold_time_anchor: Optional[str] = None,
        #for single-event exact day or small window
        single_event_time_mode: str = "around",  # "around" | "after" | "before"
        single_event_reliability: float = 1.0,

    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {"question": question}
        results["config"] = {
            "use_rewriter": use_rewriter,
            "use_implicit": use_implicit,
            "use_time_filter": use_time_filter,
            "use_reranker": use_reranker,
            "single_event": single_event,
            "gold_time_anchor": gold_time_anchor,
            "single_event_time_mode": single_event_time_mode,
        }

        if single_event:
            query = question
            anchor_ts = gold_time_anchor
            temporal_operator = None
            confidence = float(single_event_reliability)
            signal_type = None

            results["rewriter"] = {
                "original": question,
                "rewritten": question,
                "was_rewritten": False,
                "reason": "single_event_mode",
            }
            results["constraints_used"] = {
                "source": "gold_time_anchor",
                "anchor_timestamp": anchor_ts,
                "confidence": confidence,
                "mode": single_event_time_mode,
            }

        else:
            # original behavior for non-single-event
            if use_rewriter:
                rew = self.rewriter.rewrite(question)
                query = question  # current code keeps original
                constraints = rew.get("constraints", {}) or {}
                temporal_operator = constraints.get("temporal_operator")
                anchor_ts = constraints.get("anchor_timestamp")
                confidence = float(constraints.get("confidence", 0.0) or 0.0)
                signal_type = rew.get("signal_type")
                results["rewriter"] = rew
                results["constraints_used"] = {
                    "temporal_operator": temporal_operator,
                    "anchor_timestamp": anchor_ts,
                    "confidence": confidence,
                }
            else:
                query = question
                anchor_ts = None
                signal_type = None
                temporal_operator = None
                confidence = 0.0
                results["rewriter"] = {
                    "original": question,
                    "rewritten": question,
                    "was_rewritten": False,
                }
                results["constraints_used"] = None

        results["query_used"] = query
        results["anchor_timestamp"] = anchor_ts
        results["signal_type"] = signal_type
        results["temporal_operator"] = temporal_operator
        results["constraint_confidence"] = confidence

        # for single event
        extraction_text = _deframe_question(question) if single_event else question
        extraction = extract(extraction_text)

        results["extraction_text"] = extraction_text
        results["extracted_entities"] = [e["name"] for e in extraction.get("entities", [])]
        results["extracted_dates_raw"] = extraction.get("dates", [])

        entities = extraction.get("entities", [])
        dates: List[dict] = list(extraction.get("dates", []) or [])

        # Debug: override (oracle/fallback eval only) replace extracted entities entirely
        if override_entities:
            entities = [{"name": e} for e in override_entities]
            results["entities_overridden"] = True
            results["override_entities"] = list(override_entities)
        else:
            results["entities_overridden"] = False
            results["override_entities"] = None

        results["extracted_entities"] = [e["name"] for e in entities]


        if single_event:
            results["date_source"] = "gold_time_anchor" if anchor_ts else "missing_gold_time_anchor"
        else:
            results["date_source"] = "extractor" if dates else "none"


        # Step 2: Expand entities (skipped)
        expanded = [e["name"] for e in entities]
        results["expanded_entities"] = expanded

        # Step 3: Baseline retrieval
        candidates = self.retriever.retrieve(expanded)
        results["retrieved_candidates"] = len(candidates)

        # Step 4: Time filter
        filtered = candidates
        results["time_filter_applied"] = False
        results["time_filter_mode"] = None
        results["time_filter_source"] = None

        if use_time_filter:
            if single_event and anchor_ts:
                # Deterministic gold-time filtering
                mode = single_event_time_mode  #
                results["time_filter_mode"] = mode
                results["time_filter_source"] = "gold_time_anchor"

                if mode == "around":
                    dates_with_reliability = [{
                        "date": anchor_ts,
                        "format": "iso",
                        "reliability": confidence,
                    }]
                    filtered = self.time_filter.filter(candidates, dates_with_reliability)
                else:
                    filtered = self._directional_time_filter_mode(candidates, anchor_ts, mode)

                results["time_filter_applied"] = True

            elif (not single_event) and dates:
                mode = _signal_to_time_mode(signal_type)
                results["time_filter_mode"] = mode
                results["time_filter_source"] = "extractor"

                if mode == "around":
                    dates_with_reliability = [
                        {"date": d["date"], "format": d.get("format", "iso"), "reliability": 0.9}
                        for d in dates
                        if d.get("date")
                    ]
                    filtered = self.time_filter.filter(candidates, dates_with_reliability)
                else:
                    filtered = self._directional_time_filter(candidates, dates[0]["date"], signal_type)

                results["time_filter_applied"] = True

            else:
                results["time_filter_source"] = "skipped_no_temporal_info"

        results["after_time_filter"] = len(filtered)

        # cap for reranking
        filtered = filtered[:rerank_cap]
        results["filtered_candidates"] = filtered


        # Step 5: Rerank / fusion
        if filtered:
            if use_reranker and (self.tkge is not None) and (self.fusion_reranker is not None):
                pool = rerank_pool_k if rerank_pool_k is not None else max(50, encoder_top_k)

                top_triples = self.fusion_reranker.rerank(
                    query=query,
                    candidates=filtered,
                    top_k=encoder_top_k,
                    pool_k=pool,
                    anchor_timestamp=anchor_ts,
                )
                results["reranker"] = {"mode": "ff_fusion_mlp_relu", "pool_k": pool}
                results["tkge_usage"] = {"enabled": True, "reason": "learned_fusion"}
            else:
                top_triples = self.encoder.rerank(query, filtered, top_k=encoder_top_k)
                results["reranker"] = {"mode": "encoder_only"}
                results["tkge_usage"] = {"enabled": False, "reason": "tkge_disabled_or_reranker_off"}
        else:
            top_triples = []

        results["final_triples"] = top_triples

        # Step 6: Answer
        ans = self.answerer.answer(
            question=query,
            top_triples=top_triples,
            extracted_entities=[e["name"] for e in entities],
        )
        results["answer"] = ans.get("answer")
        results["answer_type"] = ans.get("answer_type")
        results["answer_evidence"] = ans.get("evidence")

        return results

