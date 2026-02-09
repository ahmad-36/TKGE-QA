import torch
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class EncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def score(self, question: str, triples: List[Dict]) -> List[float]:
        if not triples:
            return []

        texts = [self._triple_to_text(t) for t in triples]

        q_emb = self.model.encode(
            [question], convert_to_tensor=True, normalize_embeddings=True
        )  # (1, d)
        t_emb = self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )  # (n, d) #request norm, implement by SentTransf, so L2 norm: divide by vector length

        scores = torch.mm(q_emb, t_emb.T).squeeze(0)  # (n,)
        return scores.detach().cpu().tolist()

    def rerank(self, question: str, triples: List[Dict], top_k: int = 10) -> List[Dict]:

        if not triples:
            return []

        scores = self.score(question, triples)

        # attach scores to triples
        scored = []
        for t, s in zip(triples, scores):
            out = dict(t)
            out["retriever_score"] = t.get("score", 0.0)
            out["score"] = float(s) #attach encoder score
            scored.append(out)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k] #return top k

    @staticmethod
    def _triple_to_text(triple: Dict) -> str:
        head = triple.get("head", "")
        rel = triple.get("relation", "").replace("_", " ").lower()
        tail = triple.get("tail", "")
        date = triple.get("date", "")
        return f"{head} {rel} {tail} on {date}"
    # on is verbalization converting data to nl
