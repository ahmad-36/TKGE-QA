# Research-Case-Studies-WS-25-26-
# Temporal KG Embeddings for Event-Based QA (ICEWS14)

Research Case Study (WiSe 2025/26, Trier University). We evaluate whether 
**temporal KG embeddings (TTransE)** improve **event-based question 
answering** when combined with **explicit temporal grounding** 
(TimeFilter) and **semantic reranking** (BGE-large / fusion).

## What this repo contains
A modular pipeline:
**Question → Extraction → Retrieval → (optional) TimeFilter (±30d) → 
Neural reranking → Top-k events**

Evaluation regimes:
- **Time-conditioned**: TimeFilter ON (uses temporal anchor metadata; 
isolates reranking quality)
- **Unconditioned**: TimeFilter OFF (full temporal ambiguity)

Question types:
- **Explicit** (timestamp in text)
- **Implicit** (no timestamp; event reference only)

## Key folders
- `main/` — pipeline entrypoint
- `preprocess/` — entity extraction + expansion
- `retrieval/` — baseline retriever, time filter, encoder reranker, fusion 
reranker
- `eval/` — evaluation scripts
- `data/` — QA splits/resources
- `Checkpoints_TTransE/` — TTransE checkpoint + id maps (used by fusion)

## Notes
- Some older baseline code remains in the root directory; repo cleanup + 
consolidated run instructions will be added. (TBD)
- Large artifacts (env/caches) are excluded via `.gitignore`.

