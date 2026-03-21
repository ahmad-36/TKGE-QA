# Temporal KG Embeddings for Event-Based Question Answering on ICEWS14

Research Case Study repository (WiSe 2025/26, Trier University) for event-based temporal question answering over ICEWS14.

The project studies whether TTransE-based structural plausibility adds useful signal beyond text-based semantic matching in a modular QA pipeline.

**Pipeline:** Question → Extraction → Retrieval → TimeFilter (ON/OFF) → Neural reranking → Top-k event candidates

**Settings:** time-conditioned vs. unconditioned, explicit vs. implicit questions, encoder-only vs. fusion reranking

The repository contains the final pipeline code, evaluation scripts, QA resources, fusion checkpoint, and TKGE artifacts. Exploratory components not used in the final reported experiments are kept under `experimental/`.

## Repository structure

- `pipeline.py` — main pipeline
- `preprocess/` — entity extraction
- `retrieval/` — retriever, temporal filter, rerankers, TKGE scorer
- `eval/` — evaluation scripts
- `data/` — QA splits and ICEWS14 retrieval data
- `checkpoints/` — fusion checkpoint
- `tkge_artifacts/` — TTransE checkpoint and ID maps
- `scripts/` — utility scripts
- `experimental/` — exploratory modules not used in the final reported experiments

## Setup

In Colab, the code was run with:

```bash
pip install -q torch sentence-transformers transformers huggingface-hub requests spacy
python -m spacy download en_core_web_sm

python eval/run_eval.py --run_both \
  --dataset data/official_QA_test_split.json \
  --tkge_dir tkge_artifacts/icews14_ttranse_v1 \
  --question_mode explicit \
  --top_k 10 --pool_k 200 --rerank_cap 200 \
  --time_tolerance 30

Configurable via command-line arguments for split, question type, filtering, and reranking.
