# Temporal KG Embeddings for Event-Based Question Answering (Fork)

This repository is a fork of a research case study project developed at Trier University (WiSe 2025/26), focusing on temporal question answering over the ICEWS14 dataset.

🔗 Original repository: https://github.com/siwer/Research-Case-Studies-WS-25-26-

---

## 📌 My Contributions

My work focused on **semantic reranking using sentence encoders** within the QA pipeline and evaluating their effectiveness compared to temporal signals.

### 🔹 Encoder Exploration and Integration

* Implemented semantic reranking using sentence-transformer models
* Evaluated multiple encoders:

  * `all-MiniLM-L6-v2`
  * `BAAI/bge-large-en-v1.5` (final choice)
* Designed verbalization strategy for structured quadruples:

  ```
  {subject} {predicate} {object} on {timestamp}
  ```

### 🔹 Evaluation Pipeline

* Built:

  * `run_reranker_eval.py` → evaluation (Hit@10, MRR)
  * `build_reranker_eval_bundle_vA.py` → candidate pool construction
* Compared semantic reranking vs temporal filtering

### 🔹 Retrieval + Infrastructure

* Implemented:

  * `baseline_retriever.py` (entity-based retrieval with substring fallback)
  * `id_mapping.py` (consistent entity/relation mapping)

### 🔹 Fine-Tuning Experiments

* Implemented reranker fine-tuning using:

  * Triplet loss with hard negatives
  * Frozen encoder + projection head
* Result:

  * MRR improved (~0.73 → ~0.76)
  * Hit@10 unchanged

---

## 📊 Key Findings

* Temporal filtering had the **largest impact** (+0.257 Hit@10)
* Semantic reranking alone provided **no overall improvement**
* TKGE fusion gave **minor gains only for implicit questions**
* Conclusion:

  > Temporal information is more critical than semantic similarity for ICEWS-style event QA

---

## 📁 My Relevant Files

* `encoder_reranker.py`
* `baseline_retriever.py`
* `id_mapping.py`
* `run_reranker_eval.py`
* `build_reranker_eval_bundle_vA.py`

---

## 📖 Full Report

A detailed report including methodology, experiments, and reflections is available here:

👉 `docs/TKGQA_Report.pdf`

---

## ⚙️ Original Project 

This project explores **event-based temporal question answering** over the ICEWS14 dataset using a modular pipeline:

**Pipeline:**
Question → Extraction → Retrieval → Time Filtering → Neural Reranking → Top-k Results

**Goal:**
Evaluate whether **temporal knowledge graph embeddings (TTransE)** provide additional signal beyond semantic matching.

**Includes:**

* Full QA pipeline implementation
* Evaluation scripts
* Temporal KG embedding artifacts
* Fusion-based reranking model

---

## 🛠️ Setup (Original)

```bash
pip install -q torch sentence-transformers transformers huggingface-hub requests spacy
python -m spacy download en_core_web_sm

python eval/run_eval.py --run_both \
  --dataset data/official_QA_test_split.json \
  --tkge_dir tkge_artifacts/icews14_ttranse_v1 \
  --question_mode explicit \
  --top_k 10 --pool_k 200 --rerank_cap 200 \
  --time_tolerance 30
```
