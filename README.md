# RAG + Retrieval Evaluation System

> **Stack:** Python · scikit-learn (TF-IDF + cosine) · custom vector index · evaluation metrics · pytest
> **Proves:** modern AI engineering — building a retrieval-augmented (RAG) pipeline from scratch and, critically, **evaluating it** with retrieval metrics. No cloud or API keys, so it runs anywhere and is fully verifiable.

This is a genuinely hard project: it builds a retrieval engine, then measures whether it actually works (hit-rate, MRR, recall@k) — the same "evaluation" skill that separates senior from junior AI work.

---

## What it does
1. **Ingests a small knowledge corpus** (synthetic company policy/FAQ docs, generated locally).
2. **Chunks & embeds** them with TF-IDF (a self-contained vectorization, no external API).
3. **Builds a vector index** (cosine similarity) and a `query()` function for semantic retrieval.
4. **Evaluates retrieval quality** against a set of query→expected-doc labels using **Recall@k, Hit-Rate@k, and MRR** (mean reciprocal rank).
5. Optionally scores retrieved passages against a **quality rubric** and computes a weighted LLM-eval score.

## Repository structure
```
rag-llm-eval/
├── README.md
├── requirements.txt
├── rag/
│   ├── corpus.py         # generates the synthetic knowledge corpus
│   ├── index.py          # TF-IDF embedding + cosine vector index + query()
│   └── evaluate.py       # Recall@k / Hit-Rate@k / MRR + rubric scoring
├── tests/
│   └── test_rag.py
└── output/               # generated eval report
```

## Why it matters
Retrieval without evaluation is guesswork. This project demonstrates the full loop: **build → retrieve → measure → report**. The evaluation metrics (Recall@k, MRR) are the standard way teams judge RAG quality, and the rubric scoring shows how AI outputs get graded — all of which is exactly the skill AfterQuery and similar labs screen for.

## Run it
```bash
pip install -r requirements.txt
python -m rag.evaluate          # builds index, runs queries, prints the eval report + saves it
pytest tests/ -v                # verifies the index + metrics are correct
```
