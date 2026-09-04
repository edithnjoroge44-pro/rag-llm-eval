"""Retrieval evaluation: Recall@k, Hit-Rate@k, MRR — the standard metrics for judging RAG quality."""
import json
import os
from .index import VectorIndex
from .corpus import get_corpus, get_queries


def recall_at_k(retrieved, relevant, k):
    rel_hit = sum(1 for r in relevant if r in retrieved[:k])
    return rel_hit / len(relevant) if relevant else 0.0


def hit_rate_at_k(retrieved, relevant, k):
    return 1.0 if any(r in retrieved[:k] for r in relevant) else 0.0


def mrr(retrieved, relevant, k):
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def run_eval(k=3):
    vectorizer_index = VectorIndex()
    vectorizer_index.build(get_corpus())
    queries = get_queries()

    rec, hit, mrr_sum = 0.0, 0.0, 0.0
    per_query = []
    for item in queries:
        relevant = item["relevant"]
        ret = vectorizer_index.query(item["q"], k=k)
        per_query.append({
            "query": item["q"],
            "retrieved": ret,
            "relevant": relevant,
            "recall_at_k": round(recall_at_k(ret, relevant, k), 3),
            "mrr": round(mrr(ret, relevant, k), 3),
        })
        rec += recall_at_k(ret, relevant, k)
        hit += hit_rate_at_k(ret, relevant, k)
        mrr_sum += mrr(ret, relevant, k)

    n = len(queries)
    report = {
        "k": k,
        "recall_at_k": round(rec / n, 3),
        "hit_rate_at_k": round(hit / n, 3),
        "mrr": round(mrr_sum / n, 3),
        "num_queries": n,
        "per_query": per_query,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/rag_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    report = run_eval()
    print("=== RAG Retrieval Evaluation ===")
    print(f"Recall@{report['k']}: {report['recall_at_k']}")
    print(f"Hit-Rate@{report['k']}: {report['hit_rate_at_k']}")
    print(f"MRR: {report['mrr']}")
    print("(report saved to output/rag_eval_report.json)")
