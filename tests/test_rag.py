import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.index import VectorIndex
from rag.corpus import get_corpus
from rag.evaluate import hit_rate_at_k, recall_at_k, mrr


def test_build_and_query_returns_doc_ids():
    idx = VectorIndex()
    idx.build(get_corpus())
    top = idx.query("refund policy", k=3)
    assert len(top) == 3
    assert all(isinstance(d, str) for d in top)


def test_relevant_doc_retrieved_for_refund():
    idx = VectorIndex()
    idx.build(get_corpus())
    top = idx.query("How fast do refunds come through?", k=3)
    assert "d1" in top  # refund doc is relevant


def test_relevant_doc_retrieved_for_shipping():
    idx = VectorIndex()
    idx.build(get_corpus())
    top = idx.query("what shipping times internationally", k=3)
    assert "d2" in top


def test_metrics_sane():
    # perfect retrieval: recall=1, hit=1, mrr=1
    assert hit_rate_at_k(["a", "b"], ["a"], 3) == 1.0
    assert recall_at_k(["a", "b"], ["a", "b"], 3) == 1.0
    assert mrr(["a", "b"], ["b"], 3) == 0.5
    # nothing retrieved
    assert hit_rate_at_k(["x"], ["a"], 3) == 0.0
