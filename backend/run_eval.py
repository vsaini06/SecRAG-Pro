import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.eval_framework import GoldenDataset, run_eval
from utils.retriever import retrieve_top_k
from utils.llm import generate_answer
from utils.citation_verifier import verify_citations
from utils.vector_store import get_collection

PDF_NAME = "1810.04805v2.pdf"
CHROMA_DIR = "../data/chroma"
DATA_DIR = "../data"

ds = GoldenDataset("data/eval/bert_golden_qa.json")
print(f"Loaded {len(ds)} questions")

c = get_collection(PDF_NAME, persist_dir=CHROMA_DIR)
print(f"ChromaDB chunks: {c.count()}")

def retrieve_fn(query, top_k=5, mode="hybrid"):
    return retrieve_top_k(
        query=query,
        pdf_name=PDF_NAME,
        top_k=top_k,
        mode=mode,
        use_reranker=True,
        chroma_dir=CHROMA_DIR,
    )

report = run_eval(
    dataset=ds,
    pdf_name=PDF_NAME,
    retrieve_fn=retrieve_fn,
    answer_fn=generate_answer,
    verify_fn=verify_citations,
    retrieval_mode="hybrid",
    chunk_strategy="sentence",
    run_label="bert_hybrid_sentence_v1",
)

os.makedirs("data/eval", exist_ok=True)
with open("data/eval/bert_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report["summary"], indent=2))