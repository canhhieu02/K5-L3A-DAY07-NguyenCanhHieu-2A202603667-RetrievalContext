"""Evaluate and compare 3 retrieval strategies on Dev split."""
import json
import statistics
import time
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "starter"))
sys.path.insert(0, str(root_dir / "src"))

from baseline import hit, pack_context
from retrieval import PgVectorRetriever, LexicalRetriever, HybridRetriever

def run_evaluation(retriever, strategy_name, questions, budget=800, use_filters=False):
    traces = []
    latencies = []
    
    for q in questions:
        filters = q.get("filter") if use_filters else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=3)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        context, context_ids = pack_context(results, budget=budget)
        
        # Check hits
        hit1 = bool(results and hit(results[0], q["gold"]))
        hit3 = any(hit(r, q["gold"]) for r in results[:3])
        context_hit = any(cid in context_ids for cid in [r["id"] for r in results if hit(r, q["gold"])])

        traces.append({
            "query_id": q["id"],
            "question": q["question"],
            "strategy": strategy_name,
            "latency_ms": round(latency_ms, 2),
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
            "results": results,
            "context_ids": context_ids
        })

    n = len(questions)
    hit1_count = sum(t["hit1"] for t in traces)
    hit3_count = sum(t["hit3"] for t in traces)
    context_hit_count = sum(t["context_hit"] for t in traces)

    summary = {
        "strategy": strategy_name,
        "n": n,
        "hit1": f"{hit1_count}/{n} ({hit1_count/n*100:.1f}%)",
        "hit3": f"{hit3_count}/{n} ({hit3_count/n*100:.1f}%)",
        "context_evidence_rate": f"{context_hit_count}/{n} ({context_hit_count/n*100:.1f}%)",
        "median_latency_ms": round(statistics.median(latencies), 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2)
    }
    return summary, traces

def main():
    print("=" * 65)
    print("DAY 07 · COMPARING 3 RETRIEVAL METHODS ON DEV SPLIT (N = 15)")
    print("=" * 65)

    data_dir = root_dir / "data"
    with open(data_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(data_dir / "questions.json", "r", encoding="utf-8") as f:
        all_q = json.load(f)
    dev_q = [q for q in all_q if q["split"] == "dev"]

    print(f"Loaded {len(chunks)} chunks and {len(dev_q)} dev questions.")

    # Initialize retrievers
    dense = PgVectorRetriever()
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical, rrf_k=60)

    results_dir = root_dir / "starter" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Dense Retrieval (pgvector)", dense, False),
        ("Dense + Filter (pgvector)", dense, True),
        ("Lexical Retrieval (TF-IDF)", lexical, False),
        ("Hybrid Retrieval (RRF)", hybrid, False),
        ("Hybrid + Filter (RRF)", hybrid, True)
    ]

    summaries = []
    for name, ret, use_filt in experiments:
        tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        summary, traces = run_evaluation(ret, name, dev_q, budget=800, use_filters=use_filt)
        summaries.append(summary)
        
        # Save traces
        out_file = results_dir / f"eval_{tag}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "traces": traces}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"{'Strategy':<30} | {'Hit@1':<12} | {'Hit@3':<12} | {'Context Hit':<14} | {'Latency':<8}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['strategy']:<30} | {s['hit1']:<12} | {s['hit3']:<12} | {s['context_evidence_rate']:<14} | {s['median_latency_ms']} ms")
    print("=" * 65)

    # Save summary table
    with open(results_dir / "retrieval_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

