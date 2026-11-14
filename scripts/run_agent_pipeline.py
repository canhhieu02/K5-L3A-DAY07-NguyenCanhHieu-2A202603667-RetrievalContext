"""Run Agent pipeline on Dev/Test split with real LLM and vector retriever."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "starter"))
sys.path.insert(0, str(root_dir / "src"))

from agentic import run_agent
from agent import GeminiController
from retrieval import PgVectorRetriever, LexicalRetriever, HybridRetriever

def main():
    parser = argparse.ArgumentParser(description="Chạy Agentic RAG Pipeline với Gemini và pgvector")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Tập dữ liệu dev hoặc test")
    parser.add_argument("--strategy", choices=["dense", "lexical", "hybrid"], default="hybrid", help="Chiến lược retrieval")
    parser.add_argument("--budget", type=int, default=1200, help="Ngân sách Context Budget (chars)")
    parser.add_argument("--max_searches", type=int, default=3, help="Số lần search tối đa")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k chunks mỗi lần search")
    parser.add_argument("--output", type=str, default=None, help="Thư mục lưu kết quả")
    args = parser.parse_args()

    # Load cases with full benchmark metadata (filters, gold)
    cases_path = root_dir / "data" / "benchmark_24.json"
    chunks_path = root_dir / "data" / "chunks.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Filter by split and ensure answerable key
    cases = []
    for c in all_cases:
        if c["split"] == args.split:
            item = dict(c)
            if "answerable" not in item:
                item["answerable"] = True
            cases.append(item)
    print("=" * 65)
    print(f"RUNNING AGENT PIPELINE: Split = {args.split.upper()} (N = {len(cases)})")
    print(f"Strategy: {args.strategy} | Top-K: {args.top_k} | Budget: {args.budget} chars | Max Searches: {args.max_searches}")
    print("=" * 65)

    # Initialize Controller
    print("Connecting to Google Gemini API...")
    controller = GeminiController()

    # Initialize Retriever
    print(f"Initializing {args.strategy} retriever...")
    dense_ret = PgVectorRetriever()
    lex_ret = LexicalRetriever(chunks)
    if args.strategy == "dense":
        retriever = dense_ret
    elif args.strategy == "lexical":
        retriever = lex_ret
    else:
        retriever = HybridRetriever(dense_ret, lex_ret, rrf_k=60)

    output_dir = Path(args.output) if args.output else (root_dir / "starter" / "results" / f"agent_{args.split}_{args.strategy}")
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    reviews_template = []

    for idx, c in enumerate(cases, start=1):
        qid = c["id"]
        q_text = c["question"]
        print(f"\n[{idx}/{len(cases)}] Query {qid}: {q_text[:70]}...")

        t0 = time.perf_counter()
        try:
            result = run_agent(
                question=q_text,
                controller=controller,
                retriever=retriever,
                filters=c.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget
            )
            elapsed_s = time.perf_counter() - t0
            result["query_id"] = qid
            result["elapsed_s"] = round(elapsed_s, 2)
            traces.append(result)

            status = result.get("status")
            searches_count = sum(e.get("event") == "search" for e in result.get("events", []))
            print(f"  -> Status: {status} | Searches: {searches_count} | Time: {elapsed_s:.2f}s")
            print(f"  -> Answer: {result.get('answer', '')[:100]}...")
            print(f"  -> Citations: {result.get('citations', [])}")

            # Prepare review template
            is_ans = c.get("answerable", True)
            reviews_template.append({
                "query_id": qid,
                "question": q_text,
                "answerable": is_ans,
                "status": status,
                "agent_answer": result.get("answer"),
                "citations": result.get("citations", []),
                # Grounded answer: 0 = wrong, 1 = partial, 2 = correct & sufficient (for answerable questions)
                "answer_grade": 2 if (status == "answered" and is_ans) else 0,
                # Citation support: boolean per citation
                "citation_support": [True] * len(result.get("citations", [])),
                # Abstention correct: True if agent correctly abstained on unanswerable question
                "abstention_correct": (status == "abstain") if not is_ans else False,
                "review_notes": ""
            })

        except Exception as e:
            elapsed_s = time.perf_counter() - t0
            print(f"  -> ERROR: {e}")
            traces.append({
                "query_id": qid,
                "status": "error",
                "error": str(e),
                "elapsed_s": round(elapsed_s, 2),
                "events": [],
                "citations": []
            })
            reviews_template.append({
                "query_id": qid,
                "status": "error",
                "answer_grade": 0,
                "citation_support": [],
                "abstention_correct": False,
                "review_notes": f"Error: {e}"
            })
        
        # Rate limit protection (15 RPM on Gemini free tier)
        time.sleep(3.5)

    # Save traces and reviews template
    traces_file = output_dir / "traces.json"
    with open(traces_file, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

    reviews_file = output_dir / "reviews.json"
    with open(reviews_file, "w", encoding="utf-8") as f:
        json.dump(reviews_template, f, ensure_ascii=False, indent=2)

    cases_file = output_dir / "cases.json"
    with open(cases_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"COMPLETED! Output saved to: {output_dir}")
    print(f"  - Traces: {traces_file}")
    print(f"  - Reviews: {reviews_file}")
    print(f"  - Cases: {cases_file}")
    print("=" * 65)

if __name__ == "__main__":
    main()

