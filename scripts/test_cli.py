"""Interactive CLI to test arbitrary questions with the Agentic RAG system."""
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

def format_result(res: dict, elapsed: float, chunks_map: dict):
    status = res.get("status")
    answer = res.get("answer", "")
    citations = res.get("citations", [])
    events = res.get("events", [])

    print("\n" + "=" * 70)
    if status == "answered":
        print(f"🟢 [TRẠNG THÁI: ĐÃ TRẢ LỜI CÓ BẰNG CHỨNG] (Thời gian: {elapsed:.2f}s)")
    else:
        print(f"🟠 [TRẠNG THÁI: TỪ CHỐI - KHÔNG ĐỦ BẰNG CHỨNG] (Thời gian: {elapsed:.2f}s)")
    print("=" * 70)

    print(f"\n📝 CÂU TRẢ LỜI:\n{answer}\n")

    if citations:
        print("📌 TRÍCH DẪN & NGUỒN KIỂM CHỨNG:")
        for cid in citations:
            c = chunks_map.get(cid, {})
            page = c.get("page", "?")
            src = c.get("source", "Tài liệu")
            txt = c.get("text", "")[:180].strip()
            print(f"  • [{cid}] (Trang {page} - {src}):")
            print(f"    \"{txt}...\"\n")

    print("🔍 QUY TRÌNH RA QUYẾT ĐỊNH CỦA AGENT:")
    step_num = 1
    for e in events:
        ev_type = e.get("event")
        if ev_type == "search":
            q_str = e.get("query")
            results = e.get("results", [])
            print(f"  [Bước {step_num}] Agent gọi Tool Search:")
            print(f"    - Từ khóa tìm kiếm: \"{q_str}\"")
            print(f"    - Tìm thấy {len(results)} chunks:")
            for r in results:
                print(f"      + {r['id']} (Trang {r['page']}, score: {r.get('score', 0):.4f})")
            step_num += 1
        elif ev_type == "answer":
            print(f"  [Bước {step_num}] Agent đọc context và quyết định trả lời có trích dẫn.")
            step_num += 1
        elif ev_type == "search_limit":
            print(f"  [Bước {step_num}] Đã đạt giới hạn tìm kiếm tối đa (search_limit).")
            step_num += 1
        elif ev_type == "invalid_or_missing_citations":
            print(f"  [Bước {step_num}] Context không đủ bằng chứng tin cậy -> Từ chối (Abstain).")
            step_num += 1
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Test Agentic RAG tương tác")
    parser.add_argument("--question", "-q", type=str, default=None, help="Câu hỏi cần kiểm tra (nếu bỏ trống sẽ vào chế độ tương tác)")
    parser.add_argument("--strategy", choices=["hybrid", "dense", "lexical"], default="hybrid", help="Chiến lược retrieval")
    parser.add_argument("--budget", type=int, default=1200, help="Context budget")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k chunks")
    args = parser.parse_args()

    print("=" * 70)
    print("HỆ THỐNG AGENTIC RAG - SỔ TAY SINH VIÊN BÁCH KHOA 2026")
    print(f"Chiến lược: {args.strategy.upper()} | Context Budget: {args.budget} chars | Top-K: {args.top_k}")
    print("Đang khởi tạo Agent Controller (Gemini) và Retriever...")
    print("=" * 70)

    chunks_list = json.load(open(root_dir / "data" / "chunks.json", encoding="utf-8"))
    chunks_map = {c["id"]: c for c in chunks_list}

    controller = GeminiController()
    dense_ret = PgVectorRetriever()
    lex_ret = LexicalRetriever(chunks_list)

    if args.strategy == "dense":
        retriever = dense_ret
    elif args.strategy == "lexical":
        retriever = lex_ret
    else:
        retriever = HybridRetriever(dense_ret, lex_ret, rrf_k=60)

    print("✅ Hệ thống đã sẵn sàng!\n")

    if args.question:
        # One-shot test
        t0 = time.perf_counter()
        res = run_agent(args.question, controller, retriever, budget=args.budget, top_k=args.top_k)
        format_result(res, time.perf_counter() - t0, chunks_map)
        return

    # Interactive REPL
    print("Nhập câu hỏi của bạn (hoặc gõ 'exit' / 'q' để thoát):")
    print("Gợi ý câu hỏi thử:")
    print("  1. Ngày thành lập trường Đại học Chuyên nghiệp Bách khoa theo nghị định của Bộ Giáo dục là ngày nào?")
    print("  2. Mô hình và chương trình đào tạo tích hợp hệ chính quy được tổ chức như thế nào?")
    print("  3. Học bổng Trần Đại Nghĩa dành cho đối tượng nào?")
    print("  4. Mức học phí năm 2035 là bao nhiêu? (Câu hỏi ngoài tài liệu)")
    print("-" * 70)

    while True:
        try:
            q = input("\n👉 Nhập câu hỏi: ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Tạm biệt!")
                break

            print("⏳ Agent đang suy nghĩ và truy xuất bằng chứng...")
            t0 = time.perf_counter()
            res = run_agent(q, controller, retriever, budget=args.budget, top_k=args.top_k)
            format_result(res, time.perf_counter() - t0, chunks_map)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
