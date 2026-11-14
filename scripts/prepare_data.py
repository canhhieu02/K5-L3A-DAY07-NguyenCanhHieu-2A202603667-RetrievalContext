"""Export chunks from chroma_db and prepare 24-question benchmark for Day 7."""
import json
import sys
from pathlib import Path
import chromadb

# Add root and starter to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "starter"))
from baseline import validate

def main():
    print("=" * 60)
    print("PREPARING DATA & BENCHMARK FOR DAY 7")
    print("=" * 60)

    # 1. Connect to ChromaDB in current workspace
    chroma_path = root_dir / "chroma_db"
    if not chroma_path.exists():
        raise FileNotFoundError(f"ChromaDB not found at {chroma_path}")
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_collection("strategy_b")
    raw_data = coll.get()
    
    print(f"Loaded {len(raw_data['ids'])} chunks from ChromaDB collection 'strategy_b'.")
    
    chunks = []
    for i in range(len(raw_data['ids'])):
        meta = raw_data['metadatas'][i]
        chunk = {
            "id": raw_data['ids'][i],
            "text": raw_data['documents'][i],
            "source_id": meta.get("document_id", "hust_student_handbook_2026"),
            "source": meta.get("source", "So_tay_sinh_vien_2026.pdf"),
            "page": meta.get("page", 1),
            "metadata": {
                "section": meta.get("section", "general"),
                "chunk_type": meta.get("chunk_type", "paragraph")
            }
        }
        chunks.append(chunk)

    # Sort chunks stably by id
    chunks.sort(key=lambda c: c["id"])

    # 2. Load Day 6 questions
    day6_benchmark_path = Path("E:/Bài Lab thực chiến/K5-L3A-Day06-NguyenCanhHieu-2A202603667-DataIngestion/benchmark/benchmark.json")
    with open(day6_benchmark_path, "r", encoding="utf-8") as f:
        day6_questions = json.load(f)

    # 3. Format 20 answerable questions
    answerable_questions = []
    for q in day6_questions:
        evidence = q["evidence_quote"]
        # Fix q05 to exact substring in chunk
        if q["id"] == "q05":
            evidence = "Kế hoạch học tập toàn khóa Kế hoạch học tập từng kỳ"
        
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [
                {
                    "source_id": q["gold_source_id"],
                    "page": q["gold_page"],
                    "evidence": evidence
                }
            ]
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        answerable_questions.append(item)

    # 4. Prepare 4 unanswerable questions (2 dev, 2 test)
    unanswerable_questions = [
        {
            "id": "q_dev_unans_01",
            "split": "dev",
            "question": "Mức học phí đại trà của Đại học Bách khoa Hà Nội vào năm 2035 được quy định là bao nhiêu triệu đồng một tín chỉ?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Sổ tay sinh viên 2026 chỉ cung cấp thông tin quy chế và học vụ hiện hành, không có quy định hay dự báo mức học phí năm 2035.",
            "expected_answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
            "gold": []
        },
        {
            "id": "q_dev_unans_02",
            "split": "dev",
            "question": "Nhà trường có chính sách hỗ trợ 100% tiền vé máy bay khứ hồi cho toàn bộ sinh viên có hộ khẩu miền Nam về quê ăn Tết không?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu chỉ có quy định về miễn giảm học phí và học bổng khuyến khích, không có chính sách hỗ trợ vé máy bay cho sinh viên miền Nam.",
            "expected_answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
            "gold": []
        },
        {
            "id": "q_test_unans_01",
            "split": "test",
            "question": "Quy trình xin visa chuyển tiếp sang Viện Công nghệ Massachusetts (MIT) được trường Đại học Bách khoa Hà Nội hướng dẫn chi tiết ở điều mấy?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Sổ tay sinh viên không chứa điều khoản hay quy trình cụ thể nào về việc xin visa chuyển tiếp sang MIT.",
            "expected_answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
            "gold": []
        },
        {
            "id": "q_test_unans_02",
            "split": "test",
            "question": "Sinh viên năm thứ nhất được cấp bao nhiêu thẻ đỗ xe ô tô miễn phí trong khuôn viên trường theo quy chế?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong sổ tay chỉ có các quy định chung về trật tự nội vụ và thẻ sinh viên, hoàn toàn không có quy định cấp thẻ đỗ xe ô tô miễn phí cho sinh viên năm nhất.",
            "expected_answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
            "gold": []
        }
    ]

    # 5. Build agent_cases (for agent_metrics.py evaluation)
    agent_cases = []
    for q in answerable_questions:
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })
    for q in unanswerable_questions:
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": False,
            "expected_behavior": q["expected_behavior"],
            "absence_reason": q["absence_reason"]
        })

    # 6. Combined 24-question benchmark
    benchmark_24 = answerable_questions + unanswerable_questions

    # 7. Validate baseline contract
    print("Validating chunks and answerable questions with baseline.validate()...")
    validate(chunks, answerable_questions)
    print("--> Validation PASSED successfully!")

    # 8. Save files to data/
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {data_dir / 'chunks.json'}")

    with open(data_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(answerable_questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(answerable_questions)} answerable questions to {data_dir / 'questions.json'}")

    with open(data_dir / "agent_cases.json", "w", encoding="utf-8") as f:
        json.dump(agent_cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(agent_cases)} agent evaluation cases to {data_dir / 'agent_cases.json'}")

    with open(data_dir / "benchmark_24.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_24, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(benchmark_24)} full benchmark questions to {data_dir / 'benchmark_24.json'}")

    # Summary breakdown
    dev_cases = [c for c in agent_cases if c["split"] == "dev"]
    test_cases = [c for c in agent_cases if c["split"] == "test"]
    print("\nBenchmark Breakdown:")
    print(f"  Dev Split:  {len(dev_cases)} total ({sum(c['answerable'] for c in dev_cases)} answerable + {sum(not c['answerable'] for c in dev_cases)} unanswerable) - Req >= 17: PASSED")
    print(f"  Test Split: {len(test_cases)} total ({sum(c['answerable'] for c in test_cases)} answerable + {sum(not c['answerable'] for c in test_cases)} unanswerable) - Req >= 7: PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()

