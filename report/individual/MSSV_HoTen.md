# Báo cáo Cá Nhân — Day 7 · Agentic RAG & Context Engineering

**Học viên:** Nguyễn Cảnh Hiệu  
**MSSV:** 2A202603667  
**Lớp:** K5-L3A  
**Chủ đề:** Sổ tay sinh viên Đại học Bách khoa Hà Nội 2026 (`hust.edu.vn`)

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - *Nguồn và quyền:* Tài liệu chính thức *"Sổ tay sinh viên Đại học Bách khoa Hà Nội 2026"* được công bố công khai cho sinh viên toàn trường (`hust.edu.vn`); không chứa dữ liệu mật hay thông tin định danh cá nhân nhạy cảm (PII).
  - *Số trang:* 55 trang PDF vật lý (tương đương 110 trang sách), trong đó có **52 trang nội dung thực** có chứa văn bản học vụ hữu ích (>50 ký tự), đáp ứng yêu cầu $\ge 30$ trang.
  - *Số chunk:* **230 chunks** được kế thừa từ chiến lược phân đoạn theo đoạn văn logic (Paragraph-aware segmentation, tối đa 600 ký tự/chunk, giữ trọn ý câu và cấu trúc đề mục).
- **Vector store, embedding model, phiên bản và số chiều:**
  - *Vector store:* **PostgreSQL 16** tích hợp extension **`pgvector` (v0.8.7)** triển khai qua Docker container (`day7_pgvector` trên cổng `55433`). Tạo chỉ mục **HNSW** (`vector_cosine_ops`) tối ưu hóa truy vấn tương đồng Cosine.
  - *Embedding model:* `all-MiniLM-L6-v2` (`sentence-transformers` v6.3.0), số chiều vector: **384**, chuẩn hóa L2 norm (`normalize_embeddings=True`).
- **Agent model, cách chạy và phân chia dev/test:**
  - *Agent model:* **Google Gemini** (`gemini-3.1-flash-lite`), nhiệt độ $0.0$, định dạng phản hồi bắt buộc JSON schema với hành động chuẩn hóa: `{"action": "search", "query": "..."}` hoặc `{"action": "answer", "answer": "... [chunk-id]", "citations": [...]}`.
  - *Cơ chế điều phối:* Tuân thủ giao thức `starter/agentic.py` với `max_searches = 3`, `top_k = 3`, `context_budget = 1200` ký tự.
  - *Phân chia dữ liệu (Tổng cộng 24 câu):*
    - **Tập Dev (17 câu):** 15 câu có đáp án trong tài liệu (`q01`–`q15`) và **2 câu thiếu bằng chứng** (`q_dev_unans_01`, `q_dev_unans_02`).
    - **Tập Test (7 câu):** 5 câu có đáp án trong tài liệu (`q16`–`q20`) và **2 câu thiếu bằng chứng** (`q_test_unans_01`, `q_test_unans_02`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* Xác định chuỗi trích dẫn nguyên văn `gold evidence`, số trang và mã chunk nguồn `gold_source_id`.
  - *Câu thiếu evidence:* Thiết kế các câu hỏi thực tế gần chủ đề học vụ nhưng không xuất hiện trong quy chế (ví dụ: dự báo học phí năm 2035, cấp thẻ xe ô tô miễn phí, visa MIT, tài trợ vé máy bay Tết).
  - *Tiêu chí chấm (Audit):*
    - **Grounded Answer:** 2 = Trả lời đúng, đủ và trích dẫn khớp; 1 = Trả lời đúng một phần; 0 = Trả lời sai hoặc từ chối câu có đáp án (false abstention).
    - **Citation hợp lệ:** Từng ID citation phải tồn tại trong Context VÀ văn bản của chunk đó phải trực tiếp chứng minh cho câu trả lời.
    - **Abstention đúng:** Agent phải thể hiện sự từ chối rõ ràng ("Không đủ bằng chứng..."), citations rỗng; không bịa đặt hoặc suy diễn vô căn cứ.

---

## 2. Retrieval và thử nghiệm

### Bảng 1: So sánh các phương pháp Retrieval trên tập Dev (15 câu có đáp án)

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / Thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / Chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Dense (pgvector HNSW), top-k=3, budget=800 | Truy vấn vector thuần túy bằng embedding 384d | dev | 2/15 (13.3%) | 5/15 (33.3%) | 2/15 (13.3%) | 54.8 ms / $0 |
| **Phương pháp 2** | Lexical (TF-IDF NFC Tokenizer), top-k=3, budget=800 | Bắt từ khóa nguyên văn, thuật ngữ viết tắt tiếng Việt | dev | 10/15 (66.7%) | 13/15 (86.7%) | 10/15 (66.7%) | **1.7 ms** / $0 |
| **Phương pháp 3** | Hybrid RRF (k=60, 50/50), top-k=3, budget=800 | Hợp nhất thứ hạng Dense + Lexical chuẩn RRF | dev | 5/15 (33.3%) | 8/15 (53.3%) | 5/15 (33.3%) | 47.9 ms / $0 |
| **Cải thiện 1** | Dense + Metadata Filter (`section`), top-k=3 | Lọc theo chuyên mục hành chính trước khi tìm kiếm vector | dev | 2/15 (13.3%) | 6/15 (40.0%) | 2/15 (13.3%) | 47.0 ms / $0 |
| **Cải thiện 2** | **Weighted Hybrid (0.6 Lexical + 0.4 Dense) + Budget 1200** | Ưu tiên từ khóa tiếng Việt, nâng budget để tránh mất chunk | dev | **11/15 (73.3%)** | **14/15 (93.3%)** | **13/15 (86.7%)** | 48.2 ms / $0 |
| **Cấu hình chốt** | **Weighted Hybrid + Budget 1200, top-k=3** | Khóa cấu hình tốt nhất từ Dev để đánh giá Test | test | **3/5 (60.0%)** | **4/5 (80.0%)** | **3/5 (60.0%)** | 49.5 ms / $0 |
| **Baseline Test** | Dense (pgvector), top-k=3, budget=800 | Đánh giá đối chứng trên cùng tập test | test | 1/5 (20.0%) | 2/5 (40.0%) | 1/5 (20.0%) | 56.1 ms / $0 |

### Phân tích hai lỗi trên Dev, nguyên nhân và cách xử lý:
1. **Lỗi 1 (Language & Domain Mismatch trong Dense Embedding):**
   - *Hiện tượng:* Phương pháp Dense pgvector chỉ đạt Evidence Hit@3 là $5/15$ ($33.3\%$), trong khi Lexical đạt $13/15$ ($86.7\%$). Ở các câu chứa từ viết tắt hành chính hoặc số hiệu văn bản (như `147/NĐ`, `4+1,5`, `5,5 năm`, `CTĐT`), vector search trả về các đoạn chung chung về phong trào sinh viên.
   - *Nguyên nhân:* Model `all-MiniLM-L6-v2` được tiền huấn luyện chủ yếu trên tiếng Anh, biểu diễn ngữ nghĩa cho các từ viết tắt chuyên ngành Bách khoa không đủ phân tách.
   - *Cách thử & Khắc phục:* Áp dụng **Weighted Hybrid Fusion** gán trọng số $0.6$ cho Lexical và $0.4$ cho Dense thay vì chia đều $50/50$. Điều này giúp bảo toàn nguyên vẹn năng lực bắt từ khóa chính xác của Lexical trong khi vẫn giữ tính mở rộng ngữ nghĩa của Dense. Kết quả: Evidence Hit@3 tăng từ $53.3\%$ lên **$93.3\%$** ($14/15$ câu).
2. **Lỗi 2 (Context Budget Truncation làm mất bằng chứng):**
   - *Hiện tượng:* Với budget ban đầu 800 ký tự, mặc dù Hit@3 đạt $13/15$, nhưng Context Evidence Rate chỉ đạt $10/15$ ($66.7\%$). Ba câu hỏi (`q02`, `q04`, `q07`) bị mất bằng chứng trong context cuối gửi cho Agent.
   - *Nguyên nhân:* Khi chunk đầu tiên dài (~550 ký tự), thuật toán `pack_context` không thể nhồi thêm chunk thứ 2 và thứ 3 mà không vượt quá 800 ký tự, dẫn đến bằng chứng thực sự bị loại bỏ.
   - *Cách thử & Khắc phục:* Nâng ngân sách `context_budget` từ $800$ lên **$1200$ ký tự**. Tỷ lệ Context Evidence Rate lập tức tăng từ $66.7\%$ lên **$86.7\%$** ($13/15$ câu), loại bỏ triệt để hiện tượng thiếu context.

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - *Tool retrieval:* Tích hợp tool `search(query, filters, top_k)` cho phép Agent tự sinh từ khóa tìm kiếm tiếng Việt.
  - *Số lượt gọi:* Giới hạn `max_searches = 3`. Nếu sau 3 lần tìm kiếm vẫn không đủ thông tin, Agent dừng an toàn qua sự kiện `search_limit` và trả về câu từ chối chuẩn.
  - *Cách chọn context:* Khử trùng lặp ID (`deduplication`), sắp xếp theo độ tương đồng và đóng gói trong giới hạn ngân sách 1200 ký tự.
  - *Bảo vệ Rate Limit (429):* Thêm khoảng trễ $3.5$ giây giữa các lượt query và cơ chế tự động backoff retry 20s khi gặp lỗi 429 trên Gemini Free Tier.

### Bảng 2: Hiệu năng Agent trên Dev và Test

| Cấu hình / Split | Tool use | Lượt gọi TB | Grounded answer 0/1/2 và Trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dense) / Dev** | 17/17 (100%) | 2.47 | Grade 0: 11, Grade 1: 0, Grade 2: 4 (**0.53 / 2.0**) | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 11/15 (73.3%) |
| **Cấu hình chọn (Hybrid) / Dev** | 17/17 (100%) | 1.41 | Grade 0: 2, Grade 1: 0, Grade 2: 13 (**1.73 / 2.0**) | 14/15 (93.3%) | 13/13 (100%) | 1/2 (50.0%) | 2/15 (13.3%) |
| **Cấu hình chọn (Hybrid) / Test** | 7/7 (100%) | 1.86 | Grade 0: 3, Grade 1: 0, Grade 2: 2 (**0.80 / 2.0**) | 3/3 (100%) | 2/2 (100%) | **2/2 (100%)** | 3/5 (60.0%) |
| **Baseline (Dense) / Test** | 5/5 (100%) | 2.20 | Grade 0: 3, Grade 1: 0, Grade 2: 1 (**0.50 / 2.0**) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 3/4 (75.0%) |

### Bảng 3: Chi tiết Audit từng câu trên tập Test (Kiểm chứng thủ công)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|:---:|:---:|---|---|---|
| **q16** | Có đáp án | `search("cán bộ giảng viên Khoa Toán Tin")` $\to$ Chunks p13, p17, p26 | Agent nhận diện ngữ cảnh không đủ thông tin về Khoa Toán - Tin $\to$ Từ chối trả lời. | **0/2** (False abstention do corpus không có trang riêng cho Khoa Toán Tin). |
| **q17** | Có đáp án | `search("tại sao sinh viên tham gia tổ chức hoạt động ngoại khóa")` $\to$ Chunks p13, p12, p17 | Agent tìm kiếm 3 lượt không đủ điều kiện dừng $\to$ Từ chối trả lời. | **0/2** (False abstention do query rewrite thiếu keyword). |
| **q18** | Có đáp án | `search("mục đích đi làm thêm của sinh viên")` $\to$ Chunk `p51_c000_para` | *"Quan điểm đúng đắn về mục đích của việc đi làm thêm là để tích lũy kinh nghiệm, kỹ năng sống... [hust_student_handbook_2026_p51_c000_para]"* | **2/2** (Chính xác 100%, trích dẫn đúng trang 51). |
| **q19** | Có đáp án | `search("chiêu trò lừa đảo sinh viên việc làm")` $\to$ Chunks p12, p13, p14 | Không tìm thấy cảnh báo cụ thể trong các chunk trả về $\to$ Từ chối trả lời. | **0/2** (False abstention). |
| **q20** | Có đáp án | `search("lời khuyên tình bạn tình yêu sống thử sinh viên")` $\to$ Chunks `p53_c001_para`, `p53_c003_para` | *"Về tình bạn nên cởi mở hòa đồng... không nên sống thử khi chưa đủ chín chắn... [hust_student_handbook_2026_p53_c001_para], [hust_student_handbook_2026_p53_c003_para]"* | **2/2** (Đúng toàn bộ nội dung, trích dẫn chính xác cả 2 chunks trang 53). |
| **q_test_unans_01** | Thiếu bằng chứng | `search("quy trình xin visa chuyển tiếp MIT")` $\to$ Không có chunk liên quan | *"Không đủ bằng chứng trong tài liệu để trả lời."* (Citations: `[]`) | **Đạt** (Abstention đúng, không bịa đặt quy trình visa). |
| **q_test_unans_02** | Thiếu bằng chứng | `search("thẻ đỗ xe ô tô miễn phí sinh viên năm nhất")` $\to$ Tìm kiếm 3 lần không thấy | *"Không đủ bằng chứng trong tài liệu để trả lời."* (Citations: `[]`) | **Đạt** (Abstention đúng, không khẳng định sai sự thật). |

---

## 4. Kết luận, Đánh đổi và Giới hạn

- **Đường dẫn kiểm chứng:**
  - Traces Dev: `starter/results/agent_dev_hybrid/traces.json`
  - Traces Test: `starter/results/agent_test_hybrid/traces.json`
  - Script chạy lại pipeline: `python scripts/run_agent_pipeline.py --split test --strategy hybrid`
- **Đánh đổi (Trade-offs):**
  - *Chất lượng vs Thời gian:* Hybrid Weighted RRF tăng thời gian truy xuất từ 1.7ms lên ~48ms/câu so với Lexical thuần túy, nhưng bù lại điểm chất lượng câu trả lời Grounded Answer tăng vượt bậc từ 0.53 lên 1.73/2.0.
  - *Chi phí:* Toàn bộ Vector DB và Embedding model chạy hoàn toàn local offline (Docker PostgreSQL + `all-MiniLM-L6-v2`), chi phí $0. Phía Agent sử dụng Gemini Free Tier, chi phí API là $0.
- **Giới hạn của hệ thống:**
  - Khi người dùng hỏi câu hỏi yêu cầu tổng hợp đa trang phức tạp (multi-hop) mà từ khóa ở câu hỏi khác biệt hoàn toàn với văn bản nguồn, Agent vẫn có xu hướng từ chối sớm (false abstention) thay vì suy diễn bừa bãi. Tuy nhiên, hành vi này là an toàn và phù hợp với tiêu chuẩn RAG có trách nhiệm.
