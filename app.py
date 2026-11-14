"""Web UI for testing Day 7 Agentic RAG System (HUST Student Handbook 2026).
Runs on http://127.0.0.1:7860 using standard library HTTP server.
"""
from __future__ import annotations
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir / "starter"))
sys.path.insert(0, str(root_dir / "src"))

from agentic import run_agent
from agent import GeminiController
from retrieval import PgVectorRetriever, LexicalRetriever, HybridRetriever

# Global state
print("Initializing Agentic RAG system for Web UI...")
chunks_list = json.load(open(root_dir / "data" / "chunks.json", encoding="utf-8"))
chunks_map = {c["id"]: c for c in chunks_list}

controller = GeminiController()
dense_ret = PgVectorRetriever()
lex_ret = LexicalRetriever(chunks_list)
hybrid_ret = HybridRetriever(dense_ret, lex_ret, rrf_k=60)
print(f"Loaded {len(chunks_list)} chunks. System ready!")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic RAG — Sổ Tay Sinh Viên Bách Khoa 2026</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
    .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .chip { transition: all 0.2s ease; }
    .chip:hover { transform: translateY(-1px); }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <!-- Header -->
  <header class="glass sticky top-0 z-50 px-6 py-4 border-b border-slate-800">
    <div class="max-w-6xl mx-auto flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-red-600 to-amber-500 flex items-center justify-center font-bold text-white shadow-lg shadow-red-500/20">
          <i class="fa-solid fa-graduation-cap text-lg"></i>
        </div>
        <div>
          <h1 class="font-bold text-lg text-white">Agentic RAG Assistant</h1>
          <p class="text-xs text-slate-400">Sổ tay sinh viên ĐHBK Hà Nội 2026 • PostgreSQL pgvector & Gemini 3.1</p>
        </div>
      </div>
      <div class="flex items-center space-x-3 text-xs">
        <span class="inline-flex items-center px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
          pgvector :55433
        </span>
        <span class="inline-flex items-center px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
          <span class="w-2 h-2 rounded-full bg-blue-400 mr-1.5 animate-pulse"></span>
          Gemini 3.1 Flash Lite
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-6xl mx-auto px-6 py-8 flex-1 w-full space-y-6">
    <!-- Quick Suggestions -->
    <div class="space-y-2">
      <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
        <i class="fa-regular fa-lightbulb text-amber-400"></i>
        Câu hỏi gợi ý kiểm thử
      </div>
      <div class="flex flex-wrap gap-2">
        <button onclick="setQuery('Ngày thành lập trường Đại học Chuyên nghiệp Bách khoa theo nghị định của Bộ Giáo dục là ngày nào?')" 
                class="chip text-xs bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700/60">
          📜 Ngày thành lập ĐHBK Hà Nội?
        </button>
        <button onclick="setQuery('Học bổng Trần Đại Nghĩa của ĐHBK Hà Nội dành cho đối tượng sinh viên nào?')" 
                class="chip text-xs bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700/60">
          🎓 Học bổng Trần Đại Nghĩa?
        </button>
        <button onclick="setQuery('Mô hình và chương trình đào tạo tích hợp hệ chính quy được tổ chức theo cấu trúc thời gian như thế nào?')" 
                class="chip text-xs bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700/60">
          ⏱️ Mô hình đào tạo tích hợp 4+1,5?
        </button>
        <button onclick="setQuery('Mức học phí đại trà của Đại học Bách khoa Hà Nội vào năm 2035 được quy định là bao nhiêu triệu đồng một tín chỉ?')" 
                class="chip text-xs bg-red-950/40 hover:bg-red-900/40 text-red-300 px-3 py-1.5 rounded-lg border border-red-800/40">
          🚫 Học phí năm 2035? (Câu thiếu evidence)
        </button>
        <button onclick="setQuery('Quy trình xin visa chuyển tiếp sang Viện Công nghệ Massachusetts (MIT) được trường Đại học Bách khoa Hà Nội hướng dẫn chi tiết ở điều mấy?')" 
                class="chip text-xs bg-red-950/40 hover:bg-red-900/40 text-red-300 px-3 py-1.5 rounded-lg border border-red-800/40">
          🚫 Visa chuyển tiếp MIT? (Câu thiếu evidence)
        </button>
      </div>
    </div>

    <!-- Query Input Card -->
    <div class="glass p-5 rounded-2xl shadow-xl space-y-4">
      <div class="flex flex-col md:flex-row gap-3">
        <div class="relative flex-1">
          <input type="text" id="questionInput" 
                 placeholder="Nhập câu hỏi học vụ, quy chế, học bổng sinh viên Bách khoa..." 
                 class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all pr-10"
                 onkeydown="if(event.key==='Enter') submitQuery()">
          <button onclick="clearInput()" class="absolute right-3 top-3.5 text-slate-500 hover:text-slate-300 text-xs">
            <i class="fa-solid fa-circle-xmark"></i>
          </button>
        </div>
        <button id="sendBtn" onclick="submitQuery()" 
                class="bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 text-white font-medium px-6 py-3 rounded-xl text-sm shadow-lg shadow-red-600/30 flex items-center justify-center gap-2 transition-all disabled:opacity-50">
          <i class="fa-solid fa-paper-plane"></i>
          <span>Gửi câu hỏi</span>
        </button>
      </div>

      <!-- Config Toolbar -->
      <div class="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-800 text-xs text-slate-400">
        <div class="flex items-center gap-4">
          <label class="flex items-center gap-1.5">
            <span>Chiến lược:</span>
            <select id="strategySelect" class="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 focus:outline-none">
              <option value="hybrid" selected>Hybrid Weighted (RRF)</option>
              <option value="dense">Dense (pgvector HNSW)</option>
              <option value="lexical">Lexical (TF-IDF)</option>
            </select>
          </label>
          <label class="flex items-center gap-1.5">
            <span>Context Budget:</span>
            <select id="budgetSelect" class="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 focus:outline-none">
              <option value="800">800 chars</option>
              <option value="1200" selected>1200 chars (Tối ưu)</option>
              <option value="1600">1600 chars</option>
            </select>
          </label>
          <label class="flex items-center gap-1.5">
            <span>Top-K:</span>
            <select id="topkSelect" class="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 focus:outline-none">
              <option value="3" selected>3 chunks</option>
              <option value="5">5 chunks</option>
            </select>
          </label>
        </div>
        <div class="text-slate-500">
          Max Searches: 3 lượt • Free Quota Guard Active
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div id="loadingBox" class="hidden glass p-8 rounded-2xl text-center space-y-3">
      <div class="inline-block animate-spin text-red-500 text-3xl">
        <i class="fa-solid fa-circle-notch"></i>
      </div>
      <p class="text-sm font-medium text-slate-300">Agent đang suy luận, gọi công cụ tìm kiếm và đối chiếu bằng chứng...</p>
      <p class="text-xs text-slate-500">Mô hình Google Gemini đang xử lý ngữ cảnh</p>
    </div>

    <!-- Result Area -->
    <div id="resultBox" class="hidden space-y-6">
      <!-- Answer Header Card -->
      <div class="glass p-6 rounded-2xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div class="flex items-center gap-2.5">
            <span id="statusBadge" class="px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider"></span>
            <span id="elapsedBadge" class="text-xs text-slate-400 bg-slate-800/60 px-2.5 py-1 rounded-lg"></span>
          </div>
          <div class="text-xs text-slate-400">
            Chi phí: <span class="text-emerald-400 font-semibold">$0.00</span> (Free Tier)
          </div>
        </div>

        <!-- Answer Text -->
        <div class="space-y-2">
          <div class="text-xs font-semibold uppercase tracking-wider text-slate-400">Câu trả lời từ Agent:</div>
          <div id="answerText" class="text-slate-100 text-sm leading-relaxed p-4 bg-slate-900/60 rounded-xl border border-slate-800 whitespace-pre-wrap"></div>
        </div>

        <!-- Citations List -->
        <div id="citationsSection" class="space-y-2 pt-2">
          <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <i class="fa-solid fa-bookmark text-red-400"></i>
            Trích dẫn & Nguồn bằng chứng đối chiếu:
          </div>
          <div id="citationsList" class="grid grid-cols-1 md:grid-cols-2 gap-3"></div>
        </div>
      </div>

      <!-- Agent Trace Timeline -->
      <div class="glass p-6 rounded-2xl space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-sm font-semibold text-white flex items-center gap-2">
            <i class="fa-solid fa-timeline text-indigo-400"></i>
            Quy trình suy luận của Agent (Tool Calls & Reasoning Trace)
          </h3>
          <span id="eventsCount" class="text-xs text-slate-400"></span>
        </div>
        <div id="traceTimeline" class="space-y-3"></div>
      </div>
    </div>
  </main>

  <footer class="py-4 text-center text-xs text-slate-500 border-t border-slate-800/60">
    Day 7 — Agentic RAG Lab • Nguyễn Cảnh Hiệu (2A202603667) • K5-L3A
  </footer>

  <script>
    function setQuery(text) {
      document.getElementById('questionInput').value = text;
      submitQuery();
    }

    function clearInput() {
      document.getElementById('questionInput').value = '';
      document.getElementById('questionInput').focus();
    }

    async function submitQuery() {
      const q = document.getElementById('questionInput').value.trim();
      if (!q) return;

      const strategy = document.getElementById('strategySelect').value;
      const budget = parseInt(document.getElementById('budgetSelect').value);
      const top_k = parseInt(document.getElementById('topkSelect').value);

      document.getElementById('loadingBox').classList.remove('hidden');
      document.getElementById('resultBox').classList.add('hidden');
      document.getElementById('sendBtn').disabled = true;

      try {
        const resp = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, strategy, budget, top_k })
        });
        const data = await resp.json();
        renderResult(data);
      } catch (err) {
        alert('Lỗi kết nối server: ' + err.message);
      } finally {
        document.getElementById('loadingBox').classList.add('hidden');
        document.getElementById('sendBtn').disabled = false;
      }
    }

    function renderResult(data) {
      document.getElementById('resultBox').classList.remove('hidden');

      // Status Badge
      const statusBadge = document.getElementById('statusBadge');
      if (data.status === 'answered') {
        statusBadge.className = 'px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
        statusBadge.innerHTML = '<i class="fa-solid fa-circle-check mr-1.5"></i> ĐÃ TRẢ LỜI CÓ BẰNG CHỨNG';
      } else {
        statusBadge.className = 'px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/30';
        statusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation mr-1.5"></i> TỪ CHỐI - THIẾU BẰNG CHỨNG';
      }

      document.getElementById('elapsedBadge').innerText = `Thời gian: ${data.elapsed_s.toFixed(2)}s`;

      // Answer formatting with highlighted citation badges
      let formattedAnswer = data.answer;
      if (data.citations && data.citations.length > 0) {
        data.citations.forEach(c => {
          const badge = `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono bg-red-950/70 text-red-300 border border-red-800/80 mx-1 cursor-pointer">[${c.id}]</span>`;
          formattedAnswer = formattedAnswer.replaceAll(`[${c.id}]`, badge);
        });
      }
      document.getElementById('answerText').innerHTML = formattedAnswer;

      // Citations Section
      const citSection = document.getElementById('citationsSection');
      const citList = document.getElementById('citationsList');
      citList.innerHTML = '';

      if (data.citations && data.citations.length > 0) {
        citSection.classList.remove('hidden');
        data.citations.forEach(c => {
          const card = document.createElement('div');
          card.className = 'p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5';
          card.innerHTML = `
            <div class="flex items-center justify-between text-xs font-mono text-red-400 font-medium">
              <span><i class="fa-regular fa-file-pdf mr-1"></i> ${c.id}</span>
              <span class="text-slate-400">Trang ${c.page}</span>
            </div>
            <p class="text-xs text-slate-300 line-clamp-3 italic">"${c.text}"</p>
          `;
          citList.appendChild(card);
        });
      } else {
        citSection.classList.add('hidden');
      }

      // Timeline Events
      const timeline = document.getElementById('traceTimeline');
      timeline.innerHTML = '';
      document.getElementById('eventsCount').innerText = `${data.events.length} sự kiện`;

      data.events.forEach((e, idx) => {
        const item = document.createElement('div');
        item.className = 'p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-xs space-y-2';

        if (e.event === 'search') {
          item.innerHTML = `
            <div class="flex items-center justify-between font-semibold text-slate-200">
              <span class="text-blue-400"><i class="fa-solid fa-magnifying-glass mr-1.5"></i> Bước ${idx+1}: Tool Call Search</span>
              <span class="text-slate-400">${e.results ? e.results.length : 0} chunks tìm thấy</span>
            </div>
            <div class="text-slate-300">Từ khóa: <code class="bg-slate-800 px-2 py-0.5 rounded text-indigo-300 font-mono">"${e.query}"</code></div>
            <div class="space-y-1 pl-2 border-l-2 border-slate-800">
              ${(e.results || []).map(r => `
                <div class="text-slate-400 flex justify-between font-mono">
                  <span>• ${r.id} (Trang ${r.page})</span>
                  <span class="text-slate-500">score: ${r.score}</span>
                </div>
              `).join('')}
            </div>
          `;
        } else if (e.event === 'answer') {
          item.innerHTML = `
            <div class="font-semibold text-emerald-400">
              <i class="fa-solid fa-check-circle mr-1.5"></i> Bước ${idx+1}: Agent đọc đủ thông tin và kết luận câu trả lời
            </div>
          `;
        } else if (e.event === 'search_limit') {
          item.innerHTML = `
            <div class="font-semibold text-amber-400">
              <i class="fa-solid fa-ban mr-1.5"></i> Bước ${idx+1}: Đạt giới hạn tìm kiếm tối đa (search_limit) -> Từ chối an toàn
            </div>
          `;
        } else {
          item.innerHTML = `
            <div class="text-slate-400">
              <i class="fa-solid fa-info-circle mr-1.5"></i> Bước ${idx+1}: ${e.event}
            </div>
          `;
        }
        timeline.appendChild(item);
      });
    }
  </script>
</body>
</html>
"""

class RAGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean console log
        return

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/query":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)

            question = data.get("question", "").strip()
            strategy = data.get("strategy", "hybrid")
            budget = int(data.get("budget", 1200))
            top_k = int(data.get("top_k", 3))

            if strategy == "dense":
                ret = dense_ret
            elif strategy == "lexical":
                ret = lex_ret
            else:
                ret = hybrid_ret

            t0 = time.perf_counter()
            try:
                res = run_agent(
                    question=question,
                    controller=controller,
                    retriever=ret,
                    max_searches=3,
                    top_k=top_k,
                    budget=budget
                )
                elapsed_s = time.perf_counter() - t0

                # Enrich citations with text & page
                detailed_citations = []
                for cid in res.get("citations", []):
                    c = chunks_map.get(cid, {})
                    detailed_citations.append({
                        "id": cid,
                        "page": c.get("page", 1),
                        "source": c.get("source", "So_tay_sinh_vien_2026.pdf"),
                        "text": c.get("text", "")[:250].strip()
                    })

                output = {
                    "status": res.get("status"),
                    "answer": res.get("answer"),
                    "citations": detailed_citations,
                    "events": res.get("events", []),
                    "elapsed_s": round(elapsed_s, 2)
                }
            except Exception as e:
                output = {
                    "status": "error",
                    "answer": f"Đã xảy ra lỗi: {e}",
                    "citations": [],
                    "events": [],
                    "elapsed_s": round(time.perf_counter() - t0, 2)
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(output, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def main():
    port = 7860
    server = ThreadedHTTPServer(("127.0.0.1", port), RAGHandler)
    print("=" * 70)
    print(f"🚀 WEB UI SẴN SÀNG! ĐANG CHẠY TẠI: http://127.0.0.1:{port}")
    print("Mở trình duyệt của bạn và truy cập địa chỉ trên để gửi câu hỏi kiểm thử.")
    print("Nhấn Ctrl + C để dừng server.")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")

if __name__ == "__main__":
    main()
