"""Agent Controller using Google Gemini API for Day 7."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Các mã lỗi tạm thời, đáng để thử lại. Lỗi khác (404 sai model, 400/403 sai key) dừng ngay.
RETRYABLE_CODES = (429, 500, 503)
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_MS = 30_000  # 30 giây mỗi lần gọi


class GeminiController:
    """Controller calling Google Gemini model with strict JSON actions."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-3.1-flash-lite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY chưa được thiết lập! "
                "Vui lòng thêm GEMINI_API_KEY vào file .env"
            )
        # Đặt timeout và tắt retry ngầm của SDK để tự kiểm soát retry bên dưới
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self.model = model

    def _build_prompts(
        self, question: str, history: list[dict[str, Any]]
    ) -> tuple[str, str]:
        system_instruction = (
            "Bạn là một Agent chuyên truy xuất thông tin chính xác từ tài liệu 'Sổ tay sinh viên Bách khoa 2026'.\n"
            "Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh được cung cấp (context) hoặc yêu cầu tìm kiếm thêm.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Nếu chưa có ngữ cảnh (history rỗng) hoặc ngữ cảnh hiện tại chưa đủ thông tin, bạn PHẢI chọn action 'search':\n"
            '   {"action": "search", "query": "<từ khóa tìm kiếm tiếng Việt ngắn gọn>"}\n'
            "   - LƯU Ý: Ở lượt tìm kiếm đầu tiên, hãy giữ nguyên các từ khóa cốt lõi quan trọng từ câu hỏi của người dùng (tên riêng, số hiệu nghị định, số liệu, từ viết tắt).\n"
            "2. Nếu ngữ cảnh đã chứa ĐỦ bằng chứng để trả lời câu hỏi, bạn PHẢI chọn action 'answer':\n"
            '   {"action": "answer", "answer": "<nội dung câu trả lời kèm citation [chunk-id]>", "citations": ["<chunk-id>"]}\n'
            "   - Trong phần 'answer', bắt buộc phải chèn đúng định dạng [chunk-id] ngay sau thông tin được dẫn chứng (ví dụ: 'Thành lập ngày 6/3/1956 [hust_student_handbook_2026_p2_c000_para]').\n"
            "   - Danh sách 'citations' phải chứa các chunk-id tương ứng có mặt trong ngữ cảnh và hỗ trợ câu trả lời.\n"
            "3. Nếu ngữ cảnh KHÔNG có bằng chứng sau khi tìm kiếm và câu hỏi nằm ngoài tài liệu (hoặc không thể tìm thấy), bạn BẮT BUỘC phải trả về:\n"
            '   {"action": "answer", "answer": "Không đủ bằng chứng trong tài liệu để trả lời.", "citations": []}\n'
            "   - TUYỆT ĐỐI KHÔNG đưa bất kỳ chunk-id nào vào danh sách 'citations' khi từ chối trả lời.\n"
            "4. Đầu ra của bạn BẮT BUỘC chỉ là một chuỗi JSON hợp lệ duy nhất, không thêm bất kỳ văn bản nào bên ngoài JSON."
        )

        context_str = ""
        past_queries: list[str] = []
        if history:
            latest = history[-1]
            past_queries = latest.get("queries", [])
            context_str = latest.get("context", "")

        user_prompt = f"CÂU HỎI CỦA NGƯỜI DÙNG: {question}\n\n"
        if past_queries:
            user_prompt += f"Các truy vấn tìm kiếm đã thực hiện: {', '.join(past_queries)}\n\n"
        if context_str:
            user_prompt += f"NGỮ CẢNH TÀI LIỆU THU THẬP ĐƯỢC:\n{context_str}\n\n"
        else:
            user_prompt += "Hiện tại chưa có ngữ cảnh tài liệu nào.\n\n"

        user_prompt += (
            "Hãy quyết định hành động tiếp theo bằng JSON:\n"
            "- Hoặc {'action': 'search', 'query': '...'}\n"
            "- Hoặc {'action': 'answer', 'answer': '... [chunk-id]', 'citations': ['chunk-id']}\n"
        )
        return system_instruction, user_prompt

    def _call_llm(self, system_instruction: str, user_prompt: str) -> Any:
        """Gọi Gemini, chỉ retry lỗi tạm thời, tối đa MAX_ATTEMPTS lần."""
        last_err: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
            except errors.APIError as e:
                last_err = e
                if e.code not in RETRYABLE_CODES:
                    # Lỗi vĩnh viễn (sai model, sai key...): dừng ngay, không chờ vô ích
                    raise
                wait_time = 20 if e.code == 429 else (attempt * 3)
                print(
                    f"  [Gemini Retry {attempt}/{MAX_ATTEMPTS}] lỗi {e.code} -> chờ {wait_time}s",
                    flush=True,
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(wait_time)
            except Exception as e:  # timeout mạng, mất kết nối...
                last_err = e
                print(
                    f"  [Gemini Retry {attempt}/{MAX_ATTEMPTS}] {e!r}",
                    flush=True,
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(3)

        raise RuntimeError(
            f"Gemini API failed after {MAX_ATTEMPTS} attempts: {last_err}"
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        raw_text = raw_text.strip()
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"^```\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return raw_text.strip()

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide next action: search or answer with citations."""
        system_instruction, user_prompt = self._build_prompts(question, history)

        response = self._call_llm(system_instruction, user_prompt)
        raw_text = self._clean_json_text(response.text or "")

        try:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict) or "action" not in parsed:
                raise ValueError("JSON không chứa key 'action'")
            return parsed
        except Exception as e:
            # In ra để phân biệt "lỗi parse" với "abstention đúng" khi đánh giá
            print(
                f"  [Parse lỗi] {e!r} | raw_text={raw_text[:200]!r}",
                flush=True,
            )
            if "search" in raw_text.lower():
                return {"action": "search", "query": question, "_parse_error": True}
            return {
                "action": "answer",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "_parse_error": True,
            }