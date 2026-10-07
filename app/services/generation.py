import httpx


SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp dựa trên tài liệu tiếng Việt được cung cấp.
Chỉ dùng thông tin trong CONTEXT. Không suy đoán hoặc thêm kiến thức bên ngoài.
Gắn trích dẫn [số] vào các ý được hỗ trợ bởi nguồn tương ứng.
Nếu tài liệu không có câu trả lời, hãy nói rõ: 'Tôi không tìm thấy thông tin này trong các tài liệu đã lập chỉ mục.'"""


class AnswerService:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    async def answer(self, question: str, sources: list[dict]) -> tuple[str, str]:
        if not sources:
            return "Tôi không tìm thấy thông tin liên quan trong các tài liệu đã lập chỉ mục.", "grounded"

        context = "\n\n".join(
            f"[{index}] {source['filename']}{self._page_label(source)}\n{source['content']}"
            for index, source in enumerate(sources, start=1)
        )
        if not self.model:
            excerpts = []
            for index, source in enumerate(sources[:3], start=1):
                excerpt = " ".join(source["content"].split())
                if len(excerpt) > 700:
                    excerpt = excerpt[:697].rsplit(" ", 1)[0] + "..."
                excerpts.append(f"[{index}] {excerpt}")
            return "Các trích đoạn liên quan trong tài liệu:\n\n" + "\n\n".join(excerpts), "extractive"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nCÂU HỎI:\n{question}"},
                    ],
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            payload = response.json()
        answer = payload.get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama trả về câu trả lời trống.")
        return answer, "ollama"

    @staticmethod
    def _page_label(source: dict) -> str:
        start, end = source.get("page_start"), source.get("page_end")
        if start is None:
            return ""
        return f" – trang {start}" if start == end else f" – trang {start}–{end}"
