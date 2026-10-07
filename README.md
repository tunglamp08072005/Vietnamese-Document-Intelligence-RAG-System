# Vietnamese Document Intelligence & RAG

MVP hỏi đáp trên tài liệu tiếng Việt. Hệ thống nhận PDF/DOCX, trích xuất văn bản, chia đoạn có metadata trang, tạo embedding bằng BGE-M3, lập chỉ mục trong Qdrant và kết hợp dense search với BM25 bằng Reciprocal Rank Fusion (RRF). API dùng Ollama để tổng hợp câu trả lời tiếng Việt có citation; nếu Ollama chưa sẵn sàng, giao diện hướng dẫn bật model và vẫn hiển thị các nguồn tìm được.

## Tính năng

- Tải lên PDF/DOCX, chống lập chỉ mục trùng theo SHA-256 và xóa tài liệu.
- Lưu metadata và nội dung chunk trong SQLite; lưu vector trong Qdrant.
- Tìm kiếm dense + BM25 + RRF, lọc theo tài liệu và tùy chọn cross-encoder reranker.
- Trích dẫn file và số trang PDF trong API/UI.
- API FastAPI, giao diện Streamlit, chạy local hoặc Docker Compose.

PDF scan chưa được OCR; PDF cần có lớp văn bản. DOCX không có thông tin số trang đáng tin cậy nên citation chỉ tên file.

## Chạy local

Cần Python 3.11 trở lên. Tạo môi trường và cài thư viện:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Khởi động API trong terminal thứ nhất:

```powershell
uvicorn app.main:app --reload
```

Khởi động UI trong terminal thứ hai:

```powershell
streamlit run frontend/streamlit_app.py
```

Mở `http://localhost:8501`; tài liệu API ở `http://localhost:8000/docs`. Lần đầu lập chỉ mục sẽ tải model `BAAI/bge-m3` từ Hugging Face và cần kết nối mạng; những lần sau model đã nằm trong cache. Qdrant chạy embedded, lưu dữ liệu tại `./data/qdrant`; chỉ chạy một process API khi dùng chế độ này.

Quá trình upload ghi thời gian đọc file, trích xuất, chunking, embedding và Qdrant vào terminal API. Mặc định Qdrant ghi theo lô 128 chunks để giảm số lượt ghi; có thể điều chỉnh `INDEX_BATCH_SIZE`. Nếu đủ bộ nhớ, tăng `EMBEDDING_BATCH_SIZE` từ 8 lên 16 hoặc 32 để thử tăng tốc embedding; nếu gặp lỗi thiếu bộ nhớ, giảm về 4 hoặc 8.

## Bật Ollama để tổng hợp câu trả lời

Cài và khởi động Ollama riêng, sau đó tải một model chat. Ví dụ:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

Tải model bằng `ollama pull qwen2.5:7b`. Tên trong `.env` phải khớp model đã cài. Nếu `OLLAMA_MODEL` để trống, API tự chọn model chat Ollama đã cài (ưu tiên Qwen); câu trả lời được yêu cầu bằng tiếng Việt, ngắn gọn và có citation. Nếu Ollama chưa chạy hoặc chưa có model chat, giao diện sẽ báo rõ cách bật thay vì hiển thị các đoạn trích dài. Có thể bật reranker bằng `RERANKER_ENABLED=true`; lần đầu dùng sẽ tải `BAAI/bge-reranker-v2-m3`.

## Chạy bằng Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

UI ở `http://localhost:8501`, API ở `http://localhost:8000/docs`, Qdrant dashboard ở `http://localhost:6333/dashboard`. Để dùng Ollama chạy trên máy host, đặt `OLLAMA_MODEL` trong `.env`; Compose mặc định trỏ API tới `host.docker.internal:11434`.

## API

- `POST /documents/upload` — multipart field `file`, tự trích xuất, chunk, embed và index.
- `POST /documents/index` — lập chỉ mục lại tất cả hoặc `{"document_ids": ["..."]}`.
- `GET /documents` — danh sách tài liệu.
- `DELETE /documents/{id}` — xóa metadata, chunks và vector.
- `POST /query` — `{"question": "...", "document_ids": null}`. Model tự chọn và trích dẫn các nguồn thực sự dùng trong câu trả lời; `RETRIEVAL_TOP_K` chỉ giới hạn số đoạn ứng viên tối đa gửi vào context.

Mỗi nguồn trong câu trả lời có `citation`, `filename`, `page_start`, `page_end`, `chunk_id`, điểm retrieval và nội dung dùng làm ngữ cảnh.

## Cấu hình chính

Các biến môi trường có trong `.env.example`. `QDRANT_URL` để trống sẽ dùng Qdrant embedded; đặt URL để kết nối Qdrant server. `EMBEDDING_MODEL` mặc định là BGE-M3 (1024 chiều). Thay model embedding cần tương thích kích thước collection Qdrant.

## Hướng phát triển

MVP này hoàn thành luồng ingestion, hybrid retrieval, citation và generation cơ bản. Các phần tiếp theo có thể thêm OCR, bộ dữ liệu đánh giá Recall@K/MRR, đánh giá faithfulness/citation, query rewriting, phân quyền và vận hành production.
