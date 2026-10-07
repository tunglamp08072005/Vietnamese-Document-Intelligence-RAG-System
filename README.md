# Vietnamese Document Intelligence & RAG

MVP hỏi đáp trên tài liệu tiếng Việt. Hệ thống nhận PDF/DOCX, trích xuất văn bản, chia đoạn có metadata trang, tạo embedding bằng BGE-M3, lập chỉ mục trong Qdrant và kết hợp dense search với BM25 bằng Reciprocal Rank Fusion (RRF). Kết quả được trả lời qua Ollama nếu đã cấu hình; nếu chưa, API trả các trích đoạn có nguồn để tránh tự tạo câu trả lời.

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

Mở `http://localhost:8501`; tài liệu API ở `http://localhost:8000/docs`. Lần đầu lập chỉ mục sẽ tải model `BAAI/bge-m3` từ Hugging Face và cần kết nối mạng. Qdrant chạy embedded, lưu dữ liệu tại `./data/qdrant`; chỉ chạy một process API khi dùng chế độ này.

## Bật LLM Ollama

Cài và khởi động Ollama riêng, tải một model chat phù hợp, rồi đặt trong `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

Tên model cần khớp model đã cài trong Ollama. Nếu để trống, API dùng chế độ trích đoạn trực tiếp. Có thể bật reranker bằng `RERANKER_ENABLED=true`; lần đầu dùng sẽ tải `BAAI/bge-reranker-v2-m3`.

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
- `POST /query` — `{"question": "...", "top_k": 5, "document_ids": null}`.

Mỗi nguồn trong câu trả lời có `citation`, `filename`, `page_start`, `page_end`, `chunk_id`, điểm retrieval và nội dung dùng làm ngữ cảnh.

## Cấu hình chính

Các biến môi trường có trong `.env.example`. `QDRANT_URL` để trống sẽ dùng Qdrant embedded; đặt URL để kết nối Qdrant server. `EMBEDDING_MODEL` mặc định là BGE-M3 (1024 chiều). Thay model embedding cần tương thích kích thước collection Qdrant.

## Hướng phát triển

MVP này hoàn thành luồng ingestion, hybrid retrieval, citation và generation cơ bản. Các phần tiếp theo có thể thêm OCR, bộ dữ liệu đánh giá Recall@K/MRR, đánh giá faithfulness/citation, query rewriting, phân quyền và vận hành production.
