import asyncio
import os
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import httpx
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_QUERY_TIMEOUT_SECONDS = int(os.getenv("API_QUERY_TIMEOUT_SECONDS", "660"))
st.set_page_config(page_title="Vietnamese Document Intelligence", page_icon="📚", layout="wide")
st.title("Vietnamese Document Intelligence")
st.caption("Tải tài liệu PDF/DOCX lên, sau đó hỏi đáp dựa trên nội dung đã lập chỉ mục.")

st.session_state.setdefault("active_operation", None)
st.session_state.setdefault("operation_feedback", None)
st.session_state.setdefault("last_query_result", None)
st.session_state.setdefault("chat_history", [])
if not st.session_state.chat_history and st.session_state.last_query_result:
    st.session_state.chat_history.append({"role": "assistant", "result": st.session_state.last_query_result})


@st.cache_resource
def get_operation_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-ui")


def api_error(response: httpx.Response | requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail", body))
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def show_request_error(action: str, error: requests.RequestException) -> None:
    if isinstance(error, requests.ConnectionError):
        st.error(f"{action}: Không kết nối được API tại {API_URL}. API có thể chưa chạy.")
        st.caption("Với chế độ chạy local, mở terminal khác trong thư mục dự án và chạy:")
        st.code("uvicorn app.main:app --reload", language="powershell")
        st.caption("Nếu dùng Docker Compose, chạy `docker compose up --build` tại thư mục dự án.")
    elif isinstance(error, requests.Timeout):
        st.error(f"{action}: API tại {API_URL} phản hồi quá thời gian chờ. Hãy kiểm tra terminal đang chạy API.")
    else:
        st.error(f"{action}: {error}")


@st.cache_data(ttl=10, max_entries=4, show_spinner=False)
def load_documents(api_url: str) -> list[dict]:
    response = requests.get(f"{api_url}/documents", timeout=10)
    response.raise_for_status()
    return response.json()


def start_operation(kind: str, submit_request, *args) -> None:
    operation_id = str(uuid4())
    try:
        registration = requests.post(f"{API_URL}/operations/{operation_id}", timeout=10)
        if not registration.ok:
            st.error(api_error(registration))
            return
    except requests.RequestException as error:
        show_request_error("Không thể bắt đầu thao tác", error)
        return

    cancel_event = Event()
    future = get_operation_executor().submit(submit_request, operation_id, cancel_event, *args)
    st.session_state.active_operation = {
        "id": operation_id,
        "kind": kind,
        "future": future,
        "cancel_event": cancel_event,
        "cancel_requested": False,
    }
    st.session_state.operation_feedback = None


async def post_with_cancellation(url: str, cancel_event: Event, **kwargs):
    timeout = kwargs.pop("timeout")
    if cancel_event.is_set():
        return None
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout[1], connect=timeout[0])) as client:
        request_task = asyncio.create_task(client.post(url, **kwargs))
        while not request_task.done():
            if cancel_event.is_set():
                request_task.cancel()
                with suppress(asyncio.CancelledError):
                    await request_task
                return None
            await asyncio.wait({request_task}, timeout=0.1)
        return await request_task


def submit_upload(operation_id: str, cancel_event: Event, filename: str, content: bytes, mime_type: str | None):
    try:
        return asyncio.run(
            post_with_cancellation(
                f"{API_URL}/documents/upload",
                cancel_event,
                files={"file": (filename, content, mime_type or "application/octet-stream")},
                headers={"X-Operation-ID": operation_id},
                timeout=(10, 1800),
            )
        )
    except httpx.ConnectError as error:
        raise requests.ConnectionError(str(error)) from error
    except httpx.TimeoutException as error:
        raise requests.Timeout(str(error)) from error
    except httpx.RequestError as error:
        raise requests.RequestException(str(error)) from error


def submit_query(operation_id: str, cancel_event: Event, question: str):
    try:
        return asyncio.run(
            post_with_cancellation(
                f"{API_URL}/query",
                cancel_event,
                json={"question": question},
                headers={"X-Operation-ID": operation_id},
                timeout=(10, API_QUERY_TIMEOUT_SECONDS),
            )
        )
    except httpx.ConnectError as error:
        raise requests.ConnectionError(str(error)) from error
    except httpx.TimeoutException as error:
        raise requests.Timeout(str(error)) from error
    except httpx.RequestError as error:
        raise requests.RequestException(str(error)) from error


def finish_operation(operation: dict) -> None:
    try:
        response = operation["future"].result()
    except requests.RequestException as error:
        if operation["cancel_requested"]:
            feedback = {"kind": operation["kind"], "cancelled": True}
        else:
            error_type = "connection" if isinstance(error, requests.ConnectionError) else (
                "timeout" if isinstance(error, requests.Timeout) else "other"
            )
            feedback = {"kind": operation["kind"], "error_type": error_type, "message": str(error)}
    except Exception as error:
        if operation["cancel_requested"]:
            feedback = {"kind": operation["kind"], "cancelled": True}
        else:
            feedback = {"kind": operation["kind"], "error_type": "other", "message": str(error)}
    else:
        if response is None or (operation["cancel_requested"] and not response.is_success):
            feedback = {"kind": operation["kind"], "cancelled": True}
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = {"detail": response.text or f"HTTP {response.status_code}"}
            if response.is_success:
                feedback = {"kind": operation["kind"], "success": True, "payload": payload}
                if operation["kind"] == "upload":
                    load_documents.clear()
                else:
                    st.session_state.last_query_result = payload
                    st.session_state.chat_history.append({"role": "assistant", "result": payload})
            else:
                feedback = {"kind": operation["kind"], "error": api_error(response)}

    st.session_state.operation_feedback = feedback
    st.session_state.active_operation = None
    st.rerun()


def render_feedback(kind: str) -> None:
    feedback = st.session_state.get("operation_feedback")
    if not feedback or feedback.get("kind") != kind:
        return
    if feedback.get("cancelled"):
        st.info("Thao tác đã được dừng.")
    elif feedback.get("success") and kind == "upload":
        payload = feedback["payload"]
        if payload.get("already_indexed"):
            st.success(f"{payload['filename']} đã được lập chỉ mục trước đó.")
        else:
            st.success(f"{payload['filename']} · {payload['chunk_count']} đoạn.")
    elif feedback.get("error_type") == "connection":
        st.error(f"API bị ngắt kết nối: {feedback['message']}")
        st.caption("Kiểm tra terminal đang chạy API rồi thử lại.")
    elif feedback.get("error_type") == "timeout":
        if kind == "upload":
            st.error(
                "API phản hồi quá thời gian chờ khi xử lý tài liệu. "
                "Có thể do tài liệu có nhiều trang ảnh cần OCR, hoặc lần đầu khởi động "
                "API cần tải model (BAAI/bge-m3 ~40 giây). "
                "Hãy khởi động lại API và thử lại — lần sau sẽ nhanh hơn vì model đã được nạp sẵn."
            )
        else:
            st.error(
                f"API phản hồi quá thời gian chờ: {feedback['message'] or 'Ollama mất quá nhiều thời gian trả lời.'}"
            )
    elif feedback.get("error_type"):
        st.error(feedback["message"])
    elif feedback.get("error"):
        st.error(feedback["error"])


def request_operation_cancel(operation: dict, label: str) -> None:
    if operation["future"].done():
        finish_operation(operation)
        return
    try:
        response = requests.post(f"{API_URL}/operations/{operation['id']}/cancel", timeout=5)
        if not response.ok or not response.json().get("cancelled"):
            st.warning("API không còn nhận diện thao tác này; có thể thao tác đã hoàn tất.")
            return
    except requests.RequestException as error:
        operation["cancel_requested"] = True
        operation["cancel_event"].set()
        show_request_error("Không gửi được lệnh dừng", error)
        st.warning("Giao diện đã ngắt yêu cầu; API có thể còn làm việc nếu lệnh hủy không tới được.")
        return
    operation["cancel_requested"] = True
    operation["cancel_event"].set()
    st.info(f"Đang dừng {label}…")


@st.fragment(run_every=0.5, key="upload_operation_monitor")
def render_upload_operation() -> None:
    operation = st.session_state.get("active_operation")
    if not operation or operation["kind"] != "upload":
        return
    if operation["future"].done():
        finish_operation(operation)
    st.info("Đang tải lên và lập chỉ mục tài liệu…")
    if operation["cancel_requested"]:
        st.caption("Đang chờ API dừng tác vụ hiện tại.")
    elif st.button("Dừng tải lên", key=f"stop-upload-{operation['id']}", width="stretch"):
        request_operation_cancel(operation, "tải lên")


@st.fragment(run_every=0.5, key="query_operation_monitor")
def render_query_operation() -> None:
    operation = st.session_state.get("active_operation")
    if not operation or operation["kind"] != "query":
        return
    if operation["future"].done():
        finish_operation(operation)
    st.info("Đang tìm trong tài liệu và tạo câu trả lời…")
    if operation["cancel_requested"]:
        st.caption("Đang chờ API dừng tác vụ hiện tại.")
    elif st.button("Dừng tìm kiếm", key=f"stop-query-{operation['id']}"):
        request_operation_cancel(operation, "tìm kiếm")


def render_query_result(result: dict) -> None:
    if result["answer_mode"] == "configuration_required":
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])
    if result["sources"]:
        st.caption("Nguồn")
        for source in result["sources"]:
            page = ""
            if source["page_start"] is not None:
                page = f" · trang {source['page_start']}"
                if source["page_end"] != source["page_start"]:
                    page += f"–{source['page_end']}"
            with st.expander(f"[{source['citation']}] {source['filename']}{page}"):
                st.markdown(source["content"])


with st.sidebar:
    st.header("Tài liệu")
    uploaded = st.file_uploader("Chọn tài liệu PDF hoặc DOCX", type=["pdf", "docx"])
    active_operation = st.session_state.get("active_operation")
    if st.button(
        "Tải lên và lập chỉ mục",
        disabled=uploaded is None or active_operation is not None,
        width="stretch",
    ):
        if uploaded is not None:
            start_operation("upload", submit_upload, uploaded.name, uploaded.getvalue(), uploaded.type)
    if (st.session_state.get("active_operation") or {}).get("kind") == "upload":
        render_upload_operation()
    render_feedback("upload")

    st.divider()
    st.subheader("Đã lập chỉ mục")
    try:
        documents = load_documents(API_URL)
        if not documents:
            st.caption("Chưa có tài liệu.")
        for document in documents:
            with st.container(border=True):
                st.markdown(f"**{document['filename']}**")
                st.caption(f"{document['chunk_count']} đoạn · {document['created_at'][:10]}")
                if st.button(
                    "Xóa",
                    key=f"delete-{document['id']}",
                    disabled=st.session_state.get("active_operation") is not None,
                ):
                    try:
                        deletion = requests.delete(f"{API_URL}/documents/{document['id']}", timeout=20)
                        if deletion.ok:
                            load_documents.clear()
                            st.rerun()
                        st.error(api_error(deletion))
                    except requests.RequestException as error:
                        show_request_error("Không xóa được tài liệu", error)
    except requests.RequestException as error:
        show_request_error("Không tải được danh sách tài liệu", error)

st.header("Hỏi đáp")
st.caption("AI sẽ tự chọn các nguồn phù hợp từ tài liệu đã lập chỉ mục.")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            render_query_result(message["result"])

active_operation = st.session_state.get("active_operation")
if (st.session_state.get("active_operation") or {}).get("kind") == "query":
    render_query_operation()
render_feedback("query")
question = st.chat_input(
    "Hỏi tiếp về các tài liệu đã lập chỉ mục…",
    disabled=active_operation is not None,
)
if question:
    if len(question.strip()) < 2:
        st.warning("Câu hỏi cần có ít nhất 2 ký tự.")
    else:
        start_operation("query", submit_query, question.strip())
        if st.session_state.active_operation and st.session_state.active_operation["kind"] == "query":
            st.session_state.chat_history.append({"role": "user", "content": question.strip()})
            st.rerun()
