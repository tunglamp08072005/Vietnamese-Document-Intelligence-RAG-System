import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Vietnamese Document Intelligence", page_icon="📚", layout="wide")
st.title("Vietnamese Document Intelligence")
st.caption("Tải tài liệu PDF/DOCX lên, sau đó hỏi đáp dựa trên nội dung đã lập chỉ mục.")


def api_error(response: requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail", body))
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


@st.cache_data(ttl=10, max_entries=4, show_spinner=False)
def load_documents(api_url: str) -> list[dict]:
    response = requests.get(f"{api_url}/documents", timeout=10)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Tài liệu")
    uploaded = st.file_uploader("Chọn tài liệu PDF hoặc DOCX", type=["pdf", "docx"])
    if st.button("Tải lên và lập chỉ mục", disabled=uploaded is None, width="stretch"):
        file_content = uploaded.getvalue()
        try:
            with st.status("Đang tải lên và lập chỉ mục...", expanded=True) as upload_status:
                st.write(f"Tệp: {uploaded.name} ({len(file_content) / 1024 / 1024:.1f} MB)")
                st.write("PDF đang được trích xuất nội dung và mã hóa bằng BGE-M3.")
                st.caption("Lần đầu chạy có thể lâu hơn vì cần tải model về máy.")
                try:
                    response = requests.post(
                        f"{API_URL}/documents/upload",
                        files={"file": (uploaded.name, file_content, uploaded.type)},
                        timeout=(10, 1800),
                    )
                except requests.RequestException:
                    upload_status.update(label="Không kết nối được API", state="error", expanded=True)
                    raise
                if response.ok:
                    load_documents.clear()
                    payload = response.json()
                    if payload.get("already_indexed"):
                        upload_status.update(label="Tài liệu đã được lập chỉ mục", state="complete", expanded=False)
                    else:
                        upload_status.update(label="Đã lập chỉ mục xong", state="complete", expanded=False)
                    st.success(f"{payload['filename']} · {payload['chunk_count']} đoạn.")
                else:
                    upload_status.update(label="Lập chỉ mục thất bại", state="error", expanded=True)
                    st.error(api_error(response))
        except requests.RequestException as error:
            st.error(f"Không kết nối được API: {error}")

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
                if st.button("Xóa", key=f"delete-{document['id']}"):
                    try:
                        deletion = requests.delete(f"{API_URL}/documents/{document['id']}", timeout=20)
                        if deletion.ok:
                            load_documents.clear()
                            st.rerun()
                        st.error(api_error(deletion))
                    except requests.RequestException as error:
                        st.error(f"Không xóa được tài liệu: {error}")
    except requests.RequestException as error:
        st.warning(f"Không tải được danh sách tài liệu: {error}")

st.header("Hỏi đáp")
question = st.text_area("Câu hỏi", placeholder="Ví dụ: Điều kiện để sinh viên được xét tốt nghiệp là gì?")
top_k = st.slider("Số đoạn nguồn", min_value=1, max_value=10, value=5)
if st.button("Tìm câu trả lời", type="primary", disabled=len(question.strip()) < 2):
    with st.spinner("Đang tìm trong tài liệu..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=300,
            )
            if not response.ok:
                st.error(api_error(response))
            else:
                result = response.json()
                st.subheader("Trả lời")
                if result["answer_mode"] == "configuration_required":
                    st.warning(result["answer"])
                else:
                    st.markdown(result["answer"])
                if result["sources"]:
                    st.subheader("Nguồn")
                    for source in result["sources"]:
                        page = ""
                        if source["page_start"] is not None:
                            page = f" · trang {source['page_start']}"
                            if source["page_end"] != source["page_start"]:
                                page += f"–{source['page_end']}"
                        with st.expander(f"[{source['citation']}] {source['filename']}{page}"):
                            st.write(source["content"])
        except requests.RequestException as error:
            st.error(f"Không kết nối được API: {error}")
