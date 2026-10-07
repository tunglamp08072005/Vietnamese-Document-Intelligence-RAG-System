import io
import re
import threading
import unicodedata
from dataclasses import dataclass
from threading import Event

import pymupdf
from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.operations import raise_if_cancelled


@dataclass
class TextPage:
    number: int | None
    text: str


@dataclass
class TextChunk:
    content: str
    page_start: int | None
    page_end: int | None


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\x00", " ")
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _strip_page_number_artifacts(value: str) -> str:
    lines = value.splitlines()
    while lines and re.fullmatch(r"\s*\d+\s*", lines[0]):
        lines.pop(0)
    while lines and re.fullmatch(r"\s*\d+\s*", lines[-1]):
        lines.pop()
    return clean_text("\n".join(lines))


def _docx_blocks(document: DocumentType):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


OCR_MIN_EMBEDDED_TEXT_CHARS = 1200
OCR_MIN_IMAGE_COVERAGE = 0.35
_rapid_ocr_engine = None
_rapid_ocr_engine_lock = threading.Lock()
_rapid_ocr_run_lock = threading.Lock()


def _page_needs_ocr(page: pymupdf.Page, text: str) -> bool:
    page_area = page.rect.get_area()
    if len(text) >= OCR_MIN_EMBEDDED_TEXT_CHARS or page_area <= 0:
        return False
    return any(
        image_rect.get_area() / page_area >= OCR_MIN_IMAGE_COVERAGE
        for image in page.get_images(full=True)
        for image_rect in page.get_image_rects(image[0])
    )


def _get_rapid_ocr_engine():
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        with _rapid_ocr_engine_lock:
            if _rapid_ocr_engine is None:
                from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

                _rapid_ocr_engine = RapidOCR(
                    params={
                        "Det.engine_type": EngineType.ONNXRUNTIME,
                        "Det.lang_type": LangDet.CH,
                        "Det.model_type": ModelType.SMALL,
                        "Det.ocr_version": OCRVersion.PPOCRV6,
                        "Rec.engine_type": EngineType.ONNXRUNTIME,
                        "Rec.lang_type": LangRec.VI,
                        "Rec.model_type": ModelType.SMALL,
                        "Rec.ocr_version": OCRVersion.PPOCRV6,
                    }
                )
    return _rapid_ocr_engine


def _extract_page_text_with_rapid_ocr(page: pymupdf.Page, dpi: int) -> str:
    import numpy as np

    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n >= 3:
        image = image[:, :, :3][:, :, ::-1].copy()
    with _rapid_ocr_run_lock:
        result = _get_rapid_ocr_engine()(image)
    return "\n".join(text.strip() for text in (result.txts or ()) if text.strip())


def extract_pdf(
    data: bytes,
    ocr_language: str = "vie+eng",
    ocr_dpi: int = 250,
    cancel_event: Event | None = None,
) -> list[TextPage]:
    pages: list[TextPage] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for index, page in enumerate(pdf, start=1):
            raise_if_cancelled(cancel_event)
            text = _strip_page_number_artifacts(page.get_text("text"))
            if _page_needs_ocr(page, text):
                try:
                    textpage = page.get_textpage_ocr(language=ocr_language, dpi=ocr_dpi, full=True)
                    ocr_text = _strip_page_number_artifacts(page.get_text("text", textpage=textpage))
                except Exception as tesseract_error:
                    raise_if_cancelled(cancel_event)
                    try:
                        ocr_text = clean_text(_extract_page_text_with_rapid_ocr(page, ocr_dpi))
                    except Exception as rapid_ocr_error:
                        raise_if_cancelled(cancel_event)
                        raise ValueError(
                            f"Không thể OCR trang {index} bằng Tesseract hoặc RapidOCR. "
                            "Hãy cài lại phụ thuộc bằng `pip install -r requirements.txt`; "
                            "lần OCR đầu tiên cần Internet để tải model. "
                            f"Tesseract: {tesseract_error}. RapidOCR: {rapid_ocr_error}"
                        ) from rapid_ocr_error
                if len(ocr_text) > len(text):
                    text = ocr_text
            raise_if_cancelled(cancel_event)
            if text:
                pages.append(TextPage(number=index, text=text))
    if not pages:
        raise ValueError("PDF không có nội dung văn bản có thể trích xuất hoặc OCR.")
    return pages


def extract_docx(data: bytes) -> list[TextPage]:
    document = Document(io.BytesIO(data))
    blocks: list[str] = []
    for block in _docx_blocks(document):
        if isinstance(block, Paragraph):
            value = clean_text(block.text)
            if value:
                blocks.append(value)
        else:
            for row in block.rows:
                cells = [clean_text(cell.text) for cell in row.cells]
                line = " | ".join(cell for cell in cells if cell)
                if line:
                    blocks.append(line)
    text = "\n\n".join(blocks)
    if not text:
        raise ValueError("DOCX không chứa văn bản có thể đọc.")
    return [TextPage(number=None, text=text)]


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?;。])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        fragments = [sentence]
        if len(sentence) > max_chars:
            fragments = []
            fragment = ""
            for word in sentence.split():
                if len(word) > max_chars:
                    if fragment:
                        fragments.append(fragment)
                        fragment = ""
                    fragments.extend(word[index : index + max_chars] for index in range(0, len(word), max_chars))
                    continue
                candidate = f"{fragment} {word}".strip()
                if len(candidate) > max_chars and fragment:
                    fragments.append(fragment)
                    fragment = word
                else:
                    fragment = candidate
            if fragment:
                fragments.append(fragment)
        for item in fragments:
            candidate = f"{current} {item}".strip()
            if len(candidate) > max_chars and current:
                pieces.append(current)
                current = item
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(pages: list[TextPage], max_chars: int, overlap_chars: int) -> list[TextChunk]:
    units: list[tuple[str, int | None]] = []
    for page in pages:
        # PDF extraction inserts single newlines at visual line wraps. Join those
        # lines before chunking so a sentence is not exposed as a fragment.
        page_text = re.sub(r"(?<!\n)\n(?!\n)", " ", page.text)
        for paragraph in re.split(r"\n{2,}", page_text):
            paragraph = clean_text(paragraph)
            if paragraph:
                units.extend((part, page.number) for part in _split_oversized(paragraph, max_chars))

    chunks: list[TextChunk] = []
    current = ""
    current_pages: list[int] = []
    for text, page_number in units:
        candidate = f"{current}\n\n{text}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(
                TextChunk(
                    content=current,
                    page_start=min(current_pages) if current_pages else None,
                    page_end=max(current_pages) if current_pages else None,
                )
            )
            tail_limit = max(0, min(overlap_chars, max_chars - len(text) - 2))
            # Keep overlap only at a complete sentence/list-item boundary.
            tail = ""
            if tail_limit:
                sentences = re.split(r"(?<=[.!?;。])\s+", current)
                for sentence in reversed(sentences):
                    candidate_tail = f"{sentence} {tail}".strip()
                    if len(candidate_tail) > tail_limit:
                        break
                    tail = candidate_tail
            current = tail
            current_pages = [current_pages[-1]] if tail and current_pages else []
        current = f"{current}\n\n{text}".strip()
        if page_number is not None:
            current_pages.append(page_number)

    if current:
        chunks.append(
            TextChunk(
                content=current,
                page_start=min(current_pages) if current_pages else None,
                page_end=max(current_pages) if current_pages else None,
            )
        )
    return chunks
