import io
import re
import unicodedata
from dataclasses import dataclass

import pymupdf
from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph


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


def _docx_blocks(document: DocumentType):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


OCR_MIN_EMBEDDED_TEXT_CHARS = 1200
OCR_MIN_IMAGE_COVERAGE = 0.35


def _page_needs_ocr(page: pymupdf.Page, text: str) -> bool:
    page_area = page.rect.get_area()
    if len(text) >= OCR_MIN_EMBEDDED_TEXT_CHARS or page_area <= 0:
        return False
    return any(
        image_rect.get_area() / page_area >= OCR_MIN_IMAGE_COVERAGE
        for image in page.get_images(full=True)
        for image_rect in page.get_image_rects(image[0])
    )


def extract_pdf(
    data: bytes,
    ocr_language: str = "vie+eng",
    ocr_dpi: int = 250,
) -> list[TextPage]:
    pages: list[TextPage] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for index, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if _page_needs_ocr(page, text):
                try:
                    textpage = page.get_textpage_ocr(language=ocr_language, dpi=ocr_dpi, full=True)
                    ocr_text = clean_text(page.get_text("text", textpage=textpage))
                except Exception as error:
                    raise ValueError(
                        f"Không thể OCR trang {index} của PDF scan. Hãy cài Tesseract OCR cùng dữ liệu "
                        f"ngôn ngữ {ocr_language}, rồi khởi động lại API. Chi tiết: {error}"
                    ) from error
                if len(ocr_text) > len(text):
                    text = ocr_text
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
    sentences = re.split(r"(?<=[.!?。])\s+", text)
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
        for paragraph in re.split(r"\n{1,2}", page.text):
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
            words = current.split()
            tail = ""
            tail_limit = max(0, min(overlap_chars, max_chars - len(text) - 2))
            for word in reversed(words):
                candidate_tail = f"{word} {tail}".strip()
                if len(candidate_tail) > tail_limit:
                    break
                tail = candidate_tail
            current = tail
            current_pages = [current_pages[-1]] if current_pages else []
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
