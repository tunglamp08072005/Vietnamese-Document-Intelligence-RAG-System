import html
import re
import unicodedata
from collections import Counter


_STOP_WORDS = {
    "các", "của", "được", "để", "đó", "gì", "hay", "khi", "là", "mà", "một",
    "những", "này", "nào", "nên", "nếu", "ra", "sau", "sẽ", "thì", "trong",
    "từ", "và", "về", "với", "cho", "the", "and", "for", "from", "that", "this",
    "what", "which", "who", "with",
}
_PAGE_NUMBER_RE = re.compile(r"\b\d{1,3}\s*/\s*\d{2,3}\b")
_PAGE_HEADER_RE = re.compile(
    r"^\s*KHAI\s+PHÁ\s+DỮ\s+LIỆU\b.*?\b\d{1,3}\s*/\s*\d{2,3}\b\s*",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"\s*[•▪◦]\s*")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _normalize_characters(value: str) -> str:
    value = html.unescape(unicodedata.normalize("NFC", value))
    value = value.replace("\uf0e0", "→").replace("\uf0a7", "•")
    value = "".join(
        char if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Co"} else " "
        for char in value
    )
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _is_navigation_noise(paragraph: str) -> bool:
    lowered = paragraph.casefold()
    if "the first driver" in lowered:
        return True
    if "google scholar" in lowered and any(
        marker in lowered
        for marker in ("bài viết có liên quan", "trích dẫn", "sắp xếp theo", "sort by")
    ):
        return True
    if "bài viết có liên quan" in lowered and "trích dẫn" in lowered:
        return True
    return False


def _clean_paragraph(paragraph: str) -> str:
    paragraph = _normalize_characters(paragraph)
    paragraph = _PAGE_HEADER_RE.sub("", paragraph)
    paragraph = re.sub(r"^\s*\d{1,3}\s*/\s*\d{2,3}\s*", "", paragraph)
    paragraph = re.sub(r"\s+\d{1,3}\s*/\s*\d{2,3}\s*$", "", paragraph)
    paragraph = paragraph.strip()
    if not paragraph or _is_navigation_noise(paragraph):
        return ""

    numbered_items = re.findall(r"(?<!\w)(?:0[1-9]|1\d)(?!\w)", paragraph)
    if paragraph.casefold().startswith("nội dung") and len(numbered_items) >= 3:
        return ""

    paragraph = _BULLET_RE.sub("\n• ", paragraph).strip()
    paragraph = re.sub(r"\s+([,.;:!?])", r"\1", paragraph)
    paragraph = re.sub(r"\n•\s*\n", "\n• ", paragraph)
    return paragraph.strip()


def clean_source_paragraphs(content: str) -> list[str]:
    normalized = _normalize_characters(content)
    paragraphs = re.split(r"\n\s*\n+", normalized)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        value = _clean_paragraph(paragraph)
        if value:
            cleaned.extend(
                part.strip()
                for part in re.split(r"\n(?=•\s)", value)
                if part.strip()
            )
    return cleaned


def clean_source_text(content: str) -> str:
    return "\n\n".join(clean_source_paragraphs(content))


def clean_pdf_page_text(content: str) -> str:
    """Remove slide chrome and preserve readable headings and list boundaries."""
    lines = [line.strip() for line in _normalize_characters(content).splitlines() if line.strip()]
    page_counter = next(
        (index for index, line in enumerate(lines[:6]) if re.fullmatch(r"\d{1,3}\s*/\s*\d{2,3}", line)),
        None,
    )
    if page_counter is not None:
        lines = lines[page_counter + 1 :]

    paragraphs: list[str] = []
    current = ""
    bullet_pending = False
    for line in lines:
        if re.fullmatch(r"[•▪◦]", line):
            bullet_pending = True
            continue
        if _PAGE_NUMBER_RE.fullmatch(line):
            continue
        if bullet_pending:
            line = f"• {line}"
            bullet_pending = False

        is_list_item = line.startswith(("• ", "▪ ", "◦ ")) or re.match(r"^(?:\d{1,2}[.)]|[-–])\s", line)
        if is_list_item:
            if current:
                paragraphs.append(current)
            current = line
        elif current and re.search(r"[.!?;:]\s*$", current):
            paragraphs.append(current)
            current = line
        else:
            current = f"{current} {line}".strip()

    if current:
        paragraphs.append(current)

    return clean_source_text("\n\n".join(paragraphs))


def _keywords(value: str) -> Counter[str]:
    words = Counter(token.casefold() for token in _TOKEN_RE.findall(value))
    return Counter({word: count for word, count in words.items() if len(word) > 2 and word not in _STOP_WORDS})


def citation_excerpt(content: str, question: str, answer: str, max_chars: int = 1500) -> str:
    """Return a short, exact, readable excerpt from the retrieved passage."""
    paragraphs = clean_source_paragraphs(content)
    if not paragraphs:
        return ""

    question_terms = set(_keywords(question))
    answer_text = re.sub(r"\[\d+\]", "", answer)
    answer_terms = set(_keywords(answer_text))
    ranked: list[tuple[float, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        if len(paragraph) < 48 and not re.search(r"[.!?;}]\s*$", paragraph):
            continue
        if paragraph.casefold().startswith("• động lực:") and "động lực" not in f"{question} {answer_text}".casefold():
            continue
        terms = set(_keywords(paragraph))
        score = 3 * len(terms & question_terms) + len(terms & answer_terms)
        if score:
            ranked.append((score / max(1, len(terms)) ** 0.5, index, paragraph))

    if not ranked:
        selected = paragraphs[:2]
    else:
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected_rows = sorted(ranked[:6], key=lambda item: item[1])
        selected = [item[2] for item in selected_rows]

    excerpt = "\n\n".join(selected)
    if len(excerpt) <= max_chars:
        return excerpt

    sentences = re.split(r"(?<=[.!?])\s+", excerpt)
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*kept, sentence]).strip()
        if len(candidate) > max_chars:
            break
        kept.append(sentence)
    if kept:
        return " ".join(kept)
    return sentences[0].strip()
