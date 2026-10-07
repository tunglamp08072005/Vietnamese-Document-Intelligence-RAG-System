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
_SOURCE_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:"
    r"BỘ\s+TÀI\s+NGUYÊN\s+VÀ\s+MÔI\s+TRƯỜNG\b|"
    r"CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\b|"
    r"CỘNG\s+HOÀ\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\b|"
    r"THÔNG\s+TƯ(?=\s+Ban\s+hành\b)|"
    r"(?i:Độc\s+lập\s*[-–]|Hà\s+Nội,\s*ngày\b|Số\s*:|Căn\s+cứ\b|"
    r"Điều\s+\d+\s*[.:]|Chương\s+[IVXLCDM\d]+\b|"
    r"Nơi\s+nhận\s*:|KT\.\s*BỘ\s+TRƯỞNG\b|"
    r"\d{1,2}\.\s+(?=(?:QCVN|Thông\s+tư\s+này|Kể\s+từ|Các\s+))|"
    r"(?:[a-gđ]\)\s+(?=(?:QCVN|Thông\s+tư|Các\s+)))"
    r")))"
)
_INCOMPLETE_ISSUE_DATE_RE = re.compile(
    r"\bngày\s+tháng\s+năm\s+\d{4}\b", re.IGNORECASE
)
_COMPLETE_DATE_RE = re.compile(
    r"\bngày\s*(\d{1,2})\s+tháng\s*(\d{1,2})\s+năm\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_QCVN_CODE_RE = re.compile(r"QCVN\s+\d{1,2}:\d{4}/BTNMT", re.IGNORECASE)
_NAMED_TERM_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:-[A-Za-z0-9]+)*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]-[A-Z][a-z]+)\b"
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
    if paragraph.casefold().startswith("hà nội, ngày"):
        paragraph = re.sub(r"(?i)(\bnăm\s+\d{4})\d{4}\b", r"\1", paragraph)
    if paragraph.casefold().startswith("hà nội, ngày") and _INCOMPLETE_ISSUE_DATE_RE.search(paragraph):
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
    # Older indexed chunks may have flattened an entire PDF page into one line.
    # Recover obvious legal-document headings and list boundaries for citations.
    normalized = _SOURCE_BOUNDARY_RE.sub("\n\n", normalized)
    normalized = re.sub(r"(?<=THÔNG TƯ)\s+(?=Ban hành\b)", "\n\n", normalized)
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

        if line.casefold().startswith("hà nội, ngày"):
            # Some PDFs contain an overlapping, stale year in the text layer
            # (for example 20232017 while the visible year is 2023).
            line = re.sub(r"(?i)(\bnăm\s+\d{4})\d{4}\b", r"\1", line)
            # Do not expose a visibly incomplete issue date as malformed text.
            if _INCOMPLETE_ISSUE_DATE_RE.search(line):
                continue

        is_list_item = line.startswith(("• ", "▪ ", "◦ ")) or re.match(r"^(?:\d{1,2}[.)]|[-–])\s", line)
        is_heading = bool(
            re.match(
                r"^(?:BỘ\s+TÀI\s+NGUYÊN|CỘNG\s+HOÀ?\s+XÃ\s+HỘI|Độc\s+lập\b|"
                r"Hà\s+Nội,\s*ngày|Số\s*:|Căn\s+cứ\b|THÔNG\s+TƯ\b|Điều\s+\d+\s*[.:]|"
                r"Chương\s+[IVXLCDM\d]+\b|Nơi\s+nhận\s*:|KT\.\s*BỘ\s+TRƯỞNG\b)",
                line,
                re.IGNORECASE,
            )
        ) or (len(line) > 3 and line == line.upper() and any(char.isalpha() for char in line))
        if is_list_item or is_heading or current.strip().casefold() == "thông tư":
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


def add_qcvn_citations(answer: str, sources: list[dict], cited_numbers: set[int]) -> tuple[str, set[int]]:
    """Attach page-specific citations when an answer names QCVN codes across chunks."""
    answer_codes = {
        re.sub(r"\s+", "", match.group()).casefold()
        for match in _QCVN_CODE_RE.finditer(answer)
    }
    if not answer_codes:
        return answer, cited_numbers

    for index, source in enumerate(sources, start=1):
        if index in cited_numbers:
            continue
        for match in _QCVN_CODE_RE.finditer(source.get("content", "")):
            code = re.sub(r"\s+", "", match.group()).casefold()
            if code not in answer_codes:
                continue
            parts = re.fullmatch(r"QCVN\s+(\d{1,2}):(\d{4})/BTNMT", match.group(), re.IGNORECASE)
            code_pattern = re.compile(
                rf"QCVN\s+{parts.group(1)}:{parts.group(2)}/BTNMT(?!\s*\[{index}\])",
                re.IGNORECASE,
            )
            answer, inserted = code_pattern.subn(lambda item: f"{item.group()} [{index}]", answer, count=1)
            if inserted:
                cited_numbers.add(index)
    return answer, cited_numbers


def align_named_term_citations(answer: str, sources: list[dict]) -> str:
    """Align citations for named algorithms/acronyms to chunks containing those terms."""
    answer_terms = {
        term.casefold()
        for term in _NAMED_TERM_RE.findall(answer)
        if term.casefold() not in {"qcvn", "btnmt"}
    }
    if not answer_terms:
        return answer

    support_by_source = {
        index: {
            term
            for term in answer_terms
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", source.get("content", ""), re.IGNORECASE)
        }
        for index, source in enumerate(sources, start=1)
    }
    if not any(support_by_source.values()):
        return answer

    # A source citation that supports none of the answer's named terms is likely
    # a misleading citation when another retrieved chunk contains those terms.
    unsupported_numbers = {
        index for index, terms in support_by_source.items() if not terms
    }
    for index in unsupported_numbers:
        answer = re.sub(rf"\s*\[{index}\]", "", answer)

    lines = answer.splitlines(keepends=True)
    aligned_lines: list[str] = []
    for line in lines:
        line_terms = {
            term.casefold()
            for term in _NAMED_TERM_RE.findall(line)
            if term.casefold() not in {"qcvn", "btnmt"}
        }
        supported = {
            index: len(terms & line_terms)
            for index, terms in support_by_source.items()
            if terms & line_terms
        }
        if not supported:
            aligned_lines.append(line)
            continue

        best_count = max(supported.values())
        best_sources = {index for index, count in supported.items() if count == best_count}
        body = re.sub(
            r"\s*\[(\d+)\]",
            lambda match: match.group() if int(match.group(1)) in best_sources else "",
            line.rstrip("\r\n"),
        ).rstrip()
        existing = {
            int(number)
            for number in re.findall(r"\[(\d+)\]", body)
        }
        body += "".join(f" [{index}]" for index in sorted(best_sources - existing))
        ending = line[len(line.rstrip("\r\n")):]
        aligned_lines.append(body + ending)

    return "".join(aligned_lines)


def _keywords(value: str) -> Counter[str]:
    words = Counter(token.casefold() for token in _TOKEN_RE.findall(value))
    return Counter({word: count for word, count in words.items() if len(word) > 2 and word not in _STOP_WORDS})


def align_answer_citations(answer: str, sources: list[dict]) -> str:
    """Align each answer line to the retrieved passage that best supports it."""
    parts = re.split(r"(\n+)", answer)
    source_terms = {
        index: set(_keywords(source.get("content", "")))
        for index, source in enumerate(sources, start=1)
    }
    for part_index in range(0, len(parts), 2):
        paragraph = parts[part_index]
        paragraph_text = re.sub(r"\[\d+\]", "", paragraph).strip()
        paragraph_terms = set(_keywords(paragraph_text))
        if len(paragraph_terms) < 3:
            continue

        qcvn_codes = {
            re.sub(r"\s+", "", match.group()).casefold()
            for match in _QCVN_CODE_RE.finditer(paragraph_text)
        }
        answer_dates = {
            tuple(match.groups())
            for match in _COMPLETE_DATE_RE.finditer(paragraph_text)
        }
        date_source_indexes: set[int] = set()
        date_scores: dict[int, float] = {}
        if answer_dates:
            for index, source in enumerate(sources, start=1):
                source_dates = {
                    tuple(match.groups())
                    for match in _COMPLETE_DATE_RE.finditer(source.get("content", ""))
                }
                overlap = answer_dates & source_dates
                if overlap:
                    date_source_indexes.add(index)
                    date_scores[index] = len(overlap) / len(answer_dates)
            if not date_source_indexes:
                years = {date[2] for date in answer_dates}
                date_source_indexes = {
                    index
                    for index, source in enumerate(sources, start=1)
                    if any(
                        re.search(
                            rf"\bngày\s+tháng\s+năm\s+{re.escape(year)}\b",
                            source.get("content", ""),
                            re.IGNORECASE,
                        )
                        for year in years
                    )
                }
                date_scores = {index: 1.0 for index in date_source_indexes}
        scores: dict[int, float] = {}
        for index, source in enumerate(sources, start=1):
            content = source.get("content", "")
            terms = source_terms[index]
            if answer_dates:
                if index in date_source_indexes:
                    scores[index] = date_scores[index]
                continue
            if qcvn_codes:
                matching_contexts = []
                for match in _QCVN_CODE_RE.finditer(content):
                    code = re.sub(r"\s+", "", match.group()).casefold()
                    if code not in qcvn_codes:
                        continue
                    matching_contexts.append(
                        content[max(0, match.start() - 180):min(len(content), match.end() + 220)]
                    )
                if not matching_contexts:
                    continue

                # A legal code can be mentioned again in a later transition clause.
                # Match its nearby subject (for example, "chất lượng nước mặt") so
                # the citation points to the clause that defines the listed standard.
                subject_terms = paragraph_terms - {"qcvn", "btnmt", "2023"}
                best_local_overlap = max(
                    len(subject_terms & set(_keywords(context)))
                    for context in matching_contexts
                )
                source_text = _normalize_characters(content).casefold()
                normalized_paragraph = _normalize_characters(paragraph_text).casefold()
                descriptors = [
                    phrase for phrase in ("chất lượng đất", "chất lượng không khí", "chất lượng nước mặt",
                                          "chất lượng nước dưới đất", "chất lượng nước biển")
                    if phrase in normalized_paragraph
                ]
                phrase_bonus = 3 if any(phrase in source_text for phrase in descriptors) else 0
                scores[index] = best_local_overlap + phrase_bonus
                continue

            overlap = len(paragraph_terms & terms)
            recall = overlap / len(paragraph_terms)
            if overlap >= 3 and recall >= 0.3:
                scores[index] = recall

        if not scores:
            continue
        best_score = max(scores.values())
        # Choose the strongest supporting passage. Allow exact ties so a sentence
        # that genuinely spans adjacent pages can retain both page references.
        if qcvn_codes:
            supported = {index for index, score in scores.items() if score == best_score}
        else:
            supported = {
                index for index, score in scores.items()
                if score >= best_score * 0.98
            }
        if not supported:
            continue

        clean_paragraph = re.sub(r"\s*\[\d+\]", "", paragraph.rstrip()).rstrip()
        clean_paragraph += "".join(f" [{index}]" for index in sorted(supported))
        parts[part_index] = clean_paragraph + paragraph[len(paragraph.rstrip()):]
    return "".join(parts)


def citation_excerpt(content: str, question: str, answer: str, max_chars: int = 1500) -> str:
    """Return a short, exact, readable excerpt from the retrieved passage."""
    paragraphs = clean_source_paragraphs(content)
    if not paragraphs:
        return ""

    question_terms = set(_keywords(question))
    answer_text = re.sub(r"\[\d+\]", "", answer)
    answer_terms = set(_keywords(answer_text))
    cited_qcvn_codes = {
        re.sub(r"\s+", "", code).casefold()
        for code in _QCVN_CODE_RE.findall(f"{question} {answer}")
    }
    ranked: list[tuple[float, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        paragraph_qcvn_codes = {
            re.sub(r"\s+", "", code).casefold()
            for code in _QCVN_CODE_RE.findall(paragraph)
        }
        if cited_qcvn_codes and (
            not paragraph_qcvn_codes or not cited_qcvn_codes.intersection(paragraph_qcvn_codes)
        ):
            continue
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
