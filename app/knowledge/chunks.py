from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCES = re.compile(r".+?(?:[。！？!?]|(?<=\.)|$)", re.DOTALL)
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5"), ("######", "h6")]
_HEADER_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
_RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""],
)


@dataclass(frozen=True)
class ChunkDraft:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool = False
    is_table: bool = False


@dataclass(frozen=True)
class ImportResult:
    inserted: int
    skipped: int
    ids: tuple[int, ...]


def _sentences(text: str) -> list[str]:
    return [piece.strip() for piece in _SENTENCES.findall(text.strip()) if piece.strip()]


def _overlap(sentences: list[str]) -> list[str]:
    selected: list[str] = []
    length = 0
    for sentence in reversed(sentences):
        if len(sentence) > OVERLAP_CHARS:
            continue
        if selected and length + len(sentence) > OVERLAP_CHARS:
            break
        selected.insert(0, sentence)
        length += len(sentence)
    return selected


def _text_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    raw_sentences = _sentences(text)
    # Recursive splitting finds the normal paragraph/line boundaries for ordinary
    # prose. When it would need its character fallback, retain the original
    # sentence instead: an oversized sentence is a valid indivisible chunk.
    if all(len(sentence) <= MAX_CHUNK_CHARS for sentence in raw_sentences):
        sentences = _sentences("\n".join(_RECURSIVE_SPLITTER.split_text(text)))
    else:
        sentences = raw_sentences
    for sentence in sentences:
        if len(sentence) > MAX_CHUNK_CHARS:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_size = 0
            chunks.append(sentence)
            continue
        if current and current_size + len(sentence) > MAX_CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = _overlap(current)
            current_size = sum(map(len, current))
        current.append(sentence)
        current_size += len(sentence)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _table_chunks(lines: list[str]) -> list[str]:
    if len(lines) <= 2:
        return ["\n".join(lines)]
    header = lines[:2]
    chunks: list[str] = []
    current = header.copy()
    size = sum(len(line) + 1 for line in current)
    for row in lines[2:]:
        if len(current) > 2 and size + len(row) + 1 > MAX_CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = header.copy()
            size = sum(len(line) + 1 for line in current)
        current.append(row)
        size += len(row) + 1
    if len(current) > 2:
        chunks.append("\n".join(current))
    return chunks


def _section_drafts(
    lines: list[str], source: str, headings: list[str], content_type: str
) -> list[ChunkDraft]:
    text_lines = [line.rstrip() for line in lines]
    if not headings or not any(line.strip() for line in text_lines):
        return []
    section_path = " / ".join([source, *headings])
    heading_question = content_type == "faq" and headings[-1].rstrip().endswith(("?", "？"))
    category_headings = headings[:-1] if heading_question else headings
    category = " / ".join(category_headings) or headings[-1]
    heading_question_text = headings[-1] if heading_question else None
    active_question = heading_question_text or headings[-1]
    drafts: list[ChunkDraft] = []
    index = 0
    while index < len(text_lines):
        if not text_lines[index].strip():
            index += 1
            continue
        if text_lines[index].lstrip().startswith("|"):
            table: list[str] = []
            while index < len(text_lines) and text_lines[index].lstrip().startswith("|"):
                table.append(text_lines[index].strip())
                index += 1
            for answer in _table_chunks(table):
                drafts.append(ChunkDraft(category, active_question, answer, section_path, content_type, False, True))
            continue
        if text_lines[index].lstrip().startswith(">"):
            callout: list[str] = []
            while index < len(text_lines) and text_lines[index].lstrip().startswith(">"):
                callout.append(text_lines[index].strip())
                index += 1
            key_clause = any("[!IMPORTANT]" in line for line in callout)
            for answer in _text_chunks("\n".join(callout)):
                drafts.append(ChunkDraft(category, active_question, answer, section_path, content_type, key_clause, False))
            continue
        prose: list[str] = []
        while (index < len(text_lines) and text_lines[index].strip()
               and not text_lines[index].lstrip().startswith(("|", ">"))):
            prose.append(text_lines[index].strip())
            index += 1
        prose_groups: list[tuple[str, list[str]]] = []
        current_prose: list[str] = []
        for line in prose:
            if content_type == "faq" and line.rstrip().endswith(("?", "？")):
                if current_prose:
                    prose_groups.append((active_question, current_prose))
                    current_prose = []
                active_question = line
                continue
            current_prose.append(line)
        if current_prose:
            prose_groups.append((active_question, current_prose))
        for question, group in prose_groups:
            key_clause = any("[!IMPORTANT]" in line for line in group)
            for answer in _text_chunks("\n".join(group)):
                drafts.append(ChunkDraft(category, question, answer, section_path, content_type, key_clause, False))
    return drafts


def parse_markdown(text: str, source: str, content_type: str) -> list[ChunkDraft]:
    """Split Markdown by heading, preserving sentences and complete table rows."""
    if not text.strip():
        return []
    drafts: list[ChunkDraft] = []
    for document in _HEADER_SPLITTER.split_text(text):
        headings = [document.metadata[name] for _, name in _HEADERS if name in document.metadata]
        drafts.extend(_section_drafts(document.page_content.splitlines(), source, headings, content_type))
    return drafts
