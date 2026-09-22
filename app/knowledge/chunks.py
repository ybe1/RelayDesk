from __future__ import annotations

import re
from dataclasses import dataclass


MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCES = re.compile(r".+?(?:[。！？!?]|(?<=\.)|$)", re.DOTALL)


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
        if selected and length + len(sentence) > OVERLAP_CHARS:
            break
        selected.insert(0, sentence)
        length += len(sentence)
    return selected


def _text_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for sentence in _sentences(text):
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
    category = " / ".join(headings[:-1]) or headings[-1]
    key_clause = any("[!IMPORTANT]" in line for line in text_lines)
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
                drafts.append(ChunkDraft(category, headings[-1], answer, section_path, content_type, key_clause, True))
            continue
        prose: list[str] = []
        while index < len(text_lines) and text_lines[index].strip() and not text_lines[index].lstrip().startswith("|"):
            prose.append(text_lines[index].strip())
            index += 1
        for answer in _text_chunks("\n".join(prose)):
            drafts.append(ChunkDraft(category, headings[-1], answer, section_path, content_type, key_clause, False))
    return drafts


def parse_markdown(text: str, source: str, content_type: str) -> list[ChunkDraft]:
    """Split Markdown by heading, preserving sentences and complete table rows."""
    if not text.strip():
        return []
    drafts: list[ChunkDraft] = []
    headings: list[str] = []
    content: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            content.append(line)
            continue
        drafts.extend(_section_drafts(content, source, headings, content_type))
        content = []
        level = len(match.group(1))
        headings = headings[: level - 1]
        headings.append(match.group(2))
    drafts.extend(_section_drafts(content, source, headings, content_type))
    return drafts
