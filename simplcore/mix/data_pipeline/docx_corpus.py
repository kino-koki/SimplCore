import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List
from xml.etree import ElementTree as ET

from .clean_zh import clean_record, normalize_text
from .shard_writer import write_jsonl_shard


_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_text(docx_path: Path) -> str:
    docx_path = Path(docx_path)
    with zipfile.ZipFile(docx_path) as zf:
        document_xml = zf.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", _W_NS):
        parts = [
            node.text or ""
            for node in paragraph.findall(".//w:t", _W_NS)
            if node.text
        ]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return normalize_text("\n".join(paragraphs))


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        extra_len = len(paragraph) + (1 if current else 0)
        if current and current_len + extra_len > max_chars:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += extra_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def iter_docx_records(input_dir: Path, max_chars: int = 3000) -> Iterable[Dict[str, str]]:
    input_dir = Path(input_dir)
    for docx_path in sorted(input_dir.glob("*.docx")):
        text = extract_docx_text(docx_path)
        for index, chunk in enumerate(chunk_text(text, max_chars=max_chars)):
            cleaned = clean_record({"text": chunk})
            if not cleaned:
                continue
            yield {
                "text": cleaned["text"],
                "source": docx_path.name,
                "source_type": "user_authorized_docx",
                "chunk_index": str(index),
            }


def prepare_docx_corpus(
    input_dir: Path,
    output_dir: Path,
    shard_name: str = "user_docx_corpus",
    max_chars: int = 3000,
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    records = list(iter_docx_records(input_dir, max_chars=max_chars))
    manifest = write_jsonl_shard(records, output_dir, shard_name=shard_name)
    source_files = sorted(path.name for path in input_dir.glob("*.docx"))
    manifest.update(
        {
            "source_format": "docx",
            "source_type": "user_authorized_docx",
            "source_files": source_files,
            "max_chars": max_chars,
        }
    )
    manifest_path = output_dir / f"{shard_name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
