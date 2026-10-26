import json
import zipfile

from simplcore.mix.data_pipeline.docx_corpus import (
    chunk_text,
    extract_docx_text,
    prepare_docx_corpus,
)


def _write_minimal_docx(path, paragraphs):
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(
            f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
            for paragraph in paragraphs
        )
        + "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", document_xml)


def test_extract_docx_text_reads_word_paragraphs(tmp_path):
    docx_path = tmp_path / "安魂曲.docx"
    _write_minimal_docx(
        docx_path,
        ["第一章 风雪之后。", "青小团站在新元纪年的门口。"],
    )

    text = extract_docx_text(docx_path)

    assert text == "第一章 风雪之后。\n青小团站在新元纪年的门口。"


def test_chunk_text_preserves_order_and_splits_on_paragraphs():
    text = "\n".join(
        [
            "第一段是很长的中文内容，用于确认切分不会打乱顺序。",
            "第二段继续描述新元世界中的人物关系。",
            "第三段保留在后续训练样本里。",
        ]
    )

    chunks = chunk_text(text, max_chars=30)

    assert len(chunks) == 3
    assert chunks[0].startswith("第一段")
    assert chunks[1].startswith("第二段")
    assert chunks[2].startswith("第三段")


def test_prepare_docx_corpus_writes_jsonl_and_manifest(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_minimal_docx(
        input_dir / "安魂曲  完整世界观编年史.docx",
        [
            "安魂曲新元世界观设定，人物与城市都使用简体中文记录。",
            "这些文字属于作者授权语料，可用于青小团文本模型继续训练。",
        ],
    )

    manifest = prepare_docx_corpus(input_dir, output_dir, max_chars=120)

    shard_path = output_dir / "user_docx_corpus.jsonl"
    manifest_path = output_dir / "user_docx_corpus.manifest.json"
    assert shard_path.exists()
    assert manifest_path.exists()
    assert manifest["num_records"] == 1
    assert manifest["source_format"] == "docx"
    assert manifest["source_files"] == ["安魂曲  完整世界观编年史.docx"]

    record = json.loads(shard_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["source"] == "安魂曲  完整世界观编年史.docx"
    assert record["source_type"] == "user_authorized_docx"
    assert "作者授权语料" in record["text"]
