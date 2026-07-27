from simplcore.mix.data_pipeline.clean_zh import clean_record, is_simplified_chinese_text


def test_simplified_chinese_text_is_kept():
    text = "人工智能正在改变中文信息处理方式。这个句子是简体中文。"
    assert is_simplified_chinese_text(text)


def test_traditional_text_is_rejected():
    text = "人工智慧正在改變中文資訊處理方式。這個句子是繁體中文。"
    assert not is_simplified_chinese_text(text)


def test_english_heavy_text_is_rejected():
    text = "This is mostly English text with only 少量中文 words."
    assert not is_simplified_chinese_text(text)


def test_clean_record_returns_normalized_json_object():
    record = {"text": "  人工智能   正在改变世界。\n\n这是简体中文。  "}
    cleaned = clean_record(record)
    assert cleaned == {"text": "人工智能 正在改变世界。\n这是简体中文。"}
