from simplcore.mix.config import SimplCoreMixConfig, mix_7b_text, mix_14b_text


def build_target_config(size: str) -> SimplCoreMixConfig:
    normalized = size.lower().strip()
    if normalized == "7b":
        return mix_7b_text()
    if normalized == "14b":
        return mix_14b_text()
    raise ValueError(f"unsupported target size: {size}")
