# -*- coding: utf-8 -*-
"""多模态路由与 ViT 编码测试。"""
import torch

from simplcore_pg.multimodal import ModalityRouter, SimplCoreViTEncoder
from simplcore_pg.exceptions import ModalityError


def test_router_text():
    r = ModalityRouter(hidden_size=32, use_vision=False)
    out = r(text_embeds=torch.randn(2, 8, 32))
    assert out["latent"].shape == (2, 8, 32)
    assert out["modality_mask"].shape == (2, 8)
    assert (out["modality_mask"] == 0).all()


def test_router_num_modalities_aligns_with_config():
    # 与 SimplCoreConfig.num_modalities 对齐，且默认沿用内置词表长度
    r_default = ModalityRouter(hidden_size=32)
    assert r_default.num_modalities == 4
    r_cfg = ModalityRouter(hidden_size=32, num_modalities=3)
    assert r_cfg.num_modalities == 3
    assert r_cfg.modality_embeddings.shape == (3, 32)
    # 索引 0/1/2 在 num_modalities=3 时仍可用
    out = r_cfg(text_embeds=torch.randn(1, 5, 32))
    assert out["latent"].shape == (1, 5, 32)


def test_router_requires_modality():
    r = ModalityRouter(hidden_size=32, use_vision=False)
    try:
        r()
        assert False, "应抛出 ModalityError"
    except ModalityError:
        pass


def test_router_image_disabled_raises():
    r = ModalityRouter(hidden_size=32, use_vision=False)
    try:
        r(image=torch.randn(1, 3, 32, 32))
        assert False, "应抛出 NotImplementedError"
    except NotImplementedError:
        pass


def test_router_multimodal_text_image():
    r = ModalityRouter(hidden_size=32, use_vision=True, image_size=32, patch_size=8)
    out = r(text_embeds=torch.randn(1, 4, 32), image=torch.randn(1, 3, 32, 32))
    # patch 数 = (32/8)^2 = 16，加 cls 后 encode_image 去 cls -> 16
    assert out["latent"].shape == (1, 4 + 16, 32)


def test_vit_encoder_forward():
    enc = SimplCoreViTEncoder(image_size=32, patch_size=8, hidden_size=32,
                              num_layers=2, num_heads=4, output_dim=32)
    out = enc(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 17, 32)  # cls + 16 patches


def test_vit_encode_image_video():
    enc = SimplCoreViTEncoder(image_size=32, patch_size=8, hidden_size=32,
                              num_layers=2, num_heads=4, output_dim=32)
    img = enc.encode_image(torch.randn(2, 3, 32, 32))
    assert img.shape == (2, 16, 32)
    vid = enc.encode_video(torch.randn(1, 3, 3, 32, 32))
    assert vid.shape == (1, 3 * 16, 32)


def test_vit_bad_input():
    try:
        SimplCoreViTEncoder(image_size=31, patch_size=8, hidden_size=32, output_dim=32)
        assert False, "应抛 ValueError (image_size 不能被 patch_size 整除)"
    except ValueError:
        pass
