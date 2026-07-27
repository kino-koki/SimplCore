import torch


def test_tiny_text_model_forward_backward():
    from simplcore.mix import SimplCoreMix, mix_tiny_text

    torch.manual_seed(7)
    cfg = mix_tiny_text()
    model = SimplCoreMix(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    logits = model(tokens)

    assert logits.shape == (2, 16, cfg.vocab_size)

    loss = logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size).float().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
