import torch


def test_train_smoke_runs_backward_optimizer_and_checkpoint(tmp_path):
    from simplcore.mix.training.train_smoke import run_train_smoke

    torch.manual_seed(1234)
    result = run_train_smoke(
        output_dir=tmp_path,
        steps=1,
        batch_size=1,
        seq_len=8,
        seed=1234,
        device="cpu",
    )

    assert result["steps"] == 1
    assert result["batch_size"] == 1
    assert result["seq_len"] == 8
    assert result["loss"] > 0
    assert result["optimizer_steps"] == 1
    assert result["backward_ran"] is True

    checkpoint_path = tmp_path / "train_smoke.pt"
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["step"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["loss"] == result["loss"]
