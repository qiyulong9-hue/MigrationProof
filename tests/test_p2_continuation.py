import importlib
import importlib.util


def test_builds_isolated_low_lr_continuation_from_step_30():
    assert importlib.util.find_spec("migrationproof.p2_continuation") is not None
    build_continuation_config = importlib.import_module(
        "migrationproof.p2_continuation"
    ).build_continuation_config
    source = {
        "parallel": {"fsdp_plan": {}},
        "features": {"recompute": True},
        "training": {"gradient_accumulation_steps": 1},
    }

    result = build_continuation_config(
        source,
        load="/checkpoints/p2-c03",
        save="/checkpoints/p2-c04r",
        resume_iteration=30,
        additional_iterations=30,
        learning_rate=5e-7,
        effective_batch_size=16,
    )

    assert result["training"]["load"] == "/checkpoints/p2-c03"
    assert result["training"]["save"] == "/checkpoints/p2-c04r"
    assert result["training"]["train_iters"] == 60
    assert result["training"]["save_interval"] == 60
    assert result["training"]["lr"] == 5e-7
    assert result["training"]["gradient_accumulation_steps"] == 8
    assert result["features"]["recompute"] is False
    assert result["parallel"]["fsdp_plan"]["num_to_forward_prefetch"] == 1
    assert result["parallel"]["fsdp_plan"]["num_to_backward_prefetch"] == 1
    assert source["features"]["recompute"] is True
