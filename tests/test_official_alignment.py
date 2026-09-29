from pathlib import Path

import pytest

from migrationproof.official_alignment import compare_accuracy_logs, parse_training_log


def _log(losses, *, gbs=8, total=4, implementation="ascendc"):
    header = (
        f"model.gdn_implementation: {implementation}\n"
        "model.causal_conv1d_implementation: triton\n"
        "training.micro_batch_size: 4\n"
        "training.gradient_accumulation_steps: 1\n"
        "training.seed: 42\n"
        "training.lr: 1e-05\n"
        "data.dataset_param.basic_parameters.cutoff_len: 1024\n"
    )
    rows = []
    for step, loss in enumerate(losses, 1):
        rows.append(
            f"iteration {step:8d}/{total:8d} | elapsed time per iteration (ms): "
            f"{300 + step:.1f} | global batch size: {gbs:5d} | loss: {loss:.6E} | grad norm: 1.0 |"
        )
    return header + "\n".join(rows)


def test_parse_training_log_rejects_non_finite_loss(tmp_path: Path):
    path = tmp_path / "bad.log"
    path.write_text(_log([1.0, float("nan"), 0.9, 0.8]), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        parse_training_log(path)


def test_compare_accuracy_logs_passes_equivalent_curves(tmp_path: Path):
    ascendc = tmp_path / "ascendc.log"
    triton = tmp_path / "triton.log"
    ascendc.write_text(_log([1.5, 1.4, 1.3, 1.2]), encoding="utf-8")
    triton.write_text(_log([1.501, 1.399, 1.301, 1.201], implementation="triton"), encoding="utf-8")
    result = compare_accuracy_logs(ascendc, triton, expected_iterations=4)
    assert result["gate"]["passed"] is True
    assert result["protocol"]["comparable"] is True


def test_compare_accuracy_logs_rejects_batch_mismatch(tmp_path: Path):
    ascendc = tmp_path / "ascendc.log"
    triton = tmp_path / "triton.log"
    ascendc.write_text(_log([1.5, 1.4, 1.3, 1.2], gbs=8), encoding="utf-8")
    triton.write_text(_log([1.5, 1.4, 1.3, 1.2], gbs=16, implementation="triton"), encoding="utf-8")
    result = compare_accuracy_logs(ascendc, triton, expected_iterations=4)
    assert result["gate"]["passed"] is False
    assert "global_batch_size_mismatch" in result["gate"]["reasons"]

