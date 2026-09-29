from migrationproof.official_config import compare_configs, render_template


def _config(gdn):
    return {
        "parallel": {"fsdp_plan": {"param_dtype": "bf16", "reduce_dtype": "fp32"}},
        "model": {"model_id": "qwen3_5", "gdn_implementation": gdn,
                  "causal_conv1d_implementation": "triton"},
        "data": {"dataset_param": {"basic_parameters": {"cutoff_len": 1024,
                 "dataset": "/data/train.json"}}},
        "training": {"micro_batch_size": 4, "gradient_accumulation_steps": 1,
                     "seed": 42, "lr": 1e-5, "train_iters": 100,
                     "load": "/model/dcp"},
    }


def test_compare_configs_allows_only_gdn_implementation_difference():
    result = compare_configs(_config("ascendc"), _config("triton"))
    assert result["comparable"] is True
    assert result["differences"] == {
        "model.gdn_implementation": {"ascendc": "ascendc", "triton": "triton"}
    }


def test_compare_configs_rejects_dataset_difference():
    ascendc = _config("ascendc")
    triton = _config("triton")
    triton["data"]["dataset_param"]["basic_parameters"]["dataset"] = "/data/other.json"
    result = compare_configs(ascendc, triton)
    assert result["comparable"] is False
    assert "data.dataset_param.basic_parameters.dataset" in result["forbidden_differences"]


def test_compare_configs_allows_run_local_output_paths():
    ascendc = _config("ascendc")
    triton = _config("triton")
    ascendc["data"]["dataset_param"]["basic_parameters"]["cache_dir"] = "/runs/a/cache"
    triton["data"]["dataset_param"]["basic_parameters"]["cache_dir"] = "/runs/t/cache"
    ascendc["training"]["save"] = "/runs/a/checkpoints"
    triton["training"]["save"] = "/runs/t/checkpoints"

    result = compare_configs(ascendc, triton)

    assert result["comparable"] is True
    assert result["forbidden_differences"] == []


def test_render_template_replaces_all_declared_placeholders(tmp_path):
    template = tmp_path / "template.yaml"
    output = tmp_path / "rendered.yaml"
    template.write_text("model: __MODEL__\ndata: __DATA__\n", encoding="utf-8")

    render_template(template, output, {"MODEL": "/models/qwen", "DATA": "/data/train.json"})

    assert output.read_text(encoding="utf-8") == (
        "model: /models/qwen\ndata: /data/train.json\n"
    )


def test_render_template_rejects_unresolved_placeholders(tmp_path):
    template = tmp_path / "template.yaml"
    output = tmp_path / "rendered.yaml"
    template.write_text("model: __MODEL__\ndata: __DATA__\n", encoding="utf-8")

    try:
        render_template(template, output, {"MODEL": "/models/qwen"})
    except ValueError as exc:
        assert "DATA" in str(exc)
    else:
        raise AssertionError("unresolved placeholder was not rejected")
