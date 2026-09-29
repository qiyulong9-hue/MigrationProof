import re
from pathlib import Path


_PLACEHOLDER_RE = re.compile(r"__([A-Z0-9_]+)__")


def render_template(template_path, output_path, values: dict) -> Path:
    template_path = Path(template_path)
    output_path = Path(output_path)
    rendered = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        rendered = rendered.replace(f"__{key}__", str(value))
    unresolved = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if unresolved:
        raise ValueError(f"unresolved placeholders: {', '.join(unresolved)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
    else:
        result[prefix] = value
    return result


def compare_configs(ascendc: dict, triton: dict) -> dict:
    left, right = _flatten(ascendc), _flatten(triton)
    keys = sorted(set(left) | set(right))
    differences = {
        key: {"ascendc": left.get(key), "triton": right.get(key)}
        for key in keys if left.get(key) != right.get(key)
    }
    allowed = {
        "model.gdn_implementation",
        "data.dataset_param.basic_parameters.cache_dir",
        "training.save",
    }
    forbidden = sorted(set(differences) - allowed)
    return {"comparable": not forbidden, "differences": differences,
            "allowed_differences": sorted(set(differences) & allowed),
            "forbidden_differences": forbidden}
