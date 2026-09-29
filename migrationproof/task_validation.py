REQUIRED = {
    "model": ("name", "source"),
    "ascend": ("device_model", "baseline_config", "optimized_config"),
    "dataset": ("name", "version", "task_type", "test_split"),
    "accuracy_gate": ("metric", "minimum", "max_drop_from_reference"),
    "performance_gate": (
        "primary_metric",
        "minimum_improvement_percent",
        "repetitions",
    ),
}


def validate_task(data: dict) -> list[str]:
    errors = []
    if not data.get("task_id"):
        errors.append("task_id is required")
    for section, fields in REQUIRED.items():
        value = data.get(section)
        if not isinstance(value, dict):
            errors.append(f"{section} is required")
            continue
        for field in fields:
            if field not in value:
                errors.append(f"{section}.{field} is required")
    if data.get("dataset", {}).get("task_type") != "classification":
        errors.append("dataset.task_type must be classification")
    if data.get("performance_gate", {}).get("primary_metric") not in {
        "throughput",
        "latency",
    }:
        errors.append("performance_gate.primary_metric must be throughput or latency")
    repetitions = data.get("performance_gate", {}).get("repetitions")
    if isinstance(repetitions, int) and repetitions < 5:
        errors.append("performance_gate.repetitions must be at least 5")
    return errors
