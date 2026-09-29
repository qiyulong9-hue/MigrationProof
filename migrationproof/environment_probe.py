import platform

import torch
import torch_npu


def normalize_fingerprint(raw: dict) -> dict:
    names = raw.get("device_names", [])
    unique = sorted(set(names))
    return {
        **raw,
        "device_model": unique[0] if len(unique) == 1 else "mixed",
        "device_count": raw.get("device_count", len(names)),
    }


def collect_fingerprint() -> dict:
    count = torch.npu.device_count()
    raw = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "npu_available": torch.npu.is_available(),
        "device_count": count,
        "device_names": [torch.npu.get_device_name(i) for i in range(count)],
    }
    return normalize_fingerprint(raw)
