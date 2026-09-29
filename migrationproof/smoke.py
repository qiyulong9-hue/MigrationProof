import io

import torch
import torch_npu


def run_smoke(device: str) -> dict:
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.ones((2, 4), device=device)
    target = torch.zeros((2, 2), device=device)
    output = model(inputs)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    if device.startswith("npu"):
        torch.npu.synchronize()

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.nn.Linear(4, 2).to(device)
    restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
    return {
        "status": "pass",
        "device": device,
        "forward": True,
        "backward": True,
        "optimizer_update": True,
        "checkpoint_roundtrip": True,
        "loss": float(loss.detach().cpu()),
    }
