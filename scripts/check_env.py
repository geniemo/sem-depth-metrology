"""GPU/PyTorch smoke test: prints env info and times a matmul on CUDA."""
import time

import torch


def main() -> None:
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(dev)}")
    print(f"capability: {torch.cuda.get_device_capability(dev)}")
    x = torch.randn(4096, 4096, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        x = x @ x
        x = x / x.norm()
    torch.cuda.synchronize()
    print(f"10x 4096^2 matmul: {time.perf_counter() - t0:.3f}s")
    print("bf16 autocast:", end=" ")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = (x @ x).float().sum()
    print(f"ok ({y.item():.3e})")
    print("ENV OK")


if __name__ == "__main__":
    main()
