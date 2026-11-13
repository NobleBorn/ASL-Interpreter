
import torch

def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your NVIDIA driver "
            "and PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)

    print(f"GPU memory: {vram_gb:.2f} GB")

    # Verify actual computation on the GPU.
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x

    print("GPU computation successful:", y.shape)


if __name__ == "__main__":
    main()
