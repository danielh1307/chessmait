import torch


def check_cuda():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # device = "cpu" # uncomment if you want to use "cpu"
    print(f"Using {device} device")

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")

    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")


if __name__ == "__main__":
    print("Starting training process ...")

    check_cuda()
