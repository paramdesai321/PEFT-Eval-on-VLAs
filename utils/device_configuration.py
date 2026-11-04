import torch

def get_device():
  """
    Initialize the device to use for computations
    Returns:
        device: torch.device object representing the device to use
  """
  if torch.cuda.is_available():
    # NVIDIA GPU with CUDA
    device = torch.device("cuda:0")
    print("Backend: CUDA")
    print("GPU:", torch.cuda.get_device_name(device))
    print(
        "GPU memory (GB):",
        round(
            torch.cuda.get_device_properties(device).total_memory / 1e9,
            2,
        ),
    )

  elif torch.backends.mps.is_available():
    # Apple Silicon MPS
    device = torch.device("mps")
    print("Backend: Apple Metal (MPS)")
    print("GPU:", torch.backends.mps.get_name())

  else:
      # Intel Linux, unsupported Mac, or GPU drivers not installed
      device = torch.device("cpu")
      print("Backend: CPU")
  return device