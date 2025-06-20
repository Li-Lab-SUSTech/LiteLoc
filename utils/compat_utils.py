import torch
import platform


def get_device(dev_print = True):
    """
    Detects the operating system and determines the appropriate PyTorch device
    (GPU or CPU) to use.

    Returns:
        torch.device: The PyTorch device ('cuda', 'mps', or 'cpu').
    """
    # 1. Determine OS for specific GPU backends
    current_os = platform.system()
    if dev_print:
        print(f"Operating System: {current_os}")

    # 2. Check for GPU availability based on OS
    if current_os == "Windows" or current_os == "Linux":
        if torch.cuda.is_available():
            if dev_print:
                print("CUDA GPU is available!")
            # For systems with multiple GPUs, you can specify a particular one
            # using torch.cuda.set_device(device_id) or by appending :device_id
            # to 'cuda' (e.g., 'cuda:0', 'cuda:1').
            # We'll default to the primary GPU (device 0) if CUDA is available.
            return torch.device("cuda")
        else:
            if dev_print:
                print("No CUDA GPU found. Using CPU.")
            return torch.device("cpu")
    elif current_os == "Darwin":  # macOS
        # On macOS, Apple Silicon (M1/M2/etc.) uses MPS (Metal Performance Shaders)
        # for GPU acceleration.
        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                if dev_print:
                    print("MPS (Metal Performance Shaders) GPU is available on macOS!")
                return torch.device("mps")
            else:
                if dev_print:
                    print("MPS is available but not built with this PyTorch install. Using CPU.")
                return torch.device("cpu")
        else:
            if dev_print:
                print("No MPS GPU found on macOS. Using CPU.")
            return torch.device("cpu")
    else:
        # Fallback for unknown operating systems or if no specific GPU backend is detected
        if dev_print:
            print("Unknown operating system or no specific GPU backend detected. Using CPU.")
        return torch.device("cpu")
