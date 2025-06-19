import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import utils.compat_utils


# --- Example Usage ---
if __name__ == "__main__":
    device = utils.compat_utils.get_device()
    print(f"PyTorch will use: {device}")

    # Create a tensor and move it to the determined device
    x = torch.randn(2, 3)
    x = x.to(device)
    print(f"Tensor on device: {x.device}")

    # Example of moving a model to the device
    import torch.nn as nn
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)
        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    model.to(device)
    print(f"Model on device: {next(model.parameters()).device}")

    # You can also create tensors directly on the device
    y = torch.ones(5, 5, device=device)
    print(f"Another tensor directly on device: {y.device}")