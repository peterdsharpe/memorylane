from memorylane import profile
import torch


@profile
def my_function():
    x = torch.randn(5120, 5120, device="cuda")
    x = x @ x
    x = x.relu()
    x = x.mean()
    return x

my_function()