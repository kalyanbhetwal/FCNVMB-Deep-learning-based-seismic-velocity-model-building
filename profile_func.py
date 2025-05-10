import torch
import torch.nn as nn
import torch.profiler

# Define your layers
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
bn = nn.BatchNorm2d(16)

# Dummy input tensor
x = torch.randn(8, 3, 64, 64)  # (batch, channels, height, width)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv.to(device)
bn.to(device)
x = x.to(device)

# Profile only conv and bn
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    with torch.profiler.record_function("conv2d"):
        x = conv(x)
    with torch.profiler.record_function("batchnorm2d"):
        x = bn(x)

# Print results
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=200))