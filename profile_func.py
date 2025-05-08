import torch
import torch.nn as nn
import torch.profiler

# Define layers
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
bn = nn.BatchNorm2d(16)

# Dummy input
x = torch.randn(8, 3, 64, 64)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv.to(device)
bn.to(device)
x = x.to(device)

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    with torch.profiler.record_function("conv2d"):
        x = conv(x)
    with torch.profiler.record_function("batchnorm2d"):
        x = bn(x)

# Filter results
events = prof.key_averages()
conv_bn_events = [e for e in events if "conv2d" in e.name or "batchnorm2d" in e.name]

# Print only conv2d and batchnorm2d results
for evt in conv_bn_events:
    print(evt)

# Or show as a table
print(prof.key_averages(group_by_input_shape=False).table(
    sort_by="self_cuda_time_total", 
    row_limit=20, 
    filter_fn=lambda e: "conv2d" in e.name or "batchnorm2d" in e.name
))
