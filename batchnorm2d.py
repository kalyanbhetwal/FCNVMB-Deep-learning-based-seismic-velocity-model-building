import torch
import triton
import triton.language as tl
import time

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=32),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=32),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=32),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=32)
    ],
    key=[],
)

@triton.jit
def batchnorm2d_mean_pass(x_ptr, mean_ptr, var_ptr, N, C, H, W, stride_n, stride_c, stride_h, stride_w, BLOCK_SIZE: tl.constexpr):
    c = tl.program_id(0)
    layer = N * H * W
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < layer

    mean = 0.0
    mean_for_var = 0.0
    count = 0.0

    for i in range(0, layer, BLOCK_SIZE):
        index = i + offset
        masking_index = index < layer

        n = index // (H * W)
        hw = index % (H * W)
        h = hw // W
        w = hw % W

        x_offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w
        x = tl.load(x_ptr + x_offset, mask=masking_index, other=0.0)

        count_i = tl.sum(masking_index)
        mean += tl.sum(x, axis=0)
        mean_for_var += tl.sum(x * x, axis=0)
        count += count_i

    mean_final = mean / count
    var_final = (mean_for_var / count) - (mean_final * mean_final)

    tl.store(mean_ptr + c, mean_final)
    tl.store(var_ptr + c, var_final)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=32),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=32),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=32),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=32)
    ],
    key=[],
)

@triton.jit
def batchnorm2d_norm_pass(x_ptr, y_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, N, C, H, W, stride_n, stride_c, stride_h, stride_w, eps, BLOCK_SIZE: tl.constexpr):

    c = tl.program_id(0)
    layer = N * H * W
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < layer

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    inverse = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)


    for i in range(0, layer, BLOCK_SIZE):
        index = i + offset
        masking_index = index < layer

        n = index // (H * W)
        hw = index % (H * W)
        h = hw // W
        w = hw % W

        offset_ptr = n * stride_n + c * stride_c + h * stride_h + w * stride_w
        x = tl.load(x_ptr + offset_ptr, mask=masking_index, other=0.0)
        y = weight * (x - mean) * inverse + bias
        tl.store(y_ptr + offset_ptr, y, mask=masking_index)

#@torch.compile(fullgraph=True)
def batchnorm2d_triton(x, weight, bias, eps=1e-5):

    N, C, H, W = x.shape
    y = torch.empty_like(x)

    mean = torch.empty(C, device=x.device, dtype=torch.float32)
    var = torch.empty(C, device=x.device, dtype=torch.float32)

    grid = (C,)
    batchnorm2d_mean_pass[grid](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        N=N, C=C, H=H, W=W,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3)
    )

    batchnorm2d_norm_pass[grid](
        x_ptr=x,
        y_ptr=y,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        N=N, C=C, H=H, W=W,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3),
        eps=eps
    )

    return y, mean, var


def test_and_benchmark():

    device = 'cuda'

    N, C, H, W = 32, 3, 224, 224
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    weight = torch.randn(C, device=device)
    bias = torch.randn(C, device=device)

    bn = torch.nn.BatchNorm2d(C, affine=True, track_running_stats=False).to(device)
    bn.weight.data = weight.clone()
    bn.bias.data = bias.clone()
    with torch.no_grad():
        y_ref = bn(x)

    y_triton, mean, var = batchnorm2d_triton(x, weight, bias)

    torch.testing.assert_close(y_triton, y_ref, atol=1e-5, rtol=1e-5)

    def bench_torch():
        for _ in range(100):
            bn(x)

    def bench_triton():
        for _ in range(100):
            batchnorm2d_triton(x, weight, bias)

    t0 = time.time(); bench_torch(); torch.cuda.synchronize(); t1 = time.time()
    t2 = time.time(); bench_triton(); torch.cuda.synchronize(); t3 = time.time()

    torch_time = (t1 - t0)/100
    triton_time = (t3 - t2)/100

    speedup = ((triton_time - torch_time)/(torch_time))*100
    t_speed = torch_time / triton_time
    print(f"PyTorch time: {(torch_time) * 1000:.4f} ms")
    print(f"Triton time:  {(triton_time) * 1000:.4f} ms")
    print(f"Percentage Speedup:  {speedup:.4f} .")
    print(f"Times Speedup:  {t_speed:.4f} .")

test_and_benchmark()


