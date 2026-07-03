"""GPU 最小验证：随机张量放到 GPU 做矩阵乘法，打印设备与耗时。
用法: python gpu_test.py
"""
import time, sys, torch

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | 编译CUDA {torch.version.cuda}")
print("cuda.is_available():", torch.cuda.is_available())
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}: {torch.cuda.get_device_name(i)}")
    print("  arch_list:", torch.cuda.get_arch_list())

N = 4096
a = torch.randn(N, N, device=dev)
b = torch.randn(N, N, device=dev)
for _ in range(3):                      # 预热
    _ = a @ b
if dev == "cuda":
    torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    c = a @ b
if dev == "cuda":
    torch.cuda.synchronize()
avg = (time.time() - t0) / 20 * 1000
tflops = 2 * N**3 / (avg / 1000) / 1e12
print(f"matmul {N}x{N} x20: 平均 {avg:.2f} ms/次 on {dev} (~{tflops:.1f} TFLOPS)")
print("GPU_TEST_CUDA_OK" if dev == "cuda" else "GPU_TEST_CPU_FALLBACK")
