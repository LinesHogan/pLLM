"""玩具实验：GPU 经 UVA 直接读 pinned 主机内存里的行，能否与主 stream 上的模型 forward 重叠？

场景（pllm 里的真实问题缩小版）：
  * 一张大表放在 pinned host 内存里，GPU 通过 UVA 视图直接经 PCIe 读取选中的行；
  * 行号是 GPU 上算出来的，CPU 不知道（不允许 device->host 同步）；
  * 读行跑在副 stream，希望和主 stream 的 forward 同时进行，把传输时间"藏"起来。

对比两种 Triton gather kernel：
  naive   : grid = 行数 * 每行 tile 数（几万个 CTA），每个 CTA 大部分时间在等 PCIe，
            却占满所有 SM，forward 的 matmul 抢不到 SM，传输藏不住；
  bounded : 持久化 kernel，grid = SM 数 // 4，每个 program 用 tl.range 循环处理多个 tile。
            少量 CTA 就能打满 PCIe，其余 SM 留给 forward。

为什么不用 DMA / copy engine（cudaMemcpyBatchAsync、cudaMemcpy2DAsync、vLLM swap_blocks）？
  它们都要求源地址 / 指针数组在 host 端已知，而我们的行号在 GPU 上，
  用它们就得先把行号读回 CPU（一次同步，会打断异步流水线），所以只能用 GPU kernel 自己去读。

运行（需要一块 CUDA GPU）： python tools/toy_host_gather.py
"""

import statistics
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

ROWS_TABLE, WIDTH, N_PICK = 2048, 151936, 128  # 宽行：每行 ~297 KiB，共 ~594 MiB
BLOCK = 2048  # 每个 tile 读 2048 个 bf16 = 4 KiB 连续字节


@triton.jit
def naive_rows(
    source: torch.Tensor, rows: torch.Tensor, output: torch.Tensor, width: tl.constexpr, block: tl.constexpr
) -> None:
    # 每个 program 只负责一个 (行, 列 tile)；grid 跟数据量成正比。
    tiles = tl.cdiv(width, block)
    row = tl.program_id(0) // tiles
    column = tl.program_id(0) % tiles * block + tl.arange(0, block)
    index = tl.load(rows + row).to(tl.int64)  # 行号在 GPU 上，kernel 内部自己取
    value = tl.load(source + index * width + column, column < width, 0)
    tl.store(output + row * width + column, value, column < width)


@triton.jit
def bounded_rows(
    source: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    count: int,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    # 同样的循环体，但 grid 固定；program 以 num_programs 为步长跨步领取 tile。
    tiles = tl.cdiv(width, block)
    for tile in tl.range(tl.program_id(0), count * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = tile % tiles * block + tl.arange(0, block)
        index = tl.load(rows + row).to(tl.int64)
        value = tl.load(source + index * width + column, column < width, 0)
        tl.store(output + row * width + column, value, column < width)


def sm_count() -> int:
    return int(torch.cuda.get_device_properties(0).multi_processor_count)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


def run_naive(view: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    tiles = triton.cdiv(WIDTH, BLOCK)
    naive_rows[(N_PICK * tiles,)](view, idx, out, WIDTH, BLOCK)


def run_bounded(view: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    bounded_rows[(max(1, sm_count() // 4),)](view, idx, out, N_PICK, WIDTH, BLOCK)


def make_forward(target_ms: float = 3.0) -> tuple[Callable[[], torch.Tensor], int]:
    """假 forward：一串 bf16 matmul；先标定循环次数，让它在本机约 3 ms。"""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

    def forward(n: int) -> torch.Tensor:
        x = a
        for _ in range(n):
            x = x @ b  # 4096^3 的大 matmul，会铺满几乎所有 SM（计算受限）
            x = x * 0.01  # 防止数值溢出成 inf
        return x

    forward(3)
    torch.cuda.synchronize()
    n = 1
    while timed(lambda: forward(n)) < target_ms:
        n += 1
    return lambda: forward(n), n


def timed(fn: Callable[[], object]) -> float:
    """用 CUDA event 计时（比 CPU 计时准，因为 kernel 是异步的）。"""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def median_ms(fn: Callable[[], object], repeats: int = 15, warmup: int = 3) -> float:
    for _ in range(warmup):  # 预热：Triton 编译、PCIe/GPU 频率爬升
        fn()
    torch.cuda.synchronize()
    return statistics.median(timed(fn) for _ in range(repeats))


def main() -> None:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    # 1) pinned 主机表 + 它的 CUDA 视图（同一块物理内存，GPU 读它就走 PCIe）
    table = torch.randint(-(2**15), 2**15, (ROWS_TABLE, WIDTH), dtype=torch.int16).view(torch.bfloat16)
    table = table.pin_memory()
    view = get_accelerator_view_from_cpu_tensor(table)
    assert view.is_cuda and view.data_ptr() == table.data_ptr()  # 确认是别名而不是拷贝

    # 行号在 GPU 上生成，CPU 从头到尾不知道（只有最后验证时才读回一次）
    idx = torch.randint(0, ROWS_TABLE, (N_PICK,), device="cuda")
    out = torch.empty(N_PICK, WIDTH, device="cuda", dtype=torch.bfloat16)
    nbytes = out.numel() * out.element_size()

    # 2) 正确性：与 CPU 上 torch 的参考 gather 逐比特比较（当 int16 比，避免 NaN 位型干扰）
    reference = table[idx.cpu()].view(torch.int16)
    for name, run in (("naive", run_naive), ("bounded", run_bounded)):
        out.zero_()
        run(view, idx, out)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16).cpu(), reference), name
    sms = sm_count()
    print(f"SM 数={sms}  naive grid={N_PICK * triton.cdiv(WIDTH, BLOCK)}  bounded grid={max(1, sms // 4)}")

    # 3) 假 forward
    forward, n = make_forward()
    print(f"forward = {n} 组 4096^3 matmul；传输量 {nbytes / 2**20:.1f} MiB\n")

    main_stream, side = torch.cuda.current_stream(), torch.cuda.Stream()

    def concurrent(run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]) -> Callable[[], None]:
        def step() -> None:
            side.wait_stream(main_stream)  # 副 stream 要等 idx 等输入就绪
            forward()  # 主 stream：模型 forward
            with torch.cuda.stream(side):  # 副 stream：同时读 host 行
                run(view, idx, out)
            main_stream.wait_stream(side)  # 两边都做完才算一步结束

        return step

    # 4) 测量
    t_fwd = median_ms(forward)
    print(f"{'kernel':<9}{'gather':>9}{'GB/s':>8}{'forward':>9}{'concurrent':>12}{'overlap saved':>15}")
    for name, run in (("naive", run_naive), ("bounded", run_bounded)):
        t_gather = median_ms(lambda: run(view, idx, out))
        t_both = median_ms(concurrent(run))
        saved = t_fwd + t_gather - t_both  # 完全串行为 0；完全藏住则等于 min(forward, gather)
        gbps = nbytes / (t_gather * 1e-3) / 1e9
        print(f"{name:<9}{t_gather:>7.2f}ms{gbps:>8.1f}{t_fwd:>7.2f}ms{t_both:>10.2f}ms{saved:>13.2f}ms")


if __name__ == "__main__":
    main()
