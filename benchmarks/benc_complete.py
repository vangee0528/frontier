"""
Frontier 旗舰基准：符号梯度流水线的百万点批量求值。

场景
----
对带超越函数的三元标量场自动求梯度，然后在 N 个 float64 点上批量求值：

    f(x, y, z)
      = sin(x*y)
      + exp(-z**2) * (x + y)**2
      + 100 * (y - x**2)**2

比较对象
--------
1. frontier
2. frontier(fastmath)
3. sympy.lambdify(modules="numpy")
4. sympy.lambdify(modules="numpy", cse=True)
5. 手写 NumPy 向量化梯度
6. Numba 融合循环（可选，安装 numba 时自动启用）

本脚本补充了：
- Frontier 与 SymPy 的表达式构造、求导、编译/生成函数耗时；
- 首次调用与热执行分开统计；
- best / median / p95，而不是只报 best；
- 每轮随机化实现的测试顺序；
- fastmath 正确性验证；
- 显式 rtol / atol；
- 最大绝对误差与缩放相对误差；
- 可选 JSON 报告；
- 基于 median 的编译成本摊销估算。

注意
----
- 热执行会重复使用同一批输入数组，代表拟合、优化等“同一采样点反复求值”场景。
- 所有热执行计时均包含输出数组分配。
- Frontier 编译耗时是否命中磁盘缓存，取决于 Frontier 当前缓存配置。
- NumPy 是“手写向量化基线”，不是 CPU 性能上限。
- Numba 基线用于提供一个更强的“融合机器码循环”参照。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

# 允许直接从源码仓库运行。
REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.is_dir():
    sys.path.insert(0, str(PYTHON_DIR))

import frontier as fr  # noqa: E402
import sympy as sp  # noqa: E402


ArrayTuple = tuple[np.ndarray, ...]
BenchmarkFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], Any]

_SINK = 0.0


@dataclass(frozen=True)
class TimingStats:
    best_s: float
    median_s: float
    p95_s: float
    mean_s: float
    stdev_s: float
    samples_s: list[float]


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    max_scaled_rel: float


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_s(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) * 1e-9


def normalize_outputs(result: Any, expected_outputs: int = 3) -> ArrayTuple:
    """
    将不同实现的返回值统一成 tuple[np.ndarray, ...]。

    支持：
    - tuple/list of arrays
    - shape=(outputs, N) 的 ndarray
    """
    if isinstance(result, (tuple, list)):
        outputs = tuple(np.asarray(item) for item in result)
    else:
        array = np.asarray(result)
        if array.ndim >= 2 and array.shape[0] == expected_outputs:
            outputs = tuple(array[i] for i in range(expected_outputs))
        elif expected_outputs == 1:
            outputs = (array,)
        else:
            raise TypeError(
                "无法识别函数输出格式：期望 tuple/list，"
                f"或第一维大小为 {expected_outputs} 的 ndarray；"
                f"实际 shape={array.shape!r}"
            )

    if len(outputs) != expected_outputs:
        raise ValueError(
            f"输出数量错误：期望 {expected_outputs}，实际 {len(outputs)}"
        )
    return outputs


def consume_result(result: Any) -> None:
    """
    在计时结束后轻量读取结果，避免未来接入惰性执行后端时只测到提交时间。

    NumPy、Frontier 和 Numba 当前均为同步执行，因此这部分不计入函数耗时。
    """
    global _SINK
    outputs = normalize_outputs(result)
    value = 0.0
    for array in outputs:
        flat = np.ravel(array)
        if flat.size:
            value += float(flat[0]) + float(flat[-1])
    _SINK += value


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(samples_s: Sequence[float]) -> TimingStats:
    values = [float(value) for value in samples_s]
    if not values:
        raise ValueError("计时样本不能为空")

    return TimingStats(
        best_s=min(values),
        median_s=statistics.median(values),
        p95_s=percentile(values, 95.0),
        mean_s=statistics.fmean(values),
        stdev_s=statistics.stdev(values) if len(values) >= 2 else 0.0,
        samples_s=values,
    )


def timed_call(fn: BenchmarkFunction, *args: np.ndarray) -> tuple[Any, float]:
    start = now_ns()
    result = fn(*args)
    seconds = elapsed_s(start)
    consume_result(result)
    return result, seconds


def time_hot_functions(
    functions: Mapping[str, BenchmarkFunction],
    args: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, TimingStats]:
    """
    交错、随机顺序测量所有实现。

    每轮都会随机排列实现顺序，减少 CPU 频率、温度、内存分配器状态和
    固定顺序对单个实现的系统性偏差。
    """
    if repeats < 1:
        raise ValueError("repeats 必须至少为 1")
    if warmups < 0:
        raise ValueError("warmups 不能为负数")

    for name, fn in functions.items():
        for _ in range(warmups):
            result = fn(*args)
            consume_result(result)
            del result

    gc.collect()
    was_gc_enabled = gc.isenabled()
    gc.disable()

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in functions}
    names = list(functions)

    try:
        for _ in range(repeats):
            order = names.copy()
            rng.shuffle(order)

            for name in order:
                fn = functions[name]
                start = now_ns()
                result = fn(*args)
                seconds = elapsed_s(start)

                # 计时结束后消费结果。
                consume_result(result)
                del result

                samples[name].append(seconds)
    finally:
        if was_gc_enabled:
            gc.enable()

    return {name: summarize(values) for name, values in samples.items()}


def compare_outputs(
    name: str,
    actual_result: Any,
    reference_result: Any,
    *,
    rtol: float,
    atol: float,
) -> ErrorStats:
    actual_outputs = normalize_outputs(actual_result)
    reference_outputs = normalize_outputs(reference_result)

    max_abs = 0.0
    max_scaled_rel = 0.0

    for index, (actual, reference) in enumerate(
        zip(actual_outputs, reference_outputs)
    ):
        if actual.shape != reference.shape:
            raise AssertionError(
                f"{name} 第 {index} 个输出 shape 不匹配："
                f"{actual.shape!r} vs {reference.shape!r}"
            )

        np.testing.assert_allclose(
            actual,
            reference,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"{name} 第 {index} 个输出与 NumPy 参考值不一致",
        )

        difference = np.abs(actual - reference)
        if difference.size:
            max_abs = max(max_abs, float(np.nanmax(difference)))

            # 接近零时使用 atol 作为分母下限，避免无意义的巨大相对误差。
            denominator = np.maximum(np.abs(reference), atol)
            scaled_rel = difference / denominator
            max_scaled_rel = max(
                max_scaled_rel,
                float(np.nanmax(scaled_rel)),
            )

    return ErrorStats(max_abs=max_abs, max_scaled_rel=max_scaled_rel)


def make_sympy_lambdify(
    args: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    expressions: Sequence[sp.Expr],
    *,
    cse: bool,
) -> Callable[..., Any]:
    kwargs = {
        "modules": "numpy",
        "cse": cse,
    }

    # 新版 SymPy 支持关闭超长 docstring 生成，避免把文档生成时间混入基准。
    try:
        return sp.lambdify(
            args,
            expressions,
            docstring_limit=0,
            **kwargs,
        )
    except TypeError:
        return sp.lambdify(args, expressions, **kwargs)


def build_numba_baseline() -> tuple[BenchmarkFunction | None, str | None]:
    try:
        import numba
    except ImportError:
        return None, None

    @numba.njit(fastmath=False, cache=False)
    def g_numba(a, b, c):
        n = a.size
        gx = np.empty_like(a)
        gy = np.empty_like(a)
        gz = np.empty_like(a)

        for i in range(n):
            x = a[i]
            y = b[i]
            z = c[i]

            cos_xy = math.cos(x * y)
            exp_z = math.exp(-(z * z))
            xy_sum = x + y
            rosen = y - x * x

            gx[i] = (
                y * cos_xy
                + 2.0 * exp_z * xy_sum
                - 400.0 * x * rosen
            )
            gy[i] = (
                x * cos_xy
                + 2.0 * exp_z * xy_sum
                + 200.0 * rosen
            )
            gz[i] = -2.0 * z * exp_z * xy_sum * xy_sum

        return gx, gy, gz

    return g_numba, getattr(numba, "__version__", "unknown")


def build_all(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    include_numba: bool,
) -> tuple[
    dict[str, BenchmarkFunction],
    dict[str, float],
    dict[str, float],
    dict[str, ErrorStats],
    dict[str, str],
]:
    build_times: dict[str, float] = {}
    first_call_times: dict[str, float] = {}
    correctness: dict[str, ErrorStats] = {}
    versions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 手写 NumPy 基线
    # ------------------------------------------------------------------
    def g_numpy(a, b, c):
        cos_ab = np.cos(a * b)
        exp_z = np.exp(-(c**2))
        xy_sum = a + b
        rosen = b - a**2

        gx = (
            b * cos_ab
            + 2.0 * exp_z * xy_sum
            - 400.0 * a * rosen
        )
        gy = (
            a * cos_ab
            + 2.0 * exp_z * xy_sum
            + 200.0 * rosen
        )
        gz = -2.0 * c * exp_z * xy_sum**2
        return gx, gy, gz

    reference, first_call_times["numpy_manual"] = timed_call(
        g_numpy, xs, ys, zs
    )

    # ------------------------------------------------------------------
    # Frontier：表达式构造、符号梯度、普通编译、fastmath 编译
    # ------------------------------------------------------------------
    start = now_ns()
    x, y, z = fr.symbols("x y z")
    f = (
        fr.sin(x * y)
        + fr.exp(-(z**2)) * (x + y) ** 2
        + 100 * (y - x**2) ** 2
    )
    build_times["frontier_expr"] = elapsed_s(start)

    start = now_ns()
    grad = fr.grad(f, [x, y, z])
    build_times["frontier_grad"] = elapsed_s(start)

    start = now_ns()
    g_frontier = fr.compile(grad, args=(x, y, z))
    build_times["frontier_compile"] = elapsed_s(start)

    start = now_ns()
    g_frontier_fastmath = fr.compile(
        grad,
        args=(x, y, z),
        fastmath=True,
    )
    build_times["frontier_fastmath_compile"] = elapsed_s(start)

    result, first_call_times["frontier"] = timed_call(
        g_frontier, xs, ys, zs
    )
    correctness["frontier"] = compare_outputs(
        "frontier",
        result,
        reference,
        rtol=1e-10,
        atol=1e-12,
    )
    del result

    result, first_call_times["frontier_fastmath"] = timed_call(
        g_frontier_fastmath, xs, ys, zs
    )
    correctness["frontier_fastmath"] = compare_outputs(
        "frontier(fastmath)",
        result,
        reference,
        rtol=1e-9,
        atol=1e-11,
    )
    del result

    # ------------------------------------------------------------------
    # SymPy：表达式构造、符号梯度、默认 lambdify、CSE lambdify
    # ------------------------------------------------------------------
    start = now_ns()
    sx, sy, sz = sp.symbols("x y z")
    sf = (
        sp.sin(sx * sy)
        + sp.exp(-(sz**2)) * (sx + sy) ** 2
        + 100 * (sy - sx**2) ** 2
    )
    build_times["sympy_expr"] = elapsed_s(start)

    start = now_ns()
    sympy_grad = [sp.diff(sf, variable) for variable in (sx, sy, sz)]
    build_times["sympy_grad"] = elapsed_s(start)

    start = now_ns()
    g_sympy = make_sympy_lambdify(
        (sx, sy, sz),
        sympy_grad,
        cse=False,
    )
    build_times["sympy_lambdify"] = elapsed_s(start)

    start = now_ns()
    g_sympy_cse = make_sympy_lambdify(
        (sx, sy, sz),
        sympy_grad,
        cse=True,
    )
    build_times["sympy_lambdify_cse"] = elapsed_s(start)

    result, first_call_times["sympy_lambdify"] = timed_call(
        g_sympy, xs, ys, zs
    )
    correctness["sympy_lambdify"] = compare_outputs(
        "sympy.lambdify",
        result,
        reference,
        rtol=1e-10,
        atol=1e-12,
    )
    del result

    result, first_call_times["sympy_lambdify_cse"] = timed_call(
        g_sympy_cse, xs, ys, zs
    )
    correctness["sympy_lambdify_cse"] = compare_outputs(
        "sympy.lambdify(cse=True)",
        result,
        reference,
        rtol=1e-10,
        atol=1e-12,
    )
    del result

    functions: dict[str, BenchmarkFunction] = {
        "frontier": g_frontier,
        "frontier_fastmath": g_frontier_fastmath,
        "sympy_lambdify": g_sympy,
        "sympy_lambdify_cse": g_sympy_cse,
        "numpy_manual": g_numpy,
    }

    # ------------------------------------------------------------------
    # 可选 Numba：更强的融合循环基线
    # ------------------------------------------------------------------
    if include_numba:
        g_numba, numba_version = build_numba_baseline()
        if g_numba is not None:
            versions["numba"] = numba_version or "unknown"

            # 第一次调用包含 Numba JIT 编译。
            result, first_call_times["numba_fused"] = timed_call(
                g_numba, xs, ys, zs
            )
            correctness["numba_fused"] = compare_outputs(
                "numba fused",
                result,
                reference,
                rtol=1e-10,
                atol=1e-12,
            )
            del result

            functions["numba_fused"] = g_numba

    del reference
    gc.collect()

    return (
        functions,
        build_times,
        first_call_times,
        correctness,
        versions,
    )


def ms(seconds: float) -> float:
    return seconds * 1e3


def format_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def environment_info(extra_versions: Mapping[str, str]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": format_version(np),
        "sympy": format_version(sp),
        "frontier": format_version(fr),
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
            if os.environ.get(key) is not None
        },
    }

    try:
        import llvmlite
    except ImportError:
        pass
    else:
        info["llvmlite"] = format_version(llvmlite)

    info.update(extra_versions)
    return info


def print_environment(info: Mapping[str, Any]) -> None:
    print("\n运行环境")
    print("-" * 78)
    for key in (
        "python",
        "platform",
        "machine",
        "processor",
        "numpy",
        "sympy",
        "frontier",
        "llvmlite",
        "numba",
    ):
        if key in info:
            print(f"{key:>12}: {info[key]}")

    thread_env = info.get("thread_env", {})
    if thread_env:
        print(f"{'thread env':>12}: {thread_env}")


def print_build_times(
    build_times: Mapping[str, float],
    first_call_times: Mapping[str, float],
) -> None:
    print("\n构建与首次调用")
    print("-" * 78)

    rows = [
        ("Frontier 表达式构造", build_times["frontier_expr"]),
        ("Frontier 符号梯度", build_times["frontier_grad"]),
        ("Frontier CSE + 代码生成 + JIT", build_times["frontier_compile"]),
        (
            "Frontier fastmath CSE + 代码生成 + JIT",
            build_times["frontier_fastmath_compile"],
        ),
        ("SymPy 表达式构造", build_times["sympy_expr"]),
        ("SymPy 符号梯度", build_times["sympy_grad"]),
        ("SymPy lambdify", build_times["sympy_lambdify"]),
        ("SymPy lambdify(cse=True)", build_times["sympy_lambdify_cse"]),
    ]

    for name, seconds in rows:
        print(f"{name:<43} {ms(seconds):10.3f} ms")

    print()
    for name, seconds in first_call_times.items():
        print(f"{'首次调用 ' + name:<43} {ms(seconds):10.3f} ms")


def print_correctness(correctness: Mapping[str, ErrorStats]) -> None:
    print("\n正确性：相对手写 NumPy 梯度")
    print("-" * 78)
    print(f"{'实现':<30} {'最大绝对误差':>18} {'最大缩放相对误差':>22}")

    for name, stats in correctness.items():
        print(
            f"{name:<30} "
            f"{stats.max_abs:18.6e} "
            f"{stats.max_scaled_rel:22.6e}"
        )


def print_hot_results(
    n: int,
    stats: Mapping[str, TimingStats],
) -> None:
    labels = {
        "frontier": "Frontier",
        "frontier_fastmath": "Frontier fastmath",
        "sympy_lambdify": "SymPy lambdify",
        "sympy_lambdify_cse": "SymPy lambdify CSE",
        "numpy_manual": "手写 NumPy",
        "numba_fused": "Numba 融合循环",
    }

    lambdify_median = stats["sympy_lambdify"].median_s
    numpy_median = stats["numpy_manual"].median_s

    print(f"\n热执行 @ N = {n:,}，3 输入 -> 3 输出")
    print("-" * 118)
    print(
        f"{'实现':<24}"
        f"{'best(ms)':>12}"
        f"{'median(ms)':>14}"
        f"{'p95(ms)':>12}"
        f"{'M点/秒':>12}"
        f"{'vs lambdify':>16}"
        f"{'vs NumPy':>13}"
    )

    preferred_order = [
        "frontier",
        "frontier_fastmath",
        "sympy_lambdify",
        "sympy_lambdify_cse",
        "numpy_manual",
        "numba_fused",
    ]

    for key in preferred_order:
        if key not in stats:
            continue

        item = stats[key]
        throughput = n / item.median_s / 1e6
        versus_lambdify = lambdify_median / item.median_s
        versus_numpy = numpy_median / item.median_s

        print(
            f"{labels.get(key, key):<24}"
            f"{ms(item.best_s):12.3f}"
            f"{ms(item.median_s):14.3f}"
            f"{ms(item.p95_s):12.3f}"
            f"{throughput:12.2f}"
            f"{versus_lambdify:16.2f}x"
            f"{versus_numpy:13.2f}x"
        )


def print_break_even(
    build_times: Mapping[str, float],
    stats: Mapping[str, TimingStats],
) -> None:
    frontier_build = (
        build_times["frontier_expr"]
        + build_times["frontier_grad"]
        + build_times["frontier_compile"]
    )
    frontier_time = stats["frontier"].median_s

    print("\nFrontier 构建成本摊销估算")
    print("-" * 78)
    print(
        "按 Frontier 的“表达式构造 + 符号求导 + 普通编译”总耗时，"
        "并使用热执行 median 估算。"
    )
    print(f"Frontier 总构建耗时: {ms(frontier_build):.3f} ms")

    for key, label in (
        ("sympy_lambdify", "SymPy lambdify"),
        ("sympy_lambdify_cse", "SymPy lambdify CSE"),
        ("numpy_manual", "手写 NumPy"),
        ("numba_fused", "Numba 融合循环"),
    ):
        if key not in stats:
            continue

        saving = stats[key].median_s - frontier_time
        if saving <= 0:
            print(f"相对 {label:<20}: 热执行没有正向节省")
            continue

        calls = frontier_build / saving
        points = calls
        print(
            f"相对 {label:<20}: 约 {calls:.2f} 次当前 N 点调用可摊销"
        )


def report_to_json(
    path: Path,
    *,
    n: int,
    warmups: int,
    repeats: int,
    environment: Mapping[str, Any],
    build_times: Mapping[str, float],
    first_call_times: Mapping[str, float],
    correctness: Mapping[str, ErrorStats],
    hot_stats: Mapping[str, TimingStats],
) -> None:
    payload = {
        "n": n,
        "warmups": warmups,
        "repeats": repeats,
        "environment": dict(environment),
        "build_times_s": dict(build_times),
        "first_call_times_s": dict(first_call_times),
        "correctness": {
            name: asdict(stats)
            for name, stats in correctness.items()
        },
        "hot_stats": {
            name: asdict(stats)
            for name, stats in hot_stats.items()
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frontier 梯度流水线旗舰基准"
    )
    parser.add_argument(
        "N",
        nargs="?",
        type=int,
        default=1_000_000,
        help="批量点数，默认 1,000,000",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=15,
        help="热执行重复次数，默认 15",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="每个实现的预热次数，默认 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="输入数据和计时顺序随机种子，默认 0",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="即使安装了 numba，也不运行 Numba 基线",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="可选：将完整结果写入 JSON 文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.N <= 0:
        raise SystemExit("N 必须大于 0")
    if args.repeats <= 0:
        raise SystemExit("--repeats 必须大于 0")
    if args.warmups < 0:
        raise SystemExit("--warmups 不能为负数")

    rng = np.random.default_rng(args.seed)
    xs = np.ascontiguousarray(
        rng.uniform(0.1, 1.5, size=args.N),
        dtype=np.float64,
    )
    ys = np.ascontiguousarray(
        rng.uniform(0.1, 1.5, size=args.N),
        dtype=np.float64,
    )
    zs = np.ascontiguousarray(
        rng.uniform(0.1, 1.5, size=args.N),
        dtype=np.float64,
    )

    (
        functions,
        build_times,
        first_call_times,
        correctness,
        extra_versions,
    ) = build_all(
        xs,
        ys,
        zs,
        include_numba=not args.no_numba,
    )

    environment = environment_info(extra_versions)
    print_environment(environment)
    print_build_times(build_times, first_call_times)
    print_correctness(correctness)

    hot_stats = time_hot_functions(
        functions,
        (xs, ys, zs),
        warmups=args.warmups,
        repeats=args.repeats,
        seed=args.seed + 1,
    )

    print_hot_results(args.N, hot_stats)
    print_break_even(build_times, hot_stats)

    print("\n说明")
    print("-" * 78)
    print("1. 热执行重复使用同一组输入数组，属于热数据场景。")
    print("2. 热执行时间包含每次调用创建三个输出数组的成本。")
    print("3. fastmath 不保证更快，具体取决于 LLVM IR、向量数学库和 CPU。")
    print("4. Frontier 编译时间可能受磁盘缓存命中影响。")
    print("5. Numba 首次调用包含其 JIT 编译；热执行表不包含首次编译。")
    print(f"6. 防止惰性执行的结果消费校验值: {_SINK:.6e}")

    if args.json is not None:
        report_to_json(
            args.json,
            n=args.N,
            warmups=args.warmups,
            repeats=args.repeats,
            environment=environment,
            build_times=build_times,
            first_call_times=first_call_times,
            correctness=correctness,
            hot_stats=hot_stats,
        )
        print(f"\nJSON 报告已写入: {args.json}")


if __name__ == "__main__":
    main()