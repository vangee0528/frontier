"""一键运行完整基准套件：正确性、性能、实用性三重验证。

    python benchmarks/run_all.py            # 完整（数分钟）
    python benchmarks/run_all.py --quick    # 快速（约 1 分钟）

三重验证的分工：
- 正确性：每个基准内置与 SymPy / 手写实现的数值对拍断言，
  任何一处不一致都会让整个套件失败退出（另有 tests/ 的完整
  测试套件与差分模糊测试覆盖更广的表达式空间）；
- 性能：四个基准分别覆盖批量求值（gradient）、逐点求值（ode）、
  端到端拟合（fitting）与规模效应（scaling）；
- 实用性：fitting 与 ode 两个基准即真实 scipy 工作流原样运行，
  证明适配器在实际求解器回调约定下工作。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SUITE = [
    ("gradient  批量梯度求值 vs lambdify/NumPy", "bench_gradient.py",
     {"quick": ["200000"], "full": []}),
    ("ode       刚性系统逐点 RHS/Jacobian",      "bench_ode.py",
     {"quick": ["--quick"], "full": []}),
    ("fitting   非线性拟合四路线对比",            "bench_fitting.py",
     {"quick": ["--quick"], "full": []}),
    ("scaling   加速比随规模的变化",              "bench_scaling.py",
     {"quick": ["--quick"], "full": []}),
]


def main() -> int:
    quick = "--quick" in sys.argv
    mode = "quick" if quick else "full"
    print(f"Frontier 基准套件（{mode}）—— 每项内置正确性对拍断言\n"
          + "=" * 64)

    failures = []
    t_start = time.perf_counter()
    for label, script, extra in SUITE:
        print(f"\n>>> {label}")
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(HERE / script),
                            *extra[mode]], cwd=str(HERE))
        dt = time.perf_counter() - t0
        status = "OK" if r.returncode == 0 else "FAILED"
        print(f"<<< {script}: {status}（{dt:.0f}s）")
        if r.returncode != 0:
            failures.append(script)

    total = time.perf_counter() - t_start
    print("\n" + "=" * 64)
    if failures:
        print(f"套件失败：{failures}（总耗时 {total:.0f}s）")
        return 1
    print(f"全部基准通过，正确性断言零失败（总耗时 {total:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
