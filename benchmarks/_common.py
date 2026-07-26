"""基准套件公共设施：路径注入、计时、结果表。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PY = Path(__file__).resolve().parent.parent / "python"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))


def timeit_best(fn, *args, repeats: int = 7):
    """多次运行取最佳单次耗时（秒）；返回 (耗时, 最后一次返回值)。"""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


class Table:
    """对齐的结果表：row(名称, {列: 值})，print_table() 输出。"""

    def __init__(self, title: str, columns: list[str]):
        self.title = title
        self.columns = columns
        self.rows: list[tuple[str, dict]] = []

    def row(self, name: str, values: dict) -> None:
        self.rows.append((name, values))

    def print_table(self) -> None:
        w0 = max([len(n) for n, _ in self.rows] + [8]) + 2
        print(f"\n== {self.title} ==")
        header = " " * w0 + "".join(f"{c:>14s}" for c in self.columns)
        print(header)
        for name, vals in self.rows:
            line = f"  {name:<{w0 - 2}s}"
            for c in self.columns:
                v = vals.get(c, "")
                line += f"{v:>14s}" if isinstance(v, str) else f"{v:>14.2f}"
            print(line)


def ms(seconds: float) -> str:
    return f"{seconds * 1e3:.2f} ms"


def us(seconds: float) -> str:
    return f"{seconds * 1e6:.1f} us"
