"""差分模糊冒烟：每次测试运行固定跑一小批随机表达式 vs SymPy。

完整批量用 `python tests/fuzz/fuzz_differential.py --seed S --n N`；
这里只保证 CI 每次都过一遍生成-转换-编译-对拍全链路。
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sympy")

FUZZ = Path(__file__).parent / "fuzz" / "fuzz_differential.py"


def test_fuzz_smoke_batch():
    r = subprocess.run(
        [sys.executable, str(FUZZ), "--seed", "12345", "--n", "40",
         "--no-skip-known"],
        capture_output=True, text=True, timeout=300,
        cwd=str(FUZZ.parent.parent.parent))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    # 允许预期类别（复数中间值 / sympy 自身精度差 / 明确不支持的转换），
    # 但真 bug 类别必须为零
    forbidden = ("CONVERT_BUG", "VALUE_BUG", "DERIV_BUG", "CRASH",
                 "HARNESS_ERROR")
    for line in r.stdout.splitlines():
        for tag in forbidden:
            assert f'"{tag}"' not in line, line
