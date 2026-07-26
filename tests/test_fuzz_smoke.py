"""差分模糊冒烟：CI 每次运行一小批固定种子的随机表达式对拍。

与 fuzzer 共享同一套类别定义与退出判定（直接 import 其 run()），
不在此处复制任何类别字符串。完整批量：
    python tests/fuzz/fuzz_differential.py --seed S --n N
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("sympy")

_FUZZ_PATH = Path(__file__).parent / "fuzz" / "fuzz_differential.py"


def _load_fuzzer():
    spec = importlib.util.spec_from_file_location("fuzz_differential", _FUZZ_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fuzz_smoke_batch(tmp_path):
    fz = _load_fuzzer()

    # 类别三分法必须互斥且非空（结构自检，防止将来定义漂移）
    assert fz.BUG_CATEGORIES and fz.EXPECTED_CATEGORIES
    assert not (fz.BUG_CATEGORIES & fz.REVIEW_CATEGORIES)
    assert not (fz.BUG_CATEGORIES & fz.EXPECTED_CATEGORIES)
    assert not (fz.REVIEW_CATEGORIES & fz.EXPECTED_CATEGORIES)

    out = io.StringIO()
    with redirect_stdout(out):
        summary = fz.run(seed=12345, n=40,
                         json_out=str(tmp_path / "records.jsonl"))

    # 退出判定与 fuzzer 完全一致：BUG 类记录即失败
    assert summary["exit_code"] == 0, (summary, out.getvalue()[-2000:])
    assert summary["bug_records"] == 0, summary
    assert not summary["unknown_categories"], summary
    # 该固定种子批当前也不应有待裁定记录；若将来出现，先人工裁定
    # 归类（EXPECTED 或修复），不要直接放宽这个断言
    assert summary["review_records"] == 0, summary
