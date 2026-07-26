"""pickle / deepcopy：Expr 与 CompiledFunction 的序列化契约。"""
import copy
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import frontier as fr


def test_expr_pickle_roundtrip():
    x, y = fr.symbols("x y")
    cases = [
        x,
        fr.as_expr(7),
        fr.rational(22, 7),
        fr.as_expr(0.5),
        fr.sin(x * y) + fr.rational(1, 3) * x**2,
        fr.where(x > 0, x, -x) + fr.atan2(y, x),
        fr.diff(fr.exp(-(x**2)) * fr.cos(y), x),
    ]
    for e in cases:
        r = pickle.loads(pickle.dumps(e))
        assert r == e, e          # 结构相等（interning：即指针相等）
        assert repr(r) == repr(e)


def test_expr_pickle_preserves_dag_sharing():
    x = fr.symbols("x")
    u = fr.sin(x)
    e = u + u**2 + u**3
    blob = pickle.dumps(e)
    # 共享子树只编码一次：序列化大小应远小于线性展开
    e_dup = fr.sin(x) + fr.sin(x + 1) ** 2 + fr.sin(x + 2) ** 3
    assert len(blob) < len(pickle.dumps(e_dup))
    assert pickle.loads(blob) == e


def test_expr_deepcopy_and_copy():
    x, y = fr.symbols("x y")
    e = x * y + 1
    assert copy.deepcopy(e) == e
    assert copy.copy(e) == e
    # 容器深拷贝
    d = {"expr": [e, e]}
    d2 = copy.deepcopy(d)
    assert d2["expr"][0] == e


def test_expr_pickle_bad_state_raises():
    """损坏的 pickle 状态应报 FrontierError，而非崩溃或静默错果。"""
    x = fr.symbols("x")
    blob = pickle.dumps(x + 1)
    # 把节点标签字节破坏成未知标签（'s'/'c'/'+' → '?'）
    for tag in (b"s", b"+"):
        bad = blob.replace(tag, b"?", 1)
        if bad == blob:
            continue
        with pytest.raises(Exception) as exc_info:
            pickle.loads(bad)
        assert not isinstance(exc_info.value, SystemError)


def test_compiled_function_pickle_roundtrip():
    x, y = fr.symbols("x y")
    f = fr.compile([fr.sin(x * y), x + y], args=(x, y),
                   uniform=(y,), workers=2)
    g = pickle.loads(pickle.dumps(f))
    xs = np.linspace(0.1, 2.0, 501)
    for a, b in zip(f(xs, 0.7), g(xs, 0.7)):
        np.testing.assert_allclose(a, b)
    # 选项随 pickle 保留
    assert g.workers == f.workers
    assert g._uniform == f._uniform


def test_compiled_function_pickle_subprocess():
    """跨进程：pickle 的 CompiledFunction 在子进程反序列化并求值。"""
    x = fr.symbols("x")
    f = fr.compile(x**2 + fr.exp(x), args=(x,))
    blob = pickle.dumps(f)
    # 让子进程从与当前进程相同的位置导入 frontier（源码树或 site-packages）
    py_dir = Path(fr.__file__).resolve().parent.parent
    code = (
        "import pickle, sys, numpy as np\n"
        "f = pickle.loads(sys.stdin.buffer.read())\n"
        "out = f(np.array([0.5, 1.0]))\n"
        "print(round(float(out[0]), 10), round(float(out[1]), 10))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], input=blob,
                       capture_output=True, timeout=120,
                       env={**__import__('os').environ,
                            "PYTHONPATH": str(py_dir)})
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-800:]
    v1, v2 = map(float, r.stdout.split())
    assert abs(v1 - (0.25 + np.exp(0.5))) < 1e-9
    assert abs(v2 - (1.0 + np.e)) < 1e-9
