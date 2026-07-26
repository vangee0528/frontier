"""并发与缓存的健壮性（外部评审 P2 项）。"""
import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import frontier as fr


def test_concurrent_first_compile():
    """多线程同时首次编译不同表达式：intern 表锁 + JIT 各自独立。"""
    x = fr.symbols("x")
    xs = np.linspace(0.1, 1.0, 101)

    def build_and_eval(i):
        e = fr.sin(x * (i + 1)) + x**i if i else fr.cos(x)
        g = fr.compile(e, args=(x,))
        return g(xs)

    with cf.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(build_and_eval, range(16)))
    for i, r in enumerate(results):
        want = np.sin(xs * (i + 1)) + xs**i if i else np.cos(xs)
        np.testing.assert_allclose(r, want, rtol=1e-12)


def test_concurrent_same_expression_compile():
    """多线程同时编译同一表达式：jit_compile 的 LRU 竞争路径。"""
    x, y = fr.symbols("x y")
    xs = np.linspace(0.1, 1.0, 101)

    def build(_):
        return fr.compile(fr.exp(x) * fr.sin(y), args=(x, y))

    with cf.ThreadPoolExecutor(8) as pool:
        fns = list(pool.map(build, range(16)))
    for g in fns:
        np.testing.assert_allclose(g(xs, xs), np.exp(xs) * np.sin(xs))


def test_cache_corruption_recovery(tmp_path, monkeypatch):
    """截断/损坏的缓存对象码：校验失败 → 忽略并重新编译，结果正确。"""
    monkeypatch.setenv("FRONTIER_CACHE_DIR", str(tmp_path))
    from frontier import _jit
    _jit.jit_compile.cache_clear()

    x = fr.symbols("x")
    e = fr.sin(x) * fr.exp(x) + x**5

    g1 = fr.compile(e, args=(x,), cache=True)
    objs = list(tmp_path.glob("*.o"))
    shas = list(tmp_path.glob("*.sha"))
    assert len(objs) == 1 and len(shas) == 1

    # 场景 1：对象码被截断
    data = objs[0].read_bytes()
    objs[0].write_bytes(data[: len(data) // 2])
    _jit.jit_compile.cache_clear()
    g2 = fr.compile(e, args=(x,), cache=True)
    assert not g2._kernel.cache_hit          # 坏缓存被拒，走正常编译
    xs = np.linspace(0, 1, 101)
    np.testing.assert_allclose(g2(xs), g1(xs))

    # 场景 2：校验和文件缺失
    for p in tmp_path.glob("*.sha"):
        p.unlink()
    _jit.jit_compile.cache_clear()
    g3 = fr.compile(e, args=(x,), cache=True)
    assert not g3._kernel.cache_hit
    np.testing.assert_allclose(g3(xs), g1(xs))

    # 修复后的缓存再次命中
    _jit.jit_compile.cache_clear()
    g4 = fr.compile(e, args=(x,), cache=True)
    assert g4._kernel.cache_hit


def test_multiprocess_cache_write(tmp_path):
    """多进程同时写同一缓存键：原子替换保证无损坏，全部结果正确。"""
    code = f"""
import os, sys
os.environ["FRONTIER_CACHE_DIR"] = {str(tmp_path)!r}
sys.path.insert(0, {str(Path(fr.__file__).resolve().parent.parent)!r})
import numpy as np
import frontier as fr
x = fr.symbols("x")
g = fr.compile(fr.sin(x) + x**3, args=(x,), cache=True)
out = g(np.array([0.5]))
assert abs(out[0] - (np.sin(0.5) + 0.125)) < 1e-12
print("ok")
"""
    procs = [subprocess.Popen([sys.executable, "-c", code],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for _ in range(4)]
    for p in procs:
        out, err = p.communicate(timeout=180)
        assert p.returncode == 0, err.decode(errors="replace")[-500:]
        assert b"ok" in out
    # 缓存文件完好（校验和匹配）
    from frontier._jit import _obj_digest
    objs = list(tmp_path.glob("*.o"))
    assert objs, "no cache written"
    for o in objs:
        sha = o.with_suffix(".sha").read_text().strip()
        assert _obj_digest(o.read_bytes()) == sha


def test_sustained_expression_churn():
    """长时间连续构造与释放表达式：intern 表自清理、无累积错误。"""
    x, y = fr.symbols("x y")
    for round_ in range(30):
        exprs = [fr.sin(x * i) + y**(i % 7) + fr.exp(x - i) for i in range(60)]
        s = exprs[0]
        for e in exprs[1:]:
            s = s + e
        assert s.free_symbols  # 触达遍历路径
        del exprs, s           # 整批释放（迭代式 teardown 路径）
    # interning 语义在 churn 后保持
    assert fr.sin(x * 5) + fr.sin(x * 5) == 2 * fr.sin(x * 5)
