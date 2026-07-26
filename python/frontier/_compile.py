"""fr.compile()：符号表达式 → 机器码批量核函数。

管线：Expr → (C++) lower/CSE → LLVM IR 文本 → (llvmlite) O3 JIT
     → CompiledFunction（直接吃 NumPy 数组）。
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from frontier import _core
from frontier._jit import jit_compile

_pool: ThreadPoolExecutor | None = None


def _shared_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    return _pool


# 低于此规模多线程调度开销得不偿失
_PARALLEL_THRESHOLD = 8192


class CompiledFunction:
    """JIT 编译后的批量数值函数。

    调用约定：按 ``args`` 顺序传入等长一维 float64 数组（标量自动广播），
    返回与输出表达式等长的数组元组（单输出直接返回数组）。
    """

    def __init__(self, exprs, args, *, fastmath: bool = False,
                 vecmath: bool = True, workers: int | str = 1,
                 cache: bool = False, uniform=None,
                 name: str = "frontier_kernel"):
        if workers == "auto":
            workers = os.cpu_count() or 1
        self.workers = max(1, int(workers))
        self._single = not isinstance(exprs, (list, tuple))
        expr_list = [exprs] if self._single else list(exprs)
        arg_list = list(args)
        if not expr_list:
            raise ValueError("compile(): no expressions given")
        # 零参（纯常量表达式）合法：调用返回长度 1 的数组

        expr_list = [_core.as_expr(e) for e in expr_list]
        self.exprs = expr_list
        self.args = arg_list
        self.n_outputs = len(expr_list)
        self.n_inputs = len(arg_list)

        # uniform：这些参数在一个批量中取同一标量值（如拟合参数），
        # 内核在循环外载入一次，依赖它们的子表达式被 LLVM 提升出循环
        if uniform is None:
            self._uniform = [False] * self.n_inputs
        else:
            uset = {_core.as_expr(u) for u in uniform}  # 结构相等（interning）
            self._uniform = [a in uset for a in arg_list]
            missing = uset - set(arg_list)
            if missing:
                raise ValueError(
                    "uniform contains expression(s) not in args: "
                    f"{[str(m) for m in missing]}")

        self.ir = _core.emit_llvm_ir(expr_list, arg_list, name, fastmath, vecmath,
                                     self._uniform)
        self._kernel = jit_compile(self.ir, name, cache)
        self._opts = dict(fastmath=fastmath, vecmath=vecmath, cache=cache)
        self._tls = threading.local()  # eval_scalars 的线程局部缓冲

    def __call__(self, *arrays):
        ins, n = self._prepare_inputs(arrays)
        buf = self._run(ins, n)
        if self._single:
            return buf[0]
        return tuple(buf)  # 行视图，零拷贝

    def _prepare_inputs(self, arrays):
        if len(arrays) != self.n_inputs:
            raise TypeError(
                f"expected {self.n_inputs} input array(s) "
                f"(arguments: {[str(a) for a in self.args]}), got {len(arrays)}")
        for i, a in enumerate(arrays):
            if np.iscomplexobj(a):
                raise TypeError(
                    f"argument '{self.args[i]}' is complex; frontier kernels "
                    "compute in float64 (complex support is on the roadmap). "
                    "Pass .real explicitly if that is what you want")
        ins = [np.ascontiguousarray(a, dtype=np.float64) for a in arrays]
        n = 1
        for i, (a, is_uni) in enumerate(zip(ins, self._uniform)):
            if is_uni or a.ndim == 0:
                continue
            if a.ndim != 1:
                raise ValueError(
                    f"argument '{self.args[i]}' has shape {a.shape}; compiled "
                    "functions take scalars or 1-D arrays — for grids, pass "
                    "X.ravel() and reshape the output")
            if n != 1 and a.shape[0] not in (1, n):
                raise ValueError(
                    f"argument '{self.args[i]}' has length {a.shape[0]}, "
                    f"but an earlier argument has length {n}")
            n = max(n, a.shape[0])

        prepared = []
        for i, (a, is_uni) in enumerate(zip(ins, self._uniform)):
            if is_uni:
                # uniform：只需 1 元素缓冲，不物化整条数组
                flat = np.atleast_1d(a)
                if flat.shape[0] != 1:
                    raise ValueError(
                        f"argument '{self.args[i]}' was declared uniform and "
                        "must be a scalar (one value for the whole batch)")
                prepared.append(np.ascontiguousarray(flat))
            elif a.ndim == 0:
                prepared.append(np.full(n, a, dtype=np.float64))
            elif a.shape[0] == 1 and n > 1:
                prepared.append(np.full(n, a[0], dtype=np.float64))
            else:
                prepared.append(a)
        return prepared, n

    def _run(self, ins, n: int) -> np.ndarray:
        """执行内核，输出写入单块连续 (n_outputs, n) 缓冲。"""
        buf = np.empty((self.n_outputs, n), dtype=np.float64)
        if self.workers > 1 and n >= _PARALLEL_THRESHOLD:
            self._call_parallel(ins, buf, n)
        else:
            in_ptrs = (ctypes.c_void_p * self.n_inputs)(
                *(a.ctypes.data for a in ins))
            row = buf.ctypes.data
            stride = 8 * n
            out_ptrs = (ctypes.c_void_p * self.n_outputs)(
                *(row + stride * i for i in range(self.n_outputs)))
            self._kernel.cfunc(in_ptrs, out_ptrs, n)
        return buf

    def eval_stacked(self, *arrays) -> np.ndarray:
        """批量求值，直接返回 (n_outputs, n) 二维数组（无 stack 拷贝）。

        等价于 np.stack(self(*arrays))，但输出本就写在同一块连续内存里。
        大批量多输出场景（Jacobian、质量矩阵）建议使用。
        """
        ins, n = self._prepare_inputs(arrays)
        return self._run(ins, n)

    def _call_parallel(self, ins, buf: np.ndarray, n: int) -> None:
        """分块多线程求值：ctypes 调用释放 GIL，内核跨块无共享状态。"""
        w = min(self.workers, max(1, n // (_PARALLEL_THRESHOLD // 4)))
        bounds = [n * i // w for i in range(w + 1)]
        base = buf.ctypes.data
        stride = 8 * n

        def run(lo: int, hi: int) -> None:
            # uniform 输入只有 1 个元素的缓冲，绝不能加块偏移
            in_ptrs = (ctypes.c_void_p * self.n_inputs)(
                *(a.ctypes.data if is_uni else a.ctypes.data + 8 * lo
                  for a, is_uni in zip(ins, self._uniform)))
            out_ptrs = (ctypes.c_void_p * self.n_outputs)(
                *(base + stride * i + 8 * lo for i in range(self.n_outputs)))
            self._kernel.cfunc(in_ptrs, out_ptrs, hi - lo)

        futures = [_shared_pool().submit(run, bounds[i], bounds[i + 1])
                   for i in range(w)]
        for f in futures:
            f.result()

    def eval_scalars(self, *vals) -> np.ndarray:
        """单点快路径：标量输入 → shape (n_outputs,) 数组。

        复用线程局部的预分配缓冲，绕过 __call__ 的广播/物化机制；
        ODE 右端函数这类「每步只算一个点」的场景专用。线程安全。
        """
        if len(vals) != self.n_inputs:
            raise TypeError(
                f"eval_scalars: expected {self.n_inputs} value(s) "
                f"(one per argument {[str(a) for a in self.args]}), "
                f"got {len(vals)}")
        bufs = getattr(self._tls, "scalar_buffers", None)
        if bufs is None:
            ins = np.empty(self.n_inputs)
            out = np.empty(self.n_outputs)
            in_ptrs = (ctypes.c_void_p * self.n_inputs)(
                *(ins.ctypes.data + 8 * i for i in range(self.n_inputs)))
            out_ptrs = (ctypes.c_void_p * self.n_outputs)(
                *(out.ctypes.data + 8 * i for i in range(self.n_outputs)))
            bufs = self._tls.scalar_buffers = (ins, out, in_ptrs, out_ptrs)
        ins, out, in_ptrs, out_ptrs = bufs
        ins[:] = vals
        self._kernel.cfunc(in_ptrs, out_ptrs, 1)
        return out.copy()

    @property
    def optimized_ir(self) -> str:
        """优化后的 LLVM IR（调试/教学用）。"""
        return self._kernel.optimized_ir

    def __repr__(self) -> str:
        return (f"<CompiledFunction {self.n_inputs} input(s) -> "
                f"{self.n_outputs} output(s)>")

    # pickle：序列化符号表达式与编译选项，反序列化时重新 JIT
    #（配合 cache=True 时重载几乎零开销）。支持 joblib/multiprocessing。
    def __reduce__(self):
        state = dict(fastmath=self._opts["fastmath"],
                     vecmath=self._opts["vecmath"],
                     workers=self.workers, cache=self._opts["cache"],
                     uniform=[a for a, u in zip(self.args, self._uniform) if u]
                     or None)
        exprs = self.exprs[0] if self._single else list(self.exprs)
        return (_rebuild_compiled, (exprs, list(self.args), state))


def _rebuild_compiled(exprs, args, state):
    return CompiledFunction(exprs, args, **state)


def compile(exprs, args: Sequence, *, fastmath: bool = False,
            vecmath: bool = True, workers: int | str = 1,
            cache: bool = False, uniform=None) -> CompiledFunction:  # noqa: A001
    """把符号表达式编译为机器码批量函数。

    Parameters
    ----------
    exprs : Expr | list[Expr]
        输出表达式（列表 → 多输出）。
    args : sequence[Expr]
        输入符号，调用时按此顺序传数组。
    fastmath : bool
        允许浮点重结合等松弛优化（默认关闭，保证与逐点求值一致）。
    vecmath : bool
        sin/cos/exp/log 使用可内联多项式实现（1-2 ulp 误差），
        解锁循环 SIMD 向量化；False 则严格调用 libm。
    workers : int | "auto"
        批量求值的线程数（"auto" = CPU 核数）。内核逐块独立执行，
        ctypes 调用释放 GIL；小批量自动退回单线程。
    cache : bool
        把编译产物持久化到磁盘（~/.cache/frontier 或 $FRONTIER_CACHE_DIR），
        跨进程复用：同一表达式第二次启动免优化免编译。
    uniform : sequence[Expr] | None
        批量中取同一标量值的参数（如拟合参数）。内核在循环外载入一次，
        只依赖它们的子表达式被 LLVM 提升出循环（符号级 LICM）。
    """
    return CompiledFunction(exprs, args, fastmath=fastmath, vecmath=vecmath,
                            workers=workers, cache=cache, uniform=uniform)
