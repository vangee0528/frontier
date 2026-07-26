"""llvmlite JIT 引擎封装。

C++ 后端产出 LLVM IR 文本，本模块负责：解析 → O3 优化（含循环/SLP
向量化）→ MCJIT 编译 → 返回内核函数指针。模块按 IR 文本缓存。

llvmlite 只出现在这一层（架构决策：规避 Windows 上链接 LLVM C++ 库）。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import llvmlite
import llvmlite.binding as llvm

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    # llvmlite ≥0.48 弃用了 initialize()（调用即抛 RuntimeError），
    # 但 native target/asmprinter 仍需显式注册；旧版本三者都要调用
    try:
        llvm.initialize()
    except RuntimeError:
        pass
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _register_libm_symbols()
    _initialized = True


def _register_libm_symbols() -> None:
    """把 libm 符号显式注册进 JIT 符号表。

    llvm.sin.f64 等 intrinsic 会降级为对 sin/cos/... 的调用；
    MCJIT 默认从进程符号解析，Windows 上不一定可靠，显式注册兜底。
    """
    names = [
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "exp", "log", "pow", "sqrt", "fabs", "fmod",
    ]
    candidates = []
    for lib in ("ucrtbase", "msvcrt", "m", "c"):
        try:
            candidates.append(ctypes.CDLL(ctypes.util.find_library(lib) or lib))
        except OSError:
            continue
    for name in names:
        for lib in candidates:
            fn = getattr(lib, name, None)
            if fn is not None:
                llvm.add_symbol(name, ctypes.cast(fn, ctypes.c_void_p).value)
                break


def _new_target_machine():
    """每个引擎独立的 TargetMachine。

    注意：不能跨引擎共享——create_mcjit_compiler 会接管 TM 所有权，
    共享会在引擎析构时 use-after-free（access violation）。
    """
    _ensure_initialized()
    target = llvm.Target.from_default_triple()
    return target.create_target_machine(
        cpu=llvm.get_host_cpu_name(),
        features=llvm.get_host_cpu_features().flatten(),
        opt=3,
    )


def _optimize(mod, tm) -> None:
    pto = llvm.create_pipeline_tuning_options()
    pto.speed_level = 3
    pto.loop_vectorization = True
    pto.slp_vectorization = True
    pb = llvm.create_pass_builder(tm, pto)
    pm = pb.getModulePassManager()
    pm.run(mod, pb)


# 内核 C 签名：void kernel(double** ins, double** outs, int64 n)
KERNEL_CFUNC = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int64,
)


def _cache_dir() -> Path:
    d = os.environ.get("FRONTIER_CACHE_DIR")
    return Path(d) if d else Path.home() / ".cache" / "frontier"


def _cache_key(ir_text: str) -> str:
    """磁盘缓存键：IR 文本 + llvmlite 版本 + 宿主 CPU（对象码不可跨机复用）。"""
    salt = f"{llvmlite.__version__}|{llvm.get_host_cpu_name()}|"
    return hashlib.sha256((salt + ir_text).encode()).hexdigest()[:32]


class JitKernel:
    """持有 JIT 引擎与内核函数指针（引擎存活期 = 对象存活期）。

    cache=True 时把编译产物（目标文件）持久化到磁盘：
    第二次进程启动同一表达式可跳过整个 LLVM 优化+编译（大模型收益显著）。
    """

    def __init__(self, ir_text: str, kernel_name: str, cache: bool = False):
        _ensure_initialized()
        tm = _new_target_machine()

        obj_path: Path | None = None
        hit = False
        if cache:
            obj_path = _cache_dir() / f"{_cache_key(ir_text)}.o"
            hit = obj_path.is_file()

        try:
            mod = llvm.parse_assembly(ir_text)
            mod.verify()
        except RuntimeError as exc:  # pragma: no cover - 后端产出非法 IR 属于 bug
            raise RuntimeError(f"invalid LLVM IR from backend:\n{exc}") from exc

        if not hit:  # 缓存命中时跳过优化——对象码直接复用
            _optimize(mod, tm)
        self.optimized_ir = str(mod)
        self.cache_hit = hit

        self._engine = llvm.create_mcjit_compiler(mod, tm)
        if cache and obj_path is not None:
            path = obj_path

            def _notify(_mod, buf) -> None:
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".tmp{os.getpid()}")
                    tmp.write_bytes(buf)
                    tmp.replace(path)  # 原子替换，进程间安全

            def _getbuffer(_mod):
                return path.read_bytes() if hit else None

            self._engine.set_object_cache(notify_func=_notify,
                                          getbuffer_func=_getbuffer)
        self._engine.finalize_object()
        addr = self._engine.get_function_address(kernel_name)
        if not addr:
            raise RuntimeError(f"kernel '{kernel_name}' not found after JIT")
        self.cfunc = KERNEL_CFUNC(addr)


@lru_cache(maxsize=256)
def jit_compile(ir_text: str, kernel_name: str, cache: bool = False) -> JitKernel:
    """按 (IR 文本, 内核名) 缓存的 JIT 编译入口（进程内 LRU + 可选磁盘缓存）。"""
    return JitKernel(ir_text, kernel_name, cache)
