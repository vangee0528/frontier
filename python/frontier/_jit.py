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
    """磁盘缓存键。对象码严格绑定生成环境——键包含：

    IR 文本、llvmlite 版本、target triple、操作系统、宿主 CPU 名与
    完整特性串。缓存目录被跨机器/容器复制时自动失效而非误用。
    """
    _ensure_initialized()
    salt = "|".join([
        llvmlite.__version__,
        llvm.get_process_triple(),
        os.name,
        llvm.get_host_cpu_name(),
        llvm.get_host_cpu_features().flatten(),
        "",
    ])
    return hashlib.sha256((salt + ir_text).encode()).hexdigest()[:32]


def _obj_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class JitKernel:
    """持有 JIT 引擎与内核函数指针（引擎存活期 = 对象存活期）。

    cache=True 时把编译产物（目标文件）持久化到磁盘：
    第二次进程启动同一表达式可跳过整个 LLVM 优化+编译（大模型收益显著）。
    """

    def __init__(self, ir_text: str, kernel_name: str, cache: bool = False):
        _ensure_initialized()
        tm = _new_target_machine()

        obj_path: Path | None = None
        cached_obj: bytes | None = None
        if cache:
            obj_path = _cache_dir() / f"{_cache_key(ir_text)}.o"
            cached_obj = self._load_cached(obj_path)
        hit = cached_obj is not None

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
            obj = cached_obj

            def _notify(_mod, buf) -> None:
                sha_path = path.with_suffix(".sha")
                if path.is_file() and sha_path.is_file():
                    return  # 完整缓存已存在（并发写者先到）
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}")
                tmp_sha = path.with_suffix(f".sha_tmp{os.getpid()}")
                tmp.write_bytes(buf)
                tmp_sha.write_text(_obj_digest(bytes(buf)))
                # 原子替换，进程间安全；先 .o 后 .sha（读取端以 .sha 为准）
                tmp.replace(path)
                tmp_sha.replace(sha_path)

            def _getbuffer(_mod):
                return obj  # 已校验的缓存字节；None 则触发正常编译

            self._engine.set_object_cache(notify_func=_notify,
                                          getbuffer_func=_getbuffer)
        self._engine.finalize_object()
        addr = self._engine.get_function_address(kernel_name)
        if not addr:
            raise RuntimeError(f"kernel '{kernel_name}' not found after JIT")
        self.cfunc = KERNEL_CFUNC(addr)

    @staticmethod
    def _load_cached(path: Path) -> bytes | None:
        """读缓存对象码；校验和缺失/不符（截断、损坏、部分写入）时
        忽略缓存并删除坏文件，回到正常编译路径。"""
        sha_path = path.with_suffix(".sha")
        try:
            data = path.read_bytes()
            expect = sha_path.read_text().strip()
        except OSError:
            return None
        if _obj_digest(data) != expect:
            for p in (path, sha_path):
                try:
                    p.unlink()
                except OSError:
                    pass
            return None
        return data


@lru_cache(maxsize=256)
def jit_compile(ir_text: str, kernel_name: str, cache: bool = False) -> JitKernel:
    """按 (IR 文本, 内核名) 缓存的 JIT 编译入口（进程内 LRU + 可选磁盘缓存）。"""
    return JitKernel(ir_text, kernel_name, cache)
