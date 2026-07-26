"""开发树内跑 pytest 时注入源码路径；已安装（wheel/CI 测试）时不干预。

守卫条件：仅当源码树 python/frontier/ 下存在本平台已构建的 _core
扩展、且当前环境未安装 frontier 时，才把源码树加入 sys.path——
避免在 cibuildwheel 等场景下遮蔽刚安装进 site-packages 的 wheel。
"""
import importlib.util
import sys
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parent.parent / "python"


def _has_built_core() -> bool:
    pkg = _PY_DIR / "frontier"
    return pkg.is_dir() and any(pkg.glob("_core.*.pyd")) or any(pkg.glob("_core.*.so"))


def _frontier_installed() -> bool:
    spec = importlib.util.find_spec("frontier")
    if spec is None or spec.origin is None:
        return False
    # find_spec 可能命中我们即将注入的源码树——只认 site-packages 安装
    return "site-packages" in spec.origin.replace("\\", "/")


if _has_built_core() and not _frontier_installed():
    if str(_PY_DIR) not in sys.path:
        sys.path.insert(0, str(_PY_DIR))
