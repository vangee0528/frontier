"""Python API 门面行为测试（符号层）。"""
import pytest

import frontier as fr


def test_symbols():
    x, y = fr.symbols("x y")
    assert repr(x) == "x"
    # 单个名字直接返回符号本身
    z = fr.symbols("z")
    assert repr(z) == "z"
    # 逗号分隔同样支持
    a, b = fr.symbols("a, b")
    assert repr(b) == "b"
    with pytest.raises(ValueError):
        fr.symbols("  ")


def test_operator_overloading():
    x, y = fr.symbols("x y")
    assert repr(2 * x + 3 * x) == "5*x"
    assert repr(x - x) == "0"
    assert repr(x / x) == "1"
    assert repr(x**2 * x) == "x^3"
    assert repr(2 ** x) == "2^x"
    assert repr(-x) == "-x"
    assert repr(1 / x) == "x^(-1)"


def test_structural_equality_and_hash():
    x, y = fr.symbols("x y")
    assert x + y == y + x
    assert hash(x + y) == hash(y + x)
    assert x != y
    assert (x == "not an expr") is False
    # 可作 dict key
    d = {x + y: 1}
    assert d[y + x] == 1


def test_exact_arithmetic():
    x = fr.symbols("x")
    # 精确有理数不掉精度
    e = fr.rational(1, 3) * x * 3
    assert repr(e) == "x"
    # sqrt(2) 保持符号（经 rewrite 为 2^(1/2)）
    assert repr(fr.sqrt(fr.as_expr(2))) == "2^(1/2)"
    # sqrt(x) 与 x^(-1/2) 可合并
    assert repr(fr.sqrt(x) * x ** fr.rational(-1, 2)) == "1"


def test_registry_functions_installed():
    x = fr.symbols("x")
    for name in ("sin", "cos", "tan", "exp", "log", "sinh", "tanh"):
        f = getattr(fr, name)
        assert repr(f(x)) == f"{name}(x)"


def test_diff():
    x, y = fr.symbols("x y")
    assert fr.diff(fr.sin(x), x) == fr.cos(x)
    assert fr.diff(fr.sin(x * y), x) == y * fr.cos(x * y)
    # 连续求导
    assert fr.diff(x**3, x, x) == 6 * x
    # 常数求导
    assert fr.diff(fr.as_expr(5), x).is_zero


def test_grad_jacobian():
    x, y = fr.symbols("x y")
    g = fr.grad(x**2 + y**2, [x, y])
    assert g == [2 * x, 2 * y]

    j = fr.jacobian([x * y, x + y], [x, y])
    assert j == [[y, x], [1, 1]]


def test_errors():
    x, y = fr.symbols("x y")
    with pytest.raises(fr.FrontierError):
        fr.diff(x, x + y)  # 对非符号求导
    with pytest.raises(fr.DomainError):
        fr.rational(1, 0)
    with pytest.raises(TypeError):
        x + "hello"
    with pytest.raises(TypeError):
        x + True  # bool 不允许静默当作数
