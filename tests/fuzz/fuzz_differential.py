"""Differential fuzzer: Frontier vs SymPy.

Randomly generates sympy expressions (depth 3-8, mixed arithmetic / pow /
functions / rational & float constants, 1-4 variables), then checks:
  (a) fr.from_sympy conversion succeeds;
  (b) fr.compile'd value matches sympy at random sample points (points where
      sympy's own high-precision evaluation is non-finite / complex are
      filtered out first), rtol 1e-9;
  (c) first derivatives (fr.diff vs sympy.diff) match the same way.

Reference strategy: the primary comparison is Frontier (double) vs
sympy.lambdify/numpy (double).  On mismatch, sympy's 30-digit evalf acts as
arbiter: whichever side is closer to the high-precision value is "right";
if neither side matches evalf the point is classified as ill-conditioned
(double roundoff, not a bug).

Usage:
    PYTHONPATH=python python tests/fuzz/fuzz_differential.py --seed 1 --n 500
Options:
    --seed N        base seed (each expression i uses subseed seed*1000003+i)
    --n N           number of expressions
    --only-idx I    re-run just expression index I (verbose repro mode)
    --json-out F    append failure records as JSON lines to file F
    --trace         print every index to stderr (crash localisation)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
import warnings

import numpy as np
import sympy as sp

import frontier as fr

RTOL = 1e-9
ATOL = 1e-12
PREC = 30          # evalf digits for the high-precision arbiter
MAX_MAG = 1e100    # reference values beyond this are treated as overflow-zone
N_POINTS = 3       # valid sample points wanted per expression
N_CANDIDATES = 40  # candidate points tried to find valid ones

# --------------------------------------------------------------------------
# 失败类别的唯一定义（CI 冒烟测试从这里 import，禁止在别处复制字符串）。
#
# BUG_CATEGORIES     —— Frontier 缺陷：出现任意一条，进程退出码非零。
# REVIEW_CATEGORIES  —— 需要人工裁定（高条件数点等）：默认不置非零，
#                       --strict 下同样视为失败。
# EXPECTED_CATEGORIES—— 已裁定的预期行为（实数域语义 / sympy 侧更差 /
#                       明确不支持的转换），永不置非零。
# --------------------------------------------------------------------------
BUG_CATEGORIES = frozenset({
    "CONVERT_BUG",        # from_sympy 对受支持节点报错/错译
    "DERIV_RAISE",        # fr.diff 意外抛异常
    "VALUE_MISMATCH",     # 求值偏差且高精度仲裁站在 sympy 一侧
    "DERIV_MISMATCH",     # 导数偏差且高精度仲裁站在 sympy 一侧
    "HARD_CRASH",         # 崩溃（由 runner 检出）
    "HARNESS_ERROR",      # 测试器自身异常（视为未证清白）
})
REVIEW_CATEGORIES = frozenset({
    "VALUE_SUSPECT",      # 双方都偏离高精度值：病态/精度损失候审
    "DERIV_SUSPECT",
})
EXPECTED_CATEGORIES = frozenset({
    "EXPECTED_COMPLEX_INTERMEDIATE",
    "SYMPY_SIDE_DIFF",
    "CONVERT_UNSUPPORTED",
})

FUNCS1 = ["sin", "cos", "tan", "asin", "acos", "atan",
          "sinh", "cosh", "tanh", "exp", "log", "sqrt", "abs", "sign"]
FUNCS2 = ["atan2", "max", "min"]

_SP_FUNC = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "sign": sp.sign,
    "atan2": sp.atan2, "max": sp.Max, "min": sp.Min,
}


# --------------------------------------------------------------------------
# random expression generator
# --------------------------------------------------------------------------

def _leaf(rng: random.Random, syms):
    r = rng.random()
    if r < 0.55:
        return rng.choice(syms)
    if r < 0.72:
        return sp.Integer(rng.randint(-5, 5))
    if r < 0.87:
        p = rng.randint(-9, 9)
        q = rng.randint(1, 9)
        return sp.Rational(p, q)
    return sp.Float(round(rng.uniform(-3.0, 3.0), 4))


def _exponent(rng: random.Random, syms, depth):
    r = rng.random()
    if r < 0.5:
        return sp.Integer(rng.choice([-3, -2, -1, 2, 3]))
    if r < 0.8:
        return rng.choice([sp.Rational(1, 2), sp.Rational(-1, 2),
                           sp.Rational(3, 2), sp.Rational(1, 3),
                           sp.Rational(2, 3), sp.Rational(-1, 3)])
    if r < 0.9:
        return sp.Float(round(rng.uniform(-2.0, 2.0), 3))
    # small symbolic exponent
    return gen_expr(rng, syms, min(depth - 1, 2))


def gen_expr(rng: random.Random, syms, depth):
    if depth <= 0 or rng.random() < 0.12:
        return _leaf(rng, syms)
    r = rng.random()
    if r < 0.52:  # binary arithmetic
        a = gen_expr(rng, syms, depth - 1)
        b = gen_expr(rng, syms, depth - 1)
        op = rng.random()
        if op < 0.3:
            return a + b
        if op < 0.55:
            return a - b
        if op < 0.85:
            return a * b
        return a / b
    if r < 0.66:  # power
        base = gen_expr(rng, syms, depth - 1)
        return base ** _exponent(rng, syms, depth)
    if r < 0.94:  # unary function
        f = rng.choice(FUNCS1)
        return _SP_FUNC[f](gen_expr(rng, syms, depth - 1))
    # binary function
    f = rng.choice(FUNCS2)
    return _SP_FUNC[f](gen_expr(rng, syms, depth - 1),
                       gen_expr(rng, syms, depth - 1))


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------

def sym_ref(expr, subs):
    """High-precision sympy evaluation -> finite real float, or None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # maxn caps precision escalation near branch points (sign/Min/
            # atan2 nests can otherwise take ~minutes per point in sympy)
            v = expr.evalf(PREC, subs=subs, maxn=100)
    except Exception:
        return None
    if not getattr(v, "is_number", False):
        return None
    if v.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return None
    try:
        c = complex(v)
    except Exception:
        return None
    if not math.isfinite(c.real):
        return None
    if c.imag != 0.0 and abs(c.imag) > 1e-18 * max(1.0, abs(c.real)):
        return None
    if abs(c.real) > MAX_MAG:
        return None
    return c.real


def has_nonreal_subexpr(expr, subs) -> bool:
    """True if some subexpression is complex / non-finite / huge at the point.

    Frontier is real-double-domain: pow(neg, frac) = nan, log(neg) = nan, ...
    SymPy evaluates on the principal complex branch, so e.g. Abs(x**(2/7))
    is real for x<0 in sympy but nan in frontier (and numpy).  A NaN from
    frontier with such an intermediate is an expected domain difference."""
    for node in sp.preorder_traversal(expr):
        if node.is_Atom:
            continue
        if sym_ref(node, subs) is None:
            return True
    return False


def close(a, b, rtol=RTOL, atol=ATOL):
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def double_pair_equal(f, l):
    """Frontier vs lambdify, both double semantics."""
    if math.isnan(f) and math.isnan(l):
        return True
    if math.isinf(f) and math.isinf(l) and (f > 0) == (l > 0):
        return True
    if math.isnan(f) != math.isnan(l):
        return False
    if math.isnan(f):
        return True
    return close(f, l)


# --------------------------------------------------------------------------
# per-expression check
# --------------------------------------------------------------------------

class Failure:
    def __init__(self, category, idx, subseed, expr_str, detail):
        self.category = category
        self.idx = idx
        self.subseed = subseed
        self.expr_str = expr_str
        self.detail = detail

    def to_dict(self):
        return {"category": self.category, "idx": self.idx,
                "subseed": self.subseed, "expr": self.expr_str,
                **self.detail}


def check_one(idx, subseed, failures, verbose=False):
    rng = random.Random(subseed)
    nvars = rng.randint(1, 4)
    svars = [sp.Symbol(f"x{i}", real=True) for i in range(nvars)]
    depth = rng.randint(3, 8)
    try:
        sexpr = gen_expr(rng, svars, depth)
    except Exception as ex:  # sympy itself refused (e.g. zoo in auto-simplify)
        return "gen_skip", None
    expr_str = str(sexpr)
    if verbose:
        print(f"[repro] idx={idx} subseed={subseed} nvars={nvars} "
              f"depth={depth}\n  expr = {expr_str}")

    if sexpr.has(sp.zoo, sp.oo, -sp.oo, sp.nan, sp.I):
        return "gen_skip", None  # sympy folded to an unsupported special value
    # 历史注记：这里曾有 Pow(Mul/Pow 底) 崩溃类的跳过逻辑；该缺陷已修复
    # 并由 tests/test_regressions.py 独立回归，跳过逻辑随之删除。

    # ---- (a) conversion --------------------------------------------------
    try:
        fexpr = fr.from_sympy(sexpr)
    except NotImplementedError as ex:
        failures.append(Failure("CONVERT_UNSUPPORTED", idx, subseed, expr_str,
                                {"error": str(ex)}))
        return "convert_unsupported", None
    except Exception as ex:
        failures.append(Failure("CONVERT_BUG", idx, subseed, expr_str,
                                {"error": f"{type(ex).__name__}: {ex}"}))
        return "convert_bug", None

    # ---- derivatives ------------------------------------------------------
    fvars = [fr.symbol(v.name) for v in svars]
    test_derivs = True
    fderivs, sderivs = [], []
    if test_derivs:
        try:
            fderivs = [fr.diff(fexpr, v) for v in fvars]
        except Exception as ex:
            failures.append(Failure("DERIV_RAISE", idx, subseed, expr_str,
                                    {"error": f"{type(ex).__name__}: {ex}"}))
            return "deriv_raise", None
        try:
            sderivs = [sp.diff(sexpr, v) for v in svars]
        except Exception:
            sderivs = []
            fderivs = []
            test_derivs = False

    # ---- (b) compile ------------------------------------------------------
    try:
        fn = fr.compile([fexpr, *fderivs], args=fvars)
    except Exception as ex:
        failures.append(Failure("COMPILE_BUG", idx, subseed, expr_str,
                                {"error": f"{type(ex).__name__}: {ex}"}))
        return "compile_bug", None

    # sympy double-precision side.  NumPyPrinter hangs (minutes) on
    # derivative expressions containing Heaviside/DiracDelta (from
    # Min/Max/sign differentiation) — only lambdify those when clean;
    # otherwise lambdify the value alone and check derivatives vs evalf.
    custom = {"DiracDelta": lambda *a: 0.0}
    hard_print = any(d.has(sp.Heaviside, sp.DiracDelta) for d in sderivs)
    sfn, sfn_nout = None, 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not hard_print:
                sfn = sp.lambdify(svars, [sexpr, *sderivs],
                                  modules=[custom, "numpy"])
                sfn_nout = 1 + len(sderivs)
            else:
                sfn = sp.lambdify(svars, [sexpr], modules=[custom, "numpy"])
                sfn_nout = 1
    except Exception:
        sfn, sfn_nout = None, 0

    # ---- sample points (filtered through sympy high-precision eval) -------
    outputs = [sexpr, *sderivs]
    labels = ["value"] + [f"d/d{v.name}" for v in svars][:len(sderivs)]
    points, refs = [], []
    t_budget = time.time() + 20.0   # cap sympy-evalf pathologies per expr
    for _ in range(N_CANDIDATES):
        if len(points) >= N_POINTS or time.time() > t_budget:
            break
        if rng.random() < 0.6:
            pt = [round(rng.uniform(0.05, 2.5), 6) for _ in range(nvars)]
        else:
            pt = [round(rng.uniform(-3.0, 3.0), 6) for _ in range(nvars)]
        # fast double-precision pre-screen: skip points where numpy/sympy
        # already yields nan/inf/complex — avoids pathological evalf calls
        if sfn is not None:
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    quick = sfn(*[float(p) for p in pt])
                bad = False
                for v in quick:
                    c = complex(v)
                    if (not math.isfinite(c.real) or c.imag != 0.0
                            or abs(c.real) > MAX_MAG):
                        bad = True
                        break
                if bad:
                    continue
            except Exception:
                continue
        subs = {v: sp.Rational(p) for v, p in zip(svars, pt)}
        vals = []
        ok = True
        for out in outputs:
            if time.time() > t_budget:
                ok = False
                break
            r = sym_ref(out, subs)
            if r is None:
                ok = False
                break
            vals.append(r)
        if ok:
            points.append(pt)
            refs.append(vals)
    if not points:
        return "no_valid_points", None

    # ---- evaluate & compare ------------------------------------------------
    status = "ok"
    for pt, ref in zip(points, refs):
        try:
            raw_f = fn(*[np.array([p]) for p in pt])
            fvals = [float(np.asarray(v).ravel()[0]) for v in raw_f]
        except Exception as ex:
            failures.append(Failure("RUNTIME_CRASH", idx, subseed, expr_str,
                                    {"point": pt,
                                     "error": f"{type(ex).__name__}: {ex}"}))
            return "runtime_crash", None
        lvals = None
        if sfn is not None:
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    raw = sfn(*[float(p) for p in pt])
                lvals = []
                for v in raw:
                    v = complex(v)
                    lvals.append(v.real if abs(v.imag) <= 1e-18 *
                                 max(1.0, abs(v.real)) else None)
                while len(lvals) < len(labels):
                    lvals.append(None)   # derivatives not lambdified
            except Exception:
                lvals = None

        for j, (label, f_v, r_v) in enumerate(zip(labels, fvals, ref)):
            l_v = lvals[j] if lvals is not None else None
            if l_v is not None:
                if double_pair_equal(f_v, l_v):
                    continue
                fr_ok = close(f_v, r_v)
                lam_ok = close(l_v, r_v)
                if fr_ok and not lam_ok:
                    cat = "SYMPY_SIDE_DIFF"       # sympy/numpy quirk, not fr
                elif lam_ok and not fr_ok:
                    cat = ("VALUE_MISMATCH" if j == 0 else "DERIV_MISMATCH")
                elif not fr_ok and not lam_ok:
                    cat = "ILL_CONDITIONED"
                else:
                    continue
            else:
                if not math.isnan(f_v) and close(f_v, r_v):
                    continue
                cat = ("VALUE_SUSPECT" if j == 0 else "DERIV_SUSPECT")
            # frontier NaN with a complex/overflow intermediate on the sympy
            # side is an expected real-domain difference, not a bug
            if math.isnan(f_v) and cat in ("VALUE_MISMATCH", "DERIV_MISMATCH",
                                           "VALUE_SUSPECT", "DERIV_SUSPECT"):
                subs = {v: sp.Rational(p) for v, p in zip(svars, pt)}
                if has_nonreal_subexpr(outputs[j], subs):
                    cat = "EXPECTED_COMPLEX_INTERMEDIATE"
            failures.append(Failure(cat, idx, subseed, expr_str, {
                "output": label, "point": pt,
                "frontier": f_v, "sympy_lambdify": l_v,
                "sympy_evalf": r_v,
                "rel_fr_vs_ref": _rel(f_v, r_v),
                "rel_lam_vs_ref": _rel(l_v, r_v) if l_v is not None else None,
            }))
            status = "mismatch"
    if status == "ok" and not test_derivs:
        status = "ok_value_only(diff_crash_skipped)"
    return status, None


def _rel(a, b):
    if a is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    d = max(abs(a), abs(b))
    return abs(a - b) / d if d else 0.0


# --------------------------------------------------------------------------

def run(seed: int, n: int, *, start: int = 0, only_idx=None,
        json_out=None, trace: bool = False, strict: bool = False) -> dict:
    """跑一批随机表达式，返回结构化汇总（供 CLI 与 pytest 共用）。

    返回 dict 含 exit_code：BUG_CATEGORIES 有记录即非零；
    strict=True 时 REVIEW_CATEGORIES 同样非零。
    """
    jf = open(json_out, "a", encoding="utf-8") if json_out else None
    failures: list[Failure] = []
    counts: dict[str, int] = {}
    emitted = 0
    indices = [only_idx] if only_idx is not None else range(start, n)
    for i in indices:
        subseed = seed * 1000003 + i
        if trace:
            print(f"idx={i} subseed={subseed}", file=sys.stderr, flush=True)
        try:
            status, _ = check_one(i, subseed, failures,
                                  verbose=only_idx is not None)
        except Exception:
            failures.append(Failure("HARNESS_ERROR", i, subseed, "?",
                                    {"error": traceback.format_exc(limit=3)}))
            status = "harness_error"
        counts[status] = counts.get(status, 0) + 1
        # emit new failure records immediately (crash-safe)
        while emitted < len(failures):
            rec = failures[emitted].to_dict()
            print("FAIL " + json.dumps(rec, default=str), flush=True)
            if jf:
                jf.write(json.dumps(rec, default=str) + "\n")
                jf.flush()
            emitted += 1
        if only_idx is None and (i + 1) % 100 == 0:
            print(f"... {i + 1}/{n} done, {len(failures)} failure "
                  f"records", flush=True)
    if jf:
        jf.close()

    by_cat: dict[str, int] = {}
    for f in failures:
        by_cat[f.category] = by_cat.get(f.category, 0) + 1

    unknown = set(by_cat) - BUG_CATEGORIES - REVIEW_CATEGORIES - EXPECTED_CATEGORIES
    bug_n = sum(v for k, v in by_cat.items() if k in BUG_CATEGORIES)
    review_n = sum(v for k, v in by_cat.items() if k in REVIEW_CATEGORIES)
    # 未登记的类别一律按 bug 处理（新类别必须先归档再上线）
    bug_n += sum(v for k, v in by_cat.items() if k in unknown)

    exit_code = 1 if bug_n or (strict and review_n) else 0
    summary = {
        "seed": seed,
        "n": (1 if only_idx is not None else n - start),
        "statuses": counts,
        "failures_by_category": by_cat,
        "bug_records": bug_n,
        "review_records": review_n,
        "expected_records": sum(v for k, v in by_cat.items()
                                if k in EXPECTED_CATEGORIES),
        "unknown_categories": sorted(unknown),
        "strict": strict,
        "exit_code": exit_code,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--only-idx", type=int, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="resume from this index (after a hard crash)")
    ap.add_argument("--json-out", type=str, default=None,
                    help="append failure records as JSON lines to this file")
    ap.add_argument("--json-summary", type=str, default=None,
                    help="write the run summary as JSON to this file")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="REVIEW categories (ill-conditioned suspects) also fail")
    args = ap.parse_args()

    summary = run(args.seed, args.n, start=args.start, only_idx=args.only_idx,
                  json_out=args.json_out, trace=args.trace, strict=args.strict)

    print(f"\n=== summary (seed={summary['seed']}, n={summary['n']}) ===")
    for k in sorted(summary["statuses"]):
        print(f"  {k:24s} {summary['statuses'][k]}")
    print("=== failure records by category ===")
    for cat in sorted(summary["failures_by_category"]):
        print(f"  {cat:24s} {summary['failures_by_category'][cat]}")
    print(f"TOTAL expressions={summary['n']} "
          f"bug={summary['bug_records']} review={summary['review_records']} "
          f"expected={summary['expected_records']}")
    print("SUMMARY " + json.dumps(summary, default=str))
    if args.json_summary:
        with open(args.json_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
