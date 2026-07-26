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


def has_pow_of_mul_or_pow(sexpr) -> bool:
    """Known Frontier crash (assert cpp/src/expr.cpp:263): building a Mul
    whose factor is a Pow with Mul/Pow base aborts the process.  Both
    from_sympy and fr.diff can hit it.  Minimal repro:
        fr: 2 * ((x*y)**z)   or   2 * ((x**fr.rational(3,2))**y)
    Detect on the sympy side so the fuzzer can skip these expressions."""
    return any(isinstance(p.base, (sp.Mul, sp.Pow))
               for p in sexpr.atoms(sp.Pow))


def check_one(idx, subseed, failures, verbose=False, skip_known_crash=True):
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

    if skip_known_crash and has_pow_of_mul_or_pow(sexpr):
        # KNOWN CRASH class (expr.cpp:263) — skip entirely, see docstring
        return "skip_known_crash(pow_base)", None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--only-idx", type=int, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="resume from this index (after a hard crash)")
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--no-skip-known", action="store_true",
                    help="do not skip the known fr.diff pow-base crash")
    args = ap.parse_args()

    jf = open(args.json_out, "a", encoding="utf-8") if args.json_out else None
    failures: list[Failure] = []
    counts: dict[str, int] = {}
    emitted = 0
    indices = ([args.only_idx] if args.only_idx is not None
               else range(args.start, args.n))
    for i in indices:
        subseed = args.seed * 1000003 + i
        if args.trace:
            print(f"idx={i} subseed={subseed}", file=sys.stderr, flush=True)
        try:
            status, _ = check_one(i, subseed, failures,
                                  verbose=args.only_idx is not None,
                                  skip_known_crash=not args.no_skip_known)
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
        if args.only_idx is None and (i + 1) % 100 == 0:
            print(f"... {i + 1}/{args.n} done, {len(failures)} failure "
                  f"records", flush=True)

    print("\n=== summary (seed={}, n={}) ===".format(args.seed, args.n))
    for k in sorted(counts):
        print(f"  {k:24s} {counts[k]}")
    by_cat: dict[str, list[Failure]] = {}
    for f in failures:
        by_cat.setdefault(f.category, []).append(f)
    print("=== failure records by category ===")
    for cat in sorted(by_cat):
        print(f"  {cat:24s} {len(by_cat[cat])}")
    if jf:
        jf.close()
    n_run = 1 if args.only_idx is not None else args.n - args.start
    print(f"TOTAL expressions={n_run} failure_records={len(failures)}")


if __name__ == "__main__":
    main()
