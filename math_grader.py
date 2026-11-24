"""Deterministic math answer extractor & symbolic/numeric equivalence grader.

Supports:
- integer        : exact integer match
- set            : unordered set of numeric/fractional roots (e.g., quadratic equations)
- expr           : symbolic expression in x (uses SymPy if installed + safe AST numeric probing)
- antiderivative : equivalence up to an arbitrary additive constant + C (SymPy diff or numeric delta)

Security: Never calls raw eval(); uses an allowlisted Python AST evaluator.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

try:
    import sympy as _sympy  # type: ignore
except ImportError:
    _sympy = None


SAFE_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sec": lambda v: 1.0 / math.cos(v),
    "csc": lambda v: 1.0 / math.sin(v),
    "cot": lambda v: 1.0 / math.tan(v),
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}

SAFE_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

PROBE_POINTS = (0.17, 0.31, 0.47, 0.63, 0.89, 1.13, 1.37, 1.61, 1.83, 2.09)
ODE_PROBE_PAIRS = (
    (0.5, 1.2),
    (1.1, 1.7),
    (-0.7, 2.1),
    (2.3, 1.5),
    (0.2, 2.8),
    (-1.4, 3.1),
    (1.8, 0.85),
    (3.2, 2.4),
)


def extract_boxed(text: str) -> str | None:
    """Extract the last balanced \\boxed{...} payload (handles nested braces like \\frac{1}{2})."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def _strip_latex_spacing(s: str) -> str:
    s = re.sub(r"\\(?:qquad|quad|,|;|:|!)\s*", " ", s)
    return s.replace(r"\ ", " ")


def _replace_latex_frac(s: str) -> str:
    """Iteratively rewrite \\frac/\\dfrac/\\tfrac (braced or single-char TeX args) into ((A)/(B))."""
    def _read_arg(text: str, pos: int) -> tuple[str | None, int]:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return None, pos
        if text[pos] == "{":
            pos += 1
            depth = 1
            start = pos
            while pos < len(text) and depth > 0:
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    depth -= 1
                pos += 1
            if depth != 0:
                return None, pos
            return text[start : pos - 1], pos
        return text[pos], pos + 1

    while True:
        m = re.search(r"\\[dt]?frac(?![a-zA-Z])", s)
        if not m:
            return s
        num, pos = _read_arg(s, m.end())
        if num is None:
            return s
        den, pos = _read_arg(s, pos)
        if den is None:
            return s
        s = s[: m.start()] + f"(({num})/({den}))" + s[pos:]


def _replace_latex_sqrt(s: str) -> str:
    while True:
        m = re.search(r"\\sqrt\s*\{", s)
        if not m:
            return s
        i = m.end()
        depth = 1
        arg_start = i
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return s
        arg = s[arg_start : i - 1]
        s = s[: m.start()] + f"sqrt({arg})" + s[i:]


def normalize_math_expr(
    raw: str,
    strip_constant_c: bool = False,
    strip_leading_lhs: bool = True,
) -> str:
    """Convert common LaTeX/math notation into a clean Python arithmetic expression."""
    s = raw.strip()
    s = s.replace("$", "").replace(r"\left", "").replace(r"\right", "")
    s = _strip_latex_spacing(s)
    s = re.sub(r"\\text\{([^}]*)\}", r" \1 ", s)

    if strip_leading_lhs:
        # Strip leading LHS like "y(x) = ", "F(x) = ", "f(x) = ", "x = "
        s = re.sub(r"^\s*[a-zA-Z](?:\s*\(\s*[a-zA-Z]\s*\))?\s*=\s*", "", s)

    s = _replace_latex_frac(s)
    s = _replace_latex_sqrt(s)

    # Common LaTeX commands (add leading space so y\ln y -> y ln y, x\sin x -> x sin x)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    for fn in ("sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "ln"):
        s = re.sub(rf"\\{fn}\b", f" {fn}", s)
    s = re.sub(r"\\pi\b", " pi", s)

    # Convert absolute value bars |expr| -> (abs(expr)) (handles \ln|y| -> ln(abs(y)))
    s = re.sub(r"\|([^|]+)\|", r"(abs(\1))", s)

    # Convert e^{...} or e^(...) or e^x to exp(...)
    s = re.sub(r"\be\s*\^\s*\{([^}]+)\}", r"exp(\1)", s)
    s = re.sub(r"\be\s*\^\s*\(([^)]+)\)", r"exp(\1)", s)
    s = re.sub(r"\be\s*\^\s*([a-zA-Z0-9]+)", r"exp(\1)", s)

    # Braces used for grouping exponents: x^{3} -> x**(3)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")

    if strip_constant_c:
        # Strip trailing/leading "+ C", "- C", "+ c", "+ k"
        s = re.sub(r"[+\-]\s*[CcKk]\s*$", "", s)
        s = re.sub(r"^\s*[CcKk]\s*\+\s*", "", s)

    fn_names = "|".join(sorted(SAFE_FUNCS.keys(), key=len, reverse=True))

    # Wrap bare-argument function calls like "ln y", "sin x", "tan x/2" (single variable/number token)
    s = re.sub(rf"\b({fn_names})\s+([a-zA-Z0-9]+)\b(?!\s*\()", r"\1(\2)", s)

    # Insert implicit multiplication carefully while preserving function calls
    # 3x, 3sin(x), 3(x+1) -> 3*...
    s = re.sub(r"(\d)\s*([a-zA-Z(])", r"\1*\2", s)
    # )x, )(, )sin, )2 -> )*...
    s = re.sub(r"\)\s*([a-zA-Z0-9(])", r")*\1", s)
    # x( -> x*(  (only when identifier is not a known function name)
    s = re.sub(rf"\b(?!({fn_names})\b)([a-zA-Z])\s*\(", r"\2*(", s)
    # x sin(...) or y log(...) -> x*sin(...)
    s = re.sub(rf"\b([a-zA-Z])\s+({fn_names})\b", r"\1*\2", s)
    # Space-separated single variables like "x y" or "y C" -> "x*y"
    s = re.sub(rf"\b(?!({fn_names})\b)([a-zA-Z])\s+(?!({fn_names})\b)([a-zA-Z])\b", r"\2*\4", s)
    # Adjacent 2-letter variable pairs like "xy" or "yx" (not known functions/consts)
    s = re.sub(r"\b(xy|yx)\b", lambda m: f"{m.group(1)[0]}*{m.group(1)[1]}", s)

    return s.strip()


def _eval_ast_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        if node.id in SAFE_CONSTS:
            return float(SAFE_CONSTS[node.id])
        raise ValueError(f"Unknown symbol: {node.id}")
    if isinstance(node, ast.UnaryOp):
        val = _eval_ast_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, env)
        right = _eval_ast_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname not in SAFE_FUNCS or len(node.args) != 1 or node.keywords:
            raise ValueError(f"Unsupported function call: {fname}")
        arg = _eval_ast_node(node.args[0], env)
        return float(SAFE_FUNCS[fname](arg))
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def validate_safe_ast(expr_str: str, allowed_vars: set[str]) -> ast.Expression:
    """Parse expr_str and ensure every AST node is strictly safe arithmetic/math."""
    tree = ast.parse(expr_str, mode="eval")
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Expression, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub),
        ):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(node, ast.Name):
            if node.id in allowed_vars or node.id in SAFE_CONSTS or node.id in SAFE_FUNCS:
                continue
            raise ValueError(f"Unknown symbol: {node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in SAFE_FUNCS and len(node.args) == 1 and not node.keywords:
                continue
            raise ValueError(f"Unsupported function call: {node.func.id}")
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")
    return tree


def safe_eval_expr(expr_str: str, var_values: dict[str, float] | None = None) -> float:
    env = var_values or {}
    tree = validate_safe_ast(expr_str, set(env.keys()))
    return _eval_ast_node(tree, env)


def _sympy_check_equal(expr_a: str, expr_b: str, derivative_mode: bool = False, var_name: str = "x") -> bool | None:
    """Return True/False via SymPy if installed, or None if SymPy is unavailable/unparseable."""
    if _sympy is None:
        return None
    try:
        # Gate BOTH expressions through our strict AST allowlist before sympy.sympify (which uses eval!)
        validate_safe_ast(expr_a, {var_name})
        validate_safe_ast(expr_b, {var_name})
        x = _sympy.Symbol(var_name)
        locals_map = {
            var_name: x,
            "pi": _sympy.pi,
            "e": _sympy.E,
            "sin": _sympy.sin,
            "cos": _sympy.cos,
            "tan": _sympy.tan,
            "sec": _sympy.sec,
            "csc": _sympy.csc,
            "cot": _sympy.cot,
            "exp": _sympy.exp,
            "log": _sympy.log,
            "ln": _sympy.log,
            "sqrt": _sympy.sqrt,
            "abs": _sympy.Abs,
        }
        a = _sympy.sympify(expr_a, locals=locals_map)
        b = _sympy.sympify(expr_b, locals=locals_map)
        if derivative_mode:
            return _sympy.simplify(_sympy.diff(a, x) - b) == 0
        return _sympy.simplify(a - b) == 0
    except Exception:
        return None


def extract_raw_math_answer(workspace: Path, final_text: str, allow_tools: bool) -> tuple[str | None, str]:
    if allow_tools:
        ans_file = workspace / "answer.txt"
        if ans_file.exists():
            raw = ans_file.read_text(encoding="utf-8", errors="replace").strip()
            if raw:
                return raw, f"answer.txt={raw!r}"
    boxed = extract_boxed(final_text)
    if boxed is not None:
        return boxed, f"boxed={boxed!r}"
    return None, "no answer.txt or \\boxed{...} found"


def _grade_ode_implicit(task: dict, raw: str, source: str) -> tuple[bool, str, str]:
    """Verify an implicit general ODE solution G(x, y, C) = 0 against dy/dx = ode_rhs(x, y).

    Accepts any algebraically equivalent form and any reparameterization of the constant C
    (e.g., x = y*ln(y) + C/y, xy = y^2*ln|y| + C, y*(x - y*ln(y)) = k, etc.).
    """
    if _sympy is None:
        return False, "harness_error", "SymPy is required for ode_implicit grading (pip install sympy)"

    ode_rhs_str = str(task["ode_rhs"])
    s = raw.strip().replace("$", "")
    # Split on '=' (allow implicit LHS fallback if model omitted 'x =')
    if "=" in s:
        parts = s.split("=")
        if len(parts) != 2:
            return False, "wrong_answer", f"Expected a single equation LHS = RHS ({source})"
        lhs_norm = normalize_math_expr(parts[0], strip_constant_c=False, strip_leading_lhs=False)
        rhs_norm = normalize_math_expr(parts[1], strip_constant_c=False, strip_leading_lhs=False)
        g_str = f"({lhs_norm}) - ({rhs_norm})"
    else:
        default_lhs = str(task.get("default_lhs", "y"))
        rhs_norm = normalize_math_expr(s, strip_constant_c=False, strip_leading_lhs=False)
        g_str = f"({default_lhs}) - ({rhs_norm})"

    const_candidates = {"C", "c", "K", "k", "A"}
    allowed_vars = {"x", "y"} | const_candidates
    try:
        tree = validate_safe_ast(g_str, allowed_vars)
        validate_safe_ast(ode_rhs_str, {"x", "y"})
    except Exception as exc:
        return False, "wrong_answer", f"Could not parse ODE equation {g_str!r} ({source}): {exc}"

    used_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    found_consts = sorted(used_names & const_candidates)
    if len(found_consts) != 1:
        return False, "wrong_answer", f"Expected exactly 1 integration constant (C/c/k), found {found_consts} in {g_str!r} ({source})"
    if "x" not in used_names or "y" not in used_names:
        return False, "wrong_answer", f"Equation must relate both x and y: {g_str!r} ({source})"

    cname = found_consts[0]
    try:
        x_sym = _sympy.Symbol("x", real=True)
        # Positive y when domain is positive (makes abs(y) simplify to y for ln|y|)
        y_positive = bool(task.get("y_positive", True))
        y_sym = _sympy.Symbol("y", positive=True) if y_positive else _sympy.Symbol("y", real=True)
        c_sym = _sympy.Symbol(cname, real=True)
        locals_map = {
            "x": x_sym,
            "y": y_sym,
            cname: c_sym,
            "pi": _sympy.pi,
            "e": _sympy.E,
            "sin": _sympy.sin,
            "cos": _sympy.cos,
            "tan": _sympy.tan,
            "sec": _sympy.sec,
            "csc": _sympy.csc,
            "cot": _sympy.cot,
            "exp": _sympy.exp,
            "log": _sympy.log,
            "ln": _sympy.log,
            "sqrt": _sympy.sqrt,
            "abs": _sympy.Abs,
        }
        G = _sympy.sympify(g_str, locals=locals_map)
        rhs_ode = _sympy.sympify(ode_rhs_str, locals=locals_map)

        dG_dx = _sympy.diff(G, x_sym)
        dG_dy = _sympy.diff(G, y_sym)
        dG_dc = _sympy.diff(G, c_sym)
        if dG_dc == 0 or dG_dy == 0:
            return False, "wrong_answer", f"Degenerate equation (constant or y derivative is zero): {g_str!r} ({source})"

        c_sols = _sympy.solve(G, c_sym)
        if not c_sols:
            return False, "wrong_answer", f"Could not isolate constant {cname} in {g_str!r} ({source})"

        # Implicit slope dy/dx = -(dG/dx)/(dG/dy) evaluated along G(x, y, C) = 0
        raw_slope = -dG_dx / dG_dy
        for c_expr in c_sols:
            slope_xy = raw_slope.subs(c_sym, c_expr)
            # First try exact symbolic simplification
            try:
                if _sympy.simplify(slope_xy - rhs_ode) == 0:
                    return True, "passed", f"ODE implicit derivative matched symbolically ({source})"
            except Exception:
                pass
            # Numeric cross-check across probe pairs (handles tricky trig/log forms)
            ok_pts = 0
            max_err = 0.0
            for xv, yv in ODE_PROBE_PAIRS:
                try:
                    v1 = complex(slope_xy.subs({x_sym: xv, y_sym: yv}).evalf())
                    v2 = complex(rhs_ode.subs({x_sym: xv, y_sym: yv}).evalf())
                    if abs(v1.imag) > 1e-6 or abs(v2.imag) > 1e-6:
                        continue
                    err = abs(v1.real - v2.real)
                    max_err = max(max_err, err)
                    ok_pts += 1
                except Exception:
                    continue
            if ok_pts >= 5 and max_err < 1e-5:
                return True, "passed", f"ODE implicit derivative matched numerically (max_err={max_err:.2e}, {source})"

        return False, "wrong_answer", f"Implicit dy/dx does not satisfy ODE ({source})"
    except Exception as exc:
        return False, "wrong_answer", f"ODE verification error for {g_str!r} ({source}): {exc}"


def grade_math(task: dict, workspace: Path, final_text: str) -> tuple[bool, str, str]:
    """Grade integer, root-set, symbolic expression, or antiderivative math tasks."""
    allow_tools = bool(task.get("allow_tools", True))
    grader = str(task.get("grader", "integer")).lower()
    raw, source = extract_raw_math_answer(workspace, final_text, allow_tools)
    if raw is None:
        return False, "wrong_answer", source

    if grader == "integer":
        expected = str(task["expected"]).strip()
        cleaned = raw.replace(",", "").replace(" ", "")
        m = re.search(r"-?\d+", cleaned)
        if not m:
            return False, "wrong_answer", f"Expected {expected}; could not parse integer from {source}"
        got = m.group(0)
        if got == expected:
            return True, "passed", f"Matched {expected} ({source})"
        return False, "wrong_answer", f"Expected {expected}, got {got} ({source})"

    if grader == "set":
        expected_vals = sorted(float(v) for v in task["expected_set"])
        cleaned = _strip_latex_spacing(raw.strip()).strip()
        # Strip outer matching set/list delimiters only when both ends match
        for open_d, close_d in ((r"\{", r"\}"), ("{", "}"), ("[", "]"), ("(", ")")):
            if cleaned.startswith(open_d) and cleaned.endswith(close_d):
                cleaned = cleaned[len(open_d) : -len(close_d)].strip()
                break
        # Unwrap \text{ or } / \text{ and } before splitting roots
        cleaned = re.sub(r"\\text\{([^}]*)\}", r" \1 ", cleaned)
        parts = re.split(r"\s*(?:,|;|\bor\b|\band\b)\s*", cleaned, flags=re.IGNORECASE)
        got_vals: list[float] = []
        try:
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                expr = normalize_math_expr(p)
                got_vals.append(safe_eval_expr(expr))
        except Exception as exc:
            return False, "wrong_answer", f"Could not parse root set from {source}: {exc}"
        got_vals.sort()
        if len(got_vals) == len(expected_vals) and all(abs(a - b) < 1e-6 for a, b in zip(got_vals, expected_vals)):
            return True, "passed", f"Matched root set {expected_vals} ({source})"
        return False, "wrong_answer", f"Expected set {expected_vals}, got {got_vals} ({source})"

    if grader == "expr":
        var_name = str(task.get("var", "x"))
        expected_expr = str(task["expected_expr"])
        norm = normalize_math_expr(raw, strip_constant_c=False)
        sym_res = _sympy_check_equal(norm, expected_expr, derivative_mode=False, var_name=var_name)
        if sym_res is True:
            return True, "passed", f"SymPy exact match: {norm!r} == {expected_expr!r} ({source})"
        try:
            diffs = [
                abs(safe_eval_expr(norm, {var_name: xv}) - safe_eval_expr(expected_expr, {var_name: xv}))
                for xv in PROBE_POINTS
            ]
        except Exception as exc:
            return False, "wrong_answer", f"Failed to evaluate {norm!r} ({source}): {exc}"
        max_err = max(diffs)
        if max_err < 1e-6:
            return True, "passed", f"Numeric equivalence passed (max_err={max_err:.2e}, parsed={norm!r}, {source})"
        return False, "wrong_answer", f"Expected {expected_expr!r}, parsed {norm!r}, max_err={max_err:.3g} ({source})"

    if grader == "antiderivative":
        var_name = str(task.get("var", "x"))
        integrand = str(task["integrand"])
        expected_expr = task.get("expected_expr")
        norm = normalize_math_expr(raw, strip_constant_c=True)

        sym_res = _sympy_check_equal(norm, integrand, derivative_mode=True, var_name=var_name)
        if sym_res is True:
            return True, "passed", f"SymPy d/d{var_name}({norm}) == {integrand} ({source})"

        try:
            if expected_expr:
                offsets = [
                    safe_eval_expr(norm, {var_name: xv}) - safe_eval_expr(str(expected_expr), {var_name: xv})
                    for xv in PROBE_POINTS
                ]
                spread = max(offsets) - min(offsets)
                if spread < 1e-6:
                    return True, "passed", f"Matches antiderivative up to constant (spread={spread:.2e}, parsed={norm!r}, {source})"
                return False, "wrong_answer", f"Not equivalent up to constant (spread={spread:.3g}, parsed={norm!r}, {source})"
            # Fallback: central finite-difference derivative check against integrand
            h = 1e-6
            errs = []
            for xv in PROBE_POINTS:
                df = (safe_eval_expr(norm, {var_name: xv + h}) - safe_eval_expr(norm, {var_name: xv - h})) / (2 * h)
                target = safe_eval_expr(integrand, {var_name: xv})
                errs.append(abs(df - target))
            max_err = max(errs)
            if max_err < 1e-4:
                return True, "passed", f"Numeric derivative matches integrand (max_err={max_err:.2e}, parsed={norm!r}, {source})"
            return False, "wrong_answer", f"Derivative mismatch (max_err={max_err:.3g}, parsed={norm!r}, {source})"
        except Exception as exc:
            return False, "wrong_answer", f"Failed to evaluate antiderivative {norm!r} ({source}): {exc}"

    if grader == "ode_implicit":
        return _grade_ode_implicit(task, raw, source)

    return False, "harness_error", f"Unknown math grader type: {grader}"
