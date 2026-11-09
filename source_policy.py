"""Source policy for model-written candidates (AIC-007, AIC-1.1 §7).

A candidate may only be a pure JAX definition: imports from a small allowlist, function definitions,
plain assignments and a docstring. Everything that could reach the host, the harness or the clock is
rejected before the candidate is ever run: other imports, dunder or private attributes, dynamic code
(eval, exec, getattr, ...), file, process and network access, callbacks into Python, device or config
control, and module-level actions or mutation.

An AST check is not a sandbox: tracing a JAX function executes Python. The policy narrows what the
code can express; the container (AIC-009) contains what it still does.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_MODULES = frozenset({
    "jax", "jax.numpy", "jax.lax", "jax.nn", "jax.scipy", "jax.scipy.special", "jax.scipy.linalg",
    "math", "functools",
})

# Builtins that give dynamic access to code, objects, the filesystem or the interpreter.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr", "breakpoint", "help", "exit", "quit", "memoryview",
    "classmethod", "staticmethod", "property", "super", "type", "object",
})

# Attributes that reach Python callbacks, devices, configuration, profiling or files.
FORBIDDEN_ATTRIBUTES = frozenset({
    "config", "profiler", "debug", "experimental", "extend", "distributed", "lib", "interpreters",
    "pure_callback", "io_callback", "callback", "host_callback",
    "devices", "local_devices", "default_device", "device_put", "device_get", "clear_caches",
    "effects_barrier", "live_arrays", "make_array_from_callback",
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt", "savetxt",
    "genfromtxt", "memmap",
})

_FORBIDDEN_NODES = (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield,
                    ast.YieldFrom, ast.AsyncFor, ast.AsyncWith)
_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)


@dataclass(frozen=True)
class PolicyResult:
    ok: bool
    violations: tuple[str, ...]

    def summary(self, limit: int = 5) -> str:
        shown = "; ".join(self.violations[:limit])
        more = len(self.violations) - limit
        return shown + (f"; and {more} more" if more > 0 else "")


def check_source(source: str, function_name: str, *, max_source_bytes: int) -> PolicyResult:
    """Return every policy violation in `source`, each with its line number."""
    if len(source.encode("utf-8")) > max_source_bytes:
        return PolicyResult(False, (f"source is larger than {max_source_bytes} bytes",))
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return PolicyResult(False, (f"line {error.lineno}: syntax error: {error.msg}",))

    found: list[str] = []

    def bad(node, message):
        found.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    for index, node in enumerate(tree.body):
        is_docstring = index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
            and isinstance(node.value.value, str)
        if not (isinstance(node, _TOP_LEVEL) or is_docstring):
            bad(node, f"module-level {type(node).__name__} is not allowed (only imports, defs, assignments)")
    if not any(isinstance(n, ast.FunctionDef) and n.name == function_name for n in tree.body):
        found.append(f"line 1: no top-level function named {function_name!r}")

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            bad(node, f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    bad(node, f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module not in ALLOWED_MODULES:
                bad(node, f"import from {module or '.'!r} is not allowed")
            for alias in node.names:
                full = f"{module}.{alias.name}"
                if alias.name == "*" or (alias.name in FORBIDDEN_ATTRIBUTES and full not in ALLOWED_MODULES) \
                        or alias.name.startswith("_"):
                    bad(node, f"import of {full!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                bad(node, f"use of {node.id!r} is not allowed")
            elif node.id.startswith("__"):
                bad(node, f"dunder name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                bad(node, f"private or dunder attribute {node.attr!r} is not allowed")
            elif node.attr in FORBIDDEN_ATTRIBUTES:
                bad(node, f"attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("__"):
                bad(node, f"dunder function name {node.name!r} is not allowed")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Attribute):
                        bad(node, "assigning or deleting an attribute is not allowed (module or object mutation)")
        elif isinstance(node, ast.keyword) and node.arg in ("device", "backend"):
            bad(node, f"keyword {node.arg}= is not allowed (the harness selects the device)")
    return PolicyResult(not found, tuple(sorted(set(found), key=found.index)))
