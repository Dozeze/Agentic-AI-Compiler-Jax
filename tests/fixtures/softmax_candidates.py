"""Softmax candidate fixtures for the correctness matrix (AIC-004).

Each fixture is a source string defining `softmax`, exactly like a real agent candidate, so the same
fixtures can later be fed through the isolated runner (AIC-007). EXPECTED maps each fixture to
(passed, category of the first failing case).
"""
import numpy as np

from workloads.registry import SOFTMAX

HEADER = "import jax\nimport jax.numpy as jnp\n\n"
_STABLE = """    m = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - m)
"""


def _src(body: str) -> str:
    return HEADER + "def softmax(x):\n" + body


# x[0, 0] of seed 0 at shape (32, 128), taken from the registry when this module is imported.
# standard_normal fills in C order, so every shape with seed 0 shares this first element.
X00 = str(np.float32(SOFTMAX.make_inputs(0, (32, 128))[0][0, 0]))

REFERENCE = SOFTMAX.reference_source
JAX_NN = _src("    return jax.nn.softmax(x, axis=-1)\n")
RECIPROCAL_MUL = _src(_STABLE + "    return e * (1.0 / jnp.sum(e, axis=-1, keepdims=True))\n")
LOGSUMEXP = _src("    from jax.scipy.special import logsumexp\n"
                 "    return jnp.exp(x - logsumexp(x, axis=-1, keepdims=True))\n")
_CORRECT = _STABLE + "    out = e / jnp.sum(e, axis=-1, keepdims=True)\n"
NUMERIC_TINY_OK = _src(_CORRECT + "    return out * (1 + 1e-7)\n")
WRONG_TREE = _src(_CORRECT + "    return (out,)\n")
EXTRA_LEAF = _src(_CORRECT + '    return {"y": out, "aux": x}\n')
WRONG_SHAPE = _src(_CORRECT + "    return out[:, :-1]\n")
WRONG_DTYPE_BF16 = _src(_CORRECT + "    return out.astype(jnp.bfloat16)\n")
WRONG_DTYPE_INT = _src(_CORRECT + "    return out.astype(jnp.int32)\n")
WRONG_AXIS = _src("    m = jnp.max(x, axis=0, keepdims=True)\n"
                  "    e = jnp.exp(x - m)\n"
                  "    return e / jnp.sum(e, axis=0, keepdims=True)\n")
NUMERIC_OFFSET = _src(_CORRECT + "    return out + 1e-3\n")
NAIVE_UNSTABLE = _src("    e = jnp.exp(x)\n    return e / jnp.sum(e, axis=-1, keepdims=True)\n")
INJECT_NAN = _src(_CORRECT + "    return out.at[0, 0].set(jnp.nan)\n")
OVERFIT_SHAPE = _src(_CORRECT + "    if x.shape == (32, 128):\n        return out\n    return jnp.zeros_like(x)\n")
OVERFIT_VALUE = _src(_CORRECT +
                     "    uniform = jnp.full_like(x, 1.0 / x.shape[-1])\n"
                     f"    return jnp.where(x[0, 0] == {X00}, out, uniform)\n")
COMPILE_ERROR = _src("    return jnp.matmul(x, x)\n")
TRACE_ERROR = _src('    raise ValueError("boom")\n')

EXPECTED = {
    "REFERENCE": (True, None),
    "JAX_NN": (True, None),
    "RECIPROCAL_MUL": (True, None),
    "LOGSUMEXP": (True, None),                    # admitted at the default OFFSET = 200 (P5)
    "NUMERIC_TINY_OK": (True, None),              # the tolerance is not zero-tolerance
    "WRONG_TREE": (False, "STRUCTURE_MISMATCH"),
    "EXTRA_LEAF": (False, "STRUCTURE_MISMATCH"),
    "WRONG_SHAPE": (False, "STRUCTURE_MISMATCH"),
    "WRONG_DTYPE_BF16": (False, "STRUCTURE_MISMATCH"),
    "WRONG_DTYPE_INT": (False, "STRUCTURE_MISMATCH"),
    "WRONG_AXIS": (False, "NUMERIC_MISMATCH"),    # right shape is not right answer
    "NUMERIC_OFFSET": (False, "NUMERIC_MISMATCH"),
    "NAIVE_UNSTABLE": (False, "NONFINITE"),       # development cases pass; edge cases overflow
    "INJECT_NAN": (False, "NONFINITE"),
    "OVERFIT_SHAPE": (False, "NUMERIC_MISMATCH"),
    "OVERFIT_VALUE": (False, "NUMERIC_MISMATCH"), # only seed 0 passes
    "COMPILE_ERROR": (False, "COMPILE_FAILED"),
    "TRACE_ERROR": (False, "COMPILE_FAILED"),
}
FIXTURES = {name: globals()[name] for name in EXPECTED}
