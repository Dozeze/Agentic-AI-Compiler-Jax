"""Source policy (AIC-007): what a model-written candidate may contain."""
import pytest

from fixtures.softmax_candidates import FIXTURES
from source_policy import check_source
from workloads.registry import REGISTRY

LIMIT = 64 << 10


def violations(source, name="f"):
    return check_source(source, name, max_source_bytes=LIMIT).violations


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_correctness_fixture_is_expressible(name):
    """The policy restricts what code can do, not whether it is correct: all 18 fixtures pass it."""
    assert check_source(FIXTURES[name], "softmax", max_source_bytes=LIMIT).ok, name


@pytest.mark.parametrize("workload", sorted(REGISTRY))
def test_registered_references_pass(workload):
    spec = REGISTRY[workload]
    assert check_source(spec.reference_source, spec.function_name, max_source_bytes=LIMIT).ok


def test_typical_rewrite_passes():
    source = ('"""Batched heads."""\nimport functools\nimport math\nimport jax\nimport jax.numpy as jnp\n'
              'from jax import lax\n\nSCALE = 1.0 / math.sqrt(64)\n\n'
              '@functools.partial(jax.jit, static_argnums=1)\ndef f(x, n):\n'
              '    y = lax.dot_general(x, x, (((1,), (0,)), ((), ())))\n'
              '    return jnp.where(y > 0, y * SCALE, jax.nn.relu(y))\n')
    assert violations(source) == ()


@pytest.mark.parametrize("snippet, reason", [
    ("import os\ndef f(x):\n    return x", "import of 'os'"),
    ("import numpy as np\ndef f(x):\n    return x", "import of 'numpy'"),
    ("def f(x):\n    import subprocess\n    return x", "import of 'subprocess'"),
    ("from jax import config\ndef f(x):\n    return x", "import of 'jax.config'"),
    ("from jax.experimental import io_callback\ndef f(x):\n    return x", "import from 'jax.experimental'"),
    ("from . import x\ndef f(x):\n    return x", "import from '.'"),
    ("from jax.numpy import *\ndef f(x):\n    return x", "import of 'jax.numpy.*'"),
    ("def f(x):\n    return eval('x')", "use of 'eval'"),
    ("def f(x):\n    return open('/etc/passwd').read()", "use of 'open'"),
    ("def f(x):\n    return getattr(x, 'shape')", "use of 'getattr'"),
    ("def f(x):\n    return x.__class__", "attribute '__class__'"),
    ("def f(x):\n    return __builtins__", "dunder name '__builtins__'"),
    ("import jax\ndef f(x):\n    return jax._src", "attribute '_src'"),
    ("import jax\ndef f(x):\n    jax.debug.print('{}', x)\n    return x", "attribute 'debug'"),
    ("import jax\ndef f(x):\n    return jax.pure_callback(g, x, x)", "attribute 'pure_callback'"),
    ("import jax\ndef f(x):\n    return jax.device_put(x)", "attribute 'device_put'"),
    ("import jax\ndef f(x):\n    jax.config.update('jax_enable_x64', True)\n    return x", "attribute 'config'"),
    ("import jax.numpy as jnp\ndef f(x):\n    return jnp.load('w.npy')", "attribute 'load'"),
    ("import jax\nf = jax.jit(lambda x: x, device=None)", "keyword device="),
    ("import jax.numpy as jnp\njnp.exp = abs\ndef f(x):\n    return x", "assigning or deleting an attribute"),
    ("def f(x):\n    global y\n    return x", "Global is not allowed"),
    ("class A:\n    pass\ndef f(x):\n    return x", "ClassDef is not allowed"),
    ("print('hi')\ndef f(x):\n    return x", "module-level Expr"),
    ("for i in range(3):\n    pass\ndef f(x):\n    return x", "module-level For"),
    ("def g(x):\n    return x", "no top-level function named 'f'"),
    ("def f(x) return x", "syntax error"),
])
def test_forbidden_constructs_are_rejected(snippet, reason):
    found = violations(snippet)
    assert any(reason in v for v in found), found


def test_size_limit():
    result = check_source("def f(x):\n    return x\n" + "#" * 100, "f", max_source_bytes=50)
    assert not result.ok and "larger than 50 bytes" in result.violations[0]


def test_summary_is_bounded():
    source = "\n".join(f"import os{i}" for i in range(10)) + "\ndef f(x):\n    return x"
    assert "and 5 more" in check_source(source, "f", max_source_bytes=LIMIT).summary()
