"""Hand-written Pallas kernels (G-11 prototype) pass the same correctness gate as any candidate.

On CPU the kernels run in Pallas interpret mode (CLAUDE.md rule 3). The extra shapes are chosen for
the padding paths (widths that are not powers of two, a row count below one tile, a single row);
they are deliberately not the registry's held-out shapes, which stay unused until AIC-018.
The needs_gpu test checks the native Triton lowering and correctness on CUDA; it makes no speed claim.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from backends import REPO, gpu_env
from benchmark import build_cases, check_correctness, load_function
from workloads.registry import LAYER_NORM, SOFTMAX

KERNELS = {SOFTMAX.workload_id: (SOFTMAX, REPO / "kernels" / "softmax_pallas.py"),
           LAYER_NORM.workload_id: (LAYER_NORM, REPO / "kernels" / "layer_norm_pallas.py")}
EXTRA_SHAPES = ((48, 200), (7, 1000), (3, 5), (1, 4096))


def _cases(spec):
    assert not set(EXTRA_SHAPES) & set(spec.heldout_shapes), "extra shapes must not touch the held-out split"
    return build_cases(spec) + [(f"extra{shape}", spec.make_inputs(0, shape)) for shape in EXTRA_SHAPES]


@pytest.mark.parametrize("workload", sorted(KERNELS))
def test_pallas_kernel_passes_correctness_gate_on_cpu(workload):
    spec, path = KERNELS[workload]
    candidate = load_function(path.read_text(encoding="utf-8"), spec.function_name, str(path))
    report = check_correctness(candidate, spec, _cases(spec))
    assert report.passed, [case for case in report.to_json()["cases"] if not case["passed"]]


@pytest.mark.parametrize("workload", sorted(KERNELS))
def test_pallas_kernel_keeps_the_reference_signature(workload):
    spec, path = KERNELS[workload]
    candidate = load_function(path.read_text(encoding="utf-8"), spec.function_name, str(path))
    args = spec.make_inputs(0, spec.development_shapes[0])
    assert candidate(*args).shape == args[0].shape


_GPU_DRIVER = """
import sys
sys.path[:0] = [{repo!r}, {tests!r}]
import jax
from environment import select_backend
select_backend("cuda")
from benchmark import load_function, check_correctness
from test_pallas_kernels import KERNELS, _cases
for workload, (spec, path) in sorted(KERNELS.items()):
    fn = load_function(path.read_text(encoding="utf-8"), spec.function_name, str(path))
    args = spec.make_inputs(0, spec.development_shapes[-1])
    hlo = jax.jit(fn).lower(*args).compile().as_text()
    native = "triton" in hlo.lower()
    report = check_correctness(fn, spec, _cases(spec))
    print(workload, "native" if native else "NOT-NATIVE", "passed" if report.passed else "FAILED")
"""


@pytest.mark.needs_gpu
def test_pallas_kernels_lower_natively_and_pass_on_cuda():
    driver = _GPU_DRIVER.format(repo=str(REPO), tests=str(Path(__file__).resolve().parent))
    proc = subprocess.run([sys.executable, "-c", driver], cwd=REPO, env=gpu_env(),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = dict(line.split(" ", 1) for line in proc.stdout.strip().splitlines())
    assert lines == {workload: "native passed" for workload in KERNELS}, proc.stdout
