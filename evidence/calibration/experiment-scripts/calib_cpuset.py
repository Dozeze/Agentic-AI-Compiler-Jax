"""Scratch experiment: rerun calibrate_aa with --cpuset-cpus=<argv[1]> instead of a --cpus quota."""
import runpy, sys
repo = sys.argv[2]
sys.path.insert(0, repo)
import runner
cpuset = sys.argv[1]
_orig = runner.DockerRunner.argv
def argv(self, workdir, container):
    a = _orig(self, workdir, container)
    i = a.index("--cpus")
    a[i:i + 2] = ["--cpuset-cpus", cpuset]
    return a
runner.DockerRunner.argv = argv
sys.argv = [repo + "/scripts/calibrate_aa.py", *sys.argv[3:]]
runpy.run_path(sys.argv[0], run_name="__main__")
