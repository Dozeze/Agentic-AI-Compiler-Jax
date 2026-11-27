"""Scratch experiment: rerun calibrate_aa with a different DockerRunner CPU quota (argv[1])."""
import functools, runpy, sys
repo = sys.argv[2]
sys.path.insert(0, repo)
import runner
cpus = float(sys.argv[1])
runner.DockerRunner.__init__ = functools.partialmethod(runner.DockerRunner.__init__, cpus=cpus)
sys.argv = [repo + "/scripts/calibrate_aa.py", *sys.argv[3:]]
runpy.run_path(sys.argv[0], run_name="__main__")
