import os

# MUST happen BEFORE importing bad_script or Functester
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from bad_script import bad_func, X_test, Y_test
from Functester import timeit_version_1

timeit_version_1(bad_func, X_test, Y_test)

