"""Pickle-free array exchange (AIC-008)."""
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

import array_io
from array_io import ArrayIOError, FlatTree, flatten


def test_roundtrip_keeps_structure_dtypes_and_values(tmp_path):
    trees = {"case/a": flatten((np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(4, np.int32))),
             "case/b": flatten({"y": jnp.ones((2,), jnp.bfloat16), "aux": np.zeros((), np.float32)})}
    array_io.save(tmp_path / "x.npz", trees)
    loaded = array_io.load(tmp_path / "x.npz", base=tmp_path, max_bytes=1 << 20)
    assert set(loaded) == set(trees)
    for name, tree in trees.items():
        assert loaded[name].treedef == tree.treedef
        for (p1, a1), (p2, a2) in zip(loaded[name].leaves, tree.leaves):
            assert p1 == p2 and a1.dtype == a2.dtype and a1.shape == a2.shape
            assert np.array_equal(a1.astype(np.float64), a2.astype(np.float64))
    assert str(loaded["case/b"].leaves[1][1].dtype) == "bfloat16"


def _saved(tmp_path):
    array_io.save(tmp_path / "x.npz", {"t": flatten((np.ones(3, np.float32),))})
    return tmp_path / "x.npz"


def test_size_limit(tmp_path):
    with pytest.raises(ArrayIOError, match="limit"):
        array_io.load(_saved(tmp_path), base=tmp_path, max_bytes=10)


def test_symlink_and_outside_paths_refused(tmp_path):
    path = _saved(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    os.symlink(path, work / "x.npz")
    with pytest.raises(ArrayIOError, match="symlinks"):
        array_io.load(work / "x.npz", base=work, max_bytes=1 << 20)
    with pytest.raises(ArrayIOError, match="outside"):
        array_io.load(path, base=work, max_bytes=1 << 20)


def test_undeclared_array_refused(tmp_path):
    path = _saved(tmp_path)
    np.savez(path, **{"000000": np.ones(3, np.float32), "extra": np.ones(1)})
    with pytest.raises(ArrayIOError, match="index declares"):
        array_io.load(path, base=tmp_path, max_bytes=1 << 20)


def test_changed_dtype_refused(tmp_path):
    path = _saved(tmp_path)
    np.savez(path, **{"000000": np.ones(3, np.float64)})
    with pytest.raises(ArrayIOError, match="dtype"):
        array_io.load(path, base=tmp_path, max_bytes=1 << 20)


def test_pickled_objects_refused(tmp_path):
    path = _saved(tmp_path)
    np.savez(path, **{"000000": np.array([{"evil": 1}], dtype=object)})
    with pytest.raises(ArrayIOError, match="unreadable|dtype"):
        array_io.load(path, base=tmp_path, max_bytes=1 << 20)


def test_bad_index_refused(tmp_path):
    path = _saved(tmp_path)
    (tmp_path / "x.npz.json").write_text(json.dumps({"schema_version": "other"}))
    with pytest.raises(ArrayIOError, match="index"):
        array_io.load(path, base=tmp_path, max_bytes=1 << 20)


def test_flat_tree_arrays():
    tree = flatten((np.zeros(1), np.ones(2)))
    assert isinstance(tree, FlatTree) and [a.shape for a in tree.arrays] == [(1,), (2,)]
