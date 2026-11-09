"""Pickle-free, bounded array exchange between the host and a worker process (AIC-008).

Arrays travel as one `.npz` file (loaded with allow_pickle=False) plus a JSON index that records each
tree's structure and every leaf's path, dtype and shape. The index is the contract: on load, the file
must hold exactly the declared leaves with exactly the declared dtypes and shapes, so a worker cannot
smuggle in objects, extra arrays or silently changed precision. Paths are checked too: no symlinks,
nothing outside the directory the caller expects, nothing above the size limit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_SCHEMA = "array_index/v1"


class ArrayIOError(ValueError):
    """An array file or its index is missing, oversized, outside the work directory or inconsistent."""


@dataclass(frozen=True)
class FlatTree:
    """A PyTree as data: its structure string and its leaves in flattening order."""
    treedef: str                                   # str(jax.tree_util PyTreeDef), e.g. "PyTreeDef((*, *))"
    leaves: tuple[tuple[str, np.ndarray], ...]     # (keystr path, array)

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(array for _, array in self.leaves)


def flatten(tree) -> FlatTree:
    import jax
    pairs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    return FlatTree(str(treedef), tuple((jax.tree_util.keystr(p), np.asarray(leaf)) for p, leaf in pairs))


def _dtype(name: str) -> np.dtype:
    try:
        return np.dtype(name)
    except TypeError:                              # bfloat16 and friends are registered by ml_dtypes/jax
        import jax.numpy as jnp
        return np.dtype(jnp.dtype(name))


def save(path: Path, trees: dict[str, FlatTree]) -> None:
    """Write `trees` to `path` (.npz) and its index next to it (`path` + '.json')."""
    path = Path(path)
    arrays, index = {}, {"schema_version": INDEX_SCHEMA, "trees": {}}
    for name, tree in trees.items():
        entries = []
        for i, (leaf_path, array) in enumerate(tree.leaves):
            key = f"{len(arrays):06d}"
            arrays[key] = np.asarray(array, order="C")        # keeps 0-d arrays 0-d
            entries.append({"key": key, "path": leaf_path, "dtype": str(array.dtype), "shape": list(array.shape)})
        index["trees"][name] = {"treedef": tree.treedef, "leaves": entries}
    np.savez(path, **arrays)
    _index_path(path).write_text(json.dumps(index, allow_nan=False), encoding="utf-8")


def _index_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def checked_path(path: Path, base: Path, max_bytes: int) -> Path:
    """`path` if it is a regular file inside `base`, not a symlink, and at most `max_bytes`."""
    path, base = Path(path), Path(base).resolve()
    if path.is_symlink():
        raise ArrayIOError(f"{path.name}: symlinks are not accepted")
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ArrayIOError(f"{path}: outside the work directory")
    if not resolved.is_file():
        raise ArrayIOError(f"{path.name}: missing")
    if resolved.stat().st_size > max_bytes:
        raise ArrayIOError(f"{path.name}: {resolved.stat().st_size} bytes > limit {max_bytes}")
    return resolved


def load(path: Path, *, base: Path, max_bytes: int) -> dict[str, FlatTree]:
    """Load and validate trees written by `save`. Raises ArrayIOError on any inconsistency."""
    path = checked_path(path, base, max_bytes)
    index_file = checked_path(_index_path(path), base, 16 << 20)
    try:
        index = json.loads(index_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArrayIOError(f"{index_file.name}: not valid JSON") from error
    if index.get("schema_version") != INDEX_SCHEMA:
        raise ArrayIOError(f"{index_file.name}: not an {INDEX_SCHEMA} index")
    try:
        npz = np.load(path, allow_pickle=False)
    except Exception as error:   # noqa: BLE001 - any parse failure is a bad file
        raise ArrayIOError(f"{path.name}: unreadable ({type(error).__name__})") from error
    with npz:
        declared = {e["key"] for t in index["trees"].values() for e in t["leaves"]}
        if set(npz.files) != declared:
            raise ArrayIOError(f"{path.name}: holds {len(npz.files)} arrays, index declares {len(declared)}")
        trees = {}
        for name, tree in index["trees"].items():
            leaves = []
            for entry in tree["leaves"]:
                try:
                    array = npz[entry["key"]]                # object arrays fail here: no pickle
                except ValueError as error:
                    raise ArrayIOError(f"{name}{entry['path']}: unreadable ({error})") from error
                dtype = _dtype(entry["dtype"])
                if array.dtype != dtype:
                    if array.dtype.kind != "V" or array.dtype.itemsize != dtype.itemsize:
                        raise ArrayIOError(f"{name}{entry['path']}: dtype {array.dtype} != declared {dtype}")
                    array = array.view(dtype)          # void-stored extension dtypes such as bfloat16
                if list(array.shape) != entry["shape"]:
                    raise ArrayIOError(f"{name}{entry['path']}: shape {array.shape} != declared {entry['shape']}")
                leaves.append((entry["path"], array))
            trees[name] = FlatTree(tree["treedef"], tuple(leaves))
    return trees
