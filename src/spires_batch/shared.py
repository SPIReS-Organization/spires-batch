"""Scatter shared inputs (LUT, R0, masks) once; small values ride along as kwargs."""
import hashlib
import pickle

import numpy as np
import xarray as xr

#: anything below this is passed by value to every task rather than scattered.
SCATTER_THRESHOLD_BYTES = 1 << 20


class SharedInputLost(RuntimeError):
    """Every worker holding a scattered shared input died; it cannot be recomputed."""


def _is_lazy(value) -> bool:
    if isinstance(value, xr.DataArray):
        return value.chunks is not None
    if isinstance(value, xr.Dataset):
        return any(value[n].chunks is not None for n in value.data_vars)
    return False


def check_shared(shared: dict) -> None:
    """Raise before the cluster is touched if a value cannot travel to workers."""
    for name, value in shared.items():
        if _is_lazy(value):
            raise TypeError(f"shared[{name!r}] is dask-backed; load it into memory before scattering")
        if isinstance(value, (np.ndarray, xr.DataArray, xr.Dataset)):
            continue  # always picklable; serializing a multi-GB R0 just to probe it is waste
        try:
            pickle.dumps(value)
        except Exception as exc:
            raise TypeError(f"shared[{name!r}] is not picklable ({type(value).__name__}: {exc}); "
                            "lambdas and open file handles cannot be shared") from exc


def _nbytes(value) -> int:
    if isinstance(value, (xr.DataArray, xr.Dataset, np.ndarray)):
        return int(value.nbytes)
    return 0


def scatter_shared(client, shared: dict, replicas: int = 2) -> dict:
    """Return ``shared`` with every large array value replaced by a Future resident on ``replicas`` workers.

    ``broadcast=False`` on purpose: broadcasting puts every block on every worker
    (eight copies of a multi-GB R0). ``hash=False`` so keys are random and two
    runs with identical content cannot race on release/re-register.
    """
    check_shared(shared)
    large = {k: v for k, v in shared.items() if _nbytes(v) >= SCATTER_THRESHOLD_BYTES}
    out = dict(shared)
    if large:
        names = list(large)
        futures = client.scatter([large[n] for n in names], direct=True, hash=False)
        if replicas > 1 and len(client.scheduler_info()["workers"]) > 1:
            client.replicate(futures, n=replicas)
        out.update(zip(names, futures))
    return out


def _hash_array(h, arr) -> None:
    """Feed dtype, shape and raw bytes so equal bytes with different layout hash differently."""
    arr = np.asarray(arr)
    if arr.dtype.kind == "O":
        # object arrays serialize to pointer addresses; hash their text instead
        arr = arr.astype(str)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def digests(shared: dict) -> dict[str, str]:
    """Short, content-sensitive identifiers for the workspace handle."""
    out = {}
    for name, value in shared.items():
        if isinstance(value, xr.Dataset):
            h = hashlib.sha256()
            for var in sorted(value.data_vars):
                h.update(str(value[var].dims).encode())
                _hash_array(h, value[var].values)
            for coord in sorted(value.coords):
                _hash_array(h, value[coord].values)
            out[name] = h.hexdigest()[:12]
        elif isinstance(value, xr.DataArray):
            h = hashlib.sha256()
            h.update(str(value.dims).encode())
            _hash_array(h, value.values)
            for coord in sorted(value.coords):
                _hash_array(h, value.coords[coord].values)
            out[name] = h.hexdigest()[:12]
        elif isinstance(value, np.ndarray):
            h = hashlib.sha256()
            _hash_array(h, value)
            out[name] = h.hexdigest()[:12]
        elif hasattr(value, "sensor"):          # a SensorSpec
            out[name] = str(value.sensor)
        else:
            out[name] = str(value)
    return out
