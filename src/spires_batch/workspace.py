"""Content-addressed handles to on-disk artifacts.

A handle is ``<kind>-<sha256(json(stage, sorted params, parents))[:12]>``, so
asking for the same stage on the same parents with the same parameters names
the same artifact and a re-run skips completed units instead of redoing them.
Lifted from spipyservice_pipeline/workspace.py; kinds reduced to ``retrieval``
(a netCDF Dataset) and ``benchmark`` (a JSON document). Standard library at
module top; xarray is imported where needed.
"""
import datetime
import hashlib
import json
import os
import pathlib
import shutil

KINDS = ("retrieval", "benchmark")


class UnknownHandle(KeyError):
    pass


def root() -> pathlib.Path:
    configured = os.environ.get("SPIRES_BATCH_WORKSPACE")
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path.home() / ".cache" / "spires-batch" / "workspace"


def _dir(handle) -> pathlib.Path:
    return root() / "handles" / handle


def _kind(handle) -> str:
    return handle.split("-", 1)[0]


def new_handle(kind, stage, params, parents) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {KINDS}")
    payload = json.dumps(
        {"stage": stage, "params": params, "parents": list(parents)}, sort_keys=True, default=str
    )
    return f"{kind}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def exists(handle) -> bool:
    return (_dir(handle) / "meta.json").is_file()


def meta(handle) -> dict:
    meta_path = _dir(handle) / "meta.json"
    if not meta_path.is_file():
        raise UnknownHandle(handle)
    return json.loads(meta_path.read_text())


def list_handles(kind=None) -> list[dict]:
    base = root() / "handles"
    if not base.is_dir():
        return []
    found = []
    for entry in base.iterdir():
        if entry.name.startswith(".") or not (entry / "meta.json").is_file():
            continue
        record = json.loads((entry / "meta.json").read_text())
        if kind is None or record["kind"] == kind:
            found.append(record)
    return sorted(found, key=lambda record: (record["created_utc"], record["handle"]))


def forget(handle) -> None:
    if not exists(handle):
        raise UnknownHandle(handle)
    shutil.rmtree(_dir(handle))


def _netcdf_safe_attrs(obj):
    """netCDF attrs must be scalars/strings/1-D numeric; JSON-encode anything else."""
    import numpy as np

    def fix(attrs):
        out = {}
        for key, value in attrs.items():
            if value is None:
                out[key] = ""
            elif isinstance(value, (str, int, float, np.integer, np.floating, np.ndarray)):
                out[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
                out[key] = list(value)
            else:
                out[key] = json.dumps(value, default=str)
        return out

    obj = obj.copy(deep=False)
    obj.attrs = fix(obj.attrs)
    for name in list(obj.data_vars) + list(obj.coords):
        obj[name].attrs = fix(obj[name].attrs)
    return obj


def _stats(obj) -> dict:
    return {
        "shape": {str(k): int(v) for k, v in obj.sizes.items()},
        "dtype": {str(k): str(v.dtype) for k, v in obj.data_vars.items()},
    }


def store(handle, obj, meta_in) -> None:
    """Write ``obj`` under ``handle`` atomically; no-op if the handle already exists."""
    kind = _kind(handle)
    final = _dir(handle)
    if (final / "meta.json").is_file():
        return
    tmp = root() / "handles" / f".tmp-{handle}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    record = dict(meta_in)
    record.update(
        handle=handle,
        kind=kind,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    if kind == "retrieval":
        import xarray

        if not isinstance(obj, xarray.Dataset):
            raise TypeError(f"retrieval artifacts must be xarray.Dataset, got {type(obj).__name__}")
        record["stats"] = _stats(obj)
        _netcdf_safe_attrs(obj).to_netcdf(tmp / "data.nc", engine="h5netcdf")
    else:
        (tmp / "data.json").write_text(json.dumps(obj, indent=2, default=str))
    (tmp / "meta.json").write_text(json.dumps(record, indent=2, default=str))
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)


def load(handle):
    record = meta(handle)
    directory = _dir(handle)
    if record["kind"] == "benchmark":
        return json.loads((directory / "data.json").read_text())
    import xarray

    return xarray.load_dataset(directory / "data.nc", engine="h5netcdf")


def path(handle) -> pathlib.Path:
    """On-disk file of a stored artifact (for `ncdump`, QGIS, ...)."""
    record = meta(handle)
    return _dir(handle) / ("data.json" if record["kind"] == "benchmark" else "data.nc")
