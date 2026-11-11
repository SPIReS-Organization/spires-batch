import numpy as np
import pytest
import xarray as xr
from distributed import Future
from spires_io.hls import L30_SPEC

from spires_batch import shared


def test_check_shared_rejects_unpicklable_and_lazy():
    with pytest.raises(TypeError, match="lambda"):
        shared.check_shared({"f": lambda x: x})
    lazy = xr.DataArray(np.zeros((4, 4)), dims=("y", "x")).chunk({"y": 2})
    with pytest.raises(TypeError, match="dask-backed"):
        shared.check_shared({"r0": lazy})


def test_check_shared_accepts_arrays_specs_and_scalars():
    shared.check_shared({"lut": xr.Dataset({"a": ("x", np.zeros(3))}),
                         "r0": xr.DataArray(np.zeros((2, 2)), dims=("y", "x")),
                         "spec": L30_SPEC, "bbox": (0.0, 0.0, 1.0, 1.0), "max_eval": 50})


def test_scatter_shared_scatters_large_and_passes_small(client):
    values = {"r0": xr.DataArray(np.zeros((512, 512, 2), np.float32), dims=("y", "x", "band")),
              "spec": L30_SPEC, "max_eval": 50}
    out = shared.scatter_shared(client, values, replicas=2)
    assert isinstance(out["r0"], Future)
    assert out["spec"] is L30_SPEC and out["max_eval"] == 50
    who = client.who_has(out["r0"])
    assert len(who[out["r0"].key]) == 2


def test_digests_are_stable_and_content_sensitive():
    a = xr.DataArray(np.ones((2, 2), np.float32), dims=("y", "x"))
    b = xr.DataArray(np.ones((2, 2), np.float32), dims=("y", "x"))
    c = xr.DataArray(np.zeros((2, 2), np.float32), dims=("y", "x"))
    d = shared.digests({"r0": a, "spec": L30_SPEC, "n": 3})
    assert d == shared.digests({"r0": b, "spec": L30_SPEC, "n": 3})
    assert d["r0"] != shared.digests({"r0": c})["r0"]
    assert d["spec"] == "hls_l30" and d["n"] == "3"
