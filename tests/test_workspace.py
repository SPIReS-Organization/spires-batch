import json
import numpy as np
import pytest
import xarray as xr

from spires_batch import workspace


@pytest.fixture(autouse=True)
def ws_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIRES_BATCH_WORKSPACE", str(tmp_path / "ws"))
    return tmp_path / "ws"


def _ds():
    return xr.Dataset({"fsnow": (("y", "x"), np.ones((2, 3), np.float32))},
                      coords={"y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
                      attrs={"note": "x", "nested": {"a": 1}})


def test_handle_is_deterministic_and_order_insensitive():
    a = workspace.new_handle("retrieval", "invert_scene", {"b": 2, "a": 1}, ["p1"])
    b = workspace.new_handle("retrieval", "invert_scene", {"a": 1, "b": 2}, ["p1"])
    c = workspace.new_handle("retrieval", "invert_scene", {"a": 1, "b": 3}, ["p1"])
    assert a == b and a != c and a.startswith("retrieval-") and len(a) == len("retrieval-") + 12


def test_store_and_load_retrieval_roundtrip(ws_root):
    h = workspace.new_handle("retrieval", "s", {}, [])
    assert not workspace.exists(h)
    workspace.store(h, _ds(), {"unit": "scene-1"})
    assert workspace.exists(h)
    back = workspace.load(h)
    np.testing.assert_array_equal(back["fsnow"].values, 1.0)
    assert back.attrs["nested"] == '{"a": 1}'   # non-scalar attrs are JSON-encoded for netCDF
    assert workspace.meta(h)["unit"] == "scene-1"
    assert (ws_root / "handles" / h / "data.nc").is_file()


def test_store_skips_if_exists(ws_root):
    h = workspace.new_handle("retrieval", "s", {}, [])
    workspace.store(h, _ds(), {"n": 1})
    workspace.store(h, _ds().assign(fsnow=lambda d: d.fsnow * 0), {"n": 2})
    assert workspace.meta(h)["n"] == 1
    np.testing.assert_array_equal(workspace.load(h)["fsnow"].values, 1.0)


def test_store_is_atomic_no_tmp_left(ws_root):
    h = workspace.new_handle("retrieval", "s", {}, [])
    workspace.store(h, _ds(), {})
    assert not any(p.name.startswith(".tmp-") for p in (ws_root / "handles").iterdir())


def test_bool_attrs_are_stored_as_int(ws_root):
    h = workspace.new_handle("retrieval", "s", {}, [])
    ds = _ds().assign_attrs(flag=True, nflag=np.bool_(False))
    workspace.store(h, ds, {})
    back = workspace.load(h)
    assert back.attrs["flag"] == 1
    assert back.attrs["nflag"] == 0


def test_store_failure_leaves_no_tmp(ws_root):
    h = workspace.new_handle("retrieval", "s", {}, [])
    with pytest.raises(TypeError):
        workspace.store(h, "not a dataset", {})
    assert not any(p.name.startswith(".tmp-") for p in (ws_root / "handles").iterdir())
    assert not workspace.exists(h)


def test_benchmark_kind_roundtrip():
    h = workspace.new_handle("benchmark", "bench", {"workers": 8}, [])
    workspace.store(h, {"scenes_per_hour": 12.5, "records": []}, {})
    assert workspace.load(h)["scenes_per_hour"] == 12.5


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        workspace.new_handle("cube", "s", {}, [])


def test_unknown_handle():
    with pytest.raises(workspace.UnknownHandle):
        workspace.meta("retrieval-000000000000")
