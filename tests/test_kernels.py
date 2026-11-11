import numpy as np
import pytest
import xarray as xr
from spires_contract import ContractError
from spires_contract.results import validate_results
from spires_io.hls import L30_SPEC

from spires_batch import kernels
from conftest import make_lut, make_r0


def test_band_map_from_spec_hls_l30():
    assert kernels.band_map_from_spec(L30_SPEC) == {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
        "nir": "B05", "swir1": "B06", "swir2": "B07"}


def test_lut_for_spec_selects_and_relabels(lut):
    out = kernels.lut_for_spec(lut, L30_SPEC)
    assert list(out["reflectance"].band.values) == list(L30_SPEC.band_files)
    np.testing.assert_array_equal(out["reflectance"].sel(band="nir").values,
                                  lut["reflectance"].sel(band="B05").values)


def test_lut_for_spec_missing_band_raises():
    lut = make_lut(bands=["B01", "B02", "B03"])
    with pytest.raises(ValueError, match="B04"):
        kernels.lut_for_spec(lut, L30_SPEC)


def test_load_masks_invalid_pixels_and_keeps_valid(scene_dirs):
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC)
    assert set(scene.data_vars) >= {"reflectance", "valid", "solar_zenith"}
    assert scene["reflectance"].dtype == np.float32
    assert bool(scene["valid"].all())   # fake Fmask is all-clear


def test_load_bbox(scene_dirs):
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC, bbox=(300000.0, 4169520.0, 300480.0, 4170000.0))
    assert scene["reflectance"].shape == (16, 16, 7)


def test_align_r0_same_grid_passthrough(scene_dirs):
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC)
    r0 = make_r0(scene)
    out = kernels.align_r0(scene, r0)
    assert out.dims == ("y", "x", "band") and out.dtype == np.float32
    np.testing.assert_array_equal(out.band.values, scene.band.values)


def test_align_r0_subsets_full_tile_to_bbox(scene_dirs):
    full = kernels.load(scene_dirs[0], spec=L30_SPEC)
    r0 = make_r0(full)
    sub = kernels.load(scene_dirs[0], spec=L30_SPEC, bbox=(300000.0, 4169520.0, 300480.0, 4170000.0))
    out = kernels.align_r0(sub, r0)
    assert out.shape == (16, 16, 7)
    np.testing.assert_array_equal(out.y.values, sub.y.values)


def test_align_r0_band_mismatch_raises(scene_dirs):
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC)
    r0 = make_r0(scene).assign_coords(band=["B01", "B02", "B03", "B04", "B05", "B06", "B07"])
    with pytest.raises(ValueError, match="band"):
        kernels.align_r0(scene, r0)


def test_invert_scene_end_to_end(scene_dirs, lut):
    shared = {"lut": kernels.lut_for_spec(lut, L30_SPEC), "spec": L30_SPEC}
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC)
    shared["r0"] = make_r0(scene)
    out = kernels.invert_scene(scene_dirs[0], n_threads=2, **shared)
    assert isinstance(out, xr.Dataset)
    validate_results(out)
    assert "valid" in out.data_vars
    assert out.attrs["source_product_id"] == scene_dirs[0].name
    assert np.isfinite(out["fsnow"].values).all()


def test_invert_threads_do_not_change_result(scene_dirs, lut):
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC)
    r0 = make_r0(scene)
    l = kernels.lut_for_spec(lut, L30_SPEC)
    a = kernels.invert(scene, r0, l, n_threads=1)
    b = kernels.invert(scene, r0, l, n_threads=4)
    for name in ("fsnow", "fshade", "lap_concentration", "grain_radius"):
        np.testing.assert_array_equal(a[name].values, b[name].values)


def test_scene_bytes(scene_dirs):
    assert kernels.scene_bytes(scene_dirs[0], L30_SPEC) == 64 * 64 * 7 * 4 * kernels.BYTES_FACTOR
