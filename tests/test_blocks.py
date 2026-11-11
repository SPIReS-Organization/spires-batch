import numpy as np
import xarray as xr

from spires_batch import blocks


def _scene(ny=64, nx=64, nb=3):
    rng = np.random.default_rng(0)
    refl = xr.DataArray(rng.random((ny, nx, nb), dtype=np.float32), dims=("y", "x", "band"),
                        coords={"y": np.arange(ny, 0, -1) * 30.0, "x": np.arange(nx) * 30.0,
                                "band": ["a", "b", "c"][:nb]})
    valid = xr.DataArray(rng.random((ny, nx)) > 0.1, dims=("y", "x"), coords={"y": refl.y, "x": refl.x})
    sza = xr.DataArray(np.full((ny, nx), 45.0, np.float32), dims=("y", "x"), coords={"y": refl.y, "x": refl.x})
    return xr.Dataset({"reflectance": refl, "valid": valid, "solar_zenith": sza}, attrs={"sensor": "t"})


def test_block_windows_ragged_edges():
    windows = blocks.block_windows({"y": 64, "x": 64}, chunk_pixels=24)
    assert len(windows) == 9   # 24+24+16 along each axis
    sizes = sorted({(w["y"].stop - w["y"].start, w["x"].stop - w["x"].start) for w in windows})
    assert sizes == [(16, 16), (16, 24), (24, 16), (24, 24)]


def test_cut_covers_scene_exactly_and_blocks_are_contiguous_copies():
    scene = _scene()
    parts = blocks.cut(scene, chunk_pixels=24)
    assert len(parts) == 9
    assert sum(p.sizes["y"] * p.sizes["x"] for p in parts) == 64 * 64
    for p in parts:
        assert p["reflectance"].values.flags["C_CONTIGUOUS"]
        assert p.sizes["band"] == 3
        assert p.attrs == scene.attrs
    assert not np.shares_memory(parts[0]["reflectance"].values, scene["reflectance"].values)


def test_reassemble_is_exact_roundtrip():
    scene = _scene(50, 70)
    scene["reflectance"].attrs["units"] = "1"
    back = blocks.reassemble(blocks.cut(scene, chunk_pixels=16))
    xr.testing.assert_identical(back["reflectance"], scene["reflectance"])
    xr.testing.assert_identical(back["valid"], scene["valid"])
    assert back["reflectance"].attrs == {"units": "1"}
    assert back.attrs == scene.attrs


def test_cut_pairs_scene_and_background_on_same_windows():
    scene = _scene()
    background = scene["reflectance"] * 0.2
    pairs = blocks.cut_pair(scene, background, chunk_pixels=24)
    assert len(pairs) == 9
    for s, b in pairs:
        np.testing.assert_array_equal(s.y.values, b.y.values)
        np.testing.assert_array_equal(s.x.values, b.x.values)


def test_chunk_pixels_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="chunk_pixels"):
        blocks.block_windows({"y": 4, "x": 4}, chunk_pixels=0)
