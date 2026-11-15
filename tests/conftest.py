"""Synthetic HLS L30 scene directories, a tiny canonical LUT and a matching R0.

The scene writer mirrors spires-io/tests/conftest.py (same CRS, origin and
file layout) so spires_io.hls.L30_SPEC opens it unchanged. 64x64 at 30 m.
"""
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine

CRS = "EPSG:32611"
X0, Y0 = 300000.0, 4170000.0
SHAPE = (64, 64)
L30_BANDS = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
             "nir": "B05", "swir1": "B06", "swir2": "B07"}
LUT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09"]


def write_raster(path: Path, values: np.ndarray, resolution_m: int, nodata=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype=values.dtype, crs=CRS, nodata=nodata,
        transform=Affine(resolution_m, 0.0, X0, 0.0, -resolution_m, Y0),
    ) as dst:
        dst.write(values, 1)
    return path


def write_l30_scene(root: Path, doy: int, *, dn=4000, shape=SHAPE, drop_band: str | None = None) -> Path:
    """One fake HLS L30 scene dir. ``drop_band`` omits a band file so the loader raises."""
    scene_id = f"HLS.L30.T11SLB.2025{doy:03d}T182301.v2.0"
    scene = root / scene_id
    rng = np.random.default_rng(doy)
    for name, code in L30_BANDS.items():
        if name == drop_band:
            continue
        values = (dn + rng.integers(-200, 200, size=shape)).astype(np.int16)
        path = write_raster(scene / f"{scene_id}.{code}.tif", values, 30, nodata=-9999)
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(SENSING_TIME="2025-02-14T18:23:01.1234567Z")
    write_raster(scene / f"{scene_id}.Fmask.tif", np.zeros(shape, np.uint8), 30, nodata=255)
    write_raster(scene / f"{scene_id}.SZA.tif", np.full(shape, 4500, np.uint16), 30)
    return scene


def make_lut(bands=LUT_BANDS) -> xr.Dataset:
    """Tiny canonical reflectance LUT: linspace axes, values in (0, 1), lap_type dust."""
    solar = np.linspace(0.0, 80.0, 5)
    lap = np.linspace(0.0, 1000.0, 4)
    sqrt_grain = np.linspace(5.0, 40.0, 6)
    nb = len(bands)
    base = np.linspace(0.9, 0.3, nb)[:, None, None, None]
    refl = (base
            * (1 - 0.2 * (lap / lap.max()))[None, None, :, None]
            * (1 - 0.3 * ((sqrt_grain - 5) / 35))[None, None, None, :]
            * (1 - 0.1 * (solar / 80))[None, :, None, None]).astype(np.float32)
    da = xr.DataArray(
        refl, dims=("band", "solar_angle", "lap_concentration", "sqrt_grain_radius"),
        coords={
            "band": list(bands),
            "solar_angle": ("solar_angle", solar, {"units": "degrees"}),
            "lap_concentration": ("lap_concentration", lap, {"units": "ppm", "lap_type": "dust"}),
            "sqrt_grain_radius": ("sqrt_grain_radius", sqrt_grain, {"units": "um^0.5"}),
        },
        name="reflectance",
    )
    return da.to_dataset()


def make_r0(scene: xr.Dataset, level=0.08) -> xr.DataArray:
    """Snow-free background on the scene grid with the scene's band names."""
    refl = scene["reflectance"]
    values = np.full(refl.shape, level, dtype=np.float32)
    return xr.DataArray(values, dims=refl.dims, coords={"y": refl.y, "x": refl.x, "band": refl.band},
                        name="background_reflectance")


@pytest.fixture
def scene_dirs(tmp_path):
    return [write_l30_scene(tmp_path, doy) for doy in (10, 20, 30)]


@pytest.fixture
def broken_scene_dir(tmp_path):
    return write_l30_scene(tmp_path, 40, drop_band="nir")


@pytest.fixture
def lut():
    return make_lut()


@pytest.fixture(scope="module")
def client():
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, processes=True,
                           dashboard_address=None, silence_logs=40)
    c = Client(cluster)
    yield c
    c.close()
    cluster.close()
