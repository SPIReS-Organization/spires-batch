import json
import numpy as np
import pytest
import xarray as xr

from spires_batch import cli
from conftest import make_lut, write_l30_scene
from spires_io.hls import open_hls_l30_scene


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIRES_BATCH_WORKSPACE", str(tmp_path / "ws"))
    scenes = [write_l30_scene(tmp_path / "scenes", d) for d in (10, 20)]
    broken = write_l30_scene(tmp_path / "scenes", 30, drop_band="red")
    lut_path = tmp_path / "lut.nc"
    make_lut().to_netcdf(lut_path)
    scene = open_hls_l30_scene(scenes[0])
    r0 = xr.DataArray(np.full(scene["reflectance"].shape, 0.08, np.float32), dims=("y", "x", "band"),
                      coords={"y": scene.y, "x": scene.x, "band": scene.band}, name="r0_reflectance")
    r0_path = tmp_path / "r0.nc"
    r0.to_dataset().to_netcdf(r0_path)
    return scenes, broken, lut_path, r0_path


def test_parse_sensor():
    assert cli.SPECS["hls_l30"].sensor == "hls_l30"
    assert set(cli.SPECS) == {"hls_l30", "hls_s30", "landsat", "sentinel2"}


def test_stack_cli_exit_code_and_records(inputs, tmp_path):
    scenes, broken, lut_path, r0_path = inputs
    records = tmp_path / "records.json"
    rc = cli.main(["stack", "--sensor", "hls_l30", "--lut", str(lut_path), "--r0", str(r0_path),
                   "--workers", "2", "--threads", "1", "--records", str(records), *map(str, scenes)])
    assert rc == 0
    rc = cli.main(["stack", "--sensor", "hls_l30", "--lut", str(lut_path), "--r0", str(r0_path),
                   "--workers", "2", "--threads", "1", "--records", str(records), *map(str, scenes + [broken])])
    assert rc == 1
    statuses = sorted(r["status"] for r in json.loads(records.read_text()))
    assert statuses == ["failed", "skipped", "skipped"]


def test_scene_cli(inputs, tmp_path):
    scenes, _, lut_path, r0_path = inputs
    rc = cli.main(["scene", "--sensor", "hls_l30", "--lut", str(lut_path), "--r0", str(r0_path),
                   "--workers", "2", "--threads", "1", "--chunk-pixels", "32",
                   "--bbox", "300000", "4169520", "300480", "4170000", str(scenes[0])])
    assert rc == 0
