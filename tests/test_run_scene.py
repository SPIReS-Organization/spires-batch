import dataclasses

import numpy as np
import pytest
from spires_io.hls import L30_SPEC

from spires_batch import kernels, run, workspace
from conftest import make_r0


@pytest.fixture(autouse=True)
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIRES_BATCH_WORKSPACE", str(tmp_path / "ws"))


@pytest.fixture
def shared(scene_dirs, lut):
    scene = kernels.load(scene_dirs[0], spec=L30_SPEC)
    return {"lut": kernels.lut_for_spec(lut, L30_SPEC), "r0": make_r0(scene), "spec": L30_SPEC}


def test_run_scene_matches_invert_scene_bit_for_bit(client, scene_dirs, shared):
    record = run.run_scene(scene_dirs[0], shared, client, chunk_pixels=24)
    assert record.status == "done"
    blocked = workspace.load(record.handle)
    whole = kernels.invert_scene(scene_dirs[0], n_threads=3, **shared)
    for name in ("fsnow", "fshade", "lap_concentration", "grain_radius"):
        np.testing.assert_array_equal(blocked[name].values, whole[name].values)
    np.testing.assert_array_equal(blocked["valid"].values, whole["valid"].values)
    np.testing.assert_array_equal(blocked.y.values, whole.y.values)


def test_run_scene_bbox(client, scene_dirs, shared):
    bbox = (300000.0, 4169520.0, 300480.0, 4170000.0)
    record = run.run_scene(scene_dirs[0], shared, client, bbox=bbox, chunk_pixels=8)
    out = workspace.load(record.handle)
    assert out["fsnow"].shape == (16, 16)
    whole = kernels.invert_scene(scene_dirs[0], n_threads=1, bbox=bbox, **shared)
    np.testing.assert_array_equal(out["fsnow"].values, whole["fsnow"].values)


def test_run_scene_failed_block_yields_failed_record(client, broken_scene_dir, shared):
    record = run.run_scene(broken_scene_dir, shared, client)
    assert record.status == "failed" and record.error_type == "SceneLayoutError"
    assert not workspace.exists(record.handle)


def test_run_scene_skips_if_stored(client, scene_dirs, shared):
    first = run.run_scene(scene_dirs[0], shared, client)
    second = run.run_scene(scene_dirs[0], shared, client)
    assert first.handle == second.handle and second.status == "skipped"


def _boom_invert(scene, background, lut, *, n_threads=1, **kw):
    raise RuntimeError("block exploded")


def test_run_scene_block_failure_is_failed_record(client, scene_dirs, shared):
    boom = dataclasses.replace(kernels.DEFAULT, name="boom", invert=_boom_invert)
    record = run.run_scene(scene_dirs[0], shared, client, kernel=boom, chunk_pixels=32)
    assert record.status == "failed" and record.error_type == "RuntimeError"
    assert "block exploded" in record.error_message
    assert not workspace.exists(record.handle)
