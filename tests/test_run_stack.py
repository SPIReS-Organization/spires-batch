import json

import numpy as np
import pytest
import xarray as xr
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


def test_record_roundtrip(tmp_path):
    r = run.Record(unit="u", handle="retrieval-abc", status="failed", seconds=1.5, worker="tcp://x",
                   error_type="ValueError", error_message="boom", traceback="tb")
    path = tmp_path / "records.json"
    run.write_records(path, [r])
    back = json.loads(path.read_text())
    assert back[0]["status"] == "failed" and back[0]["error_type"] == "ValueError"


def test_run_stack_four_units_one_failing(client, scene_dirs, broken_scene_dir, shared):
    units = scene_dirs + [broken_scene_dir]
    records = run.run_stack(units, kernels.invert_scene, shared, client)
    assert [r.status for r in records].count("done") == 3
    failed = [r for r in records if r.status == "failed"]
    assert len(failed) == 1 and failed[0].unit == str(broken_scene_dir)
    assert failed[0].error_type == "SceneLayoutError"
    assert all(r.seconds > 0 for r in records)
    assert all(r.worker for r in records if r.status != "skipped")
    done = [r for r in records if r.status == "done"]
    assert all(workspace.exists(r.handle) for r in done)
    assert not workspace.exists(failed[0].handle)
    out = workspace.load(done[0].handle)
    assert "fsnow" in out and out.attrs["source_product_id"] == records[0].unit.split("/")[-1]


def test_run_stack_rerun_skips_done_and_retries_failed(client, scene_dirs, broken_scene_dir, shared):
    units = scene_dirs + [broken_scene_dir]
    run.run_stack(units, kernels.invert_scene, shared, client)
    again = run.run_stack(units, kernels.invert_scene, shared, client)
    assert sorted(r.status for r in again) == ["failed", "skipped", "skipped", "skipped"]


def test_run_stack_handle_depends_on_shared_content(client, scene_dirs, shared):
    a = run.run_stack(scene_dirs[:1], kernels.invert_scene, shared, client)[0]
    other = dict(shared, r0=shared["r0"] * 0.5)
    b = run.run_stack(scene_dirs[:1], kernels.invert_scene, other, client)[0]
    assert a.handle != b.handle and b.status == "done"


def test_run_stack_preconditions(client, scene_dirs, shared):
    with pytest.raises(ValueError, match="zero units"):
        run.run_stack([], kernels.invert_scene, shared, client)
    with pytest.raises(TypeError, match="lambda"):
        run.run_stack(scene_dirs, kernels.invert_scene, dict(shared, f=lambda: 1), client)
    with pytest.raises(run.MemoryCheckError, match="memory_limit"):
        run.run_stack(scene_dirs, kernels.invert_scene, shared, client, unit_bytes=lambda u: 1 << 60)


def test_run_stack_passes_worker_threads_to_kernel(client, scene_dirs, shared):
    seen = run.run_stack(scene_dirs[:1], _echo_threads, {}, client, store=False)
    assert seen[0].status == "done" and seen[0].result == 2   # fixture cluster is 2x2


def _echo_threads(unit, *, n_threads):
    return n_threads


def _returns_int(unit, *, n_threads):
    return 7


def test_run_stack_store_failure_is_a_failed_record(client, scene_dirs):
    records = run.run_stack(scene_dirs[:1], _returns_int, {}, client)
    assert len(records) == 1
    record = records[0]
    assert record.status == "failed" and record.error_type == "TypeError"
    assert record.traceback and record.seconds > 0
    assert not workspace.exists(record.handle)


def test_run_stack_resolves_scattered_shared_inputs(client, scene_dirs):
    # 512*512*2 float32 = 2 MiB, above the 1 MiB scatter threshold, so `big`
    # reaches the worker as a Future and dask must resolve it inside the kwargs.
    big = xr.DataArray(np.full((512, 512, 2), 0.5, np.float32), dims=("y", "x", "band"))
    record = run.run_stack(scene_dirs[:1], _sum_big, {"big": big}, client, store=False)[0]
    assert record.status == "done", record.traceback
    assert record.result == 512 * 512 * 2 * 0.5


def _sum_big(unit, *, n_threads, big):
    return float(big.sum())
