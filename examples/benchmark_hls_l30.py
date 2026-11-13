"""Poster benchmark: scenes/hour and per-scene wall-clock vs cluster shape.

Stack: HLS L30 T11SLB 2024-10-01..2025-06-30 (fetch first with
    spires-fetch --dest /scratch/spires/scenes --sensor hls_l30 --aoi mammoth --start 2024-10-01 --end 2025-06-30
). Configurations: 128x1 64x2 32x4 16x8 8x16 (stack) and 1x8 1x32 1x128 (scene).
Each configuration runs in a fresh cluster and a fresh workspace root so nothing
is skipped. Results are stored as a ``benchmark`` artifact and printed as JSON.

    python examples/benchmark_hls_l30.py --scenes-glob '/scratch/spires/scenes/hls_l30/HLS.L30.T11SLB.*' \
        --configs 128x1 64x2 32x4 16x8 8x16 --scene-configs 1x8 1x32 1x128
"""
import argparse
import glob
import json
import os
import statistics
import sys
import tempfile
import time

from distributed.diagnostics import MemorySampler
from spires_io.hls import L30_SPEC
from spires_io.background import load_background_reflectance
from spires_inversion import load_reflectance_lut

from spires_batch import kernels, run, workspace
from spires_batch.cluster import local_cluster

LUT = "/data/spires/luts/spires_reflectance_lut_hlsl30_dust_skiles_size3.nc"
R0 = "/data/spires/r0/mammoth_hls_l30_r0.nc"


def parse_shape(s):
    w, t = s.lower().split("x")
    return int(w), int(t)


def _peak(ms):
    return int(ms.to_pandas()["run"].max()) if len(ms.samples["run"]) else None


def _close(client):
    cluster = client.cluster
    client.close()
    cluster.close()


def bench_stack(scenes, shared, workers, threads):
    ms = MemorySampler()
    client = local_cluster(workers, threads)
    try:
        t0 = time.perf_counter()
        with ms.sample("run"):
            records = run.run_stack(scenes, kernels.invert_scene, shared, client)
        wall = time.perf_counter() - t0
    finally:
        _close(client)
    done = [r for r in records if r.status == "done"]
    secs = sorted(r.seconds for r in done)
    return {
        "mode": "stack", "workers": workers, "threads": threads, "scenes": len(scenes),
        "done": len(done), "failed": len(records) - len(done), "wall_s": wall,
        "scenes_per_hour": len(done) / wall * 3600 if wall else None,
        "median_scene_s": statistics.median(secs) if secs else None,
        "p90_scene_s": secs[int(0.9 * (len(secs) - 1))] if secs else None,
        "peak_cluster_memory_bytes": _peak(ms),
        "records": [r.to_dict() for r in records],
    }


def bench_scene(scene, shared, workers, threads, chunk_pixels):
    ms = MemorySampler()
    client = local_cluster(workers, threads)
    try:
        t0 = time.perf_counter()
        with ms.sample("run"):
            record = run.run_scene(scene, shared, client, chunk_pixels=chunk_pixels)
        wall = time.perf_counter() - t0
    finally:
        _close(client)
    return {"mode": "scene", "workers": workers, "threads": threads, "chunk_pixels": chunk_pixels,
            "scene": record.unit, "status": record.status, "wall_s": wall,
            "peak_cluster_memory_bytes": _peak(ms), "record": record.to_dict()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-glob", default="/scratch/spires/scenes/hls_l30/HLS.L30.T11SLB.*")
    p.add_argument("--configs", nargs="*", default=["128x1", "64x2", "32x4", "16x8", "8x16"])
    p.add_argument("--scene-configs", nargs="*", default=["1x8", "1x32", "1x128"])
    p.add_argument("--chunk-pixels", type=int, default=512)
    p.add_argument("--max-scenes", type=int, default=None)
    args = p.parse_args(argv)

    scenes = sorted(glob.glob(args.scenes_glob))[: args.max_scenes]
    if not scenes:
        sys.exit(f"no scenes match {args.scenes_glob}")
    shared = {"lut": kernels.lut_for_spec(load_reflectance_lut(LUT, expected_lap_type="dust"), L30_SPEC),
              "r0": load_background_reflectance(R0), "spec": L30_SPEC}

    results = []
    keep_root = os.environ.get("SPIRES_BATCH_WORKSPACE")
    for cfg in args.configs:
        w, t = parse_shape(cfg)
        os.environ["SPIRES_BATCH_WORKSPACE"] = tempfile.mkdtemp(prefix=f"bench-{cfg}-")
        r = bench_stack(scenes, shared, w, t)
        print(f"{cfg:>7s} stack  {r['scenes_per_hour']:.1f} scenes/h  median {r['median_scene_s']:.1f}s  "
              f"peak {(r['peak_cluster_memory_bytes'] or 0) / 2**30:.1f} GiB  failed {r['failed']}", file=sys.stderr)
        results.append(r)
    for cfg in args.scene_configs:
        w, t = parse_shape(cfg)
        os.environ["SPIRES_BATCH_WORKSPACE"] = tempfile.mkdtemp(prefix=f"bench-{cfg}-")
        r = bench_scene(scenes[0], shared, w, t, args.chunk_pixels)
        print(f"{cfg:>7s} scene  {r['wall_s']:.1f}s  {r['status']}", file=sys.stderr)
        results.append(r)

    if keep_root:
        os.environ["SPIRES_BATCH_WORKSPACE"] = keep_root
    else:
        os.environ.pop("SPIRES_BATCH_WORKSPACE", None)
    handle = workspace.new_handle("benchmark", "benchmark_hls_l30",
                                  {"configs": args.configs + args.scene_configs, "scenes": len(scenes)}, [])
    workspace.store(handle, {"results": results}, {"scenes_glob": args.scenes_glob})
    print(json.dumps({"handle": handle, "path": str(workspace.path(handle)),
                      "summary": [{k: v for k, v in r.items() if k not in ("records", "record")} for r in results]},
                     indent=2))


if __name__ == "__main__":
    main()
