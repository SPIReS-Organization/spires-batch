"""Both modes on real hls_l30 scenes + their R0. Not a test; run by hand.

    python examples/smoke_hls_l30.py --lut lut.nc --r0 r0.nc --scenes-glob 'scenes/hls_l30/HLS.L30.*' --workers 4 --threads 8
"""
import argparse
import glob
import sys
import time

from spires_io.hls import L30_SPEC
from spires_io.background import load_background_reflectance
from spires_inversion import load_reflectance_lut

from spires_batch import kernels, run, workspace
from spires_batch.cluster import local_cluster


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lut", required=True)
    p.add_argument("--r0", required=True)
    p.add_argument("--scenes-glob", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--chunk-pixels", type=int, default=512)
    args = p.parse_args()

    scenes = sorted(glob.glob(args.scenes_glob))
    if not scenes:
        sys.exit(f"no scenes match {args.scenes_glob}")
    shared = {"lut": kernels.lut_for_spec(load_reflectance_lut(args.lut, expected_lap_type="dust"), L30_SPEC),
              "r0": load_background_reflectance(args.r0), "spec": L30_SPEC}
    client = local_cluster(args.workers, args.threads)
    try:
        t = time.perf_counter()
        records = run.run_stack(scenes, kernels.invert_scene, shared, client,
                                unit_bytes=lambda u: kernels.scene_bytes(u, L30_SPEC))
        print(f"stack: {len(scenes)} scenes in {time.perf_counter() - t:.1f}s", file=sys.stderr)
        for r in records:
            print(f"  {r.status:8s} {r.seconds:7.1f}s {r.unit.split('/')[-1]} {r.error_message or ''}", file=sys.stderr)
        t = time.perf_counter()
        rec = run.run_scene(scenes[0], shared, client, chunk_pixels=args.chunk_pixels, params={"mode": "scene"})
        print(f"scene: {rec.status} in {time.perf_counter() - t:.1f}s -> {workspace.path(rec.handle) if rec.status == 'done' else rec.error_message}", file=sys.stderr)
    finally:
        cluster = client.cluster
        client.close()
        cluster.close()


if __name__ == "__main__":   # workers spawn and re-import this module
    main()
