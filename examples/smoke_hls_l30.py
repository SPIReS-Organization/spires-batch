"""Both modes on the real hls_l30 scenes + Mammoth R0. Not a test; run by hand in spipy14.

    python examples/smoke_hls_l30.py --workers 4 --threads 8
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

LUT = "/data/spires/luts/spires_reflectance_lut_hlsl30_dust_skiles_size3.nc"
R0 = "/data/spires/r0/mammoth_hlsl30_r0.nc"
SCENES = sorted(glob.glob("/data/spires/scenes/hls_l30/HLS.L30.T11SLB.*"))

p = argparse.ArgumentParser()
p.add_argument("--workers", type=int, default=4)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--chunk-pixels", type=int, default=512)
args = p.parse_args()

shared = {"lut": kernels.lut_for_spec(load_reflectance_lut(LUT, expected_lap_type="dust"), L30_SPEC),
          "r0": load_background_reflectance(R0), "spec": L30_SPEC}
client = local_cluster(args.workers, args.threads)
try:
    t = time.perf_counter()
    records = run.run_stack(SCENES, kernels.invert_scene, shared, client,
                            unit_bytes=lambda u: kernels.scene_bytes(u, L30_SPEC))
    print(f"stack: {len(SCENES)} scenes in {time.perf_counter() - t:.1f}s", file=sys.stderr)
    for r in records:
        print(f"  {r.status:8s} {r.seconds:7.1f}s {r.unit.split('/')[-1]} {r.error_message or ''}", file=sys.stderr)
    t = time.perf_counter()
    rec = run.run_scene(SCENES[0], shared, client, chunk_pixels=args.chunk_pixels, params={"mode": "scene"})
    print(f"scene: {rec.status} in {time.perf_counter() - t:.1f}s -> {workspace.path(rec.handle) if rec.status == 'done' else rec.error_message}", file=sys.stderr)
finally:
    cluster = client.cluster
    client.close()
    cluster.close()
