"""``spires-batch stack`` and ``spires-batch scene``: thin wrappers, exit 1 if any unit failed."""
import argparse
import sys
from pathlib import Path

from spires_io import hls, landsat, sentinel2
from spires_io.background import load_background_reflectance
from spires_inversion import load_reflectance_lut

from spires_batch import kernels, run, workspace
from spires_batch.cluster import local_cluster, slurm_cluster

SPECS = {"hls_l30": hls.L30_SPEC, "hls_s30": hls.S30_SPEC, "landsat": landsat.SPEC, "sentinel2": sentinel2.SPEC}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sensor", required=True, choices=sorted(SPECS))
    p.add_argument("--lut", required=True, type=Path, help="canonical reflectance LUT (.nc)")
    p.add_argument("--r0", required=True, type=Path, help="background reflectance (.nc with r0_reflectance)")
    p.add_argument("--workers", type=int, default=8, help="local backend only")
    p.add_argument("--threads", type=int, default=8, help="local backend only: threads per worker")
    p.add_argument("--memory-limit", default=None, help="local backend only: per worker, e.g. 64GB (default: RAM / workers)")
    p.add_argument("--records", type=Path, default=None, help="write per-unit JSON records here")
    p.add_argument("--backend", choices=["local", "slurm"], default="local",
                   help="dask cluster back-end; slurm submits jobs via dask-jobqueue")
    p.add_argument("--slurm-jobs", type=int, default=1, help="slurm backend: SLURM jobs to submit")
    p.add_argument("--slurm-cores", type=int, default=1, help="slurm backend: cores per SLURM job")
    p.add_argument("--slurm-memory", default="16GB", help="slurm backend: memory per SLURM job, e.g. 64GB")
    p.add_argument("--slurm-queue", default=None, help="slurm backend: SLURM partition (required)")
    p.add_argument("--slurm-walltime", default="01:00:00", help="slurm backend: HH:MM:SS per SLURM job")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spires-batch")
    sub = parser.add_subparsers(dest="command", required=True)
    stack = sub.add_parser("stack", help="one task per scene")
    _common(stack)
    stack.add_argument("scenes", nargs="+", type=Path)
    scene = sub.add_parser("scene", help="one scene, block-parallel")
    _common(scene)
    scene.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                       help="in the scene CRS")
    scene.add_argument("--chunk-pixels", type=int, default=512)
    scene.add_argument("scene", type=Path)
    return parser


def load_shared(args) -> dict:
    spec = SPECS[args.sensor]
    lut = kernels.lut_for_spec(load_reflectance_lut(args.lut, expected_lap_type="dust"), spec)
    r0 = load_background_reflectance(args.r0)
    return {"lut": lut, "r0": r0, "spec": spec}


def make_client(args):
    if args.backend == "local":
        return local_cluster(args.workers, args.threads, memory_limit=args.memory_limit)
    return slurm_cluster(jobs=args.slurm_jobs, cores=args.slurm_cores, memory=args.slurm_memory,
                         queue=args.slurm_queue, walltime=args.slurm_walltime)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == "slurm" and not args.slurm_queue:
        parser.error("--backend slurm requires --slurm-queue")
    shared = load_shared(args)
    client = make_client(args)
    try:
        if args.command == "stack":
            records = run.run_stack(args.scenes, kernels.invert_scene, shared, client,
                                    unit_bytes=lambda u: kernels.scene_bytes(u, shared["spec"]),
                                    records_path=args.records)
        else:
            records = [run.run_scene(args.scene, shared, client, bbox=tuple(args.bbox) if args.bbox else None,
                                     chunk_pixels=args.chunk_pixels)]
            if args.records:
                run.write_records(args.records, records)
    finally:
        cluster = client.cluster
        client.close()
        cluster.close()
    for r in records:
        line = f"{r.status:8s} {r.seconds:8.1f}s  {r.unit}"
        if r.status == "failed":
            line += f"  {r.error_type}: {r.error_message}"
        elif workspace.exists(r.handle):
            line += f"  -> {workspace.path(r.handle)}"
        print(line, file=sys.stderr)
    return 1 if any(r.status == "failed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
