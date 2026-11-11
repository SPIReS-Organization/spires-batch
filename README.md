# spires-batch

Batch processing for the [SPIReS](https://github.com/SPIReS-Organization)
package family — run the retrieval **at scale** over stacks of granules and
spacetime cubes.

`spires-batch` depends directly on `spires-io`, `spires-inversion` and
`spires-contract` and parallelizes the single-scene retrieval across **many**
units of work. Schedulers, clusters, chunking, and job submission live here so
the [`spires`](https://github.com/SPIReS-Organization/spires) metapackage's
convenience layer can stay light.

## Intended scope

- **In `spires-batch`:** anything that knows about *many* units of work —
  stacks of granules, spacetime-cube tiling, scheduling, cluster/job
  submission. dask-distributed is the engine; SLURM submission is an opt-in extra.
- **Not here:** the retrieval logic itself. That is `spires-inversion`, reached
  through the default `invert_scene` kernel.

```bash
pip install spires-batch          # dask distributed is a core dependency
pip install spires-batch[slurm]   # + slurm submission (dask-jobqueue)
```

## What it does

`spires-batch` is a kernel-agnostic orchestration layer: it runs many units of
work through a caller-supplied `kernel(unit, *, n_threads, **shared) -> xr.Dataset`
on a dask cluster, scatters shared inputs (LUT, R0) once, records per-unit
timings and failures, and stores outputs in a content-addressed workspace so a
re-run skips finished units and retries only failed ones. It ships one default
kernel, `invert_scene` (spires-io → R0 alignment → `spires_inversion.invert`).
No cloud/tree/water masking or temporal fill lives here — those are
sensor-specific and belong in `spires-postprocess` or your script.

| Entry point                                   | Unit of parallelism | Use                                                |
|-----------------------------------------------|---------------------|----------------------------------------------------|
| `run_stack(units, kernel, shared, client)`    | one task per scene  | many scenes; each inverts on the worker's threads  |
| `run_scene(unit, shared, client, bbox=None)`  | one task per block  | one scene or ROI, as fast as the machine allows    |

The cluster shape `workers × threads_per_worker` is the single knob:
`128×1` for "128 scenes at once", `16×8` for the memory-friendly middle,
`1×128` for one scene as fast as possible. The C++ inversion releases the GIL,
so threads inside a worker genuinely run concurrently.

```bash
spires-batch stack --sensor hls_l30 --lut lut.nc --r0 r0.nc --workers 16 --threads 8 scenes/HLS.L30.*
spires-batch scene --sensor hls_l30 --lut lut.nc --r0 r0.nc --workers 1 --threads 64 \
    --bbox 320000 4165000 330000 4175000 scenes/HLS.L30.T11SLB.2025045T182301.v2.0
```

Artifacts land under `$SPIRES_BATCH_WORKSPACE` (default `~/.cache/spires-batch/workspace`).
`examples/benchmark_hls_l30.py` produces the scenes/hour vs cluster-shape table.

## Development

```bash
conda activate spipy14
pip install -e . --no-deps          # spires-io / spires-inversion / spires-contract are already in the env
pytest tests -q
```
