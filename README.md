# spires-batch

Batch processing for the [SPIReS](https://github.com/SPIReS-Organization)
package family — run the retrieval **at scale** over stacks of granules and
spacetime cubes.

`spires-batch` builds on the single-unit convenience kernel exposed by the
[`spires`](https://github.com/SPIReS-Organization/spires) metapackage
(`spires.invert(...)`, which wires io + lut + r0 → inversion → postprocess for
**one** unit of work) and parallelizes it across **many** units. Schedulers,
clusters, chunking, and job submission live here so the metapackage's
convenience layer can stay light.

## Intended scope

- **In `spires-batch`:** anything that knows about *many* units of work —
  stacks of granules, spacetime-cube tiling, scheduling, cluster/job
  submission. Heavy execution backends (dask, slurm) are opt-in extras.
- **Not here:** the retrieval logic itself, and the single-unit wiring. That
  is the metapackage's light convenience layer (`spires.invert`).

```bash
pip install spires-batch          # base: parallelization API
pip install spires-batch[dask]    # + dask distributed backend
pip install spires-batch[slurm]   # + slurm submission (dask-jobqueue)
```

## Open questions (this is a stub — discussion welcome)

The package is scaffolding only; the API below is a sketch to argue about, not
a commitment.

- **Execution backends.** Start with dask (`distributed` for local/cluster,
  `dask-jobqueue` for slurm)? Should a plain serial/loop backend exist for
  debugging and tiny jobs?
- **Unit of work.** Is the natural unit a single granule, or a chunk of a
  spacetime cube? Both? How does the user express "this stack of granules"
  vs. "this cube, tiled" — globs, an intake catalog, an xarray of paths?
- **CLI vs. API.** Does `spires-batch` ship its own CLI (`spires-batch run
  ...`), or does it register as a backend that the metapackage's `spires` CLI
  dispatches to (`spires run --backend dask ...`)? The latter keeps one CLI
  surface; the former keeps batch fully decoupled.
- **Dependency direction.** `spires-batch` depends on `spires` (so it can call
  `spires.invert`). The metapackage must **not** depend back on
  `spires-batch`, to stay light — `spires[batch]` is the surfacing extra. Is
  depending on the whole metapackage too coarse, vs. depending directly on
  `spires-inversion` + the sub-packages?
- **State & restart.** Big jobs fail partway. Do we need checkpointing /
  resumability, or is that the caller's problem (re-run, skip existing
  outputs)?

> **Status:** scaffolding only. Package layout and the dependency on `spires`
> are in place; the implementation is to be filled in.
