# spires-batch

Scale-out batch processing for the SPIReS package family: run the retrieval over
stacks of granules and spacetime cubes (dask / slurm), parallelizing the
single-unit `spires.invert` kernel. Opt-in via `pip install spires[batch]`.

This package is part of the [SPIReS family](https://spires.readthedocs.io/).

```{note}
**Status: scaffolding.** `spires-batch` is an early scaffold — the public API is
not implemented yet, so this site is a placeholder. It will grow an API
reference (autodoc) as the scale-out code lands. Track progress in the
[repository](https://github.com/SPIReS-Organization/spires-batch).
```

## Planned scope

- Parallelize the `spires` metapackage's single-unit `invert` kernel across many
  units of work.
- Provide dask / slurm backends (opt-in extras `spires-batch[dask]`, `[slurm]`).
