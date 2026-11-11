"""Pixel-block engine for ``run_scene``: cut (y, x) into squares, scatter off-graph, reassemble.

Blocking is by dim name: ``y`` and ``x`` are cut into ``chunk_pixels`` squares
with a ragged last block; every other dim (``band``) stays whole. Each block is
an ``ascontiguousarray`` copy so the scattered payload never pins the parent
array. Scattering with ``direct=True, hash=False`` and replicating to ``n``
workers rather than ``broadcast=True`` is lifted from spipyservice
``spires/service.py:_scatter_blocks`` (measured there: in-graph chunking does
not shrink what the scheduler receives, ratio 1.00; off-graph scatter cut the
pickled graph from 79.7 MB to 32 KB).
"""
import numpy as np
import xarray as xr


def block_windows(sizes: dict, chunk_pixels: int) -> list[dict]:
    """Row-major list of ``{"y": slice, "x": slice}`` windows covering ``sizes``."""
    if not isinstance(chunk_pixels, (int, np.integer)) or chunk_pixels < 1:
        raise ValueError(f"chunk_pixels must be a positive integer, got {chunk_pixels!r}")
    edges = {}
    for dim in ("y", "x"):
        stops = list(range(chunk_pixels, sizes[dim], chunk_pixels)) + [sizes[dim]]
        starts = [0] + stops[:-1]
        edges[dim] = list(zip(starts, stops))
    return [{"y": slice(y0, y1), "x": slice(x0, x1)}
            for (y0, y1) in edges["y"] for (x0, x1) in edges["x"]]


def _contiguous(obj):
    if isinstance(obj, xr.Dataset):
        return obj.copy(data={n: np.ascontiguousarray(obj[n].values) for n in obj.data_vars})
    return obj.copy(data=np.ascontiguousarray(obj.values))


def cut(obj, chunk_pixels: int) -> list:
    """Cut a Dataset or DataArray with ``y``/``x`` dims into contiguous blocks."""
    return [_contiguous(obj.isel(**w)) for w in block_windows(dict(obj.sizes), chunk_pixels)]


def cut_pair(scene: xr.Dataset, background: xr.DataArray, chunk_pixels: int) -> list[tuple]:
    """Cut a scene and its background on identical windows; returns ``[(scene_block, bg_block), ...]``."""
    windows = block_windows(dict(scene.sizes), chunk_pixels)
    return [(_contiguous(scene.isel(**w)), _contiguous(background.isel(**w))) for w in windows]


def reassemble(parts: list) -> xr.Dataset:
    """Inverse of ``cut`` for a regular block grid; attrs taken from the first block."""
    first_x = parts[0].x.values[0]
    y_order = np.concatenate([p.y.values for p in parts if p.x.values[0] == first_x])
    first_y = parts[0].y.values[0]
    x_order = np.concatenate([p.x.values for p in parts if p.y.values[0] == first_y])
    out = xr.combine_by_coords(parts, combine_attrs="drop").reindex(y=y_order, x=x_order)
    out.attrs = dict(parts[0].attrs)
    return out


def scatter(client, parts: list, replicas: int = 2) -> list:
    """Scatter blocks off-graph, directly to workers, with random keys; replicate ``n`` ways."""
    futures = client.scatter(parts, direct=True, hash=False)
    if replicas > 1 and len(client.scheduler_info()["workers"]) > 1:
        client.replicate(futures, n=replicas)
    return futures
