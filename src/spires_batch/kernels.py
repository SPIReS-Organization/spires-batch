"""Default kernel: io -> R0 alignment -> inversion for one scene.

The orchestration layer calls ``kernel(unit, *, n_threads, **shared) -> xr.Dataset``.
This module's ``invert_scene`` is that kernel for a scene directory, written as
three reusable steps so ``run_scene`` can run ``load`` and ``align_r0`` on the
client and ``invert`` per pixel block. Nothing here is sensor specific beyond
the ``SensorSpec`` passed in; cloud/tree/water masking and temporal fill live
in spires-postprocess or the caller's script.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import xarray as xr
from spires_contract import SpiresData
from spires_contract.spectra import validate_background_spectra
from spires_io.raster_scene import open_raster_scene, resolve_one
from spires_io.sensor_spec import SensorSpec
import spires_inversion

#: target + background + float64 kernel results, roughly, per pixel-band float32.
BYTES_FACTOR = 3

_BAND_CODE = re.compile(r"(B\d{1,2}A?)(?=\.|_|$)", re.IGNORECASE)


def band_map_from_spec(spec: SensorSpec) -> dict[str, str]:
    """Scene band name -> LUT band label, parsed from the spec's file globs (``*.B05.tif`` -> ``B05``)."""
    out = {}
    for name, pattern in spec.band_files.items():
        match = _BAND_CODE.search(Path(pattern).stem)
        if match is None:
            raise ValueError(f"cannot derive a LUT band label for {name!r} from glob {pattern!r}")
        out[name] = match.group(1).upper()
    return out


def lut_for_spec(lut: xr.Dataset, spec: SensorSpec) -> xr.Dataset:
    """Select the LUT bands a sensor spec reads and relabel them with the scene's band names.

    ``spires_inversion.invert`` requires scene and LUT ``band`` coordinates to be
    equal; sensor LUTs from spires-lut carry codes (``B05``) while spires-io scenes
    carry common names (``nir``). Done once on the client before scatter.
    """
    band_map = band_map_from_spec(spec)
    have = set(np.asarray(lut["reflectance"].band.values).astype(str))
    missing = [code for code in band_map.values() if code not in have]
    if missing:
        raise ValueError(f"LUT lacks band(s) {missing} needed by {spec.sensor}; LUT has {sorted(have)}")
    selected = lut["reflectance"].sel(band=list(band_map.values()))
    selected = selected.assign_coords(band=list(band_map.keys()))
    return selected.to_dataset(name="reflectance").assign_attrs(lut.attrs)


def load(scene_dir, *, spec: SensorSpec, bbox=None) -> xr.Dataset:
    """Open one scene; pixels the QA decode rejected become NaN so the solver skips them."""
    scene = open_raster_scene(Path(scene_dir), spec, bbox=bbox)
    masked = scene["reflectance"].where(scene["valid"])
    scene["reflectance"] = masked.astype(np.float32)
    return scene


def align_r0(scene: xr.Dataset, r0: xr.DataArray) -> xr.DataArray:
    """Put an in-memory background on the scene grid with the scene's band labels."""
    target = scene["reflectance"]
    if list(map(str, r0.band.values)) != list(map(str, target.band.values)):
        raise ValueError(
            f"R0 band labels {list(r0.band.values)} differ from scene band labels "
            f"{list(target.band.values)}; relabel the background before running")
    r0 = r0.transpose("y", "x", "band")
    step = float(abs(target.x.values[1] - target.x.values[0])) if target.sizes["x"] > 1 else 1.0
    # Absolute tolerance in metres: rtol on northings near 4e6 m would be ~40 m.
    atol = step * 1e-3
    if r0.sizes["y"] == target.sizes["y"] and r0.sizes["x"] == target.sizes["x"] \
            and np.allclose(r0.y.values, target.y.values, rtol=0, atol=atol) \
            and np.allclose(r0.x.values, target.x.values, rtol=0, atol=atol):
        aligned = r0.assign_coords(y=target.y, x=target.x)
    else:
        try:
            aligned = r0.sel(y=target.y, x=target.x, method="nearest", tolerance=step / 2)
        except KeyError as exc:
            raise ValueError("background grid does not cover the scene grid "
                             "(same CRS and pixel alignment required)") from exc
        aligned = aligned.assign_coords(y=target.y, x=target.x)
    aligned = aligned.astype(np.float32)
    aligned.name = "background_reflectance"
    validate_background_spectra(aligned)
    return aligned


def invert(scene: xr.Dataset, background: xr.DataArray, lut: xr.Dataset, *, n_threads: int = 1,
           **invert_kwargs) -> xr.Dataset:
    """Run ``spires_inversion.invert`` and attach the scene's QA layers and attrs to the results."""
    data = SpiresData(scene=scene, background=background)
    out = spires_inversion.invert(data, lut=lut, n_workers=int(n_threads), **invert_kwargs).results
    out = out.assign(valid=scene["valid"])
    out.attrs.update({k: v for k, v in scene.attrs.items() if k not in out.attrs})
    return out


def invert_scene(scene_dir, *, n_threads: int = 1, lut: xr.Dataset, r0: xr.DataArray,
                 spec: SensorSpec, bbox=None, **invert_kwargs) -> xr.Dataset:
    """The default kernel: load -> align_r0 -> invert for one scene directory."""
    scene = load(scene_dir, spec=spec, bbox=bbox)
    background = align_r0(scene, r0)
    return invert(scene, background, lut, n_threads=n_threads, **invert_kwargs)


def scene_bytes(scene_dir, spec: SensorSpec) -> int:
    """Approximate working-set bytes for one scene, from one band's header only."""
    name = next(iter(spec.band_files))
    with rasterio.open(resolve_one(Path(scene_dir), name, spec.band_files[name])) as src:
        ny, nx = src.height, src.width
    return ny * nx * len(spec.band_files) * 4 * BYTES_FACTOR


@dataclass(frozen=True)
class SceneKernel:
    """A kernel decomposed into the three steps ``run_scene`` needs."""
    name: str
    load: Callable
    align_r0: Callable
    invert: Callable
    whole: Callable          # the plain kernel(unit, *, n_threads, **shared) form


DEFAULT = SceneKernel(name="invert_scene", load=load, align_r0=align_r0, invert=invert, whole=invert_scene)
