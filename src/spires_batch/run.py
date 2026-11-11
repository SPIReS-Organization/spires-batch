"""Two entry points, one engine.

``run_stack``: one dask task per unit (scene); each unit inverts on the
worker's local threads. ``run_scene``: one task per pixel block of a single
scene or bbox. Both scatter shared inputs once, record per-unit timing and
failures, and store outputs in the content-addressed workspace. A unit's
failure never aborts a run.
"""
import json
import time
import traceback as _tb
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from distributed import Client, Future, as_completed, get_worker
from distributed.client import FutureCancelledError

from spires_batch import blocks, shared as _shared, workspace
from spires_batch.cluster import worker_memory_limits
from spires_batch.shared import SharedInputLost


class MemoryCheckError(ValueError):
    pass


@dataclass
class Record:
    unit: str
    handle: str
    status: str                                 # "done" | "skipped" | "failed"
    seconds: float = 0.0
    worker: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    result: Any = field(default=None, repr=False, compare=False)   # only kept when store=False

    def to_dict(self) -> dict:
        # not dataclasses.asdict: that deep-copies `result` (a Dataset) before we drop it
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "result"}


def write_records(path, records: list) -> None:
    Path(path).write_text(json.dumps([r.to_dict() for r in records], indent=2))


def _unit_id(unit) -> str:
    return str(unit)


def _run_unit(kernel, unit, shared_kwargs):
    """Executed on a worker. Never raises: failures come back as data."""
    started = time.perf_counter()
    try:
        worker = get_worker()
        address, n_threads = worker.address, worker.state.nthreads
    except ValueError:                     # not inside a worker (synchronous debugging)
        address, n_threads = None, 1
    try:
        result = kernel(unit, n_threads=n_threads, **shared_kwargs)
        return {"ok": True, "result": result, "seconds": time.perf_counter() - started, "worker": address}
    except Exception as exc:               # noqa: BLE001 - the whole point is to record it
        return {"ok": False, "seconds": time.perf_counter() - started, "worker": address,
                "error_type": type(exc).__name__, "error_message": str(exc),
                "traceback": _tb.format_exc()}


def _check_memory(units, unit_bytes, client):
    limits = worker_memory_limits(client)
    smallest = min(limits.values()) if limits else None
    if smallest is None:
        return
    for unit in units:
        need = int(unit_bytes(unit))
        if need > smallest:
            raise MemoryCheckError(
                f"unit {unit!s} needs ~{need / 2**30:.1f} GiB but the smallest worker memory_limit is "
                f"{smallest / 2**30:.1f} GiB; use fewer workers (more memory each) or run_scene")


def _shared_lost(client, shared_futures) -> bool:
    """True if any scattered shared future has no surviving holder. Cancelled counts as lost."""
    futures = [f for f in shared_futures.values() if isinstance(f, Future)]
    if not futures:
        return False
    if any(f.status == "cancelled" for f in futures):
        return True    # who_has raises FuturesCancelledError on these; they are lost by definition
    try:
        who = client.who_has(futures)
    except Exception:  # noqa: BLE001 - if we cannot even ask, the data cannot be assumed present
        return True
    return any(not who.get(f.key) for f in futures)


def run_stack(units, kernel: Callable, shared: dict, client: Client, *,
              unit_bytes: Optional[Callable] = None, replicas: int = 2,
              store: bool = True, records_path=None, params: Optional[dict] = None) -> list:
    """Run ``kernel(unit, n_threads=..., **shared)`` once per unit on the cluster.

    Returns one ``Record`` per unit in submission order. Units whose workspace
    handle already exists are skipped without being submitted; failed units
    write no artifact, so a re-run retries exactly those.
    """
    units = list(units)
    if not units:
        raise ValueError("run_stack called with zero units")
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    _shared.check_shared(shared)
    if unit_bytes is not None:
        _check_memory(units, unit_bytes, client)

    kernel_name = getattr(kernel, "__name__", type(kernel).__name__)
    parents = sorted(f"{k}={v}" for k, v in _shared.digests(shared).items())
    handles = {u: workspace.new_handle("retrieval", kernel_name, dict(params or {}, unit=_unit_id(u)), parents)
               for u in units}

    records = {u: Record(unit=_unit_id(u), handle=handles[u], status="skipped")
               for u in units if store and workspace.exists(handles[u])}
    todo = [u for u in units if u not in records]

    if todo:
        shared_futures = _shared.scatter_shared(client, shared, replicas=replicas)
        futures = {client.submit(_run_unit, kernel, u, shared_futures, pure=False): u for u in todo}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                outcome = future.result()
            except (FutureCancelledError, Exception) as exc:    # worker died mid-task, or dask-level error
                if _shared_lost(client, shared_futures):
                    raise SharedInputLost(
                        "every worker holding a scattered shared input died; scattered data has no recipe "
                        "and cannot be recomputed — restart the run (consider replicas>2)") from exc
                outcome = {"ok": False, "seconds": 0.0, "worker": None, "error_type": type(exc).__name__,
                           "error_message": str(exc), "traceback": _tb.format_exc()}
            if outcome["ok"]:
                try:
                    if store:
                        workspace.store(handles[unit], outcome["result"],
                                        {"unit": _unit_id(unit), "kernel": kernel_name, "parents": parents,
                                         "seconds": outcome["seconds"], "worker": outcome["worker"]})
                except Exception as exc:    # bad result type or disk full: this unit fails, the run goes on
                    records[unit] = Record(unit=_unit_id(unit), handle=handles[unit], status="failed",
                                           seconds=outcome["seconds"], worker=outcome["worker"],
                                           error_type=type(exc).__name__, error_message=str(exc),
                                           traceback=_tb.format_exc())
                else:
                    records[unit] = Record(unit=_unit_id(unit), handle=handles[unit], status="done",
                                           seconds=outcome["seconds"], worker=outcome["worker"],
                                           result=None if store else outcome["result"])
            else:
                records[unit] = Record(unit=_unit_id(unit), handle=handles[unit], status="failed",
                                       seconds=outcome["seconds"], worker=outcome["worker"],
                                       error_type=outcome["error_type"], error_message=outcome["error_message"],
                                       traceback=outcome["traceback"])
            future.release()

    ordered = [records[u] for u in units]
    if records_path is not None:
        write_records(records_path, ordered)
    return ordered


def _invert_block(invert, scene_block, background_block, lut, invert_kwargs):
    """Executed on a worker for one pixel block. One thread: parallelism is across blocks."""
    return invert(scene_block, background_block, lut, n_threads=1, **invert_kwargs)


def run_scene(unit, shared: dict, client: Client, *, bbox=None, chunk_pixels: int = 512,
              kernel=None, replicas: int = 2, store: bool = True, params: Optional[dict] = None) -> Record:
    """Invert one scene (or a bbox of it) block-parallel across all workers.

    ``kernel.load`` and ``kernel.align_r0`` run on the client; the scene and
    background are cut into ``chunk_pixels`` squares, scattered off-graph, and
    ``kernel.invert`` runs per block with one thread. Output is bit-identical to
    ``kernel.whole`` on the same scene because every pixel is an independent
    solve from the same initial guess. ``shared`` must contain ``lut``, ``r0``
    and ``spec``; any other keys are forwarded to ``kernel.invert``.
    """
    from spires_batch import kernels as _kernels
    kernel = kernel or _kernels.DEFAULT
    _shared.check_shared(shared)
    required = {"lut", "r0", "spec"}
    if not required <= set(shared):
        raise ValueError(f"run_scene needs shared keys {sorted(required)}; got {sorted(shared)}")
    invert_kwargs = {k: v for k, v in shared.items() if k not in required}

    parents = sorted(f"{k}={v}" for k, v in _shared.digests(shared).items())
    handle = workspace.new_handle("retrieval", kernel.name,
                                  dict(params or {}, unit=_unit_id(unit), bbox=list(bbox) if bbox else None),
                                  parents)
    if store and workspace.exists(handle):
        return Record(unit=_unit_id(unit), handle=handle, status="skipped")

    started = time.perf_counter()
    try:
        scene = kernel.load(unit, spec=shared["spec"], bbox=bbox)
        background = kernel.align_r0(scene, shared["r0"])
    except Exception as exc:   # noqa: BLE001
        return Record(unit=_unit_id(unit), handle=handle, status="failed", seconds=time.perf_counter() - started,
                      error_type=type(exc).__name__, error_message=str(exc), traceback=_tb.format_exc())

    pairs = blocks.cut_pair(scene, background, chunk_pixels)
    n_blocks = len(pairs)
    if _shared._nbytes(shared["lut"]) >= _shared.SCATTER_THRESHOLD_BYTES:
        lut_future = _shared.scatter_shared(client, {"lut": shared["lut"]}, replicas=replicas)["lut"]
    else:
        lut_future = shared["lut"]
    futures = scene_futures = bg_futures = []
    try:
        scene_futures = blocks.scatter(client, [s for s, _ in pairs], replicas=replicas)
        bg_futures = blocks.scatter(client, [b for _, b in pairs], replicas=replicas)
        futures = [client.submit(_invert_block, kernel.invert, s, b, lut_future, invert_kwargs, pure=False)
                   for s, b in zip(scene_futures, bg_futures)]
        del scene, background, pairs   # the scattered copies are now the only ones; free the client's
        parts = client.gather(futures)
    except Exception as exc:   # noqa: BLE001 - a scatter or block failure is this unit's failure
        if _shared_lost(client, {"lut": lut_future}) or any(f.status == "cancelled" for f in scene_futures):
            raise SharedInputLost("scattered scene blocks or LUT were lost with their workers; "
                                  "restart the run") from exc
        return Record(unit=_unit_id(unit), handle=handle, status="failed", seconds=time.perf_counter() - started,
                      error_type=type(exc).__name__, error_message=str(exc), traceback=_tb.format_exc())
    finally:
        for f in futures + scene_futures + bg_futures:
            f.release()

    result = blocks.reassemble(parts)
    seconds = time.perf_counter() - started
    if store:
        try:
            workspace.store(handle, result, {"unit": _unit_id(unit), "kernel": kernel.name, "parents": parents,
                                             "bbox": list(bbox) if bbox else None, "chunk_pixels": chunk_pixels,
                                             "blocks": n_blocks, "seconds": seconds})
        except Exception as exc:   # noqa: BLE001
            return Record(unit=_unit_id(unit), handle=handle, status="failed", seconds=seconds,
                          error_type=type(exc).__name__, error_message=str(exc), traceback=_tb.format_exc())
    return Record(unit=_unit_id(unit), handle=handle, status="done", seconds=seconds,
                  worker=None, result=None if store else result)
