"""The one place that knows a dask cluster exists at the cluster level."""
import os

from distributed import Client, LocalCluster


def total_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def local_cluster(workers: int, threads_per_worker: int, memory_limit=None, **kwargs) -> Client:
    """``workers x threads_per_worker`` LocalCluster, always multi-process.

    ``processes=True`` so a crashed inversion kills one worker, not the run.
    ``memory_limit`` defaults to ``total_ram / workers``. No nested clusters
    and no per-task Client: the C++ inversion releases the GIL, so threads
    inside a worker genuinely run inversions concurrently.
    """
    if workers < 1 or threads_per_worker < 1:
        raise ValueError("workers and threads_per_worker must be >= 1")
    if memory_limit is None:
        memory_limit = total_memory_bytes() // workers
    kwargs.setdefault("dashboard_address", None)
    cluster = LocalCluster(n_workers=workers, threads_per_worker=threads_per_worker,
                           processes=True, memory_limit=memory_limit, **kwargs)
    return Client(cluster)


def worker_memory_limits(client: Client) -> dict[str, int]:
    info = client.scheduler_info()["workers"]
    return {addr: int(w["memory_limit"]) for addr, w in info.items()}
