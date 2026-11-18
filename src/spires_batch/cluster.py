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


def slurm_cluster(jobs: int, cores: int, memory: str, queue: str, walltime: str = "01:00:00",
                  wait_timeout: str = "600s", **kw) -> Client:
    """``SLURMCluster`` from dask-jobqueue, scaled to ``jobs`` SLURM jobs, returned as a Client.

    ``cores``, ``memory``, ``queue`` and ``walltime`` describe one SLURM job and are
    passed to ``SLURMCluster``; extra keyword arguments go to it unchanged. Blocks
    until one worker has connected or ``wait_timeout`` elapses. On that timeout the
    client and cluster are closed, which cancels the submitted jobs.

    Untested on the development host, which is not a SLURM client. It is provided so
    the same stack runs under SLURM via dask-jobqueue (``pip install spires-batch[slurm]``).
    """
    try:
        from dask_jobqueue import SLURMCluster
    except ImportError as exc:
        raise ImportError(
            "slurm_cluster needs dask-jobqueue; install it with `pip install dask-jobqueue` "
            "or `pip install spires-batch[slurm]`") from exc
    cluster = SLURMCluster(cores=cores, memory=memory, queue=queue, walltime=walltime, **kw)
    cluster.scale(jobs=jobs)
    client = Client(cluster)
    try:
        client.wait_for_workers(1, timeout=wait_timeout)
    except BaseException:
        client.close()
        cluster.close()
        raise
    return client


def worker_memory_limits(client: Client) -> dict[str, int]:
    info = client.scheduler_info()["workers"]
    return {addr: int(w["memory_limit"]) for addr, w in info.items()}
