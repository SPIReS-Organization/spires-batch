"""slurm_cluster with a fake dask_jobqueue and a fake Client: no SLURM, no scheduler."""
import sys
import types

import pytest

from spires_batch import cluster


class FakeSLURMCluster:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled = None
        self.closed = False
        FakeSLURMCluster.instances.append(self)

    def scale(self, **kwargs):
        self.scaled = kwargs

    def close(self):
        self.closed = True


class FakeClient:
    instances = []
    wait_error = None

    def __init__(self, cluster_obj):
        self.cluster = cluster_obj
        self.waited = None
        self.closed = False
        FakeClient.instances.append(self)

    def wait_for_workers(self, n, timeout=None):
        self.waited = (n, timeout)
        if FakeClient.wait_error is not None:
            raise FakeClient.wait_error

    def close(self):
        self.closed = True


@pytest.fixture
def fake_jobqueue(monkeypatch):
    FakeSLURMCluster.instances = []
    FakeClient.instances = []
    FakeClient.wait_error = None
    module = types.ModuleType("dask_jobqueue")
    module.SLURMCluster = FakeSLURMCluster
    monkeypatch.setitem(sys.modules, "dask_jobqueue", module)
    monkeypatch.setattr(cluster, "Client", FakeClient)
    return module


def test_slurm_cluster_records_kwargs_scales_and_waits(fake_jobqueue):
    client = cluster.slurm_cluster(jobs=4, cores=8, memory="64GB", queue="batch", walltime="02:00:00",
                                   job_extra_directives=["--account=x"])

    (slurm,) = FakeSLURMCluster.instances
    assert slurm.kwargs == {"cores": 8, "memory": "64GB", "queue": "batch", "walltime": "02:00:00",
                            "job_extra_directives": ["--account=x"]}
    assert slurm.scaled == {"jobs": 4}
    assert client.cluster is slurm
    assert client.waited == (1, "600s")
    assert not client.closed and not slurm.closed


def test_slurm_cluster_wait_timeout_is_forwarded(fake_jobqueue):
    cluster.slurm_cluster(jobs=1, cores=1, memory="4GB", queue="q", wait_timeout="30s")
    assert FakeClient.instances[0].waited == (1, "30s")


def test_slurm_cluster_closes_on_wait_timeout(fake_jobqueue):
    FakeClient.wait_error = TimeoutError("no workers")
    with pytest.raises(TimeoutError):
        cluster.slurm_cluster(jobs=2, cores=1, memory="4GB", queue="q")
    client = FakeClient.instances[0]
    assert client.closed
    assert client.cluster.closed


def test_slurm_cluster_missing_dask_jobqueue_has_helpful_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "dask_jobqueue", None)  # import raises ImportError
    with pytest.raises(ImportError, match="pip install dask-jobqueue"):
        cluster.slurm_cluster(jobs=1, cores=1, memory="4GB", queue="q")


def test_cli_backend_selects_cluster(monkeypatch):
    from spires_batch import cli

    calls = {}
    monkeypatch.setattr(cli, "local_cluster", lambda *a, **k: calls.setdefault("local", (a, k)))
    monkeypatch.setattr(cli, "slurm_cluster", lambda **k: calls.setdefault("slurm", k))

    args = cli.build_parser().parse_args(
        ["stack", "--sensor", "hls_l30", "--lut", "l.nc", "--r0", "r.nc", "--workers", "3",
         "--threads", "2", "x"])
    cli.make_client(args)
    assert calls["local"] == ((3, 2), {"memory_limit": None})

    args = cli.build_parser().parse_args(
        ["stack", "--sensor", "hls_l30", "--lut", "l.nc", "--r0", "r.nc", "--backend", "slurm",
         "--slurm-jobs", "4", "--slurm-cores", "8", "--slurm-memory", "64GB",
         "--slurm-queue", "batch", "--slurm-walltime", "02:00:00", "x"])
    cli.make_client(args)
    assert calls["slurm"] == {"jobs": 4, "cores": 8, "memory": "64GB", "queue": "batch", "walltime": "02:00:00"}


def test_cli_slurm_backend_requires_queue(capsys):
    from spires_batch import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["stack", "--sensor", "hls_l30", "--lut", "l.nc", "--r0", "r.nc", "--backend", "slurm", "x"])
    assert exc.value.code == 2
    assert "--slurm-queue" in capsys.readouterr().err
