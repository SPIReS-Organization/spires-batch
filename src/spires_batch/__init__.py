"""spires-batch: run the SPIReS retrieval at scale over stacks of granules."""
from spires_batch.cluster import local_cluster
from spires_batch.kernels import DEFAULT, SceneKernel, invert_scene, lut_for_spec
from spires_batch.run import Record, MemoryCheckError, run_scene, run_stack, write_records
from spires_batch.shared import SharedInputLost

__all__ = ["local_cluster", "DEFAULT", "SceneKernel", "invert_scene", "lut_for_spec",
           "Record", "MemoryCheckError", "run_scene", "run_stack", "write_records", "SharedInputLost"]

try:
    from importlib.metadata import version as _version
    __version__ = _version("spires-batch")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
