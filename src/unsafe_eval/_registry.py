"""Expose benchmark and model-protocol components to Inspect's entry-point registry."""

from unsafe_eval import benchmarks
from unsafe_eval._metric_artifacts import MetricsExportHooks
from unsafe_eval._progress import RunProgressHooks
from unsafe_eval.protocols import registered_protocol_components

_COMPONENTS = (
    MetricsExportHooks,
    RunProgressHooks,
    *(getattr(benchmarks, name) for name in benchmarks.__all__),
    *registered_protocol_components(),
)
__all__ = [component.__name__ for component in _COMPONENTS]  # ty: ignore[unresolved-attribute]

if len(__all__) != len(set(__all__)):
    raise RuntimeError("Public Inspect component names must be unique")

globals().update(zip(__all__, _COMPONENTS, strict=True))
