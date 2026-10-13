"""Versioned benchmark-to-model safety protocols."""

from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkMetricBuilder,
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
    register_benchmark,
    registered_benchmarks,
)
from unsafe_eval.protocols._options import protocol_options, route_option
from unsafe_eval.protocols._resolver import (
    InspectComponent,
    ProtocolBuilder,
    load_protocol_extensions,
    register_protocol,
    registered_protocol_components,
    registered_protocols,
    resolve_protocol,
)
from unsafe_eval.protocols._resource_manifest import (
    PromptResource,
    adaptation_resource,
    prompt_resources,
    register_prompt_resources,
)
from unsafe_eval.protocols._routes import (
    Route,
    register_route,
    registered_routes,
    route_spec,
)
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol

__all__ = [
    "BenchmarkMetricBuilder",
    "BenchmarkName",
    "BenchmarkProtocolSpec",
    "InspectComponent",
    "PromptResource",
    "ProtocolBuilder",
    "ProtocolName",
    "Route",
    "SafetyProtocol",
    "adaptation_resource",
    "build_protocol_metrics",
    "load_protocol_extensions",
    "prompt_resources",
    "protocol_options",
    "register_benchmark",
    "register_prompt_resources",
    "register_protocol",
    "register_route",
    "registered_benchmarks",
    "registered_protocol_components",
    "registered_protocols",
    "registered_routes",
    "resolve_protocol",
    "route_option",
    "route_spec",
]


def _initialize_protocols() -> None:
    # Imported late because protocol modules import the public names above.
    from unsafe_eval.protocols._builtins import register_builtin_protocols

    register_builtin_protocols()


_initialize_protocols()
load_protocol_extensions()
