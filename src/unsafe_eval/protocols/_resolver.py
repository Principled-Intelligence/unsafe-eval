"""Dispatch versioned benchmark-to-model protocol construction."""

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from types import MappingProxyType

from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    benchmark_protocol_spec,
)
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol

ProtocolBuilder = Callable[
    [BenchmarkName, BenchmarkProtocolSpec, dict[str, object] | None],
    SafetyProtocol,
]
InspectComponent = Callable[..., object]

_PROTOCOL_BUILDERS: dict[ProtocolName, ProtocolBuilder] = {}
_PROTOCOL_COMPONENTS: dict[ProtocolName, tuple[InspectComponent, ...]] = {}


def register_protocol(
    name: ProtocolName,
    builder: ProtocolBuilder,
    *,
    components: tuple[InspectComponent, ...] = (),
) -> None:
    """Register a model protocol builder and its public Inspect components."""
    if not name:
        raise ValueError("Protocol name must not be empty")
    if name in _PROTOCOL_BUILDERS:
        raise ValueError(f"Protocol is already registered: {name}")
    names = [component.__name__ for component in components]  # ty: ignore[unresolved-attribute]
    taken = {component.__name__ for component in registered_protocol_components()}  # ty: ignore[unresolved-attribute]
    if len(names) != len(set(names)) or taken.intersection(names):
        raise ValueError("Protocol Inspect component names must be unique")
    _PROTOCOL_BUILDERS[name] = builder
    _PROTOCOL_COMPONENTS[name] = components


def registered_protocols() -> Mapping[ProtocolName, ProtocolBuilder]:
    """Return a read-only snapshot of registered protocol builders."""
    return MappingProxyType(dict(_PROTOCOL_BUILDERS))


def registered_protocol_components() -> tuple[InspectComponent, ...]:
    """Return public Inspect components owned by registered model protocols."""
    return tuple(
        component for components in _PROTOCOL_COMPONENTS.values() for component in components
    )


def load_protocol_extensions() -> None:
    """Load installed ``unsafe_eval.protocols`` registration entry points."""
    for entry_point in entry_points(group="unsafe_eval.protocols"):
        extension = entry_point.load()
        if not callable(extension):
            raise TypeError(f"Protocol extension entry point is not callable: {entry_point.name}")
        extension()


def resolve_protocol(
    protocol: ProtocolName,
    *,
    benchmark: BenchmarkName,
    options: dict[str, object] | None = None,
) -> SafetyProtocol:
    """Build and validate one versioned benchmark-to-model protocol."""
    spec = benchmark_protocol_spec(benchmark)
    try:
        builder = _PROTOCOL_BUILDERS[protocol]
    except KeyError as error:
        raise ValueError(f"Unsupported safety protocol: {protocol}") from error
    return builder(benchmark, spec, options)
