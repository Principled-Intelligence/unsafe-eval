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
    """Register a builder under a unique, non-empty protocol identifier.

    The builder receives a benchmark identifier, its `BenchmarkProtocolSpec`, and the
    supplied options, and returns a `SafetyProtocol`. `components` lists the public
    Inspect solvers and scorers exposed through this package's entry-point registry.
    Component function names must be unique across all registered protocols.

    Raise `ValueError` for an empty or duplicate protocol name or a reused component name.
    """
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
    """Call each installed entry point in the `unsafe_eval.protocols` group.

    Each entry point must load a zero-argument registration callable. Raise `TypeError`
    for a non-callable entry point; loading and registration errors propagate unchanged.
    """
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
    """Build the registered protocol for a benchmark using its evaluation semantics.

    `options=None` lets the builder apply its defaults; some protocols still require
    options such as `route`. The builder validates options and benchmark compatibility.
    Unknown benchmark or protocol identifiers raise `ValueError`.
    """
    spec = benchmark_protocol_spec(benchmark)
    try:
        builder = _PROTOCOL_BUILDERS[protocol]
    except KeyError as error:
        raise ValueError(f"Unsupported safety protocol: {protocol}") from error
    return builder(benchmark, spec, options)
