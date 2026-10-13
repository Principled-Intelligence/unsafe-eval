"""The registry of the APIs that serve a hosted model."""

import pytest

from unsafe_eval.protocols import (
    Route,
    _routes,
    register_route,
    registered_routes,
    route_option,
    route_spec,
)


def test_a_custom_route_can_register_without_changing_dispatch_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_routes, "_ROUTES", dict(registered_routes()))
    register_route(Route(name="test-gateway", model_api="test", honours_seed=False))

    assert route_spec("test-gateway").model_api == "test"


def test_route_registration_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_route(Route(name="", model_api="test", honours_seed=False))


def test_route_registration_rejects_a_duplicate() -> None:
    with pytest.raises(ValueError, match="already registered: openrouter"):
        register_route(Route(name="openrouter", model_api="openrouter", honours_seed=True))


SUPPORTED = ("openrouter", "openai")


def test_the_route_option_must_be_a_string() -> None:
    with pytest.raises(ValueError, match="'route' must be a string"):
        route_option({"route": None}, SUPPORTED)


def test_an_unregistered_route_is_rejected_as_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model route: bedrock"):
        route_option({"route": "bedrock"}, SUPPORTED)
