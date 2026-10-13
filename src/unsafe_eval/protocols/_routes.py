"""Registry of the APIs that serve a model.

A protocol names a model family; the API that serves it is a separate `route` option,
because the same build answers on several APIs and the choice changes only what the
endpoint accepts, never the prompt, schema or scoring. `Route` carries one capability
flag for each request feature the registered routes disagree about. Route names are
open strings, so an extension can register a gateway.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Route:
    """One API that serves a hosted model.

    `model_api` is Inspect's provider prefix, so a configuration's route can be
    checked against the model string beside it.

    `honours_seed` says whether a request over this route reaches a model that
    accepts a seed. It depends on the model behind the gateway as well as the gateway,
    so a new family declaring an existing route must re-verify it: a dropped seed is
    still recorded as though it took effect.

    `forwards_chat_template_kwargs` says whether the server passes
    `chat_template_kwargs` through to the model's chat template. OpenRouter accepts the
    field and silently discards it, so a protocol that steers its model through the
    template must check this flag and refuse what the route cannot deliver.
    """

    name: str
    model_api: str
    honours_seed: bool
    forwards_chat_template_kwargs: bool = False


_ROUTES: dict[str, Route] = {}


def register_route(route: Route) -> None:
    """Register one API route. Rejects empty and duplicate names."""
    if not route.name or not route.model_api:
        raise ValueError("Route name and model API must not be empty")
    if route.name in _ROUTES:
        raise ValueError(f"Route is already registered: {route.name}")
    _ROUTES[route.name] = route


def route_spec(name: str) -> Route:
    """Return one registered route."""
    try:
        return _ROUTES[name]
    except KeyError as error:
        raise ValueError(f"Unknown model route: {name}") from error


def registered_routes() -> Mapping[str, Route]:
    """Return a read-only snapshot of the registered routes."""
    return MappingProxyType(dict(_ROUTES))


# vLLM applies the chat template itself, so the template's arguments reach the model.
register_route(
    Route(name="vllm", model_api="vllm", honours_seed=True, forwards_chat_template_kwargs=True)
)
# OpenRouter serves the registered families over chat completions, which honours
# `seed`. Sending `chat_template_kwargs` leaves a fixed message's prompt token count
# unchanged, which shows they are dropped.
register_route(Route(name="openrouter", model_api="openrouter", honours_seed=True))
# Inspect routes a GPT-5 family model on the `openai/` provider to the Responses API,
# which discards `seed` with an unsupported-parameter warning.
register_route(Route(name="openai", model_api="openai", honours_seed=False))
# Anthropic's API exposes no seed parameter.
register_route(Route(name="anthropic", model_api="anthropic", honours_seed=False))
# Inspect's `google/` provider, to Vertex AI or the Gemini Developer API. Both APIs take
# a seed, but the provider never sends one.
register_route(Route(name="google", model_api="google", honours_seed=False))
