"""Inspect model provider for CLM, Contrastive-LM's System One model.

CLM answers System One questions without generating text: two projection heads score each
option against the state over Qwen3-8B embeddings. This provider runs the heads in-process
with the `contrastive-lm` package, the code behind `clm-serve`, and runs the protocol's Jev
questions unchanged.

* The encoder is a pooling vLLM server, started through Inspect's own `vllm` provider at the
  pinned `encoder_revision`, with CLM's reference settings.
* The heads come from the pinned `revision` of the model repository, since `contrastive-lm`
  downloads them from `main`.
* `contrastive-lm` and `vllm` are imported only when the model is constructed, because
  `protocols/_builtins` imports this module on every install.
"""

import atexit
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from unsafe_eval.providers._batching import Batcher
from unsafe_eval.providers._pins import commit
from unsafe_eval.providers.d1 import model_output_from_response
from unsafe_eval.providers.typesafe import questions_from_messages, state_from_messages

ENCODER = "Qwen/Qwen3-8B"
HEADS_FILE = "CLM_v0.1-8B.pt"


class CLMAPI(ModelAPI):
    """Answer each conversation's question with pinned CLM heads over a local Qwen3-8B."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        revision: str | None = None,
        encoder_revision: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        revision = commit("CLM", "revision", revision)
        encoder_revision = commit("CLM", "encoder_revision", encoder_revision)
        self._metadata = {
            "served_model": f"{model_name}@{revision}",
            "encoder": f"{ENCODER}@{encoder_revision}",
        }
        self._engine = self._load(model_name, revision, encoder_revision)
        self._batcher = Batcher(self._answer)

    @staticmethod
    def _load(model_name: str, revision: str, encoder_revision: str) -> Any:
        try:
            from clm import Embedder, Engine
        except ImportError as error:
            raise ImportError(
                "The clm provider requires the `clm` extra: "
                "uv sync --extra clm (or pip install 'unsafe-eval[clm]')"
            ) from error

        heads = hf_hub_download(model_name, HEADS_FILE, revision=revision)
        # CLM's reference encoder server: last-token pooling over at most 2,048 tokens.
        encoder = get_model(
            f"vllm/{ENCODER}",
            revision=encoder_revision,
            lazy_init=False,
            runner="pooling",
            max_model_len=2048,
            enable_prefix_caching=True,
        ).api
        # Inspect closes only the models it runs, so the encoder server is stopped here.
        atexit.register(encoder.close)
        embedder = Embedder(
            url=f"{encoder.base_url}/embeddings", model=ENCODER, api_key=encoder.api_key
        )
        return Engine(embedder=embedder, checkpoint=heads)

    def _answer(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # ponytail: one engine call per request; embed a batch's states together if
        # concurrency above 1 ever matters.
        return [self._engine.answer(r["state"], r["questions"]) for r in requests]

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Answer the conversation's one question about its state."""
        request = {"state": state_from_messages(input), "questions": questions_from_messages(input)}
        response, seconds = await self._batcher.submit(request)
        output = model_output_from_response(
            response, model_name=self.model_name, metadata=self._metadata
        )
        # Inspect times a call as its `ModelCall` says, so the wait for the device is left out.
        return output, ModelCall.create(request, response, time=seconds)


@modelapi(name="clm")
def clm() -> type[ModelAPI]:
    """Register the CLM provider with Inspect."""
    return CLMAPI
