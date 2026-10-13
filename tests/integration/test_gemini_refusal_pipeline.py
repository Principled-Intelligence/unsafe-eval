"""End-to-end behaviour when OpenRouter relays a Gemini block inside a successful response.

OpenRouter can relay a Vertex AI block as a failed choice, `finish_reason: "error"`
with `native_finish_reason: "SAFETY"`, which Inspect reduces to an empty completion with
an `unknown` stop. Only the raw response on the model event names the block.
"""

from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval
from inspect_ai.dataset import Sample
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

from tests.integration._support import configure_test_environment, task_with_dataset

_GROUPS = {"functional_category": "standard", "semantic_category": "chemical_biological"}

SAMPLES = [
    Sample(input="refused-example", target="yes", metadata=dict(_GROUPS)),
    Sample(input="unsafe-example", target="yes", metadata=dict(_GROUPS)),
]


def _verdicts(log: Any) -> list[str]:
    assert log.status == "success", log.error
    return sorted(
        (sample.scores or {})["gemini_safety"].metadata["predicted_verdict"]
        for sample in (log.samples or [])
    )


class _RelayingGateway(ModelAPI):
    """Answers like OpenRouter relaying a Vertex AI block as a failed choice.

    The refused prompt comes back with an empty completion and Inspect's `unknown`
    stop reason. Only the raw response, recorded on the model event, names the block.
    The model name is the native finish reason the gateway relays.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None,
        api_key: str | None,
        config: GenerateConfig,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        self.native_finish_reason = model_name

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        if "refused-example" in input[-1].text:
            output = ModelOutput.from_content(
                model="relaying-gateway", content="", stop_reason="unknown"
            )
            choice = {
                "finish_reason": "error",
                "native_finish_reason": self.native_finish_reason,
            }
        else:
            output = ModelOutput.from_content(
                model="relaying-gateway",
                content='{"reasoning":"mock","user_safety":"unsafe"}',
            )
            choice = {"finish_reason": "stop", "native_finish_reason": "STOP"}
        return output, ModelCall.create(request={}, response={"choices": [choice]})


@modelapi(name="test-relaying-gateway")
def _relaying_gateway() -> type[ModelAPI]:
    return _RelayingGateway


def _verdicts_through_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_finish_reason: str
) -> list[str]:
    configure_test_environment(tmp_path, monkeypatch, f"gemini-relay-{native_finish_reason}")
    task = task_with_dataset(
        module_name="harmbench_prompt_safety",
        loader_name="load_harmbench_prompt_safety",
        task_name="harmbench_prompt_safety",
        samples=SAMPLES,
        protocol="gemini",
        protocol_options={"route": "openrouter"},
        monkeypatch=monkeypatch,
    )
    log = eval(
        task,
        model=get_model(f"test-relaying-gateway/{native_finish_reason}"),
        log_dir=str(tmp_path),
        display="none",
        ctl_server=False,
    )[0]
    return _verdicts(log)


def test_a_block_relayed_as_a_failed_choice_is_scored_as_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verdicts = _verdicts_through_gateway(tmp_path, monkeypatch, "SAFETY")

    assert verdicts == ["refused", "yes"]
