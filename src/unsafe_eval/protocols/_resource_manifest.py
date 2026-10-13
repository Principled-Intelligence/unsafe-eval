"""Immutable manifest of benchmark-to-protocol prompt resources, and their loading."""

import re
import warnings
from dataclasses import dataclass

from unsafe_eval._resources import content_sha256, load_text_resource
from unsafe_eval.protocols._benchmark_specs import BenchmarkName
from unsafe_eval.protocols._types import ProtocolName

PromptResourcePairing = tuple[BenchmarkName, ProtocolName]


@dataclass(frozen=True)
class PromptResource:
    """One packaged prompt resource and its reproducibility requirements."""

    filename: str
    sha256: str
    allow_empty: bool = False
    package: str = "unsafe_eval"
    path: tuple[str, ...] | None = None


PromptResources = tuple[PromptResource, ...]

_PROMPT_RESOURCE_MANIFEST: dict[PromptResourcePairing, PromptResources] = {}

# SHA-256 of each built-in `prompts/<benchmark>/<protocol>/<filename>`.
_BUILTIN_DIGESTS: dict[tuple[ProtocolName, str], dict[BenchmarkName, str]] = {
    ("shieldstral-1.0-3b", "instruction.txt"): {
        "aegis-2.0": "4ab93fde29c227ab5b5aa7c9fca49487230abe7b9810f192f51c9929bf23d354",
        "cosapien": "d50a6a390fa10bb3b9b1a8433f4f7e9df7259dec37a79a3c3b2b282afafc0815",
        "dynaguardrail": "a4ecba89048d38d86c7ad4e5edfa642c58349c1071145feb377b62045e3ce695",
        "harmbench-text-test": "6a589e401ec978cd1e0a8f5e32f7af8fa9babab4b441b9899af817496c52fcc4",
        "openai-moderation": "7b734d48ff1a154d086f3491a4f0bd6991fdec45de8fc7f91392dce5a3bb60ab",
        "simple-safety-tests": "2b0637733291f37bc93dbb7054e6ee2ecf2af0b4b0e2481d85b87170e5954f91",
        "toxic-chat-0124": "8f0ac15c23f0e7103da857f93acc38afc027d6f7326eb4a3edf1bbb63bdd4ebb",
        "wildguardmix-test": "a9973c97901cca61400b0c7dd722e46ec8cfafd36dc6819c4d16763664f12e37",
    },
    ("shieldstral-1.0-3b", "query.txt"): {
        "aegis-2.0": "07e5c9828c42521aafab4c3b4eefe8e12a78a6e33a9062b872de781c95e14025",
        "cosapien": "039ffc65d633e95562d61d65ed29d3cca0acd7e0aee7c9911c78657021a93b9b",
        "dynaguardrail": "5f5139a33212eb1d2159a2992546450f21b3074081c7a12c18efb3f1f403f3b9",
        "harmbench-text-test": "b34071916e2aeca428a8a4b26fa23539d2c1b1ec7fcc02609954bb2274cbefa7",
        "openai-moderation": "474cad508bc3feb8b5567888c184f2eaebde5dae5ed2bb92689478db9d021975",
        "simple-safety-tests": "b8df2fc50b37e7522c8a2386c829a41f0167419cf215fb696b4f88ea59e78367",
        "toxic-chat-0124": "d41c567c13463a0a74f92f86e6803779b3e0e98c699e5f98406b58b12b8ca37f",
        "wildguardmix-test": "3b2c092a55ff4e20aaa928bda34276b0640a8604738ecc1740905f3b2dd6d12f",
    },
    ("scope-guard-2601", "ai-service-description.txt"): {
        "aegis-2.0": "ea4192774397327d159c1e659be703a28a73055ba3114bfd02e244b8cba47d3c",
        "cosapien": "3dfe8bccc3d095323b64b9163ad4a69987e45cad023cf2ef3a1f1d01c48410e9",
        "dynaguardrail": "3dfe8bccc3d095323b64b9163ad4a69987e45cad023cf2ef3a1f1d01c48410e9",
        "harmbench-text-test": "bd67911ed3248391d41c85febe3bd864f4e74c9f71299dabae56b0a201f0580b",
        "openai-moderation": "97a4517b047a81b0e70e7a7676fe411034d2a4c063ac0a749183a8804bcdce31",
        "simple-safety-tests": "de6fef72157728e3dfa8d56501abeada083c5c80000a1482ef9496ce1da7ab94",
        "toxic-chat-0124": "ab30899685f8b662dbf4af96856fb86c68b8711c90c8d23d1438cf8bf404ab8d",
        "wildguardmix-test": "29829fa63540b0e09fa767d83994e5ead171eefc06267f76e580b04d3e92572e",
    },
    ("scope-guard-2609", "ai-service-description.txt"): {
        "aegis-2.0": "ea4192774397327d159c1e659be703a28a73055ba3114bfd02e244b8cba47d3c",
        "cosapien": "3dfe8bccc3d095323b64b9163ad4a69987e45cad023cf2ef3a1f1d01c48410e9",
        "dynaguardrail": "3dfe8bccc3d095323b64b9163ad4a69987e45cad023cf2ef3a1f1d01c48410e9",
        "harmbench-text-test": "bd67911ed3248391d41c85febe3bd864f4e74c9f71299dabae56b0a201f0580b",
        "openai-moderation": "97a4517b047a81b0e70e7a7676fe411034d2a4c063ac0a749183a8804bcdce31",
        "simple-safety-tests": "de6fef72157728e3dfa8d56501abeada083c5c80000a1482ef9496ce1da7ab94",
        "toxic-chat-0124": "ab30899685f8b662dbf4af96856fb86c68b8711c90c8d23d1438cf8bf404ab8d",
        "wildguardmix-test": "29829fa63540b0e09fa767d83994e5ead171eefc06267f76e580b04d3e92572e",
    },
    ("nemotron-3.5-content-safety", "custom-policy.txt"): {
        "cosapien": "3d10d0056a9d5cbc436b3bbf7ab5150f26cd88fc1966f625a97f6e6d58b092e2",
        "dynaguardrail": "69b5d187c96b2e1b8c99db6df6a1553b0c530105fe8a3913c55396d62acf073f",
    },
    # The hosted-family policies and System One questions name no benchmark, dataset or
    # annotation process: a model that saw the public set would recall its labels
    # instead of applying the policy.
    ("gpt", "instruction.txt"): {
        "aegis-2.0": "878dfeb6f65b6dd12431b52884bea3cae766f0197ec477d6e3a94b4a8f213c73",
        "cosapien": "140f1d53b54adf9a7c0aa7f2484b6329a4db93e3f69d57cdfb297f7dcf0b2a0d",
        "dynaguardrail": "0b31aaf33eb6c2cdc6aa278f5b5a3b3eb9220b1a06bc797af10b4b12d581e102",
        "harmbench-text-test": "a72a5ed7d9b7da0d5ba589d62f401664c74480629a6afc0b6fea8233afb351ce",
        "openai-moderation": "f2d7fde6f1081237b1e672c8e70a29e4706f9f5bc56848241d4ad11bb60a42bd",
        "simple-safety-tests": "a6478598cd24b1fc12c29b9e2894d137ae6647ad98ff3822bcf64e96bc0956a6",
        "toxic-chat-0124": "9e1c61289c47ea7438cb96d4872fbf1ffd5fd976117b45749cd598936d1c27a2",
        "wildguardmix-test": "f7634745c09cfb5fdff766cd78ba08c2b4a0d0b32ac193a8f733c047b1e45a2c",
    },
    ("claude", "instruction.txt"): {
        "aegis-2.0": "878dfeb6f65b6dd12431b52884bea3cae766f0197ec477d6e3a94b4a8f213c73",
        "cosapien": "140f1d53b54adf9a7c0aa7f2484b6329a4db93e3f69d57cdfb297f7dcf0b2a0d",
        "dynaguardrail": "0b31aaf33eb6c2cdc6aa278f5b5a3b3eb9220b1a06bc797af10b4b12d581e102",
        "harmbench-text-test": "a72a5ed7d9b7da0d5ba589d62f401664c74480629a6afc0b6fea8233afb351ce",
        "openai-moderation": "f2d7fde6f1081237b1e672c8e70a29e4706f9f5bc56848241d4ad11bb60a42bd",
        "simple-safety-tests": "a6478598cd24b1fc12c29b9e2894d137ae6647ad98ff3822bcf64e96bc0956a6",
        "toxic-chat-0124": "9e1c61289c47ea7438cb96d4872fbf1ffd5fd976117b45749cd598936d1c27a2",
        "wildguardmix-test": "f7634745c09cfb5fdff766cd78ba08c2b4a0d0b32ac193a8f733c047b1e45a2c",
    },
    ("gemini", "instruction.txt"): {
        "aegis-2.0": "878dfeb6f65b6dd12431b52884bea3cae766f0197ec477d6e3a94b4a8f213c73",
        "cosapien": "140f1d53b54adf9a7c0aa7f2484b6329a4db93e3f69d57cdfb297f7dcf0b2a0d",
        "dynaguardrail": "0b31aaf33eb6c2cdc6aa278f5b5a3b3eb9220b1a06bc797af10b4b12d581e102",
        "harmbench-text-test": "a72a5ed7d9b7da0d5ba589d62f401664c74480629a6afc0b6fea8233afb351ce",
        "openai-moderation": "f2d7fde6f1081237b1e672c8e70a29e4706f9f5bc56848241d4ad11bb60a42bd",
        "simple-safety-tests": "a6478598cd24b1fc12c29b9e2894d137ae6647ad98ff3822bcf64e96bc0956a6",
        "toxic-chat-0124": "9e1c61289c47ea7438cb96d4872fbf1ffd5fd976117b45749cd598936d1c27a2",
        "wildguardmix-test": "f7634745c09cfb5fdff766cd78ba08c2b4a0d0b32ac193a8f733c047b1e45a2c",
    },
    ("jev", "questions.json"): {
        "aegis-2.0": "2447c3aa6a92a6e30bc5d9733f8016d6e7df74899a43ec7868fcb4594efa9f74",
        "cosapien": "b9476780e39ec2b9b01a118d867fd72f9eaf5b9858b151c83ab6511d377c42ec",
        "dynaguardrail": "fc865478cb5db15a726bdea78422211295397807b141d41a17c08c8ece45b9a7",
        "harmbench-text-test": "a02c0befa7b6131576b85479b5a1faab1bcc5245d95e70d82e8721c3a199af73",
        "openai-moderation": "b9335f306f6de4fc734876158f75cceb23c5903497c114852d133460e1b59431",
        "simple-safety-tests": "06b52c10ba06cc3668ec20752911d9d946ca58ede3d29ff2ed22e1d38eb66898",
        "toxic-chat-0124": "f02a8c4c13661f58058db00621486171ddb92672f538c1d35478dc864f3f3e28",
        "wildguardmix-test": "42a23a9ee5635a6bf08cb0abbae705fa12fa02cb3f695b7c3a905385c484fb4a",
    },
    ("laya", "questions.json"): {
        "aegis-2.0": "58299e2b9120dfb4b172eb2828757f3bad8065b9cad4b0b80e1bd2dbe28f9778",
        "cosapien": "120e890f40db9cb180fb91d8545607487ae6225485ef1ce5012682804e76822e",
        "dynaguardrail": "4be78de1b516ad775f0b7515c6ee95710e44aa0397fd18450d2b36b2fb26fabc",
        "harmbench-text-test": "d77743eeadbc0af9842146637ac7d378b96a8fff2f2b500f913850a0ca788f56",
        "openai-moderation": "cde974b173a62fb7ae89cd2f9ff96051d77821500dde1179114413d633c22c65",
        "simple-safety-tests": "1130b91b4fb7a6b434a8a370dd3122a877bb957f9a71b7902b02dd1f2bc0ca21",
        "toxic-chat-0124": "02149d55962389b2971c5d7f65b1ecb19b08d1c7ba412ac17207455d0b4a87f5",
        "wildguardmix-test": "321e65d262e529b43506ab3efe986aeb3d9008956f0a2cb458423cce578c27ce",
    },
    ("d1", "questions.json"): {
        "aegis-2.0": "c5b6b3c1a2543036a9980c93e17bf5a0e5078ac38e8bc7e7789491c5108f07d9",
        "cosapien": "192dccac70f8fecfdaca51a6affb520320538c0e530aa7ddb63e46476e77077b",
        "dynaguardrail": "c5b6f20534a25d872d8150f6add3cad58c4f539d31225904206777ed9ec69f99",
        "harmbench-text-test": "66d1bac1b3b95bc00f4bedde4f8893532f32d9328093d2e0619c2143a4ed7342",
        "openai-moderation": "3fb6b5f6e04dee57fc0fdb81dd4ae165c2b6ca326ab848e6af6d92c288b7a718",
        "simple-safety-tests": "7cfbd3f2676548297d0941001388eb0e17d0713285e2130e7773ca135487f0b8",
        "toxic-chat-0124": "0bd9d3a0ba548cff3a610e3b2b33fd6ff2b25fcc2c3255b672c086aec9308468",
        "wildguardmix-test": "6d16eb2e0c74b5b48b6e62ff1c0705c0bef92ce80ba3f2577464272adaa90b17",
    },
    ("clef", "questions.json"): {
        "aegis-2.0": "c5b6b3c1a2543036a9980c93e17bf5a0e5078ac38e8bc7e7789491c5108f07d9",
        "cosapien": "192dccac70f8fecfdaca51a6affb520320538c0e530aa7ddb63e46476e77077b",
        "dynaguardrail": "c5b6f20534a25d872d8150f6add3cad58c4f539d31225904206777ed9ec69f99",
        "harmbench-text-test": "66d1bac1b3b95bc00f4bedde4f8893532f32d9328093d2e0619c2143a4ed7342",
        "openai-moderation": "3fb6b5f6e04dee57fc0fdb81dd4ae165c2b6ca326ab848e6af6d92c288b7a718",
        "simple-safety-tests": "7cfbd3f2676548297d0941001388eb0e17d0713285e2130e7773ca135487f0b8",
        "toxic-chat-0124": "0bd9d3a0ba548cff3a610e3b2b33fd6ff2b25fcc2c3255b672c086aec9308468",
        "wildguardmix-test": "6d16eb2e0c74b5b48b6e62ff1c0705c0bef92ce80ba3f2577464272adaa90b17",
    },
    ("gliner2.5-decide", "schema.json"): {
        "aegis-2.0": "00420970da7c098bbe04f2bad173d711ce0a3a3fd234654197062eefbdf41b39",
        "cosapien": "d7ed7ae16e14392985ea9cd8f7a95a54a6a42864e046e0843e34eba5e5220118",
        "dynaguardrail": "bd4521f0319e80c3a692586f458aaa4c37c423d93e71f2614151ff67dd964252",
        "harmbench-text-test": "b10698cca0d981a4cae407aca6e88974d0dbfc16d0eef9e42a7fa12d27d407e8",
        "openai-moderation": "3c8bf4a6862394401e422f5be160b02c6e83edfc51cfadd9ee33ed9d0a2aa1e9",
        "simple-safety-tests": "9ccf3e2d2a9c4d4aaf8db74fb3783a94a7664de0338931738855cb37dc6c89cd",
        "toxic-chat-0124": "8a8d9bec24f1dc717cf7e5d6ed802b676cb30e2fcc5e089831fd4ad7e7f0dd61",
        "wildguardmix-test": "f29f3c50d69c9f22a4019ed1ee14e1983fed57d79afe967264f6616b78907026",
    },
}


def register_prompt_resources(
    benchmark: BenchmarkName,
    protocol: ProtocolName,
    resources: PromptResources,
) -> None:
    """Register the immutable prompt resources for one supported pairing."""
    pairing = benchmark, protocol
    if not benchmark or not protocol:
        raise ValueError("Prompt resource benchmark and protocol names must not be empty")
    if pairing in _PROMPT_RESOURCE_MANIFEST:
        raise ValueError(
            f"Prompt resources are already registered for pairing: {benchmark}/{protocol}"
        )
    filenames = [resource.filename for resource in resources]
    if not resources or any(not filename for filename in filenames):
        raise ValueError("Prompt resource registrations must contain named resources")
    if len(filenames) != len(set(filenames)):
        raise ValueError("Prompt resource filenames must be unique within a pairing")
    for resource in resources:
        if not re.fullmatch(r"[0-9a-f]{64}", resource.sha256):
            raise ValueError(f"Prompt resource has an invalid SHA-256: {resource.filename}")
        if not resource.package:
            raise ValueError(f"Prompt resource has an empty package: {resource.filename}")
        if resource.path is not None and (
            not resource.path
            or any(
                not part or part in {".", ".."} or "/" in part or "\\" in part
                for part in resource.path
            )
        ):
            raise ValueError(f"Prompt resource has an invalid package path: {resource.filename}")
    _PROMPT_RESOURCE_MANIFEST[pairing] = resources


def prompt_resources(
    benchmark: BenchmarkName,
    protocol: ProtocolName,
) -> PromptResources:
    """Return declared prompt resources for one pairing."""
    try:
        return _PROMPT_RESOURCE_MANIFEST[(benchmark, protocol)]
    except KeyError as error:
        raise ValueError(
            f"No prompt resources declared for pairing: {benchmark}/{protocol}"
        ) from error


def adaptation_resource(
    benchmark: BenchmarkName,
    protocol: ProtocolName,
    filename: str,
    *,
    validate_hash: bool = True,
) -> tuple[str, dict[str, str]]:
    """Load and validate one declared adaptation resource."""
    resources = prompt_resources(benchmark, protocol)
    resource = next(
        (resource for resource in resources if resource.filename == filename),
        None,
    )
    if resource is None:
        raise ValueError(f"Undeclared prompt resource: {benchmark}/{protocol}/{filename}")

    path = resource.path or ("protocols", "prompts", benchmark, protocol, filename)
    content = load_text_resource(*path, package=resource.package)
    if not content and not resource.allow_empty:
        raise ValueError(f"Prompt resource must not be empty: {benchmark}/{protocol}/{filename}")

    actual_hash = content_sha256(content)
    if actual_hash != resource.sha256:
        message = (
            f"Prompt resource hash mismatch: {benchmark}/{protocol}/{filename}; "
            f"expected {resource.sha256}, got {actual_hash}"
        )
        if validate_hash:
            raise ValueError(message)
        warnings.warn(
            f"{message}; continuing because prompt resource hash validation is disabled",
            stacklevel=2,
        )
    return content, {f"protocol/{filename}": actual_hash}


def _register_builtin_resources() -> None:
    pairings: dict[PromptResourcePairing, list[PromptResource]] = {}
    for (protocol, filename), digests in _BUILTIN_DIGESTS.items():
        for benchmark, sha256 in digests.items():
            pairings.setdefault((benchmark, protocol), []).append(
                PromptResource(filename=filename, sha256=sha256)
            )
    for (benchmark, protocol), resources in pairings.items():
        register_prompt_resources(benchmark, protocol, tuple(resources))


_register_builtin_resources()
