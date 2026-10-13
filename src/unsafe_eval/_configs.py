"""Find and resolve the run configurations packaged with unsafe-eval.

A configuration file sits at ``<provider>/<configuration>/<benchmark>.yaml`` below a config
root. ``<provider>/<configuration>`` is its configuration identifier, for example
``vllm/shieldstral-1.0-3b``, and appending the benchmark identifier addresses one file, for
example ``vllm/shieldstral-1.0-3b/aegis-2.0``.
"""

from pathlib import Path

PACKAGED_CONFIG_ROOT = Path(__file__).parent / "configs"


class SweepSelectionError(ValueError):
    """Raised when a selector matches no run configuration."""


def config_id(config: Path) -> str:
    """Name a configuration by its provider directory and tree, e.g. `vllm/polyguard-qwen`.

    A file outside that layout, such as `./my-config.yaml`, is named after whichever of the
    two directories above it exist in the path as given, which may be none.
    """
    return "/".join(name for name in (config.parent.parent.name, config.parent.name) if name)


def full_config_id(config: Path) -> str:
    """Address one configuration file, e.g. `vllm/polyguard-qwen/aegis-2.0`."""
    return f"{config_id(config)}/{config.stem}"


def discover_configs(config_root: Path) -> list[Path]:
    """Return every run configuration below a config root, in sorted order."""
    return sorted(config_root.glob("*/*/*.yaml"))


def _unknown(noun: str, selector: str, available: set[str]) -> SweepSelectionError:
    choices = ", ".join(sorted(available)) if available else "none"
    return SweepSelectionError(
        f"Unknown {noun} identifier {selector!r}. Available identifiers: {choices}."
    )


def select_configs(selector: str, config_root: Path) -> list[Path]:
    """Resolve a configuration identifier, or one configuration file's identifier."""
    configs = discover_configs(config_root)
    selected = [config for config in configs if config_id(config) == selector]
    if not selected:
        selected = [config for config in configs if full_config_id(config) == selector]
    if not selected:
        # Only the configuration identifiers: every file would make a list of hundreds.
        raise _unknown("configuration", selector, {config_id(config) for config in configs})
    return selected


def select_benchmark_configs(benchmark: str, config_root: Path) -> list[Path]:
    """Return every configuration that runs one benchmark."""
    configs = discover_configs(config_root)
    selected = [config for config in configs if config.stem == benchmark]
    if not selected:
        raise _unknown("benchmark", benchmark, {config.stem for config in configs})
    return selected


def config_path(identifier: str, config_root: Path) -> Path:
    """Return the absolute path of the configuration file an identifier addresses."""
    configs = discover_configs(config_root)
    for config in configs:
        if full_config_id(config) == identifier:
            return config.resolve()
    # Name the benchmarks of the configuration the identifier starts with, if there is one,
    # since listing every file would run to hundreds of names.
    tree, _, _ = identifier.rpartition("/")
    siblings = {full_config_id(config) for config in configs if config_id(config) == tree}
    if siblings:
        raise _unknown("configuration", identifier, siblings)
    raise _unknown("configuration", identifier, {config_id(config) for config in configs})
