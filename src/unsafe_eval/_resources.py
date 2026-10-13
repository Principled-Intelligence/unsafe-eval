"""Load, render, and identify packaged text resources."""

from hashlib import sha256
from importlib.resources import files
from string import Formatter


def load_text_resource(*parts: str, package: str = "unsafe_eval") -> str:
    """Load a required UTF-8 text resource from the package."""
    resource = files(package).joinpath(*parts)
    if not resource.is_file():
        raise FileNotFoundError(f"Missing packaged text resource: {package}:{'/'.join(parts)}")
    return resource.read_text(encoding="utf-8").removesuffix("\n")


def render_template(template: str, /, **values: str) -> str:
    """Render a template only when its named fields exactly match the supplied values."""
    fields = {
        field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None
    }
    supplied = set(values)
    if fields != supplied:
        missing = sorted(fields - supplied)
        unexpected = sorted(supplied - fields)
        details = []
        if missing:
            details.append(f"missing={missing!r}")
        if unexpected:
            details.append(f"unexpected={unexpected!r}")
        raise ValueError(f"Template variables do not match values: {', '.join(details)}")
    return template.format(**values)


def content_sha256(content: str) -> str:
    """Return a stable identifier for packaged text after loading normalization."""
    return sha256(content.encode()).hexdigest()
