"""Load, render, and identify packaged text resources."""

from hashlib import sha256
from importlib.resources import files
from string import Formatter


def load_text_resource(*parts: str, package: str = "unsafe_eval") -> str:
    """Read a package-relative UTF-8 resource and remove one trailing newline.

    `parts` gives the path components within the importable `package`. Raise
    `FileNotFoundError` if the path is missing or does not name a file.
    """
    resource = files(package).joinpath(*parts)
    if not resource.is_file():
        raise FileNotFoundError(f"Missing packaged text resource: {package}:{'/'.join(parts)}")
    return resource.read_text(encoding="utf-8").removesuffix("\n")


def render_template(template: str, /, **values: str) -> str:
    """Fill named Python format fields using exactly the supplied keyword values.

    Raise `ValueError` if a field has no value or a supplied value has no field.
    Repeated uses of the same field require only one value. Literal braces must be
    escaped as `{{` and `}}`, as with `str.format`.
    """
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
    """Return the hexadecimal SHA-256 digest of `content` encoded as UTF-8.

    This function does not normalize text. For packaged resources, pass the text returned
    by `load_text_resource`, which removes one trailing newline before hashing.
    """
    return sha256(content.encode()).hexdigest()
