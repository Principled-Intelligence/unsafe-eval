# Contributing

Thanks for helping improve `unsafe-eval`.

## Setup

```bash
uv sync
uv run pre-commit install
```

The pre-commit hooks sort imports and run the ruff formatter on the files you commit. If a
hook rewrites a file, the commit stops; stage the changes and commit again.

## Before opening a pull request

Run the full quality gate. CI runs the same commands on Python 3.11 and 3.12:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Pytest enforces at least 90% branch coverage. The suite runs offline and needs no GPU or
credentials.

## Conventions

[AGENTS.md](AGENTS.md) is the authoritative description of the repository layout, typing and
style rules, registration and extension APIs, and evaluation invariants. Read it before changing
anything under `src/`. In short:

- Keep parsers strict.
- Never change a pinned revision as incidental cleanup.
- Keep docs and configs in sync with behavior.
- Test each behavior change once, where no other test already covers it.

For adding a benchmark, model, or metric, start from [docs/extending.md](docs/extending.md).

## Sensitive content

Benchmarks and logs contain explicit or upsetting material. Do not paste prompts or model
completions into issues, pull requests, source, or tests. Never commit `.env`, datasets, model
weights, caches, or anything from `runs/` or `logs/`.

## Reporting problems

Open an issue using one of the issue templates. For security concerns, see
[SECURITY.md](SECURITY.md).
