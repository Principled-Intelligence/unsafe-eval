## Summary

## Checklist

- [ ] `uv run ruff check . && uv run ty check && uv run pytest` is green
- [ ] Each new test reaches code no other test does or pins an evaluation invariant
- [ ] README and `docs/` pages that describe the changed behaviour are updated
- [ ] No pinned revision, prompt resource, threshold, or metric changed without a stated reason
- [ ] No benchmark prompts, model outputs, credentials, or logs are included
