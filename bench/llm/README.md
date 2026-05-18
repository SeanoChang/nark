# nark-bench LLM Prompts

This directory holds the prompt templates used by Task B's generation
and judging steps. Each task (LongMemEval, LOCOMO) has two prompts:

- `prompts/<task>-generate.md` — sent to the generation LLM after retrieval
- `prompts/<task>-judge.md` — sent to the judging LLM to score the generated answer

## Versioning convention

Every prompt file's first line MUST be `<!-- prompt-version: N -->`.

The prompt loader extracts N and uses it as part of the cache key:
`sha256(prompt + model_id + prompt_version)`. When you change a prompt's
intent (not just whitespace or wording polish), bump N. The cache will
re-query the LLM for affected entries; old cached responses become
unreachable.

Whitespace-only or comment-only changes do not require a version bump —
the rendered prompt is identical, so the cache key is identical.

When in doubt, bump. Cost of an unnecessary bump: re-querying ~500
questions × 1-2 backends = a one-time subscription-limits hit (then
cached). Cost of NOT bumping when you should: silently mixed-methodology
results in the same baseline JSON, hard to detect after the fact.

## Placeholder syntax

Templates use `{{var}}` placeholders. The loader substitutes from a
`HashMap<&str, &str>` at render time. Missing keys are left in place —
this lets multi-pass substitution work cleanly.

Common placeholders:

- `{{question}}` — the question text from the dataset
- `{{gold}}` — the gold answer (judge prompts only)
- `{{candidate}}` — the generated answer being judged (judge prompts only)
- `{{context}}` — retrieved context snippets joined with `\n---\n` (generation prompts only)

## Iteration discipline

1. Write `<!-- prompt-version: 1 -->` initial prompt.
2. Run a subset of questions (10-20) with `--task longmemeval --systems vector` and inspect verdicts manually.
3. If verdicts look wrong (judge too harsh, too lenient, misclassifying), edit prompt and bump version.
4. Cache invalidates affected entries on next run. Re-run subset.
5. Once stable, run full sweep. Commit the locked-in prompt.

Prompt history lives in git — `git log -p bench/llm/prompts/<file>` shows
methodology evolution, which is the credibility argument when publishing
numbers.
