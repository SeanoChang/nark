# LongMemEval Dataset

Public benchmark from ICLR 2025 — 500 curated questions across 5 ability
classes designed to test long-term conversational memory in LLM agents.

Upstream git repo: https://github.com/xiaowu0162/LongMemEval
Pinned commit SHA: `9e0b455f4ef0e2ab8f2e582289761153549043fc` (recorded in `fetch.sh`)

Dataset (HuggingFace): https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## How to fetch

```bash
bash bench/datasets/longmemeval/fetch.sh
```

Clones the upstream git repo at the pinned SHA into `upstream/repo/` (for eval
scripts) and downloads the three dataset JSON files into `upstream/data/`.
`upstream/` is gitignored (data files are ~265 MB each and live on HuggingFace).

## Dataset structure

Discovered in Step 8.2 of the Phase 2 plan by downloading and inspecting
`longmemeval_s_cleaned.json` directly from HuggingFace.

**Data file paths** (relative to `upstream/`):
- `data/longmemeval_s_cleaned.json` — 500 questions, short haystacks (~53 sessions each)
- `data/longmemeval_m_cleaned.json` — 500 questions, medium haystacks
- `data/longmemeval_oracle.json`    — 500 questions, oracle haystacks (answer sessions only)

**Each question entry has these fields:**

| Field | Type | Description |
|---|---|---|
| `question_id` | `string` | Unique ID (e.g. `"e47becba"`) |
| `question` | `string` | The question to answer |
| `answer` | `string` | Gold answer |
| `question_type` | `string` | Ability class (see below) |
| `question_date` | `string` | Date the question is asked (e.g. `"2023/05/30 (Tue) 23:40"`) |
| `answer_session_ids` | `string[]` | IDs of sessions containing the answer |
| `haystack_session_ids` | `string[]` | IDs of all sessions in this question's haystack |
| `haystack_dates` | `string[]` | Timestamp for each session in the haystack |
| `haystack_sessions` | `TurnObject[][]` | Nested array: list of sessions, each session is a list of turns |

**TurnObject shape** (NOT flat strings — each turn is an object):
```json
{ "role": "user" | "assistant", "content": "..." }
```

**`question_type` values** (6 classes, not 5 — "single-session" splits into 3 subtypes):

| Value | Description |
|---|---|
| `single-session-user` | Info from user's own utterances in one session |
| `single-session-assistant` | Info from assistant responses in one session |
| `single-session-preference` | User preference expressed in one session |
| `multi-session` | Reasoning across multiple sessions |
| `knowledge-update` | Tracking changes/updates over time |
| `temporal-reasoning` | Questions requiring temporal reasoning |

**Total questions:** 500 per file variant.

**Haystack size example** (longmemeval_s): ~53 sessions per question, each session
with 2–N turns.

## Five ability classes (paper taxonomy)

From the LongMemEval paper (ICLR 2025):

1. Information Extraction — the `single-session-*` variants
2. Multi-Session Reasoning — `multi-session`
3. Knowledge Updates — `knowledge-update`
4. Temporal Reasoning — `temporal-reasoning`
5. Abstention — correctly saying "I don't know" (not a separate `question_type`;
   handled by the judge)

Note: the dataset uses 6 `question_type` values but they map to the 5 paper
ability classes. The `single-session-user`, `single-session-assistant`, and
`single-session-preference` subtypes all fall under "Information Extraction."

## Methodology

This bench uses the upstream dataset and gold answers but runs the
gen + judge pipeline through our `LlmBackend` infrastructure. See
`bench/llm/README.md` for the prompt-versioning convention; see
`bench/src/tasks/longmemeval.rs` for the runner (added in Task 9).

## Phase 2 status

Phase 2 ships baseline numbers but does NOT include validation that
our methodology reproduces published mem0/Zep numbers — that requires
Phase 3's mem0 adapter. Until then, our LongMemEval numbers are
"internal directional" not "publishable comparable."
