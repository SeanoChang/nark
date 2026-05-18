# LOCOMO Dataset

Long-Conversation Memory benchmark from Snap Research (ACL 2024).

Upstream: https://github.com/snap-research/locomo
Pinned commit SHA: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` (recorded in `fetch.sh`)

## Methodology disclaimer

LOCOMO scores reported in the literature are CONTESTED across papers:
- mem0 reports 91.6% (paper)
- Zep's independent reproduction shows ~58% with corrected eval
  (https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Letta showed a trivial filesystem-tool agent hits ~74%
  (https://www.letta.com/blog/benchmarking-ai-agent-memory)

For this reason, our bench treats LOCOMO as a SECONDARY signal, not
a headline. The baseline JSON's `properties.methodology_disclaimer`
field embeds these references inline so anyone reading the results
sees the caveat.

## How to fetch

```bash
bash bench/datasets/locomo/fetch.sh
```

The data file is included directly in the upstream git repo — no
separate HuggingFace download is required (unlike LongMemEval).

## Dataset structure

Discovered in Step 12.2 of the Phase 2 plan by cloning upstream and
inspecting `data/locomo10.json` directly.

**Data file path:** `upstream/repo/data/locomo10.json`

**Top-level shape:** flat JSON array of 10 sample objects.

**Sample entry keys:** `sample_id`, `conversation`, `qa`,
`observation`, `session_summary`, `event_summary`

**Sample entry shape:**

| Field | Type | Description |
|---|---|---|
| `sample_id` | `string` | Unique ID (e.g. `"conv-26"`) |
| `conversation` | `object` | Dict with keys `speaker_a`, `speaker_b`, `session_1`, `session_1_date_time`, `session_2`, ... Each `session_<N>` is an array of turn objects; each `session_<N>_date_time` is a timestamp string. |
| `qa` | `QA[]` | Array of QA pairs annotated for this conversation |
| `observation` | `object` | Generated per-session observations (for RAG eval) |
| `session_summary` | `object` | Generated per-session summaries (for RAG eval) |
| `event_summary` | `object` | Annotated event summaries per speaker per session |

**Conversation turn object shape** (each element of `session_<N>`):

```json
{ "speaker": "Caroline", "dia_id": "D1:3", "text": "..." }
```

- `speaker`: speaker name (matches `conversation.speaker_a` or `speaker_b`)
- `dia_id`: dialog ID of form `"D<session>:<turn>"` (used as evidence reference in QA)
- `text`: dialog text

**QA entry shape:**

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The question to answer |
| `answer` | `string \| number \| null` | Gold answer (may be integer year or null) |
| `category` | `integer` | Category label 1–5 (see below) |
| `evidence` | `string[]` | Dialog IDs containing the answer (e.g. `["D1:3"]`) |

Note: there is no `question_id` field. QAs are identified by their
index within the sample's `qa` array.

**Category values** (integer, not string — differs from LongMemEval):

| Value | Label | Count in locomo10 |
|---|---|---|
| 1 | multi-hop | 282 |
| 2 | single-hop | 321 |
| 3 | temporal | 96 |
| 4 | commonsense / open-domain knowledge | 841 |
| 5 | adversarial | 446 |

**Total samples:** 10 conversations
**Total QAs:** 1986 (range 105–260 per sample)

**Sessions per sample:** up to 19 sessions with dialog turns; later
session keys may appear only as `session_<N>_date_time` (timestamp
only, no dialog — those sessions have no actual turns in locomo10).

## Key structural difference from LongMemEval

- LongMemEval: list of QUESTIONS, each with its own independent haystack
- LOCOMO: list of SAMPLES (conversations), each with ALL its QA pairs

In LOCOMO, the memory context for all QAs in a sample is the same
conversation. The runner must ingest the conversation once per sample,
then answer all attached QA pairs.

## Running

(After Task 13 runner ships:)

```bash
cargo run -p nark-bench --release -- run \
  --task locomo \
  --systems fts5,nark,vector \
  --gen-backend claude-cli --gen-model claude-opus-4-7 \
  --judge-backend claude-cli --judge-model claude-opus-4-7 \
  --output bench/results/main
```
