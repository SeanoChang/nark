<!-- prompt-version: 1 -->

You are evaluating whether a candidate answer matches a gold answer for a long-conversation memory benchmark.

Question: {{question}}
Gold answer: {{gold}}
Candidate answer: {{candidate}}

Respond with ONLY a JSON object on a single line:
{"verdict": "correct" | "partial" | "incorrect", "reason": "<one-sentence rationale>"}

Rules:
- "correct" if the candidate conveys the same factual content as gold, even if worded differently
- "partial" if it captures part of gold's content but omits important specifics
- "incorrect" if the candidate states something contradictory or unrelated to gold
- If gold is "I don't know" or similar abstention, "correct" iff candidate also abstains
