//! Loader for the LOCOMO dataset.
//!
//! Adjusted based on actual upstream format discovered in Step 12.1-12.2
//! of the Phase 2 plan. LOCOMO's structure: a flat JSON array of 10 "samples",
//! each containing a conversation + a list of QA pairs attached to that
//! conversation.
//!
//! **Key deviations from the spec template:**
//!
//! - `conversation` is NOT a `Vec<String>`. It is a JSON object with dynamic
//!   string keys: `speaker_a`, `speaker_b`, `session_1`, `session_1_date_time`,
//!   `session_2`, `session_2_date_time`, ... Each `session_<N>` maps to an array
//!   of turn objects `{speaker, dia_id, text}`. Some session numbers appear only
//!   as `_date_time` keys with no dialog array (those sessions have no turns
//!   in locomo10).
//!
//! - QA entries have NO `question_id` field. QAs are identified by their
//!   index within the sample's `qa` array. We synthesize an ID as
//!   `<sample_id>_q<idx>` in `load_samples`.
//!
//! - `answer` can be a string, an integer (e.g. year `2022`), or `null`.
//!   Stored as `serde_json::Value`; use `answer_as_string()` for a display form.
//!
//! - `category` is an integer (1–5), not a string:
//!   1 = multi-hop, 2 = single-hop, 3 = temporal,
//!   4 = commonsense/open-domain, 5 = adversarial.
//!
//! - The top-level JSON is a flat array (not `{"samples": [...]}`).

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// A single conversational turn within a LOCOMO session.
#[derive(Debug, Deserialize, Clone)]
pub struct Turn {
    /// Speaker name (matches `conversation.speaker_a` or `speaker_b`).
    pub speaker: String,
    /// Dialog ID of the form `"D<session>:<turn>"` (e.g. `"D1:3"`).
    /// Used as an evidence reference in QA entries.
    pub dia_id: String,
    /// Content of the dialog turn.
    pub text: String,
}

/// The conversation field of a LOCOMO sample.
///
/// Contains two named speakers and a dynamic set of session arrays.
/// Deserialized via a custom struct that captures the session turns.
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Name of speaker A.
    pub speaker_a: String,
    /// Name of speaker B.
    pub speaker_b: String,
    /// Sessions in chronological order (session index, date_time, turns).
    /// Sorted by session number at load time.
    pub sessions: Vec<Session>,
}

/// One session within a LOCOMO conversation.
#[derive(Debug, Clone)]
pub struct Session {
    /// Session number (1-based, from the `session_<N>` key).
    pub session_num: u32,
    /// Timestamp string (e.g. `"1:56 pm on 8 May, 2023"`). Empty if absent.
    pub date_time: String,
    /// Dialog turns in this session (empty if this session had no dialog array).
    pub turns: Vec<Turn>,
}

impl<'de> Deserialize<'de> for Conversation {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw: HashMap<String, Value> = HashMap::deserialize(deserializer)?;

        let speaker_a = raw
            .get("speaker_a")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let speaker_b = raw
            .get("speaker_b")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Collect all session numbers that have a dialog array.
        let mut session_nums: Vec<u32> = raw
            .keys()
            .filter_map(|k| k.strip_prefix("session_"))
            .filter(|s| !s.contains('_')) // exclude session_<N>_date_time
            .filter_map(|s| s.parse::<u32>().ok())
            .collect();
        session_nums.sort_unstable();

        // Also collect session numbers that appear only as _date_time (no dialog).
        let mut all_session_nums: std::collections::HashSet<u32> = session_nums
            .iter()
            .cloned()
            .collect();
        for k in raw.keys() {
            if let Some(rest) = k.strip_prefix("session_") {
                if let Some(num_str) = rest.strip_suffix("_date_time") {
                    if let Ok(n) = num_str.parse::<u32>() {
                        all_session_nums.insert(n);
                    }
                }
            }
        }
        let mut all_sorted: Vec<u32> = all_session_nums.into_iter().collect();
        all_sorted.sort_unstable();

        let mut sessions = Vec::new();
        for n in all_sorted {
            let date_key = format!("session_{}_date_time", n);
            let dialog_key = format!("session_{}", n);

            let date_time = raw
                .get(&date_key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let turns: Vec<Turn> = raw
                .get(&dialog_key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| serde_json::from_value::<Turn>(t.clone()).ok())
                        .collect()
                })
                .unwrap_or_default();

            sessions.push(Session {
                session_num: n,
                date_time,
                turns,
            });
        }

        Ok(Conversation {
            speaker_a,
            speaker_b,
            sessions,
        })
    }
}

/// A QA pair attached to a LOCOMO sample.
#[derive(Debug, Clone)]
pub struct LocomoQA {
    /// Synthesized ID: `<sample_id>_q<idx>` (no `question_id` in upstream).
    pub question_id: String,
    /// The question text.
    pub question: String,
    /// Gold answer — may be string, integer (e.g. year), or null in upstream.
    pub answer: Value,
    /// Question category (integer 1–5):
    /// 1=multi-hop, 2=single-hop, 3=temporal, 4=commonsense/open-domain, 5=adversarial.
    pub category: u32,
    /// Dialog IDs containing the answer (e.g. `["D1:3"]`). May be empty.
    pub evidence: Vec<String>,
}

impl LocomoQA {
    /// Return the answer as a display string.
    pub fn answer_as_string(&self) -> String {
        match &self.answer {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    /// Map the integer category to a human-readable ability label.
    pub fn category_label(&self) -> &'static str {
        category_label(self.category)
    }
}

/// A LOCOMO sample: one conversation + all its QA pairs.
#[derive(Debug, Clone)]
pub struct LocomoSample {
    /// Unique conversation ID (e.g. `"conv-26"`).
    pub sample_id: String,
    /// The multi-session conversation.
    pub conversation: Conversation,
    /// All QA pairs annotated for this conversation.
    pub qa: Vec<LocomoQA>,
}

// Raw deserialization helper — we post-process into LocomoSample.
#[derive(Deserialize)]
struct RawSample {
    sample_id: String,
    conversation: Conversation,
    qa: Vec<RawQA>,
}

#[derive(Deserialize)]
struct RawQA {
    question: String,
    answer: Value,
    category: u32,
    #[serde(default)]
    evidence: Vec<String>,
}

/// Load all samples from a LOCOMO JSON file.
///
/// Typical path:
/// `bench/datasets/locomo/upstream/repo/data/locomo10.json`
pub fn load_samples(path: &Path) -> Result<Vec<LocomoSample>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read LOCOMO data file {:?}", path))?;
    let raw_samples: Vec<RawSample> = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse LOCOMO JSON at {:?}", path))?;

    let samples = raw_samples
        .into_iter()
        .map(|rs| {
            let qa = rs
                .qa
                .into_iter()
                .enumerate()
                .map(|(idx, rq)| LocomoQA {
                    question_id: format!("{}_q{}", rs.sample_id, idx),
                    question: rq.question,
                    answer: rq.answer,
                    category: rq.category,
                    evidence: rq.evidence,
                })
                .collect();
            LocomoSample {
                sample_id: rs.sample_id,
                conversation: rs.conversation,
                qa,
            }
        })
        .collect();

    Ok(samples)
}

/// Map a LOCOMO category integer to a human-readable ability label.
///
/// Category mapping (from upstream evaluation code and paper):
/// - 1 = multi-hop
/// - 2 = single-hop
/// - 3 = temporal
/// - 4 = commonsense / open-domain knowledge
/// - 5 = adversarial
pub fn category_label(category: u32) -> &'static str {
    match category {
        1 => "multi-hop",
        2 => "single-hop",
        3 => "temporal",
        4 => "commonsense",
        5 => "adversarial",
        _ => "unknown",
    }
}

/// Flatten a sample's conversation into stable `(doc_id, text)` tuples,
/// one per turn across all sessions.
///
/// Document IDs: `<sample_id>_s<session_num>_t<turn_idx>`.
/// Turn text is formatted as `"<speaker>: <text>"` to preserve attribution.
pub fn conversation_to_documents(s: &LocomoSample) -> Vec<(String, String)> {
    s.conversation
        .sessions
        .iter()
        .flat_map(|session| {
            session.turns.iter().enumerate().map(|(turn_idx, turn)| {
                (
                    format!("{}_s{}_t{}", s.sample_id, session.session_num, turn_idx),
                    format!("{}: {}", turn.speaker, turn.text),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_sample() {
        let json = r#"[
            {
                "sample_id": "conv-1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "2:00 pm on 1 Jan, 2023",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Hello!"},
                        {"speaker": "Bob",   "dia_id": "D1:2", "text": "Hi back!"}
                    ]
                },
                "qa": [
                    {
                        "question": "What did Alice say?",
                        "answer": "Hello!",
                        "category": 2,
                        "evidence": ["D1:1"]
                    }
                ],
                "observation": {},
                "session_summary": {},
                "event_summary": {}
            }
        ]"#;
        let samples = load_samples_from_str(json).unwrap();
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        assert_eq!(s.sample_id, "conv-1");
        assert_eq!(s.conversation.speaker_a, "Alice");
        assert_eq!(s.conversation.sessions.len(), 1);
        assert_eq!(s.conversation.sessions[0].session_num, 1);
        assert_eq!(s.conversation.sessions[0].turns.len(), 2);
        assert_eq!(s.conversation.sessions[0].turns[0].speaker, "Alice");
        assert_eq!(s.qa.len(), 1);
        assert_eq!(s.qa[0].question_id, "conv-1_q0");
        assert_eq!(s.qa[0].category, 2);
        assert_eq!(s.qa[0].answer_as_string(), "Hello!");
        assert_eq!(s.qa[0].category_label(), "single-hop");
    }

    #[test]
    fn parses_integer_answer() {
        let json = r#"[
            {
                "sample_id": "conv-2",
                "conversation": {
                    "speaker_a": "A", "speaker_b": "B",
                    "session_1": []
                },
                "qa": [
                    {"question": "When?", "answer": 2022, "category": 3, "evidence": []}
                ],
                "observation": {}, "session_summary": {}, "event_summary": {}
            }
        ]"#;
        let samples = load_samples_from_str(json).unwrap();
        assert_eq!(samples[0].qa[0].answer_as_string(), "2022");
        assert_eq!(samples[0].qa[0].category_label(), "temporal");
    }

    #[test]
    fn parses_null_answer() {
        let json = r#"[
            {
                "sample_id": "conv-3",
                "conversation": {
                    "speaker_a": "A", "speaker_b": "B",
                    "session_1": []
                },
                "qa": [
                    {"question": "Q?", "answer": null, "category": 5, "evidence": []}
                ],
                "observation": {}, "session_summary": {}, "event_summary": {}
            }
        ]"#;
        let samples = load_samples_from_str(json).unwrap();
        assert_eq!(samples[0].qa[0].answer_as_string(), "");
        assert_eq!(samples[0].qa[0].category_label(), "adversarial");
    }

    #[test]
    fn flattens_conversation_to_documents() {
        let json = r#"[
            {
                "sample_id": "conv-4",
                "conversation": {
                    "speaker_a": "Alice", "speaker_b": "Bob",
                    "session_1_date_time": "noon",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Hi"},
                        {"speaker": "Bob",   "dia_id": "D1:2", "text": "Hey"}
                    ],
                    "session_2_date_time": "eve",
                    "session_2": [
                        {"speaker": "Alice", "dia_id": "D2:1", "text": "Bye"}
                    ]
                },
                "qa": [],
                "observation": {}, "session_summary": {}, "event_summary": {}
            }
        ]"#;
        let samples = load_samples_from_str(json).unwrap();
        let docs = conversation_to_documents(&samples[0]);
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].0, "conv-4_s1_t0");
        assert_eq!(docs[0].1, "Alice: Hi");
        assert_eq!(docs[1].0, "conv-4_s1_t1");
        assert_eq!(docs[1].1, "Bob: Hey");
        assert_eq!(docs[2].0, "conv-4_s2_t0");
        assert_eq!(docs[2].1, "Alice: Bye");
    }

    #[test]
    fn category_labels_all_values() {
        assert_eq!(category_label(1), "multi-hop");
        assert_eq!(category_label(2), "single-hop");
        assert_eq!(category_label(3), "temporal");
        assert_eq!(category_label(4), "commonsense");
        assert_eq!(category_label(5), "adversarial");
        assert_eq!(category_label(99), "unknown");
    }

    /// Helper for tests: parse samples from a JSON string.
    fn load_samples_from_str(s: &str) -> Result<Vec<LocomoSample>> {
        let raw_samples: Vec<RawSample> = serde_json::from_str(s)
            .with_context(|| "failed to parse test JSON")?;
        let samples = raw_samples
            .into_iter()
            .map(|rs| {
                let qa = rs
                    .qa
                    .into_iter()
                    .enumerate()
                    .map(|(idx, rq)| LocomoQA {
                        question_id: format!("{}_q{}", rs.sample_id, idx),
                        question: rq.question,
                        answer: rq.answer,
                        category: rq.category,
                        evidence: rq.evidence,
                    })
                    .collect();
                LocomoSample {
                    sample_id: rs.sample_id,
                    conversation: rs.conversation,
                    qa,
                }
            })
            .collect();
        Ok(samples)
    }
}
