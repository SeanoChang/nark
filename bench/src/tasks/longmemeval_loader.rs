//! Loader for the LongMemEval dataset.
//!
//! Parses the upstream JSON files (downloaded via bench/datasets/longmemeval/fetch.sh
//! from HuggingFace: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
//! into typed Rust structs the runner can consume.
//!
//! **Adjusted from spec to match actual upstream format (discovered in Step 8.2):**
//! - `haystack_sessions` is `Vec<Vec<Turn>>` where `Turn` is `{role, content}`,
//!   NOT `Vec<Vec<String>>` as the spec assumed.
//! - Three additional fields are present: `question_date`, `answer_session_ids`,
//!   `haystack_session_ids`, and `haystack_dates`.
//! - `question_type` has 6 values (not 5): `single-session-user`,
//!   `single-session-assistant`, `single-session-preference`, `multi-session`,
//!   `knowledge-update`, `temporal-reasoning`.
//!
//! If upstream changes their format (rare; the dataset is version-pinned via
//! fetch.sh), update this file accordingly.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// A single conversational turn in a haystack session.
#[derive(Debug, Deserialize, Clone)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// A single question entry from the LongMemEval dataset.
///
/// Format as discovered in Step 8.2 from `longmemeval_s_cleaned.json`.
#[derive(Debug, Deserialize, Clone)]
pub struct LongMemEvalQuestion {
    pub question_id: String,
    pub question: String,
    /// Gold answer.
    pub answer: String,
    /// Ability class — drives per-ability breakdown in the result JSON.
    /// One of: `single-session-user`, `single-session-assistant`,
    /// `single-session-preference`, `multi-session`, `knowledge-update`,
    /// `temporal-reasoning`.
    pub question_type: String,
    /// Date the question is asked (e.g. `"2023/05/30 (Tue) 23:40"`).
    pub question_date: String,
    /// IDs of the sessions that contain the answer.
    pub answer_session_ids: Vec<String>,
    /// IDs of all sessions in this question's haystack (parallel to
    /// `haystack_sessions` and `haystack_dates`).
    pub haystack_session_ids: Vec<String>,
    /// Timestamp string for each session in the haystack (parallel to
    /// `haystack_sessions`).
    pub haystack_dates: Vec<String>,
    /// Haystack: nested list of sessions; each session is a list of turns.
    /// Each turn has `role` (`"user"` or `"assistant"`) and `content`.
    pub haystack_sessions: Vec<Vec<Turn>>,
}

/// Load all questions from a LongMemEval JSON file.
///
/// Typical path:
/// `bench/datasets/longmemeval/upstream/data/longmemeval_s_cleaned.json`
pub fn load_questions(path: &Path) -> Result<Vec<LongMemEvalQuestion>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read LongMemEval data file {:?}", path))?;
    let questions: Vec<LongMemEvalQuestion> = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse LongMemEval JSON at {:?}", path))?;
    Ok(questions)
}

/// Flatten a question's haystack into a list of `(document_id, text)` tuples,
/// one per turn. Document IDs are stable: `s{session_idx}_t{turn_idx}`.
///
/// The `text` is formatted as `"{role}: {content}"` to preserve speaker
/// attribution. This matches the typical format used in memory-system ingestion.
pub fn haystack_to_documents(q: &LongMemEvalQuestion) -> Vec<(String, String)> {
    q.haystack_sessions
        .iter()
        .enumerate()
        .flat_map(|(session_idx, session)| {
            session.iter().enumerate().map(move |(turn_idx, turn)| {
                (
                    format!("s{}_t{}", session_idx, turn_idx),
                    format!("{}: {}", turn.role, turn.content),
                )
            })
        })
        .collect()
}

/// Map a `question_type` string to the paper's 5 ability-class labels.
///
/// The dataset uses 6 `question_type` values; the three `single-session-*`
/// subtypes all fall under the paper's "Information Extraction" class.
pub fn ability_class(question_type: &str) -> &'static str {
    match question_type {
        "single-session-user" | "single-session-assistant" | "single-session-preference" => {
            "information-extraction"
        }
        "multi-session" => "multi-session-reasoning",
        "knowledge-update" => "knowledge-updates",
        "temporal-reasoning" => "temporal-reasoning",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_question() {
        let json = r#"[
            {
                "question_id": "q1",
                "question": "What did Alice say?",
                "answer": "She said hello.",
                "question_type": "single-session-user",
                "question_date": "2023/05/30 (Tue) 23:40",
                "answer_session_ids": ["session_abc"],
                "haystack_session_ids": ["session_abc", "session_xyz"],
                "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:00"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "Alice: Hello"},
                        {"role": "assistant", "content": "Hi there!"}
                    ],
                    [
                        {"role": "user", "content": "How are you"},
                        {"role": "assistant", "content": "Fine"}
                    ]
                ]
            }
        ]"#;
        let qs: Vec<LongMemEvalQuestion> = serde_json::from_str(json).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].question_id, "q1");
        assert_eq!(qs[0].question_type, "single-session-user");
        assert_eq!(qs[0].haystack_sessions.len(), 2);
        assert_eq!(qs[0].haystack_sessions[0].len(), 2);
        assert_eq!(qs[0].haystack_sessions[0][0].role, "user");
        assert_eq!(qs[0].haystack_sessions[0][0].content, "Alice: Hello");
        assert_eq!(qs[0].haystack_dates.len(), 2);
        assert_eq!(qs[0].answer_session_ids, vec!["session_abc"]);
    }

    #[test]
    fn flattens_haystack_to_documents() {
        let q = LongMemEvalQuestion {
            question_id: "q1".into(),
            question: "Q".into(),
            answer: "A".into(),
            question_type: "single-session-user".into(),
            question_date: "2023/01/01 (Sun) 00:00".into(),
            answer_session_ids: vec![],
            haystack_session_ids: vec!["s0".into(), "s1".into()],
            haystack_dates: vec!["2023/01/01 (Sun) 00:00".into(), "2023/01/02 (Mon) 00:00".into()],
            haystack_sessions: vec![
                vec![
                    Turn { role: "user".into(), content: "hello".into() },
                    Turn { role: "assistant".into(), content: "hi".into() },
                ],
                vec![
                    Turn { role: "user".into(), content: "bye".into() },
                ],
            ],
        };
        let docs = haystack_to_documents(&q);
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].0, "s0_t0");
        assert_eq!(docs[0].1, "user: hello");
        assert_eq!(docs[1].0, "s0_t1");
        assert_eq!(docs[1].1, "assistant: hi");
        assert_eq!(docs[2].0, "s1_t0");
        assert_eq!(docs[2].1, "user: bye");
    }

    #[test]
    fn ability_class_mapping() {
        assert_eq!(ability_class("single-session-user"), "information-extraction");
        assert_eq!(ability_class("single-session-assistant"), "information-extraction");
        assert_eq!(ability_class("single-session-preference"), "information-extraction");
        assert_eq!(ability_class("multi-session"), "multi-session-reasoning");
        assert_eq!(ability_class("knowledge-update"), "knowledge-updates");
        assert_eq!(ability_class("temporal-reasoning"), "temporal-reasoning");
        assert_eq!(ability_class("unknown-type"), "unknown");
    }
}
