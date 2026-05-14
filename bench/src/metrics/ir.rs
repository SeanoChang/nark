//! Classical IR metrics. All deterministic, no LLM calls.

use std::collections::HashSet;

/// Recall@k: fraction of relevant docs found in the top-k ranked results.
///
/// `ranked` is the system's output ordered by descending score; only doc_ids matter here.
/// `relevant` is the gold-labelled set of relevant doc_ids for this query.
/// `k` is the cutoff; common values are 1, 5, 10.
pub fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let top: HashSet<&String> = ranked.iter().take(k).collect();
    let hits = relevant.iter().filter(|r| top.contains(*r)).count();
    hits as f64 / relevant.len() as f64
}

/// Mean Reciprocal Rank for a single query: 1 / rank of the first relevant doc, or 0 if none.
/// Average across queries is done by the caller.
pub fn reciprocal_rank(ranked: &[String], relevant: &HashSet<String>) -> f64 {
    for (i, doc_id) in ranked.iter().enumerate() {
        if relevant.contains(doc_id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// nDCG@k with binary relevance (1.0 if doc is in `relevant`, else 0.0).
/// Uses the standard log2 discount.
pub fn ndcg_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .filter_map(|(i, doc_id)| {
            if relevant.contains(doc_id) {
                Some(1.0 / ((i + 2) as f64).log2())
            } else {
                None
            }
        })
        .sum();
    let ideal_count = relevant.len().min(k);
    let idcg: f64 = (0..ideal_count)
        .map(|i| 1.0 / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn ranked(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_perfect() {
        // All 2 relevant docs appear in top-5
        let r = recall_at_k(&ranked(&["a", "b", "c", "d", "e"]), &rel(&["b", "e"]), 5);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn recall_partial() {
        // Only 1 of 2 relevant docs in top-3
        let r = recall_at_k(&ranked(&["a", "b", "c", "d", "e"]), &rel(&["b", "e"]), 3);
        assert!((r - 0.5).abs() < 1e-9, "expected 0.5, got {}", r);
    }

    #[test]
    fn recall_none() {
        let r = recall_at_k(&ranked(&["a", "b", "c"]), &rel(&["x", "y"]), 5);
        assert!(r.abs() < 1e-9, "expected 0.0, got {}", r);
    }

    #[test]
    fn recall_no_relevant_set_is_one_by_convention() {
        // When there are no relevant docs, recall is undefined; we return 1.0
        // to avoid penalising systems on queries with empty gold sets.
        let r = recall_at_k(&ranked(&["a", "b"]), &HashSet::new(), 5);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn rr_first_position() {
        let r = reciprocal_rank(&ranked(&["a", "b", "c"]), &rel(&["a"]));
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rr_third_position() {
        let r = reciprocal_rank(&ranked(&["a", "b", "c"]), &rel(&["c"]));
        assert!((r - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn rr_none_present() {
        let r = reciprocal_rank(&ranked(&["a", "b", "c"]), &rel(&["x"]));
        assert!(r.abs() < 1e-9);
    }

    #[test]
    fn ndcg_perfect_ordering() {
        // All relevant docs at the top in best possible order
        let r = ndcg_at_k(&ranked(&["a", "b", "c", "d"]), &rel(&["a", "b"]), 4);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn ndcg_worst_ordering() {
        // Relevant docs at the bottom — DCG is low but nonzero
        let r = ndcg_at_k(&ranked(&["a", "b", "c", "d"]), &rel(&["c", "d"]), 4);
        // DCG = 0 + 0 + 1/log2(4) + 1/log2(5) = 0.5 + 0.4306... ≈ 0.9307
        // IDCG = 1 + 1/log2(3) ≈ 1.6309
        // nDCG ≈ 0.5706
        assert!((r - 0.5706).abs() < 1e-3, "expected ~0.5706, got {}", r);
    }

    #[test]
    fn ndcg_empty_relevant() {
        let r = ndcg_at_k(&ranked(&["a", "b", "c"]), &HashSet::new(), 5);
        // Convention: when no relevant docs exist, nDCG is 1.0 (vacuously satisfied).
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }
}
