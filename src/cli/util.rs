/// Truncate a string at a word boundary, respecting multi-byte UTF-8.
pub fn truncate_at_word(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let boundary = s.char_indices()
        .take_while(|(i, _)| *i < max)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    match s[..boundary].rfind(' ') {
        Some(i) => &s[..i],
        None => &s[..boundary],
    }
}
