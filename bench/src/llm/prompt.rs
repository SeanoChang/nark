//! Prompt template loader.
//!
//! Convention: each prompt file begins with `<!-- prompt-version: N -->`
//! on the first line. The loader extracts N and uses it as the
//! `prompt_version` in cache keys. When the prompt's intent changes,
//! bumping N invalidates cache entries that used the old prompt.
//!
//! Templates use mustache-style `{{var}}` placeholders. The loader
//! substitutes from a key-value map at render time.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub struct PromptTemplate {
    pub version: String,
    pub template: String,
}

impl PromptTemplate {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt file {:?}", path))?;
        Self::parse(&raw)
    }

    pub fn parse(raw: &str) -> Result<Self> {
        // First line should be `<!-- prompt-version: N -->`.
        let first_line = raw.lines().next()
            .ok_or_else(|| anyhow!("prompt is empty"))?;
        let version = first_line
            .strip_prefix("<!-- prompt-version: ")
            .and_then(|s| s.strip_suffix(" -->"))
            .ok_or_else(|| anyhow!(
                "prompt missing or malformed first-line `<!-- prompt-version: N -->` header; got: {}",
                first_line
            ))?
            .trim()
            .to_string();

        if version.is_empty() {
            return Err(anyhow!("prompt-version is empty"));
        }

        // Template is everything after the first line + its trailing newline.
        let template = raw.strip_prefix(first_line)
            .map(|rest| rest.strip_prefix('\n').unwrap_or(rest))
            .unwrap_or(raw)
            .to_string();

        Ok(PromptTemplate { version, template })
    }

    /// Substitute `{{var}}` placeholders using the given map. Missing keys
    /// are left in place (with the `{{var}}` syntax intact) — this lets the
    /// caller stage substitutions in multiple passes if needed.
    pub fn render(&self, vars: &HashMap<&str, &str>) -> String {
        let mut output = self.template.clone();
        for (key, value) in vars {
            let placeholder = format!("{{{{{}}}}}", key);
            output = output.replace(&placeholder, value);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_prompt() {
        let raw = "<!-- prompt-version: 1 -->\n\nHello {{name}}!";
        let p = PromptTemplate::parse(raw).unwrap();
        assert_eq!(p.version, "1");
        assert_eq!(p.template, "\nHello {{name}}!");
    }

    #[test]
    fn errors_on_missing_header() {
        let raw = "Hello world";
        let err = PromptTemplate::parse(raw).unwrap_err();
        assert!(err.to_string().contains("prompt-version"));
    }

    #[test]
    fn errors_on_empty_prompt() {
        let raw = "";
        let err = PromptTemplate::parse(raw).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn substitutes_placeholders() {
        let p = PromptTemplate { version: "1".into(), template: "Hi {{name}}, age {{age}}.".into() };
        let mut vars = HashMap::new();
        vars.insert("name", "Alice");
        vars.insert("age", "30");
        assert_eq!(p.render(&vars), "Hi Alice, age 30.");
    }

    #[test]
    fn leaves_unmatched_placeholders_intact() {
        let p = PromptTemplate { version: "1".into(), template: "Hi {{name}}, age {{age}}.".into() };
        let mut vars = HashMap::new();
        vars.insert("name", "Alice");
        assert_eq!(p.render(&vars), "Hi Alice, age {{age}}.");
    }

    #[test]
    fn parses_version_with_whitespace() {
        let raw = "<!-- prompt-version:   42   -->\nbody";
        let p = PromptTemplate::parse(raw).unwrap();
        assert_eq!(p.version, "42");
    }
}
