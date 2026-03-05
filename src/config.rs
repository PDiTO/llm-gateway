//! TOML configuration: parsing, defaults and startup validation.
//!
//! Validation collects every problem it finds instead of stopping at the first
//! one, so a broken config file can be fixed in a single pass.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

/// Request fields that the gateway itself depends on and that a target is not
/// allowed to strip with `drop_params`.
const PROTECTED_PARAMS: &[&str] = &["model", "messages", "stream"];

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read config file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("{}", InvalidList(.0))]
    Invalid(Vec<String>),
}

struct InvalidList<'a>(&'a [String]);

impl fmt::Display for InvalidList<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid configuration ({} problem", self.0.len())?;
        if self.0.len() != 1 {
            write!(f, "s")?;
        }
        write!(f, "):")?;
        for problem in self.0 {
            write!(f, "\n  - {problem}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub models: Vec<ModelRoute>,
    #[serde(default)]
    pub prices: BTreeMap<String, Price>,
    #[serde(default)]
    pub keys: Vec<ApiKeyConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "defaults::listen")]
    pub listen: String,
    #[serde(default = "defaults::max_body_bytes")]
    pub max_body_bytes: usize,
    /// How long in-flight requests get to finish after SIGTERM/ctrl-c.
    #[serde(default = "defaults::shutdown_grace_ms")]
    pub shutdown_grace_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: defaults::listen(),
            max_body_bytes: defaults::max_body_bytes(),
            shutdown_grace_ms: defaults::shutdown_grace_ms(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryConfig {
    /// Attempts per target, including the first one.
    #[serde(default = "defaults::max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "defaults::base_delay_ms")]
    pub base_delay_ms: u64,
    #[serde(default = "defaults::max_delay_ms")]
    pub max_delay_ms: u64,
    /// A `Retry-After` longer than this makes the gateway move on to the next
    /// target instead of waiting.
    #[serde(default = "defaults::max_retry_after_ms")]
    pub max_retry_after_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: defaults::max_attempts(),
            base_delay_ms: defaults::base_delay_ms(),
            max_delay_ms: defaults::max_delay_ms(),
            max_retry_after_ms: defaults::max_retry_after_ms(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures that open the circuit.
    #[serde(default = "defaults::failure_threshold")]
    pub failure_threshold: u32,
    /// How long an open circuit rejects requests before letting a probe through.
    #[serde(default = "defaults::cooldown_ms")]
    pub cooldown_ms: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: defaults::failure_threshold(),
            cooldown_ms: defaults::cooldown_ms(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "defaults::cache_max_entries")]
    pub max_entries: usize,
    #[serde(default = "defaults::cache_ttl_secs")]
    pub ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: defaults::cache_max_entries(),
            ttl_secs: defaults::cache_ttl_secs(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// Any upstream that speaks the OpenAI Chat Completions API.
    OpenAi,
    /// The Anthropic Messages API, translated to and from the OpenAI shape.
    Anthropic,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub base_url: String,
    /// Literal API key. Prefer `api_key_env` outside of tests.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Name of the environment variable holding the API key.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Time allowed until response headers (streams) or the whole response.
    #[serde(default = "defaults::provider_timeout_ms")]
    pub timeout_ms: u64,
    /// Longest gap allowed between chunks of a stream once it has started.
    #[serde(default = "defaults::stream_idle_timeout_ms")]
    pub stream_idle_timeout_ms: u64,
    /// Sent as `anthropic-version`. Only used by `kind = "anthropic"`.
    #[serde(default = "defaults::anthropic_version")]
    pub anthropic_version: String,
    /// Anthropic requires `max_tokens`; this is used when the client sent none.
    #[serde(default = "defaults::default_max_tokens")]
    pub default_max_tokens: u32,
    /// Extra headers added to every upstream request.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Filled in from `api_key` or `api_key_env` during validation.
    #[serde(skip)]
    pub resolved_api_key: Option<String>,
}

// Hand-written so the API key never ends up in logs.
impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("api_key_env", &self.api_key_env)
            .field("timeout_ms", &self.timeout_ms)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRoute {
    /// The model name clients send, e.g. `"smart"` or `"claude-sonnet-4-5-20250929"`.
    pub name: String,
    /// Tried in order until one succeeds.
    pub targets: Vec<Target>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub provider: String,
    /// Model ID sent to the upstream.
    pub model: String,
    /// Overrides the provider's `timeout_ms` for this target.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Top-level request fields to remove before sending, for models that
    /// reject them (for example `temperature` on reasoning models).
    #[serde(default)]
    pub drop_params: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Price {
    /// USD per million input tokens.
    pub input_per_mtok: f64,
    /// USD per million output tokens.
    pub output_per_mtok: f64,
    /// USD per million cached input tokens (cache reads). Defaults to the
    /// input price.
    #[serde(default)]
    pub cached_input_per_mtok: Option<f64>,
    /// USD per million tokens written to the prompt cache. Anthropic reports
    /// these separately and charges more for them. Defaults to the input price.
    #[serde(default)]
    pub cache_write_per_mtok: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyConfig {
    pub name: String,
    /// Hex SHA-256 of the key. Generate with `llm-gateway hash-key`.
    pub key_sha256: String,
    pub requests_per_minute: u32,
    /// Bucket size. Defaults to `requests_per_minute`.
    #[serde(default)]
    pub burst: Option<u32>,
    /// Spend cap in USD per UTC day.
    #[serde(default)]
    pub daily_budget_usd: Option<f64>,
}

mod defaults {
    pub fn listen() -> String {
        "0.0.0.0:8080".to_string()
    }
    pub fn max_body_bytes() -> usize {
        4 * 1024 * 1024
    }
    pub fn shutdown_grace_ms() -> u64 {
        30_000
    }
    pub fn max_attempts() -> u32 {
        2
    }
    pub fn base_delay_ms() -> u64 {
        250
    }
    pub fn max_delay_ms() -> u64 {
        4_000
    }
    pub fn max_retry_after_ms() -> u64 {
        10_000
    }
    pub fn failure_threshold() -> u32 {
        5
    }
    pub fn cooldown_ms() -> u64 {
        30_000
    }
    pub fn cache_max_entries() -> usize {
        1_000
    }
    pub fn cache_ttl_secs() -> u64 {
        300
    }
    pub fn provider_timeout_ms() -> u64 {
        60_000
    }
    pub fn stream_idle_timeout_ms() -> u64 {
        60_000
    }
    pub fn anthropic_version() -> String {
        "2023-06-01".to_string()
    }
    pub fn default_max_tokens() -> u32 {
        4_096
    }
}

impl Config {
    /// Reads, parses and validates a config file, resolving API keys from the
    /// process environment.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text)
    }

    /// Parses and validates, resolving API keys from the process environment.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        Self::from_toml_with_env(text, |name| std::env::var(name).ok())
    }

    /// Parses and validates, resolving `api_key_env` through `env`. Tests use
    /// this to avoid touching the real environment.
    pub fn from_toml_with_env(
        text: &str,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ConfigError> {
        let mut config: Config = toml::from_str(text)?;
        config.validate(&env)?;
        Ok(config)
    }

    fn validate(&mut self, env: &dyn Fn(&str) -> Option<String>) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        if self.server.listen.parse::<SocketAddr>().is_err() {
            errors.push(format!(
                "server.listen: {:?} is not a socket address like \"0.0.0.0:8080\"",
                self.server.listen
            ));
        }
        if self.server.max_body_bytes == 0 {
            errors.push("server.max_body_bytes must be greater than 0".into());
        }

        if self.retry.max_attempts == 0 {
            errors.push("retry.max_attempts must be at least 1".into());
        }
        if self.retry.base_delay_ms > self.retry.max_delay_ms {
            errors.push(format!(
                "retry.base_delay_ms ({}) is larger than retry.max_delay_ms ({})",
                self.retry.base_delay_ms, self.retry.max_delay_ms
            ));
        }
        if self.circuit_breaker.failure_threshold == 0 {
            errors.push("circuit_breaker.failure_threshold must be at least 1".into());
        }
        if self.circuit_breaker.cooldown_ms == 0 {
            errors.push("circuit_breaker.cooldown_ms must be greater than 0".into());
        }
        if self.cache.enabled {
            if self.cache.max_entries == 0 {
                errors.push(
                    "cache.max_entries must be greater than 0 when the cache is enabled".into(),
                );
            }
            if self.cache.ttl_secs == 0 {
                errors
                    .push("cache.ttl_secs must be greater than 0 when the cache is enabled".into());
            }
        }

        self.validate_providers(env, &mut errors);
        self.validate_models(&mut errors);
        self.validate_prices(&mut errors);
        self.validate_keys(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(errors))
        }
    }

    fn validate_providers(
        &mut self,
        env: &dyn Fn(&str) -> Option<String>,
        errors: &mut Vec<String>,
    ) {
        if self.providers.is_empty() {
            errors.push("no providers defined; add at least one [providers.<name>] table".into());
        }
        for (name, provider) in &mut self.providers {
            let at = format!("providers.{name}");
            match reqwest::Url::parse(&provider.base_url) {
                Ok(url) if matches!(url.scheme(), "http" | "https") => {}
                Ok(url) => errors.push(format!(
                    "{at}.base_url: scheme must be http or https, got {:?}",
                    url.scheme()
                )),
                Err(e) => errors.push(format!(
                    "{at}.base_url: {:?} is not a URL ({e})",
                    provider.base_url
                )),
            }
            if provider.timeout_ms == 0 {
                errors.push(format!("{at}.timeout_ms must be greater than 0"));
            }
            if provider.stream_idle_timeout_ms == 0 {
                errors.push(format!(
                    "{at}.stream_idle_timeout_ms must be greater than 0"
                ));
            }
            if provider.default_max_tokens == 0 {
                errors.push(format!("{at}.default_max_tokens must be greater than 0"));
            }
            for header in provider.headers.keys() {
                if reqwest::header::HeaderName::from_bytes(header.as_bytes()).is_err() {
                    errors.push(format!(
                        "{at}.headers: {header:?} is not a valid header name"
                    ));
                }
            }
            provider.resolved_api_key = match (&provider.api_key, &provider.api_key_env) {
                (Some(_), Some(_)) => {
                    errors.push(format!("{at}: set api_key or api_key_env, not both"));
                    None
                }
                (Some(key), None) => Some(key.clone()),
                (None, Some(var)) => match env(var) {
                    Some(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
                    _ => {
                        errors.push(format!(
                            "{at}.api_key_env: environment variable {var} is not set or is empty"
                        ));
                        None
                    }
                },
                (None, None) if provider.kind == ProviderKind::Anthropic => {
                    errors.push(format!(
                        "{at}: the anthropic provider needs api_key or api_key_env"
                    ));
                    None
                }
                // Local OpenAI-compatible servers (vLLM, llama.cpp, Ollama) often
                // run without auth.
                (None, None) => None,
            };
        }
    }

    fn validate_models(&self, errors: &mut Vec<String>) {
        if self.models.is_empty() {
            errors.push("no models defined; add at least one [[models]] entry".into());
        }
        let mut seen = HashSet::new();
        for (i, route) in self.models.iter().enumerate() {
            let at = if route.name.is_empty() {
                format!("models[{i}]")
            } else {
                format!("models[{i}] ({:?})", route.name)
            };
            if route.name.trim().is_empty() {
                errors.push(format!("{at}.name must not be empty"));
            } else if !seen.insert(route.name.as_str()) {
                errors.push(format!(
                    "{at}: model name {:?} is defined more than once",
                    route.name
                ));
            }
            if route.targets.is_empty() {
                errors.push(format!("{at}: needs at least one target"));
            }
            for (j, target) in route.targets.iter().enumerate() {
                let at = format!("{at}.targets[{j}]");
                if !self.providers.contains_key(&target.provider) {
                    let known: Vec<_> = self.providers.keys().map(String::as_str).collect();
                    errors.push(format!(
                        "{at}: unknown provider {:?} (known: {})",
                        target.provider,
                        if known.is_empty() {
                            "none".to_string()
                        } else {
                            known.join(", ")
                        }
                    ));
                }
                if target.model.trim().is_empty() {
                    errors.push(format!("{at}.model must not be empty"));
                }
                if target.timeout_ms == Some(0) {
                    errors.push(format!("{at}.timeout_ms must be greater than 0"));
                }
                for param in &target.drop_params {
                    if PROTECTED_PARAMS.contains(&param.as_str()) {
                        errors.push(format!("{at}.drop_params: {param:?} cannot be dropped"));
                    }
                }
            }
        }
    }

    fn validate_prices(&self, errors: &mut Vec<String>) {
        for (model, price) in &self.prices {
            let fields = [
                ("input_per_mtok", Some(price.input_per_mtok)),
                ("output_per_mtok", Some(price.output_per_mtok)),
                ("cached_input_per_mtok", price.cached_input_per_mtok),
                ("cache_write_per_mtok", price.cache_write_per_mtok),
            ];
            for (field, value) in fields {
                if let Some(v) = value
                    && (!v.is_finite() || v < 0.0)
                {
                    errors.push(format!(
                        "prices.{model:?}.{field} must be a non-negative number, got {v}"
                    ));
                }
            }
        }
    }

    fn validate_keys(&self, errors: &mut Vec<String>) {
        if self.keys.is_empty() {
            errors.push(
                "no API keys defined; add at least one [[keys]] entry (see `llm-gateway gen-key`)"
                    .into(),
            );
        }
        let mut names = HashSet::new();
        let mut hashes = HashSet::new();
        for (i, key) in self.keys.iter().enumerate() {
            let at = format!("keys[{i}] ({:?})", key.name);
            if key.name.trim().is_empty() {
                errors.push(format!("keys[{i}].name must not be empty"));
            } else if !names.insert(key.name.as_str()) {
                errors.push(format!("{at}: key name is used more than once"));
            }
            let hash = key.key_sha256.to_ascii_lowercase();
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                errors.push(format!(
                    "{at}.key_sha256 must be 64 hex characters; store the hash from \
                     `llm-gateway hash-key`, not the key itself"
                ));
            } else if !hashes.insert(hash) {
                errors.push(format!("{at}: the same key hash appears more than once"));
            }
            if key.requests_per_minute == 0 {
                errors.push(format!("{at}.requests_per_minute must be greater than 0"));
            }
            if key.burst == Some(0) {
                errors.push(format!("{at}.burst must be greater than 0"));
            }
            if let Some(budget) = key.daily_budget_usd
                && (!budget.is_finite() || budget <= 0.0)
            {
                errors.push(format!(
                    "{at}.daily_budget_usd must be a positive number, got {budget}"
                ));
            }
        }
    }

    /// Non-fatal problems worth logging at startup.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let mut unpriced = HashSet::new();
        for route in &self.models {
            for target in &route.targets {
                if !self.prices.contains_key(&target.model)
                    && unpriced.insert(target.model.as_str())
                {
                    warnings.push(format!(
                        "no [prices] entry for upstream model {:?}; its cost will be recorded as 0 \
                         and it will not count towards budgets",
                        target.model
                    ));
                }
            }
        }
        for (name, provider) in &self.providers {
            if provider.api_key.is_some() {
                warnings.push(format!(
                    "providers.{name} has a literal api_key in the config file; api_key_env is safer"
                ));
            }
        }
        warnings
    }
}

impl RetryConfig {
    pub fn base_delay(&self) -> Duration {
        Duration::from_millis(self.base_delay_ms)
    }
    pub fn max_delay(&self) -> Duration {
        Duration::from_millis(self.max_delay_ms)
    }
    pub fn max_retry_after(&self) -> Duration {
        Duration::from_millis(self.max_retry_after_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "a3f1c2d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90";
    const HASH_B: &str = "0000000000000000000000000000000000000000000000000000000000000001";

    fn minimal() -> String {
        format!(
            r#"
            [providers.openai]
            kind = "openai"
            base_url = "https://api.openai.com/v1"
            api_key_env = "OPENAI_API_KEY"

            [[models]]
            name = "fast"
            targets = [{{ provider = "openai", model = "gpt-4.1-mini" }}]

            [[keys]]
            name = "dev"
            key_sha256 = "{HASH_A}"
            requests_per_minute = 60
            "#
        )
    }

    fn env_with_openai(name: &str) -> Option<String> {
        (name == "OPENAI_API_KEY").then(|| "sk-test".to_string())
    }

    fn problems(text: &str) -> Vec<String> {
        match Config::from_toml_with_env(text, env_with_openai) {
            Err(ConfigError::Invalid(problems)) => problems,
            Err(other) => panic!("expected validation errors, got {other}"),
            Ok(_) => panic!("expected validation errors, config was accepted"),
        }
    }

    #[test]
    fn minimal_config_gets_defaults() {
        let config = Config::from_toml_with_env(&minimal(), env_with_openai).unwrap();
        assert_eq!(config.server.listen, "0.0.0.0:8080");
        assert_eq!(config.retry.max_attempts, 2);
        assert_eq!(config.circuit_breaker.failure_threshold, 5);
        assert!(!config.cache.enabled);
        let provider = &config.providers["openai"];
        assert_eq!(provider.kind, ProviderKind::OpenAi);
        assert_eq!(provider.resolved_api_key.as_deref(), Some("sk-test"));
        assert_eq!(provider.timeout_ms, 60_000);
    }

    #[test]
    fn full_config_parses() {
        let text = format!(
            r#"
            [server]
            listen = "127.0.0.1:9000"

            [retry]
            max_attempts = 3
            base_delay_ms = 100
            max_delay_ms = 1000

            [circuit_breaker]
            failure_threshold = 3
            cooldown_ms = 5000

            [cache]
            enabled = true
            max_entries = 10
            ttl_secs = 60

            [providers.anthropic]
            kind = "anthropic"
            base_url = "https://api.anthropic.com"
            api_key = "sk-ant-test"
            headers = {{ "x-team" = "platform" }}

            [providers.openai]
            kind = "openai"
            base_url = "https://api.openai.com/v1"
            api_key_env = "OPENAI_API_KEY"

            [[models]]
            name = "smart"
            targets = [
              {{ provider = "anthropic", model = "claude-sonnet-4-5-20250929", timeout_ms = 30000, drop_params = ["temperature", "top_p"] }},
              {{ provider = "openai", model = "gpt-4.1" }},
            ]

            [prices."claude-sonnet-4-5-20250929"]
            input_per_mtok = 2.0
            output_per_mtok = 10.0
            cached_input_per_mtok = 0.2

            [[keys]]
            name = "alice"
            key_sha256 = "{HASH_A}"
            requests_per_minute = 120
            burst = 10
            daily_budget_usd = 5.0
            "#
        );
        let config = Config::from_toml_with_env(&text, env_with_openai).unwrap();
        assert_eq!(config.models[0].targets.len(), 2);
        assert_eq!(config.models[0].targets[0].timeout_ms, Some(30_000));
        assert_eq!(
            config.models[0].targets[0].drop_params,
            ["temperature", "top_p"]
        );
        assert_eq!(
            config.prices["claude-sonnet-4-5-20250929"].cached_input_per_mtok,
            Some(0.2)
        );
        assert_eq!(config.keys[0].daily_budget_usd, Some(5.0));
        assert_eq!(
            config.providers["anthropic"].anthropic_version,
            "2023-06-01"
        );
        assert_eq!(config.providers["anthropic"].headers["x-team"], "platform");
        // gpt-4.1 has no price entry, and anthropic uses a literal key.
        assert_eq!(config.warnings().len(), 2);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let text = minimal().replace(
            "requests_per_minute = 60",
            "requests_per_minute = 60\nrpm = 5",
        );
        let err = Config::from_toml_with_env(&text, env_with_openai).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        assert!(err.to_string().contains("rpm"), "{err}");
    }

    #[test]
    fn unknown_provider_kind_is_a_parse_error() {
        let text = minimal().replace(r#"kind = "openai""#, r#"kind = "gemini""#);
        let err = Config::from_toml_with_env(&text, env_with_openai).unwrap_err();
        assert!(err.to_string().contains("gemini"), "{err}");
    }

    #[test]
    fn missing_env_var_is_reported() {
        let err = Config::from_toml_with_env(&minimal(), |_| None).unwrap_err();
        assert!(
            err.to_string().contains("OPENAI_API_KEY is not set"),
            "{err}"
        );
    }

    #[test]
    fn route_to_unknown_provider_is_reported() {
        let text = minimal().replace(r#"provider = "openai""#, r#"provider = "opnai""#);
        let problems = problems(&text);
        assert_eq!(problems.len(), 1);
        assert!(
            problems[0].contains(r#"unknown provider "opnai" (known: openai)"#),
            "{problems:?}"
        );
    }

    #[test]
    fn plaintext_key_instead_of_hash_is_reported() {
        let text = minimal().replace(HASH_A, "sk-gw-my-secret");
        let problems = problems(&text);
        assert!(problems[0].contains("64 hex characters"), "{problems:?}");
    }

    #[test]
    fn all_problems_are_collected() {
        let text = format!(
            r#"
            [server]
            listen = "not-an-address"

            [retry]
            max_attempts = 0
            base_delay_ms = 5000
            max_delay_ms = 100

            [providers.openai]
            kind = "openai"
            base_url = "ftp://example.com"

            [providers.claude]
            kind = "anthropic"
            base_url = "https://api.anthropic.com"

            [[models]]
            name = "a"
            targets = []

            [[models]]
            name = "a"
            targets = [{{ provider = "openai", model = "x", drop_params = ["messages"] }}]

            [prices.x]
            input_per_mtok = -1.0
            output_per_mtok = 1.0

            [[keys]]
            name = "k"
            key_sha256 = "{HASH_A}"
            requests_per_minute = 0
            daily_budget_usd = 0.0

            [[keys]]
            name = "k"
            key_sha256 = "{HASH_A}"
            requests_per_minute = 1
            "#
        );
        let problems = problems(&text);
        let expected = [
            "server.listen",
            "retry.max_attempts",
            "retry.base_delay_ms (5000) is larger",
            "scheme must be http or https",
            "anthropic provider needs api_key",
            "needs at least one target",
            "defined more than once",
            "\"messages\" cannot be dropped",
            "input_per_mtok must be a non-negative number",
            "requests_per_minute must be greater than 0",
            "daily_budget_usd must be a positive number",
            "key name is used more than once",
            "same key hash appears more than once",
        ];
        for needle in expected {
            assert!(
                problems.iter().any(|p| p.contains(needle)),
                "missing problem containing {needle:?} in {problems:#?}"
            );
        }
        assert_eq!(problems.len(), expected.len(), "{problems:#?}");
    }

    #[test]
    fn empty_config_explains_what_is_missing() {
        let problems = problems("");
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems[0].contains("no providers"));
        assert!(problems[1].contains("no models"));
        assert!(problems[2].contains("no API keys"));
    }

    #[test]
    fn invalid_error_lists_every_problem() {
        let err = ConfigError::Invalid(vec!["first".into(), "second".into()]);
        assert_eq!(
            err.to_string(),
            "invalid configuration (2 problems):\n  - first\n  - second"
        );
    }

    #[test]
    fn debug_output_redacts_api_keys() {
        let text = minimal().replace(
            r#"api_key_env = "OPENAI_API_KEY""#,
            r#"api_key = "sk-secret""#,
        );
        let config = Config::from_toml_with_env(&text, |_| None).unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret"), "{debug}");
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn example_config_in_the_repo_is_valid() {
        let config =
            Config::from_toml_with_env(include_str!("../gateway.toml"), |_| Some("sk-test".into()))
                .unwrap();
        assert_eq!(config.warnings(), Vec::<String>::new());
        assert!(config.models.iter().any(|m| m.name == "smart"));
    }

    #[test]
    fn distinct_hashes_are_accepted() {
        let text = format!(
            "{}\n[[keys]]\nname = \"other\"\nkey_sha256 = \"{HASH_B}\"\nrequests_per_minute = 1\n",
            minimal()
        );
        let config = Config::from_toml_with_env(&text, env_with_openai).unwrap();
        assert_eq!(config.keys.len(), 2);
    }
}
