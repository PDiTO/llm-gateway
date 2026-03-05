//! Gateway API keys, per-key rate limits and daily spend budgets.
//!
//! The config holds only SHA-256 hashes of keys, so a leaked config file
//! doesn't leak usable keys. An incoming key is hashed and looked up by hash,
//! which also means lookup time doesn't depend on how much of a guessed key
//! is right. Keys are long random strings, so an unsalted fast hash is enough
//! here; this is not a password store.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;
use sha2::{Digest, Sha256};

use crate::clock::Clock;
use crate::config::ApiKeyConfig;
use crate::error::ApiError;
use crate::ratelimit::TokenBucket;

pub fn hash_key(key: &str) -> String {
    hex::encode(Sha256::digest(key.as_bytes()))
}

/// A new random key: `sk-gw-` followed by 32 random bytes in hex.
pub fn generate_key() -> String {
    let bytes: [u8; 32] = rand::random();
    format!("sk-gw-{}", hex::encode(bytes))
}

#[derive(Debug)]
pub struct KeyState {
    pub name: String,
    bucket: Mutex<TokenBucket>,
    daily_budget_usd: Option<f64>,
    spend: Mutex<DailySpend>,
}

#[derive(Debug, Clone, Copy)]
struct DailySpend {
    day: u64,
    usd: f64,
}

impl KeyState {
    /// Takes a token from the key's bucket.
    pub fn check_rate_limit(&self, clock: &dyn Clock) -> Result<(), ApiError> {
        self.bucket
            .lock()
            .unwrap()
            .try_acquire(clock.now())
            .map_err(|wait| ApiError::rate_limited(&self.name, wait))
    }

    /// Rejects the request if today's spend has reached the budget. Cost is
    /// only known after a response, so the request that crosses the line is
    /// allowed and the next one is refused.
    pub fn check_budget(&self, clock: &dyn Clock) -> Result<(), ApiError> {
        let Some(budget) = self.daily_budget_usd else {
            return Ok(());
        };
        let spent = self.spent_today(clock);
        if spent >= budget {
            Err(ApiError::budget_exceeded(&self.name, spent, budget))
        } else {
            Ok(())
        }
    }

    pub fn charge(&self, usd: f64, clock: &dyn Clock) {
        if usd <= 0.0 || !usd.is_finite() {
            return;
        }
        let today = clock.utc_day();
        let mut spend = self.spend.lock().unwrap();
        if spend.day != today {
            *spend = DailySpend {
                day: today,
                usd: 0.0,
            };
        }
        spend.usd += usd;
    }

    pub fn spent_today(&self, clock: &dyn Clock) -> f64 {
        let spend = self.spend.lock().unwrap();
        if spend.day == clock.utc_day() {
            spend.usd
        } else {
            0.0
        }
    }

    pub fn daily_budget_usd(&self) -> Option<f64> {
        self.daily_budget_usd
    }
}

#[derive(Debug)]
pub struct KeyStore {
    by_hash: HashMap<String, Arc<KeyState>>,
}

impl KeyStore {
    pub fn new(keys: &[ApiKeyConfig], clock: &dyn Clock) -> Self {
        let now = clock.now();
        let today = clock.utc_day();
        let by_hash = keys
            .iter()
            .map(|key| {
                let state = KeyState {
                    name: key.name.clone(),
                    bucket: Mutex::new(TokenBucket::per_minute(
                        key.requests_per_minute,
                        key.burst,
                        now,
                    )),
                    daily_budget_usd: key.daily_budget_usd,
                    spend: Mutex::new(DailySpend {
                        day: today,
                        usd: 0.0,
                    }),
                };
                (key.key_sha256.to_ascii_lowercase(), Arc::new(state))
            })
            .collect();
        Self { by_hash }
    }

    /// Finds the key from `Authorization: Bearer <key>` or, for clients
    /// written against Anthropic's API, `x-api-key: <key>`.
    pub fn authenticate(&self, headers: &HeaderMap) -> Result<Arc<KeyState>, ApiError> {
        let presented = headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| {
                v.strip_prefix("Bearer ")
                    .or_else(|| v.strip_prefix("bearer "))
            })
            .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .ok_or_else(|| {
                ApiError::unauthorized("missing API key; send it as `Authorization: Bearer <key>`")
            })?;
        self.by_hash
            .get(&hash_key(presented))
            .cloned()
            .ok_or_else(|| ApiError::unauthorized("invalid API key"))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<KeyState>> {
        self.by_hash.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use axum::http::{HeaderValue, StatusCode};
    use std::time::Duration;

    fn key(name: &str, plaintext: &str, rpm: u32, budget: Option<f64>) -> ApiKeyConfig {
        ApiKeyConfig {
            name: name.into(),
            key_sha256: hash_key(plaintext),
            requests_per_minute: rpm,
            burst: None,
            daily_budget_usd: budget,
        }
    }

    fn bearer(key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).unwrap(),
        );
        headers
    }

    #[test]
    fn hash_is_sha256_hex() {
        assert_eq!(
            hash_key("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn generated_keys_are_unique_and_prefixed() {
        let a = generate_key();
        let b = generate_key();
        assert!(a.starts_with("sk-gw-"));
        assert_eq!(a.len(), 6 + 64);
        assert_ne!(a, b);
    }

    #[test]
    fn authenticates_by_bearer_or_x_api_key() {
        let clock = ManualClock::new();
        let store = KeyStore::new(&[key("alice", "sk-alice", 10, None)], &clock);
        assert_eq!(
            store.authenticate(&bearer("sk-alice")).unwrap().name,
            "alice"
        );

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("sk-alice"));
        assert_eq!(store.authenticate(&headers).unwrap().name, "alice");
    }

    #[test]
    fn rejects_missing_and_wrong_keys() {
        let clock = ManualClock::new();
        let store = KeyStore::new(&[key("alice", "sk-alice", 10, None)], &clock);
        let missing = store.authenticate(&HeaderMap::new()).unwrap_err();
        assert_eq!(missing.status, StatusCode::UNAUTHORIZED);
        assert!(missing.message.contains("missing"));
        let wrong = store.authenticate(&bearer("sk-bob")).unwrap_err();
        assert_eq!(wrong.message, "invalid API key");
        // The hash itself is not a valid key.
        assert!(store.authenticate(&bearer(&hash_key("sk-alice"))).is_err());
    }

    #[test]
    fn rate_limit_is_per_key() {
        let clock = ManualClock::new();
        let store = KeyStore::new(&[key("a", "ka", 2, None), key("b", "kb", 2, None)], &clock);
        let a = store.authenticate(&bearer("ka")).unwrap();
        let b = store.authenticate(&bearer("kb")).unwrap();
        a.check_rate_limit(&clock).unwrap();
        a.check_rate_limit(&clock).unwrap();
        let err = a.check_rate_limit(&clock).unwrap_err();
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.retry_after, Some(Duration::from_secs(30)));
        b.check_rate_limit(&clock).unwrap();

        clock.advance(Duration::from_secs(30));
        a.check_rate_limit(&clock).unwrap();
    }

    #[test]
    fn budget_blocks_after_it_is_spent_and_resets_at_midnight_utc() {
        // One minute before midnight UTC.
        let clock = ManualClock::starting_at_unix(Duration::from_secs(20_000 * 86_400 - 60));
        let store = KeyStore::new(&[key("a", "ka", 100, Some(1.0))], &clock);
        let a = store.authenticate(&bearer("ka")).unwrap();

        a.check_budget(&clock).unwrap();
        a.charge(0.6, &clock);
        a.check_budget(&clock).unwrap();
        a.charge(0.6, &clock);
        let err = a.check_budget(&clock).unwrap_err();
        assert_eq!(err.status, StatusCode::PAYMENT_REQUIRED);
        assert_eq!(err.code, Some("budget_exceeded"));
        assert!(err.message.contains("$1.2000 of $1.00"), "{}", err.message);

        clock.advance(Duration::from_secs(60));
        assert_eq!(a.spent_today(&clock), 0.0);
        a.check_budget(&clock).unwrap();
    }

    #[test]
    fn keys_without_budget_are_never_blocked() {
        let clock = ManualClock::new();
        let store = KeyStore::new(&[key("a", "ka", 100, None)], &clock);
        let a = store.authenticate(&bearer("ka")).unwrap();
        a.charge(1e9, &clock);
        a.check_budget(&clock).unwrap();
    }

    #[test]
    fn nonsense_charges_are_ignored() {
        let clock = ManualClock::new();
        let store = KeyStore::new(&[key("a", "ka", 100, Some(1.0))], &clock);
        let a = store.authenticate(&bearer("ka")).unwrap();
        a.charge(-5.0, &clock);
        a.charge(f64::NAN, &clock);
        assert_eq!(a.spent_today(&clock), 0.0);
    }
}
