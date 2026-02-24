//! In-memory LRU cache with TTL for deterministic, non-streaming responses.
//!
//! Only requests with `temperature: 0` are cached, since those are the only
//! ones where a client could reasonably expect the same answer twice. The key
//! is a SHA-256 over a canonical form of the request body, so field order and
//! `0` vs `0.0` don't cause misses.
//!
//! Entries are scoped per gateway key. Sharing across keys would let one
//! tenant learn, from a fast response, that another tenant sent the same
//! prompt.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use lru::LruCache;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::clock::Clock;

pub type CacheKey = [u8; 32];

/// Fields that don't change a non-streaming response and so are left out of
/// the key.
const IGNORED_FIELDS: &[&str] = &["stream", "stream_options"];

#[derive(Debug, Clone, PartialEq)]
pub struct CachedResponse {
    pub body: Bytes,
    pub provider: String,
    pub model: String,
}

struct Entry {
    value: Arc<CachedResponse>,
    expires_at: Instant,
}

pub struct ResponseCache {
    entries: Mutex<LruCache<CacheKey, Entry>>,
    ttl: Duration,
    clock: Arc<dyn Clock>,
}

impl ResponseCache {
    pub fn new(max_entries: usize, ttl: Duration, clock: Arc<dyn Clock>) -> Self {
        let capacity = NonZeroUsize::new(max_entries).expect("cache size must be positive");
        Self {
            entries: Mutex::new(LruCache::new(capacity)),
            ttl,
            clock,
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<Arc<CachedResponse>> {
        let now = self.clock.now();
        let mut entries = self.entries.lock().unwrap();
        match entries.get(key) {
            Some(entry) if entry.expires_at > now => Some(Arc::clone(&entry.value)),
            Some(_) => {
                entries.pop(key);
                None
            }
            None => None,
        }
    }

    pub fn insert(&self, key: CacheKey, value: CachedResponse) {
        let entry = Entry {
            value: Arc::new(value),
            expires_at: self.clock.now() + self.ttl,
        };
        self.entries.lock().unwrap().put(key, entry);
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A request can be served from cache if it isn't streaming and asks for
/// `temperature: 0`.
pub fn is_cacheable(body: &Value) -> bool {
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let temperature = body.get("temperature").and_then(Value::as_f64);
    !streaming && temperature == Some(0.0)
}

pub fn cache_key(scope: &str, body: &Value) -> CacheKey {
    let mut canonical = String::with_capacity(256);
    match body {
        Value::Object(map) => {
            let filtered: serde_json::Map<String, Value> = map
                .iter()
                .filter(|(k, _)| !IGNORED_FIELDS.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            write_canonical(&Value::Object(filtered), &mut canonical);
        }
        other => write_canonical(other, &mut canonical),
    }
    let mut hasher = Sha256::new();
    hasher.update(scope.as_bytes());
    hasher.update([0]);
    hasher.update(canonical.as_bytes());
    hasher.finalize().into()
}

/// Writes JSON with object keys sorted and numbers normalised, so equal
/// requests produce equal bytes. Keys are sorted explicitly instead of relying
/// on `serde_json::Map` being a `BTreeMap`, which stops being true if any
/// crate in the build turns on serde_json's `preserve_order` feature.
pub fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(n, out),
        Value::String(s) => out.push_str(&Value::String(s.clone()).to_string()),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
    }
}

fn write_number(n: &serde_json::Number, out: &mut String) {
    use std::fmt::Write;
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else if let Some(f) = n.as_f64() {
        // Integral floats inside the exactly-representable range print as
        // integers, so `1.0` and `1` hash the same.
        if f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0 {
            let _ = write!(out, "{}", f as i64);
        } else {
            // Debug gives the shortest round-trip form, with an exponent for
            // very large or small values.
            let _ = write!(out, "{f:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use serde_json::json;

    fn canonical(value: &Value) -> String {
        let mut out = String::new();
        write_canonical(value, &mut out);
        out
    }

    fn response(body: &str) -> CachedResponse {
        CachedResponse {
            body: Bytes::from(body.to_string()),
            provider: "p".into(),
            model: "m".into(),
        }
    }

    #[test]
    fn canonical_form_sorts_keys_at_every_level() {
        let value = json!({"b": 1, "a": {"z": [3, {"y": true, "x": null}], "c": "s"}});
        assert_eq!(
            canonical(&value),
            r#"{"a":{"c":"s","z":[3,{"x":null,"y":true}]},"b":1}"#
        );
    }

    #[test]
    fn canonical_form_normalises_numbers_and_escapes_strings() {
        assert_eq!(
            canonical(&json!([0.0, 1.0, -2.0, 0.5, 1e300])),
            "[0,1,-2,0.5,1e300]"
        );
        assert_eq!(
            canonical(&json!("quote\" and\nnewline")),
            r#""quote\" and\nnewline""#
        );
    }

    #[test]
    fn key_ignores_field_order_and_number_spelling() {
        let a: Value = serde_json::from_str(
            r#"{"model":"m","temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let b: Value = serde_json::from_str(
            r#"{"messages":[{"content":"hi","role":"user"}],"temperature":0.0,"model":"m","stream":false}"#,
        )
        .unwrap();
        assert_eq!(cache_key("k", &a), cache_key("k", &b));
    }

    #[test]
    fn key_changes_with_content_and_scope() {
        let a = json!({"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hi"}]});
        let b = json!({"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hi!"}]});
        let c = json!({"model": "m", "temperature": 0, "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]});
        assert_ne!(cache_key("k", &a), cache_key("k", &b));
        assert_ne!(cache_key("k", &a), cache_key("k", &c));
        assert_ne!(cache_key("alice", &a), cache_key("bob", &a));
    }

    #[test]
    fn only_deterministic_non_streaming_requests_are_cacheable() {
        assert!(is_cacheable(&json!({"temperature": 0})));
        assert!(is_cacheable(&json!({"temperature": 0.0, "stream": false})));
        assert!(!is_cacheable(&json!({"temperature": 0, "stream": true})));
        assert!(!is_cacheable(&json!({"temperature": 0.2})));
        // The OpenAI default temperature is 1.
        assert!(!is_cacheable(&json!({})));
        assert!(!is_cacheable(&json!({"temperature": "0"})));
    }

    #[test]
    fn entries_expire_after_ttl() {
        let clock = Arc::new(ManualClock::new());
        let cache = ResponseCache::new(10, Duration::from_secs(60), clock.clone());
        let key = cache_key("k", &json!({"a": 1}));
        cache.insert(key, response("one"));
        assert_eq!(cache.get(&key).unwrap().body, "one");
        clock.advance(Duration::from_secs(59));
        assert!(cache.get(&key).is_some());
        clock.advance(Duration::from_secs(1));
        assert!(cache.get(&key).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn least_recently_used_entry_is_evicted() {
        let clock = Arc::new(ManualClock::new());
        let cache = ResponseCache::new(2, Duration::from_secs(60), clock);
        let [a, b, c] = ["a", "b", "c"].map(|s| cache_key("k", &json!(s)));
        cache.insert(a, response("a"));
        cache.insert(b, response("b"));
        cache.get(&a); // a is now more recent than b
        cache.insert(c, response("c"));
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&b).is_none());
        assert!(cache.get(&c).is_some());
        assert_eq!(cache.len(), 2);
    }
}
