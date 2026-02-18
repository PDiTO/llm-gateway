//! Retry delays: exponential backoff with full jitter, and `Retry-After`
//! parsing.

use std::time::{Duration, SystemTime};

use reqwest::header::HeaderMap;

#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    base: Duration,
    max: Duration,
}

impl Backoff {
    pub fn new(base: Duration, max: Duration) -> Self {
        Self { base, max }
    }

    /// Upper bound of the delay before retry number `retry` (0-based):
    /// `base * 2^retry`, capped at `max`.
    pub fn ceiling(&self, retry: u32) -> Duration {
        let factor = 2u32.saturating_pow(retry.min(31));
        self.base.saturating_mul(factor).min(self.max)
    }

    /// "Full jitter": a uniform random delay in `[0, ceiling]`. Spreading
    /// retries out like this stops many clients that failed together from
    /// retrying together.
    pub fn delay(&self, retry: u32) -> Duration {
        self.delay_with(retry, rand::random::<f64>())
    }

    /// Deterministic version of [`Backoff::delay`]; `unit` is clamped to `[0, 1]`.
    pub fn delay_with(&self, retry: u32, unit: f64) -> Duration {
        self.ceiling(retry).mul_f64(unit.clamp(0.0, 1.0))
    }
}

/// Reads how long the upstream asked us to wait. Understands `retry-after-ms`
/// (sent by OpenAI) and both forms of `Retry-After`: delta-seconds and an
/// HTTP date.
pub fn retry_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
    {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| parse_retry_after(v, now))
}

pub fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // Not in the RFC, but some servers send fractional seconds.
    if let Ok(secs) = value.parse::<f64>() {
        return (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs));
    }
    let at = httpdate::parse_http_date(value).ok()?;
    // A date in the past means "retry now".
    Some(at.duration_since(now).unwrap_or(Duration::ZERO))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    fn backoff() -> Backoff {
        Backoff::new(Duration::from_millis(100), Duration::from_secs(1))
    }

    #[test]
    fn ceiling_doubles_then_caps() {
        let b = backoff();
        let ceilings: Vec<u64> = (0..6).map(|n| b.ceiling(n).as_millis() as u64).collect();
        assert_eq!(ceilings, [100, 200, 400, 800, 1000, 1000]);
    }

    #[test]
    fn ceiling_survives_huge_retry_counts() {
        assert_eq!(backoff().ceiling(u32::MAX), Duration::from_secs(1));
    }

    #[test]
    fn jitter_stays_within_bounds() {
        let b = backoff();
        assert_eq!(b.delay_with(2, 0.0), Duration::ZERO);
        assert_eq!(b.delay_with(2, 0.5), Duration::from_millis(200));
        assert_eq!(b.delay_with(2, 1.0), Duration::from_millis(400));
        assert_eq!(b.delay_with(2, 7.0), Duration::from_millis(400));
        for retry in 0..8 {
            for _ in 0..200 {
                assert!(b.delay(retry) <= b.ceiling(retry));
            }
        }
    }

    #[test]
    fn random_delays_are_spread_out() {
        let b = backoff();
        let delays: std::collections::HashSet<_> = (0..50).map(|_| b.delay(3)).collect();
        assert!(delays.len() > 10, "expected jitter, got {delays:?}");
    }

    #[test]
    fn parses_delta_seconds() {
        let now = SystemTime::now();
        assert_eq!(parse_retry_after("7", now), Some(Duration::from_secs(7)));
        assert_eq!(parse_retry_after(" 0 ", now), Some(Duration::ZERO));
        assert_eq!(
            parse_retry_after("1.5", now),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(parse_retry_after("-3", now), None);
        assert_eq!(parse_retry_after("soon", now), None);
    }

    #[test]
    fn parses_http_dates() {
        let now = httpdate::parse_http_date("Wed, 21 Oct 2026 07:28:00 GMT").unwrap();
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2026 07:28:30 GMT", now),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2026 07:27:00 GMT", now),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn prefers_retry_after_ms() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("20"));
        assert_eq!(
            retry_after(&headers, SystemTime::now()),
            Some(Duration::from_secs(20))
        );
        headers.insert("retry-after-ms", HeaderValue::from_static("350"));
        assert_eq!(
            retry_after(&headers, SystemTime::now()),
            Some(Duration::from_millis(350))
        );
        assert_eq!(retry_after(&HeaderMap::new(), SystemTime::now()), None);
    }
}
