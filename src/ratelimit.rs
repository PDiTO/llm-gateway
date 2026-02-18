//! Token bucket rate limiter.
//!
//! Each gateway key owns one bucket. The bucket holds up to `capacity` tokens
//! and refills continuously at `refill_per_sec`. A request takes one token; if
//! none is available the caller gets back how long to wait for the next one,
//! which becomes the `Retry-After` header.

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_sec: f64,
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// A bucket that starts full.
    pub fn new(capacity: u32, refill_per_sec: f64, now: Instant) -> Self {
        assert!(capacity > 0, "bucket capacity must be positive");
        assert!(refill_per_sec > 0.0, "refill rate must be positive");
        Self {
            capacity: f64::from(capacity),
            refill_per_sec,
            tokens: f64::from(capacity),
            last_refill: now,
        }
    }

    /// `requests_per_minute` sustained, with bursts up to `burst` (defaulting
    /// to one minute's worth).
    pub fn per_minute(requests_per_minute: u32, burst: Option<u32>, now: Instant) -> Self {
        Self::new(
            burst.unwrap_or(requests_per_minute),
            f64::from(requests_per_minute) / 60.0,
            now,
        )
    }

    fn refill(&mut self, now: Instant) {
        // A clock that goes backwards (it shouldn't, Instant is monotonic) is
        // treated as no time passing.
        let elapsed = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        self.last_refill = now;
    }

    /// Takes one token, or returns how long until one will be available.
    pub fn try_acquire(&mut self, now: Instant) -> Result<(), Duration> {
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            let missing = 1.0 - self.tokens;
            Err(Duration::from_secs_f64(missing / self.refill_per_sec))
        }
    }

    pub fn available(&mut self, now: Instant) -> f64 {
        self.refill(now);
        self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_full_and_drains() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::per_minute(3, None, t0);
        for _ in 0..3 {
            assert!(bucket.try_acquire(t0).is_ok());
        }
        assert!(bucket.try_acquire(t0).is_err());
    }

    #[test]
    fn reports_wait_until_next_token() {
        let t0 = Instant::now();
        // 60/min is one token per second.
        let mut bucket = TokenBucket::per_minute(60, Some(1), t0);
        bucket.try_acquire(t0).unwrap();
        let wait = bucket.try_acquire(t0).unwrap_err();
        assert_eq!(wait, Duration::from_secs(1));

        let wait = bucket
            .try_acquire(t0 + Duration::from_millis(250))
            .unwrap_err();
        assert!((wait.as_secs_f64() - 0.75).abs() < 1e-9, "{wait:?}");
    }

    #[test]
    fn refills_over_time() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::per_minute(120, Some(2), t0);
        bucket.try_acquire(t0).unwrap();
        bucket.try_acquire(t0).unwrap();
        assert!(bucket.try_acquire(t0).is_err());

        // Two tokens per second, so 500ms buys exactly one request.
        let t1 = t0 + Duration::from_millis(500);
        assert!(bucket.try_acquire(t1).is_ok());
        assert!(bucket.try_acquire(t1).is_err());
    }

    #[test]
    fn never_exceeds_capacity() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::per_minute(60, Some(5), t0);
        let later = t0 + Duration::from_secs(3_600);
        assert_eq!(bucket.available(later), 5.0);
        for _ in 0..5 {
            bucket.try_acquire(later).unwrap();
        }
        assert!(bucket.try_acquire(later).is_err());
    }

    #[test]
    fn failed_attempts_do_not_consume_tokens() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::per_minute(60, Some(1), t0);
        bucket.try_acquire(t0).unwrap();
        for ms in [100, 200, 300] {
            assert!(bucket.try_acquire(t0 + Duration::from_millis(ms)).is_err());
        }
        assert!(bucket.try_acquire(t0 + Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn burst_is_separate_from_rate() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::per_minute(6, Some(3), t0);
        for _ in 0..3 {
            bucket.try_acquire(t0).unwrap();
        }
        // 6/min is one token every 10 seconds.
        assert_eq!(bucket.try_acquire(t0).unwrap_err(), Duration::from_secs(10));
    }
}
