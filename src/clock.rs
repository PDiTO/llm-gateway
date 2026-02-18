//! Time source abstraction so that the circuit breaker, rate limiter, cache
//! and budgets can be tested without sleeping.

use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub trait Clock: Send + Sync + 'static {
    /// Monotonic time, used for intervals.
    fn now(&self) -> Instant;
    /// Wall-clock time as a duration since the Unix epoch, used for daily
    /// budgets and `Retry-After` dates.
    fn unix_time(&self) -> Duration;

    /// Days since the Unix epoch in UTC. Budgets reset when this changes.
    fn utc_day(&self) -> u64 {
        self.unix_time().as_secs() / 86_400
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn unix_time(&self) -> Duration {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    }
}

/// A clock that only moves when told to. Both the monotonic and wall-clock
/// readings advance together.
#[derive(Debug)]
pub struct ManualClock {
    start: Instant,
    start_unix: Duration,
    elapsed: Mutex<Duration>,
}

impl ManualClock {
    pub fn new() -> Self {
        Self::starting_at_unix(SystemClock.unix_time())
    }

    /// Starts the wall clock at a fixed point, for tests that care about day
    /// boundaries.
    pub fn starting_at_unix(start_unix: Duration) -> Self {
        Self {
            start: Instant::now(),
            start_unix,
            elapsed: Mutex::new(Duration::ZERO),
        }
    }

    pub fn advance(&self, by: Duration) {
        *self.elapsed.lock().unwrap() += by;
    }

    fn elapsed(&self) -> Duration {
        *self.elapsed.lock().unwrap()
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.start + self.elapsed()
    }

    fn unix_time(&self) -> Duration {
        self.start_unix + self.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_moves_only_when_advanced() {
        let clock = ManualClock::starting_at_unix(Duration::from_secs(86_400 * 10 + 86_399));
        let t0 = clock.now();
        assert_eq!(clock.now(), t0);
        assert_eq!(clock.utc_day(), 10);

        clock.advance(Duration::from_secs(1));
        assert_eq!(clock.now() - t0, Duration::from_secs(1));
        assert_eq!(clock.utc_day(), 11);
    }
}
