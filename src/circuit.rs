//! Per-provider circuit breaker.
//!
//! ```text
//!            failures >= threshold                cooldown elapsed
//!   Closed ─────────────────────────▶ Open ─────────────────────────▶ HalfOpen
//!     ▲                                ▲                                 │
//!     │                                └────────── probe fails ──────────┤
//!     └──────────────────────────────────────────── probe succeeds ──────┘
//! ```
//!
//! While open, callers are turned away immediately so a dead provider costs
//! nothing. Once the cooldown has passed, exactly one request is let through
//! as a probe. Its result decides whether the circuit closes or opens again.
//!
//! Callers hold a [`Permit`] for the duration of an attempt and report the
//! outcome on it. If a permit is dropped without an outcome (the client went
//! away mid-request), a probe slot is released so the breaker can't get stuck
//! half-open.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::clock::Clock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitState {
    /// Numeric value for the Prometheus gauge.
    pub fn as_gauge(self) -> i64 {
        match self {
            CircuitState::Closed => 0,
            CircuitState::Open => 1,
            CircuitState::HalfOpen => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half_open",
        }
    }
}

#[derive(Debug)]
struct Inner {
    state: CircuitState,
    consecutive_failures: u32,
    opened_at: Option<Instant>,
    probe_in_flight: bool,
}

pub struct CircuitBreaker {
    inner: Mutex<Inner>,
    failure_threshold: u32,
    cooldown: Duration,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("inner", &self.inner)
            .field("failure_threshold", &self.failure_threshold)
            .field("cooldown", &self.cooldown)
            .finish()
    }
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown: Duration, clock: Arc<dyn Clock>) -> Self {
        assert!(failure_threshold > 0);
        Self {
            inner: Mutex::new(Inner {
                state: CircuitState::Closed,
                consecutive_failures: 0,
                opened_at: None,
                probe_in_flight: false,
            }),
            failure_threshold,
            cooldown,
            clock,
        }
    }

    /// The state as a caller would see it right now. An open circuit whose
    /// cooldown has passed reports `HalfOpen`, since the next request will be
    /// let through.
    pub fn state(&self) -> CircuitState {
        let inner = self.inner.lock().unwrap();
        match inner.state {
            CircuitState::Open if self.cooldown_elapsed(&inner) => CircuitState::HalfOpen,
            state => state,
        }
    }

    fn cooldown_elapsed(&self, inner: &Inner) -> bool {
        inner
            .opened_at
            .is_some_and(|at| self.clock.now().saturating_duration_since(at) >= self.cooldown)
    }

    /// Asks to send a request. `None` means the circuit is open (or a probe
    /// is already in flight) and this upstream should be skipped.
    pub fn try_acquire(self: &Arc<Self>) -> Option<Permit> {
        let mut inner = self.inner.lock().unwrap();
        let probe = match inner.state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                if !self.cooldown_elapsed(&inner) {
                    return None;
                }
                inner.state = CircuitState::HalfOpen;
                inner.probe_in_flight = true;
                true
            }
            CircuitState::HalfOpen => {
                if inner.probe_in_flight {
                    return None;
                }
                inner.probe_in_flight = true;
                true
            }
        };
        Some(Permit {
            breaker: Arc::clone(self),
            probe,
            resolved: false,
        })
    }

    /// Time until an open circuit will allow a probe. Zero if not open.
    pub fn retry_in(&self) -> Duration {
        let inner = self.inner.lock().unwrap();
        match (inner.state, inner.opened_at) {
            (CircuitState::Open, Some(at)) => self
                .cooldown
                .saturating_sub(self.clock.now().saturating_duration_since(at)),
            _ => Duration::ZERO,
        }
    }

    fn on_success(&self, probe: bool) -> Option<CircuitState> {
        let mut inner = self.inner.lock().unwrap();
        match inner.state {
            CircuitState::Closed => {
                inner.consecutive_failures = 0;
                None
            }
            CircuitState::HalfOpen if probe => {
                inner.state = CircuitState::Closed;
                inner.consecutive_failures = 0;
                inner.opened_at = None;
                inner.probe_in_flight = false;
                Some(CircuitState::Closed)
            }
            // A request that was admitted before the circuit opened finished
            // late. One stale success is not evidence the upstream recovered.
            _ => None,
        }
    }

    fn on_failure(&self, probe: bool) -> Option<CircuitState> {
        let mut inner = self.inner.lock().unwrap();
        match inner.state {
            CircuitState::Closed => {
                inner.consecutive_failures += 1;
                if inner.consecutive_failures >= self.failure_threshold {
                    inner.state = CircuitState::Open;
                    inner.opened_at = Some(self.clock.now());
                    Some(CircuitState::Open)
                } else {
                    None
                }
            }
            CircuitState::HalfOpen if probe => {
                inner.state = CircuitState::Open;
                inner.opened_at = Some(self.clock.now());
                inner.probe_in_flight = false;
                Some(CircuitState::Open)
            }
            _ => None,
        }
    }

    fn on_abandoned(&self, probe: bool) {
        if probe {
            let mut inner = self.inner.lock().unwrap();
            if inner.state == CircuitState::HalfOpen {
                inner.probe_in_flight = false;
            }
        }
    }
}

/// Permission to send one request. Report the outcome with
/// [`Permit::success`] or [`Permit::failure`]; each returns the new state if
/// the circuit changed state.
#[must_use = "report the outcome with success() or failure()"]
pub struct Permit {
    breaker: Arc<CircuitBreaker>,
    probe: bool,
    resolved: bool,
}

impl Permit {
    pub fn is_probe(&self) -> bool {
        self.probe
    }

    pub fn success(mut self) -> Option<CircuitState> {
        self.resolved = true;
        self.breaker.on_success(self.probe)
    }

    pub fn failure(mut self) -> Option<CircuitState> {
        self.resolved = true;
        self.breaker.on_failure(self.probe)
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if !self.resolved {
            self.breaker.on_abandoned(self.probe);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;

    const COOLDOWN: Duration = Duration::from_secs(30);

    fn breaker(threshold: u32) -> (Arc<CircuitBreaker>, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let breaker = Arc::new(CircuitBreaker::new(threshold, COOLDOWN, clock.clone()));
        (breaker, clock)
    }

    fn fail(breaker: &Arc<CircuitBreaker>) -> Option<CircuitState> {
        breaker.try_acquire().expect("should be admitted").failure()
    }

    fn succeed(breaker: &Arc<CircuitBreaker>) -> Option<CircuitState> {
        breaker.try_acquire().expect("should be admitted").success()
    }

    #[test]
    fn opens_after_threshold_consecutive_failures() {
        let (b, _) = breaker(3);
        assert_eq!(fail(&b), None);
        assert_eq!(fail(&b), None);
        assert_eq!(b.state(), CircuitState::Closed);
        assert_eq!(fail(&b), Some(CircuitState::Open));
        assert_eq!(b.state(), CircuitState::Open);
        assert!(b.try_acquire().is_none());
    }

    #[test]
    fn success_resets_the_failure_count() {
        let (b, _) = breaker(3);
        fail(&b);
        fail(&b);
        succeed(&b);
        fail(&b);
        fail(&b);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn stays_open_until_cooldown_passes() {
        let (b, clock) = breaker(1);
        fail(&b);
        clock.advance(COOLDOWN - Duration::from_millis(1));
        assert!(b.try_acquire().is_none());
        assert_eq!(b.retry_in(), Duration::from_millis(1));
        clock.advance(Duration::from_millis(1));
        assert_eq!(b.state(), CircuitState::HalfOpen);
        assert_eq!(b.retry_in(), Duration::ZERO);
    }

    #[test]
    fn half_open_admits_a_single_probe() {
        let (b, clock) = breaker(1);
        fail(&b);
        clock.advance(COOLDOWN);
        let probe = b.try_acquire().expect("probe should be admitted");
        assert!(probe.is_probe());
        assert!(
            b.try_acquire().is_none(),
            "second request must wait for the probe"
        );
        assert_eq!(probe.success(), Some(CircuitState::Closed));
    }

    #[test]
    fn successful_probe_closes_the_circuit() {
        let (b, clock) = breaker(2);
        fail(&b);
        fail(&b);
        clock.advance(COOLDOWN);
        assert_eq!(succeed(&b), Some(CircuitState::Closed));
        assert_eq!(b.state(), CircuitState::Closed);
        // The failure count starts from zero again.
        assert_eq!(fail(&b), None);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn failed_probe_reopens_for_a_full_cooldown() {
        let (b, clock) = breaker(1);
        fail(&b);
        clock.advance(COOLDOWN);
        assert_eq!(fail(&b), Some(CircuitState::Open));
        assert_eq!(b.state(), CircuitState::Open);
        clock.advance(COOLDOWN / 2);
        assert!(b.try_acquire().is_none());
        clock.advance(COOLDOWN / 2);
        assert!(b.try_acquire().is_some());
    }

    #[test]
    fn abandoned_probe_frees_the_slot() {
        let (b, clock) = breaker(1);
        fail(&b);
        clock.advance(COOLDOWN);
        let probe = b.try_acquire().unwrap();
        drop(probe);
        assert_eq!(b.state(), CircuitState::HalfOpen);
        assert!(b.try_acquire().unwrap().is_probe());
    }

    #[test]
    fn late_results_from_before_opening_are_ignored() {
        let (b, clock) = breaker(1);
        let slow = b.try_acquire().unwrap();
        fail(&b);
        assert_eq!(b.state(), CircuitState::Open);
        // The slow request finishes successfully after the circuit opened.
        assert_eq!(slow.success(), None);
        assert_eq!(b.state(), CircuitState::Open);

        clock.advance(COOLDOWN);
        let probe = b.try_acquire().unwrap();
        let slow = {
            // Can't get a normal permit while half-open, so simulate a late
            // failure from a request admitted while closed.
            Permit {
                breaker: Arc::clone(&b),
                probe: false,
                resolved: false,
            }
        };
        assert_eq!(slow.failure(), None);
        assert_eq!(b.state(), CircuitState::HalfOpen);
        assert_eq!(probe.success(), Some(CircuitState::Closed));
    }

    #[test]
    fn gauge_values_are_distinct() {
        let values = [
            CircuitState::Closed.as_gauge(),
            CircuitState::Open.as_gauge(),
            CircuitState::HalfOpen.as_gauge(),
        ];
        assert_eq!(values, [0, 1, 2]);
    }
}
