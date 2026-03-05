//! Prometheus metrics. Each gateway owns its own registry, which keeps tests
//! that run several gateways in one process from seeing each other's counts.

use prometheus::{
    CounterVec, Encoder, GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts,
    Registry, TextEncoder,
};

pub struct Metrics {
    registry: Registry,
    /// Client requests by key, requested model and HTTP status.
    pub requests: IntCounterVec,
    /// Time until response headers were sent. For streams this is time to
    /// first byte, not the length of the stream.
    pub request_duration: HistogramVec,
    /// Every upstream attempt by provider and outcome.
    pub upstream_attempts: IntCounterVec,
    /// Times a request moved on to a later target in its route.
    pub fallbacks: IntCounterVec,
    pub tokens: IntCounterVec,
    pub cost_usd: CounterVec,
    pub cache_lookups: IntCounterVec,
    pub rate_limited: IntCounterVec,
    pub budget_rejected: IntCounterVec,
    pub key_spend_today_usd: GaugeVec,
    pub circuit_state: IntGaugeVec,
    pub circuit_transitions: IntCounterVec,
    /// Streams that ended without any usage from the upstream, so they were
    /// billed at zero. Usually a client hanging up on an OpenAI stream before
    /// the final usage chunk.
    pub unbilled_streams: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        fn counter(registry: &Registry, name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
            let c = IntCounterVec::new(Opts::new(name, help), labels).expect("valid metric");
            registry
                .register(Box::new(c.clone()))
                .expect("unique metric");
            c
        }

        let requests = counter(
            &registry,
            "llm_gateway_requests_total",
            "Client requests by gateway key, requested model and HTTP status.",
            &["key", "model", "status"],
        );
        let upstream_attempts = counter(
            &registry,
            "llm_gateway_upstream_attempts_total",
            "Upstream attempts by provider and outcome.",
            &["provider", "outcome"],
        );
        let fallbacks = counter(
            &registry,
            "llm_gateway_fallbacks_total",
            "Requests that moved on to a later target in their route.",
            &["model"],
        );
        let tokens = counter(
            &registry,
            "llm_gateway_tokens_total",
            "Tokens by gateway key, upstream model and kind (prompt, completion, cached_prompt).",
            &["key", "model", "kind"],
        );
        let cache_lookups = counter(
            &registry,
            "llm_gateway_cache_lookups_total",
            "Response cache lookups by result (hit, miss).",
            &["result"],
        );
        let rate_limited = counter(
            &registry,
            "llm_gateway_rate_limited_total",
            "Requests rejected by the per-key rate limit.",
            &["key"],
        );
        let budget_rejected = counter(
            &registry,
            "llm_gateway_budget_rejected_total",
            "Requests rejected because the key's daily budget was spent.",
            &["key"],
        );
        let circuit_transitions = counter(
            &registry,
            "llm_gateway_circuit_transitions_total",
            "Circuit breaker state changes by provider and new state.",
            &["provider", "to"],
        );
        let unbilled_streams = counter(
            &registry,
            "llm_gateway_unbilled_streams_total",
            "Streams that ended without usage from the upstream and were billed at zero.",
            &["provider"],
        );

        let cost_usd = CounterVec::new(
            Opts::new(
                "llm_gateway_cost_usd_total",
                "Estimated spend in USD by gateway key and upstream model.",
            ),
            &["key", "model"],
        )
        .expect("valid metric");
        registry
            .register(Box::new(cost_usd.clone()))
            .expect("unique metric");

        let key_spend_today_usd = GaugeVec::new(
            Opts::new(
                "llm_gateway_key_spend_today_usd",
                "Spend so far in the current UTC day by gateway key.",
            ),
            &["key"],
        )
        .expect("valid metric");
        registry
            .register(Box::new(key_spend_today_usd.clone()))
            .expect("unique metric");

        let circuit_state = IntGaugeVec::new(
            Opts::new(
                "llm_gateway_circuit_state",
                "Circuit breaker state by provider: 0 closed, 1 open, 2 half-open.",
            ),
            &["provider"],
        )
        .expect("valid metric");
        registry
            .register(Box::new(circuit_state.clone()))
            .expect("unique metric");

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                "llm_gateway_request_duration_seconds",
                "Time from request to response headers by requested model.",
            )
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
            ]),
            &["model"],
        )
        .expect("valid metric");
        registry
            .register(Box::new(request_duration.clone()))
            .expect("unique metric");

        Self {
            registry,
            requests,
            request_duration,
            upstream_attempts,
            fallbacks,
            tokens,
            cost_usd,
            cache_lookups,
            rate_limited,
            budget_rejected,
            key_spend_today_usd,
            circuit_state,
            circuit_transitions,
            unbilled_streams,
        }
    }

    /// Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = Vec::new();
        TextEncoder::new()
            .encode(&self.registry.gather(), &mut out)
            .expect("encoding to a Vec cannot fail");
        String::from_utf8(out).expect("prometheus output is UTF-8")
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_labelled_counters() {
        let metrics = Metrics::new();
        metrics
            .requests
            .with_label_values(&["alice", "smart", "200"])
            .inc_by(2);
        metrics
            .cost_usd
            .with_label_values(&["alice", "claude-sonnet-4-5-20250929"])
            .inc_by(0.25);
        let text = metrics.render();
        assert!(
            text.contains(
                r#"llm_gateway_requests_total{key="alice",model="smart",status="200"} 2"#
            ),
            "{text}"
        );
        assert!(
            text.contains(
                r#"llm_gateway_cost_usd_total{key="alice",model="claude-sonnet-4-5-20250929"} 0.25"#
            ),
            "{text}"
        );
        assert!(text.contains("# TYPE llm_gateway_requests_total counter"));
    }

    #[test]
    fn registries_are_independent() {
        let a = Metrics::new();
        let b = Metrics::new();
        a.fallbacks.with_label_values(&["m"]).inc();
        assert!(!b.render().contains("llm_gateway_fallbacks_total{"));
    }
}
