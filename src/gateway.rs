//! The request path: authenticate, rate limit, check the budget, try the
//! cache, then walk the route's targets with retries, fallback and circuit
//! breakers until one of them answers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::Value;
use tracing::{info, warn};

use crate::auth::{KeyState, KeyStore};
use crate::backoff::{self, Backoff};
use crate::cache::{self, CachedResponse, ResponseCache};
use crate::circuit::{CircuitBreaker, CircuitState, Permit};
use crate::clock::{Clock, SystemClock};
use crate::config::{Config, RetryConfig, Target};
use crate::cost::PriceTable;
use crate::error::ApiError;
use crate::metrics::Metrics;
use crate::openai::TokenUsage;
use crate::providers::{Provider, StreamRecorder, StreamSummary};

pub const HEADER_PROVIDER: &str = "x-gateway-provider";
pub const HEADER_MODEL: &str = "x-gateway-model";
pub const HEADER_ATTEMPTS: &str = "x-gateway-attempts";
pub const HEADER_CACHE: &str = "x-gateway-cache";
pub const HEADER_COST: &str = "x-gateway-cost-usd";

/// Largest successful non-streaming body the gateway will read from an upstream.
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
/// Largest error body the gateway will read. Error bodies are short; this
/// only guards against an upstream sending something huge with a 5xx.
const MAX_ERROR_BYTES: usize = 64 * 1024;

pub struct Upstream {
    pub provider: Provider,
    pub breaker: Arc<CircuitBreaker>,
}

pub struct Gateway {
    routes: HashMap<String, Vec<Target>>,
    model_names: Vec<String>,
    upstreams: HashMap<String, Upstream>,
    keys: KeyStore,
    prices: PriceTable,
    cache: Option<ResponseCache>,
    pub metrics: Metrics,
    clock: Arc<dyn Clock>,
    http: reqwest::Client,
    retry: RetryConfig,
    backoff: Backoff,
    pub max_body_bytes: usize,
}

/// Labels a finished request is recorded under.
struct RequestLabels {
    key: String,
    model: String,
}

enum Reply {
    Json {
        body: Bytes,
        usage: Option<TokenUsage>,
    },
    /// The permit travels with the stream, so the circuit hears how the
    /// stream ended rather than only that it started.
    Stream(reqwest::Response, Permit),
}

struct Served<'a> {
    upstream: &'a Upstream,
    target: &'a Target,
    reply: Reply,
    attempts: u32,
}

enum AttemptError {
    /// This target couldn't serve the request; another one might.
    Failed {
        reason: String,
        outcome: &'static str,
        retry_after: Option<Duration>,
        /// False when retrying the same upstream is pointless (bad
        /// credentials, unknown model), but another target might still work.
        retry_same_target: bool,
        /// False when the upstream answered sensibly and so shouldn't be
        /// counted as unhealthy (a 404 for a model it doesn't have).
        counts_against_circuit: bool,
        kind: FailureKind,
    },
    /// The upstream rejected the request itself (400, 422...). Other targets
    /// get the same messages and parameters, so this goes straight back to
    /// the client.
    Rejected { status: StatusCode, body: Value },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FailureKind {
    CircuitOpen,
    RateLimited,
    Timeout,
    /// The upstream doesn't know the model.
    NotFound,
    /// The request can't be expressed for this target's API.
    Unsupported,
    Other,
}

enum DispatchError {
    Api(ApiError),
    Rejected {
        status: StatusCode,
        body: Value,
        provider: String,
        attempts: u32,
    },
}

impl Gateway {
    pub fn new(config: Config) -> Self {
        Self::with_clock(config, Arc::new(SystemClock))
    }

    /// Builds a gateway with an injected clock, so tests can move time
    /// forward for circuit breaker cooldowns, rate limits and budgets.
    pub fn with_clock(config: Config, clock: Arc<dyn Clock>) -> Self {
        let cooldown = Duration::from_millis(config.circuit_breaker.cooldown_ms);
        let upstreams = config
            .providers
            .iter()
            .map(|(name, provider)| {
                let upstream = Upstream {
                    provider: Provider::from_config(name, provider),
                    breaker: Arc::new(CircuitBreaker::new(
                        config.circuit_breaker.failure_threshold,
                        cooldown,
                        Arc::clone(&clock),
                    )),
                };
                (name.clone(), upstream)
            })
            .collect();
        let cache = config.cache.enabled.then(|| {
            ResponseCache::new(
                config.cache.max_entries,
                Duration::from_secs(config.cache.ttl_secs),
                Arc::clone(&clock),
            )
        });
        let http = reqwest::Client::builder()
            .tcp_nodelay(true)
            // Redirects would carry provider credentials (x-api-key, custom
            // headers) and request bodies to another host. No LLM API
            // redirects a POST, so treat one as an error instead.
            .redirect(reqwest::redirect::Policy::none())
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("HTTP client configuration is static and valid");
        Self {
            routes: config
                .models
                .iter()
                .map(|m| (m.name.clone(), m.targets.clone()))
                .collect(),
            model_names: config.models.iter().map(|m| m.name.clone()).collect(),
            upstreams,
            keys: KeyStore::new(&config.keys, &*clock),
            prices: PriceTable::new(&config.prices),
            cache,
            metrics: Metrics::new(),
            backoff: Backoff::new(config.retry.base_delay(), config.retry.max_delay()),
            retry: config.retry,
            max_body_bytes: config.server.max_body_bytes,
            http,
            clock,
        }
    }

    pub fn model_names(&self) -> &[String] {
        &self.model_names
    }

    pub fn keys(&self) -> &KeyStore {
        &self.keys
    }

    pub fn clock(&self) -> &dyn Clock {
        &*self.clock
    }

    /// Provider name and current circuit state, sorted by name.
    pub fn circuit_states(&self) -> Vec<(&str, CircuitState)> {
        let mut states: Vec<_> = self
            .upstreams
            .iter()
            .map(|(name, u)| (name.as_str(), u.breaker.state()))
            .collect();
        states.sort_by_key(|(name, _)| *name);
        states
    }

    /// Updates gauges that are computed rather than counted, then renders.
    pub fn render_metrics(&self) -> String {
        for (name, state) in self.circuit_states() {
            self.metrics
                .circuit_state
                .with_label_values(&[name])
                .set(state.as_gauge());
        }
        for key in self.keys.iter() {
            self.metrics
                .key_spend_today_usd
                .with_label_values(&[&key.name])
                .set(key.spent_today(&*self.clock));
        }
        self.metrics.render()
    }

    fn unix_now(&self) -> u64 {
        self.clock.unix_time().as_secs()
    }

    /// Handles `POST /v1/chat/completions` end to end.
    pub async fn chat_completions(
        self: &Arc<Self>,
        request_id: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Response {
        let started = Instant::now();
        let mut labels = RequestLabels {
            key: "-".into(),
            model: "-".into(),
        };
        let response = match self
            .chat_inner(request_id, headers, body, &mut labels)
            .await
        {
            Ok(response) => response,
            Err(err) => {
                if err.status.is_server_error() {
                    warn!(status = err.status.as_u16(), error = %err.message, "request failed");
                } else {
                    info!(status = err.status.as_u16(), error = %err.message, "request rejected");
                }
                err.into_response()
            }
        };
        let status = response.status().as_u16().to_string();
        self.metrics
            .requests
            .with_label_values(&[&labels.key, &labels.model, &status])
            .inc();
        self.metrics
            .request_duration
            .with_label_values(&[&labels.model])
            .observe(started.elapsed().as_secs_f64());
        response
    }

    async fn chat_inner(
        self: &Arc<Self>,
        request_id: &str,
        headers: &HeaderMap,
        body: Bytes,
        labels: &mut RequestLabels,
    ) -> Result<Response, ApiError> {
        let key = self.keys.authenticate(headers)?;
        labels.key = key.name.clone();
        if let Err(err) = key.check_rate_limit(&*self.clock) {
            self.metrics
                .rate_limited
                .with_label_values(&[&key.name])
                .inc();
            return Err(err);
        }
        if let Err(err) = key.check_budget(&*self.clock) {
            self.metrics
                .budget_rejected
                .with_label_values(&[&key.name])
                .inc();
            return Err(err);
        }

        let request: Value = serde_json::from_slice(&body).map_err(|e| {
            ApiError::invalid_request(format!("request body is not valid JSON: {e}"))
        })?;
        let object = request
            .as_object()
            .ok_or_else(|| ApiError::invalid_request("request body must be a JSON object"))?;
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid_request("`model` is required"))?;
        let targets = self
            .routes
            .get(model)
            .ok_or_else(|| ApiError::model_not_found(model))?;
        labels.model = model.to_string();
        if !object.get("messages").is_some_and(Value::is_array) {
            return Err(ApiError::invalid_request("`messages` must be an array"));
        }
        let stream = object
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let include_usage = object
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let cache_slot = match &self.cache {
            Some(cache) if cache::is_cacheable(&request) => {
                let cache_key = cache::cache_key(&key.name, &request);
                if let Some(hit) = cache.get(&cache_key) {
                    self.metrics.cache_lookups.with_label_values(&["hit"]).inc();
                    info!(key = %key.name, model = %model, provider = %hit.provider, "served from cache");
                    return Ok(json_response(
                        hit.body.clone(),
                        &hit.provider,
                        &hit.model,
                        0,
                        Some("hit"),
                        Some(0.0),
                    ));
                }
                self.metrics
                    .cache_lookups
                    .with_label_values(&["miss"])
                    .inc();
                Some((cache, cache_key))
            }
            _ => None,
        };
        let cache_header = match (&self.cache, &cache_slot) {
            (None, _) => None,
            (Some(_), Some(_)) => Some("miss"),
            (Some(_), None) => Some("bypass"),
        };

        let served = match self
            .dispatch(model, targets, &request, stream, request_id)
            .await
        {
            Ok(served) => served,
            Err(DispatchError::Api(err)) => return Err(err),
            Err(DispatchError::Rejected {
                status,
                body,
                provider,
                attempts,
            }) => {
                info!(%provider, status = status.as_u16(), "upstream rejected the request");
                let mut response = (status, axum::Json(body)).into_response();
                set_header(&mut response, HEADER_PROVIDER, &provider);
                set_header(&mut response, HEADER_ATTEMPTS, &attempts.to_string());
                return Ok(response);
            }
        };

        let provider_name = served.upstream.provider.name.clone();
        let upstream_model = served.target.model.clone();
        match served.reply {
            Reply::Json { body, usage } => {
                let cost = self.account(&key, &upstream_model, usage);
                info!(
                    key = %key.name,
                    model = %model,
                    provider = %provider_name,
                    upstream_model = %upstream_model,
                    attempts = served.attempts,
                    prompt_tokens = usage.map(|u| u.prompt_tokens),
                    completion_tokens = usage.map(|u| u.completion_tokens),
                    cost_usd = cost,
                    "completed"
                );
                if let Some((cache, cache_key)) = cache_slot {
                    cache.insert(
                        cache_key,
                        CachedResponse {
                            body: body.clone(),
                            provider: provider_name.clone(),
                            model: upstream_model.clone(),
                        },
                    );
                }
                Ok(json_response(
                    body,
                    &provider_name,
                    &upstream_model,
                    served.attempts,
                    cache_header,
                    Some(cost),
                ))
            }
            Reply::Stream(upstream_response, permit) => {
                let gateway = Arc::clone(self);
                let key_for_billing = Arc::clone(&key);
                let (model_log, provider_log, upstream_model_log, request_id_log) = (
                    model.to_string(),
                    provider_name.clone(),
                    upstream_model.clone(),
                    request_id.to_string(),
                );
                let started = Instant::now();
                let recorder = StreamRecorder::new(move |summary: StreamSummary| {
                    // A client hanging up says nothing about the upstream, so
                    // only upstream errors count against the circuit.
                    let transition = if summary.error.is_some() {
                        gateway.record_attempt(&provider_log, "stream_failed");
                        permit.failure()
                    } else {
                        permit.success()
                    };
                    gateway.note_transition(&provider_log, transition);
                    if summary.usage.is_none() {
                        gateway
                            .metrics
                            .unbilled_streams
                            .with_label_values(&[provider_log.as_str()])
                            .inc();
                    }
                    let cost =
                        gateway.account(&key_for_billing, &upstream_model_log, summary.usage);
                    // The body outlives the request span, so the request ID
                    // is logged explicitly here.
                    info!(
                        request_id = %request_id_log,
                        key = %key_for_billing.name,
                        model = %model_log,
                        provider = %provider_log,
                        upstream_model = %upstream_model_log,
                        completed = summary.completed,
                        error = summary.error.as_deref(),
                        prompt_tokens = summary.usage.map(|u| u.prompt_tokens),
                        completion_tokens = summary.usage.map(|u| u.completion_tokens),
                        cost_usd = cost,
                        stream_ms = started.elapsed().as_millis() as u64,
                        "stream finished"
                    );
                });
                let body = served.upstream.provider.stream_body(
                    upstream_response,
                    include_usage,
                    recorder,
                    self.unix_now(),
                );
                let mut response = Response::new(Body::from_stream(body));
                let headers = response.headers_mut();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("text/event-stream"),
                );
                headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
                // Stops nginx-style proxies from buffering the stream.
                headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
                set_header(&mut response, HEADER_PROVIDER, &provider_name);
                set_header(&mut response, HEADER_MODEL, &upstream_model);
                set_header(&mut response, HEADER_ATTEMPTS, &served.attempts.to_string());
                if let Some(cache_header) = cache_header {
                    set_header(&mut response, HEADER_CACHE, cache_header);
                }
                Ok(response)
            }
        }
    }

    /// Records tokens and cost, charges the key, and returns the cost.
    fn account(&self, key: &KeyState, upstream_model: &str, usage: Option<TokenUsage>) -> f64 {
        let Some(usage) = usage else {
            return 0.0;
        };
        let labels = |kind: &'static str| {
            [
                key.name.clone(),
                upstream_model.to_string(),
                kind.to_string(),
            ]
        };
        let tokens = &self.metrics.tokens;
        tokens
            .with_label_values(&labels("prompt"))
            .inc_by(usage.prompt_tokens);
        tokens
            .with_label_values(&labels("completion"))
            .inc_by(usage.completion_tokens);
        if usage.cached_prompt_tokens > 0 {
            tokens
                .with_label_values(&labels("cached_prompt"))
                .inc_by(usage.cached_prompt_tokens);
        }
        if usage.cache_write_prompt_tokens > 0 {
            tokens
                .with_label_values(&labels("cache_write_prompt"))
                .inc_by(usage.cache_write_prompt_tokens);
        }
        let cost = self.prices.cost(upstream_model, usage).unwrap_or(0.0);
        if cost > 0.0 {
            self.metrics
                .cost_usd
                .with_label_values(&[key.name.as_str(), upstream_model])
                .inc_by(cost);
            key.charge(cost, &*self.clock);
        }
        cost
    }

    /// Walks the route's targets in order. Each target gets up to
    /// `retry.max_attempts` tries, with backoff between them, unless its
    /// circuit is open.
    async fn dispatch<'a>(
        &'a self,
        model: &str,
        targets: &'a [Target],
        request: &Value,
        stream: bool,
        request_id: &str,
    ) -> Result<Served<'a>, DispatchError> {
        let mut failures: Vec<String> = Vec::new();
        let mut kinds: Vec<FailureKind> = Vec::new();
        let mut unsupported: Option<ApiError> = None;
        let mut last_retry_after = None;
        let mut attempts = 0u32;

        for (index, target) in targets.iter().enumerate() {
            let upstream = &self.upstreams[&target.provider];
            if index > 0 {
                self.metrics.fallbacks.with_label_values(&[model]).inc();
                info!(
                    provider = %target.provider,
                    upstream_model = %target.model,
                    "falling back to next target"
                );
            }
            let timeout = target
                .timeout_ms
                .map(Duration::from_millis)
                .unwrap_or(upstream.provider.timeout);
            let mut body = None;

            for attempt in 0..self.retry.max_attempts {
                let Some(permit) = upstream.breaker.try_acquire() else {
                    // If the circuit opened during our own retries, the
                    // failure that opened it is already recorded.
                    if attempt == 0 {
                        self.record_attempt(&target.provider, "circuit_open");
                        failures.push(format!(
                            "{} ({}): circuit open",
                            target.provider, target.model
                        ));
                        kinds.push(FailureKind::CircuitOpen);
                        last_retry_after = Some(upstream.breaker.retry_in());
                    }
                    break;
                };
                // Translate once per target, and only if we get this far. A
                // request this target's API can't express (say `n: 2` for
                // Anthropic) may still suit a later target.
                let body = match &body {
                    Some(body) => Bytes::clone(body),
                    None => match upstream.provider.prepare_body(request, target, stream) {
                        Ok(prepared) => body.insert(prepared).clone(),
                        Err(err) => {
                            drop(permit);
                            failures.push(format!(
                                "{} ({}): {}",
                                target.provider, target.model, err.message
                            ));
                            kinds.push(FailureKind::Unsupported);
                            unsupported.get_or_insert(err);
                            break;
                        }
                    },
                };
                attempts += 1;
                let result = self
                    .attempt(upstream, body, stream, timeout, request_id)
                    .await;
                match result {
                    Ok(fetched) => {
                        self.record_attempt(&target.provider, "success");
                        let reply = match fetched {
                            Fetched::Json { body, usage } => {
                                self.note_transition(&target.provider, permit.success());
                                Reply::Json { body, usage }
                            }
                            Fetched::Stream(response) => Reply::Stream(response, permit),
                        };
                        return Ok(Served {
                            upstream,
                            target,
                            reply,
                            attempts,
                        });
                    }
                    Err(AttemptError::Rejected { status, body }) => {
                        // The upstream is healthy; it just didn't like the request.
                        self.record_attempt(&target.provider, "rejected");
                        self.note_transition(&target.provider, permit.success());
                        return Err(DispatchError::Rejected {
                            status,
                            body,
                            provider: target.provider.clone(),
                            attempts,
                        });
                    }
                    Err(AttemptError::Failed {
                        reason,
                        outcome,
                        retry_after,
                        retry_same_target,
                        counts_against_circuit,
                        kind,
                    }) => {
                        self.record_attempt(&target.provider, outcome);
                        let transition = if counts_against_circuit {
                            permit.failure()
                        } else {
                            permit.success()
                        };
                        self.note_transition(&target.provider, transition);
                        warn!(
                            provider = %target.provider,
                            upstream_model = %target.model,
                            attempt = attempt + 1,
                            %reason,
                            "upstream attempt failed"
                        );
                        failures.push(format!("{} ({}): {reason}", target.provider, target.model));
                        kinds.push(kind);
                        last_retry_after = retry_after;

                        if !retry_same_target || attempt + 1 >= self.retry.max_attempts {
                            break;
                        }
                        let delay = match retry_after {
                            Some(wait) if wait > self.retry.max_retry_after() => {
                                info!(
                                    provider = %target.provider,
                                    retry_after_ms = wait.as_millis() as u64,
                                    "Retry-After is longer than retry.max_retry_after_ms; moving on"
                                );
                                break;
                            }
                            Some(wait) => wait,
                            None => self.backoff.delay(attempt),
                        };
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(DispatchError::Api(exhausted(
            model,
            &failures,
            &kinds,
            last_retry_after,
            unsupported,
        )))
    }

    async fn attempt(
        &self,
        upstream: &Upstream,
        body: Bytes,
        stream: bool,
        timeout: Duration,
        request_id: &str,
    ) -> Result<Fetched, AttemptError> {
        let provider = &upstream.provider;
        let request = provider.build_request(&self.http, body, request_id);
        let exchange = async {
            let response = request.send().await.map_err(network_failure)?;
            let status = response.status();
            if status.is_success() {
                if stream {
                    return Ok(Fetched::Stream(response));
                }
                let bad_response = |reason: String| AttemptError::Failed {
                    reason,
                    outcome: "bad_response",
                    retry_after: None,
                    retry_same_target: true,
                    counts_against_circuit: true,
                    kind: FailureKind::Other,
                };
                let bytes = read_body(response, MAX_RESPONSE_BYTES)
                    .await
                    .map_err(|e| match e {
                        BodyError::Network(e) => network_failure(e),
                        BodyError::TooLarge => bad_response(format!(
                            "response body is larger than {MAX_RESPONSE_BYTES} bytes"
                        )),
                    })?;
                let (body, usage) = provider
                    .parse_response(&bytes, self.unix_now())
                    .map_err(|e| bad_response(format!("unusable response body: {e}")))?;
                return Ok(Fetched::Json { body, usage });
            }

            let retry_after =
                backoff::retry_after(response.headers(), UNIX_EPOCH + self.clock.unix_time());
            let bytes = read_body(response, MAX_ERROR_BYTES)
                .await
                .unwrap_or_default();
            let snippet = error_snippet(&bytes);
            let code = status.as_u16();
            let failed = |outcome, kind, retry_same_target, counts_against_circuit, reason| {
                Err(AttemptError::Failed {
                    reason,
                    outcome,
                    retry_after,
                    retry_same_target,
                    counts_against_circuit,
                    kind,
                })
            };
            match code {
                429 => failed(
                    "rate_limited",
                    FailureKind::RateLimited,
                    true,
                    true,
                    format!("HTTP 429{snippet}"),
                ),
                408 | 500..=599 => failed(
                    "server_error",
                    FailureKind::Other,
                    true,
                    true,
                    format!("HTTP {code}{snippet}"),
                ),
                // The gateway's credentials for this provider are wrong. That
                // won't fix itself on retry, but another provider may work.
                401 | 403 => failed(
                    "auth_error",
                    FailureKind::Other,
                    false,
                    true,
                    format!("HTTP {code} (check this provider's API key){snippet}"),
                ),
                // Usually a model ID this upstream doesn't serve. Targets use
                // different model IDs, so the next one may be fine, and the
                // upstream itself is healthy.
                404 => failed(
                    "not_found",
                    FailureKind::NotFound,
                    false,
                    false,
                    format!("HTTP 404{snippet}"),
                ),
                // Redirects are not followed (see the client builder), and no
                // LLM API should send one. Most likely a wrong base_url.
                300..=399 => failed(
                    "redirect",
                    FailureKind::Other,
                    false,
                    true,
                    format!("unexpected redirect, HTTP {code} (check base_url)"),
                ),
                _ => Err(AttemptError::Rejected {
                    status: StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY),
                    body: provider.error_body(&bytes),
                }),
            }
        };
        match tokio::time::timeout(timeout, exchange).await {
            Ok(result) => result,
            Err(_) => Err(AttemptError::Failed {
                reason: format!("timed out after {}ms", timeout.as_millis()),
                outcome: "timeout",
                retry_after: None,
                retry_same_target: true,
                counts_against_circuit: true,
                kind: FailureKind::Timeout,
            }),
        }
    }

    fn record_attempt(&self, provider: &str, outcome: &str) {
        self.metrics
            .upstream_attempts
            .with_label_values(&[provider, outcome])
            .inc();
    }

    fn note_transition(&self, provider: &str, transition: Option<CircuitState>) {
        if let Some(state) = transition {
            self.metrics
                .circuit_transitions
                .with_label_values(&[provider, state.as_str()])
                .inc();
            match state {
                CircuitState::Open => warn!(%provider, "circuit opened"),
                _ => info!(%provider, state = state.as_str(), "circuit state changed"),
            }
        }
    }
}

fn network_failure(err: reqwest::Error) -> AttemptError {
    let (outcome, kind) = if err.is_timeout() {
        ("timeout", FailureKind::Timeout)
    } else if err.is_connect() {
        ("connect_error", FailureKind::Other)
    } else {
        ("network_error", FailureKind::Other)
    };
    AttemptError::Failed {
        reason: format!("{outcome}: {err}"),
        outcome,
        retry_after: None,
        retry_same_target: true,
        counts_against_circuit: true,
        kind,
    }
}

/// What a successful attempt produced, before the circuit permit is attached.
enum Fetched {
    Json {
        body: Bytes,
        usage: Option<TokenUsage>,
    },
    Stream(reqwest::Response),
}

enum BodyError {
    Network(reqwest::Error),
    TooLarge,
}

/// Reads a response body, refusing to buffer more than `limit` bytes.
async fn read_body(mut response: reqwest::Response, limit: usize) -> Result<Bytes, BodyError> {
    if response
        .content_length()
        .is_some_and(|len| len > limit as u64)
    {
        return Err(BodyError::TooLarge);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(BodyError::Network)? {
        if body.len() + chunk.len() > limit {
            return Err(BodyError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

/// First part of an upstream error body, for logs and the final error message.
fn error_snippet(body: &[u8]) -> String {
    const MAX: usize = 200;
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }
    let cut: String = text.chars().take(MAX).collect();
    let ellipsis = if cut.len() < text.len() { "..." } else { "" };
    format!(": {cut}{ellipsis}")
}

/// The error returned when every target failed. The status reflects what
/// went wrong: 400 if no target could express the request at all, 404 if
/// every upstream said it didn't know the model, 503 if every circuit was
/// open, 429 if the last upstream was rate limiting (so clients back off),
/// 504 on a timeout, and 502 otherwise.
fn exhausted(
    model: &str,
    failures: &[String],
    kinds: &[FailureKind],
    last_retry_after: Option<Duration>,
    unsupported: Option<ApiError>,
) -> ApiError {
    let tried: Vec<FailureKind> = kinds
        .iter()
        .copied()
        .filter(|k| *k != FailureKind::Unsupported)
        .collect();
    if tried.is_empty()
        && let Some(err) = unsupported
    {
        return err;
    }
    let message = format!(
        "all upstream targets for model {model:?} failed: {}",
        failures.join("; ")
    );
    let last = tried.last().copied().unwrap_or(FailureKind::Other);
    let error = if tried.iter().all(|k| *k == FailureKind::CircuitOpen) {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "upstream_error", message)
            .with_code("circuit_open")
    } else if tried.iter().all(|k| *k == FailureKind::NotFound) {
        ApiError::new(StatusCode::NOT_FOUND, "invalid_request_error", message)
            .with_code("model_not_found")
    } else {
        match last {
            FailureKind::RateLimited => {
                ApiError::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", message)
                    .with_code("upstream_rate_limited")
            }
            FailureKind::Timeout => {
                ApiError::new(StatusCode::GATEWAY_TIMEOUT, "upstream_error", message)
                    .with_code("upstream_timeout")
            }
            _ => ApiError::new(StatusCode::BAD_GATEWAY, "upstream_error", message)
                .with_code("upstream_unavailable"),
        }
    };
    match last_retry_after {
        Some(wait) if matches!(last, FailureKind::RateLimited | FailureKind::CircuitOpen) => {
            error.with_retry_after(wait)
        }
        _ => error,
    }
}

fn json_response(
    body: Bytes,
    provider: &str,
    model: &str,
    attempts: u32,
    cache: Option<&str>,
    cost: Option<f64>,
) -> Response {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    set_header(&mut response, HEADER_PROVIDER, provider);
    set_header(&mut response, HEADER_MODEL, model);
    set_header(&mut response, HEADER_ATTEMPTS, &attempts.to_string());
    if let Some(cache) = cache {
        set_header(&mut response, HEADER_CACHE, cache);
    }
    if let Some(cost) = cost {
        set_header(&mut response, HEADER_COST, &format!("{cost:.8}"));
    }
    response
}

fn set_header(response: &mut Response, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        response.headers_mut().insert(name, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_status_follows_the_failures() {
        let failures = vec!["a".to_string(), "b".to_string()];
        let status = |kinds: &[FailureKind]| exhausted("m", &failures, kinds, None, None).status;
        assert_eq!(
            status(&[FailureKind::Other, FailureKind::RateLimited]),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            status(&[FailureKind::CircuitOpen, FailureKind::CircuitOpen]),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            status(&[FailureKind::CircuitOpen, FailureKind::Timeout]),
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(
            status(&[FailureKind::RateLimited, FailureKind::Other]),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            status(&[FailureKind::NotFound, FailureKind::NotFound]),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            status(&[FailureKind::NotFound, FailureKind::Other]),
            StatusCode::BAD_GATEWAY
        );
        // Untranslatable targets are ignored when others were tried.
        assert_eq!(
            status(&[FailureKind::Unsupported, FailureKind::Timeout]),
            StatusCode::GATEWAY_TIMEOUT
        );
    }

    #[test]
    fn untranslatable_everywhere_returns_the_translation_error() {
        let err = exhausted(
            "m",
            &["a (m1): n > 1 is not supported".into()],
            &[FailureKind::Unsupported],
            None,
            Some(ApiError::invalid_request("n > 1 is not supported")),
        );
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "n > 1 is not supported");
    }

    #[test]
    fn exhausted_message_lists_every_failure() {
        let err = exhausted(
            "smart",
            &[
                "a (m1): HTTP 500".into(),
                "b (m2): timed out after 100ms".into(),
            ],
            &[FailureKind::Other, FailureKind::Timeout],
            None,
            None,
        );
        assert_eq!(
            err.message,
            "all upstream targets for model \"smart\" failed: a (m1): HTTP 500; b (m2): timed out after 100ms"
        );
    }

    #[test]
    fn retry_after_is_passed_on_for_rate_limits() {
        let err = exhausted(
            "m",
            &["x".into()],
            &[FailureKind::RateLimited],
            Some(Duration::from_secs(20)),
            None,
        );
        assert_eq!(err.retry_after, Some(Duration::from_secs(20)));
        let err = exhausted(
            "m",
            &["x".into()],
            &[FailureKind::Other],
            Some(Duration::from_secs(20)),
            None,
        );
        assert_eq!(err.retry_after, None);
    }

    #[test]
    fn snippets_are_trimmed_and_bounded() {
        assert_eq!(error_snippet(b""), "");
        assert_eq!(error_snippet(b"  boom \n"), ": boom");
        let long = "x".repeat(500);
        let snippet = error_snippet(long.as_bytes());
        assert_eq!(snippet.len(), 2 + 200 + 3);
    }
}
