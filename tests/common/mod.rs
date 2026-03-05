//! Shared helpers for the integration tests: a scriptable fake provider and a
//! gateway running the real router on a random port.

#![allow(dead_code)] // each test binary uses a different subset

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde_json::{Value, json};

use llm_gateway::auth::hash_key;
use llm_gateway::clock::ManualClock;
use llm_gateway::config::Config;
use llm_gateway::gateway::Gateway;
use llm_gateway::server;

pub const KEY: &str = "sk-gw-test-alice";
pub const OTHER_KEY: &str = "sk-gw-test-bob";

/// Retry settings that keep tests fast: short backoff, two attempts per target.
pub const FAST_RETRY: &str = r#"
[retry]
max_attempts = 2
base_delay_ms = 5
max_delay_ms = 20
max_retry_after_ms = 3000
"#;

// ---------------------------------------------------------------------------
// Mock upstream

#[derive(Debug, Clone)]
pub enum Behavior {
    /// 200 with a JSON body.
    Json(Value),
    /// Any status, with a JSON body and extra headers.
    Status {
        status: u16,
        body: Value,
        headers: Vec<(&'static str, String)>,
    },
    /// Accepts the request and never answers.
    Hang,
    /// 200 `text/event-stream`, sending each chunk after `delay`.
    Stream {
        chunks: Vec<String>,
        delay: Duration,
    },
}

impl Behavior {
    pub fn status(status: u16) -> Self {
        Behavior::Status {
            status,
            body: json!({"error": {"message": format!("mock {status}"), "type": "mock_error"}}),
            headers: Vec::new(),
        }
    }

    pub fn status_with_retry_after(status: u16, retry_after: &str) -> Self {
        Behavior::Status {
            status,
            body: json!({"error": {"message": format!("mock {status}"), "type": "mock_error"}}),
            headers: vec![("retry-after", retry_after.to_string())],
        }
    }

    pub fn sse(chunks: Vec<String>) -> Self {
        Behavior::Stream {
            chunks,
            delay: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Recorded {
    pub path: String,
    pub headers: HeaderMap,
    pub body: Value,
    pub at: Instant,
}

struct MockState {
    script: Mutex<VecDeque<Behavior>>,
    default: Mutex<Behavior>,
    requests: Mutex<Vec<Recorded>>,
}

/// A fake provider on a random local port. It answers every POST with the
/// next scripted behaviour, or the default once the script runs out, and
/// records what it received.
pub struct MockUpstream {
    pub addr: SocketAddr,
    state: Arc<MockState>,
}

impl MockUpstream {
    pub async fn start(default: Behavior) -> Self {
        let state = Arc::new(MockState {
            script: Mutex::new(VecDeque::new()),
            default: Mutex::new(default),
            requests: Mutex::new(Vec::new()),
        });
        let app = Router::new()
            .fallback(mock_handler)
            .with_state(Arc::clone(&state));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { addr, state }
    }

    /// Base URL for an OpenAI-kind provider (includes `/v1`).
    pub fn openai_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// Base URL for an Anthropic-kind provider.
    pub fn anthropic_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Queues a one-off behaviour, used before falling back to the default.
    pub fn push(&self, behavior: Behavior) {
        self.state.script.lock().unwrap().push_back(behavior);
    }

    pub fn set_default(&self, behavior: Behavior) {
        *self.state.default.lock().unwrap() = behavior;
    }

    pub fn hits(&self) -> usize {
        self.state.requests.lock().unwrap().len()
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.state.requests.lock().unwrap().clone()
    }

    pub fn last_request(&self) -> Recorded {
        self.requests().pop().expect("mock received no requests")
    }
}

async fn mock_handler(State(state): State<Arc<MockState>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .unwrap_or_default();
    state.requests.lock().unwrap().push(Recorded {
        path: parts.uri.path().to_string(),
        headers: parts.headers,
        body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        at: Instant::now(),
    });
    let behavior = state
        .script
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or_else(|| state.default.lock().unwrap().clone());

    match behavior {
        Behavior::Json(body) => json_response(StatusCode::OK, &body, &[]),
        Behavior::Status {
            status,
            body,
            headers,
        } => json_response(StatusCode::from_u16(status).unwrap(), &body, &headers),
        Behavior::Hang => {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            json_response(StatusCode::OK, &json!({}), &[])
        }
        Behavior::Stream { chunks, delay } => {
            let stream = async_stream::stream! {
                for chunk in chunks {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    yield Ok::<_, std::io::Error>(Bytes::from(chunk));
                }
            };
            Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }
    }
}

fn json_response(status: StatusCode, body: &Value, headers: &[(&'static str, String)]) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, value);
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

/// A URL where nothing is listening, for connection-refused tests.
pub async fn dead_url() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}/v1")
}

// ---------------------------------------------------------------------------
// Response builders

pub fn openai_completion(content: &str, prompt_tokens: u64, completion_tokens: u64) -> Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

pub fn anthropic_message(text: &str, input_tokens: u64, output_tokens: u64) -> Value {
    json!({
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
    })
}

/// An OpenAI stream as OpenAI sends it with `include_usage` on (which the
/// gateway always sets): one chunk per word, each with `"usage":null`, then a
/// usage-only chunk and `[DONE]`.
pub fn openai_stream(words: &[&str], prompt_tokens: u64, completion_tokens: u64) -> Vec<String> {
    let mut chunks = Vec::new();
    let chunk = |delta: Value, finish: Value| {
        format!(
            "data: {}\n\n",
            json!({
                "id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 1, "model": "mock-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": null
            })
        )
    };
    chunks.push(chunk(
        json!({"role": "assistant", "content": ""}),
        Value::Null,
    ));
    for word in words {
        chunks.push(chunk(json!({"content": word}), Value::Null));
    }
    chunks.push(chunk(json!({}), json!("stop")));
    chunks.push(format!(
        "data: {}\n\n",
        json!({
            "id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 1, "model": "mock-model",
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}
        })
    ));
    chunks.push("data: [DONE]\n\n".to_string());
    chunks
}

pub fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// Splits an SSE fixture into one chunk per event, as a server would flush them.
pub fn sse_events(text: &str) -> Vec<String> {
    text.split("\n\n")
        .filter(|e| !e.trim().is_empty())
        .map(|e| format!("{e}\n\n"))
        .collect()
}

// ---------------------------------------------------------------------------
// Gateway harness

pub fn key_entry(name: &str, key: &str, rpm: u32, budget: Option<f64>) -> String {
    let budget = budget
        .map(|b| format!("daily_budget_usd = {b}\n"))
        .unwrap_or_default();
    format!(
        "[[keys]]\nname = \"{name}\"\nkey_sha256 = \"{}\"\nrequests_per_minute = {rpm}\n{budget}",
        hash_key(key)
    )
}

/// The default key section: alice with a generous rate limit and no budget.
pub fn default_keys() -> String {
    key_entry("alice", KEY, 10_000, None)
}

pub struct TestGateway {
    pub url: String,
    pub gateway: Arc<Gateway>,
    pub clock: Arc<ManualClock>,
    pub client: reqwest::Client,
}

impl TestGateway {
    /// Starts the real router with `config` on a random port. The clock is
    /// manual, so circuit cooldowns, rate limits and budgets only move when a
    /// test advances it.
    pub async fn start(config: &str) -> Self {
        let config = Config::from_toml_with_env(config, |_| None)
            .unwrap_or_else(|e| panic!("test config is invalid: {e}\n---\n{config}"));
        let clock = Arc::new(ManualClock::new());
        let gateway = Arc::new(Gateway::with_clock(config, clock.clone()));
        let app = server::router(Arc::clone(&gateway));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            url: format!("http://{addr}"),
            gateway,
            clock,
            client: reqwest::Client::new(),
        }
    }

    pub fn post_chat(&self) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}/v1/chat/completions", self.url))
            .bearer_auth(KEY)
    }

    pub async fn chat(&self, body: Value) -> reqwest::Response {
        self.post_chat().json(&body).send().await.unwrap()
    }

    pub async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
    }

    pub async fn metrics(&self) -> String {
        self.get("/metrics").await.text().await.unwrap()
    }
}

pub fn chat_body(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "Hello"}]})
}

pub fn header<'a>(response: &'a reqwest::Response, name: &str) -> &'a str {
    response
        .headers()
        .get(name)
        .unwrap_or_else(|| panic!("missing header {name}"))
        .to_str()
        .unwrap()
}

/// Reads a streaming response and returns each `data:` payload with the time
/// it arrived.
pub async fn read_sse(response: reqwest::Response) -> Vec<(String, Instant)> {
    use futures_util::StreamExt;
    let mut parser = llm_gateway::sse::SseParser::new();
    let mut out = Vec::new();
    let mut body = response.bytes_stream();
    while let Some(chunk) = body.next().await {
        let chunk = chunk.unwrap();
        let now = Instant::now();
        for event in parser.push(&chunk).unwrap() {
            out.push((event.data, now));
        }
    }
    out
}

/// Value of a Prometheus sample line, e.g.
/// `metric_value(&text, r#"llm_gateway_requests_total{key="alice",model="m",status="200"}"#)`.
pub fn metric_value(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find_map(|line| line.strip_prefix(series)?.trim().parse().ok())
}
