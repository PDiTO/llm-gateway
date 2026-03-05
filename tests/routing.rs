//! Routing, retries, fallback and circuit breaking, end to end through the
//! real router against scripted mock upstreams.

mod common;

use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};

/// Two OpenAI-kind providers, `a` then `b`, behind the alias `smart`.
fn two_provider_config(a: &str, b: &str, extra: &str) -> String {
    format!(
        r#"
{FAST_RETRY}
{extra}

[providers.a]
kind = "openai"
base_url = "{a}"
api_key = "sk-upstream-a"
timeout_ms = 2000

[providers.b]
kind = "openai"
base_url = "{b}"
api_key = "sk-upstream-b"

[[models]]
name = "smart"
targets = [
  {{ provider = "a", model = "model-a" }},
  {{ provider = "b", model = "model-b" }},
]

{keys}
"#,
        keys = default_keys()
    )
}

async fn pair() -> (MockUpstream, MockUpstream) {
    let a = MockUpstream::start(Behavior::Json(openai_completion("from a", 10, 5))).await;
    let b = MockUpstream::start(Behavior::Json(openai_completion("from b", 10, 5))).await;
    (a, b)
}

#[tokio::test]
async fn passes_the_request_through_to_the_first_target() {
    let (a, b) = pair().await;
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let mut body = chat_body("smart");
    body["seed"] = json!(42);
    body["logit_bias"] = json!({"50256": -100});
    let response = gw
        .post_chat()
        .header("x-request-id", "trace-123")
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "a");
    assert_eq!(header(&response, "x-gateway-model"), "model-a");
    assert_eq!(header(&response, "x-gateway-attempts"), "1");
    assert_eq!(header(&response, "x-request-id"), "trace-123");
    let json: Value = response.json().await.unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "from a");

    let upstream = a.last_request();
    assert_eq!(upstream.path, "/v1/chat/completions");
    // Model is rewritten, unknown fields survive, and the gateway key is
    // replaced by the provider's key.
    assert_eq!(upstream.body["model"], "model-a");
    assert_eq!(upstream.body["seed"], 42);
    assert_eq!(upstream.body["logit_bias"], json!({"50256": -100}));
    assert_eq!(upstream.headers["authorization"], "Bearer sk-upstream-a");
    assert_eq!(upstream.headers["x-request-id"], "trace-123");
    assert_eq!(b.hits(), 0);
}

#[tokio::test]
async fn generates_a_request_id_when_the_client_sends_none() {
    let (a, b) = pair().await;
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;
    let response = gw.chat(chat_body("smart")).await;
    let id = header(&response, "x-request-id").to_string();
    assert_eq!(id.len(), 36, "expected a UUID, got {id}");
    assert_eq!(a.last_request().headers["x-request-id"], id.as_str());
}

#[tokio::test]
async fn retries_then_falls_back_in_order_on_5xx() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert_eq!(header(&response, "x-gateway-attempts"), "3");
    assert_eq!(a.hits(), 2, "a gets max_attempts tries");
    assert_eq!(b.hits(), 1);
    assert!(a.last_request().at <= b.last_request().at);
    assert_eq!(b.last_request().body["model"], "model-b");
}

#[tokio::test]
async fn a_retry_can_succeed_on_the_same_target() {
    let (a, b) = pair().await;
    a.push(Behavior::status(503));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "a");
    assert_eq!(header(&response, "x-gateway-attempts"), "2");
    assert_eq!(b.hits(), 0);
}

#[tokio::test]
async fn falls_back_when_the_first_provider_refuses_connections() {
    let b = MockUpstream::start(Behavior::Json(openai_completion("from b", 1, 1))).await;
    let gw = TestGateway::start(&two_provider_config(&dead_url().await, &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "b");

    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_upstream_attempts_total{outcome="connect_error",provider="a"}"#
        ),
        Some(2.0),
        "{metrics}"
    );
}

#[tokio::test]
async fn falls_back_when_the_first_provider_hangs() {
    let (a, b) = pair().await;
    a.set_default(Behavior::Hang);
    let config = two_provider_config(&a.openai_url(), &b.openai_url(), "")
        .replace("timeout_ms = 2000", "timeout_ms = 150");
    let gw = TestGateway::start(&config).await;

    let started = Instant::now();
    let response = gw.chat(chat_body("smart")).await;
    let elapsed = started.elapsed();
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert_eq!(a.hits(), 2);
    // Two 150ms timeouts plus a short backoff, nowhere near the hour the mock sleeps.
    assert!(elapsed >= Duration::from_millis(300), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
}

#[tokio::test]
async fn per_target_timeout_overrides_the_provider_timeout() {
    let (a, b) = pair().await;
    a.set_default(Behavior::Hang);
    let config = two_provider_config(&a.openai_url(), &b.openai_url(), "").replace(
        r#"{ provider = "a", model = "model-a" }"#,
        r#"{ provider = "a", model = "model-a", timeout_ms = 50 }"#,
    );
    let gw = TestGateway::start(&config).await;

    let started = Instant::now();
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert!(started.elapsed() < Duration::from_millis(1000));
}

#[tokio::test]
async fn retry_honours_retry_after() {
    let (a, b) = pair().await;
    a.push(Behavior::status_with_retry_after(429, "1"));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let started = Instant::now();
    let response = gw.chat(chat_body("smart")).await;
    let elapsed = started.elapsed();
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "a");
    assert_eq!(a.hits(), 2);
    assert_eq!(b.hits(), 0);
    let requests = a.requests();
    let gap = requests[1].at - requests[0].at;
    assert!(
        gap >= Duration::from_secs(1),
        "retried after {gap:?}, expected >= 1s"
    );
    assert!(elapsed < Duration::from_secs(3), "{elapsed:?}");
}

#[tokio::test]
async fn retry_after_ms_is_understood() {
    let (a, b) = pair().await;
    a.push(Behavior::Status {
        status: 429,
        body: json!({"error": {"message": "slow down"}}),
        headers: vec![
            ("retry-after-ms", "300".into()),
            ("retry-after", "30".into()),
        ],
    });
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    gw.chat(chat_body("smart")).await;
    let requests = a.requests();
    let gap = requests[1].at - requests[0].at;
    assert!(gap >= Duration::from_millis(300), "{gap:?}");
    assert!(gap < Duration::from_secs(5), "{gap:?}");
}

#[tokio::test]
async fn long_retry_after_moves_on_to_the_next_target() {
    let (a, b) = pair().await;
    a.push(Behavior::status_with_retry_after(429, "120"));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let started = Instant::now();
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert_eq!(a.hits(), 1, "no point waiting two minutes to retry a");
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn client_errors_are_returned_without_retry_or_fallback() {
    let (a, b) = pair().await;
    a.set_default(Behavior::Status {
        status: 400,
        body: json!({"error": {"message": "messages: too long", "type": "invalid_request_error", "param": "messages", "code": null}}),
        headers: vec![],
    });
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 400);
    assert_eq!(header(&response, "x-gateway-provider"), "a");
    let json: Value = response.json().await.unwrap();
    assert_eq!(json["error"]["message"], "messages: too long");
    assert_eq!(a.hits(), 1);
    assert_eq!(b.hits(), 0);
}

#[tokio::test]
async fn bad_upstream_credentials_fall_back_without_retrying() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(401));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert_eq!(a.hits(), 1);
}

#[tokio::test]
async fn reports_every_failure_when_all_targets_fail() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    b.set_default(Behavior::status(502));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 502);
    let json: Value = response.json().await.unwrap();
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.starts_with(r#"all upstream targets for model "smart" failed"#),
        "{message}"
    );
    assert!(message.contains("a (model-a): HTTP 500"), "{message}");
    assert!(message.contains("b (model-b): HTTP 502"), "{message}");
    assert_eq!(json["error"]["code"], "upstream_unavailable");
}

#[tokio::test]
async fn upstream_rate_limits_everywhere_become_a_429_for_the_client() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status_with_retry_after(429, "60"));
    b.set_default(Behavior::status_with_retry_after(429, "45"));
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 429);
    assert_eq!(header(&response, "retry-after"), "45");
}

/// Circuit test config: one attempt per target, open after 2 failures,
/// 30 second cooldown (on the manual clock).
fn breaker_config(a: &str, b: &str) -> String {
    two_provider_config(
        a,
        b,
        "[circuit_breaker]\nfailure_threshold = 2\ncooldown_ms = 30000\n",
    )
    .replace("max_attempts = 2", "max_attempts = 1")
}

#[tokio::test]
async fn circuit_opens_after_repeated_failures_and_skips_the_provider() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    let gw = TestGateway::start(&breaker_config(&a.openai_url(), &b.openai_url())).await;

    for _ in 0..2 {
        let response = gw.chat(chat_body("smart")).await;
        assert_eq!(header(&response, "x-gateway-provider"), "b");
    }
    assert_eq!(a.hits(), 2);

    let health: Value = gw.get("/healthz").await.json().await.unwrap();
    assert_eq!(health["circuits"]["a"], "open");
    assert_eq!(health["circuits"]["b"], "closed");

    // With the circuit open, a is not contacted at all.
    for _ in 0..5 {
        let response = gw.chat(chat_body("smart")).await;
        assert_eq!(header(&response, "x-gateway-provider"), "b");
        assert_eq!(header(&response, "x-gateway-attempts"), "1");
    }
    assert_eq!(a.hits(), 2);

    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(&metrics, r#"llm_gateway_circuit_state{provider="a"}"#),
        Some(1.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_upstream_attempts_total{outcome="circuit_open",provider="a"}"#
        ),
        Some(5.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_circuit_transitions_total{provider="a",to="open"}"#
        ),
        Some(1.0)
    );
}

#[tokio::test]
async fn circuit_half_opens_after_cooldown_and_closes_on_success() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    let gw = TestGateway::start(&breaker_config(&a.openai_url(), &b.openai_url())).await;
    gw.chat(chat_body("smart")).await;
    gw.chat(chat_body("smart")).await;
    assert_eq!(a.hits(), 2);

    // Not yet.
    gw.clock.advance(Duration::from_secs(29));
    gw.chat(chat_body("smart")).await;
    assert_eq!(a.hits(), 2);

    // Cooldown over: the next request is a probe, and a has recovered.
    gw.clock.advance(Duration::from_secs(1));
    let health: Value = gw.get("/healthz").await.json().await.unwrap();
    assert_eq!(health["circuits"]["a"], "half_open");
    a.set_default(Behavior::Json(openai_completion("a is back", 1, 1)));
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "a");
    assert_eq!(a.hits(), 3);

    let health: Value = gw.get("/healthz").await.json().await.unwrap();
    assert_eq!(health["circuits"]["a"], "closed");
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "a");
}

#[tokio::test]
async fn failed_probe_reopens_the_circuit() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    let gw = TestGateway::start(&breaker_config(&a.openai_url(), &b.openai_url())).await;
    gw.chat(chat_body("smart")).await;
    gw.chat(chat_body("smart")).await;

    gw.clock.advance(Duration::from_secs(30));
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert_eq!(a.hits(), 3, "one probe went to a");

    // Open again for another full cooldown.
    gw.clock.advance(Duration::from_secs(15));
    gw.chat(chat_body("smart")).await;
    assert_eq!(a.hits(), 3);
    let health: Value = gw.get("/healthz").await.json().await.unwrap();
    assert_eq!(health["circuits"]["a"], "open");
}

#[tokio::test]
async fn all_circuits_open_returns_503_quickly() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    b.set_default(Behavior::status(500));
    let gw = TestGateway::start(&breaker_config(&a.openai_url(), &b.openai_url())).await;
    for _ in 0..2 {
        assert_eq!(gw.chat(chat_body("smart")).await.status(), 502);
    }

    let started = Instant::now();
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 503);
    assert_eq!(header(&response, "retry-after"), "30");
    let json: Value = response.json().await.unwrap();
    assert_eq!(json["error"]["code"], "circuit_open");
    assert!(started.elapsed() < Duration::from_millis(200));
    assert_eq!(a.hits() + b.hits(), 4);
}

#[tokio::test]
async fn client_errors_do_not_trip_the_circuit() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(422));
    let gw = TestGateway::start(&breaker_config(&a.openai_url(), &b.openai_url())).await;
    for _ in 0..5 {
        assert_eq!(gw.chat(chat_body("smart")).await.status(), 422);
    }
    assert_eq!(a.hits(), 5);
    let health: Value = gw.get("/healthz").await.json().await.unwrap();
    assert_eq!(health["circuits"]["a"], "closed");
}

#[tokio::test]
async fn drop_params_strips_fields_for_one_target_only() {
    let (a, b) = pair().await;
    a.set_default(Behavior::status(500));
    let config = two_provider_config(&a.openai_url(), &b.openai_url(), "").replace(
        r#"{ provider = "b", model = "model-b" }"#,
        r#"{ provider = "b", model = "model-b", drop_params = ["temperature", "top_p"] }"#,
    );
    let gw = TestGateway::start(&config).await;
    let mut body = chat_body("smart");
    body["temperature"] = json!(0.7);
    body["top_p"] = json!(0.9);
    gw.chat(body).await;
    assert_eq!(a.last_request().body["temperature"], 0.7);
    let sent_to_b = b.last_request().body;
    assert!(sent_to_b.get("temperature").is_none());
    assert!(sent_to_b.get("top_p").is_none());
}

#[tokio::test]
async fn rejects_bad_requests_before_contacting_upstreams() {
    let (a, b) = pair().await;
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("gpt-17")).await;
    assert_eq!(response.status(), 404);
    let json: Value = response.json().await.unwrap();
    assert_eq!(json["error"]["code"], "model_not_found");

    let response = gw.post_chat().body("{not json").send().await.unwrap();
    assert_eq!(response.status(), 400);

    let response = gw.chat(json!({"messages": []})).await;
    assert_eq!(response.status(), 400);

    let response = gw.chat(json!({"model": "smart", "messages": "hi"})).await;
    assert_eq!(response.status(), 400);

    assert_eq!(a.hits() + b.hits(), 0);
}

#[tokio::test]
async fn lists_models_and_reports_health() {
    let (a, b) = pair().await;
    let config = two_provider_config(&a.openai_url(), &b.openai_url(), "")
        + "\n[[models]]\nname = \"fast\"\ntargets = [{ provider = \"b\", model = \"model-b-mini\" }]\n";
    let gw = TestGateway::start(&config).await;

    let models: Value = gw.get("/v1/models").await.json().await.unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["smart", "fast"]);
    assert_eq!(models["object"], "list");

    let unauthenticated = gw
        .client
        .get(format!("{}/v1/models", gw.url))
        .send()
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), 401);

    let health = gw
        .client
        .get(format!("{}/healthz", gw.url))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), 200);

    let missing = gw.get("/v1/embeddings").await;
    assert_eq!(missing.status(), 404);
}

#[tokio::test]
async fn upstream_404_moves_on_without_tripping_the_circuit() {
    let (a, b) = pair().await;
    a.set_default(Behavior::Status {
        status: 404,
        body: json!({"error": {"message": "The model `model-a` does not exist", "type": "invalid_request_error"}}),
        headers: vec![],
    });
    let gw = TestGateway::start(&breaker_config(&a.openai_url(), &b.openai_url())).await;
    for _ in 0..3 {
        let response = gw.chat(chat_body("smart")).await;
        assert_eq!(header(&response, "x-gateway-provider"), "b");
    }
    assert_eq!(
        a.hits(),
        3,
        "a 404 is not retried, and a keeps getting asked"
    );
    let health: Value = gw.get("/healthz").await.json().await.unwrap();
    assert_eq!(health["circuits"]["a"], "closed");

    // When nobody knows the model, the client gets a 404.
    b.set_default(Behavior::status(404));
    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(response.status(), 404);
}

#[tokio::test]
async fn redirects_are_not_followed() {
    let (a, b) = pair().await;
    let elsewhere = MockUpstream::start(Behavior::Json(openai_completion(
        "should not be reached",
        1,
        1,
    )))
    .await;
    a.set_default(Behavior::Status {
        status: 307,
        body: json!({}),
        headers: vec![(
            "location",
            format!("{}/chat/completions", elsewhere.openai_url()),
        )],
    });
    let gw = TestGateway::start(&two_provider_config(&a.openai_url(), &b.openai_url(), "")).await;

    let response = gw.chat(chat_body("smart")).await;
    assert_eq!(header(&response, "x-gateway-provider"), "b");
    assert_eq!(
        elsewhere.hits(),
        0,
        "credentials must not follow a redirect"
    );
    assert_eq!(a.hits(), 1);
}

#[tokio::test]
async fn circuit_opening_mid_retry_keeps_the_real_failure_status() {
    let (a, b) = pair().await;
    a.set_default(Behavior::Hang);
    // Only target a; the circuit opens on the first timeout, which blocks
    // the retry. The client should still see a timeout, not "circuit open".
    let config = format!(
        r#"
{FAST_RETRY}
[circuit_breaker]
failure_threshold = 1

[providers.a]
kind = "openai"
base_url = "{}"
timeout_ms = 100

[[models]]
name = "only-a"
targets = [{{ provider = "a", model = "model-a" }}]

{}
"#,
        a.openai_url(),
        default_keys()
    );
    let gw = TestGateway::start(&config).await;
    let response = gw.chat(chat_body("only-a")).await;
    assert_eq!(response.status(), 504);
    assert_eq!(a.hits(), 1);
    drop(b);
}
