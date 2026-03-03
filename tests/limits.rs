//! Authentication, per-key rate limits, spend budgets and cost accounting.

mod common;

use std::time::Duration;

use common::*;
use serde_json::Value;

/// One OpenAI-kind provider. Every response uses 100k prompt and 20k
/// completion tokens, which at $2/M in and $10/M out costs exactly $0.40.
async fn setup(keys: &str) -> (MockUpstream, TestGateway) {
    let upstream =
        MockUpstream::start(Behavior::Json(openai_completion("ok", 100_000, 20_000))).await;
    let config = format!(
        r#"
[providers.p]
kind = "openai"
base_url = "{url}"

[[models]]
name = "m"
targets = [{{ provider = "p", model = "upstream-model" }}]

[prices.upstream-model]
input_per_mtok = 2.0
output_per_mtok = 10.0

{keys}
"#,
        url = upstream.openai_url()
    );
    let gw = TestGateway::start(&config).await;
    (upstream, gw)
}

#[tokio::test]
async fn missing_or_unknown_keys_are_rejected() {
    let (upstream, gw) = setup(&default_keys()).await;
    let url = format!("{}/v1/chat/completions", gw.url);

    let response = gw
        .client
        .post(&url)
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "authentication_error");

    let response = gw
        .client
        .post(&url)
        .bearer_auth("sk-gw-wrong")
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);

    let response = gw
        .client
        .post(&url)
        .header("x-api-key", KEY)
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "x-api-key is accepted too");
    assert_eq!(upstream.hits(), 1);
}

#[tokio::test]
async fn rate_limit_returns_429_with_retry_after_and_recovers() {
    // 2 requests per minute: one token every 30 seconds.
    let keys = key_entry("alice", KEY, 2, None) + &key_entry("bob", OTHER_KEY, 2, None);
    let (upstream, gw) = setup(&keys).await;

    assert_eq!(gw.chat(chat_body("m")).await.status(), 200);
    assert_eq!(gw.chat(chat_body("m")).await.status(), 200);
    let limited = gw.chat(chat_body("m")).await;
    assert_eq!(limited.status(), 429);
    assert_eq!(header(&limited, "retry-after"), "30");
    let body: Value = limited.json().await.unwrap();
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    assert_eq!(
        upstream.hits(),
        2,
        "limited requests never reach the upstream"
    );

    // Other keys have their own bucket.
    let bob = gw
        .client
        .post(format!("{}/v1/chat/completions", gw.url))
        .bearer_auth(OTHER_KEY)
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(bob.status(), 200);

    gw.clock.advance(Duration::from_secs(30));
    assert_eq!(gw.chat(chat_body("m")).await.status(), 200);
    assert_eq!(gw.chat(chat_body("m")).await.status(), 429);

    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(&metrics, r#"llm_gateway_rate_limited_total{key="alice"}"#),
        Some(2.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_requests_total{key="alice",model="-",status="429"}"#
        ),
        Some(2.0),
        "{metrics}"
    );
}

#[tokio::test]
async fn budget_exhaustion_returns_402_until_the_next_utc_day() {
    // $1.00 a day; each request costs $0.40.
    let (upstream, gw) = setup(&key_entry("alice", KEY, 1000, Some(1.0))).await;

    for _ in 0..3 {
        let response = gw.chat(chat_body("m")).await;
        assert_eq!(response.status(), 200);
        assert_eq!(header(&response, "x-gateway-cost-usd"), "0.40000000");
    }
    // $1.20 spent: the request that crossed the line went through, the next
    // one is refused.
    let refused = gw.chat(chat_body("m")).await;
    assert_eq!(refused.status(), 402);
    let body: Value = refused.json().await.unwrap();
    assert_eq!(body["error"]["code"], "budget_exceeded");
    assert_eq!(body["error"]["type"], "insufficient_quota");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains(r#"key "alice""#), "{message}");
    assert!(message.contains("$1.2000 of $1.00"), "{message}");
    assert!(message.contains("00:00 UTC"), "{message}");
    assert_eq!(upstream.hits(), 3);

    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_budget_rejected_total{key="alice"}"#
        ),
        Some(1.0)
    );
    let spend = metric_value(&metrics, r#"llm_gateway_key_spend_today_usd{key="alice"}"#).unwrap();
    assert!((spend - 1.2).abs() < 1e-9, "{spend}");

    gw.clock.advance(Duration::from_secs(86_400));
    assert_eq!(gw.chat(chat_body("m")).await.status(), 200);
}

#[tokio::test]
async fn cost_and_tokens_are_tracked_per_key_and_model() {
    let keys = default_keys() + &key_entry("bob", OTHER_KEY, 1000, None);
    let (_, gw) = setup(&keys).await;

    gw.chat(chat_body("m")).await;
    gw.chat(chat_body("m")).await;
    gw.client
        .post(format!("{}/v1/chat/completions", gw.url))
        .bearer_auth(OTHER_KEY)
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();

    let metrics = gw.metrics().await;
    let alice_cost = metric_value(
        &metrics,
        r#"llm_gateway_cost_usd_total{key="alice",model="upstream-model"}"#,
    )
    .unwrap();
    let bob_cost = metric_value(
        &metrics,
        r#"llm_gateway_cost_usd_total{key="bob",model="upstream-model"}"#,
    )
    .unwrap();
    assert!((alice_cost - 0.8).abs() < 1e-9, "{alice_cost}");
    assert!((bob_cost - 0.4).abs() < 1e-9, "{bob_cost}");
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_tokens_total{key="alice",kind="prompt",model="upstream-model"}"#
        ),
        Some(200_000.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_requests_total{key="alice",model="m",status="200"}"#
        ),
        Some(2.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_request_duration_seconds_count{model="m"}"#
        ),
        Some(3.0)
    );
}
