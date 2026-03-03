//! Response cache behaviour through the full request path.

mod common;

use std::time::Duration;

use common::*;
use serde_json::{Value, json};

async fn setup(ttl_secs: u64) -> (MockUpstream, TestGateway) {
    let upstream =
        MockUpstream::start(Behavior::Json(openai_completion("cached answer", 50, 10))).await;
    let config = format!(
        r#"
[cache]
enabled = true
max_entries = 100
ttl_secs = {ttl_secs}

[providers.p]
kind = "openai"
base_url = "{url}"

[[models]]
name = "m"
targets = [{{ provider = "p", model = "upstream-model" }}]

[prices.upstream-model]
input_per_mtok = 1.0
output_per_mtok = 1.0

{keys}
"#,
        url = upstream.openai_url(),
        keys = default_keys() + &key_entry("bob", OTHER_KEY, 1000, None)
    );
    (upstream, TestGateway::start(&config).await)
}

fn deterministic() -> Value {
    json!({"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "What is 2+2?"}]})
}

#[tokio::test]
async fn second_identical_request_is_a_cache_hit() {
    let (upstream, gw) = setup(300).await;

    let first = gw.chat(deterministic()).await;
    assert_eq!(header(&first, "x-gateway-cache"), "miss");
    let first_body: Value = first.json().await.unwrap();

    // Same request with keys in a different order and 0.0 instead of 0.
    let reordered: Value = serde_json::from_str(
        r#"{"messages":[{"content":"What is 2+2?","role":"user"}],"temperature":0.0,"model":"m"}"#,
    )
    .unwrap();
    let second = gw.chat(reordered).await;
    assert_eq!(second.status(), 200);
    assert_eq!(header(&second, "x-gateway-cache"), "hit");
    assert_eq!(header(&second, "x-gateway-provider"), "p");
    assert_eq!(header(&second, "x-gateway-cost-usd"), "0.00000000");
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(first_body, second_body);
    assert_eq!(upstream.hits(), 1);

    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(&metrics, r#"llm_gateway_cache_lookups_total{result="hit"}"#),
        Some(1.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_cache_lookups_total{result="miss"}"#
        ),
        Some(1.0)
    );
    // Only the upstream call was billed.
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_tokens_total{key="alice",kind="prompt",model="upstream-model"}"#
        ),
        Some(50.0)
    );
}

#[tokio::test]
async fn non_deterministic_and_streaming_requests_bypass_the_cache() {
    let (upstream, gw) = setup(300).await;

    let mut warm = deterministic();
    warm["temperature"] = json!(0.7);
    for _ in 0..2 {
        let response = gw.chat(warm.clone()).await;
        assert_eq!(header(&response, "x-gateway-cache"), "bypass");
    }
    assert_eq!(upstream.hits(), 2);

    upstream.set_default(Behavior::sse(openai_stream(&["4"], 5, 1)));
    let mut stream = deterministic();
    stream["stream"] = json!(true);
    for _ in 0..2 {
        let response = gw.chat(stream.clone()).await;
        assert_eq!(header(&response, "x-gateway-cache"), "bypass");
        read_sse(response).await;
    }
    assert_eq!(upstream.hits(), 4);
}

#[tokio::test]
async fn entries_expire_after_the_ttl() {
    let (upstream, gw) = setup(60).await;
    gw.chat(deterministic()).await;
    gw.clock.advance(Duration::from_secs(59));
    assert_eq!(
        header(&gw.chat(deterministic()).await, "x-gateway-cache"),
        "hit"
    );
    gw.clock.advance(Duration::from_secs(1));
    assert_eq!(
        header(&gw.chat(deterministic()).await, "x-gateway-cache"),
        "miss"
    );
    assert_eq!(upstream.hits(), 2);
}

#[tokio::test]
async fn cache_is_scoped_to_the_gateway_key() {
    let (upstream, gw) = setup(300).await;
    gw.chat(deterministic()).await;
    let bob = gw
        .client
        .post(format!("{}/v1/chat/completions", gw.url))
        .bearer_auth(OTHER_KEY)
        .json(&deterministic())
        .send()
        .await
        .unwrap();
    assert_eq!(header(&bob, "x-gateway-cache"), "miss");
    assert_eq!(upstream.hits(), 2);
}

#[tokio::test]
async fn failed_responses_are_not_cached() {
    let (upstream, gw) = setup(300).await;
    upstream.push(Behavior::Status {
        status: 400,
        body: json!({"error": {"message": "nope"}}),
        headers: vec![],
    });
    assert_eq!(gw.chat(deterministic()).await.status(), 400);
    let retry = gw.chat(deterministic()).await;
    assert_eq!(retry.status(), 200);
    assert_eq!(header(&retry, "x-gateway-cache"), "miss");
    assert_eq!(upstream.hits(), 2);
}
