//! Streaming and the Anthropic adapter, end to end.

mod common;

use std::time::Duration;

use common::*;
use serde_json::{Value, json};

fn config(openai: &str, anthropic: &str) -> String {
    format!(
        r#"
{FAST_RETRY}

[providers.openai]
kind = "openai"
base_url = "{openai}"
api_key = "sk-upstream-openai"

[providers.anthropic]
kind = "anthropic"
base_url = "{anthropic}"
api_key = "sk-ant-upstream"
default_max_tokens = 2048

[[models]]
name = "gpt"
targets = [{{ provider = "openai", model = "gpt-4.1-mini" }}]

[[models]]
name = "claude"
targets = [{{ provider = "anthropic", model = "claude-sonnet-4-5-20250929" }}]

[[models]]
name = "claude-then-gpt"
targets = [
  {{ provider = "anthropic", model = "claude-sonnet-4-5-20250929" }},
  {{ provider = "openai", model = "gpt-4.1-mini" }},
]

[prices."gpt-4.1-mini"]
input_per_mtok = 0.4
output_per_mtok = 1.6

[prices."claude-sonnet-4-5-20250929"]
input_per_mtok = 2.0
output_per_mtok = 10.0
cached_input_per_mtok = 0.2

{keys}
"#,
        keys = default_keys()
    )
}

fn stream_body(model: &str, include_usage: bool) -> Value {
    let mut body = json!({
        "model": model,
        "stream": true,
        "messages": [{"role": "user", "content": "Tell me a story"}]
    });
    if include_usage {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

async fn setup() -> (MockUpstream, MockUpstream, TestGateway) {
    let openai = MockUpstream::start(Behavior::Json(openai_completion("hi", 1, 1))).await;
    let anthropic = MockUpstream::start(Behavior::Json(anthropic_message("hi", 1, 1))).await;
    let gw = TestGateway::start(&config(&openai.openai_url(), &anthropic.anthropic_url())).await;
    (openai, anthropic, gw)
}

#[tokio::test]
async fn openai_stream_is_relayed_chunk_by_chunk() {
    let (openai, _, gw) = setup().await;
    let words = ["Once", " upon", " a", " time"];
    openai.set_default(Behavior::Stream {
        chunks: openai_stream(&words, 12, 4),
        delay: Duration::from_millis(80),
    });

    let response = gw.chat(stream_body("gpt", false)).await;
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "content-type"), "text/event-stream");
    assert_eq!(header(&response, "x-gateway-provider"), "openai");
    let events = read_sse(response).await;

    // role chunk, 4 words, finish chunk, [DONE]. The usage chunk the gateway
    // asked for on the client's behalf is hidden.
    let payloads: Vec<&str> = events.iter().map(|(data, _)| data.as_str()).collect();
    assert_eq!(payloads.len(), 7, "{payloads:#?}");
    assert_eq!(payloads.last(), Some(&"[DONE]"));
    assert!(payloads.iter().all(|p| !p.contains("\"usage\"")));
    let text: String = payloads[..payloads.len() - 1]
        .iter()
        .map(|p| serde_json::from_str::<Value>(p).unwrap())
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(text, "Once upon a time");

    // Chunks arrived spread out, as the upstream sent them, rather than all
    // at once at the end.
    let first = events.first().unwrap().1;
    let last = events.last().unwrap().1;
    assert!(
        last - first >= Duration::from_millis(400),
        "stream arrived in {:?}; looks buffered",
        last - first
    );
    let distinct_arrivals = events
        .windows(2)
        .filter(|w| w[1].1 - w[0].1 >= Duration::from_millis(40))
        .count();
    assert!(
        distinct_arrivals >= 5,
        "only {distinct_arrivals} separate arrivals"
    );

    // The gateway asked for usage and billed from it.
    assert_eq!(
        openai.last_request().body["stream_options"]["include_usage"],
        true
    );
    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_tokens_total{key="alice",kind="completion",model="gpt-4.1-mini"}"#
        ),
        Some(4.0),
        "{metrics}"
    );
}

#[tokio::test]
async fn usage_chunk_is_kept_when_the_client_asks_for_it() {
    let (openai, _, gw) = setup().await;
    openai.set_default(Behavior::sse(openai_stream(&["a", "b"], 3, 2)));
    let events = read_sse(gw.chat(stream_body("gpt", true)).await).await;
    let usage: Vec<Value> = events
        .iter()
        .filter_map(|(d, _)| serde_json::from_str::<Value>(d).ok())
        .filter(|c| c.get("usage").is_some())
        .collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn stream_falls_back_before_the_first_byte() {
    let (openai, anthropic, gw) = setup().await;
    anthropic.set_default(Behavior::status(529));
    openai.set_default(Behavior::sse(openai_stream(&["fallback"], 3, 1)));

    let response = gw.chat(stream_body("claude-then-gpt", false)).await;
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "openai");
    assert_eq!(header(&response, "x-gateway-attempts"), "3");
    let events = read_sse(response).await;
    assert_eq!(events.last().unwrap().0, "[DONE]");
    assert_eq!(anthropic.hits(), 2);
}

#[tokio::test]
async fn anthropic_request_is_translated_and_response_comes_back_as_openai() {
    let (_, anthropic, gw) = setup().await;
    anthropic.set_default(Behavior::Json(
        serde_json::from_str(&fixture("anthropic/anthropic_response_tool_use.json")).unwrap(),
    ));

    let request: Value =
        serde_json::from_str(&fixture("anthropic/openai_request_tools.json")).unwrap();
    let mut request = request;
    request["model"] = json!("claude");
    let response = gw
        .post_chat()
        .header("x-request-id", "req-anthropic")
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "x-gateway-provider"), "anthropic");
    let body: Value = response.json().await.unwrap();

    let mut expected: Value =
        serde_json::from_str(&fixture("anthropic/openai_response_tool_use.json")).unwrap();
    // `created` comes from the gateway's clock rather than the fixture.
    expected["created"] = body["created"].clone();
    assert_eq!(body, expected);

    let upstream = anthropic.last_request();
    assert_eq!(upstream.path, "/v1/messages");
    assert_eq!(upstream.headers["x-api-key"], "sk-ant-upstream");
    assert_eq!(upstream.headers["anthropic-version"], "2023-06-01");
    assert_eq!(upstream.headers["x-request-id"], "req-anthropic");
    assert!(upstream.headers.get("authorization").is_none());
    let expected_request: Value =
        serde_json::from_str(&fixture("anthropic/anthropic_request_tools.json")).unwrap();
    assert_eq!(upstream.body, expected_request);

    // 1,200 uncached + 300 cache-write tokens at $2/M, 2,000 cache reads at
    // $0.20/M, 95 output tokens at $10/M.
    let expected_cost = (1_500.0 * 2.0 + 2_000.0 * 0.2 + 95.0 * 10.0) / 1e6;
    let metrics = gw.metrics().await;
    let cost = metric_value(
        &metrics,
        r#"llm_gateway_cost_usd_total{key="alice",model="claude-sonnet-4-5-20250929"}"#,
    )
    .unwrap();
    assert!(
        (cost - expected_cost).abs() < 1e-12,
        "{cost} vs {expected_cost}"
    );
}

#[tokio::test]
async fn anthropic_errors_are_translated_into_openai_errors() {
    let (_, anthropic, gw) = setup().await;
    anthropic.set_default(Behavior::Status {
        status: 400,
        body: json!({"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: 999999 > 128000"}}),
        headers: vec![],
    });
    let response = gw.chat(chat_body("claude")).await;
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["message"], "max_tokens: 999999 > 128000");
    assert_eq!(anthropic.hits(), 1);
}

#[tokio::test]
async fn untranslatable_requests_are_rejected_with_400() {
    let (_, anthropic, gw) = setup().await;
    let mut body = chat_body("claude");
    body["n"] = json!(3);
    let response = gw.chat(body).await;
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("n > 1"));
    assert_eq!(anthropic.hits(), 0);
}

#[tokio::test]
async fn anthropic_text_stream_becomes_valid_openai_chunks() {
    let (_, anthropic, gw) = setup().await;
    anthropic.set_default(Behavior::Stream {
        chunks: sse_events(&fixture("anthropic/stream_text.sse")),
        delay: Duration::from_millis(10),
    });

    let response = gw.chat(stream_body("claude", true)).await;
    assert_eq!(response.status(), 200);
    assert_eq!(header(&response, "content-type"), "text/event-stream");
    let events = read_sse(response).await;
    let payloads: Vec<String> = events.into_iter().map(|(d, _)| d).collect();

    let expected_text = fixture("anthropic/stream_text.expected.jsonl");
    let expected: Vec<&str> = expected_text.lines().collect();
    assert_eq!(payloads.len(), expected.len(), "{payloads:#?}");
    for (got, want) in payloads.iter().zip(&expected) {
        if *want == "[DONE]" {
            assert_eq!(got, want);
            continue;
        }
        let mut got: Value = serde_json::from_str(got).unwrap();
        let want: Value = serde_json::from_str(want).unwrap();
        got["created"] = want["created"].clone();
        assert_eq!(got, want);
    }
    assert_eq!(anthropic.last_request().body["stream"], true);
}

#[tokio::test]
async fn anthropic_tool_stream_reassembles_into_tool_calls() {
    let (_, anthropic, gw) = setup().await;
    anthropic.set_default(Behavior::sse(sse_events(&fixture(
        "anthropic/stream_tool_use.sse",
    ))));

    let events = read_sse(gw.chat(stream_body("claude", false)).await).await;
    let payloads: Vec<String> = events.into_iter().map(|(d, _)| d).collect();
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));

    let mut arguments = [String::new(), String::new()];
    let mut names = [None, None];
    for payload in &payloads[..payloads.len() - 1] {
        let chunk: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"].as_array().unwrap().len(), 1);
        for call in chunk["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let i = call["index"].as_u64().unwrap() as usize;
            if let Some(name) = call["function"]["name"].as_str() {
                names[i] = Some(name.to_string());
            }
            arguments[i].push_str(call["function"]["arguments"].as_str().unwrap_or(""));
        }
    }
    assert_eq!(
        names,
        [
            Some("get_weather".to_string()),
            Some("get_weather".to_string())
        ]
    );
    let parsed: Vec<Value> = arguments
        .iter()
        .map(|a| serde_json::from_str(a).unwrap())
        .collect();
    assert_eq!(parsed, [json!({"city": "Paris"}), json!({"city": "Tokyo"})]);

    // Billed from the stream's usage: 412 + 256 prompt tokens, 89 completion.
    let metrics = gw.metrics().await;
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_tokens_total{key="alice",kind="prompt",model="claude-sonnet-4-5-20250929"}"#
        ),
        Some(668.0)
    );
    assert_eq!(
        metric_value(
            &metrics,
            r#"llm_gateway_tokens_total{key="alice",kind="cached_prompt",model="claude-sonnet-4-5-20250929"}"#
        ),
        Some(256.0)
    );
}

#[tokio::test]
async fn anthropic_mid_stream_error_is_surfaced_to_the_client() {
    let (_, anthropic, gw) = setup().await;
    anthropic.set_default(Behavior::sse(sse_events(&fixture(
        "anthropic/stream_error.sse",
    ))));

    let events = read_sse(gw.chat(stream_body("claude", false)).await).await;
    let last: Value = serde_json::from_str(&events.last().unwrap().0).unwrap();
    assert_eq!(last["error"]["type"], "overloaded_error");
    assert!(events.iter().all(|(d, _)| d != "[DONE]"));
}

#[tokio::test]
async fn truncated_upstream_stream_ends_with_an_error() {
    let (openai, _, gw) = setup().await;
    let mut chunks = openai_stream(&["cut", " off"], 1, 1);
    chunks.truncate(3); // role chunk and two words, then the connection closes
    openai.set_default(Behavior::sse(chunks));

    let events = read_sse(gw.chat(stream_body("gpt", false)).await).await;
    assert_eq!(events.len(), 4);
    let last: Value = serde_json::from_str(&events[3].0).unwrap();
    assert!(
        last["error"]["message"]
            .as_str()
            .unwrap()
            .contains("ended before")
    );
}
