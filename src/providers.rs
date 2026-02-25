//! Upstream providers: how to build a request for each kind of API, and how
//! to turn its response (or stream) back into the OpenAI shape.

use std::io;
use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::anthropic::{self, StreamTranslator};
use crate::config::{ProviderConfig, ProviderKind, Target};
use crate::error::ApiError;
use crate::openai::{ChatCompletionRequest, TokenUsage};
use crate::sse::{SseEvent, SseParser, encode_data};

#[derive(Debug)]
pub struct Provider {
    pub name: String,
    pub kind: ProviderKind,
    endpoint: String,
    api_key: Option<String>,
    anthropic_version: String,
    default_max_tokens: u32,
    extra_headers: HeaderMap,
    pub timeout: Duration,
}

impl Provider {
    /// Builds a provider from validated config.
    pub fn from_config(name: &str, config: &ProviderConfig) -> Self {
        let base = config.base_url.trim_end_matches('/');
        let endpoint = match config.kind {
            // OpenAI-style base URLs already include the version, e.g.
            // https://api.openai.com/v1.
            ProviderKind::OpenAi => format!("{base}/chat/completions"),
            ProviderKind::Anthropic => format!("{base}/v1/messages"),
        };
        let mut extra_headers = HeaderMap::new();
        for (key, value) in &config.headers {
            if let (Ok(key), Ok(value)) = (
                HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                extra_headers.insert(key, value);
            }
        }
        Self {
            name: name.to_string(),
            kind: config.kind,
            endpoint,
            api_key: config.resolved_api_key.clone(),
            anthropic_version: config.anthropic_version.clone(),
            default_max_tokens: config.default_max_tokens,
            extra_headers,
            timeout: Duration::from_millis(config.timeout_ms),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Produces the upstream request body for `target` from the client's
    /// OpenAI-shaped request.
    pub fn prepare_body(
        &self,
        request: &Value,
        target: &Target,
        stream: bool,
    ) -> Result<Bytes, ApiError> {
        let mut request = request.clone();
        let object = request
            .as_object_mut()
            .ok_or_else(|| ApiError::invalid_request("request body must be a JSON object"))?;
        object.insert("model".into(), Value::String(target.model.clone()));
        for param in &target.drop_params {
            object.remove(param);
        }

        let body = match self.kind {
            ProviderKind::OpenAi => {
                if stream {
                    // Always ask for the final usage chunk so the stream can
                    // be billed. It is filtered out again if the client
                    // didn't ask for it.
                    let options = object.entry("stream_options").or_insert_with(|| json!({}));
                    if let Some(options) = options.as_object_mut() {
                        options.insert("include_usage".into(), Value::Bool(true));
                    }
                }
                serde_json::to_vec(&request)
            }
            ProviderKind::Anthropic => {
                let parsed: ChatCompletionRequest = serde_json::from_value(request)
                    .map_err(|e| ApiError::invalid_request(format!("invalid request: {e}")))?;
                let translated =
                    anthropic::translate_request(parsed, &target.model, self.default_max_tokens)?;
                serde_json::to_vec(&translated)
            }
        };
        body.map(Bytes::from)
            .map_err(|e| ApiError::internal(format!("could not encode upstream request: {e}")))
    }

    pub fn build_request(
        &self,
        client: &reqwest::Client,
        body: Bytes,
        request_id: &str,
    ) -> reqwest::RequestBuilder {
        let mut request = client
            .post(&self.endpoint)
            .header(CONTENT_TYPE, "application/json")
            .header("x-request-id", request_id)
            .headers(self.extra_headers.clone())
            .body(body);
        match self.kind {
            ProviderKind::OpenAi => {
                if let Some(key) = &self.api_key {
                    request = request.header(AUTHORIZATION, format!("Bearer {key}"));
                }
            }
            ProviderKind::Anthropic => {
                request = request.header("anthropic-version", &self.anthropic_version);
                if let Some(key) = &self.api_key {
                    request = request.header("x-api-key", key);
                }
            }
        }
        request
    }

    /// Converts a successful non-streaming response body into OpenAI JSON and
    /// pulls out the usage. An error here means the upstream sent something
    /// unusable, which the caller treats like any other upstream failure.
    pub fn parse_response(
        &self,
        body: &[u8],
        created: u64,
    ) -> Result<(Bytes, Option<TokenUsage>), String> {
        match self.kind {
            ProviderKind::OpenAi => {
                let value: Value = serde_json::from_slice(body).map_err(|e| e.to_string())?;
                if !value.get("choices").is_some_and(Value::is_array) {
                    return Err("response has no choices array".into());
                }
                let usage = value.get("usage").and_then(TokenUsage::from_openai);
                // Forward the upstream bytes untouched.
                Ok((Bytes::copy_from_slice(body), usage))
            }
            ProviderKind::Anthropic => {
                let response: anthropic::MessagesResponse =
                    serde_json::from_slice(body).map_err(|e| e.to_string())?;
                let usage = response.usage.to_token_usage();
                let translated = anthropic::translate_response(response, created);
                let bytes = serde_json::to_vec(&translated).map_err(|e| e.to_string())?;
                Ok((Bytes::from(bytes), Some(usage)))
            }
        }
    }

    /// Error body for a non-retryable upstream response, in OpenAI shape.
    pub fn error_body(&self, body: &[u8]) -> Value {
        match self.kind {
            ProviderKind::OpenAi => match serde_json::from_slice::<Value>(body) {
                Ok(value) if value.get("error").is_some() => value,
                _ => json!({
                    "error": {
                        "message": String::from_utf8_lossy(body),
                        "type": "upstream_error",
                        "param": null,
                        "code": null
                    }
                }),
            },
            ProviderKind::Anthropic => anthropic::translate_error(body),
        }
    }

    /// Turns an upstream streaming response into an OpenAI SSE byte stream.
    /// `recorder` is dropped when the stream ends or the client disconnects,
    /// which is when the request gets billed.
    pub fn stream_body(
        &self,
        response: reqwest::Response,
        include_usage: bool,
        recorder: StreamRecorder,
        created: u64,
    ) -> impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
        let upstream = response.bytes_stream().boxed();
        let handler: Box<dyn EventHandler> = match self.kind {
            ProviderKind::OpenAi => Box::new(OpenAiPassthrough {
                include_usage,
                usage: None,
                done: false,
            }),
            ProviderKind::Anthropic => Box::new(AnthropicAdapter {
                translator: StreamTranslator::new(include_usage, created),
            }),
        };
        relay(upstream, handler, recorder)
    }
}

/// What a finished (or abandoned) stream reports for billing and logs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamSummary {
    pub usage: Option<TokenUsage>,
    /// The upstream sent its end-of-stream marker.
    pub completed: bool,
    pub error: Option<String>,
}

/// Collects stream results and hands them to a callback exactly once, when
/// dropped. Dropping happens both on normal completion and when the client
/// disconnects halfway, so partial streams are still accounted for.
pub struct StreamRecorder {
    summary: StreamSummary,
    on_finish: Option<Box<dyn FnOnce(StreamSummary) + Send>>,
}

impl StreamRecorder {
    pub fn new(on_finish: impl FnOnce(StreamSummary) + Send + 'static) -> Self {
        Self {
            summary: StreamSummary::default(),
            on_finish: Some(Box::new(on_finish)),
        }
    }
}

impl Drop for StreamRecorder {
    fn drop(&mut self) {
        if let Some(on_finish) = self.on_finish.take() {
            on_finish(std::mem::take(&mut self.summary));
        }
    }
}

trait EventHandler: Send {
    /// Payloads to send to the client for one upstream event.
    fn on_event(&mut self, event: &SseEvent) -> Vec<String>;
    fn usage(&self) -> Option<TokenUsage>;
    fn is_finished(&self) -> bool;
    /// Called when the upstream fails or ends early; returns a final payload.
    fn fail(&mut self, message: &str) -> Vec<String>;
}

struct OpenAiPassthrough {
    include_usage: bool,
    usage: Option<TokenUsage>,
    done: bool,
}

impl EventHandler for OpenAiPassthrough {
    fn on_event(&mut self, event: &SseEvent) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        if event.data == "[DONE]" {
            self.done = true;
            return vec![event.data.clone()];
        }
        // Forward the upstream payload verbatim. Only chunks that mention
        // usage are parsed, which skips the JSON work for almost every chunk.
        if event.data.contains("\"usage\"")
            && let Ok(chunk) = serde_json::from_str::<Value>(&event.data)
            && let Some(usage) = chunk.get("usage").and_then(TokenUsage::from_openai)
        {
            self.usage = Some(usage);
            let no_choices = chunk
                .get("choices")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty);
            if no_choices && !self.include_usage {
                // We added include_usage ourselves; hide the extra chunk.
                return Vec::new();
            }
        }
        vec![event.data.clone()]
    }

    fn usage(&self) -> Option<TokenUsage> {
        self.usage
    }

    fn is_finished(&self) -> bool {
        self.done
    }

    fn fail(&mut self, message: &str) -> Vec<String> {
        self.done = true;
        vec![
            json!({"error": {"message": message, "type": "upstream_error", "param": null, "code": null}})
                .to_string(),
        ]
    }
}

struct AnthropicAdapter {
    translator: StreamTranslator,
}

impl EventHandler for AnthropicAdapter {
    fn on_event(&mut self, event: &SseEvent) -> Vec<String> {
        self.translator.on_event(event)
    }

    fn usage(&self) -> Option<TokenUsage> {
        self.translator.usage()
    }

    fn is_finished(&self) -> bool {
        self.translator.is_finished()
    }

    fn fail(&mut self, message: &str) -> Vec<String> {
        self.translator.fail("upstream_error", message)
    }
}

fn relay(
    mut upstream: futures_util::stream::BoxStream<'static, reqwest::Result<Bytes>>,
    mut handler: Box<dyn EventHandler>,
    mut recorder: StreamRecorder,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
    async_stream::stream! {
        let mut parser = SseParser::new();
        let mut failed = false;
        'read: loop {
            let events = match upstream.next().await {
                Some(Ok(chunk)) => parser.push(&chunk),
                Some(Err(e)) => {
                    let message = format!("upstream stream failed: {e}");
                    for payload in handler.fail(&message) {
                        yield Ok(encode_data(&payload));
                    }
                    recorder.summary.error = Some(message);
                    failed = true;
                    break 'read;
                }
                None => {
                    let events: Vec<SseEvent> = parser.finish().into_iter().collect();
                    for event in &events {
                        for payload in handler.on_event(event) {
                            yield Ok(encode_data(&payload));
                        }
                    }
                    break 'read;
                }
            };
            for event in &events {
                for payload in handler.on_event(event) {
                    yield Ok(encode_data(&payload));
                }
                recorder.summary.usage = handler.usage();
                if handler.is_finished() {
                    break 'read;
                }
            }
        }
        recorder.summary.usage = handler.usage();
        if !failed && !handler.is_finished() {
            let message = "upstream stream ended before it was complete".to_string();
            for payload in handler.fail(&message) {
                yield Ok(encode_data(&payload));
            }
            recorder.summary.error = Some(message);
        } else if !failed {
            recorder.summary.completed = true;
        }
        // Dropping the recorder at the end of the generator reports the
        // summary. It is also dropped if the client goes away mid-stream.
        drop(recorder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    fn provider(kind: ProviderKind, base_url: &str) -> Provider {
        Provider::from_config(
            "p",
            &ProviderConfig {
                kind,
                base_url: base_url.into(),
                api_key: None,
                api_key_env: None,
                timeout_ms: 1000,
                anthropic_version: "2023-06-01".into(),
                default_max_tokens: 256,
                headers: BTreeMap::new(),
                resolved_api_key: Some("secret".into()),
            },
        )
    }

    fn target(model: &str, drop_params: &[&str]) -> Target {
        Target {
            provider: "p".into(),
            model: model.into(),
            timeout_ms: None,
            drop_params: drop_params.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn body_json(bytes: &Bytes) -> Value {
        serde_json::from_slice(bytes).unwrap()
    }

    #[test]
    fn endpoints_by_kind() {
        assert_eq!(
            provider(ProviderKind::OpenAi, "https://api.openai.com/v1/").endpoint(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            provider(ProviderKind::Anthropic, "https://api.anthropic.com").endpoint(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn openai_body_rewrites_model_and_drops_params() {
        let p = provider(ProviderKind::OpenAi, "http://x/v1");
        let request = json!({
            "model": "alias", "temperature": 0.3, "top_p": 0.9, "seed": 1,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let body = body_json(
            &p.prepare_body(&request, &target("o3", &["temperature", "top_p"]), false)
                .unwrap(),
        );
        assert_eq!(
            body,
            json!({"model": "o3", "seed": 1, "messages": [{"role": "user", "content": "hi"}]})
        );
    }

    #[test]
    fn openai_stream_body_requests_usage() {
        let p = provider(ProviderKind::OpenAi, "http://x/v1");
        let request = json!({"model": "a", "stream": true, "messages": []});
        let body = body_json(&p.prepare_body(&request, &target("m", &[]), true).unwrap());
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn anthropic_body_is_translated() {
        let p = provider(ProviderKind::Anthropic, "http://x");
        let request = json!({"model": "a", "temperature": 0.5, "messages": [{"role": "user", "content": "hi"}]});
        let body = body_json(
            &p.prepare_body(
                &request,
                &target("claude-sonnet-4-5-20250929", &["temperature"]),
                false,
            )
            .unwrap(),
        );
        assert_eq!(
            body,
            json!({
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
            })
        );
    }

    #[test]
    fn anthropic_translation_errors_are_client_errors() {
        let p = provider(ProviderKind::Anthropic, "http://x");
        let err = p
            .prepare_body(
                &json!({"model": "a", "messages": "nope"}),
                &target("m", &[]),
                false,
            )
            .unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn auth_headers_by_kind() {
        let client = reqwest::Client::new();
        let openai = provider(ProviderKind::OpenAi, "http://x/v1")
            .build_request(&client, Bytes::new(), "req-1")
            .build()
            .unwrap();
        assert_eq!(openai.headers()["authorization"], "Bearer secret");
        assert_eq!(openai.headers()["x-request-id"], "req-1");

        let anthropic = provider(ProviderKind::Anthropic, "http://x")
            .build_request(&client, Bytes::new(), "req-2")
            .build()
            .unwrap();
        assert_eq!(anthropic.headers()["x-api-key"], "secret");
        assert_eq!(anthropic.headers()["anthropic-version"], "2023-06-01");
        assert!(anthropic.headers().get("authorization").is_none());
    }

    #[test]
    fn openai_response_is_forwarded_untouched() {
        let p = provider(ProviderKind::OpenAi, "http://x/v1");
        let raw = br#"{"id":"c","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7},"system_fingerprint":"fp"}"#;
        let (body, usage) = p.parse_response(raw, 0).unwrap();
        assert_eq!(&body[..], &raw[..]);
        assert_eq!(usage.unwrap().completion_tokens, 4);
        assert!(p.parse_response(b"{}", 0).is_err());
        assert!(p.parse_response(b"not json", 0).is_err());
    }

    #[test]
    fn upstream_error_bodies_keep_their_message() {
        let openai = provider(ProviderKind::OpenAi, "http://x/v1");
        let body =
            openai.error_body(br#"{"error":{"message":"bad","type":"invalid_request_error"}}"#);
        assert_eq!(body["error"]["message"], "bad");
        assert_eq!(openai.error_body(b"oops")["error"]["message"], "oops");
    }

    async fn collect(stream: impl Stream<Item = Result<Bytes, io::Error>>) -> Vec<String> {
        let chunks: Vec<_> = stream.collect().await;
        let mut parser = SseParser::new();
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(parser.push(&chunk.unwrap()).into_iter().map(|e| e.data));
        }
        out
    }

    fn recorder() -> (StreamRecorder, Arc<Mutex<Option<StreamSummary>>>) {
        let slot = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&slot);
        (
            StreamRecorder::new(move |summary| *sink.lock().unwrap() = Some(summary)),
            slot,
        )
    }

    fn upstream(
        chunks: Vec<reqwest::Result<Bytes>>,
    ) -> futures_util::stream::BoxStream<'static, reqwest::Result<Bytes>> {
        futures_util::stream::iter(chunks).boxed()
    }

    fn passthrough(include_usage: bool) -> Box<dyn EventHandler> {
        Box::new(OpenAiPassthrough {
            include_usage,
            usage: None,
            done: false,
        })
    }

    const OPENAI_STREAM: &str = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\n",
        "data: [DONE]\n\n",
    );

    #[tokio::test]
    async fn passthrough_hides_usage_chunk_unless_requested() {
        let (rec, summary) = recorder();
        let out = collect(relay(
            upstream(vec![Ok(Bytes::from(OPENAI_STREAM))]),
            passthrough(false),
            rec,
        ))
        .await;
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[1], "[DONE]");
        let summary = summary.lock().unwrap().clone().unwrap();
        assert!(summary.completed);
        assert_eq!(summary.usage.unwrap().prompt_tokens, 5);

        let (rec, _) = recorder();
        let out = collect(relay(
            upstream(vec![Ok(Bytes::from(OPENAI_STREAM))]),
            passthrough(true),
            rec,
        ))
        .await;
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn truncated_stream_gets_an_error_payload() {
        let (rec, summary) = recorder();
        let first = OPENAI_STREAM.split("\n\n").next().unwrap().to_string() + "\n\n";
        let out = collect(relay(
            upstream(vec![Ok(Bytes::from(first))]),
            passthrough(false),
            rec,
        ))
        .await;
        assert_eq!(out.len(), 2);
        assert!(out[1].contains("ended before it was complete"));
        let summary = summary.lock().unwrap().clone().unwrap();
        assert!(!summary.completed);
        assert!(summary.error.is_some());
    }

    #[tokio::test]
    async fn abandoned_stream_still_reports() {
        let (rec, summary) = recorder();
        let stream = relay(
            upstream(vec![Ok(Bytes::from(OPENAI_STREAM))]),
            passthrough(false),
            rec,
        );
        let mut stream = Box::pin(stream);
        let first = stream.next().await.unwrap().unwrap();
        assert!(first.starts_with(b"data: "));
        assert!(summary.lock().unwrap().is_none());
        drop(stream);
        let summary = summary.lock().unwrap().clone().expect("reported on drop");
        assert!(!summary.completed);
    }

    #[tokio::test]
    async fn anthropic_stream_is_translated_across_odd_chunk_boundaries() {
        let sse = std::fs::read_to_string(format!(
            "{}/tests/fixtures/anthropic/stream_text.sse",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let chunks: Vec<reqwest::Result<Bytes>> = sse
            .as_bytes()
            .chunks(7)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect();
        let (rec, summary) = recorder();
        let handler = Box::new(AnthropicAdapter {
            translator: StreamTranslator::new(false, 1),
        });
        let out = collect(relay(upstream(chunks), handler, rec)).await;
        assert_eq!(out.last().map(String::as_str), Some("[DONE]"));
        let text: String = out[..out.len() - 1]
            .iter()
            .filter_map(|p| serde_json::from_str::<Value>(p).ok())
            .filter_map(|c| {
                c["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(text, "Hello, world");
        let summary = summary.lock().unwrap().clone().unwrap();
        assert!(summary.completed);
        assert_eq!(summary.usage.unwrap().completion_tokens, 4);
    }
}
