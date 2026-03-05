//! Translation between OpenAI Chat Completions and the Anthropic Messages API.
//!
//! The two formats carry the same information but arrange it differently:
//!
//! | OpenAI                                   | Anthropic                                   |
//! |------------------------------------------|---------------------------------------------|
//! | `system` / `developer` messages          | top-level `system`                          |
//! | `assistant.tool_calls[]` (args as string)| `tool_use` content blocks (input as object) |
//! | one `tool` message per result            | `tool_result` blocks inside a `user` turn   |
//! | `max_tokens` optional                    | `max_tokens` required                       |
//! | `tool_choice: "required"`                | `tool_choice: {"type": "any"}`              |
//! | stream: one chunk shape, `[DONE]` at end | stream: typed events per content block      |
//!
//! Requests are translated OpenAI to Anthropic; responses and stream events
//! go the other way.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::openai::{
    self, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatMessage, Choice,
    ChunkChoice, ContentPart, Delta, FunctionCall, FunctionCallDelta, MessageContent,
    ResponseMessage, Role, TokenUsage, ToolCall, ToolCallDelta,
};
use crate::sse::SseEvent;

// ---------------------------------------------------------------------------
// Request types (what we send)

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<Block>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Metadata {
    pub user_id: String,
}

// ---------------------------------------------------------------------------
// Response types (what we get back)

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ResponseBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    /// `thinking`, `redacted_thinking`, server tool blocks and anything newer.
    /// None of these have an OpenAI Chat Completions equivalent.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

impl AnthropicUsage {
    /// Anthropic reports uncached input, cache writes and cache reads
    /// separately. OpenAI's `prompt_tokens` is the total of all three.
    pub fn to_token_usage(self) -> TokenUsage {
        let input = self.input_tokens.unwrap_or(0);
        let written = self.cache_creation_input_tokens.unwrap_or(0);
        let read = self.cache_read_input_tokens.unwrap_or(0);
        TokenUsage {
            prompt_tokens: input + written + read,
            completion_tokens: self.output_tokens.unwrap_or(0),
            cached_prompt_tokens: read,
            cache_write_prompt_tokens: written,
        }
    }

    /// Stream events carry partial usage; later values replace earlier ones.
    fn merge(&mut self, newer: AnthropicUsage) {
        self.input_tokens = newer.input_tokens.or(self.input_tokens);
        self.output_tokens = newer.output_tokens.or(self.output_tokens);
        self.cache_creation_input_tokens = newer
            .cache_creation_input_tokens
            .or(self.cache_creation_input_tokens);
        self.cache_read_input_tokens = newer
            .cache_read_input_tokens
            .or(self.cache_read_input_tokens);
    }
}

// ---------------------------------------------------------------------------
// OpenAI request -> Anthropic request

/// Translates a Chat Completions request for the upstream `model`.
pub fn translate_request(
    request: ChatCompletionRequest,
    model: &str,
    default_max_tokens: u32,
) -> Result<MessagesRequest, ApiError> {
    if request.n.is_some_and(|n| n > 1) {
        return Err(ApiError::invalid_request(
            "n > 1 is not supported for Anthropic models",
        ));
    }
    // JSON mode can't be honoured by this adapter. Failing here lets the
    // gateway try a later target that supports it, instead of quietly
    // returning prose to a client that asked for JSON.
    if let Some(format) = request.extra.get("response_format") {
        let kind = format.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "text" {
            return Err(ApiError::invalid_request(format!(
                "response_format {kind:?} is not supported for Anthropic models"
            )));
        }
    }
    let ignored: Vec<&str> = request
        .extra
        .keys()
        .map(String::as_str)
        .filter(|k| *k != "response_format")
        .collect();
    if !ignored.is_empty() {
        tracing::debug!(
            fields = ?ignored,
            "dropping request fields with no Anthropic equivalent"
        );
    }

    let mut system = Vec::new();
    let mut messages: Vec<Message> = Vec::new();
    for (i, message) in request.messages.into_iter().enumerate() {
        match message.role {
            Role::System | Role::Developer => {
                let text = text_only(message.content, i)?;
                if !text.is_empty() {
                    system.push(text);
                }
            }
            Role::User => push_blocks(&mut messages, MessageRole::User, user_blocks(message, i)?),
            Role::Assistant => push_blocks(
                &mut messages,
                MessageRole::Assistant,
                assistant_blocks(message, i)?,
            ),
            Role::Tool => {
                let tool_use_id = message.tool_call_id.clone().ok_or_else(|| {
                    ApiError::invalid_request(format!(
                        "messages[{i}]: tool message has no tool_call_id"
                    ))
                })?;
                let content = text_only(message.content, i)?;
                push_blocks(
                    &mut messages,
                    MessageRole::User,
                    vec![Block::ToolResult {
                        tool_use_id,
                        content,
                    }],
                );
            }
        }
    }
    if messages.is_empty() {
        return Err(ApiError::invalid_request(
            "at least one user or assistant message is required",
        ));
    }

    let tools = request.tools.filter(|t| !t.is_empty()).map(|tools| {
        tools
            .into_iter()
            .map(|tool| Tool {
                name: tool.function.name,
                description: tool.function.description,
                input_schema: tool
                    .function
                    .parameters
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            })
            .collect::<Vec<_>>()
    });
    let tool_choice = match &tools {
        Some(_) => translate_tool_choice(request.tool_choice, request.parallel_tool_calls)?,
        None => None,
    };

    Ok(MessagesRequest {
        model: model.to_string(),
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(default_max_tokens),
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        messages,
        // OpenAI accepts 0..=2, Anthropic 0..=1.
        temperature: request.temperature.map(|t| t.clamp(0.0, 1.0)),
        top_p: request.top_p,
        stop_sequences: request
            .stop
            .map(openai::StopSequences::into_vec)
            .filter(|s| !s.is_empty()),
        stream: request.stream.unwrap_or(false),
        tools,
        tool_choice,
        metadata: request.user.map(|user_id| Metadata { user_id }),
    })
}

/// Anthropic expects alternating turns, so consecutive messages with the same
/// role (for example several tool results followed by a user question) are
/// merged into one.
fn push_blocks(messages: &mut Vec<Message>, role: MessageRole, blocks: Vec<Block>) {
    if blocks.is_empty() {
        return;
    }
    match messages.last_mut() {
        Some(last) if last.role == role => last.content.extend(blocks),
        _ => messages.push(Message {
            role,
            content: blocks,
        }),
    }
}

fn text_only(content: Option<MessageContent>, index: usize) -> Result<String, ApiError> {
    match content {
        None => Ok(String::new()),
        Some(MessageContent::Text(text)) => Ok(text),
        Some(MessageContent::Parts(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text: t } => text.push_str(&t),
                    _ => {
                        return Err(ApiError::invalid_request(format!(
                            "messages[{index}]: only text content is supported in this message"
                        )));
                    }
                }
            }
            Ok(text)
        }
    }
}

fn user_blocks(message: ChatMessage, index: usize) -> Result<Vec<Block>, ApiError> {
    match message.content {
        None => Ok(Vec::new()),
        Some(MessageContent::Text(text)) => Ok(text_block(text).into_iter().collect()),
        Some(MessageContent::Parts(parts)) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ContentPart::Text { text } => blocks.extend(text_block(text)),
                    ContentPart::ImageUrl { image_url } => blocks.push(Block::Image {
                        source: image_source(&image_url.url, index)?,
                    }),
                    ContentPart::Refusal { .. } | ContentPart::Unsupported => {
                        return Err(ApiError::invalid_request(format!(
                            "messages[{index}]: unsupported content part for Anthropic models; \
                             text and image_url are supported"
                        )));
                    }
                }
            }
            Ok(blocks)
        }
    }
}

fn assistant_blocks(message: ChatMessage, index: usize) -> Result<Vec<Block>, ApiError> {
    let mut blocks = match message.content {
        None => Vec::new(),
        Some(MessageContent::Text(text)) => text_block(text).into_iter().collect(),
        Some(MessageContent::Parts(parts)) => parts
            .into_iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } | ContentPart::Refusal { refusal: text } => {
                    text_block(text)
                }
                _ => None,
            })
            .collect(),
    };
    for call in message.tool_calls.unwrap_or_default() {
        let input = if call.function.arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&call.function.arguments).map_err(|e| {
                ApiError::invalid_request(format!(
                    "messages[{index}]: arguments of tool call {:?} are not valid JSON: {e}",
                    call.id
                ))
            })?
        };
        blocks.push(Block::ToolUse {
            id: call.id,
            name: call.function.name,
            input,
        });
    }
    Ok(blocks)
}

/// Anthropic rejects empty text blocks.
fn text_block(text: String) -> Option<Block> {
    (!text.is_empty()).then_some(Block::Text { text })
}

fn image_source(url: &str, index: usize) -> Result<ImageSource, ApiError> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Ok(ImageSource::Url {
            url: url.to_string(),
        });
    };
    let parsed = rest
        .split_once(',')
        .and_then(|(meta, data)| Some((meta.strip_suffix(";base64")?, data)));
    match parsed {
        Some((media_type, data)) if !media_type.is_empty() => Ok(ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        }),
        _ => Err(ApiError::invalid_request(format!(
            "messages[{index}]: image data URLs must look like data:<media type>;base64,<data>"
        ))),
    }
}

fn translate_tool_choice(
    choice: Option<openai::ToolChoice>,
    parallel_tool_calls: Option<bool>,
) -> Result<Option<ToolChoice>, ApiError> {
    let disable_parallel_tool_use = parallel_tool_calls.map(|parallel| !parallel);
    let choice = match choice {
        None if disable_parallel_tool_use.is_none() => return Ok(None),
        None => ToolChoice::Auto {
            disable_parallel_tool_use,
        },
        Some(openai::ToolChoice::Mode(mode)) => match mode.as_str() {
            "auto" => ToolChoice::Auto {
                disable_parallel_tool_use,
            },
            "required" => ToolChoice::Any {
                disable_parallel_tool_use,
            },
            "none" => ToolChoice::None,
            other => {
                return Err(ApiError::invalid_request(format!(
                    "unknown tool_choice {other:?}; expected \"auto\", \"required\" or \"none\""
                )));
            }
        },
        Some(openai::ToolChoice::Named(named)) => ToolChoice::Tool {
            name: named.function.name,
            disable_parallel_tool_use,
        },
    };
    Ok(Some(choice))
}

// ---------------------------------------------------------------------------
// Anthropic response -> OpenAI response

pub fn finish_reason(stop_reason: Option<&str>) -> Option<String> {
    let reason = match stop_reason? {
        "end_turn" | "stop_sequence" | "pause_turn" => "stop",
        "max_tokens" | "model_context_window_exceeded" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        _ => "stop",
    };
    Some(reason.to_string())
}

pub fn translate_response(response: MessagesResponse, created: u64) -> ChatCompletion {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in response.content {
        match block {
            ResponseBlock::Text { text: t } => text.push_str(&t),
            ResponseBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id,
                kind: "function".to_string(),
                function: FunctionCall {
                    name,
                    arguments: input.to_string(),
                },
            }),
            ResponseBlock::Other => {}
        }
    }
    let content = if text.is_empty() && !tool_calls.is_empty() {
        None
    } else {
        Some(text)
    };
    ChatCompletion {
        id: response.id,
        object: "chat.completion".to_string(),
        created,
        model: response.model,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: Role::Assistant,
                content,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            },
            finish_reason: finish_reason(response.stop_reason.as_deref()),
        }],
        usage: response.usage.to_token_usage().to_openai(),
    }
}

/// Converts an Anthropic error body (`{"type":"error","error":{...}}`) into
/// the OpenAI error shape, keeping the message.
pub fn translate_error(body: &[u8]) -> Value {
    #[derive(Deserialize)]
    struct Envelope {
        error: Inner,
    }
    #[derive(Deserialize)]
    struct Inner {
        #[serde(rename = "type")]
        kind: String,
        message: String,
    }
    match serde_json::from_slice::<Envelope>(body) {
        Ok(Envelope { error }) => json!({
            "error": {"message": error.message, "type": error.kind, "param": null, "code": null}
        }),
        Err(_) => json!({
            "error": {
                "message": String::from_utf8_lossy(body),
                "type": "upstream_error",
                "param": null,
                "code": null
            }
        }),
    }
}

// ---------------------------------------------------------------------------
// Anthropic stream events -> OpenAI chunks

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: StartMessage,
    },
    ContentBlockStart {
        index: u32,
        content_block: ResponseBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: BlockDelta,
    },
    ContentBlockStop {},
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    MessageStop {},
    Ping {},
    Error {
        error: StreamError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct StartMessage {
    id: String,
    model: String,
    #[serde(default)]
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    /// `thinking_delta`, `signature_delta`, `citations_delta`, ...
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

/// Turns Anthropic stream events into OpenAI `chat.completion.chunk`
/// payloads. Each call returns the `data:` payloads to emit, in order; the
/// last payload of a complete stream is `[DONE]`.
#[derive(Debug)]
pub struct StreamTranslator {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    usage: AnthropicUsage,
    started: bool,
    /// Anthropic content block index -> OpenAI tool call index.
    tool_indices: HashMap<u32, u32>,
    finished: bool,
    failure: Option<String>,
}

impl StreamTranslator {
    pub fn new(include_usage: bool, created: u64) -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            created,
            include_usage,
            usage: AnthropicUsage::default(),
            started: false,
            tool_indices: HashMap::new(),
            finished: false,
            failure: None,
        }
    }

    /// True once `message_stop` or an `error` event has been handled.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// True if the stream ended with an error event.
    pub fn failed(&self) -> bool {
        self.failure.is_some()
    }

    /// The error message, if the stream ended with an error event.
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// Usage so far. `None` until `message_start` has been seen.
    pub fn usage(&self) -> Option<TokenUsage> {
        self.started.then(|| self.usage.to_token_usage())
    }

    pub fn on_event(&mut self, event: &SseEvent) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        let parsed: StreamEvent = match serde_json::from_str(&event.data) {
            Ok(parsed) => parsed,
            Err(e) => {
                return self.fail(
                    "upstream_error",
                    &format!("could not parse Anthropic stream event: {e}"),
                );
            }
        };
        match parsed {
            StreamEvent::MessageStart { message } => {
                self.id = message.id;
                self.model = message.model;
                self.usage.merge(message.usage);
                self.started = true;
                vec![self.chunk(
                    Delta {
                        role: Some(Role::Assistant),
                        content: Some(String::new()),
                        tool_calls: None,
                    },
                    None,
                )]
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ResponseBlock::Text { text } if !text.is_empty() => {
                    vec![self.content_chunk(text)]
                }
                ResponseBlock::ToolUse { id, name, .. } => {
                    let tool_index = self.tool_indices.len() as u32;
                    self.tool_indices.insert(index, tool_index);
                    vec![self.tool_chunk(ToolCallDelta {
                        index: tool_index,
                        id: Some(id),
                        kind: Some("function".to_string()),
                        function: FunctionCallDelta {
                            name: Some(name),
                            arguments: Some(String::new()),
                        },
                    })]
                }
                _ => Vec::new(),
            },
            StreamEvent::ContentBlockDelta { index, delta } => match delta {
                BlockDelta::TextDelta { text } if !text.is_empty() => {
                    vec![self.content_chunk(text)]
                }
                BlockDelta::InputJsonDelta { partial_json } if !partial_json.is_empty() => {
                    match self.tool_indices.get(&index) {
                        Some(&tool_index) => vec![self.tool_chunk(ToolCallDelta {
                            index: tool_index,
                            id: None,
                            kind: None,
                            function: FunctionCallDelta {
                                name: None,
                                arguments: Some(partial_json),
                            },
                        })],
                        None => Vec::new(),
                    }
                }
                _ => Vec::new(),
            },
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(usage) = usage {
                    self.usage.merge(usage);
                }
                match finish_reason(delta.stop_reason.as_deref()) {
                    Some(reason) => vec![self.chunk(Delta::default(), Some(reason))],
                    None => Vec::new(),
                }
            }
            StreamEvent::MessageStop {} => {
                self.finished = true;
                let mut out = Vec::with_capacity(2);
                if self.include_usage {
                    out.push(self.usage_chunk());
                }
                out.push("[DONE]".to_string());
                out
            }
            StreamEvent::Error { error } => self.fail(&error.kind, &error.message),
            StreamEvent::ContentBlockStop {} | StreamEvent::Ping {} | StreamEvent::Unknown => {
                Vec::new()
            }
        }
    }

    /// Emits an OpenAI-style error payload and stops the stream. OpenAI has no
    /// formal mid-stream error event, but its SDKs raise on a `data:` payload
    /// with an `error` key.
    pub fn fail(&mut self, kind: &str, message: &str) -> Vec<String> {
        self.finished = true;
        self.failure = Some(format!("{kind}: {message}"));
        vec![
            json!({"error": {"message": message, "type": kind, "param": null, "code": null}})
                .to_string(),
        ]
    }

    fn content_chunk(&self, text: String) -> String {
        self.chunk(
            Delta {
                content: Some(text),
                ..Delta::default()
            },
            None,
        )
    }

    fn tool_chunk(&self, call: ToolCallDelta) -> String {
        self.chunk(
            Delta {
                tool_calls: Some(vec![call]),
                ..Delta::default()
            },
            None,
        )
    }

    fn chunk(&self, delta: Delta, finish_reason: Option<String>) -> String {
        self.serialise(
            vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            None,
        )
    }

    fn usage_chunk(&self) -> String {
        self.serialise(Vec::new(), Some(self.usage.to_token_usage().to_openai()))
    }

    fn serialise(&self, choices: Vec<ChunkChoice>, usage: Option<openai::Usage>) -> String {
        let chunk = ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage,
        };
        serde_json::to_string(&chunk).expect("chunk serialisation cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::SseParser;

    const CREATED: u64 = 1_700_000_000;

    fn fixture(name: &str) -> String {
        let path = format!(
            "{}/tests/fixtures/anthropic/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
    }

    fn json_fixture(name: &str) -> Value {
        serde_json::from_str(&fixture(name)).unwrap()
    }

    fn request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    fn translate(value: Value) -> Result<MessagesRequest, ApiError> {
        translate_request(request(value), "claude-sonnet-4-5-20250929", 1024)
    }

    /// Runs an SSE fixture through the parser and translator, returning the
    /// emitted payloads.
    fn run_stream(sse: &str, include_usage: bool) -> (Vec<String>, StreamTranslator) {
        let mut parser = SseParser::new();
        let mut translator = StreamTranslator::new(include_usage, CREATED);
        let mut out = Vec::new();
        for event in parser.push(sse.as_bytes()).unwrap() {
            out.extend(translator.on_event(&event));
        }
        (out, translator)
    }

    #[test]
    fn translates_a_full_request_with_tools() {
        let translated = translate(json_fixture("openai_request_tools.json")).unwrap();
        assert_eq!(
            serde_json::to_value(&translated).unwrap(),
            json_fixture("anthropic_request_tools.json")
        );
    }

    #[test]
    fn simple_request_gets_default_max_tokens() {
        let translated = translate(json!({
            "model": "fast",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "temperature": 1.7
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(&translated).unwrap(),
            json!({
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "temperature": 1.0,
                "stream": true
            })
        );
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let translated = translate(json!({
            "model": "m", "max_tokens": 10, "max_completion_tokens": 20,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert_eq!(translated.max_tokens, 20);
    }

    #[test]
    fn tool_choice_variants() {
        let with_choice = |choice: Value, parallel: Option<bool>| {
            let mut body = json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": choice,
            });
            if let Some(parallel) = parallel {
                body["parallel_tool_calls"] = json!(parallel);
            }
            serde_json::to_value(translate(body).unwrap().tool_choice).unwrap()
        };
        assert_eq!(with_choice(json!("auto"), None), json!({"type": "auto"}));
        assert_eq!(with_choice(json!("none"), None), json!({"type": "none"}));
        assert_eq!(
            with_choice(json!("required"), Some(true)),
            json!({"type": "any", "disable_parallel_tool_use": false})
        );
        assert_eq!(
            with_choice(json!({"type": "function", "function": {"name": "f"}}), None),
            json!({"type": "tool", "name": "f"})
        );
        assert_eq!(with_choice(Value::Null, None), Value::Null);
    }

    #[test]
    fn tool_choice_is_dropped_without_tools() {
        let translated = translate(json!({
            "model": "m", "tool_choice": "required",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert!(translated.tools.is_none());
        assert!(translated.tool_choice.is_none());
    }

    #[test]
    fn rejects_invalid_tool_arguments() {
        let err = translate(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
                ]}
            ]
        }))
        .unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("call_1"), "{}", err.message);
    }

    #[test]
    fn empty_tool_arguments_become_an_empty_object() {
        let translated = translate(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "f", "arguments": ""}}
                ]}
            ]
        }))
        .unwrap();
        assert_eq!(
            translated.messages[1].content,
            vec![Block::ToolUse {
                id: "c".into(),
                name: "f".into(),
                input: json!({})
            }]
        );
    }

    #[test]
    fn rejects_unsupported_requests() {
        let n = translate(
            json!({"model": "m", "n": 2, "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert!(n.unwrap_err().message.contains("n > 1"));

        let only_system =
            translate(json!({"model": "m", "messages": [{"role": "system", "content": "x"}]}));
        assert!(only_system.unwrap_err().message.contains("at least one"));

        let audio = translate(
            json!({"model": "m", "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}
            ]}]}),
        );
        assert!(
            audio
                .unwrap_err()
                .message
                .contains("unsupported content part")
        );

        let json_mode = translate(
            json!({"model": "m", "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert!(json_mode.unwrap_err().message.contains("json_object"));
        let text_mode = translate(json!({"model": "m", "response_format": {"type": "text"},
            "messages": [{"role": "user", "content": "hi"}]}));
        assert!(text_mode.is_ok());

        let bad_data_url = translate(
            json!({"model": "m", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png,rawbytes"}}
            ]}]}),
        );
        assert!(bad_data_url.unwrap_err().message.contains("base64"));

        let orphan_tool =
            translate(json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}));
        assert!(orphan_tool.unwrap_err().message.contains("tool_call_id"));
    }

    #[test]
    fn translates_a_tool_use_response() {
        let response: MessagesResponse =
            serde_json::from_value(json_fixture("anthropic_response_tool_use.json")).unwrap();
        let translated = translate_response(response, CREATED);
        assert_eq!(
            serde_json::to_value(&translated).unwrap(),
            json_fixture("openai_response_tool_use.json")
        );
    }

    #[test]
    fn text_only_response() {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " there"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        }))
        .unwrap();
        let translated = translate_response(response, CREATED);
        let choice = &translated.choices[0];
        assert_eq!(choice.message.content.as_deref(), Some("Hello there"));
        assert_eq!(choice.message.tool_calls, None);
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
        assert_eq!(translated.usage.total_tokens, 7);
        assert_eq!(translated.usage.prompt_tokens_details, None);
    }

    #[test]
    fn stop_reason_mapping() {
        let cases = [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("pause_turn", "stop"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("tool_use", "tool_calls"),
            ("refusal", "content_filter"),
            ("something_new", "stop"),
        ];
        for (anthropic, openai) in cases {
            assert_eq!(
                finish_reason(Some(anthropic)).as_deref(),
                Some(openai),
                "{anthropic}"
            );
        }
        assert_eq!(finish_reason(None), None);
    }

    #[test]
    fn translates_errors() {
        let body = br#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: too large"}}"#;
        assert_eq!(
            translate_error(body),
            json!({"error": {"message": "max_tokens: too large", "type": "invalid_request_error", "param": null, "code": null}})
        );
        assert_eq!(
            translate_error(b"<html>bad gateway</html>")["error"]["type"],
            "upstream_error"
        );
    }

    #[test]
    fn text_stream_matches_golden_chunks() {
        let (payloads, translator) = run_stream(&fixture("stream_text.sse"), true);
        let expected: Vec<String> = fixture("stream_text.expected.jsonl")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(payloads.len(), expected.len(), "{payloads:#?}");
        for (got, want) in payloads.iter().zip(&expected) {
            if want == "[DONE]" {
                assert_eq!(got, want);
            } else {
                let got: Value = serde_json::from_str(got).unwrap();
                let want: Value = serde_json::from_str(want).unwrap();
                assert_eq!(got, want);
            }
        }
        assert!(translator.is_finished());
        assert!(!translator.failed());
        assert_eq!(
            translator.usage(),
            Some(TokenUsage {
                prompt_tokens: 25,
                completion_tokens: 4,
                ..TokenUsage::default()
            })
        );
    }

    #[test]
    fn usage_chunk_only_when_requested() {
        let (payloads, translator) = run_stream(&fixture("stream_text.sse"), false);
        assert_eq!(payloads.len(), 5);
        assert!(payloads.iter().all(|p| !p.contains("\"usage\"")));
        // Usage is still tracked for billing.
        assert_eq!(translator.usage().unwrap().completion_tokens, 4);
    }

    #[test]
    fn tool_use_stream_produces_openai_tool_call_deltas() {
        let (payloads, translator) = run_stream(&fixture("stream_tool_use.sse"), true);
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));

        let chunks: Vec<ChatCompletionChunk> = payloads[..payloads.len() - 1]
            .iter()
            .map(|p| serde_json::from_str(p).unwrap())
            .collect();
        for chunk in &chunks {
            assert_eq!(chunk.id, "msg_01ABC");
            assert_eq!(chunk.object, "chat.completion.chunk");
            assert_eq!(chunk.model, "claude-sonnet-4-5-20250929");
        }

        // Rebuild the message the way an OpenAI client would.
        let mut text = String::new();
        let mut calls: Vec<(String, String, String)> = Vec::new();
        let mut finish = None;
        for chunk in &chunks {
            for choice in &chunk.choices {
                text.push_str(choice.delta.content.as_deref().unwrap_or(""));
                for call in choice.delta.tool_calls.iter().flatten() {
                    let i = call.index as usize;
                    if i == calls.len() {
                        calls.push((
                            call.id.clone().expect("first delta carries the id"),
                            call.function
                                .name
                                .clone()
                                .expect("first delta carries the name"),
                            String::new(),
                        ));
                    }
                    calls[i]
                        .2
                        .push_str(call.function.arguments.as_deref().unwrap_or(""));
                }
                if choice.finish_reason.is_some() {
                    finish = choice.finish_reason.clone();
                }
            }
        }
        assert_eq!(text, "Checking both cities.");
        assert_eq!(finish.as_deref(), Some("tool_calls"));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "toolu_01A");
        assert_eq!(calls[1].1, "get_weather");
        let args: Vec<Value> = calls
            .iter()
            .map(|c| serde_json::from_str(&c.2).unwrap())
            .collect();
        assert_eq!(args, [json!({"city": "Paris"}), json!({"city": "Tokyo"})]);

        let usage = chunks.last().unwrap().usage.unwrap();
        assert_eq!(usage.prompt_tokens, 668);
        assert_eq!(usage.completion_tokens, 89);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 256);
        assert!(!translator.failed());
    }

    #[test]
    fn error_event_ends_the_stream_with_an_error_payload() {
        let (payloads, translator) = run_stream(&fixture("stream_error.sse"), true);
        assert_eq!(payloads.len(), 3, "{payloads:#?}");
        let error: Value = serde_json::from_str(&payloads[2]).unwrap();
        assert_eq!(error["error"]["type"], "overloaded_error");
        assert_eq!(error["error"]["message"], "Overloaded");
        assert!(translator.is_finished());
        assert_eq!(translator.failure(), Some("overloaded_error: Overloaded"));
        assert!(!payloads.contains(&"[DONE]".to_string()));
    }

    #[test]
    fn garbage_event_fails_cleanly() {
        let mut translator = StreamTranslator::new(false, CREATED);
        let out = translator.on_event(&SseEvent {
            event: None,
            data: "{oops".into(),
        });
        assert_eq!(out.len(), 1);
        assert!(translator.failed());
        assert!(
            translator
                .on_event(&SseEvent {
                    event: None,
                    data: "{}".into()
                })
                .is_empty()
        );
    }
}
