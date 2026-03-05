//! OpenAI Chat Completions wire types.
//!
//! Requests to OpenAI-compatible upstreams are forwarded as JSON values so that
//! fields the gateway doesn't know about still get through. These typed
//! structs are used where the gateway has to understand the request, which is
//! mainly translation to and from the Anthropic Messages API.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    /// Everything else (`seed`, `logprobs`, `response_format`, ...).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    /// Assistant refusals sent back in history.
    Refusal {
        refusal: String,
    },
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, as a string.
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"none"`, `"auto"` or `"required"`.
    Mode(String),
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Deserialize)]
pub struct NamedToolChoice {
    pub function: NamedFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NamedFunction {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::One(s) => vec![s],
            StopSequences::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

// ---------------------------------------------------------------------------
// Responses

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseMessage {
    pub role: Role,
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

// ---------------------------------------------------------------------------
// Streaming chunks

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub function: FunctionCallDelta,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FunctionCallDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Token accounting shared by every provider

/// Token counts the gateway bills on, normalised across providers.
/// `prompt_tokens` is the total, as in OpenAI's usage: it includes cache
/// reads (`cached_prompt_tokens`) and cache writes
/// (`cache_write_prompt_tokens`, which only Anthropic reports).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub cache_write_prompt_tokens: u64,
}

impl TokenUsage {
    /// Reads the `usage` object of an OpenAI response or final stream chunk.
    pub fn from_openai(usage: &Value) -> Option<Self> {
        let usage = usage.as_object()?;
        let field = |name: &str| usage.get(name).and_then(Value::as_u64);
        Some(Self {
            prompt_tokens: field("prompt_tokens")?,
            completion_tokens: field("completion_tokens").unwrap_or(0),
            cached_prompt_tokens: usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cache_write_prompt_tokens: 0,
        })
    }

    pub fn to_openai(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.prompt_tokens + self.completion_tokens,
            prompt_tokens_details: (self.cached_prompt_tokens > 0).then_some(PromptTokensDetails {
                cached_tokens: self.cached_prompt_tokens,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_a_request_with_tools_and_parts() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "smart",
            "messages": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "low"}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"cat\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a cat"}
            ],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "stop": "END",
            "seed": 42
        }))
        .unwrap();
        assert_eq!(request.messages.len(), 4);
        assert_eq!(request.messages[0].role, Role::Developer);
        assert!(matches!(request.tool_choice, Some(ToolChoice::Named(_))));
        assert_eq!(request.stop.unwrap().into_vec(), ["END"]);
        assert_eq!(request.extra["seed"], 42);
    }

    #[test]
    fn unknown_content_parts_are_flagged() {
        let message: ChatMessage = serde_json::from_value(json!({
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]
        }))
        .unwrap();
        let Some(MessageContent::Parts(parts)) = message.content else {
            panic!("expected parts")
        };
        assert!(matches!(parts[0], ContentPart::Unsupported));
    }

    #[test]
    fn usage_round_trip() {
        let usage = TokenUsage::from_openai(&json!({
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 60}
        }))
        .unwrap();
        assert_eq!(
            usage,
            TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                cached_prompt_tokens: 60,
                cache_write_prompt_tokens: 0,
            }
        );
        assert_eq!(usage.to_openai().total_tokens, 120);
        assert_eq!(TokenUsage::from_openai(&json!(null)), None);
    }

    #[test]
    fn chunk_serialisation_omits_empty_fields() {
        let chunk = ChatCompletionChunk {
            id: "c1".into(),
            object: "chat.completion.chunk".into(),
            created: 1,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some("hi".into()),
                    ..Delta::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        assert_eq!(
            serde_json::to_value(&chunk).unwrap(),
            json!({
                "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]
            })
        );
    }
}
