//! An OpenAI-compatible HTTP gateway that sits in front of several LLM providers.

pub mod backoff;
pub mod circuit;
pub mod clock;
pub mod config;
pub mod error;
pub mod openai;
pub mod ratelimit;
pub mod sse;
