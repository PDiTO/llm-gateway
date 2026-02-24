//! An OpenAI-compatible HTTP gateway that sits in front of several LLM providers.

pub mod anthropic;
pub mod auth;
pub mod backoff;
pub mod cache;
pub mod circuit;
pub mod clock;
pub mod config;
pub mod cost;
pub mod error;
pub mod openai;
pub mod ratelimit;
pub mod sse;
