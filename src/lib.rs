//! An OpenAI-compatible HTTP gateway that sits in front of several LLM providers.

pub mod backoff;
pub mod clock;
pub mod config;
pub mod ratelimit;
