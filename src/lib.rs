//! An OpenAI-compatible HTTP gateway that sits in front of several LLM
//! providers.
//!
//! A request comes in on `POST /v1/chat/completions`, is authenticated
//! against a hashed key, rate limited and budget checked, then routed by its
//! `model` field to an ordered list of upstream targets. Targets are tried in
//! order with retries and per-provider circuit breakers until one answers.
//! Anthropic targets are translated to and from the OpenAI format, including
//! streams.
//!
//! The binary lives in `main.rs`; everything it needs is exposed here so the
//! integration tests and benchmarks can run the real router in-process.

pub mod anthropic;
pub mod auth;
pub mod backoff;
pub mod cache;
pub mod circuit;
pub mod clock;
pub mod config;
pub mod cost;
pub mod error;
pub mod metrics;
pub mod openai;
pub mod providers;
pub mod ratelimit;
pub mod sse;
