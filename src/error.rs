//! Errors returned to clients, in the OpenAI error shape:
//!
//! ```json
//! {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
//! ```

use std::time::Duration;

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: Option<&'static str>,
    pub message: String,
    pub retry_after: Option<Duration>,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            code: None,
            message: message.into(),
            retry_after: None,
        }
    }

    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", message)
            .with_code("invalid_api_key")
    }

    pub fn model_not_found(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("the model {model:?} is not configured on this gateway; GET /v1/models lists the available ones"),
        )
        .with_code("model_not_found")
    }

    pub fn rate_limited(key: &str, retry_after: Duration) -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            format!(
                "rate limit exceeded for key {key:?}; retry in {:.1}s",
                retry_after.as_secs_f64()
            ),
        )
        .with_code("rate_limit_exceeded")
        .with_retry_after(retry_after)
    }

    pub fn budget_exceeded(key: &str, spent: f64, budget: f64) -> Self {
        Self::new(
            StatusCode::PAYMENT_REQUIRED,
            "insufficient_quota",
            format!(
                "daily budget exhausted for key {key:?}: spent ${spent:.4} of ${budget:.2}; \
                 the budget resets at 00:00 UTC"
            ),
        )
        .with_code("budget_exceeded")
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    pub fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": null,
                "code": self.code,
            }
        })
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}: {}",
            self.status.as_u16(),
            self.kind,
            self.message
        )
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, axum::Json(self.body())).into_response();
        if let Some(retry_after) = self.retry_after {
            // Retry-After is whole seconds; round up so clients don't come
            // back too early.
            let secs = retry_after.as_secs_f64().ceil().max(1.0) as u64;
            if let Ok(value) = HeaderValue::from_str(&secs.to_string()) {
                response.headers_mut().insert(header::RETRY_AFTER, value);
            }
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialises_in_openai_shape() {
        let err = ApiError::model_not_found("gpt-9");
        assert_eq!(err.body()["error"]["type"], "invalid_request_error");
        assert_eq!(err.body()["error"]["code"], "model_not_found");
        assert!(err.body()["error"]["param"].is_null());
    }

    #[test]
    fn retry_after_header_rounds_up() {
        let response = ApiError::rate_limited("k", Duration::from_millis(1200)).into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "2");
    }
}
