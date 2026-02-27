//! HTTP routes and middleware.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use bytes::Bytes;
use serde_json::{Map, Value, json};
use tracing::Instrument;

use crate::error::ApiError;
use crate::gateway::Gateway;

pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// The ID of the current request, taken from the client's `x-request-id`
/// header when it looks sane, or generated otherwise. It is logged on every
/// line for the request, echoed back, and forwarded to the upstream.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

pub fn router(gateway: Arc<Gateway>) -> Router {
    let body_limit = gateway.max_body_bytes;
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(middleware::from_fn(request_id))
        .with_state(gateway)
}

async fn request_id(mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|v| {
            !v.is_empty()
                && v.len() <= 128
                && v.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
        })
        .map(str::to_string)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    request.extensions_mut().insert(RequestId(id.clone()));
    let span = tracing::info_span!(
        "request",
        request_id = %id,
        method = %request.method(),
        path = %request.uri().path(),
    );
    let mut response = next.run(request).instrument(span).await;
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

async fn chat_completions(
    State(gateway): State<Arc<Gateway>>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    gateway.chat_completions(&request_id, &headers, body).await
}

async fn list_models(
    State(gateway): State<Arc<Gateway>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    gateway.keys().authenticate(&headers)?;
    let data: Vec<Value> = gateway
        .model_names()
        .iter()
        .map(|name| json!({"id": name, "object": "model", "created": 0, "owned_by": "llm-gateway"}))
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

async fn healthz(State(gateway): State<Arc<Gateway>>) -> Json<Value> {
    let circuits: Map<String, Value> = gateway
        .circuit_states()
        .into_iter()
        .map(|(name, state)| (name.to_string(), Value::from(state.as_str())))
        .collect();
    Json(json!({"status": "ok", "circuits": circuits}))
}

async fn metrics(State(gateway): State<Arc<Gateway>>) -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        gateway.render_metrics(),
    )
}

async fn not_found() -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "unknown route; this gateway serves POST /v1/chat/completions and GET /v1/models",
    )
}
