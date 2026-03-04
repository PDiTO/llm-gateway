//! Measures the latency the gateway adds on top of an upstream.
//!
//! A mock upstream and the real gateway router run in-process on localhost.
//! The same request is sent alternately straight to the mock and through the
//! gateway, so both see the same machine conditions, and the latency
//! distributions are compared. Run with:
//!
//! ```sh
//! cargo bench --bench overhead
//! BENCH_REQUESTS=50000 cargo bench --bench overhead
//! ```
//!
//! Client, gateway and mock share the same CPU, so absolute numbers are
//! pessimistic compared with a gateway on its own host. The difference
//! between the two paths is what matters.

#[path = "../tests/common/mod.rs"]
mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{
    Behavior, KEY, MockUpstream, TestGateway, anthropic_message, openai_completion, openai_stream,
};
use futures_util::StreamExt;
use llm_gateway::auth::hash_key;
use serde_json::{Value, json};

struct Summary {
    p50: Duration,
    p90: Duration,
    p99: Duration,
    p999: Duration,
    mean: Duration,
}

fn summarise(mut samples: Vec<Duration>) -> Summary {
    samples.sort_unstable();
    let pick = |q: f64| samples[((samples.len() as f64 - 1.0) * q).round() as usize];
    let total: Duration = samples.iter().sum();
    Summary {
        p50: pick(0.50),
        p90: pick(0.90),
        p99: pick(0.99),
        p999: pick(0.999),
        mean: total / samples.len() as u32,
    }
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn print_row(label: &str, s: &Summary) {
    println!(
        "| {label:<34} | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.0} |",
        us(s.p50),
        us(s.p90),
        us(s.p99),
        us(s.p999),
        us(s.mean)
    );
}

fn print_delta(label: &str, via: &Summary, direct: &Summary) {
    let d = |a: Duration, b: Duration| us(a) - us(b);
    println!(
        "| {label:<34} | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.0} |",
        d(via.p50, direct.p50),
        d(via.p90, direct.p90),
        d(via.p99, direct.p99),
        d(via.p999, direct.p999),
        d(via.mean, direct.mean)
    );
}

fn table_header(title: &str) {
    println!("\n{title}\n");
    println!(
        "| {:<34} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
        "path (microseconds)", "p50", "p90", "p99", "p99.9", "mean"
    );
    println!(
        "|{:-<36}|{:->10}|{:->10}|{:->10}|{:->10}|{:->10}|",
        "", "", "", "", "", ""
    );
}

async fn timed(request: reqwest::RequestBuilder) -> Duration {
    let started = Instant::now();
    let response = request.send().await.expect("request failed");
    assert!(
        response.status().is_success(),
        "status {}",
        response.status()
    );
    response.bytes().await.expect("body failed");
    started.elapsed()
}

/// Time to the first body chunk and to the end of the stream.
async fn timed_stream(request: reqwest::RequestBuilder) -> (Duration, Duration) {
    let started = Instant::now();
    let response = request.send().await.expect("request failed");
    assert!(response.status().is_success());
    let mut body = response.bytes_stream();
    let mut first = None;
    while let Some(chunk) = body.next().await {
        chunk.expect("stream failed");
        first.get_or_insert_with(|| started.elapsed());
    }
    (first.unwrap_or_default(), started.elapsed())
}

fn config(openai_url: &str, anthropic_url: &str) -> String {
    format!(
        r#"
[providers.openai]
kind = "openai"
base_url = "{openai_url}"
api_key = "sk-bench"

[providers.anthropic]
kind = "anthropic"
base_url = "{anthropic_url}"
api_key = "sk-ant-bench"

[[models]]
name = "gpt"
targets = [{{ provider = "openai", model = "gpt-4.1-mini" }}]

[[models]]
name = "claude"
targets = [{{ provider = "anthropic", model = "claude-sonnet-4-5-20250929" }}]

[prices."gpt-4.1-mini"]
input_per_mtok = 0.4
output_per_mtok = 1.6

[[keys]]
name = "bench"
key_sha256 = "{hash}"
requests_per_minute = 4000000000
"#,
        hash = hash_key(KEY)
    )
}

fn main() {
    let requests: usize = std::env::var("BENCH_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let concurrency: usize = std::env::var("BENCH_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    // Format log lines as the real binary would, but throw them away, so the
    // measured overhead includes logging.
    tracing_subscriber::fmt()
        .json()
        .with_env_filter("info")
        .with_writer(std::io::sink)
        .init();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(run(requests, concurrency));
}

async fn run(requests: usize, concurrency: usize) {
    let openai =
        MockUpstream::start(Behavior::Json(openai_completion("The answer is 4.", 20, 6))).await;
    let anthropic =
        MockUpstream::start(Behavior::Json(anthropic_message("The answer is 4.", 20, 6))).await;
    let gw = TestGateway::start(&config(&openai.openai_url(), &anthropic.anthropic_url())).await;
    let client = reqwest::Client::builder()
        .tcp_nodelay(true)
        .build()
        .unwrap();

    let body = |model: &str| -> Value {
        json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a calculator."},
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            "max_tokens": 32
        })
    };
    let direct = || {
        client
            .post(format!("{}/chat/completions", openai.openai_url()))
            .json(&body("gpt-4.1-mini"))
    };
    let direct_anthropic = || {
        client
            .post(format!("{}/v1/messages", anthropic.anthropic_url()))
            .json(&json!({"model": "claude-sonnet-4-5-20250929", "max_tokens": 32, "messages": [{"role": "user", "content": "What is 2 + 2?"}]}))
    };
    let via = |model: &str| {
        client
            .post(format!("{}/v1/chat/completions", gw.url))
            .bearer_auth(KEY)
            .json(&body(model))
    };

    println!(
        "llm-gateway overhead benchmark: {requests} sequential requests per path, \
         {concurrency}-way concurrency for the throughput run"
    );

    // Warm up connection pools on both hops.
    for _ in 0..2_000 {
        timed(direct()).await;
        timed(via("gpt")).await;
        timed(direct_anthropic()).await;
        timed(via("claude")).await;
    }

    // Sequential, alternating so both paths see the same conditions.
    let mut direct_samples = Vec::with_capacity(requests);
    let mut via_samples = Vec::with_capacity(requests);
    let mut direct_anthropic_samples = Vec::with_capacity(requests);
    let mut via_anthropic_samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        direct_samples.push(timed(direct()).await);
        via_samples.push(timed(via("gpt")).await);
        direct_anthropic_samples.push(timed(direct_anthropic()).await);
        via_anthropic_samples.push(timed(via("claude")).await);
    }
    let direct_summary = summarise(direct_samples);
    let via_summary = summarise(via_samples);
    let direct_anthropic_summary = summarise(direct_anthropic_samples);
    let via_anthropic_summary = summarise(via_anthropic_samples);

    table_header("Non-streaming, one request at a time");
    print_row("mock upstream directly (OpenAI)", &direct_summary);
    print_row("through gateway, passthrough", &via_summary);
    print_delta(
        "added by gateway, passthrough",
        &via_summary,
        &direct_summary,
    );
    print_row(
        "mock upstream directly (Anthropic)",
        &direct_anthropic_summary,
    );
    print_row("through gateway, translated", &via_anthropic_summary);
    print_delta(
        "added by gateway, translated",
        &via_anthropic_summary,
        &direct_anthropic_summary,
    );

    // Streaming: 40 content chunks, no artificial delay.
    let words: Vec<String> = (0..40).map(|i| format!(" word{i}")).collect();
    let word_refs: Vec<&str> = words.iter().map(String::as_str).collect();
    openai.set_default(Behavior::sse(openai_stream(&word_refs, 20, 40)));
    let stream_requests = (requests / 4).max(1_000);
    let mut stream_body = body("gpt");
    stream_body["stream"] = json!(true);
    let direct_stream = || {
        let mut b = stream_body.clone();
        b["model"] = json!("gpt-4.1-mini");
        client
            .post(format!("{}/chat/completions", openai.openai_url()))
            .json(&b)
    };
    let via_stream = || {
        client
            .post(format!("{}/v1/chat/completions", gw.url))
            .bearer_auth(KEY)
            .json(&stream_body)
    };
    for _ in 0..500 {
        timed_stream(direct_stream()).await;
        timed_stream(via_stream()).await;
    }
    let (mut d_first, mut d_total, mut v_first, mut v_total) = (vec![], vec![], vec![], vec![]);
    for _ in 0..stream_requests {
        let (first, total) = timed_stream(direct_stream()).await;
        d_first.push(first);
        d_total.push(total);
        let (first, total) = timed_stream(via_stream()).await;
        v_first.push(first);
        v_total.push(total);
    }
    let (d_first, d_total, v_first, v_total) = (
        summarise(d_first),
        summarise(d_total),
        summarise(v_first),
        summarise(v_total),
    );
    table_header(&format!(
        "Streaming, 40 chunks, {stream_requests} requests per path"
    ));
    print_row("direct, first byte", &d_first);
    print_row("gateway, first byte", &v_first);
    print_delta("added to first byte", &v_first, &d_first);
    print_row("direct, whole stream", &d_total);
    print_row("gateway, whole stream", &v_total);
    print_delta("added to whole stream", &v_total, &d_total);

    // Throughput under concurrency, non-streaming passthrough.
    openai.set_default(Behavior::Json(openai_completion("The answer is 4.", 20, 6)));
    let concurrent_total = requests * 2;
    let mut rows = Vec::new();
    for (label, through_gateway) in [("mock upstream directly", false), ("through gateway", true)] {
        let per_worker = concurrent_total / concurrency;
        let client = client.clone();
        let target = if through_gateway {
            format!("{}/v1/chat/completions", gw.url)
        } else {
            format!("{}/chat/completions", openai.openai_url())
        };
        let payload = Arc::new(body(if through_gateway {
            "gpt"
        } else {
            "gpt-4.1-mini"
        }));
        let started = Instant::now();
        let workers: Vec<_> = (0..concurrency)
            .map(|_| {
                let client = client.clone();
                let target = target.clone();
                let payload = Arc::clone(&payload);
                tokio::spawn(async move {
                    let mut samples = Vec::with_capacity(per_worker);
                    for _ in 0..per_worker {
                        let request = client.post(&target).bearer_auth(KEY).json(&*payload);
                        samples.push(timed(request).await);
                    }
                    samples
                })
            })
            .collect();
        let mut samples = Vec::with_capacity(concurrent_total);
        for worker in workers {
            samples.extend(worker.await.unwrap());
        }
        let elapsed = started.elapsed();
        let throughput = samples.len() as f64 / elapsed.as_secs_f64();
        rows.push((label, summarise(samples), throughput));
    }
    println!(
        "\nNon-streaming, {concurrency} concurrent clients, {concurrent_total} requests per path\n"
    );
    println!(
        "| {:<34} | {:>8} | {:>8} | {:>10} |",
        "path", "p50 us", "p99 us", "req/s"
    );
    println!("|{:-<36}|{:->10}|{:->10}|{:->12}|", "", "", "", "");
    for (label, summary, throughput) in &rows {
        println!(
            "| {label:<34} | {:>8.0} | {:>8.0} | {:>10.0} |",
            us(summary.p50),
            us(summary.p99),
            throughput
        );
    }

    let metrics = gw.metrics().await;
    let served = metrics
        .lines()
        .filter(|l| l.starts_with("llm_gateway_requests_total{"))
        .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
        .sum::<f64>();
    println!("\ngateway served {served} requests during the run");
}
