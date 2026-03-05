use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use llm_gateway::auth;
use llm_gateway::config::Config;
use llm_gateway::gateway::Gateway;
use llm_gateway::server;

#[derive(Parser)]
#[command(
    name = "llm-gateway",
    version,
    about = "OpenAI-compatible gateway for several LLM providers"
)]
struct Cli {
    /// Path to the TOML config file.
    #[arg(
        short,
        long,
        env = "LLM_GATEWAY_CONFIG",
        default_value = "gateway.toml",
        global = true
    )]
    config: PathBuf,

    /// Log format: "pretty" for humans, "json" for log pipelines.
    #[arg(long, env = "LOG_FORMAT", default_value = "pretty", global = true)]
    log_format: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the gateway (the default).
    Serve,
    /// Validate the config file and print any warnings.
    Check,
    /// Print the SHA-256 hash to put in a [[keys]] entry for an existing key.
    /// Reads the key from stdin if it isn't given, which keeps it out of shell
    /// history and the process list.
    HashKey {
        /// The plaintext key.
        key: Option<String>,
    },
    /// Generate a new random key and print it with its hash.
    GenKey,
}

fn init_tracing(format: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,llm_gateway=info"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    if format == "json" {
        builder
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .init();
    } else {
        builder.init();
    }
}

fn load_config(path: &PathBuf) -> Result<Config, ExitCode> {
    Config::load(path).map_err(|e| {
        eprintln!("error: {e}");
        ExitCode::from(2)
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::HashKey { key } => {
            let key = match key {
                Some(key) => key,
                None => {
                    let mut line = String::new();
                    if let Err(e) = std::io::stdin().read_line(&mut line) {
                        eprintln!("error: could not read the key from stdin: {e}");
                        return ExitCode::FAILURE;
                    }
                    line
                }
            };
            let key = key.trim();
            if key.is_empty() {
                eprintln!("error: empty key");
                return ExitCode::FAILURE;
            }
            println!("{}", auth::hash_key(key));
            ExitCode::SUCCESS
        }
        Command::GenKey => {
            let key = auth::generate_key();
            println!("key:        {key}");
            println!("key_sha256: {}", auth::hash_key(&key));
            ExitCode::SUCCESS
        }
        Command::Check => match load_config(&cli.config) {
            Ok(config) => {
                for warning in config.warnings() {
                    println!("warning: {warning}");
                }
                println!(
                    "{}: ok ({} providers, {} models, {} keys)",
                    cli.config.display(),
                    config.providers.len(),
                    config.models.len(),
                    config.keys.len()
                );
                ExitCode::SUCCESS
            }
            Err(code) => code,
        },
        Command::Serve => {
            init_tracing(&cli.log_format);
            let config = match load_config(&cli.config) {
                Ok(config) => config,
                Err(code) => return code,
            };
            serve(config).await
        }
    }
}

async fn serve(config: Config) -> ExitCode {
    for warning in config.warnings() {
        warn!("{warning}");
    }
    let addr: SocketAddr = config.server.listen.parse().expect("validated at load");
    let grace = Duration::from_millis(config.server.shutdown_grace_ms);
    let providers = config.providers.len();
    let models = config.models.len();

    let gateway = Arc::new(Gateway::new(config));
    let app = server::router(gateway);
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            error!(%addr, "could not bind: {e}");
            return ExitCode::FAILURE;
        }
    };
    info!(%addr, providers, models, "llm-gateway listening");

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut server = tokio::spawn(async move {
        let mut rx = shutdown_rx;
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.wait_for(|stop| *stop).await;
            })
            .await
    });

    tokio::select! {
        result = &mut server => {
            return match result {
                Ok(Ok(())) => ExitCode::SUCCESS,
                Ok(Err(e)) => { error!("server error: {e}"); ExitCode::FAILURE }
                Err(e) => { error!("server task failed: {e}"); ExitCode::FAILURE }
            };
        }
        () = shutdown_signal() => {}
    }

    info!(
        grace_ms = grace.as_millis() as u64,
        "shutting down, waiting for in-flight requests"
    );
    let _ = shutdown_tx.send(true);
    match tokio::time::timeout(grace, &mut server).await {
        Ok(_) => {
            info!("shutdown complete");
            ExitCode::SUCCESS
        }
        Err(_) => {
            warn!("grace period over, dropping remaining connections");
            server.abort();
            ExitCode::SUCCESS
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}
