# syntax=docker/dockerfile:1

# ---- build ------------------------------------------------------------------
FROM rust:1-slim-bookworm AS builder
WORKDIR /build

# Build dependencies against stub sources first, so this layer is reused
# until Cargo.toml or Cargo.lock change.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src benches \
    && echo 'fn main() {}' > src/main.rs \
    && touch src/lib.rs \
    && echo 'fn main() {}' > benches/overhead.rs \
    && cargo build --release --locked \
    && rm -rf src benches

COPY src ./src
COPY benches ./benches
# COPY keeps the original mtimes, which can be older than the stub build.
# Touching makes sure cargo rebuilds the real crate.
RUN touch src/main.rs src/lib.rs \
    && cargo build --release --locked --bin llm-gateway

# ---- runtime ----------------------------------------------------------------
# distroless/cc has glibc, libgcc and CA certificates and nothing else: no
# shell, no package manager. The builder above is Debian 12 to match it.
FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder /build/target/release/llm-gateway /usr/local/bin/llm-gateway
COPY gateway.toml /etc/llm-gateway/gateway.toml

ENV LLM_GATEWAY_CONFIG=/etc/llm-gateway/gateway.toml \
    LOG_FORMAT=json \
    RUST_LOG=info
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/llm-gateway"]
CMD ["serve"]
