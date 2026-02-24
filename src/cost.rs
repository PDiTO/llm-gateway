//! Cost calculation from token usage and the configured price table.

use std::collections::{BTreeMap, HashMap};

use crate::config::Price;
use crate::openai::TokenUsage;

const PER_MILLION: f64 = 1_000_000.0;

/// USD cost of one response. Cached prompt tokens are billed at the cached
/// rate when one is configured, and at the normal input rate otherwise.
pub fn cost_usd(price: &Price, usage: TokenUsage) -> f64 {
    let cached = usage.cached_prompt_tokens.min(usage.prompt_tokens);
    let uncached = usage.prompt_tokens - cached;
    let cached_rate = price.cached_input_per_mtok.unwrap_or(price.input_per_mtok);
    (uncached as f64 * price.input_per_mtok
        + cached as f64 * cached_rate
        + usage.completion_tokens as f64 * price.output_per_mtok)
        / PER_MILLION
}

/// Prices keyed by upstream model ID (the `model` of a route target, not the
/// alias the client used).
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    prices: HashMap<String, Price>,
}

impl PriceTable {
    pub fn new(prices: &BTreeMap<String, Price>) -> Self {
        Self {
            prices: prices.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        }
    }

    /// `None` when the model has no price entry.
    pub fn cost(&self, model: &str, usage: TokenUsage) -> Option<f64> {
        self.prices.get(model).map(|price| cost_usd(price, usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u64, completion: u64, cached: u64) -> TokenUsage {
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            cached_prompt_tokens: cached,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    const SONNET: Price = Price {
        input_per_mtok: 2.0,
        output_per_mtok: 10.0,
        cached_input_per_mtok: Some(0.2),
    };

    #[test]
    fn simple_input_and_output() {
        let price = Price {
            input_per_mtok: 1.0,
            output_per_mtok: 5.0,
            cached_input_per_mtok: None,
        };
        // 1M in at $1 + 200k out at $5.
        assert!(close(cost_usd(&price, usage(1_000_000, 200_000, 0)), 2.0));
    }

    #[test]
    fn cached_tokens_use_the_cached_rate() {
        // 3,500 prompt tokens of which 2,000 cached, 95 completion tokens.
        let expected = (1_500.0 * 2.0 + 2_000.0 * 0.2 + 95.0 * 10.0) / 1e6;
        assert!(close(cost_usd(&SONNET, usage(3_500, 95, 2_000)), expected));
    }

    #[test]
    fn cached_tokens_fall_back_to_the_input_rate() {
        let price = Price {
            cached_input_per_mtok: None,
            ..SONNET
        };
        assert!(close(
            cost_usd(&price, usage(1_000, 0, 600)),
            cost_usd(&price, usage(1_000, 0, 0))
        ));
    }

    #[test]
    fn inconsistent_cached_count_is_clamped() {
        // A buggy upstream reporting more cached than total tokens must not
        // produce a negative or inflated cost.
        let cost = cost_usd(&SONNET, usage(100, 0, 500));
        assert!(close(cost, 100.0 * 0.2 / 1e6));
    }

    #[test]
    fn zero_usage_is_free() {
        assert_eq!(cost_usd(&SONNET, TokenUsage::default()), 0.0);
    }

    #[test]
    fn table_lookup() {
        let table = PriceTable::new(&BTreeMap::from([(
            "claude-sonnet-4-5-20250929".to_string(),
            SONNET,
        )]));
        assert!(
            table
                .cost("claude-sonnet-4-5-20250929", usage(1, 1, 0))
                .is_some()
        );
        assert_eq!(table.cost("unknown-model", usage(1, 1, 0)), None);
    }
}
