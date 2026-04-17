//! Shared usage-snapshot type reported by providers that can introspect their
//! current subscription consumption. Consumed by the usage-aware router.

use serde::{Deserialize, Serialize};

/// A provider's view of its current subscription usage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageSnapshot {
    /// Fraction of the current billing period consumed, in `[0.0, 1.0]`.
    /// `None` means the provider cannot determine this (e.g. no usage
    /// endpoint, or usage is denominated in an incomparable unit).
    pub fraction_used: Option<f64>,

    /// Short human-readable window label (e.g. `"monthly"`, `"daily"`).
    pub window: String,

    /// Free-form label shown to the user (e.g. `"1.2M / 5M tokens"`).
    pub label: Option<String>,
}

impl UsageSnapshot {
    #[allow(dead_code)]
    pub fn unknown(window: impl Into<String>) -> Self {
        Self {
            fraction_used: None,
            window: window.into(),
            label: None,
        }
    }
}
