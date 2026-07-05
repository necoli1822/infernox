//! Truncated alignment support
//!
//! Handles 5' and 3' truncated hits where the sequence doesn't fully
//! match the CM from start to end.

use crate::cm::CM;
use crate::constants::*;

/// Truncation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncMode {
    /// No truncation (full alignment)
    None,
    /// 5' truncated (missing start)
    Trunc5,
    /// 3' truncated (missing end)
    Trunc3,
    /// Both ends truncated
    TruncBoth,
    /// Any truncation allowed
    TruncAny,
}

impl TruncMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "no" => Some(TruncMode::None),
            "5" | "5'" | "trunc5" => Some(TruncMode::Trunc5),
            "3" | "3'" | "trunc3" => Some(TruncMode::Trunc3),
            "both" | "trunc-both" => Some(TruncMode::TruncBoth),
            "any" | "trunc-any" => Some(TruncMode::TruncAny),
            _ => None,
        }
    }

    pub fn allows_5_trunc(&self) -> bool {
        matches!(self, TruncMode::Trunc5 | TruncMode::TruncBoth | TruncMode::TruncAny)
    }

    pub fn allows_3_trunc(&self) -> bool {
        matches!(self, TruncMode::Trunc3 | TruncMode::TruncBoth | TruncMode::TruncAny)
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            TruncMode::None => "no",
            TruncMode::Trunc5 => "5'",
            TruncMode::Trunc3 => "3'",
            TruncMode::TruncBoth => "5'&3'",
            TruncMode::TruncAny => "any",
        }
    }
}

/// Truncated hit information
#[derive(Debug, Clone)]
pub struct TruncatedHit {
    /// Truncation mode detected
    pub mode: TruncMode,
    /// Start position in model (1 = full start)
    pub model_start: i32,
    /// End position in model (clen = full end)
    pub model_end: i32,
    /// Penalty applied for truncation (in bits)
    pub trunc_penalty: f32,
}

impl TruncatedHit {
    pub fn new_full(clen: i32) -> Self {
        TruncatedHit {
            mode: TruncMode::None,
            model_start: 1,
            model_end: clen,
            trunc_penalty: 0.0,
        }
    }

    pub fn is_truncated(&self) -> bool {
        self.mode != TruncMode::None
    }
}

/// Detect truncation in an alignment based on model coverage
pub fn detect_truncation(
    cm: &CM,
    model_start: i32,
    model_end: i32,
    trunc_penalty_per_pos: f32,
) -> TruncatedHit {
    let clen = cm.clen;

    let is_5_trunc = model_start > 1;
    let is_3_trunc = model_end < clen;

    let mode = match (is_5_trunc, is_3_trunc) {
        (false, false) => TruncMode::None,
        (true, false) => TruncMode::Trunc5,
        (false, true) => TruncMode::Trunc3,
        (true, true) => TruncMode::TruncBoth,
    };

    // Calculate penalty
    let missing_5 = (model_start - 1).max(0) as f32;
    let missing_3 = (clen - model_end).max(0) as f32;
    let trunc_penalty = (missing_5 + missing_3) * trunc_penalty_per_pos;

    TruncatedHit {
        mode,
        model_start,
        model_end,
        trunc_penalty,
    }
}

/// Calculate local entry probability for truncated alignment
///
/// For 5' truncation, we can enter at internal nodes
pub fn calculate_local_entry_prob(
    cm: &CM,
    entry_node: i32,
    mode: TruncMode,
) -> f32 {
    if !mode.allows_5_trunc() {
        return 0.0; // Must enter at root
    }

    // Uniform local entry probability
    let num_internal_nodes = (cm.nodes - 2).max(1) as f32;
    (1.0 / num_internal_nodes).ln()
}

/// Calculate local exit probability for truncated alignment
///
/// For 3' truncation, we can exit at internal nodes
pub fn calculate_local_exit_prob(
    cm: &CM,
    exit_node: i32,
    mode: TruncMode,
) -> f32 {
    if !mode.allows_3_trunc() {
        return 0.0; // Must exit at end
    }

    // Uniform local exit probability
    let num_internal_nodes = (cm.nodes - 2).max(1) as f32;
    (1.0 / num_internal_nodes).ln()
}

/// Adjust score for truncation penalty
pub fn apply_trunc_penalty(score: f32, trunc_hit: &TruncatedHit) -> f32 {
    score - trunc_hit.trunc_penalty
}

/// Check if truncation mode is compatible with alignment boundaries
pub fn is_valid_truncation(
    allowed_mode: TruncMode,
    detected_mode: TruncMode,
) -> bool {
    match allowed_mode {
        TruncMode::None => detected_mode == TruncMode::None,
        TruncMode::TruncAny => true,
        TruncMode::TruncBoth => true,
        TruncMode::Trunc5 => detected_mode == TruncMode::Trunc5 || detected_mode == TruncMode::None,
        TruncMode::Trunc3 => detected_mode == TruncMode::Trunc3 || detected_mode == TruncMode::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trunc_mode_from_str() {
        assert_eq!(TruncMode::from_str("none"), Some(TruncMode::None));
        assert_eq!(TruncMode::from_str("5'"), Some(TruncMode::Trunc5));
        assert_eq!(TruncMode::from_str("3'"), Some(TruncMode::Trunc3));
        assert_eq!(TruncMode::from_str("both"), Some(TruncMode::TruncBoth));
        assert_eq!(TruncMode::from_str("any"), Some(TruncMode::TruncAny));
    }

    #[test]
    fn test_trunc_mode_allows() {
        assert!(!TruncMode::None.allows_5_trunc());
        assert!(TruncMode::Trunc5.allows_5_trunc());
        assert!(!TruncMode::Trunc3.allows_5_trunc());
        assert!(TruncMode::TruncBoth.allows_5_trunc());
        assert!(TruncMode::TruncBoth.allows_3_trunc());
    }

    #[test]
    fn test_truncated_hit_full() {
        let hit = TruncatedHit::new_full(72);
        assert!(!hit.is_truncated());
        assert_eq!(hit.model_start, 1);
        assert_eq!(hit.model_end, 72);
        assert_eq!(hit.trunc_penalty, 0.0);
    }
}
