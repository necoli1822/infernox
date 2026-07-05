//! Infernal/Easel type definitions - 1:1 port from easel.h and infernal.h
//!
//! Core type aliases matching the C implementation.

/// Digital sequence residue type
///
/// ESL_DSQ is a uint8_t in C (1 byte), used for digitized sequence representation.
/// Values 0-3 typically represent nucleotides (A=0, C=1, G=2, U/T=3).
/// Special sentinel values (251-255) are defined in easel::constants.
pub type EslDsq = u8;

/// Position type for sequence coordinates
///
/// esl_pos_t is int64_t in C (8 bytes), used for sequence positions.
/// Signed to allow for -1 sentinel values and coordinate arithmetic.
pub type EslPos = i64;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_esl_dsq_size() {
        // ESL_DSQ must be 1 byte (uint8_t)
        assert_eq!(size_of::<EslDsq>(), 1);
    }

    #[test]
    fn test_esl_pos_size() {
        // esl_pos_t must be 8 bytes (int64_t)
        assert_eq!(size_of::<EslPos>(), 8);
    }

    #[test]
    fn test_esl_dsq_range() {
        // ESL_DSQ can hold values 0-255
        let min: EslDsq = 0;
        let max: EslDsq = 255;
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn test_esl_pos_range() {
        // esl_pos_t can hold typical sequence lengths and negative sentinels
        let neg_sentinel: EslPos = -1;
        let large_pos: EslPos = 1_000_000_000_000;
        assert_eq!(neg_sentinel, -1);
        assert!(large_pos > 0);
    }
}
