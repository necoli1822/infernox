//! Easel Alphabet - 1:1 port from esl_alphabet.c
//!
//! Digital sequence alphabet handling for RNA/DNA/Amino sequences.

use crate::constants::ESL_DSQ_ILLEGAL;

/// Alphabet type constants
pub const ESL_DNA: i32 = 0;
pub const ESL_RNA: i32 = 1;
pub const ESL_AMINO: i32 = 2;
pub const ESL_COINS: i32 = 3;
pub const ESL_DICE: i32 = 4;
pub const ESL_NONSTANDARD: i32 = 5;

/// RNA alphabet structure
///
/// K  = 4  (canonical bases: A, C, G, U)
/// Kp = 18 (total including degeneracy codes, gap, missing)
///
/// Symbol ordering (index): A(0), C(1), G(2), U(3), -(4), R(5), Y(6), M(7), K(8),
/// S(9), W(10), H(11), B(12), V(13), D(14), N(15), *(16), ~(17)
#[derive(Debug, Clone)]
pub struct EslAlphabet {
    /// Alphabet type: eslRNA=1
    pub type_: i32,
    /// Number of canonical residues (4 for RNA: A,C,G,U)
    pub K: i32,
    /// Total alphabet size including degeneracy codes, gap, missing (18 for RNA)
    pub Kp: i32,
    /// Input map: ASCII char -> digital code (0..Kp-1), or ESL_DSQ_ILLEGAL
    pub inmap: [u8; 128],
    /// Output symbols: digital code -> ASCII char
    pub sym: Vec<char>,
    /// Complement mapping: digital code -> complement's digital code
    pub complement: [u8; 18],
}

impl EslAlphabet {
    /// Create a standard RNA alphabet
    ///
    /// Returns an alphabet with:
    /// - type_ = 1 (eslRNA)
    /// - K = 4 (A, C, G, U)
    /// - Kp = 18 (includes degeneracy codes, gap, missing)
    /// - Standard IUPAC degeneracy codes
    pub fn rna() -> Self {
        let type_ = ESL_RNA;
        let K = 4;
        let Kp = 18;

        // Symbol string: ACGU-RYMKSWHBVDN*~
        // Index:         0123456789...
        let sym: Vec<char> = "ACGU-RYMKSWHBVDN*~".chars().collect();

        // Initialize inmap to ESL_DSQ_ILLEGAL (254)
        let mut inmap = [ESL_DSQ_ILLEGAL; 128];

        // Canonical bases (uppercase)
        inmap[b'A' as usize] = 0;
        inmap[b'C' as usize] = 1;
        inmap[b'G' as usize] = 2;
        inmap[b'U' as usize] = 3;
        inmap[b'T' as usize] = 3;  // T maps to U

        // Canonical bases (lowercase)
        inmap[b'a' as usize] = 0;
        inmap[b'c' as usize] = 1;
        inmap[b'g' as usize] = 2;
        inmap[b'u' as usize] = 3;
        inmap[b't' as usize] = 3;  // t maps to u

        // Gap characters
        inmap[b'-' as usize] = 4;
        inmap[b'_' as usize] = 4;
        inmap[b'.' as usize] = 4;

        // IUPAC degeneracy codes (uppercase)
        // R = A or G (purine)
        inmap[b'R' as usize] = 5;
        inmap[b'r' as usize] = 5;
        // Y = C or U (pyrimidine)
        inmap[b'Y' as usize] = 6;
        inmap[b'y' as usize] = 6;
        // M = A or C
        inmap[b'M' as usize] = 7;
        inmap[b'm' as usize] = 7;
        // K = G or U
        inmap[b'K' as usize] = 8;
        inmap[b'k' as usize] = 8;
        // S = G or C (strong)
        inmap[b'S' as usize] = 9;
        inmap[b's' as usize] = 9;
        // W = A or U (weak)
        inmap[b'W' as usize] = 10;
        inmap[b'w' as usize] = 10;
        // H = A or C or U (not G)
        inmap[b'H' as usize] = 11;
        inmap[b'h' as usize] = 11;
        // B = C or G or U (not A)
        inmap[b'B' as usize] = 12;
        inmap[b'b' as usize] = 12;
        // V = A or C or G (not U)
        inmap[b'V' as usize] = 13;
        inmap[b'v' as usize] = 13;
        // D = A or G or U (not C)
        inmap[b'D' as usize] = 14;
        inmap[b'd' as usize] = 14;
        // N = any base
        inmap[b'N' as usize] = 15;
        inmap[b'n' as usize] = 15;
        inmap[b'X' as usize] = 15;  // X also maps to N
        inmap[b'x' as usize] = 15;

        // Special codes
        // * = nonresidue/stop (index 16)
        inmap[b'*' as usize] = 16;
        // ~ = missing data (index 17)
        inmap[b'~' as usize] = 17;

        // I maps to A in Easel (inosine analog)
        inmap[b'I' as usize] = 0;
        inmap[b'i' as usize] = 0;

        // Complement mapping
        // A(0) <-> U(3), C(1) <-> G(2)
        // Gap(4) <-> Gap(4)
        // Degeneracy codes follow complement rules
        let complement: [u8; 18] = [
            3,   // 0: A -> U
            2,   // 1: C -> G
            1,   // 2: G -> C
            0,   // 3: U -> A
            4,   // 4: - -> - (gap)
            6,   // 5: R(A|G) -> Y(C|U)
            5,   // 6: Y(C|U) -> R(A|G)
            8,   // 7: M(A|C) -> K(G|U)
            7,   // 8: K(G|U) -> M(A|C)
            9,   // 9: S(G|C) -> S(G|C)
            10,  // 10: W(A|U) -> W(A|U)
            14,  // 11: H(A|C|U) -> D(A|G|U)
            13,  // 12: B(C|G|U) -> V(A|C|G)
            12,  // 13: V(A|C|G) -> B(C|G|U)
            11,  // 14: D(A|G|U) -> H(A|C|U)
            15,  // 15: N -> N
            16,  // 16: * -> *
            17,  // 17: ~ -> ~
        ];

        EslAlphabet {
            type_,
            K,
            Kp,
            inmap,
            sym,
            complement,
        }
    }

    /// Convert a text sequence to digital form
    ///
    /// Each character in the input is mapped to its digital code
    /// using the inmap table.
    ///
    /// # Arguments
    /// * `seq` - Input sequence string
    ///
    /// # Returns
    /// Vector of digital codes (0..Kp-1), or ESL_DSQ_ILLEGAL for invalid chars
    pub fn digitize(&self, seq: &str) -> Vec<u8> {
        seq.bytes()
            .map(|b| {
                if b < 128 {
                    self.inmap[b as usize]
                } else {
                    ESL_DSQ_ILLEGAL
                }
            })
            .collect()
    }

    /// Convert a digital sequence back to text form
    ///
    /// Each digital code is mapped to its symbol character.
    ///
    /// # Arguments
    /// * `dsq` - Digital sequence (slice of codes)
    ///
    /// # Returns
    /// Text representation of the sequence
    pub fn textize(&self, dsq: &[u8]) -> String {
        dsq.iter()
            .map(|&code| {
                if (code as i32) < self.Kp {
                    self.sym[code as usize]
                } else {
                    '?' // Unknown/illegal code
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rna_basic() {
        let abc = EslAlphabet::rna();
        assert_eq!(abc.type_, ESL_RNA);
        assert_eq!(abc.K, 4);
        assert_eq!(abc.Kp, 18);
    }

    #[test]
    fn test_digitize_canonical() {
        let abc = EslAlphabet::rna();
        let dsq = abc.digitize("ACGU");
        assert_eq!(dsq, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_digitize_lowercase() {
        let abc = EslAlphabet::rna();
        let dsq = abc.digitize("acgu");
        assert_eq!(dsq, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_textize() {
        let abc = EslAlphabet::rna();
        let text = abc.textize(&[0, 1, 2, 3]);
        assert_eq!(text, "ACGU");
    }
}
