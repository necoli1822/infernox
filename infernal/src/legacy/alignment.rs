//! Alignment structures for CM traceback output
//!
//! This module provides data structures for representing the alignment
//! between a sequence and a covariance model, generated from traceback
//! of the dynamic programming matrices.

use crate::parsetree::Parsetree;

/// Alignment mode for each parsetree node
///
/// The mode indicates how the alignment handles fragment boundaries
/// and is used in Stockholm-style output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Standard joint mode (J) - full alignment
    J,
    /// Left fragment mode (L) - alignment starts within model
    L,
    /// Right fragment mode (R) - alignment ends within model
    R,
}

impl AlignMode {
    /// Convert mode to character for output
    pub fn to_char(self) -> char {
        match self {
            AlignMode::J => 'J',
            AlignMode::L => 'L',
            AlignMode::R => 'R',
        }
    }
}

/// Alignment output with Stockholm-style information
///
/// This structure holds the complete alignment result, including:
/// - The alignment score
/// - The parsetree (traceback path through the model)
/// - Consensus sequence and structure strings
/// - Alignment coordinates in both consensus and sequence space
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Alignment score (log-odds, in bits)
    pub score: f32,

    /// Parsetree showing the path through the CM
    pub parsetree: Parsetree,

    /// Number of nodes in parsetree
    pub parsetree_size: usize,

    /// Whether this is a standard (non-truncated) alignment
    pub is_standard: bool,

    // =========================================================================
    // Consensus information
    // =========================================================================

    /// Length of consensus sequence
    pub consensus_len: i32,

    /// Consensus sequence string (aligned model residues)
    pub consensus_seq: String,

    /// Consensus structure string (secondary structure annotation)
    pub consensus_str: String,

    // =========================================================================
    // Alignment coordinates
    // =========================================================================

    /// Consensus start position (1-indexed)
    pub cfrom: i32,

    /// Consensus end position (1-indexed)
    pub cto: i32,

    /// Sequence start position (1-indexed)
    pub sqfrom: i32,

    /// Sequence end position (1-indexed)
    pub sqto: i32,
}

impl Alignment {
    /// Create a new alignment from score and parsetree
    ///
    /// # Arguments
    /// * `score` - The alignment score (log-odds, in bits)
    /// * `parsetree` - The parsetree showing the alignment path
    ///
    /// # Returns
    /// A new Alignment with the given score and parsetree. Other fields
    /// are initialized to default values and should be filled in by
    /// the alignment generation code.
    pub fn new(score: f32, parsetree: Parsetree) -> Self {
        let size = parsetree.n as usize;
        Alignment {
            score,
            parsetree,
            parsetree_size: size,
            is_standard: true,
            consensus_len: 0,
            consensus_seq: String::new(),
            consensus_str: String::new(),
            cfrom: 1,
            cto: 0,
            sqfrom: 1,
            sqto: 0,
        }
    }

    /// Get the alignment mode based on consensus coverage
    ///
    /// Detects truncated alignments by comparing the alignment's consensus
    /// coverage to the full model length:
    /// - J mode: Full alignment (cfrom == 1 and cto == consensus_len)
    /// - L mode: Left fragment (alignment starts at consensus 1 but ends early)
    /// - R mode: Right fragment (alignment ends at consensus_len but starts late)
    ///
    /// For ambiguous cases (both truncated on left and right), defaults to J.
    pub fn mode(&self) -> AlignMode {
        if self.is_standard {
            return AlignMode::J;
        }

        // Check for truncated alignments based on consensus coverage
        // A full alignment spans from consensus position 1 to consensus_len
        let is_left_truncated = self.cfrom > 1;
        let is_right_truncated = self.consensus_len > 0 && self.cto < self.consensus_len;

        match (is_left_truncated, is_right_truncated) {
            (false, true) => AlignMode::L,   // Right end missing = L fragment
            (true, false) => AlignMode::R,   // Left end missing = R fragment
            _ => AlignMode::J,               // Full alignment or ambiguous
        }
    }

    /// Set the alignment as a truncated (non-standard) alignment
    pub fn set_truncated(&mut self) {
        self.is_standard = false;
    }

    /// Check if this is a left-truncated alignment (missing right end)
    pub fn is_left_fragment(&self) -> bool {
        self.mode() == AlignMode::L
    }

    /// Check if this is a right-truncated alignment (missing left end)
    pub fn is_right_fragment(&self) -> bool {
        self.mode() == AlignMode::R
    }

    /// Get sequence length covered by this alignment
    pub fn seq_length(&self) -> i32 {
        if self.sqto >= self.sqfrom {
            self.sqto - self.sqfrom + 1
        } else {
            0
        }
    }

    /// Get consensus length covered by this alignment
    pub fn consensus_length(&self) -> i32 {
        if self.cto >= self.cfrom {
            self.cto - self.cfrom + 1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_mode_to_char() {
        assert_eq!(AlignMode::J.to_char(), 'J');
        assert_eq!(AlignMode::L.to_char(), 'L');
        assert_eq!(AlignMode::R.to_char(), 'R');
    }

    #[test]
    fn test_alignment_new() {
        let mut pt = Parsetree::new(10);
        pt.add_node(1, 76, 0, -1, -1, -1);
        pt.add_node(2, 75, 1, -1, -1, 0);

        let aln = Alignment::new(48.0963, pt);
        assert_eq!(aln.score, 48.0963);
        assert_eq!(aln.parsetree_size, 2);
        assert_eq!(aln.is_standard, true);
        assert_eq!(aln.mode(), AlignMode::J);
    }

    #[test]
    fn test_alignment_lengths() {
        let pt = Parsetree::new(10);
        let mut aln = Alignment::new(10.0, pt);

        // Set coordinates
        aln.sqfrom = 5;
        aln.sqto = 15;
        aln.cfrom = 1;
        aln.cto = 71;

        assert_eq!(aln.seq_length(), 11); // 15 - 5 + 1
        assert_eq!(aln.consensus_length(), 71); // 71 - 1 + 1
    }

    #[test]
    fn test_truncated_alignment_detection() {
        let pt = Parsetree::new(10);
        let mut aln = Alignment::new(10.0, pt);
        aln.consensus_len = 100;

        // Full alignment (J mode)
        aln.cfrom = 1;
        aln.cto = 100;
        aln.is_standard = true;
        assert_eq!(aln.mode(), AlignMode::J);
        assert!(!aln.is_left_fragment());
        assert!(!aln.is_right_fragment());

        // Left fragment (L mode): starts at 1, ends before consensus_len
        aln.cfrom = 1;
        aln.cto = 50;
        aln.is_standard = false;
        assert_eq!(aln.mode(), AlignMode::L);
        assert!(aln.is_left_fragment());
        assert!(!aln.is_right_fragment());

        // Right fragment (R mode): starts after 1, ends at consensus_len
        aln.cfrom = 30;
        aln.cto = 100;
        aln.is_standard = false;
        assert_eq!(aln.mode(), AlignMode::R);
        assert!(!aln.is_left_fragment());
        assert!(aln.is_right_fragment());

        // Both truncated - defaults to J
        aln.cfrom = 20;
        aln.cto = 80;
        aln.is_standard = false;
        assert_eq!(aln.mode(), AlignMode::J);
    }

    #[test]
    fn test_set_truncated() {
        let pt = Parsetree::new(10);
        let mut aln = Alignment::new(10.0, pt);

        assert!(aln.is_standard);
        aln.set_truncated();
        assert!(!aln.is_standard);
    }
}
