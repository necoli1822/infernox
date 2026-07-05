//! CP9 DP Matrix - 1:1 port from cp9_mx.h
//!
//! Dynamic programming matrices for CP9 HMM algorithms including
//! Forward, Backward, and Viterbi.

use crate::cp9::CP9_IMPOSSIBLE;

// =============================================================================
// CP9MX Structure
// =============================================================================

/// CP9 Dynamic Programming Matrix
///
/// Holds the DP matrix for Forward, Backward, or Viterbi algorithms.
/// The matrix is (L+1) x (M+1) for each state type.
#[derive(Debug, Clone)]
pub struct CP9MX {
    /// HMM model length (number of match nodes)
    pub m: i32,

    /// Sequence length
    pub l: i32,

    /// Match state scores: mmx[i][k] = score at position i, match node k
    /// Dimensions: [0..L][0..M]
    pub mmx: Vec<Vec<i32>>,

    /// Insert state scores: imx[i][k] = score at position i, insert node k
    /// Dimensions: [0..L][0..M]
    pub imx: Vec<Vec<i32>>,

    /// Delete state scores: dmx[i][k] = score at position i, delete node k
    /// Dimensions: [0..L][0..M]
    pub dmx: Vec<Vec<i32>>,

    /// EL state scores (for local ends): elmx[i] = score at position i in EL
    /// Dimensions: [0..L]
    pub elmx: Vec<i32>,

    /// Begin state scores: bmx[i] = score at position i in B state
    /// Dimensions: [0..L]
    pub bmx: Vec<i32>,
}

impl CP9MX {
    /// Create a new CP9MX with given HMM length and sequence length
    pub fn new(m: i32, l: i32) -> Self {
        let l_plus_1 = (l + 1) as usize;
        let m_plus_1 = (m + 1) as usize;

        CP9MX {
            m,
            l,
            mmx: vec![vec![CP9_IMPOSSIBLE; m_plus_1]; l_plus_1],
            imx: vec![vec![CP9_IMPOSSIBLE; m_plus_1]; l_plus_1],
            dmx: vec![vec![CP9_IMPOSSIBLE; m_plus_1]; l_plus_1],
            elmx: vec![CP9_IMPOSSIBLE; l_plus_1],
            bmx: vec![CP9_IMPOSSIBLE; l_plus_1],
        }
    }

    /// Resize the matrix for a new sequence length
    pub fn resize(&mut self, l: i32) {
        let l_plus_1 = (l + 1) as usize;
        let m_plus_1 = (self.m + 1) as usize;

        if l_plus_1 > self.mmx.len() {
            self.mmx.resize(l_plus_1, vec![CP9_IMPOSSIBLE; m_plus_1]);
            self.imx.resize(l_plus_1, vec![CP9_IMPOSSIBLE; m_plus_1]);
            self.dmx.resize(l_plus_1, vec![CP9_IMPOSSIBLE; m_plus_1]);
            self.elmx.resize(l_plus_1, CP9_IMPOSSIBLE);
            self.bmx.resize(l_plus_1, CP9_IMPOSSIBLE);
        }

        self.l = l;
    }

    /// Clear/initialize the matrix with impossible scores
    pub fn clear(&mut self) {
        let m_plus_1 = (self.m + 1) as usize;
        let l_plus_1 = (self.l + 1) as usize;

        for i in 0..l_plus_1 {
            for k in 0..m_plus_1 {
                self.mmx[i][k] = CP9_IMPOSSIBLE;
                self.imx[i][k] = CP9_IMPOSSIBLE;
                self.dmx[i][k] = CP9_IMPOSSIBLE;
            }
            self.elmx[i] = CP9_IMPOSSIBLE;
            self.bmx[i] = CP9_IMPOSSIBLE;
        }
    }

    /// Get match score at position i, node k
    pub fn get_m(&self, i: usize, k: usize) -> i32 {
        self.mmx[i][k]
    }

    /// Get insert score at position i, node k
    pub fn get_i(&self, i: usize, k: usize) -> i32 {
        self.imx[i][k]
    }

    /// Get delete score at position i, node k
    pub fn get_d(&self, i: usize, k: usize) -> i32 {
        self.dmx[i][k]
    }

    /// Set match score at position i, node k
    pub fn set_m(&mut self, i: usize, k: usize, score: i32) {
        self.mmx[i][k] = score;
    }

    /// Set insert score at position i, node k
    pub fn set_i(&mut self, i: usize, k: usize, score: i32) {
        self.imx[i][k] = score;
    }

    /// Set delete score at position i, node k
    pub fn set_d(&mut self, i: usize, k: usize, score: i32) {
        self.dmx[i][k] = score;
    }

    /// Get the number of cells in the matrix
    pub fn ncells(&self) -> usize {
        let l_plus_1 = (self.l + 1) as usize;
        let m_plus_1 = (self.m + 1) as usize;
        3 * l_plus_1 * m_plus_1 + 2 * l_plus_1 // M, I, D matrices + EL, B vectors
    }

    /// Get approximate memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.ncells() * std::mem::size_of::<i32>()
    }

    /// Get approximate memory usage in megabytes
    pub fn size_mb(&self) -> f64 {
        self.size_bytes() as f64 / (1024.0 * 1024.0)
    }
}

// =============================================================================
// CP9 Shadow Matrix for Traceback
// =============================================================================

/// Traceback pointer values for CP9 algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CP9Trace {
    /// Invalid/unset
    None = 0,
    /// Came from M state
    M = 1,
    /// Came from I state
    I = 2,
    /// Came from D state
    D = 3,
    /// Came from B (begin) state
    B = 4,
    /// Came from E (end) state
    E = 5,
    /// Came from EL (local end) state
    EL = 6,
    /// Came from N state
    N = 7,
    /// Came from C state
    C = 8,
    /// Came from J state
    J = 9,
}

/// CP9 Shadow Matrix for traceback
#[derive(Debug, Clone)]
pub struct CP9Shadow {
    /// HMM model length
    pub m: i32,

    /// Sequence length
    pub l: i32,

    /// Match state traceback: mtb[i][k] = which state led to M[i][k]
    pub mtb: Vec<Vec<CP9Trace>>,

    /// Insert state traceback: itb[i][k] = which state led to I[i][k]
    pub itb: Vec<Vec<CP9Trace>>,

    /// Delete state traceback: dtb[i][k] = which state led to D[i][k]
    pub dtb: Vec<Vec<CP9Trace>>,

    /// EL state traceback
    pub eltb: Vec<CP9Trace>,
}

impl CP9Shadow {
    /// Create a new shadow matrix
    pub fn new(m: i32, l: i32) -> Self {
        let l_plus_1 = (l + 1) as usize;
        let m_plus_1 = (m + 1) as usize;

        CP9Shadow {
            m,
            l,
            mtb: vec![vec![CP9Trace::None; m_plus_1]; l_plus_1],
            itb: vec![vec![CP9Trace::None; m_plus_1]; l_plus_1],
            dtb: vec![vec![CP9Trace::None; m_plus_1]; l_plus_1],
            eltb: vec![CP9Trace::None; l_plus_1],
        }
    }

    /// Clear the shadow matrix
    pub fn clear(&mut self) {
        let l_plus_1 = (self.l + 1) as usize;
        let m_plus_1 = (self.m + 1) as usize;

        for i in 0..l_plus_1 {
            for k in 0..m_plus_1 {
                self.mtb[i][k] = CP9Trace::None;
                self.itb[i][k] = CP9Trace::None;
                self.dtb[i][k] = CP9Trace::None;
            }
            self.eltb[i] = CP9Trace::None;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp9mx_new() {
        let mx = CP9MX::new(71, 76);
        assert_eq!(mx.m, 71);
        assert_eq!(mx.l, 76);
        assert_eq!(mx.mmx.len(), 77); // 0..76
        assert_eq!(mx.mmx[0].len(), 72); // 0..71
        assert_eq!(mx.imx.len(), 77);
        assert_eq!(mx.dmx.len(), 77);
        assert_eq!(mx.elmx.len(), 77);
        assert_eq!(mx.bmx.len(), 77);
    }

    #[test]
    fn test_cp9mx_initial_values() {
        let mx = CP9MX::new(10, 20);
        // All cells should be initialized to IMPOSSIBLE
        assert_eq!(mx.get_m(5, 5), CP9_IMPOSSIBLE);
        assert_eq!(mx.get_i(5, 5), CP9_IMPOSSIBLE);
        assert_eq!(mx.get_d(5, 5), CP9_IMPOSSIBLE);
    }

    #[test]
    fn test_cp9mx_set_get() {
        let mut mx = CP9MX::new(10, 20);

        mx.set_m(5, 5, 1000);
        mx.set_i(5, 5, 2000);
        mx.set_d(5, 5, 3000);

        assert_eq!(mx.get_m(5, 5), 1000);
        assert_eq!(mx.get_i(5, 5), 2000);
        assert_eq!(mx.get_d(5, 5), 3000);
    }

    #[test]
    fn test_cp9mx_resize() {
        let mut mx = CP9MX::new(10, 20);
        assert_eq!(mx.l, 20);

        mx.resize(50);
        assert_eq!(mx.l, 50);
        assert_eq!(mx.mmx.len(), 51);
    }

    #[test]
    fn test_cp9mx_clear() {
        let mut mx = CP9MX::new(10, 20);
        mx.set_m(5, 5, 1000);
        mx.clear();
        assert_eq!(mx.get_m(5, 5), CP9_IMPOSSIBLE);
    }

    #[test]
    fn test_cp9mx_ncells() {
        let mx = CP9MX::new(71, 76);
        let l_plus_1 = 77usize;
        let m_plus_1 = 72usize;
        let expected = 3 * l_plus_1 * m_plus_1 + 2 * l_plus_1;
        assert_eq!(mx.ncells(), expected);
    }

    #[test]
    fn test_cp9shadow_new() {
        let shadow = CP9Shadow::new(71, 76);
        assert_eq!(shadow.m, 71);
        assert_eq!(shadow.l, 76);
        assert_eq!(shadow.mtb.len(), 77);
        assert_eq!(shadow.mtb[0].len(), 72);
    }

    #[test]
    fn test_cp9trace_values() {
        assert_eq!(CP9Trace::None as u8, 0);
        assert_eq!(CP9Trace::M as u8, 1);
        assert_eq!(CP9Trace::I as u8, 2);
        assert_eq!(CP9Trace::D as u8, 3);
    }
}
