//! CP9 Bands - 1:1 port from cp9_bands.h
//!
//! CP9 bands are computed from HMM forward/backward algorithms and used
//! to constrain CM dynamic programming, dramatically reducing computation.

// =============================================================================
// CP9Bands Structure
// =============================================================================

/// CP9 Bands for constrained CM alignment
///
/// Contains HMM-derived bands that constrain which cells of the CM
/// DP matrix need to be computed.
#[derive(Debug, Clone)]
pub struct CP9Bands {
    // =========================================================================
    // Model dimensions
    // =========================================================================
    /// HMM model length (number of match nodes)
    pub hmm_m: i32,

    /// CM model size (number of states)
    pub cm_m: i32,

    /// CM consensus length
    pub cm_clen: i32,

    // =========================================================================
    // Posterior probability thresholds
    // =========================================================================
    /// Threshold for defining bands (typically 0.01)
    pub thresh1: f64,

    /// Upper threshold (typically 0.98)
    pub thresh2: f64,

    /// Posterior probability tail mass (typically 0.0001)
    pub tau: f64,

    // =========================================================================
    // HMM position bands (for each sequence position i)
    // =========================================================================
    /// Minimum HMM node at position i (match state)
    pub pn_min_m: Vec<i32>,

    /// Maximum HMM node at position i (match state)
    pub pn_max_m: Vec<i32>,

    /// Minimum HMM node at position i (insert state)
    pub pn_min_i: Vec<i32>,

    /// Maximum HMM node at position i (insert state)
    pub pn_max_i: Vec<i32>,

    /// Minimum HMM node at position i (delete state)
    pub pn_min_d: Vec<i32>,

    /// Maximum HMM node at position i (delete state)
    pub pn_max_d: Vec<i32>,

    // =========================================================================
    // Sequence position bands (for each HMM node k)
    // =========================================================================
    /// Minimum sequence position for HMM node k (match state)
    pub imin: Vec<i32>,

    /// Maximum sequence position for HMM node k (match state)
    pub imax: Vec<i32>,

    // =========================================================================
    // CM state bands (for each CM state v)
    // =========================================================================
    /// Minimum j for CM state v
    pub jmin: Vec<i32>,

    /// Maximum j for CM state v
    pub jmax: Vec<i32>,

    /// For each state v: hdmin[v][jp] is min d at j = jmin[v] + jp
    pub hdmin: Vec<Vec<i32>>,

    /// For each state v: hdmax[v][jp] is max d at j = jmin[v] + jp
    pub hdmax: Vec<Vec<i32>>,

    // =========================================================================
    // Safe band limits
    // =========================================================================
    /// Safe minimum j for each state (accounting for state type)
    pub safe_hdmin: Vec<i32>,

    /// Safe maximum j for each state
    pub safe_hdmax: Vec<i32>,

    // =========================================================================
    // Status flags
    // =========================================================================
    /// Bands have been computed and are valid
    pub valid: bool,

    /// Sequence length for which bands are valid
    pub l: i32,
}

impl CP9Bands {
    /// Create new CP9Bands with given dimensions
    pub fn new(hmm_m: i32, cm_m: i32) -> Self {
        let hmm_m_plus_1 = (hmm_m + 1) as usize;
        let cm_m_usize = cm_m as usize;

        CP9Bands {
            hmm_m,
            cm_m,
            cm_clen: 0,
            thresh1: 0.01,
            thresh2: 0.98,
            tau: 0.0001,
            pn_min_m: Vec::new(),
            pn_max_m: Vec::new(),
            pn_min_i: Vec::new(),
            pn_max_i: Vec::new(),
            pn_min_d: Vec::new(),
            pn_max_d: Vec::new(),
            imin: vec![0; hmm_m_plus_1],
            imax: vec![0; hmm_m_plus_1],
            jmin: vec![0; cm_m_usize],
            jmax: vec![0; cm_m_usize],
            hdmin: vec![Vec::new(); cm_m_usize],
            hdmax: vec![Vec::new(); cm_m_usize],
            safe_hdmin: vec![0; cm_m_usize],
            safe_hdmax: vec![0; cm_m_usize],
            valid: false,
            l: 0,
        }
    }

    /// Allocate position bands for sequence length L
    pub fn alloc_for_seq(&mut self, l: i32) {
        let l_plus_1 = (l + 1) as usize;

        self.pn_min_m = vec![-1; l_plus_1];
        self.pn_max_m = vec![-1; l_plus_1];
        self.pn_min_i = vec![-1; l_plus_1];
        self.pn_max_i = vec![-1; l_plus_1];
        self.pn_min_d = vec![-1; l_plus_1];
        self.pn_max_d = vec![-1; l_plus_1];

        self.l = l;
    }

    /// Set j band for CM state v
    pub fn set_j_band(&mut self, v: usize, jmin: i32, jmax: i32) {
        self.jmin[v] = jmin;
        self.jmax[v] = jmax;

        // Allocate hdmin/hdmax arrays for this state
        let width = (jmax - jmin + 1) as usize;
        self.hdmin[v] = vec![0; width];
        self.hdmax[v] = vec![0; width];
    }

    /// Get minimum d at state v, position j
    pub fn get_hdmin(&self, v: usize, j: i32) -> i32 {
        let jp = (j - self.jmin[v]) as usize;
        if jp < self.hdmin[v].len() {
            self.hdmin[v][jp]
        } else {
            0
        }
    }

    /// Get maximum d at state v, position j
    pub fn get_hdmax(&self, v: usize, j: i32) -> i32 {
        let jp = (j - self.jmin[v]) as usize;
        if jp < self.hdmax[v].len() {
            self.hdmax[v][jp]
        } else {
            0
        }
    }

    /// Set d band at state v, position j
    pub fn set_hd_band(&mut self, v: usize, j: i32, dmin: i32, dmax: i32) {
        let jp = (j - self.jmin[v]) as usize;
        if jp < self.hdmin[v].len() {
            self.hdmin[v][jp] = dmin;
            self.hdmax[v][jp] = dmax;
        }
    }

    /// Check if bands are valid for the given sequence length
    pub fn is_valid_for(&self, l: i32) -> bool {
        self.valid && self.l == l
    }

    /// Invalidate the bands
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Get the number of cells that would be computed with these bands
    /// for a given sequence length
    pub fn ncells(&self, l: i32) -> usize {
        if !self.valid || self.l != l {
            return 0;
        }

        let mut total = 0usize;
        for v in 0..self.cm_m as usize {
            let jmin = self.jmin[v];
            let jmax = self.jmax[v];
            for j in jmin..=jmax {
                let dmin = self.get_hdmin(v, j);
                let dmax = self.get_hdmax(v, j);
                if dmax >= dmin {
                    total += (dmax - dmin + 1) as usize;
                }
            }
        }
        total
    }
}

// =============================================================================
// QDB (Query-Dependent Bands)
// =============================================================================

/// Query-Dependent Bands
///
/// These are pre-computed CM bands that don't depend on the sequence,
/// only on the model and a beta parameter.
#[derive(Debug, Clone)]
pub struct QDBBands {
    /// CM model size
    pub cm_m: i32,

    /// Beta parameter used to compute these bands
    pub beta: f64,

    /// Minimum d for each CM state
    pub dmin: Vec<i32>,

    /// Maximum d for each CM state
    pub dmax: Vec<i32>,
}

impl QDBBands {
    /// Create new QDB bands with given CM size
    pub fn new(cm_m: i32) -> Self {
        let cm_m_usize = cm_m as usize;

        QDBBands {
            cm_m,
            beta: 0.0,
            dmin: vec![0; cm_m_usize],
            dmax: vec![0; cm_m_usize],
        }
    }

    /// Create QDB bands from dmin/dmax arrays
    pub fn from_bands(cm_m: i32, beta: f64, dmin: Vec<i32>, dmax: Vec<i32>) -> Self {
        QDBBands {
            cm_m,
            beta,
            dmin,
            dmax,
        }
    }

    /// Get minimum d for state v
    pub fn get_dmin(&self, v: usize) -> i32 {
        self.dmin[v]
    }

    /// Get maximum d for state v
    pub fn get_dmax(&self, v: usize) -> i32 {
        self.dmax[v]
    }

    /// Set d band for state v
    pub fn set_band(&mut self, v: usize, dmin: i32, dmax: i32) {
        self.dmin[v] = dmin;
        self.dmax[v] = dmax;
    }
}

// =============================================================================
// DP Matrix Size Calculation
// =============================================================================

/// Calculate the number of cells in a CM DP matrix for a given sequence length
/// using QDB bands
pub fn cm_mx_ncells(cm_m: i32, l: i32, dmin: &[i32], dmax: &[i32]) -> usize {
    let mut ncells = 0usize;

    for v in 0..cm_m as usize {
        let d_min = dmin[v].max(0);
        let d_max = dmax[v].min(l);

        if d_max >= d_min {
            // For each j from 0 to L, count cells where d is in band
            // This is an approximation; actual count depends on state type
            ncells += ((l + 1) * (d_max - d_min + 1)) as usize;
        }
    }

    ncells
}

/// Calculate approximate memory usage in MB for a CM DP matrix
pub fn cm_mx_size_mb(cm_m: i32, l: i32, dmin: &[i32], dmax: &[i32]) -> f64 {
    let ncells = cm_mx_ncells(cm_m, l, dmin, dmax);
    // Each cell is a 32-bit integer
    (ncells * 4) as f64 / (1024.0 * 1024.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp9bands_new() {
        let bands = CP9Bands::new(71, 227);
        assert_eq!(bands.hmm_m, 71);
        assert_eq!(bands.cm_m, 227);
        assert_eq!(bands.imin.len(), 72); // 0..M
        assert_eq!(bands.jmin.len(), 227);
        assert!(!bands.valid);
    }

    #[test]
    fn test_cp9bands_alloc_for_seq() {
        let mut bands = CP9Bands::new(71, 227);
        bands.alloc_for_seq(76);

        assert_eq!(bands.l, 76);
        assert_eq!(bands.pn_min_m.len(), 77); // 0..L
        assert_eq!(bands.pn_max_m.len(), 77);
    }

    #[test]
    fn test_cp9bands_j_band() {
        let mut bands = CP9Bands::new(71, 227);
        bands.set_j_band(10, 5, 15);

        assert_eq!(bands.jmin[10], 5);
        assert_eq!(bands.jmax[10], 15);
        assert_eq!(bands.hdmin[10].len(), 11); // 15 - 5 + 1
    }

    #[test]
    fn test_cp9bands_hd_band() {
        let mut bands = CP9Bands::new(71, 227);
        bands.set_j_band(10, 5, 15);
        bands.set_hd_band(10, 7, 3, 10);

        assert_eq!(bands.get_hdmin(10, 7), 3);
        assert_eq!(bands.get_hdmax(10, 7), 10);
    }

    #[test]
    fn test_qdbbands_new() {
        let qdb = QDBBands::new(227);
        assert_eq!(qdb.cm_m, 227);
        assert_eq!(qdb.dmin.len(), 227);
        assert_eq!(qdb.dmax.len(), 227);
    }

    #[test]
    fn test_qdbbands_set_get() {
        let mut qdb = QDBBands::new(227);
        qdb.set_band(10, 5, 50);

        assert_eq!(qdb.get_dmin(10), 5);
        assert_eq!(qdb.get_dmax(10), 50);
    }

    #[test]
    fn test_qdbbands_from_bands() {
        let dmin = vec![0, 1, 2];
        let dmax = vec![10, 20, 30];
        let qdb = QDBBands::from_bands(3, 1e-7, dmin, dmax);

        assert_eq!(qdb.cm_m, 3);
        assert_eq!(qdb.beta, 1e-7);
        assert_eq!(qdb.get_dmin(1), 1);
        assert_eq!(qdb.get_dmax(2), 30);
    }

    #[test]
    fn test_cp9bands_thresholds() {
        let bands = CP9Bands::new(71, 227);
        assert!((bands.thresh1 - 0.01).abs() < 1e-10);
        assert!((bands.thresh2 - 0.98).abs() < 1e-10);
        assert!((bands.tau - 0.0001).abs() < 1e-10);
    }
}
