//! E-value computation for CM hits
//!
//! This module implements E-value computation based on exponential tail
//! distributions for covariance model search hits.

/// Exponential tail parameters for E-value computation
#[derive(Debug, Clone)]
pub struct ExpParams {
    pub lambda: f64,
    pub mu: f64,           // This is mu_extrap (extrapolated mu)
    pub dbsize: f64,       // Calibration database size
    pub nrandhits: i32,    // Number of random hits in calibration
    // C: ExpInfo_t also stores mu_orig and tailp (structs.h). Retained so the
    // CM file writer (cm_file_WriteASCII) can reproduce the ECMxx lines exactly.
    pub mu_orig: f64,      // original (non-extrapolated) mu
    pub tailp: f64,        // tail probability used in the fit
}

impl Default for ExpParams {
    fn default() -> Self {
        ExpParams {
            lambda: 0.693,  // ln(2) ~= 0.693
            mu: 20.0,
            dbsize: 1_000_000.0,
            nrandhits: 1,   // Calibrated by default
            mu_orig: 0.0,
            tailp: 0.0,
        }
    }
}

impl ExpParams {
    /// Convert bit score to E-value using current database size
    /// Formula: E = cur_eff_dbsize * exp(-lambda * (score - mu))
    /// where cur_eff_dbsize = (current_Z / calibration_dbsize) * nrandhits
    pub fn score_to_evalue_with_z(&self, score: f64, current_z: f64) -> f64 {
        let cur_eff_dbsize = if self.nrandhits > 0 && self.dbsize > 0.0 {
            (current_z / self.dbsize) * (self.nrandhits as f64)
        } else {
            // Uncalibrated: use current_z directly
            current_z
        };

        // esl_exp_surv: P(X > x) = exp(-lambda * (x - mu)) for x >= mu, else 1.0
        let surv = if score < self.mu {
            1.0
        } else {
            (-self.lambda * (score - self.mu)).exp()
        };

        surv * cur_eff_dbsize
    }

    /// Legacy method for backward compatibility (uses calibration dbsize directly)
    pub fn score_to_evalue(&self, score: f64) -> f64 {
        self.dbsize * (-self.lambda * (score - self.mu)).exp()
    }

    /// Convert E-value to bit score (using calibration dbsize)
    /// score = mu - ln(E / dbsize) / lambda
    pub fn evalue_to_score(&self, evalue: f64) -> f64 {
        if evalue <= 0.0 {
            return f64::INFINITY;
        }
        self.mu - (evalue / self.dbsize).ln() / self.lambda
    }

    /// Convert E-value to bit score cutoff using current database size
    /// This matches C Infernal's E2ScoreGivenExpInfo() function:
    /// score = mu_extrap + (log(E / cur_eff_dbsize) / (-lambda))
    ///
    /// This is the inverse of score_to_evalue_with_z()
    pub fn evalue_to_score_with_z(&self, evalue: f64, current_z: f64) -> f64 {
        if evalue <= 0.0 {
            return f64::INFINITY;
        }

        // Calculate cur_eff_dbsize the same way as score_to_evalue_with_z
        let cur_eff_dbsize = if self.nrandhits > 0 && self.dbsize > 0.0 {
            (current_z / self.dbsize) * (self.nrandhits as f64)
        } else {
            current_z
        };

        if cur_eff_dbsize <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // C formula: sc = mu_extrap + (log(E/cur_eff_dbsize) / (-lambda))
        self.mu + (evalue / cur_eff_dbsize).ln() / (-self.lambda)
    }

    /// Check if parameters are calibrated
    pub fn is_calibrated(&self) -> bool {
        self.lambda > 0.0 && self.mu.is_finite() && self.dbsize > 0.0 && self.nrandhits > 0
    }
}

/// P7 HMM E-value parameters (for filtering)
#[derive(Debug, Clone)]
pub struct P7ExpParams {
    pub lmmu: f64,      // local MSV mu
    pub lmlambda: f64,  // local MSV lambda
    pub lvmu: f64,      // local Vit mu
    pub lvlambda: f64,  // local Vit lambda
    pub lftau: f64,     // local Fwd tau
    pub lflambda: f64,  // local Fwd lambda
    pub gfmu: f64,      // glocal Fwd mu
    pub gflambda: f64,  // glocal Fwd lambda
}

impl Default for P7ExpParams {
    fn default() -> Self {
        P7ExpParams {
            lmmu: Self::UNCALIBRATED,
            lmlambda: Self::UNCALIBRATED,
            lvmu: Self::UNCALIBRATED,
            lvlambda: Self::UNCALIBRATED,
            lftau: Self::UNCALIBRATED,
            lflambda: Self::UNCALIBRATED,
            gfmu: Self::UNCALIBRATED,
            gflambda: Self::UNCALIBRATED,
        }
    }
}

impl P7ExpParams {
    const UNCALIBRATED: f64 = -99999.0;

    pub fn is_calibrated(&self) -> bool {
        self.lmlambda != Self::UNCALIBRATED
    }
}

/// Calculate nucleotide composition bias score
///
/// This implements a simplified null3-style bias correction.
/// Returns the bias in bits (positive = biased, 0 = unbiased)
pub fn calculate_bias_score(dsq: &[u8], l: i32) -> f32 {
    if l <= 0 {
        return 0.0;
    }

    // Count nucleotide frequencies (dsq uses 0,1,2,3 for A,C,G,U)
    let mut counts = [0u32; 4];
    let mut total = 0u32;

    for i in 1..=(l as usize) {
        if i < dsq.len() {
            let res = dsq[i];
            if res < 4 {
                counts[res as usize] += 1;
                total += 1;
            }
        }
    }

    if total == 0 {
        return 0.0;
    }

    // Calculate KL divergence from uniform (null1 model)
    // D_KL(P || Q) = sum_i P(i) * log2(P(i) / Q(i))
    // where P is observed, Q is uniform (0.25)
    let mut kl_divergence = 0.0f64;
    let total_f = total as f64;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total_f;
            // log2(p / 0.25) = log2(p) - log2(0.25) = log2(p) + 2
            kl_divergence += p * (p.log2() + 2.0);
        }
    }

    // Bias score scaled by sequence length
    // Longer sequences with same composition have higher bias
    (kl_divergence * total_f).max(0.0) as f32
}

/// Calculate GC content of a sequence
pub fn calculate_gc_content(dsq: &[u8], l: i32) -> f32 {
    if l <= 0 {
        return 0.5;
    }

    let mut gc = 0u32;
    let mut total = 0u32;

    for i in 1..=(l as usize) {
        if i < dsq.len() {
            let res = dsq[i];
            if res < 4 {
                total += 1;
                // C=1, G=2
                if res == 1 || res == 2 {
                    gc += 1;
                }
            }
        }
    }

    if total == 0 {
        0.5
    } else {
        gc as f32 / total as f32
    }
}

/// Null3 omega parameter (from HMMER)
pub const NULL3_OMEGA: f64 = 0.000015258791; // 1/65536

/// Calculate Null3 composition bias score
///
/// This implements the Null3 composition bias correction used in Infernal.
/// The bias score measures how much a hit score might be inflated due to
/// sequence composition differing from the null model.
///
/// # Arguments
/// * `dsq` - Digitized sequence (0-3 for A,C,G,U, 255 for sentinel)
/// * `l` - Sequence length
/// * `null_probs` - Background null model probabilities [A,C,G,U]
///
/// # Returns
/// Bias score in bits (positive = biased composition)
///
/// C: ScoreCorrectionNull3CompUnknown (cm_parsetree.c) — count the composition of
/// dsq[1..=l], then ScoreCorrectionNull3. The previous body here used a NON-C
/// omega-mixture form (`omega*null1 + (1-omega)*obs`) that inflated the bias on
/// skewed compositions; we delegate to the single C-faithful path so the numbers
/// match C exactly (e.g. all-A len 8 → exactly 1.0 bit, not the mixture's larger value).
pub fn calculate_null3_bias(dsq: &[u8], l: i32, null_probs: &[f32; 4]) -> f32 {
    if l <= 0 {
        return 0.0;
    }
    score_correction_null3_comp_unknown(dsq, 1, l, null_probs, NULL3_OMEGA)
}

/// LogSum2: log2(2^a + 2^b).
///
/// C `LogSum2()`/`FLogsum()` (logsum.c:165) is NOT the analytic form — it reads a
/// DISCRETIZED lookup table (INTSCALE=1000, FLOGSUM_TBL=23000). To match C's
/// numbers bit-for-bit we delegate to the single faithful table implementation in
/// cp9 rather than recomputing `log2(1+2^(min-max))` analytically (which
/// diverges by up to a table-quantum from C).
pub fn log_sum2(a: f32, b: f32) -> f32 {
    crate::cp9::flogsum(a, b)
}

/// Compute nucleotide composition from digitized sequence region
pub fn get_composition(dsq: &[u8], start: i32, stop: i32) -> [f32; 4] {
    let mut counts = [0u32; 4];
    let mut total = 0u32;

    // Count canonical residues (0=A, 1=C, 2=G, 3=U/T)
    for i in (start as usize)..=(stop as usize) {
        if i < dsq.len() {
            let residue = dsq[i];
            if residue < 4 {
                counts[residue as usize] += 1;
                total += 1;
            }
        }
    }

    // Normalize to frequencies (default to uniform if no valid residues)
    if total == 0 {
        return [0.25; 4];
    }

    let total_f = total as f32;
    [
        counts[0] as f32 / total_f,
        counts[1] as f32 / total_f,
        counts[2] as f32 / total_f,
        counts[3] as f32 / total_f,
    ]
}

/// C: ScoreCorrectionNull3 (cm_parsetree.c:2352).
///
/// This is a thin adapter onto the SINGLE C-faithful implementation in
/// cp9. C computes each `sreLOG2(comp[a]/null[a])` term in DOUBLE, then
/// accumulates into a FLOAT `score` (rounding per step), adds `sreLOG2(omega)`
/// (double), and finishes with the discretized-table `LogSum2(0., score)`. The
/// earlier "simplified" body here (exact f32 log2 + analytic log_sum2) diverged
/// from C by ~1e-4..6e-3 bit. We delegate so the numbers are C-identical and there
/// is exactly one source of truth. Note the arg order flips (comp/null vs null0/comp).
pub fn score_correction_null3(
    comp: &[f32; 4],      // Normalized composition frequencies [A, C, G, U]
    null: &[f32; 4],      // Null model probabilities
    len: i32,             // Hit length
    omega: f64,           // Null3 omega parameter (default 1/65536)
) -> f32 {
    crate::cp9::score_correction_null3(null, comp, len, omega as f32)
}

/// Simple API: compute NULL3 correction from sequence region
/// Matches C's ScoreCorrectionNull3CompUnknown()
pub fn score_correction_null3_comp_unknown(
    dsq: &[u8],
    start: i32,
    stop: i32,
    null: &[f32; 4],
    omega: f64,
) -> f32 {
    let comp = get_composition(dsq, start, stop);
    let len = stop - start + 1;
    score_correction_null3(&comp, null, len, omega)
}

/// Apply null3 correction to a raw score
///
/// # Arguments
/// * `raw_score` - Raw bit score from alignment
/// * `null3_bias` - Null3 bias score from calculate_null3_bias()
///
/// # Returns
/// Corrected score (raw_score - null3_bias)
pub fn apply_null3_correction(raw_score: f32, null3_bias: f32) -> f32 {
    raw_score - null3_bias
}

/// Calculate composition-corrected E-value
///
/// # Arguments
/// * `params` - E-value parameters
/// * `raw_score` - Raw bit score
/// * `null3_bias` - Null3 bias correction
///
/// # Returns
/// E-value after bias correction
pub fn score_to_evalue_corrected(params: &ExpParams, raw_score: f64, null3_bias: f64) -> f64 {
    let corrected_score = raw_score - null3_bias;
    params.score_to_evalue(corrected_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_to_evalue_conversion() {
        let params = ExpParams::default();

        // High scores should give low E-values
        let evalue_high = params.score_to_evalue(100.0);
        assert!(evalue_high < 1e-10, "High score should have low E-value");

        // Low scores should give high E-values
        let evalue_low = params.score_to_evalue(10.0);
        assert!(evalue_low > 1.0, "Low score should have high E-value");

        // Scores should be monotonic - higher score = lower E-value
        assert!(evalue_high < evalue_low);
    }

    #[test]
    fn test_evalue_to_score_roundtrip() {
        let params = ExpParams::default();

        let original_score = 50.0;
        let evalue = params.score_to_evalue(original_score);
        let recovered_score = params.evalue_to_score(evalue);

        assert!((original_score - recovered_score).abs() < 1e-6,
                "Roundtrip conversion should preserve score");
    }

    #[test]
    fn test_is_calibrated() {
        let params = ExpParams::default();
        assert!(params.is_calibrated());

        let uncalibrated = ExpParams {
            lambda: 0.0,
            mu: 0.0,
            dbsize: 0.0,
            nrandhits: 0,
            mu_orig: 0.0,
            tailp: 0.0,
        };
        assert!(!uncalibrated.is_calibrated());
    }

    #[test]
    fn test_p7_uncalibrated() {
        let p7_params = P7ExpParams::default();
        assert!(!p7_params.is_calibrated());

        let mut calibrated = P7ExpParams::default();
        calibrated.lmlambda = 0.5;
        assert!(calibrated.is_calibrated());
    }

    #[test]
    fn test_null3_bias_uniform() {
        // Uniform sequence should have minimal bias
        let dsq = vec![255, 0, 1, 2, 3, 0, 1, 2, 3, 255]; // sentinel, ACGUACGU, sentinel
        let null_probs = [0.25, 0.25, 0.25, 0.25];

        let bias = calculate_null3_bias(&dsq, 8, &null_probs);

        // Bias should be very small (near zero) for uniform composition
        assert!(bias < 0.1, "Uniform composition should have minimal bias, got {}", bias);
    }

    #[test]
    fn test_null3_bias_skewed() {
        // All A's - highly skewed composition
        let dsq = vec![255, 0, 0, 0, 0, 0, 0, 0, 0, 255];
        let null_probs = [0.25, 0.25, 0.25, 0.25];

        let bias = calculate_null3_bias(&dsq, 8, &null_probs);

        // Skewed composition yields a positive bias (faithful ScoreCorrectionNull3 path).
        assert!(bias > 0.0, "Skewed composition should have positive bias, got {}", bias);
    }

    #[test]
    fn test_apply_null3_correction() {
        let raw_score = 50.0;
        let null3_bias = 5.0;

        let corrected = apply_null3_correction(raw_score, null3_bias);

        assert_eq!(corrected, 45.0);
    }

    #[test]
    fn test_score_to_evalue_corrected() {
        let params = ExpParams::default();
        let raw_score = 50.0;
        let null3_bias = 5.0;

        // E-value with correction should be higher (worse) than without
        let evalue_raw = params.score_to_evalue(raw_score);
        let evalue_corrected = score_to_evalue_corrected(&params, raw_score, null3_bias);

        assert!(evalue_corrected > evalue_raw,
                "Corrected E-value should be higher (bias reduces score)");
    }

    #[test]
    fn test_null3_bias_empty() {
        let dsq = vec![255, 255]; // Just sentinels
        let null_probs = [0.25, 0.25, 0.25, 0.25];

        let bias = calculate_null3_bias(&dsq, 0, &null_probs);

        assert_eq!(bias, 0.0, "Empty sequence should have zero bias");
    }

    #[test]
    fn test_null3_omega_constant() {
        // Verify omega is 1/65536
        let expected = 1.0 / 65536.0;
        assert!((NULL3_OMEGA - expected).abs() < 1e-10,
                "NULL3_OMEGA should be 1/65536");
    }

    #[test]
    fn test_log_sum2() {
        // Basic test: log2(2^3 + 2^4) = log2(8 + 16) = log2(24) ≈ 4.585
        let result = log_sum2(3.0, 4.0);
        let expected = 24.0_f32.log2();
        assert!((result - expected).abs() < 1e-5, "Basic log_sum2 failed");

        // Test with -infinity (should return the other value)
        assert_eq!(log_sum2(f32::NEG_INFINITY, 5.0), 5.0);
        assert_eq!(log_sum2(5.0, f32::NEG_INFINITY), 5.0);

        // Test with large difference (>23 bits, max dominates)
        let result = log_sum2(0.0, 25.0);
        assert_eq!(result, 25.0);

        // Test symmetry
        assert_eq!(log_sum2(3.0, 4.0), log_sum2(4.0, 3.0));
    }

    #[test]
    fn test_get_composition() {
        // Test uniform sequence
        let dsq = vec![255, 0, 1, 2, 3, 255]; // A, C, G, U
        let comp = get_composition(&dsq, 1, 4);
        assert!((comp[0] - 0.25).abs() < 1e-5);
        assert!((comp[1] - 0.25).abs() < 1e-5);
        assert!((comp[2] - 0.25).abs() < 1e-5);
        assert!((comp[3] - 0.25).abs() < 1e-5);

        // Test all A's
        let dsq = vec![255, 0, 0, 0, 0, 255];
        let comp = get_composition(&dsq, 1, 4);
        assert!((comp[0] - 1.0).abs() < 1e-5);
        assert!((comp[1] - 0.0).abs() < 1e-5);
        assert!((comp[2] - 0.0).abs() < 1e-5);
        assert!((comp[3] - 0.0).abs() < 1e-5);

        // Test empty region (should return uniform)
        let dsq = vec![255, 255];
        let comp = get_composition(&dsq, 1, 0);
        assert!((comp[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_score_correction_null3() {
        // Test with uniform composition and uniform null
        let comp = [0.25, 0.25, 0.25, 0.25];
        let null = [0.25, 0.25, 0.25, 0.25];
        let len = 100;
        let omega = NULL3_OMEGA;

        let correction = score_correction_null3(&comp, &null, len, omega);

        // For uniform composition, correction should be close to log2(omega) soft-capped
        // which is log_sum2(0, log2(omega)) = log_sum2(0, -16) ≈ 0
        assert!(correction.abs() < 1e-4, "Uniform composition should have near-zero correction");
    }

    #[test]
    fn test_score_correction_null3_comp_unknown() {
        // Test the convenience wrapper
        let dsq = vec![255, 0, 1, 2, 3, 255]; // uniform sequence
        let null = [0.25, 0.25, 0.25, 0.25];
        let omega = NULL3_OMEGA;

        let correction = score_correction_null3_comp_unknown(&dsq, 1, 4, &null, omega);

        // Should match direct call to score_correction_null3
        let comp = get_composition(&dsq, 1, 4);
        let expected = score_correction_null3(&comp, &null, 4, omega);

        assert!((correction - expected).abs() < 1e-5);
    }
}
