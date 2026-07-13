//! Collection and display of score histograms.
//!
//! Faithful 1:1 port of `easel/esl_histogram.c` (and the `ESL_HISTOGRAM`
//! struct in `esl_histogram.h`). This is the statistical core used by
//! `cmcalibrate`'s `fit_histogram()`:
//!
//! ```text
//!   h = esl_histogram_CreateFull(-100,100,.1)
//!   esl_histogram_Add(h, score)   for each random-seq hit
//!   esl_histogram_GetTailByMass(h, tailp, &xv,&n,&z)
//!   esl_exp_FitComplete(xv, n, &mu, &lambda)
//!   esl_histogram_SetExpectedTail(h, mu, tailp, cdf, params)
//! ```
//!
//! Only the subset needed by that call sequence (plus the easy display
//! helpers) is ported. Porting-comment convention: each function is
//! annotated with `esl_histogram.c:<function>:<line>`.

use crate::easel::error::{InfernalError, Result};

/// Which kind of dataset the histogram represents.
///
/// Port of the anonymous `enum { COMPLETE, VIRTUAL_CENSORED, TRUE_CENSORED }`
/// in `esl_histogram.h:65`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetIs {
    Complete,
    VirtualCensored,
    TrueCensored,
}

/// Structure: ESL_HISTOGRAM
///
/// Keeps a score histogram, in which scores are counted into bins of
/// size (width) `w`.
///   histogram starts at `bmin <  floor(xmin/w) * w`
///   histogram ends at   `bmax >= ceil(xmax/w)*w`
///   `nb = (bmax-bmin)/w`
///   each score x is counted into bin `b = nb - (int) (bmax-x)/w`
///   each bin b contains scores `bw+bmin < x <= (b+1)w + bmin`
///
/// Port of `ESL_HISTOGRAM` (esl_histogram.h:25).
pub struct EslHistogram {
    /* The histogram is kept as counts in fixed-width bins. */
    /// observed counts in bin b, 0..nb-1 (dynamic)
    pub obs: Vec<u64>,
    /// number of bins
    pub nb: i32,
    /// fixed width of each bin
    pub w: f64,
    /// histogram lower bound: all x satisfy bmin < x <= bmax
    pub bmin: f64,
    /// histogram upper bound
    pub bmax: f64,
    /// smallest bin that contains obs[i] > 0
    pub imin: i32,
    /// largest bin that contains obs[i] > 0
    pub imax: i32,

    /* Optionally, in a "full" h, we can also keep all the raw samples in x. */
    /// smallest sample value x observed
    pub xmin: f64,
    /// largest sample value x observed
    pub xmax: f64,
    /// total number of raw data samples
    pub n: u64,
    /// optional: raw sample values x[0..n-1]
    pub x: Vec<f64>,
    /// current allocated size of x (kept for faithfulness; Vec manages storage)
    pub nalloc: u64,

    /* Censoring information for binned/censored parameter fitting. */
    /// censoring value; all x_i > phi
    pub phi: f64,
    /// smallest bin index that contains uncensored data
    pub cmin: i32,
    /// # of censored values <= phi
    pub z: u64,
    /// # samples in complete data (including unobs)
    pub Nc: u64,
    /// # of samples in observed data
    pub No: u64,

    /* Expected binned counts, set by SetExpect() or SetExpectedTail(). */
    /// expected counts in bin b, 0..nb-1 (not resized). Empty until a Set*() call.
    pub expect: Vec<f64>,
    /// smallest bin index that contains expected counts
    pub emin: i32,
    /// for tail fits: fitted x > tailbase
    pub tailbase: f64,
    /// for tail fits: fractional prob in the tail
    pub tailmass: f64,

    /* Status flags */
    /// TRUE when we're keeping raw data in x
    pub is_full: bool,
    /// TRUE if we prevent more Add()'s
    pub is_done: bool,
    /// TRUE if x is sorted smallest-to-largest
    pub is_sorted: bool,
    /// TRUE if expected dist only describes tail
    pub is_tailfit: bool,
    /// TRUE if values aren't more accurate than bins
    pub is_rounded: bool,
    pub dataset_is: DatasetIs,
}

/// `esl_histogram_Bin2LBound(h,b)`  (esl_histogram.h:69)
#[inline]
pub fn bin2lbound(h: &EslHistogram, b: i32) -> f64 {
    h.w * (b as f64) + h.bmin
}

/// `esl_histogram_Bin2UBound(h,b)`  (esl_histogram.h:70)
#[inline]
pub fn bin2ubound(h: &EslHistogram, b: i32) -> f64 {
    h.w * ((b + 1) as f64) + h.bmin
}

impl EslHistogram {
    /// Creates and returns a new (display-only) histogram object.
    ///
    /// Port of `esl_histogram_Create()` (esl_histogram.c:76).
    pub fn create(bmin: f64, bmax: f64, w: f64) -> EslHistogram {
        // h->nb = (int)((bmax-bmin)/w);
        let nb = ((bmax - bmin) / w) as i32;

        let mut h = EslHistogram {
            xmin: f64::MAX,   // DBL_MAX
            xmax: -f64::MAX,  // -DBL_MAX
            n: 0,
            obs: Vec::new(),  // allocated below
            bmin,
            bmax,
            nb,
            imin: nb,
            imax: -1,
            w,

            x: Vec::new(),
            nalloc: 0,

            phi: 0.,
            cmin: nb, // sentinel: no observed data yet (= h->imin)
            z: 0,
            Nc: 0,
            No: 0,

            expect: Vec::new(), // 'til a Set*() call
            emin: -1,           // sentinel: no expected counts yet
            tailbase: 0.,       // unused unless is_tailfit TRUE
            tailmass: 1.0,      // <= 1.0 if is_tailfit TRUE

            is_full: false,
            is_done: false,
            is_sorted: false,
            is_tailfit: false,
            is_rounded: false,
            dataset_is: DatasetIs::Complete,
        };

        // ESL_ALLOC(h->obs, sizeof(uint64_t) * h->nb); for(...) h->obs[i]=0;
        h.obs = vec![0u64; nb.max(0) as usize];
        h
    }

    /// Alternative form of `create()` that also keeps all raw sample values.
    ///
    /// Port of `esl_histogram_CreateFull()` (esl_histogram.c:136).
    pub fn create_full(bmin: f64, bmax: f64, w: f64) -> EslHistogram {
        let mut h = EslHistogram::create(bmin, bmax, w);
        h.n = 0; // make sure
        h.nalloc = 128; // arbitrary initial allocation size
        h.x = Vec::with_capacity(128);
        h.is_full = true;
        h
    }

    /// For a real-valued `x`, figure out what bin it would go into.
    ///
    /// Port of `esl_histogram_Score2Bin()` (esl_histogram.c:186).
    ///
    /// Throws `Range` if bin would exceed the range of an int (e.g. x not finite).
    pub fn score2bin(&self, x: f64) -> Result<i32> {
        if !x.is_finite() {
            return Err(InfernalError::Range);
        }

        // x = ceil( ((x - h->bmin) / h->w) - 1. );
        let xb = (((x - self.bmin) / self.w) - 1.).ceil();

        // Check for under/overflow before conversion (INT_MIN/INT_MAX = i32).
        if xb < (i32::MIN as f64) || xb > (i32::MAX as f64) {
            return Err(InfernalError::Range);
        }

        Ok(xb as i32)
    }

    /// Adds score `x` to the histogram, reallocating bins as needed.
    ///
    /// Port of `esl_histogram_Add()` (esl_histogram.c:235).
    pub fn add(&mut self, x: f64) -> Result<()> {
        // Don't allow caller to add data after configuration is declared.
        if self.is_done {
            return Err(InfernalError::Inval);
        }

        // (In C, the full-data vector is grown by 2x here; Vec::push handles that.)

        // Which bin will we want to put x into?
        let mut b = self.score2bin(x)?;

        // Make sure we have that bin. Realloc as needed.
        if b < 0 {
            // Reallocate below.
            let nnew = -b * 2; // overallocate by 2x
            if nnew > i32::MAX - self.nb {
                return Err(InfernalError::Range);
            }
            // memmove: new low bins are zeros; old obs shift up by nnew.
            let mut newobs = vec![0u64; (nnew + self.nb) as usize];
            newobs[nnew as usize..].copy_from_slice(&self.obs);
            self.obs = newobs;

            self.nb += nnew;
            b += nnew;
            self.bmin -= (nnew as f64) * self.w;
            self.imin += nnew;
            self.cmin += nnew;
            if self.imax > -1 {
                self.imax += nnew;
            }
        } else if b >= self.nb {
            // Reallocate above.
            let nnew = (b - self.nb + 1) * 2; // 2x overalloc
            if nnew > i32::MAX - self.nb {
                return Err(InfernalError::Range);
            }
            self.obs.resize((self.nb + nnew) as usize, 0u64);
            if self.imin == self.nb {
                // boundary condition of no data yet
                self.imin += nnew;
                self.cmin += nnew;
            }
            self.bmax += (nnew as f64) * self.w;
            self.nb += nnew;
        }

        // If full histogram, keep the raw x value.
        if self.is_full {
            self.x.push(x);
            if self.x.len() as u64 > self.nalloc {
                self.nalloc = self.x.capacity() as u64;
            }
        }
        self.is_sorted = false; // not any more!

        // Bump the bin counter, and all the data sample counters.
        self.obs[b as usize] += 1;
        self.n += 1;
        self.Nc += 1;
        self.No += 1;

        if b > self.imax {
            self.imax = b;
        }
        if b < self.imin {
            self.imin = b;
            self.cmin = b;
        }
        if x > self.xmax {
            self.xmax = x;
        }
        if x < self.xmin {
            self.xmin = x;
        }
        Ok(())
    }

    /// Sort the raw scores from smallest to largest.
    ///
    /// Port of `esl_histogram_sort()` (esl_histogram.c:332).
    fn sort(&mut self) {
        if self.is_sorted {
            return; // already sorted
        }
        if !self.is_full {
            return; // nothing to sort
        }
        // esl_vec_DSortIncreasing: qsort increasing. Scores are finite.
        self.x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.is_sorted = true;
    }

    /// Retrieve the `rank`'th highest score (rank in 1..n).
    ///
    /// Port of `esl_histogram_GetRank()` (esl_histogram.c:535).
    pub fn get_rank(&mut self, rank: u64) -> Result<f64> {
        if !self.is_full {
            return Err(InfernalError::Inval);
        }
        if rank > self.n {
            return Err(InfernalError::Inval);
        }
        if rank < 1 {
            return Err(InfernalError::Inval);
        }
        self.sort();
        Ok(self.x[(self.n - rank) as usize])
    }

    /// Retrieve the vector of all raw scores, sorted smallest-to-largest.
    ///
    /// Port of `esl_histogram_GetData()` (esl_histogram.c:586).
    /// Sets `is_done` and returns a slice into internal storage.
    pub fn get_data(&mut self) -> Result<(&[f64], usize)> {
        if !self.is_full {
            return Err(InfernalError::Inval);
        }
        self.sort();
        self.is_done = true;
        let n = self.n as usize;
        Ok((&self.x[..n], n))
    }

    /// Retrieve the data values in the right (high-scoring) tail, defined by
    /// a mass fraction threshold `pmass` (mass in the returned tail is `<= pmass`).
    ///
    /// Port of `esl_histogram_GetTailByMass()` (esl_histogram.c:701).
    ///
    /// Returns `(xv, n, z)` where `xv` is the sorted tail slice `[0..n-1]`,
    /// `n` is the number of tail samples, and `z` is the number not in the tail.
    pub fn get_tail_by_mass(&mut self, pmass: f64) -> Result<(&[f64], usize, u64)> {
        if !self.is_full {
            return Err(InfernalError::Inval);
        }
        if pmass < 0. || pmass > 1. {
            return Err(InfernalError::Inval);
        }

        self.sort();

        // n = (uint64_t) ((double) h->n * pmass);  /* rounds down => <= pmass */
        let n = ((self.n as f64) * pmass) as u64;

        self.is_done = true;
        let start = (self.n - n) as usize;
        let z = self.n - n;
        Ok((&self.x[start..], n as usize, z))
    }

    /// Declare that only the tail above threshold `phi` is "observed".
    ///
    /// Port of `esl_histogram_SetTail()` (esl_histogram.c:435).
    /// Returns the fractional probability mass now in the right tail.
    pub fn set_tail(&mut self, phi: f64) -> Result<f64> {
        // Usually put true phi at the next bin lower bound, but watch for the
        // case where phi is already exactly a bin upper bound.
        self.cmin = self.score2bin(phi)?;
        if phi == bin2ubound(self, self.cmin) {
            self.phi = phi;
        } else {
            self.phi = bin2lbound(self, self.cmin);
        }

        self.z = 0;
        for b in self.imin..self.cmin {
            self.z += self.obs[b as usize];
        }
        self.Nc = self.n; // (redundant)
        self.No = self.n - self.z;
        self.dataset_is = DatasetIs::VirtualCensored;
        self.is_done = true;
        Ok((self.No as f64) / (self.Nc as f64))
    }

    /// Find a cutoff score that at least fraction `pmass` of the samples exceed,
    /// declaring the binned data as a (virtually) left-censored tail dataset.
    ///
    /// Port of `esl_histogram_SetTailByMass()` (esl_histogram.c:487).
    /// Returns the actual fractional mass in the right tail.
    pub fn set_tail_by_mass(&mut self, pmass: f64) -> Result<f64> {
        let mut sum: u64 = 0;
        let mut b = self.imax;
        while b >= self.imin {
            sum += self.obs[b as usize];
            if (sum as f64) >= pmass * (self.n as f64) {
                break;
            }
            b -= 1;
        }

        self.phi = bin2lbound(self, b);
        self.z = self.n - sum;
        self.cmin = b;
        self.Nc = self.n; // (redundant)
        self.No = self.n - self.z;
        self.dataset_is = DatasetIs::VirtualCensored;
        self.is_done = true;
        Ok((self.No as f64) / (self.Nc as f64))
    }

    /// Set expected binned counts from a complete-distribution CDF.
    ///
    /// Port of `esl_histogram_SetExpect()` (esl_histogram.c:755).
    ///
    /// `cdf(x)` is the generic-interface CDF (the C `(*cdf)(x, params)`, with
    /// `params` captured by the closure).
    pub fn set_expect<F: Fn(f64) -> f64>(&mut self, cdf: F) {
        if self.expect.is_empty() {
            self.expect = vec![0.0; self.nb.max(0) as usize];
        }
        for i in 0..self.nb {
            let ai = bin2lbound(self, i);
            let bi = bin2ubound(self, i);
            self.expect[i as usize] = (self.Nc as f64) * (cdf(bi) - cdf(ai));
            if self.emin == -1 && self.expect[i as usize] > 0. {
                self.emin = i;
            }
        }
        self.is_done = true;
    }

    /// Set expected binned counts for the right tail starting at `base_val`,
    /// containing a fraction `pmass` of the complete distribution.
    ///
    /// Port of `esl_histogram_SetExpectedTail()` (esl_histogram.c:812).
    ///
    /// `cdf(x)` is the generic-interface CDF (C `(*cdf)(x, params)`).
    pub fn set_expected_tail<F: Fn(f64) -> f64>(
        &mut self,
        base_val: f64,
        pmass: f64,
        cdf: F,
    ) -> Result<()> {
        if self.expect.is_empty() {
            self.expect = vec![0.0; self.nb.max(0) as usize];
        }

        // h->emin = Score2Bin(base_val) + 1
        self.emin = self.score2bin(base_val)? + 1;

        // esl_vec_DSet(h->expect, h->emin, 0.);  /* zero bins [0..emin-1] */
        for i in 0..self.emin {
            self.expect[i as usize] = 0.;
        }

        for b in self.emin..self.nb {
            let ai = bin2lbound(self, b);
            let bi = bin2ubound(self, b);
            self.expect[b as usize] = pmass * (self.Nc as f64) * (cdf(bi) - cdf(ai));
        }

        self.tailbase = base_val;
        self.tailmass = pmass;
        self.is_tailfit = true;
        self.is_done = true;
        Ok(())
    }

    /// Print observed (and expected, if set) binned counts in xmgrace XY format.
    ///
    /// Port of `esl_histogram_Plot()` (esl_histogram.c:1041).
    pub fn plot<W: std::fmt::Write>(&self, fp: &mut W) -> std::fmt::Result {
        // First data set: the observed histogram.
        let mut i = self.imin;
        while i <= self.imax {
            let x = bin2lbound(self, i);
            writeln!(fp, "{} {}", x, self.obs[i as usize])?;
            i += 1;
        }
        // trailing y=0 (i is now imax+1)
        let x = bin2lbound(self, i);
        writeln!(fp, "{} {}", x, 0)?;
        writeln!(fp, "&")?;

        // Second data set: the theoretical (expected) histogram.
        if !self.expect.is_empty() {
            let mut imin = 0i32;
            while imin < self.nb {
                if self.expect[imin as usize] > 0. {
                    break;
                }
                imin += 1;
            }
            let mut imax = self.nb - 1;
            while imax >= 0 {
                if self.expect[imax as usize] > 0. {
                    break;
                }
                imax -= 1;
            }
            let mut j = imin;
            while j <= imax {
                let x = bin2lbound(self, j);
                writeln!(fp, "{} {}", x, self.expect[j as usize])?;
                j += 1;
            }
            writeln!(fp, "&")?;
        }
        Ok(())
    }
}

// Note: `esl_histogram_Destroy()` (esl_histogram.c:160) is unnecessary in Rust;
// the `Vec` fields free their storage automatically when `EslHistogram` drops.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::easel::exponential::{esl_exp_FitComplete, esl_exp_generic_cdf};

    // Hand-worked GetTailByMass binning semantics on a tiny sorted dataset.
    #[test]
    fn test_get_tail_by_mass_semantics() {
        let mut h = EslHistogram::create_full(-100.0, 100.0, 0.1);
        // 10 samples: 1,2,...,10
        for v in 1..=10 {
            h.add(v as f64).unwrap();
        }
        // pmass 0.5 -> n = floor(10*0.5) = 5, tail = 5 largest = [6,7,8,9,10]
        let (xv, n, z) = h.get_tail_by_mass(0.5).unwrap();
        assert_eq!(n, 5);
        assert_eq!(z, 5);
        assert_eq!(xv, &[6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_get_tail_by_mass_rounds_down() {
        let mut h = EslHistogram::create_full(-100.0, 100.0, 0.1);
        for v in 1..=10 {
            h.add(v as f64).unwrap();
        }
        // pmass 0.33 -> n = floor(10*0.33)=floor(3.3)=3 -> [8,9,10]
        let (xv, n, z) = h.get_tail_by_mass(0.33).unwrap();
        assert_eq!(n, 3);
        assert_eq!(z, 7);
        assert_eq!(xv, &[8.0, 9.0, 10.0]);
    }

    // Full cmcalibrate-style call sequence.
    #[test]
    fn test_cmcalibrate_sequence() {
        let mut h = EslHistogram::create_full(-100.0, 100.0, 0.1);
        let scores = [
            0.5, 1.2, 2.3, 0.1, 3.4, 5.6, 2.1, 4.4, 1.1, 6.7, 3.3, 2.2, 0.9, 7.8, 5.5, 4.1, 3.9,
            2.7, 1.8, 8.2,
        ];
        for &s in scores.iter() {
            h.add(s).unwrap();
        }
        let (mu, lambda) = {
            let (xv, n, _z) = h.get_tail_by_mass(0.5).unwrap();
            esl_exp_FitComplete(xv, n).unwrap()
        };
        // xv should be the 10 largest sorted; mu = min of those.
        // Verify SetExpectedTail wiring works with the generic cdf closure.
        h.set_expected_tail(mu, 0.5, |x| esl_exp_generic_cdf(x, &[mu, lambda]))
            .unwrap();
        assert!(h.is_tailfit);
        assert_eq!(h.tailmass, 0.5);
        assert_eq!(h.tailbase, mu);

        // Bit-for-bit parity against the C probe (original libeasel):
        //   n=10 z=10 mu=3.2999999999999998 lambda=0.50251256281407042
        //   cdf(mu+1)=0.39499137376548188 newmass=0.5
        assert_eq!(mu, 3.2999999999999998);
        assert_eq!(lambda, 0.50251256281407042);
        assert_eq!(
            esl_exp_generic_cdf(mu + 1.0, &[mu, lambda]),
            0.39499137376548188
        );
        let newmass = h.set_tail_by_mass(0.5).unwrap();
        assert_eq!(newmass, 0.5);

        // Independent recompute of FitComplete on the 10 largest.
        let mut sorted = scores;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tail = &sorted[10..];
        let m = tail.iter().cloned().fold(f64::MAX, f64::min);
        let mean: f64 = tail.iter().map(|v| v - m).sum::<f64>() / (tail.len() as f64);
        assert_eq!(mu, m);
        assert_eq!(lambda, 1.0 / mean);
    }

    #[test]
    fn test_score2bin() {
        let h = EslHistogram::create(-100.0, 100.0, 0.1);
        // bin b = ceil(((x-bmin)/w) - 1)
        // bmin=-100, w=0.1. x=-100+eps counts into bin 0.
        // For x just above bmin, (x-bmin)/w ~ small positive, ceil(.. -1) = 0.
        assert_eq!(h.score2bin(-99.95).unwrap(), 0);
        assert!(h.score2bin(f64::INFINITY).is_err());
    }
}
