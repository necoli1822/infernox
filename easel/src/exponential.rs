//! Exponential distribution functions — faithful port of `easel/esl_exponential.c`.

/// Survival function `P(X > x)` for the exponential distribution.
/// Port of C `esl_exp_surv` (easel/esl_exponential.c).
#[inline]
pub fn esl_exp_surv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 1.0;
    }
    (-lambda * (x - mu)).exp()
}
