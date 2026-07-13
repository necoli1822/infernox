//! Exponential distribution functions — faithful port of `easel/esl_exponential.c`.

use crate::easel::constants::ESL_SMALLX1;
use crate::easel::error::{InfernalError, Result};

/// Cumulative distribution function `P(X <= x)` for the exponential.
/// Port of C `esl_exp_cdf` (esl_exponential.c:86).
#[inline]
pub fn esl_exp_cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu); // y>=0 because lambda>0 and x>=mu

    if x < mu {
        return 0.;
    }

    // 1-e^-y ~ y for small |y|
    if y < ESL_SMALLX1 {
        y
    } else {
        1. - (-y).exp()
    }
}

/// Survival function `P(X > x)` for the exponential distribution.
/// Port of C `esl_exp_surv` (esl_exponential.c:127).
#[inline]
pub fn esl_exp_surv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 1.0;
    }
    (-lambda * (x - mu)).exp()
}

/// Log survivor function `log P(X > x)` for the exponential distribution.
/// Port of C `esl_exp_logsurv` (esl_exponential.c:141).
#[inline]
pub fn esl_exp_logsurv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        return 0.0;
    }
    -lambda * (x - mu)
}

/// Generic-API version of CDF: `params = [mu, lambda]`.
/// Port of C `esl_exp_generic_cdf` (esl_exponential.c:202).
#[inline]
pub fn esl_exp_generic_cdf(x: f64, params: &[f64]) -> f64 {
    esl_exp_cdf(x, params[0], params[1])
}

/// Generic-API version of survival function: `params = [mu, lambda]`.
/// Port of C `esl_exp_generic_surv` (esl_exponential.c:213).
#[inline]
pub fn esl_exp_generic_surv(x: f64, params: &[f64]) -> f64 {
    esl_exp_surv(x, params[0], params[1])
}

/// Given `n` samples `x[0..n-1]`, fit them to an exponential distribution;
/// return maximum-likelihood parameters `(mu, lambda)`.
///
/// The ML `mu` is the lowest score (`mu = x_i` is ok in the exponential).
/// The ML `lambda = 1 / mean(x_i - mu)` — trivial & analytic.
///
/// Port of C `esl_exp_FitComplete` (esl_exponential.c:310).
/// Throws `Inval` if `n == 0` (empty data vector).
#[allow(non_snake_case)]
pub fn esl_exp_FitComplete(x: &[f64], n: usize) -> Result<(f64, f64)> {
    if n == 0 {
        return Err(InfernalError::Inval);
    }

    // ML mu is the lowest score. mu=x is ok in the exponential.
    let mut mu = x[0];
    for i in 1..n {
        if x[i] < mu {
            mu = x[i];
        }
    }

    let mut mean = 0.;
    for i in 0..n {
        mean += x[i] - mu;
    }
    mean /= n as f64;

    let ret_mu = mu;
    let ret_lambda = 1. / mean; // ML estimate trivial & analytic
    Ok((ret_mu, ret_lambda))
}

/// Given `n` samples with known location `mu`, return the ML scale `lambda`.
/// Port of C `esl_exp_FitCompleteScale` (esl_exponential.c:355).
#[allow(non_snake_case)]
pub fn esl_exp_FitCompleteScale(x: &[f64], n: usize, mu: f64) -> f64 {
    let mut mean = 0.;
    for i in 0..n {
        mean += x[i] - mu;
    }
    mean /= n as f64;
    1. / mean // ML estimate trivial & analytic
}
