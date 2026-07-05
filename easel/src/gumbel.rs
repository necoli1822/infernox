//! Gumbel (Type I extreme value) distribution functions.
//!
//! 1:1 port from esl_gumbel.c - statistical routines for Gumbel distributions.

use crate::constants::ESL_SMALLX1;

/// Calculates the probability density function (PDF) for the Gumbel distribution.
///
/// P(X=x) = lambda * exp(-y - exp(-y))
/// where y = lambda * (x - mu)
///
/// # Arguments
/// * `x` - The quantile
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The probability density at x
#[inline]
pub fn esl_gumbel_pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    lambda * (-y - (-y).exp()).exp()
}

/// Calculates the log probability density function for the Gumbel distribution.
///
/// log P(X=x) = log(lambda) - y - exp(-y)
/// where y = lambda * (x - mu)
///
/// # Arguments
/// * `x` - The quantile
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The log probability density at x
#[inline]
pub fn esl_gumbel_logpdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    lambda.ln() - y - (-y).exp()
}

/// Calculates the cumulative distribution function (CDF) for the Gumbel distribution.
///
/// P(X <= x) = exp(-exp(-y))
/// where y = lambda * (x - mu)
///
/// # Arguments
/// * `x` - The quantile
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The cumulative probability P(X <= x)
#[inline]
pub fn esl_gumbel_cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    (-(-y).exp()).exp()
}

/// Calculates the log cumulative distribution function for the Gumbel distribution.
///
/// log P(X <= x) = -exp(-y)
/// where y = lambda * (x - mu)
///
/// # Arguments
/// * `x` - The quantile
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The log cumulative probability
#[inline]
pub fn esl_gumbel_logcdf(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    -(-y).exp()
}

/// Calculates the survival function (right tail probability) for the Gumbel distribution.
///
/// P(X > x) = 1 - CDF = 1 - exp(-exp(-y))
/// where y = lambda * (x - mu)
///
/// Uses approximation for numerical stability when exp(-y) is small.
///
/// # Arguments
/// * `x` - The quantile
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The survival probability P(X > x)
#[inline]
pub fn esl_gumbel_surv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    let ey = -(-y).exp();

    // Use 1 - e^x ~ -x approximation when e^-y is small
    if ey.abs() < ESL_SMALLX1 {
        -ey
    } else {
        1.0 - ey.exp()
    }
}

/// Calculates the log survival function for the Gumbel distribution.
///
/// log P(X > x) = log(1 - exp(-exp(-y)))
/// where y = lambda * (x - mu)
///
/// Uses approximations for numerical stability:
/// - For large y: log(1 - exp(-exp(-y))) ~ -y
/// - For small y: log(1 - x) ~ -x when x is small
///
/// # Arguments
/// * `x` - The quantile
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The log survival probability
#[inline]
pub fn esl_gumbel_logsurv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    let ey = -(-y).exp();

    // The real calculation is log(1-exp(-exp(-y))).
    // For "large" y, -exp(-y) is small, so 1-exp(-exp(-y)) ~ exp(-y),
    // and log of that gives us -y.
    // For "small" y, exp(-exp(-y)) is small, and we can use log(1-x) ~ -x.
    if ey.abs() < ESL_SMALLX1 {
        -y
    } else if ey.exp().abs() < ESL_SMALLX1 {
        -ey.exp()
    } else {
        (1.0 - ey.exp()).ln()
    }
}

/// Calculates the inverse CDF (quantile function) for the Gumbel distribution.
///
/// Returns the quantile x at which CDF(x) = p.
///
/// # Arguments
/// * `p` - The probability (0 < p < 1)
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The quantile x such that P(X <= x) = p
#[inline]
pub fn esl_gumbel_invcdf(p: f64, mu: f64, lambda: f64) -> f64 {
    mu - ((-p.ln()).ln() / lambda)
}

/// Calculates the inverse survival function for the Gumbel distribution.
///
/// Returns the quantile x at which the right tail mass equals p.
///
/// # Arguments
/// * `p` - The tail probability (0 < p < 1)
/// * `mu` - Location parameter
/// * `lambda` - Scale parameter (lambda > 0)
///
/// # Returns
/// The quantile x such that P(X > x) = p
#[inline]
pub fn esl_gumbel_invsurv(p: f64, mu: f64, lambda: f64) -> f64 {
    // The real calculation is mu - (log(-log(1-p)) / lambda).
    // For small p, use approximation: log(1-p) ~= -p
    // and log(-log(1-p)) ~= log(p)
    let log_part = if p < ESL_SMALLX1 {
        (p.powf(p) - 1.0) / p
    } else {
        (-(1.0 - p).ln()).ln()
    };

    mu - (log_part / lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        if a.is_infinite() && b.is_infinite() {
            return a.signum() == b.signum();
        }
        if a.abs() < EPSILON && b.abs() < EPSILON {
            return true;
        }
        (a - b).abs() / a.abs().max(b.abs()).max(1.0) < EPSILON
    }

    #[test]
    fn test_gumbel_pdf_at_mode() {
        // For standard Gumbel (mu=0, lambda=1), mode is at x=0
        // PDF at mode = 1/e = 0.367879...
        let pdf = esl_gumbel_pdf(0.0, 0.0, 1.0);
        let expected = 1.0 / std::f64::consts::E;
        assert!(
            approx_eq(pdf, expected),
            "PDF at mode: {} vs {}",
            pdf,
            expected
        );
    }

    #[test]
    fn test_gumbel_cdf_properties() {
        // CDF(mu) for standard Gumbel = exp(-1) = 0.367879...
        let cdf = esl_gumbel_cdf(0.0, 0.0, 1.0);
        let expected = (-1.0_f64).exp();
        assert!(
            approx_eq(cdf, expected),
            "CDF at mu: {} vs {}",
            cdf,
            expected
        );
    }

    #[test]
    fn test_gumbel_surv_cdf_relationship() {
        // surv(x) + cdf(x) = 1
        let x = 1.5;
        let mu = 0.0;
        let lambda = 1.0;
        let surv = esl_gumbel_surv(x, mu, lambda);
        let cdf = esl_gumbel_cdf(x, mu, lambda);
        assert!(
            approx_eq(surv + cdf, 1.0),
            "surv + cdf should equal 1: {} + {} = {}",
            surv,
            cdf,
            surv + cdf
        );
    }

    #[test]
    fn test_gumbel_logpdf_pdf_relationship() {
        // logpdf = ln(pdf)
        let x = 0.5;
        let mu = 0.0;
        let lambda = 1.0;
        let pdf = esl_gumbel_pdf(x, mu, lambda);
        let logpdf = esl_gumbel_logpdf(x, mu, lambda);
        assert!(
            approx_eq(logpdf, pdf.ln()),
            "logpdf should equal ln(pdf): {} vs {}",
            logpdf,
            pdf.ln()
        );
    }

    #[test]
    fn test_gumbel_invcdf_cdf_inverse() {
        // invcdf(cdf(x)) = x
        let x = 1.5;
        let mu = -20.0;
        let lambda = 0.4;
        let p = esl_gumbel_cdf(x, mu, lambda);
        let recovered = esl_gumbel_invcdf(p, mu, lambda);
        assert!(
            approx_eq(x, recovered),
            "invcdf(cdf(x)) should equal x: {} vs {}",
            x,
            recovered
        );
    }
}
