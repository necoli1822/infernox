//! SIMD-accelerated operations for P7/CP9 filters
//!
//! Provides vectorized implementations using std::simd (nightly) or
//! portable SIMD via packed_simd/simdeez for stable Rust.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD vector width (4 f32s = 128 bits for SSE)
pub const SIMD_WIDTH: usize = 4;

/// Vectorized max operation for f32 arrays
#[inline]
pub fn simd_max_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            unsafe {
                simd_max_f32_sse(a, b, result, len);
                return;
            }
        }
    }

    // Scalar fallback
    for i in 0..len {
        result[i] = a[i].max(b[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn simd_max_f32_sse(a: &[f32], b: &[f32], result: &mut [f32], len: usize) {
    let chunks = len / SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let va = _mm_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm_loadu_ps(b.as_ptr().add(offset));
        let vmax = _mm_max_ps(va, vb);
        _mm_storeu_ps(result.as_mut_ptr().add(offset), vmax);
    }

    // Handle remainder
    for i in (chunks * SIMD_WIDTH)..len {
        result[i] = a[i].max(b[i]);
    }
}

/// Vectorized add operation for f32 arrays
#[inline]
pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            unsafe {
                simd_add_f32_sse(a, b, result, len);
                return;
            }
        }
    }

    for i in 0..len {
        result[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn simd_add_f32_sse(a: &[f32], b: &[f32], result: &mut [f32], len: usize) {
    let chunks = len / SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let va = _mm_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm_loadu_ps(b.as_ptr().add(offset));
        let vsum = _mm_add_ps(va, vb);
        _mm_storeu_ps(result.as_mut_ptr().add(offset), vsum);
    }

    for i in (chunks * SIMD_WIDTH)..len {
        result[i] = a[i] + b[i];
    }
}

/// Vectorized horizontal max (find max in array)
#[inline]
pub fn simd_horizontal_max(arr: &[f32]) -> f32 {
    if arr.is_empty() {
        return f32::NEG_INFINITY;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            unsafe {
                return simd_horizontal_max_sse(arr);
            }
        }
    }

    arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn simd_horizontal_max_sse(arr: &[f32]) -> f32 {
    let len = arr.len();
    let chunks = len / SIMD_WIDTH;

    let mut vmax = _mm_set1_ps(f32::NEG_INFINITY);

    for i in 0..chunks {
        let va = _mm_loadu_ps(arr.as_ptr().add(i * SIMD_WIDTH));
        vmax = _mm_max_ps(vmax, va);
    }

    // Horizontal max within vector
    let mut result = [0.0f32; SIMD_WIDTH];
    _mm_storeu_ps(result.as_mut_ptr(), vmax);
    let mut max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Handle remainder
    for i in (chunks * SIMD_WIDTH)..len {
        max_val = max_val.max(arr[i]);
    }

    max_val
}

/// Vectorized log-sum-exp for numerical stability
#[inline]
pub fn simd_logsumexp(arr: &[f32]) -> f32 {
    if arr.is_empty() {
        return f32::NEG_INFINITY;
    }

    let max_val = simd_horizontal_max(arr);
    if max_val == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }

    let sum: f32 = arr.iter()
        .filter(|&&x| x > f32::NEG_INFINITY)
        .map(|&x| (x - max_val).exp())
        .sum();

    max_val + sum.ln()
}

/// SIMD-accelerated MSV filter inner loop
pub fn simd_msv_inner(
    emissions: &[f32],  // Match emissions for position
    prev_scores: &[f32],
    scores: &mut [f32],
    m: usize,
) {
    // MSV: score[k] = max(prev_score[k-1] + emit[k], 0)

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            unsafe {
                simd_msv_inner_sse(emissions, prev_scores, scores, m);
                return;
            }
        }
    }

    // Scalar fallback
    scores[0] = 0.0;
    for k in 1..=m.min(emissions.len() - 1).min(prev_scores.len() - 1).min(scores.len() - 1) {
        let score = prev_scores[k - 1] + emissions[k];
        scores[k] = score.max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn simd_msv_inner_sse(
    emissions: &[f32],
    prev_scores: &[f32],
    scores: &mut [f32],
    m: usize,
) {
    let vzero = _mm_setzero_ps();
    let chunks = m / SIMD_WIDTH;

    scores[0] = 0.0;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH + 1;
        if offset + SIMD_WIDTH > m + 1 { break; }

        // Load prev_scores[k-1] (shifted by 1)
        let vprev = _mm_loadu_ps(prev_scores.as_ptr().add(offset - 1));
        let vemit = _mm_loadu_ps(emissions.as_ptr().add(offset));

        let vsum = _mm_add_ps(vprev, vemit);
        let vmax = _mm_max_ps(vsum, vzero);

        _mm_storeu_ps(scores.as_mut_ptr().add(offset), vmax);
    }

    // Handle remainder
    for k in (chunks * SIMD_WIDTH + 1)..=m.min(scores.len() - 1) {
        if k > 0 && k - 1 < prev_scores.len() && k < emissions.len() {
            let score = prev_scores[k - 1] + emissions[k];
            scores[k] = score.max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_max() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 2.0, 1.0, 9.0, 0.0, 7.0, 8.0];
        let mut result = vec![0.0; 8];

        simd_max_f32(&a, &b, &mut result);

        assert_eq!(result, vec![8.0, 7.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_simd_horizontal_max() {
        let arr = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0];
        assert_eq!(simd_horizontal_max(&arr), 9.0);
    }

    #[test]
    fn test_simd_logsumexp() {
        let arr = vec![1.0, 2.0, 3.0];
        let result = simd_logsumexp(&arr);
        // ln(e^1 + e^2 + e^3) ≈ 3.41
        assert!((result - 3.41).abs() < 0.1);
    }
}
