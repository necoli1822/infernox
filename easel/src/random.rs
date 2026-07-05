//! Portable Mersenne Twister random number generator.
//!
//! 1:1 port from esl_random.c - the default Easel RNG using Mersenne Twister MT19937.

const N: usize = 624;
const M: usize = 397;

/// Mersenne Twister random number generator.
///
/// The default random number generator uses the Mersenne Twister MT19937 algorithm.
/// It has a period of 2^19937-1 and equidistribution over 2^32 values.
pub struct EslRandom {
    /// State array
    mt: [u32; N],
    /// Current index into state array (mti)
    mti: usize,
    /// The seed used to initialize the generator
    seed: u32,
}

impl EslRandom {
    /// Create a new Mersenne Twister RNG with the given seed.
    ///
    /// If seed is 0, an arbitrary seed based on system time would be chosen.
    /// For reproducibility, use a non-zero seed.
    ///
    /// # Arguments
    /// * `seed` - The random seed (should be > 0 for reproducibility)
    pub fn new(seed: u32) -> Self {
        let mut rng = EslRandom {
            mt: [0u32; N],
            mti: 0,
            seed,
        };
        rng.mersenne_seed_table(seed);
        rng.mersenne_fill_table();
        rng
    }

    /// Reinitialize the RNG with a new seed.
    pub fn init(&mut self, seed: u32) {
        self.seed = seed;
        self.mersenne_seed_table(seed);
        self.mersenne_fill_table();
    }

    /// Get the seed used to initialize this RNG.
    pub fn get_seed(&self) -> u32 {
        self.seed
    }

    /// Generate a uniform random deviate on [0,1).
    ///
    /// Returns a double-precision value x where 0.0 <= x < 1.0.
    /// All 2^32-1 possible values are exactly representable.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let x = self.mersenne_twister();
        (x as f64) / 4294967296.0 // 2^32
    }

    /// Generate a uniform random 32-bit unsigned integer.
    ///
    /// Returns a value x where 0 <= x < 2^32.
    #[inline]
    pub fn random_uint32(&mut self) -> u32 {
        self.mersenne_twister()
    }

    /// Generate a uniform random integer on [0, n).
    ///
    /// Uses rejection sampling to ensure uniform distribution.
    ///
    /// # Arguments
    /// * `n` - Upper bound (exclusive), must be > 0 and < 2^31
    ///
    /// # Returns
    /// A uniformly distributed integer in [0, n)
    #[inline]
    pub fn random_int(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let factor = u32::MAX / n;
        loop {
            let u = self.random_uint32() / factor;
            if u < n {
                return u;
            }
        }
    }

    /// Generate a uniform positive deviate on (0,1).
    ///
    /// Same as random() but guarantees 0 < x < 1.
    #[inline]
    pub fn uniform_positive(&mut self) -> f64 {
        loop {
            let x = self.random();
            if x != 0.0 {
                return x;
            }
        }
    }

    /// Initialize the state table from a seed using a Knuth LCG.
    fn mersenne_seed_table(&mut self, seed: u32) {
        self.seed = seed;
        self.mt[0] = seed;
        for z in 1..N {
            self.mt[z] = 69069u32.wrapping_mul(self.mt[z - 1]);
        }
    }

    /// Refill the table with 624 new random numbers.
    fn mersenne_fill_table(&mut self) {
        const MAG01: [u32; 2] = [0x0, 0x9908b0df];

        // First loop: z = 0..226 (N-M = 624-397 = 227)
        for z in 0..227 {
            let y = (self.mt[z] & 0x80000000) | (self.mt[z + 1] & 0x7fffffff);
            self.mt[z] = self.mt[z + M] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }

        // Second loop: z = 227..622
        for z in 227..623 {
            let y = (self.mt[z] & 0x80000000) | (self.mt[z + 1] & 0x7fffffff);
            self.mt[z] = self.mt[z - 227] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }

        // Final element: z = 623
        let y = (self.mt[623] & 0x80000000) | (self.mt[0] & 0x7fffffff);
        self.mt[623] = self.mt[396] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];

        self.mti = 0;
    }

    /// Generate the next random number from the Mersenne Twister.
    #[inline]
    fn mersenne_twister(&mut self) -> u32 {
        if self.mti >= N {
            self.mersenne_fill_table();
        }

        let mut x = self.mt[self.mti];
        self.mti += 1;

        // Tempering transformations
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680;
        x ^= (x << 15) & 0xefc60000;
        x ^= x >> 18;

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_sequence() {
        // Same seed should produce same sequence
        let mut rng1 = EslRandom::new(42);
        let mut rng2 = EslRandom::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.random_uint32(), rng2.random_uint32());
        }
    }

    #[test]
    fn test_random_range() {
        let mut rng = EslRandom::new(42);

        // All values should be in [0, 1)
        for _ in 0..1000 {
            let x = rng.random();
            assert!(x >= 0.0 && x < 1.0, "Value {} out of range [0,1)", x);
        }
    }

    #[test]
    fn test_random_int_range() {
        let mut rng = EslRandom::new(42);

        // All values should be in [0, n)
        for _ in 0..1000 {
            let x = rng.random_int(10);
            assert!(x < 10, "Value {} out of range [0,10)", x);
        }
    }

    #[test]
    fn test_uniform_positive_nonzero() {
        let mut rng = EslRandom::new(42);

        // All values should be > 0
        for _ in 0..1000 {
            let x = rng.uniform_positive();
            assert!(x > 0.0 && x < 1.0, "Value {} not in (0,1)", x);
        }
    }

    #[test]
    fn test_seed_retrieval() {
        let rng = EslRandom::new(12345);
        assert_eq!(rng.get_seed(), 12345);
    }
}
