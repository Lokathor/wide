//! Defines APIs that only exists for some SIMD types inconsistently.
//!
//! Such APIS are usually either:
//!
//! - Deprecated and replaced by consistent APIs
//!
//! - Missing for certain types and could be moved from here (and renamed if
//!   needed) once they are implemented for all applicable types
//!
//! - Cannot be made consistent yet due to language limitations
//!
//! This module does not implement any functionality. All functions here
//! simply delegate to other functions, either backend functions or other public
//! functions.

use crate::{
    f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i8x16, i8x32, i16x8, i16x16, i16x32, i32x4, i32x8,
    i32x16, i64x2, i64x4, i64x8, u8x16, u8x32, u16x8, u16x16, u32x4, u32x8, u32x16, u64x2, u64x4,
    u64x8,
};

/// The following functionality exists only for [`f32x4`], or only for
/// particular types inconsistently.
impl f32x4 {
    /// Returns a SIMD vector whose elements are selected from `self` using
    /// constant indices.
    ///
    /// If an index is out of bounds, compilation fails.
    ///
    /// Equivalent to `[self[I0], self[I1], ..., self[I{N-1}]`.
    #[inline]
    #[must_use]
    pub fn shuffle_consts<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        self,
    ) -> Self {
        Self::from_bits(self.to_bits().shuffle_consts::<I0, I1, I2, I3>())
    }

    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    pub fn from_i32x4(v: i32x4) -> Self {
        Self::from_i32x4_backend(v)
    }

    /// Returns a [mask] that checks if each element has a negative sign,
    /// including `-0.0`, NaNs with negative sign bit and negative infinity.
    ///
    /// Note that this function has a misleading name. If the sign bit is set, the
    /// result has all bits set, not just the sign bit. This function has been
    /// renamed to [`is_sign_negative`].
    ///
    /// [mask]: crate#masks
    /// [`is_sign_negative`]: Self::is_sign_negative
    #[inline]
    #[must_use]
    #[deprecated(since = "1.4.0", note = "renamed to `is_sign_negative`")]
    pub fn sign_bit(self) -> Self {
        self.is_sign_negative()
    }
}

/// The following functionality exists only for [`f32x8`], or only for
/// particular types inconsistently.
impl f32x8 {
    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    pub fn from_i32x8(v: i32x8) -> Self {
        Self::from_i32x8_backend(v)
    }

    /// Returns a [mask] that checks if each element has a negative sign,
    /// including `-0.0`, NaNs with negative sign bit and negative infinity.
    ///
    /// Note that this function has a misleading name. If the sign bit is set, the
    /// result has all bits set, not just the sign bit. This function has been
    /// renamed to [`is_sign_negative`].
    ///
    /// [mask]: crate#masks
    /// [`is_sign_negative`]: Self::is_sign_negative
    #[inline]
    #[must_use]
    #[deprecated(since = "1.4.0", note = "renamed to `is_sign_negative`")]
    pub fn sign_bit(self) -> Self {
        self.is_sign_negative()
    }
}

/// The following functionality exists only for [`f32x16`], or only for
/// particular types inconsistently.
impl f32x16 {
    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    pub fn from_i32x16(v: i32x16) -> Self {
        Self::from_i32x16_backend(v)
    }

    /// Returns a [mask] that checks if each element has a negative sign,
    /// including `-0.0`, NaNs with negative sign bit and negative infinity.
    ///
    /// Note that this function has a misleading name. If the sign bit is set, the
    /// result has all bits set, not just the sign bit. This function has been
    /// renamed to [`is_sign_negative`].
    ///
    /// [mask]: crate#masks
    /// [`is_sign_negative`]: Self::is_sign_negative
    #[inline]
    #[must_use]
    #[deprecated(since = "1.4.0", note = "renamed to `is_sign_negative`")]
    pub fn sign_bit(self) -> Self {
        self.is_sign_negative()
    }
}

/// The following functionality exists only for [`f64x2`], or only for
/// particular types inconsistently.
impl f64x2 {
    /// Converts the lower two elements of `v` from [`i32`] to [`f64`], dropping
    /// the higher two elements.
    #[inline]
    pub fn from_i32x4_lower2(v: i32x4) -> Self {
        Self::from_i32x4_lower2_backend(v)
    }
}

/// The following functionality exists only for [`f64x8`], or only for
/// particular types inconsistently.
impl f64x8 {
    /// Converts each element from [`i32`] to [`f64`].
    #[inline]
    pub fn from_i32x8(v: i32x8) -> Self {
        Self::from_i32x8_backend(v)
    }
}

/// The following functionality exists only for [`f64x4`], or only for
/// particular types inconsistently.
impl f64x4 {
    /// Converts each element from [`i32`] to [`f64`].
    #[inline]
    pub fn from_i32x4(v: i32x4) -> Self {
        Self::from_i32x4_backend(v)
    }
}

/// The following functionality exists only for [`i8x16`], or only for
/// particular types inconsistently.
impl i8x16 {
    /// Converts each element from [`i16`] to [`i8`], saturating out of range
    /// values.
    #[inline]
    #[must_use]
    pub fn from_i16x16_saturate(v: i16x16) -> i8x16 {
        Self::from_i16x16_saturate_backend(v)
    }

    /// Converts each element from [`i16`] to [`i8`], truncating out of range
    /// values (behaves like [`as`] casting).
    ///
    /// [`as`]: https://doc.rust-lang.org/stable/reference/expressions/operator-expr.html#r-expr.as.numeric
    #[inline]
    #[must_use]
    pub fn from_i16x16_truncate(v: i16x16) -> i8x16 {
        Self::from_i16x16_truncate_backend(v)
    }

    /// Converts a slice to a SIMD vector, ignoring elements beyond the first 16.
    ///
    /// # Panics
    ///
    /// Panics if `input` has less than 16 elements.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn from_slice_unaligned(input: &[i8]) -> Self {
        Self::from_slice_unaligned_backend(input)
    }

    /// Returns a new vector where each element is based on the index values in
    /// `rhs`.
    ///
    /// * Index values in the range `[0, 15]` select the i-th element of `self`.
    /// * Index values that are out of range will cause that output lane to be
    ///   `0`.
    ///
    /// This function has been deprecated and replaced with [`shuffle_zeroing`].
    ///
    /// [`shuffle_zeroing`]: Self::shuffle_zeroing
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle_zeroing`")]
    pub fn swizzle(self, rhs: i8x16) -> i8x16 {
        self.shuffle_zeroing(rhs.cast_unsigned())
    }

    /// Works like [`swizzle`](Self::swizzle) with the following additional
    /// details
    ///
    /// * Indices in the range `[0, 15]` will select the i-th element of `self`.
    /// * If the high bit of any index is set (meaning that the index is
    ///   negative), then the corresponding output lane is guaranteed to be zero.
    /// * Otherwise the output lane is either `0` or `self[rhs[i] % 16]`,
    ///   depending on the implementation.
    ///
    /// This function has been deprecated and replaced with [`shuffle`].
    ///
    /// [`shuffle`]: Self::shuffle
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle`")]
    pub fn swizzle_relaxed(self, rhs: i8x16) -> i8x16 {
        self.shuffle(rhs.cast_unsigned())
    }
}

/// The following functionality exists only for [`i8x32`], or only for
/// particular types inconsistently.
impl i8x32 {
    /// Returns a new vector with lanes selected from the lanes of the first input
    /// vector a specified in the second input vector `rhs`.
    /// The indices i in range `[0, 15]` select the i-th element of `self`. For
    /// indices outside of the range the resulting lane is `0`.
    ///
    /// This note that is the equivalent of two parallel swizzle operations on the
    /// two halves of the vector, and the indexes each refer to the
    /// corresponding half.
    #[inline]
    pub fn swizzle_half(self, rhs: i8x32) -> i8x32 {
        self.swizzle_half_backend(rhs)
    }

    /// Indices in the range `[0, 15]` will select the i-th element of `self`. If
    /// the high bit of any element of `rhs` is set (negative) then the
    /// corresponding output lane is guaranteed to be zero. Otherwise if the
    /// element of `rhs` is within the range `[32, 127]` then the output lane is
    /// either `0` or `self[rhs[i] % 16]` depending on the implementation.
    ///
    /// This is the equivalent to two parallel swizzle operations on the two
    /// halves of the vector, and the indexes each refer to their corresponding
    /// half.
    #[inline]
    pub fn swizzle_half_relaxed(self, rhs: i8x32) -> i8x32 {
        self.swizzle_half_relaxed_backend(rhs)
    }

    /// Full 32-entry byte table lookup.
    ///
    /// * An index (interpreted as unsigned) in `[0, 31]` selects `self[index]`.
    /// * Any index `>= 32` (including negative `i8` values) yields `0`.
    ///
    /// Unlike [`swizzle_half`](Self::swizzle_half), indices address the entire
    /// 32-byte vector, not just their own 16-byte half.
    ///
    /// This function has been deprecated and replaced with [`shuffle_zeroing`].
    ///
    /// [`shuffle_zeroing`]: Self::shuffle_zeroing
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle_zeroing`")]
    pub fn swizzle(self, rhs: i8x32) -> i8x32 {
        self.shuffle_zeroing(rhs.cast_unsigned())
    }

    /// Like [`swizzle`](Self::swizzle), but out-of-range indices (unsigned
    /// `>= 32`) yield an implementation-defined result (`0` or `self[index %
    /// 32]`). Prefer this when you know all indices are in range; it can be
    /// cheaper.
    ///
    /// This function has been deprecated and replaced with [`shuffle`].
    ///
    /// [`shuffle`]: Self::shuffle
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle`")]
    pub fn swizzle_relaxed(self, rhs: i8x32) -> i8x32 {
        self.shuffle(rhs.cast_unsigned())
    }
}

/// The following functionality exists only for [`i16x8`], or only for
/// particular types inconsistently.
impl i16x8 {
    /// Converts the lower eight elements of `u` from [`u8`] to [`i16`], dropping
    /// the higher eight elements.
    #[inline]
    #[must_use]
    pub fn from_u8x16_low(u: u8x16) -> Self {
        Self::from_u8x16_low_backend(u)
    }

    /// Converts the higher eight elements of `u` from [`u8`] to [`i16`], dropping
    /// the lower eight elements.
    #[inline]
    #[must_use]
    pub fn from_u8x16_high(u: u8x16) -> Self {
        Self::from_u8x16_high_backend(u)
    }

    /// Converts each element from [`i32`] to [`i16`], saturating out of range
    /// values.
    #[inline]
    #[must_use]
    pub fn from_i32x8_saturate(v: i32x8) -> Self {
        Self::from_i32x8_saturate_backend(v)
    }

    /// Converts each element from [`i32`] to [`i16`], truncating out of range
    /// values (behaves like [`as`] casting).
    ///
    /// [`as`]: https://doc.rust-lang.org/stable/reference/expressions/operator-expr.html#r-expr.as.numeric
    #[inline]
    #[must_use]
    pub fn from_i32x8_truncate(v: i32x8) -> Self {
        Self::from_i32x8_truncate_backend(v)
    }

    /// Converts a slice to a SIMD vector, ignoring elements beyond the first 8.
    ///
    /// # Panics
    ///
    /// Panics if `input` has less than 8 elements.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn from_slice_unaligned(input: &[i16]) -> Self {
        Self::from_slice_unaligned_backend(input)
    }

    /// Partially computes the dot product.
    ///
    /// First this multiplies the input 16-bit integers, producing intermediate
    /// 32-bit integers. Then this horizontally adds adjacent pairs, resulting in
    /// four 32-bit integers.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> i32x4 {
        self.dot_backend(rhs)
    }

    /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
    /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
    /// and `1`. This corresponds to the following instructions:
    /// - `vqrdmulhq_s16` instruction on neon
    /// - `i16x8_q15mulr_sat` on simd128
    /// - `_mm_mulhrs_epi16` on ssse3
    /// - emulated via `mul_i16_*` on sse2
    #[inline]
    #[must_use]
    pub fn mul_scale_round(self, rhs: Self) -> Self {
        self.mul_scale_round_backend(rhs)
    }

    #[inline]
    #[must_use]
    /// Multiply and scale, equivalent to `((self * rhs) + 0x4000) >> 15` on each
    /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
    /// and `1`. This corresponds to the following instructions:
    /// - `vqrdmulhq_n_s16` instruction on neon
    /// - `i16x8_q15mulr_sat` on simd128
    /// - `_mm_mulhrs_epi16` on ssse3
    /// - emulated via `mul_i16_*` on sse2
    pub fn mul_scale_round_n(self, rhs: i16) -> Self {
        self.mul_scale_round_n_backend(rhs)
    }

    /// Widening multiplication. Computes `self * rhs`, widening to a SIMD
    /// vector of larger integers.
    ///
    /// The returned value is always exact and can never overflow.
    ///
    /// This function has been renamed to [`widening_mul`].
    ///
    /// [`widening_mul`]: Self::widening_mul
    #[inline]
    #[must_use]
    #[deprecated(since = "1.6.0", note = "renamed to `widening_mul`")]
    pub fn mul_widen(self, rhs: Self) -> i32x8 {
        self.widening_mul(rhs)
    }
}

/// The following functionality exists only for [`i16x16`], or only for
/// particular types inconsistently.
impl i16x16 {
    /// Converts each element from [`i8`] to [`i16`].
    #[inline]
    #[must_use]
    pub fn from_i8x16(v: i8x16) -> Self {
        Self::from_i8x16_backend(v)
    }

    /// Partially computes the dot product.
    ///
    /// First this multiplies the input 16-bit integers, producing intermediate
    /// 32-bit integers. Then this horizontally adds adjacent pairs, resulting in
    /// eight 32-bit integers.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> i32x8 {
        self.dot_backend(rhs)
    }

    /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
    /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
    /// and `1`. This corresponds to the following instructions:
    /// - `vqrdmulhq_n_s16` instruction on neon
    /// - `i16x8_q15mulr_sat` on simd128
    /// - `_mm256_mulhrs_epi16` on avx2
    /// - emulated via `mul_i16_*` on sse2
    #[inline]
    #[must_use]
    pub fn mul_scale_round(self, rhs: Self) -> Self {
        self.mul_scale_round_backend(rhs)
    }

    /// Multiply and scale equivalent to `((self * rhs) + 0x4000) >> 15` on each
    /// lane, effectively multiplying by a 16 bit fixed point number between `-1`
    /// and `1`. This corresponds to the following instructions:
    /// - `vqrdmulhq_n_s16` instruction on neon
    /// - `i16x8_q15mulr_sat` on simd128
    /// - `_mm256_mulhrs_epi16` on avx2
    /// - emulated via `mul_i16_*` on sse2
    #[inline]
    #[must_use]
    pub fn mul_scale_round_n(self, rhs: i16) -> Self {
        self.mul_scale_round_n_backend(rhs)
    }
}

/// The following functionality exists only for [`i16x32`], or only for
/// particular types inconsistently.
impl i16x32 {
    /// Converts each element from [`i8`] to [`i16`].
    #[inline]
    #[must_use]
    pub fn from_i8x32(v: i8x32) -> Self {
        Self::from_i8x32_backend(v)
    }

    /// Partially computes the dot product.
    ///
    /// First this multiplies the input 16-bit integers, producing intermediate
    /// 32-bit integers. Then this horizontally adds adjacent pairs, resulting in
    /// sixteen 32-bit integers.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> i32x16 {
        self.dot_backend(rhs)
    }
}

/// The following functionality exists only for [`i32x4`], or only for
/// particular types inconsistently.
impl i32x4 {
    /// Returns a SIMD vector whose elements are selected from `self` using
    /// constant indices.
    ///
    /// If an index is out of bounds, compilation fails.
    ///
    /// Equivalent to `[self[I0], self[I1], ..., self[I{N-1}]`.
    #[inline]
    #[must_use]
    pub fn shuffle_consts<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        self,
    ) -> Self {
        self.cast_unsigned()
            .shuffle_consts::<I0, I1, I2, I3>()
            .cast_signed()
    }

    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f32x4 {
        self.round_float_backend()
    }

    /// Widening multiplication. Computes `self * rhs`, widening to a SIMD
    /// vector of larger integers.
    ///
    /// The returned value is always exact and can never overflow.
    ///
    /// This function has been renamed to [`widening_mul`].
    ///
    /// [`widening_mul`]: Self::widening_mul
    #[inline]
    #[must_use]
    #[deprecated(since = "1.6.0", note = "renamed to `widening_mul`")]
    pub fn mul_widen(self, rhs: Self) -> i64x4 {
        self.widening_mul(rhs)
    }
}

/// The following functionality exists only for [`i32x8`], or only for
/// particular types inconsistently.
impl i32x8 {
    /// Converts each element from [`i16`] to [`i32`].
    #[inline]
    #[must_use]
    pub fn from_i16x8(v: i16x8) -> Self {
        Self::from_i16x8_backend(v)
    }

    /// Converts each element from [`u16`] to [`i32`].
    #[inline]
    #[must_use]
    pub fn from_u16x8(v: u16x8) -> Self {
        Self::from_u16x8_backend(v)
    }

    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f32x8 {
        self.round_float_backend()
    }
}

/// The following functionality exists only for [`i32x16`], or only for
/// particular types inconsistently.
impl i32x16 {
    /// Converts each element from [`i16`] to [`i32`].
    #[inline]
    #[must_use]
    pub fn from_i16x16(v: i16x16) -> Self {
        Self::from_i16x16_backend(v)
    }

    /// Converts each element from [`i32`] to [`f32`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f32x16 {
        self.round_float_backend()
    }
}

/// The following functionality exists only for [`i64x2`], or only for
/// particular types inconsistently.
impl i64x2 {
    /// Converts each element from [`i64`] to [`f64`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f64x2 {
        self.round_float_backend()
    }
}

/// The following functionality exists only for [`i64x4`], or only for
/// particular types inconsistently.
impl i64x4 {
    /// Converts each element from [`i64`] to [`f64`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f64x4 {
        self.round_float_backend()
    }
}

/// The following functionality exists only for [`i64x8`], or only for
/// particular types inconsistently.
impl i64x8 {
    /// Converts each element from [`i64`] to [`f64`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f64x8 {
        self.round_float_backend()
    }
}

/// The following functionality exists only for [`u8x16`], or only for
/// particular types inconsistently.
impl u8x16 {
    /// Treats two [`i16x8`] values as a single [`i16x16`] value, then converts
    /// each element from [`i16`] to [`u8`], saturating out of range values.
    #[inline]
    #[must_use]
    pub fn narrow_i16x8(lhs: i16x8, rhs: i16x8) -> Self {
        Self::narrow_i16x8_backend(lhs, rhs)
    }

    /// Interleaves the lower halfs of two SIMD vectors, discarding the higher
    /// halfs.
    ///
    /// Equivalent to `[self[0], other[0], self[1], other[1], ...]`.
    ///
    /// This function has been deprecated and renamed to [`interleave_low`].
    ///
    /// [`interleave_low`]: Self::interleave_low
    #[inline]
    #[must_use]
    #[deprecated(since = "1.8.0", note = "renamed to `interleave_low`")]
    pub fn unpack_low(lhs: u8x16, rhs: u8x16) -> u8x16 {
        lhs.interleave_low(rhs)
    }

    /// Interleaves the higher halfs of two SIMD vectors, discarding the lower
    /// halfs.
    ///
    /// Equivalent to
    /// `[self[N / 2], other[N / 2], self[N / 2 + 1], other[N / 2 + 1], ...]`.
    ///
    /// This function has been deprecated and renamed to [`interleave_high`].
    ///
    /// [`interleave_high`]: Self::interleave_high
    #[inline]
    #[must_use]
    #[deprecated(since = "1.8.0", note = "renamed to `interleave_high`")]
    pub fn unpack_high(lhs: u8x16, rhs: u8x16) -> u8x16 {
        lhs.interleave_high(rhs)
    }

    /// Returns a new vector where each element is based on the index values in
    /// `rhs`.
    ///
    /// * Index values in the range `[0, 15]` select the i-th element of `self`.
    /// * Index values that are out of range will cause that output lane to be
    ///   `0`.
    ///
    /// This function has been deprecated and replaced with [`shuffle_zeroing`].
    ///
    /// [`shuffle_zeroing`]: Self::shuffle_zeroing
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle_zeroing`")]
    pub fn swizzle(self, rhs: i8x16) -> i8x16 {
        self.shuffle_zeroing(rhs.cast_unsigned()).cast_signed()
    }

    /// Works like [`swizzle`](Self::swizzle) with the following additional
    /// details
    ///
    /// * Indices in the range `[0, 15]` will select the i-th element of `self`.
    /// * If the high bit of any index is set (meaning that the index is
    ///   negative), then the corresponding output lane is guaranteed to be zero.
    /// * Otherwise the output lane is either `0` or `self[rhs[i] % 16]`,
    ///   depending on the implementation.
    ///
    /// This function has been deprecated and replaced with [`shuffle`].
    ///
    /// [`shuffle`]: Self::shuffle
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle`")]
    pub fn swizzle_relaxed(self, rhs: u8x16) -> u8x16 {
        self.shuffle(rhs)
    }
}

/// The following functionality exists only for [`u8x32`], or only for
/// particular types inconsistently.
impl u8x32 {
    /// Returns a new vector with lanes selected from the lanes of the first input
    /// vector a specified in the second input vector `rhs`.
    /// The indices i in range `[0, 15]` select the i-th element of `self`. For
    /// indices outside of the range the resulting lane is `0`.
    ///
    /// This note that is the equivalent of two parallel swizzle operations on the
    /// two halves of the vector, and the indexes each refer to the
    /// corresponding half.
    #[inline]
    pub fn swizzle_half(self, rhs: i8x32) -> i8x32 {
        self.swizzle_half_backend(rhs)
    }

    /// Indices in the range `[0, 15]` will select the i-th element of `self`. If
    /// the high bit of any element of `rhs` is set (negative) then the
    /// corresponding output lane is guaranteed to be zero. Otherwise if the
    /// element of `rhs` is within the range `[32, 127]` then the output lane is
    /// either `0` or `self[rhs[i] % 16]` depending on the implementation.
    ///
    /// This is the equivalent to two parallel swizzle operations on the two
    /// halves of the vector, and the indexes each refer to their corresponding
    /// half.
    #[inline]
    pub fn swizzle_half_relaxed(self, rhs: u8x32) -> u8x32 {
        self.swizzle_half_relaxed_backend(rhs)
    }

    /// Full 32-entry byte table lookup. An index in `[0, 31]` selects
    /// `self[index]`; any index `>= 32` yields `0`.
    ///
    /// This function has been deprecated and replaced with [`shuffle_zeroing`].
    ///
    /// [`shuffle_zeroing`]: Self::shuffle_zeroing
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle_zeroing`")]
    pub fn swizzle(self, rhs: u8x32) -> u8x32 {
        self.shuffle_zeroing(rhs)
    }

    /// Like [`swizzle`](Self::swizzle), but out-of-range indices yield an
    /// implementation-defined result (`0` or `self[index % 32]`).
    ///
    /// This function has been deprecated and replaced with [`shuffle`].
    ///
    /// [`shuffle`]: Self::shuffle
    #[inline]
    #[deprecated(since = "1.7.0", note = "replaced with `shuffle`")]
    pub fn swizzle_relaxed(self, rhs: u8x32) -> u8x32 {
        self.shuffle(rhs)
    }
}

/// The following functionality exists only for [`u16x8`], or only for
/// particular types inconsistently.
impl u16x8 {
    /// Converts the lower eight elements of `u` from [`u8`] to [`u16`], dropping
    /// the higher eight elements.
    #[inline]
    #[must_use]
    pub fn from_u8x16_low(u: u8x16) -> Self {
        Self::from_u8x16_low_backend(u)
    }

    /// Converts the higher eight elements of `u` from [`u8`] to [`u16`], dropping
    /// the lower eight elements.
    #[inline]
    #[must_use]
    pub fn from_u8x16_high(u: u8x16) -> Self {
        Self::from_u8x16_high_backend(u)
    }

    /// Widening multiplication. Computes `self * rhs`, widening to a SIMD
    /// vector of larger integers.
    ///
    /// The returned value is always exact and can never overflow.
    ///
    /// This function has been renamed to [`widening_mul`].
    ///
    /// [`widening_mul`]: Self::widening_mul
    #[inline]
    #[must_use]
    #[deprecated(since = "1.6.0", note = "renamed to `widening_mul`")]
    pub fn mul_widen(self, rhs: Self) -> u32x8 {
        self.widening_mul(rhs)
    }
}

/// The following functionality exists only for [`u32x4`], or only for
/// particular types inconsistently.
impl u32x4 {
    /// Returns a SIMD vector whose elements are selected from `self` using
    /// constant indices.
    ///
    /// If an index is out of bounds, compilation fails.
    ///
    /// Equivalent to `[self[I0], self[I1], ..., self[I{N-1}]`.
    #[inline]
    #[must_use]
    pub fn shuffle_consts<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        self,
    ) -> Self {
        const {
            assert!(I0 < 4);
            assert!(I1 < 4);
            assert!(I2 < 4);
            assert!(I3 < 4);
        }

        self.shuffle_consts_backend::<I0, I1, I2, I3>()
    }

    /// Widening multiplication. Computes `self * rhs`, widening to a SIMD
    /// vector of larger integers.
    ///
    /// The returned value is always exact and can never overflow.
    ///
    /// This function has been renamed to [`widening_mul`].
    ///
    /// [`widening_mul`]: Self::widening_mul
    #[inline]
    #[must_use]
    #[deprecated(since = "1.6.0", note = "renamed to `widening_mul`")]
    pub fn mul_widen(self, rhs: Self) -> u64x4 {
        self.widening_mul(rhs)
    }

    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=32`.
    ///
    /// There is no IFMA equivalent at this width, and below 17 bits no widening
    /// multiply is needed either: the whole product fits a lane, so the ordinary
    /// lane multiply already yields both halves.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_lo_backend::<W>(a, b)
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=32`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_hi_backend::<W>(a, b)
    }
}

/// The following functionality exists only for [`u32x8`], or only for
/// particular types inconsistently.
impl u32x8 {
    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=32`.
    ///
    /// There is no IFMA equivalent at this width, and below 17 bits no widening
    /// multiply is needed either: the whole product fits a lane, so the ordinary
    /// lane multiply already yields both halves.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_lo_backend::<W>(a, b)
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=32`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_hi_backend::<W>(a, b)
    }
}

/// The following functionality exists only for [`u32x16`], or only for
/// particular types inconsistently.
impl u32x16 {
    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=32`.
    ///
    /// There is no IFMA equivalent at this width, and below 17 bits no widening
    /// multiply is needed either: the whole product fits a lane, so the ordinary
    /// lane multiply already yields both halves.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_lo_backend::<W>(a, b)
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=32`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_hi_backend::<W>(a, b)
    }
}

/// The following functionality exists only for [`u64x2`], or only for
/// particular types inconsistently.
impl u64x2 {
    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_lo_backend::<W>(a, b)
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_hi_backend::<W>(a, b)
    }
}

/// The following functionality exists only for [`u64x4`], or only for
/// particular types inconsistently.
impl u64x4 {
    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_lo_backend::<W>(a, b)
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_hi_backend::<W>(a, b)
    }
}

/// The following functionality exists only for [`u64x8`], or only for
/// particular types inconsistently.
impl u64x8 {
    /// `self + ((a * b) mod 2^W)`, reading only the low `W` bits of each lane of
    /// `a` and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_lo<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_lo_backend::<W>(a, b)
    }

    /// `self + ((a * b) >> W)`, reading only the low `W` bits of each lane of `a`
    /// and `b`. `W` must be in `1..=64`.
    #[inline]
    #[must_use]
    pub fn add_mul_hi<const W: u32>(self, a: Self, b: Self) -> Self {
        self.add_mul_hi_backend::<W>(a, b)
    }
}

impl From<i32x4> for f64x2 {
    /// Converts the lower two `i32` lanes to two `f64` lanes (and dropping the
    /// higher two `i32` lanes)
    #[inline]
    fn from(v: i32x4) -> Self {
        Self::from_i32x4_lower2(v)
    }
}

impl From<i32x4> for f64x4 {
    #[inline]
    fn from(v: i32x4) -> Self {
        Self::from_i32x4(v)
    }
}

impl From<i32x8> for f64x8 {
    #[inline]
    fn from(v: i32x8) -> Self {
        Self::from_i32x8(v)
    }
}

impl From<i8x16> for i16x16 {
    /// widen with sign extend from i8 to i16
    #[inline]
    fn from(i: i8x16) -> Self {
        i16x16::from_i8x16(i)
    }
}

impl From<u8x16> for i16x16 {
    /// widen with zero extend from u8 to i16
    #[inline]
    fn from(i: u8x16) -> Self {
        u16x16::from(i).cast_signed()
    }
}

impl From<i8x32> for i16x32 {
    /// widen with sign extend from i8 to i16
    #[inline]
    fn from(i: i8x32) -> Self {
        i16x32::from_i8x32(i)
    }
}

impl From<i16x8> for i32x8 {
    #[inline]
    fn from(value: i16x8) -> Self {
        i32x8::from_i16x8(value)
    }
}

impl From<i16x16> for i32x16 {
    #[inline]
    fn from(value: i16x16) -> Self {
        i32x16::from_i16x16(value)
    }
}

impl From<u8x16> for u16x16 {
    /// widens and sign extends to u16x16
    #[inline]
    fn from(v: u8x16) -> Self {
        Self::from_u8x16_backend(v)
    }
}

impl From<u16x8> for u32x8 {
    /// widens and zero extends to u32x8
    #[inline]
    fn from(v: u16x8) -> Self {
        Self::from_u16x8_backend(v)
    }
}

impl From<u16x16> for u32x16 {
    /// Widens and zero-extends each u16 lane to u32
    #[inline]
    fn from(v: u16x16) -> Self {
        Self::from_u16x16_backend(v)
    }
}
