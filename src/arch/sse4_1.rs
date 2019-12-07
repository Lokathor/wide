#![cfg(target_feature = "sse")]
#![cfg(target_feature = "sse2")]
#![cfg(target_feature = "sse3")]
#![cfg(target_feature = "ssse3")]
#![cfg(target_feature = "sse4.1")]

use super::*;

/// # SSE4.1 Operations
impl m128 {
  /// Blend values in `self` and `rhs` using a variable `mask`.
  ///
  /// ```txt
  /// for i in 0..4 {
  ///   out[i] = if sign_bit(mask[i]) {
  ///     rhs[i]
  ///   } else {
  ///     self[i]
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn blend_var(self, rhs: Self, mask: Self) -> Self {
    Self(unsafe { _mm_blendv_ps(self.0, rhs.0, mask.0) })
  }

  /// Lanewise "ceiling" operation (round to positive infinity)
  #[inline(always)]
  #[must_use]
  pub fn ceil(self) -> Self {
    Self(unsafe { _mm_ceil_ps(self.0) })
  }

  /// Does "ceiling" on `rhs[0]`, other lanes `self`.
  #[inline(always)]
  #[must_use]
  pub fn ceil_rhs0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_ceil_ss(self.0, rhs.0) })
  }

  /// Lanewise "floor" operation (round to negative infinity)
  #[inline(always)]
  #[must_use]
  pub fn floor(self) -> Self {
    Self(unsafe { _mm_floor_ps(self.0) })
  }

  /// Does "floor" on `rhs[0]`, other lanes `self`.
  #[inline(always)]
  #[must_use]
  pub fn floor_rhs0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_floor_ss(self.0, rhs.0) })
  }

  /// Round to the nearest whole number.
  #[inline]
  #[must_use]
  pub fn round_nearest(self) -> Self {
    Self(unsafe {
      _mm_round_ps(self.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
    })
  }

  /// Truncate the fractional portion.
  #[inline]
  #[must_use]
  pub fn truncate(self) -> Self {
    Self(unsafe {
      _mm_round_ps(self.0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
    })
  }
}

/// # SSE4.1 Operations
impl m128d {
  /// Blend values in `self` and `rhs` using a variable `mask`.
  ///
  /// ```txt
  /// for i in 0..2 {
  ///   out[i] = if sign_bit(mask[i]) {
  ///     rhs[i]
  ///   } else {
  ///     self[i]
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn blend_var(self, rhs: Self, mask: Self) -> Self {
    Self(unsafe { _mm_blendv_pd(self.0, rhs.0, mask.0) })
  }

  /// Lanewise "ceiling" operation (round to positive infinity)
  #[inline(always)]
  #[must_use]
  pub fn ceil(self) -> Self {
    Self(unsafe { _mm_ceil_pd(self.0) })
  }

  /// Does "ceiling" on `rhs[0]`, other lanes `self`.
  #[inline(always)]
  #[must_use]
  pub fn ceil_rhs0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_ceil_sd(self.0, rhs.0) })
  }

  /// Lanewise "floor" operation (round to negative infinity)
  #[inline(always)]
  #[must_use]
  pub fn floor(self) -> Self {
    Self(unsafe { _mm_floor_pd(self.0) })
  }

  /// Does "floor" on `rhs[0]`, other lanes `self`.
  #[inline(always)]
  #[must_use]
  pub fn floor_rhs0(self, rhs: Self) -> Self {
    Self(unsafe { _mm_floor_sd(self.0, rhs.0) })
  }
}

/// # SSE4.1 Operations
impl m128i {
  /// `i8`: blend values in `self` and `rhs` using a variable `mask`.
  ///
  /// ```txt
  /// for i in 0..16 {
  ///   out[i] = if mask[i] < 0 {
  ///     rhs[i]
  ///   } else {
  ///     self[i]
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn blend_var_i8(self, rhs: Self, mask: Self) -> Self {
    Self(unsafe { _mm_blendv_epi8(self.0, rhs.0, mask.0) })
  }

  /// Lanewise `i64` equality comparison, bool-ish output.
  #[inline(always)]
  #[must_use]
  pub fn cmp_eq_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpeq_epi64(self.0, rhs.0) })
  }

  /// Sign extend the lower four `i16` values into `i32`.
  #[inline(always)]
  #[must_use]
  pub fn sign_extend_i16_i32(self) -> Self {
    Self(unsafe { _mm_cvtepi16_epi32(self.0) })
  }

  /// Sign extend the lower two `i16` values into `i64`.
  #[inline(always)]
  #[must_use]
  pub fn sign_extend_i16_i64(self) -> Self {
    Self(unsafe { _mm_cvtepi16_epi64(self.0) })
  }

  /// Sign extend the lower two `i32` values into `i64`.
  #[inline(always)]
  #[must_use]
  pub fn sign_extend_i32_i64(self) -> Self {
    Self(unsafe { _mm_cvtepi32_epi64(self.0) })
  }

  /// Sign extend the lower eight `i8` values into `i16`.
  #[inline(always)]
  #[must_use]
  pub fn sign_extend_i8_i16(self) -> Self {
    Self(unsafe { _mm_cvtepi8_epi16(self.0) })
  }

  /// Sign extend the lower four `i8` values into `i32`.
  #[inline(always)]
  #[must_use]
  pub fn sign_extend_i8_i32(self) -> Self {
    Self(unsafe { _mm_cvtepi8_epi32(self.0) })
  }

  /// Sign extend the lower two `i8` values into `i64`.
  #[inline(always)]
  #[must_use]
  pub fn sign_extend_i8_i64(self) -> Self {
    Self(unsafe { _mm_cvtepi8_epi64(self.0) })
  }

  /// Zero extend the lower four `u16` values into `i32`.
  #[inline(always)]
  #[must_use]
  pub fn zero_extend_u16_i32(self) -> Self {
    Self(unsafe { _mm_cvtepu16_epi32(self.0) })
  }

  /// Zero extend the lower two `u16` values into `i64`.
  #[inline(always)]
  #[must_use]
  pub fn zero_extend_u16_i64(self) -> Self {
    Self(unsafe { _mm_cvtepu16_epi64(self.0) })
  }

  /// Zero extend the lower two `u32` values into `i64`.
  #[inline(always)]
  #[must_use]
  pub fn zero_extend_u32_i64(self) -> Self {
    Self(unsafe { _mm_cvtepu32_epi64(self.0) })
  }

  /// Zero extend the lower eight `u8` values into `i16`.
  #[inline(always)]
  #[must_use]
  pub fn zero_extend_u8_i16(self) -> Self {
    Self(unsafe { _mm_cvtepu8_epi16(self.0) })
  }

  /// Zero extend the lower four `u8` values into `i32`.
  #[inline(always)]
  #[must_use]
  pub fn zero_extend_u8_i32(self) -> Self {
    Self(unsafe { _mm_cvtepu8_epi32(self.0) })
  }

  /// Zero extend the lower two `u8` values into `i64`.
  #[inline(always)]
  #[must_use]
  pub fn zero_extend_u8_i64(self) -> Self {
    Self(unsafe { _mm_cvtepu8_epi64(self.0) })
  }

  /// Lanewise `i32` maximum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn max_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i8` maximum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn max_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_epi8(self.0, rhs.0) })
  }

  /// Lanewise `u16` maximum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn max_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_epu16(self.0, rhs.0) })
  }

  /// Lanewise `u32` maximum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn max_u32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_max_epu32(self.0, rhs.0) })
  }

  /// Lanewise `i32` minimum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn min_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i8` minimum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn min_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_epi8(self.0, rhs.0) })
  }

  /// Lanewise `u16` minimum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn min_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_epu16(self.0, rhs.0) })
  }

  /// Lanewise `u32` minimum between `self` and `rhs`
  #[inline(always)]
  #[must_use]
  pub fn min_u32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_min_epu32(self.0, rhs.0) })
  }

  /// Minimum `u16` and its position.
  ///
  /// ```txt
  /// out_u16[0] = minimum lane value of self
  /// out_u16[1] = previous index of selected value
  /// the rest = zeroed
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn min_and_position_u16(self) -> Self {
    Self(unsafe { _mm_minpos_epu16(self.0) })
  }

  /// Multiply the even `i32` lanes and produce `i64` outputs
  ///
  /// ```txt
  /// out_i64[0] = self_i32[0] * rhs_i32[0]
  /// out_i64[1] = self_i32[2] * rhs_i32[2]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn widen_mul_i32_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mul_epi32(self.0, rhs.0) })
  }

  /// Lanewise `i32` multiply, keeping the low 32 bits of each result.
  #[inline(always)]
  #[must_use]
  pub fn mul_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mullo_epi32(self.0, rhs.0) })
  }

  /// Pack `self` then `rhs` `i32` lanes into `u16` lanes in the output.
  #[inline(always)]
  #[must_use]
  pub fn pack_u16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_packus_epi32(self.0, rhs.0) })
  }

  /// Sets `CF` to be `!self & All_1s`, then returns `CF`.
  #[inline(always)]
  #[must_use]
  pub fn test_all_bits_one(self) -> i32 {
    unsafe { _mm_test_all_ones(self.0) }
  }

  /// Sets `ZF` and `CF` as below, returns `CF`
  ///
  /// ```txt
  /// ZF = if self & rhs == 0 { 1 } else { 0 };
  /// CF = if (!self) & rhs == 0 { 1 } else { 0 };
  /// return CF;
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn test_cf(self, rhs: Self) -> i32 {
    unsafe { _mm_testc_si128(self.0, rhs.0) }
  }

  /// Sets `ZF` and `CF` as below, returns `ZF`.
  ///
  /// ```txt
  /// ZF = if self & rhs == 0 { 1 } else { 0 };
  /// CF = if (!self) & rhs == 0 { 1 } else { 0 };
  /// return ZF;
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn test_zf(self, rhs: Self) -> i32 {
    unsafe { _mm_testz_si128(self.0, rhs.0) }
  }

  /// Sets `ZF` and `CF` as below, returns if both are 0.
  ///
  /// ```txt
  /// ZF = if self & rhs == 0 { 1 } else { 0 };
  /// CF = if (!self) & rhs == 0 { 1 } else { 0 };
  /// return if ZF == 0 && CF == 0 { 1 } else { 0 };
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn test_not_zf_cf(self, rhs: Self) -> i32 {
    unsafe { _mm_testnzc_si128(self.0, rhs.0) }
  }
}
