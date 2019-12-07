#![cfg(target_feature="sse")]
#![cfg(target_feature="sse2")]
#![cfg(target_feature="sse3")]
#![cfg(target_feature="ssse3")]

use super::*;

/// # SSSE3 Operations
impl m128i {
  /// Lanewise `i8` wrapping absolute value.
  #[inline(always)]
  #[must_use]
  pub fn abs_i8(self) -> Self {
    Self(unsafe { _mm_abs_epi8(self.0) })
  }

  /// Lanewise `i16` wrapping absolute value.
  #[inline(always)]
  #[must_use]
  pub fn abs_i16(self) -> Self {
    Self(unsafe { _mm_abs_epi16(self.0) })
  }

  /// Lanewise `i32` wrapping absolute value.
  #[inline(always)]
  #[must_use]
  pub fn abs_i32(self) -> Self {
    Self(unsafe { _mm_abs_epi32(self.0) })
  }

  /// Horizontal add `i16` pairs in `self` and `rhs`.
  ///
  /// ```txt
  /// out[0]= self[1] + self[0]
  /// out[1]= self[3] + self[2]
  /// out[2]= self[5] + self[4]
  /// out[3]= self[7] + self[6]
  /// out[4]= rhs[1] + rhs[0]
  /// out[5]= rhs[3] + rhs[2]
  /// out[6]= rhs[5] + rhs[4]
  /// out[7]= rhs[7] + rhs[6]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn horizontal_add_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hadd_epi16(self.0, rhs.0) })
  }

  /// Horizontal saturating add `i16` pairs in `self` and `rhs`.
  ///
  /// ```txt
  /// out[0]= self[1].saturating_add(self[0])
  /// out[1]= self[3].saturating_add(self[2])
  /// out[2]= self[5].saturating_add(self[4])
  /// out[3]= self[7].saturating_add(self[6])
  /// out[4]= rhs[1].saturating_add(rhs[0])
  /// out[5]= rhs[3].saturating_add(rhs[2])
  /// out[6]= rhs[5].saturating_add(rhs[4])
  /// out[7]= rhs[7].saturating_add(rhs[6])
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn horizontal_saturating_add_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hadds_epi16(self.0, rhs.0) })
  }

  /// Horizontal add `i32` pairs in `self` and `rhs`.
  ///
  /// ```txt
  /// out[0]= self[1] + self[0]
  /// out[1]= self[3] + self[2]
  /// out[2]= rhs[5] + rhs[4]
  /// out[3]= rhs[7] + rhs[6]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn horizontal_add_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hadd_epi32(self.0, rhs.0) })
  }

  /// Horizontal subtract `i16` pairs in `self` and `rhs`.
  ///
  /// ```txt
  /// out[0]= self[0] - self[1]
  /// out[1]= self[2] - self[3]
  /// out[2]= self[4] - self[5]
  /// out[3]= self[6] - self[7]
  /// out[4]= rhs[0] - rhs[1]
  /// out[5]= rhs[2] - rhs[3]
  /// out[6]= rhs[4] - rhs[5]
  /// out[7]= rhs[6] - rhs[7]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn horizontal_sub_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hsub_epi16(self.0, rhs.0) })
  }

  /// Horizontal saturating subtract `i16` pairs in `self` and `rhs`.
  ///
  /// ```txt
  /// out[0]= self[1].saturating_sub(self[0])
  /// out[1]= self[3].saturating_sub(self[2])
  /// out[2]= self[5].saturating_sub(self[4])
  /// out[3]= self[7].saturating_sub(self[6])
  /// out[4]= rhs[1].saturating_sub(rhs[0])
  /// out[5]= rhs[3].saturating_sub(rhs[2])
  /// out[6]= rhs[5].saturating_sub(rhs[4])
  /// out[7]= rhs[7].saturating_sub(rhs[6])
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn horizontal_saturating_sub_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hsubs_epi16(self.0, rhs.0) })
  }
  /// Horizontal sub `i32` pairs in `self` and `rhs`.
  ///
  /// ```txt
  /// out[0]= self[1] - self[0]
  /// out[1]= self[3] - self[2]
  /// out[2]= rhs[5] - rhs[4]
  /// out[3]= rhs[7] - rhs[6]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn horizontal_sub_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hsub_epi32(self.0, rhs.0) })
  }

  /// Multiply `u8` values in `self` and `rhs` into `i16` intermediates, then
  /// horizontal add pack into the output.
  ///
  /// ```txt
  /// out_i16[0]= self_u8[0]*rhs_u8[0] + self_u8[1]*rhs_u8[1]
  /// out_i16[1]= self_u8[2]*rhs_u8[2] + self_u8[3]*rhs_u8[3]
  /// out_i16[2]= self_u8[4]*rhs_u8[4] + self_u8[5]*rhs_u8[5]
  /// out_i16[3]= self_u8[6]*rhs_u8[6] + self_u8[7]*rhs_u8[7]
  /// out_i16[4]= self_u8[8]*rhs_u8[8] + self_u8[9]*rhs_u8[9]
  /// out_i16[5]= self_u8[10]*rhs_u8[10] + self_u8[11]*rhs_u8[11]
  /// out_i16[6]= self_u8[12]*rhs_u8[12] + self_u8[13]*rhs_u8[13]
  /// out_i16[7]= self_u8[14]*rhs_u8[14] + self_u8[15]*rhs_u8[15]
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn mul_hadd_u8_to_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_maddubs_epi16(self.0, rhs.0) })
  }

  /// Multiply `i16` values in `self` and `rhs` into `i32`, keep high 18 bits,
  /// add 1, and then keep the middle 16 bits.
  ///
  /// ```txt
  /// for i in 0..16 {
  ///   out[i] = (((self[i] * rhs[i]) >> 14) + 1) >> 1
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn mul_higher_ish_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_mulhrs_epi16(self.0, rhs.0) })
  }

  /// Shuffle `i8` values in `self` according to control mask in `rhs`.
  ///
  /// ```txt
  /// for i in 0..16 {
  ///   out[i] = if rhs[i] < 0 {
  ///     0
  ///   } else {
  ///     self[rhs[i] & 0b1111]
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn shuffle_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_shuffle_epi8(self.0, rhs.0) })
  }

  /// `i8`: negate, zero, or no-change each lane of `self` based on `rhs`.
  ///
  /// ```txt
  /// for i in 0..16 {
  ///   out[i] = match sign(rhs[i]) {
  ///     Positive => self[i],
  ///     Zero => 0,
  ///     Negative => -self[i],
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn sign_i8(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sign_epi8(self.0, rhs.0) })
  }
  /// `i16`: negate, zero, or no-change each lane of `self` based on `rhs`.
  ///
  /// ```txt
  /// for i in 0..8 {
  ///   out[i] = match sign(rhs[i]) {
  ///     Positive => self[i],
  ///     Zero => 0,
  ///     Negative => -self[i],
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn sign_i16(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sign_epi16(self.0, rhs.0) })
  }
  /// `i32`: negate, zero, or no-change each lane of `self` based on `rhs`.
  ///
  /// ```txt
  /// for i in 0..4 {
  ///   out[i] = match sign(rhs[i]) {
  ///     Positive => self[i],
  ///     Zero => 0,
  ///     Negative => -self[i],
  ///   };
  /// }
  /// ```
  #[inline(always)]
  #[must_use]
  pub fn sign_i32(self, rhs: Self) -> Self {
    Self(unsafe { _mm_sign_epi32(self.0, rhs.0) })
  }
}
