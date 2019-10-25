#![cfg(target_feature="sse")]
#![cfg(target_feature="sse2")]
#![cfg(target_feature="sse3")]

use super::*;

/// # SSE3 Operations
impl m128 {
  /// Adds odd lanes (3 and 1) and subtracts even lanes (2 and 0).
  ///
  /// ```txt
  /// out[0]= self[0] - rhs[0]
  /// out[1]= self[1] + rhs[1]
  /// out[2]= self[2] - rhs[2]
  /// out[3]= self[3] + rhs[3]
  /// ```
  #[inline(always)]
  pub fn add_sub(self, rhs: Self) -> Self {
    Self(unsafe { _mm_addsub_ps(self.0, rhs.0) })
  }

  /// Horizontal add both `self` and `rhs`, then pack together.
  ///
  /// ```txt
  /// out[0]= self[0] + self[1]
  /// out[1]= self[2] + self[3]
  /// out[2]= rhs[0] + rhs[1]
  /// out[3]= rhs[2] + rhs[3]
  /// ```
  #[inline(always)]
  pub fn horizontal_add(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hadd_ps(self.0, rhs.0) })
  }

  /// Horizontal subtract both `self` and `rhs`, then pack together.
  ///
  /// ```txt
  /// out[0]= self[0] - self[1]
  /// out[1]= self[2] - self[3]
  /// out[2]= rhs[0] - rhs[1]
  /// out[3]= rhs[2] - rhs[3]
  /// ```
  #[inline(always)]
  pub fn horizontal_sub(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hsub_ps(self.0, rhs.0) })
  }

  /// Duplicate odd indexed lanes into a new `m128`.
  ///
  /// ```txt
  /// out[0]= self[1]
  /// out[1]= self[1]
  /// out[2]= self[3]
  /// out[3]= self[3]
  /// ```
  #[inline(always)]
  pub fn duplicate_odd(self) -> Self {
    Self(unsafe { _mm_movehdup_ps(self.0) })
  }

  /// Duplicate even indexed lanes into a new `m128`.
  ///
  /// ```txt
  /// out[0]= self[0]
  /// out[1]= self[0]
  /// out[2]= self[2]
  /// out[3]= self[2]
  /// ```
  #[inline(always)]
  pub fn duplicate_even(self) -> Self {
    Self(unsafe { _mm_moveldup_ps(self.0) })
  }
}

/// # SSE3 Operations
impl m128d {
  /// Adds the high lane (1) and subtracts the low lane (0).
  ///
  /// ```txt
  /// out[0]= self[0] - rhs[0]
  /// out[1]= self[1] + rhs[1]
  /// ```
  #[inline(always)]
  pub fn add_sub(self, rhs: Self) -> Self {
    Self(unsafe { _mm_addsub_pd(self.0, rhs.0) })
  }

  /// Horizontal add both `self` and `rhs`, then pack together.
  ///
  /// ```txt
  /// out[0]= self[0] + self[1]
  /// out[1]= rhs[0] + rhs[1]
  /// ```
  #[inline(always)]
  pub fn horizontal_add(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hadd_pd(self.0, rhs.0) })
  }

  /// Horizontal subtract both `self` and `rhs`, then pack together.
  ///
  /// ```txt
  /// out[0]= self[0] - self[1]
  /// out[1]= rhs[0] - rhs[1]
  /// ```
  #[inline(always)]
  pub fn horizontal_sub(self, rhs: Self) -> Self {
    Self(unsafe { _mm_hsub_pd(self.0, rhs.0) })
  }

  /// Load the given `f64` address, duplicating it into both lanes.
  #[inline(always)]
  #[allow(clippy::trivially_copy_pass_by_ref)]
  pub fn load_splat(addr: &f64) -> Self {
    Self(unsafe { _mm_loaddup_pd(addr) })
  }

  /// Duplicate the low lane of `self` into both lanes of a new `m128d`.
  ///
  /// ```txt
  /// out[0]= self[0]
  /// out[1]= self[0]
  /// ```
  #[inline(always)]
  pub fn duplicate_low(self) -> Self {
    Self(unsafe { _mm_movedup_pd(self.0) })
  }
}

/// # SSE3 Operations
impl m128i {
  /// Loads 128-bits of integer data without alignment requirements.
  ///
  /// This can perform faster than [`m128i::load_unaligned`] if the data would
  /// cross a cache line boundary.
  #[inline(always)]
  pub fn load_quick_unaligned(addr: *const i128) -> Self {
    #[allow(clippy::cast_ptr_alignment)]
    Self(unsafe { _mm_lddqu_si128(addr as *const _) })
  }
}
