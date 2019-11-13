#![cfg(target_feature = "fma")]

use super::*;

impl m128 {
  /// fused `(self * b) + c`
  #[inline(always)]
  pub fn fmadd(self, b: Self, c: Self) -> Self {
    Self(unsafe { _mm_fmadd_ps(self.0, b.0, c.0) })
  }

  /// fused `-(self * b) + c`
  #[inline(always)]
  pub fn fnmadd(self, b: Self, c: Self) -> Self {
    Self(unsafe { _mm_fnmadd_ps(self.0, b.0, c.0) })
  }
}
