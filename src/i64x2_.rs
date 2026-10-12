use crate::imports::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    type Inner = m128i;
  } else if #[cfg(target_feature="simd128")] {
    type Inner = v128;
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    type Inner = int64x2_t;
  } else {
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub arr: [i64; 2],
    }
  }
}

define_simd_signed! {
  unsafe {
    T = i64,
    N = 2,
    N_STR = "two",
    Simd = i64x2,
    UnsignedSimd = u64x2,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    BitmaskType = u32,
    [0, 1],
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = int64x2_t },
    optional_type_wasm_inner { WasmInner = v128 },
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        // only has gt, so flip arguments around to get lt
        Self(cmp_gt_mask_i64_m128i( rhs.0, self.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(i64x2_lt(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self(vreinterpretq_s64_u64(vcltq_s64(self.0, rhs.0)))}
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] < r[0] { -1_i64 } else { 0 },
          if s[1] < r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        Self(cmp_gt_mask_i64_m128i(self.0, rhs.0))
      } else if #[cfg(target_feature="simd128")] {
        Self(i64x2_gt(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self(vreinterpretq_s64_u64(vcgtq_s64(self.0, rhs.0)))}
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] > r[0] { -1_i64 } else { 0 },
          if s[1] > r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_gt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(i64x2_le(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_gt(rhs)
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] <= r[0] { -1_i64 } else { 0 },
          if s[1] <= r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_lt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self(i64x2_ge(self.0, rhs.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_lt(rhs)
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] >= r[0] { -1_i64 } else { 0 },
          if s[1] >= r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u64x2) -> Self::Output {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // mask the shift count to 63 to have same behavior on all platforms
          // no right shift, have to pass negative value to left shift on neon
          let shift_by = vnegq_s64(vreinterpretq_s64_u64(vandq_u64(rhs.0, vmovq_n_u64(63))));
          Self(vshlq_s64(self.0, shift_by))
        }
      } else {
        let arr: [i64; 2] = cast(self);
        let rhs: [u64; 2] = cast(rhs);
        cast([
          arr[0].wrapping_shr(rhs[0] as u32),
          arr[1].wrapping_shr(rhs[1] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="simd128")] {
        Self(i64x2_shr(self.0, rhs))
      } else {
        let arr: [i64; 2] = cast(self);
        cast([
          arr[0].wrapping_shr(rhs),
          arr[1].wrapping_shr(rhs),
        ])
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    self.simd_gt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    self.simd_lt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn reduce_max(self) -> i64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [i64; 2] = cast(self);
        array[0].max(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_s64(self.0, 0).max(vgetq_lane_s64(self.0, 1)) }
      } else {
        self.0.arr[0].max(self.0.arr[1])
      }
    }
  }

  #[inline]
  pub fn reduce_min(self) -> i64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [i64; 2] = cast(self);
        array[0].min(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_s64(self.0, 0).min(vgetq_lane_s64(self.0, 1)) }
      } else {
        self.0.arr[0].min(self.0.arr[1])
      }
    }
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u64x2) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Negate `rhs` because there is no direct shift-right intrinsic, and
          // restrict it to prevent overflow.
          Self(vshlq_s64(self.0, vnegq_s64(vreinterpretq_s64_u64(rhs.min(u64x2::splat(64)).0))))
        }
      } else {
        // Cannot use scalar `unbounded_shl` because it takes `u32`, which is
        // smaller than `u64`.
        cast::<u64x2, i64x2>(rhs.simd_lt(64)).select(
          self >> cast::<u64x2, i64x2>(rhs),
          self.is_negative(),
        )
      }
    }
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="simd128")] {
        if rhs < 64 { Self(i64x2_shr(self.0, rhs)) } else { self.is_negative() }
      } else {
        let self_array = self.to_array();

        Self::new([
          self_array[0].unbounded_shr(rhs),
          self_array[1].unbounded_shr(rhs),
        ])
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vqaddq_s64(self.0, rhs.0)) }
      } else {
        Self::new([
            self.0.arr[0].saturating_add(rhs.0.arr[0]),
            self.0.arr[1].saturating_add(rhs.0.arr[1]),
        ])
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self(vqsubq_s64(self.0, rhs.0)) }
      } else {
        Self::new([
            self.0.arr[0].saturating_sub(rhs.0.arr[0]),
            self.0.arr[1].saturating_sub(rhs.0.arr[1]),
        ])
      }
    }
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one? This intentionally avoids `mul_keep_low_high` because getting the
    // high bits of 64-bit multiplication could be slow.

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let result = [
      self_array[0].overflowing_mul(rhs_array[0]),
      self_array[1].overflowing_mul(rhs_array[1]),
    ];
    (
      Self::new([result[0].0, result[1].0]),
      Self::new([-(result[0].1 as i64), -(result[1].1 as i64)]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i128x2` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u64x2, i64x2) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
      (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
    ];

    (
      u64x2::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
      ]),
      i64x2::new([
        (widening_mul[0] >> 64) as i64,
        (widening_mul[1] >> 64) as i64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    Self::new([
      ((self_array[0] as i128).wrapping_mul(rhs_array[0] as i128) >> 64) as i64,
      ((self_array[1] as i128).wrapping_mul(rhs_array[1] as i128) >> 64) as i64,
    ])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      // x86 doesn't have this builtin
      if #[cfg(target_feature="simd128")] {
        Self(i64x2_abs(self.0))
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self(vabsq_s64(self.0))}
      } else {
        let arr: [i64; 2] = cast(self);
        cast(
          [
            arr[0].wrapping_abs(),
            arr[1].wrapping_abs(),
          ])
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        Self(unsafe { vreinterpretq_s64_u64(vcgtzq_s64(self.0)) })
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  #[inline]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        Self(unsafe { vreinterpretq_s64_u64(vcltzq_s64(self.0)) })
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

/// The following functionality exists only for [`i64x2`], or only for
/// particular types inconsistently.
impl i64x2 {
    /// Converts each element from [`i64`] to [`f64`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f64x2 {
        let arr: [i64; 2] = cast(self);
        cast([arr[0] as f64, arr[1] as f64])
    }
}
