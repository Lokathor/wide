use crate::imports::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    pub(crate) type Inner = m512i;
  } else {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    pub(crate) struct Inner {
      pub lo: i64x4,
      pub hi: i64x4,
    }
  }
}

define_simd_signed! {
  unsafe {
    T = i64,
    N = 8,
    N_STR = "eight",
    Simd = i64x8,
    UnsignedSimd = u64x8,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    BitmaskType = u32,
    [0, 1, 2, 3, 4, 5, 6, 7],
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Lt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: rhs.0.lo.simd_gt(self.0.lo),
          hi: rhs.0.hi.simd_gt(self.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Nle)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_gt(rhs.0.lo),
          hi: self.0.hi.simd_gt(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Le)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_le(rhs.0.lo),
          hi: self.0.hi.simd_le(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(cmp_op_mask_i64_m512i::<{cmp_int_op!(Nlt)}>(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.simd_ge(rhs.0.lo),
          hi: self.0.hi.simd_ge(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u64x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // TODO(safe_arch): add shr_each_i64_m512i (arithmetic right shift)
        // Self(shr_each_i64_m512i(self.0, rhs.0) }
        // Fallback for now:
        let a: [i64; 8] = cast(self);
        let r: [u64; 8] = cast(rhs);
        cast([
          a[0].wrapping_shr(r[0] as u32),
          a[1].wrapping_shr(r[1] as u32),
          a[2].wrapping_shr(r[2] as u32),
          a[3].wrapping_shr(r[3] as u32),
          a[4].wrapping_shr(r[4] as u32),
          a[5].wrapping_shr(r[5] as u32),
          a[6].wrapping_shr(r[6] as u32),
          a[7].wrapping_shr(r[7] as u32),
        ])
      } else {
        // widen via two halves
        Self(Inner {
          lo: self.0.lo.shr(rhs.0.lo),
          hi: self.0.hi.shr(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self(shr_all_i64_m512i(self.0, shift))
      } else {
        Self(Inner {
          lo: self.0.lo.shr(rhs),
          hi: self.0.hi.shr(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(max_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.max(rhs.0.lo),
          hi: self.0.hi.max(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(min_i64_m512i(self.0, rhs.0))
      } else {
        Self(Inner {
          lo: self.0.lo.min(rhs.0.lo),
          hi: self.0.hi.min(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn unbounded_shr(self, rhs: u64x8) -> Self {
    // TODO(safe_arch): add shr_each_i64_m512i (arithmetic right shift)
    // Self(shr_each_i64_m512i(self.0, rhs.0) }
    // Fallback for now:

    let [self_a, self_b] = cast::<i64x8, [i64x4; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u64x8, [u64x4; 2]>(rhs);

    cast([self_a.unbounded_shr(rhs_a), self_b.unbounded_shr(rhs_b)])
  }

  #[inline]
  pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self(shr_all_i64_m512i(self.0, rhs as u64))
      } else {
        Self(Inner {
          lo: self.0.lo.unbounded_shr_scalar(rhs),
          hi: self.0.hi.unbounded_shr_scalar(rhs),
        })
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else {
        Self(Inner {
          lo: self.0.lo.saturating_add(rhs.0.lo),
          hi: self.0.hi.saturating_add(rhs.0.hi),
        })
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else {
        Self(Inner {
          lo: self.0.lo.saturating_sub(rhs.0.lo),
          hi: self.0.hi.saturating_sub(rhs.0.hi),
        })
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
      self_array[2].overflowing_mul(rhs_array[2]),
      self_array[3].overflowing_mul(rhs_array[3]),
      self_array[4].overflowing_mul(rhs_array[4]),
      self_array[5].overflowing_mul(rhs_array[5]),
      self_array[6].overflowing_mul(rhs_array[6]),
      self_array[7].overflowing_mul(rhs_array[7]),
    ];
    (
      Self::from_array([
        result[0].0,
        result[1].0,
        result[2].0,
        result[3].0,
        result[4].0,
        result[5].0,
        result[6].0,
        result[7].0,
      ]),
      Self::from_array([
        -(result[0].1 as i64),
        -(result[1].1 as i64),
        -(result[2].1 as i64),
        -(result[3].1 as i64),
        -(result[4].1 as i64),
        -(result[5].1 as i64),
        -(result[6].1 as i64),
        -(result[7].1 as i64),
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i128x8` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u64x8, i64x8) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
      (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
      (self_array[2] as i128).wrapping_mul(rhs_array[2] as i128),
      (self_array[3] as i128).wrapping_mul(rhs_array[3] as i128),
      (self_array[4] as i128).wrapping_mul(rhs_array[4] as i128),
      (self_array[5] as i128).wrapping_mul(rhs_array[5] as i128),
      (self_array[6] as i128).wrapping_mul(rhs_array[6] as i128),
      (self_array[7] as i128).wrapping_mul(rhs_array[7] as i128),
    ];

    (
      u64x8::from_array([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
        widening_mul[4] as u64,
        widening_mul[5] as u64,
        widening_mul[6] as u64,
        widening_mul[7] as u64,
      ]),
      i64x8::from_array([
        (widening_mul[0] >> 64) as i64,
        (widening_mul[1] >> 64) as i64,
        (widening_mul[2] >> 64) as i64,
        (widening_mul[3] >> 64) as i64,
        (widening_mul[4] >> 64) as i64,
        (widening_mul[5] >> 64) as i64,
        (widening_mul[6] >> 64) as i64,
        (widening_mul[7] >> 64) as i64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    Self::from_array([
      ((self_array[0] as i128).wrapping_mul(rhs_array[0] as i128) >> 64) as i64,
      ((self_array[1] as i128).wrapping_mul(rhs_array[1] as i128) >> 64) as i64,
      ((self_array[2] as i128).wrapping_mul(rhs_array[2] as i128) >> 64) as i64,
      ((self_array[3] as i128).wrapping_mul(rhs_array[3] as i128) >> 64) as i64,
      ((self_array[4] as i128).wrapping_mul(rhs_array[4] as i128) >> 64) as i64,
      ((self_array[5] as i128).wrapping_mul(rhs_array[5] as i128) >> 64) as i64,
      ((self_array[6] as i128).wrapping_mul(rhs_array[6] as i128) >> 64) as i64,
      ((self_array[7] as i128).wrapping_mul(rhs_array[7] as i128) >> 64) as i64,
    ])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // AVX512 might have this, unsure for now
        let arr: [i64; 8] = cast(self);
        cast(
          [
            arr[0].wrapping_abs(),
            arr[1].wrapping_abs(),
            arr[2].wrapping_abs(),
            arr[3].wrapping_abs(),
            arr[4].wrapping_abs(),
            arr[5].wrapping_abs(),
            arr[6].wrapping_abs(),
            arr[7].wrapping_abs(),
          ])
      } else {
        Self(Inner {
          lo: self.0.lo.abs(),
          hi: self.0.hi.abs(),
        })
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated greater-than-zero intrinsics.
        Self(Inner {
          lo: self.0.lo.is_positive(),
          hi: self.0.hi.is_positive(),
        })
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  #[inline]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated less-than-zero intrinsics.
        Self(Inner {
          lo: self.0.lo.is_negative(),
          hi: self.0.hi.is_negative(),
        })
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

/// The following functionality exists only for [`i64x8`], or only for
/// particular types inconsistently.
impl i64x8 {
    /// Converts each element from [`i64`] to [`f64`].
    #[inline]
    #[must_use]
    pub fn round_float(self) -> f64x8 {
        let arr: [i64; 8] = cast(self);
        cast([
            arr[0] as f64,
            arr[1] as f64,
            arr[2] as f64,
            arr[3] as f64,
            arr[4] as f64,
            arr[5] as f64,
            arr[6] as f64,
            arr[7] as f64,
        ])
    }
}
