use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { pub(crate) avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { pub(crate) a: u64x2, pub(crate) b: u64x2 }
  }
}

impl_simd! {
  unsafe {
    T = u64,
    N = 4,
    Simd = u64x4,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i64_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.simd_eq(rhs.a),
          b: self.b.simd_eq(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_eq(rhs)
      } else {
        Self {
          a: self.a.simd_ne(rhs.a),
          b: self.b.simd_ne(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    // lt is just gt the other way around
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // no unsigned gt than so inverting the high bit will get the correct result
        let highbit = u64x4::splat(1 << 63);
        Self { avx2: cmp_gt_mask_i64_m256i((self ^ highbit).avx2, (rhs ^ highbit).avx2) }
      } else {
        Self {
          a: self.a.simd_gt(rhs.a),
          b: self.b.simd_gt(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_gt(rhs)
      } else {
        Self {
          a: self.a.simd_le(rhs.a),
          b: self.b.simd_le(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        !self.simd_lt(rhs)
      } else {
        Self {
          a: self.a.simd_ge(rhs.a),
          b: self.b.simd_ge(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self {
          avx2: bitor_m256i(
            bitand_m256i(if_one.avx2, self.avx2),
            bitandnot_m256i(self.avx2, if_zero.avx2),
          ),
        }
      } else {
        Self {
          a: self.a.bitselect(if_one.a, if_zero.a),
          b: self.b.bitselect(if_one.b, if_zero.b),
        }
      }
    }
  }

  #[inline]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(if_false.avx2,if_true.avx2,self.avx2) }
      } else {
        Self {
          a: self.a.select(if_true.a, if_false.a),
          b: self.b.select(if_true.b, if_false.b),
        }
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    i64x4::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i64x4::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i64x4::all(cast(self))
  }

  /// Transpose matrix of 4x4 `u64` matrix.
  #[inline]
  pub fn transpose(data: [u64x4; 4]) -> [u64x4; 4] {
    cast(i64x4::transpose(cast(data)))
  }
}

impl_simd_uint! {
  unsafe {
    T = u64,
    N = 4,
    Simd = u64x4,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3],
  }

  #[inline]
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: self.avx2.not()  }
      } else {
        Self {
          a: self.a.not(),
          b: self.b.not(),
        }
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_i64_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.add(rhs.a),
          b: self.b.add(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_i64_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.sub(rhs.a),
          b: self.b.sub(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr1: [i64; 4] = cast(self);
        let arr2: [i64; 4] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
          arr1[2].wrapping_mul(arr2[2]),
          arr1[3].wrapping_mul(arr2[3]),
        ])
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self { avx2: shl_each_u64_m256i(self.avx2, shift_by.avx2) }
      } else {
        Self {
          a: self.a.shl(rhs.a),
          b: self.b.shl(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self { avx2: shl_all_u64_m256i(self.avx2, shift) }
      } else {
        Self {
          a: self.a.shl(rhs),
          b: self.b.shl(rhs),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self { avx2: shr_each_u64_m256i(self.avx2, shift_by.avx2) }
      } else {
        Self {
          a: self.a.shr(rhs.a),
          b: self.b.shr(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self { avx2: shr_all_u64_m256i(self.avx2, shift) }
      } else {
        Self {
          a: self.a.shr(rhs),
          b: self.b.shr(rhs),
        }
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitand_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.bitand(rhs.a),
          b: self.b.bitand(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx2")] {
        Self { avx2: bitor_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.bitor(rhs.a),
          b: self.b.bitor(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitxor_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a: self.a.bitxor(rhs.a),
          b: self.b.bitxor(rhs.b),
        }
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
  pub fn reduce_add(self) -> u64 {
    cast(i64x4::reduce_add(cast(self)))
  }


  #[inline]
  pub fn reduce_mul(self) -> u64 {
    let array: [u64; 4] = cast(self);
    array[0]
      .wrapping_mul(array[1])
      .wrapping_mul(array[2])
      .wrapping_mul(array[3])
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    let array: [u64; 4] = cast(self);
    array[0].max(array[1]).max(array[2]).max(array[3])
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    let array: [u64; 4] = cast(self);
    array[0].min(array[1]).min(array[2]).min(array[3])
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let result = self + rhs;
        let overflow = result.simd_lt(self);
        // Return `MAX` (all bits set) if overflow occurs.
        result | overflow
      } else {
        Self {
          a: self.a.saturating_add(rhs.a),
          b: self.b.saturating_add(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let result = self - rhs;
        let no_overflow = result.simd_le(self);
        // Return `0` (no bits set) if overflow occurs.
        result & no_overflow
      } else {
        Self {
          a: self.a.saturating_sub(rhs.a),
          b: self.b.saturating_sub(rhs.b),
        }
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
    ];
    (
      Self::new([result[0].0, result[1].0, result[2].0, result[3].0]),
      Self::new([
        -(result[0].1 as i64) as u64,
        -(result[1].1 as i64) as u64,
        -(result[2].1 as i64) as u64,
        -(result[3].1 as i64) as u64,
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `u128x4` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (Self, Self) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as u128).wrapping_mul(rhs_array[0] as u128),
      (self_array[1] as u128).wrapping_mul(rhs_array[1] as u128),
      (self_array[2] as u128).wrapping_mul(rhs_array[2] as u128),
      (self_array[3] as u128).wrapping_mul(rhs_array[3] as u128),
    ];

    (
      Self::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
      ]),
      Self::new([
        (widening_mul[0] >> 64) as u64,
        (widening_mul[1] >> 64) as u64,
        (widening_mul[2] >> 64) as u64,
        (widening_mul[3] >> 64) as u64,
      ]),
    )
  }

  #[inline]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr1: [u64; 4] = cast(self);
        let arr2: [u64; 4] = cast(rhs);
        cast([
          (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
          (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
          (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
          (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
        ])
      } else {
        Self {
          a: self.a.mul_keep_high(rhs.a),
          b: self.b.mul_keep_high(rhs.b),
        }
      }
    }
  }
}
