use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct u64x8 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct u64x8 { pub(crate) a : u64x4, pub(crate) b : u64x4 }
  }
}

impl_simd! {
  T = u64,
  N = 8,
  Simd = u64x8,

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u64_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u64_m512i::<{cmp_int_op!(Ne)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ne(rhs.a),
          b : self.b.simd_ne(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u64_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_lt(rhs.a),
          b : self.b.simd_lt(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u64_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_gt(rhs.a),
          b : self.b.simd_gt(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u64_m512i::<{cmp_int_op!(Le)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_le(rhs.a),
          b : self.b.simd_le(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_u64_m512i::<{cmp_int_op!(Nlt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ge(rhs.a),
          b : self.b.simd_ge(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self {
          avx512: bitor_m512i(
            bitand_m512i(if_one.avx512, self.avx512),
            bitandnot_m512i(self.avx512, if_zero.avx512),
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
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_i8_m512i(if_false.avx512,if_true.avx512,movepi8_mask_m512i(self.avx512)) }
      } else {
        Self {
          a : self.a.select(if_true.a, if_false.a),
          b : self.b.select(if_true.b, if_false.b),
        }
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    i64x8::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i64x8::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i64x8::all(cast(self))
  }

  /// Transpose matrix of 8x8 `u64` matrix. Currently not accelerated.
  #[inline]
  pub fn transpose(data: [u64x8; 8]) -> [u64x8; 8] {
    cast(i64x8::transpose(cast(data)))
  }
}

impl_simd_uint! {
  T = u64,
  N = 8,
  Simd = u64x8,
  [0, 1, 2, 3, 4, 5, 6, 7],

  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512i(self.avx512, set_splat_i64_m512i(-1)) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: add_i64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sub_i64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let arr1: [u64; 8] = cast(self);
        let arr2: [u64; 8] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
          arr1[2].wrapping_mul(arr2[2]),
          arr1[3].wrapping_mul(arr2[3]),
          arr1[4].wrapping_mul(arr2[4]),
          arr1[5].wrapping_mul(arr2[5]),
          arr1[6].wrapping_mul(arr2[6]),
          arr1[7].wrapping_mul(arr2[7]),
        ])
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i64_m512i(63));
        Self { avx512: shl_each_u64_m512i(self.avx512, rhs) }
      } else {
        Self {
          a : self.a.shl(rhs.a),
          b : self.b.shl(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self { avx512: shl_all_u64_m512i(self.avx512, shift) }
      } else {
        Self {
          a : self.a.shl(rhs),
          b : self.b.shl(rhs),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i64_m512i(63));
        Self { avx512: shr_each_u64_m512i(self.avx512, rhs) }
      } else {
        Self {
          a : self.a.shr(rhs.a),
          b : self.b.shr(rhs.b),
        }
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
        Self { avx512: shr_all_u64_m512i(self.avx512, shift) }
      } else {
        Self {
          a : self.a.shr(rhs),
          b : self.b.shr(rhs),
        }
      }
    }
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitand_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitand(rhs.a),
          b : self.b.bitand(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitor_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitor(rhs.a),
          b : self.b.bitor(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitxor(rhs.a),
          b : self.b.bitxor(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_u64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.max(rhs.a),
          b: self.b.max(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_u64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.min(rhs.a),
          b: self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  pub fn reduce_add(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }


  #[inline]
  pub fn reduce_mul(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
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
      if #[cfg(target_feature="avx512f")] {
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
  pub fn saturating_mul(self, rhs: Self) -> Self {
    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    Self::new([
      self_array[0].saturating_mul(rhs_array[0]),
      self_array[1].saturating_mul(rhs_array[1]),
      self_array[2].saturating_mul(rhs_array[2]),
      self_array[3].saturating_mul(rhs_array[3]),
      self_array[4].saturating_mul(rhs_array[4]),
      self_array[5].saturating_mul(rhs_array[5]),
      self_array[6].saturating_mul(rhs_array[6]),
      self_array[7].saturating_mul(rhs_array[7]),
    ])
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

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
      Self::new([
        result[0].0,
        result[1].0,
        result[2].0,
        result[3].0,
        result[4].0,
        result[5].0,
        result[6].0,
        result[7].0,
      ]),
      Self::new([
        -(result[0].1 as i64) as u64,
        -(result[1].1 as i64) as u64,
        -(result[2].1 as i64) as u64,
        -(result[3].1 as i64) as u64,
        -(result[4].1 as i64) as u64,
        -(result[5].1 as i64) as u64,
        -(result[6].1 as i64) as u64,
        -(result[7].1 as i64) as u64,
      ]),
    )
  }
}

unsafe impl Zeroable for u64x8 {}
unsafe impl Pod for u64x8 {}

impl AlignTo for u64x8 {
  type Elem = u64;
}

impl u64x8 {
  #[inline]
  #[must_use]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let arr1: [u64; 8] = cast(self);
        let arr2: [u64; 8] = cast(rhs);
        cast([
          (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
          (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
          (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
          (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
          (arr1[4] as u128 * arr2[4] as u128 >> 64) as u64,
          (arr1[5] as u128 * arr2[5] as u128 >> 64) as u64,
          (arr1[6] as u128 * arr2[6] as u128 >> 64) as u64,
          (arr1[7] as u128 * arr2[7] as u128 >> 64) as u64,
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
