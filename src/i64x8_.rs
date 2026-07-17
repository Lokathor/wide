use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i64x8 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i64x8 { pub(crate) a : i64x4, pub(crate) b : i64x4 }
  }
}

impl_simd! {
  unsafe {
    T = i64,
    N = 8,
    Simd = i64x8,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i64_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
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
        Self { avx512: cmp_op_mask_i64_m512i::<{cmp_int_op!(Ne)}>(self.avx512, rhs.avx512) }
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
        Self { avx512: cmp_op_mask_i64_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : rhs.a.simd_gt(self.a),
          b : rhs.b.simd_gt(self.b),
        }
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i64_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
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
        Self { avx512: cmp_op_mask_i64_m512i::<{cmp_int_op!(Le)}>(self.avx512, rhs.avx512) }
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
        Self { avx512: cmp_op_mask_i64_m512i::<{cmp_int_op!(Nlt)}>(self.avx512, rhs.avx512) }
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
          a : self.a.bitselect(if_one.a, if_zero.a),
          b : self.b.bitselect(if_one.b, if_zero.b),
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

  /// returns the bit mask for each high bit set in the vector with the lowest
  /// lane being the lowest bit
  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        // use f64 move_mask since it is the same size as i64
        movepi64_mask_m512d(cast(self.avx512)) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 4)
      }
    }
  }

  /// true if any high bits are set for any value in the vector
  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi64_mask_m512d(cast(self.avx512)) != 0
      } else {
        let [a, b]: [i64x4; 2] = cast(self);
        (a | b).any()
      }
    }
  }

  /// true if all high bits are set for every value in the vector
  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        movepi64_mask_m512d(cast(self.avx512)) == 0b11111111
      } else {
        let [a, b]: [i64x4; 2] = cast(self);
        (a & b).all()
      }
    }
  }

  /// Transpose matrix of 8x8 `i64` matrix. Currently not accelerated.
  #[inline]
  pub fn transpose(data: [i64x8; 8]) -> [i64x8; 8] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i64x8; 8], index: usize) -> i64x8 {
      i64x8::new([
        data[0].as_array()[index],
        data[1].as_array()[index],
        data[2].as_array()[index],
        data[3].as_array()[index],
        data[4].as_array()[index],
        data[5].as_array()[index],
        data[6].as_array()[index],
        data[7].as_array()[index],
      ])
    }

    [
      transpose_column(&data, 0),
      transpose_column(&data, 1),
      transpose_column(&data, 2),
      transpose_column(&data, 3),
      transpose_column(&data, 4),
      transpose_column(&data, 5),
      transpose_column(&data, 6),
      transpose_column(&data, 7),
    ]
  }
}

impl_simd_int! {
  unsafe {
    T = i64,
    N = 8,
    Simd = i64x8,
    UnsignedSimd = u64x8,
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    [0, 1, 2, 3, 4, 5, 6, 7],
  }

  #[inline]
  fn shr(self, rhs: u64x8) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // TODO(safe_arch): add shr_each_i64_m512i (arithmetic right shift)
        // Self { avx512: shr_each_i64_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512f")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = rhs as u64 & 63;
        Self { avx512: shr_all_i64_m512i(self.avx512, shift) }
      } else {
        Self {
          a : self.a.shr(rhs),
          b : self.b.shr(rhs),
        }
      }
    }
  }

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_i64_m512i(self.avx512, rhs.avx512) }
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
        Self { avx512: min_i64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.min(rhs.a),
          b: self.b.min(rhs.b),
        }
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
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
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
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
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
      u64x8::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
        widening_mul[4] as u64,
        widening_mul[5] as u64,
        widening_mul[6] as u64,
        widening_mul[7] as u64,
      ]),
      i64x8::new([
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

    Self::new([
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
        Self {
          a : self.a.abs(),
          b : self.b.abs(),
        }
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        // `neon` has dedicated greater-than-zero intrinsics.
        Self {
          a: self.a.is_positive(),
          b: self.b.is_positive(),
        }
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
        Self {
          a: self.a.is_negative(),
          b: self.b.is_negative(),
        }
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

impl i64x8 {
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
