use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    /// A SIMD vector with four elements of type [`i64`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { pub(crate) avx2: m256i }
  } else {
    /// A SIMD vector with four elements of type [`i64`].
    ///
    /// See the [crate level documentation] for more information about SIMD
    /// vectors.
    ///
    /// [crate level documentation]: crate
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { pub(crate) a : i64x2, pub(crate) b : i64x2 }
  }
}

impl_simd! {
  unsafe {
    T = i64,
    N = 4,
    Simd = i64x4,
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
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
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
          a : self.a.simd_ne(rhs.a),
          b : self.b.simd_ne(rhs.b),
        }
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: !(cmp_gt_mask_i64_m256i(self.avx2, rhs.avx2) ^ cmp_eq_mask_i64_m256i(self.avx2, rhs.avx2)) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i64_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        !self.simd_gt(rhs)
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
      if #[cfg(target_feature="avx2")] {
        !self.simd_lt(rhs)
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
      if #[cfg(target_feature="avx2")] {
        // use f64 move_mask since it is the same size as i64
        move_mask_m256d(cast(self.avx2)) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 2)
      }
    }
  }

  /// true if any high bits are set for any value in the vector
  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256d(cast(self.avx2)) != 0
      } else {
        (self.a | self.b).any()
      }
    }
  }

  /// true if all high bits are set for every value in the vector
  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        move_mask_m256d(cast(self.avx2)) == 0b1111
      } else {
        (self.a & self.b).all()
      }
    }
  }

  ///
  /// Currently this function is only accelerated on `avx2`.
  #[inline]
  pub fn transpose(data: [i64x4; 4]) -> [i64x4; 4] {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // Can this be optimized?
        let a = data[0].unpack_lo(data[2]);
        let b = data[1].unpack_lo(data[3]);
        let c = data[0].unpack_hi(data[2]);
        let d = data[1].unpack_hi(data[3]);
        [
          a.unpack_lo(b),
          a.unpack_hi(b),
          c.unpack_lo(d),
          c.unpack_hi(d),
        ]
      } else {
        #[inline(always)]
        fn transpose_column(data: &[i64x4; 4], index: usize) -> i64x4 {
          i64x4::new([
            data[0].as_array()[index],
            data[1].as_array()[index],
            data[2].as_array()[index],
            data[3].as_array()[index],
          ])
        }

        [
          transpose_column(&data, 0),
          transpose_column(&data, 1),
          transpose_column(&data, 2),
          transpose_column(&data, 3),
        ]
      }
    }
  }
}

impl_simd_int! {
  unsafe {
    T = i64,
    N = 4,
    Simd = i64x4,
    UnsignedSimd = u64x4,
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
          a : self.a.not(),
          b : self.b.not(),
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
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
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
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
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
          a : self.a.shl(rhs.a),
          b : self.b.shl(rhs.b),
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
          a : self.a.shl(rhs),
          b : self.b.shl(rhs),
        }
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr: [i64; 4] = cast(self);
        let rhs: [i64; 4] = cast(rhs);
        cast([
          arr[0].wrapping_shr(rhs[0] as u32),
          arr[1].wrapping_shr(rhs[1] as u32),
          arr[2].wrapping_shr(rhs[2] as u32),
          arr[3].wrapping_shr(rhs[3] as u32),
        ])
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
    // there is no signed right shift in AVX2
    let [a,b] : [i64x2; 2] = cast(self);
    cast([a.shr(rhs), b.shr(rhs)])
  }

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitand_m256i(self.avx2, rhs.avx2) }
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
    if #[cfg(target_feature="avx2")] {
            Self { avx2: bitor_m256i(self.avx2, rhs.avx2) }
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
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitxor_m256i(self.avx2, rhs.avx2) }
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
    self.simd_gt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    self.simd_lt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn reduce_add(self) -> i64 {
    pick! {
      if #[cfg(all(target_arch="x86_64", target_feature="avx2"))] {
        let zwxx  = shuffle_ai_i64_all_m256i::<0b00_00_11_10>(self.avx2);
        let xz_yw = add_i64_m256i(zwxx, self.avx2);
        let yw_xz  = shuffle_ai_i64_all_m256i::<0b00_00_00_01>(xz_yw);
        let sum = add_i64_m256i(xz_yw, yw_xz);
        extract_i64_from_m256i::<0>(sum)
      } else {
        let array: [i64; 4] = cast(self);
        array[0]
          .wrapping_add(array[1])
          .wrapping_add(array[2])
          .wrapping_add(array[3])
      }
    }
  }

  #[inline]
  pub fn reduce_mul(self) -> i64 {
    let array: [i64; 4] = cast(self);
    array[0]
      .wrapping_mul(array[1])
      .wrapping_mul(array[2])
      .wrapping_mul(array[3])
  }

  #[inline]
  pub fn reduce_max(self) -> i64 {
    let array: [i64; 4] = cast(self);
    array[0].max(array[1]).max(array[2]).max(array[3])
  }

  #[inline]
  pub fn reduce_min(self) -> i64 {
    let array: [i64; 4] = cast(self);
    array[0].min(array[1]).min(array[2]).min(array[3])
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
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
      if #[cfg(target_feature="avx2")] {
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
    ];
    (
      Self::new([result[0].0, result[1].0, result[2].0, result[3].0]),
      Self::new([
        -(result[0].1 as i64),
        -(result[1].1 as i64),
        -(result[2].1 as i64),
        -(result[3].1 as i64),
      ]),
    )
  }

  optional_fn_widening_mul {
    // Cannot have `widening_mul` because there is no `i128x4` type.
  }

  #[inline]
  pub fn mul_keep_low_high(self, rhs: Self) -> (u64x4, i64x4) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let widening_mul = [
      (self_array[0] as i128).wrapping_mul(rhs_array[0] as i128),
      (self_array[1] as i128).wrapping_mul(rhs_array[1] as i128),
      (self_array[2] as i128).wrapping_mul(rhs_array[2] as i128),
      (self_array[3] as i128).wrapping_mul(rhs_array[3] as i128),
    ];

    (
      u64x4::new([
        widening_mul[0] as u64,
        widening_mul[1] as u64,
        widening_mul[2] as u64,
        widening_mul[3] as u64,
      ]),
      i64x4::new([
        (widening_mul[0] >> 64) as i64,
        (widening_mul[1] >> 64) as i64,
        (widening_mul[2] >> 64) as i64,
        (widening_mul[3] >> 64) as i64,
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
    ])
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // avx x86 doesn't have this builtin
        let arr: [i64; 4] = cast(self);
        cast(
          [
            arr[0].wrapping_abs(),
            arr[1].wrapping_abs(),
            arr[2].wrapping_abs(),
            arr[3].wrapping_abs(),
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

/// The following functionality exists only for [`i64x4`], or only for
/// particular types inconsistently.
impl i64x4 {
  /// Converts each element from [`i64`] to [`f64`].
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f64x4 {
    let arr: [i64; 4] = cast(self);
    cast([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64])
  }

  // Sometimes used for `transpose`.
  #[must_use]
  #[inline]
  #[allow(dead_code)]
  pub(crate) fn unpack_lo(self, b: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let [aa, _]: [i64x2; 2] = cast(self);
        let [ba, _]: [i64x2; 2] = cast(b);
        cast([aa.unpack_lo(ba), aa.unpack_hi(ba)])
      } else {
        Self { a: self.a.unpack_lo(b.a), b: self.a.unpack_hi(b.a) }
      }
    }
  }

  // Sometimes used for `transpose`.
  #[must_use]
  #[inline]
  #[allow(dead_code)]
  pub(crate) fn unpack_hi(self, b: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let [_, ab]: [i64x2; 2] = cast(self);
        let [_, bb]: [i64x2; 2] = cast(b);
        cast([ab.unpack_lo(bb), ab.unpack_hi(bb)])
      } else {
        Self { a: self.b.unpack_lo(b.b), b: self.b.unpack_hi(b.b) }
      }
    }
  }
}
