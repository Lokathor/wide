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

int_uint_consts!(i64, 8, i64x8, 512);

unsafe impl Zeroable for i64x8 {}
unsafe impl Pod for i64x8 {}

impl AlignTo for i64x8 {
  type Elem = i64;
}

impl Add for i64x8 {
  type Output = Self;
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
}

impl Sub for i64x8 {
  type Output = Self;
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
}

impl Mul for i64x8 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let arr1: [i64; 8] = cast(self);
        let arr2: [i64; 8] = cast(rhs);
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
}

integer_impl_div_rem!(i64, i64x8, [0, 1, 2, 3, 4, 5, 6, 7]);

impl Shr for i64x8 {
  type Output = Self;

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // TODO(safe_arch): add shr_each_i64_m512i (arithmetic right shift)
        // Self { avx512: shr_each_i64_m512i(self.avx512, rhs.avx512) }
        // Fallback for now:
        let a: [i64; 8] = cast(self);
        let r: [i64; 8] = cast(rhs);
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
}

impl Shl for i64x8 {
  type Output = Self;

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // TODO(safe_arch): add shl_each_i64_m512i
        // Self { avx512: shl_each_i64_m512i(self.avx512, rhs.avx512) }
        // Fallback for now:
        let a: [i64; 8] = cast(self);
        let r: [i64; 8] = cast(rhs);
        cast([
          a[0].wrapping_shl(r[0] as u32),
          a[1].wrapping_shl(r[1] as u32),
          a[2].wrapping_shl(r[2] as u32),
          a[3].wrapping_shl(r[3] as u32),
          a[4].wrapping_shl(r[4] as u32),
          a[5].wrapping_shl(r[5] as u32),
          a[6].wrapping_shl(r[6] as u32),
          a[7].wrapping_shl(r[7] as u32),
        ])
      } else {
        // widen via two halves
        Self {
          a: self.a.shl(rhs.a),
          b: self.b.shl(rhs.b),
        }
      }
    }
  }
}

impl Add<i64> for i64x8 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: i64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i64> for i64x8 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: i64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i64> for i64x8 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: i64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i64x8> for i64 {
  type Output = i64x8;
  #[inline]
  fn add(self, rhs: i64x8) -> Self::Output {
    i64x8::splat(self).add(rhs)
  }
}

impl Sub<i64x8> for i64 {
  type Output = i64x8;
  #[inline]
  fn sub(self, rhs: i64x8) -> Self::Output {
    i64x8::splat(self).sub(rhs)
  }
}

impl Mul<i64x8> for i64 {
  type Output = i64x8;
  #[inline]
  fn mul(self, rhs: i64x8) -> Self::Output {
    i64x8::splat(self).mul(rhs)
  }
}

impl BitAnd for i64x8 {
  type Output = Self;
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
}

impl BitOr for i64x8 {
  type Output = Self;
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
}

impl BitXor for i64x8 {
  type Output = Self;
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
}

macro_rules! impl_shl_t_for_i64x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i64x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u64);
            Self { avx512: shl_all_u64_m512i(self.avx512, shift) }
          } else {
            Self {
              a : self.a.shl(rhs),
              b : self.b.shl(rhs),
            }
          }
        }
      }
    })+
  };
}
impl_shl_t_for_i64x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i64x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i64x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u64);
            Self { avx512: shr_all_i64_m512i(self.avx512, shift) }
          } else {
            Self {
              a : self.a.shr(rhs),
              b : self.b.shr(rhs),
            }
          }
        }
      }
    })+
  };
}
impl_shr_t_for_i64x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

#[expect(deprecated)]
impl CmpEq for i64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGt for i64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLt for i64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpNe for i64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLe for i64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGe for i64x8 {
  type Output = Self;
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
}

impl i64x8 {
  #[inline]
  #[must_use]
  pub const fn new(array: [i64; 8]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  simd_comparison_fns!();

  /// Bitwise selection.
  ///
  /// For each bit of `self`:
  ///
  /// - If the bit is one, return the corresponding bit of `if_one`
  /// - If the bit is zero, return the corresponding bit of `if_zero`
  ///
  /// If you know `self` is a mask, meaning each lane is either all zeros or all
  /// ones, consider using [`select`] which is faster.
  ///
  /// [`select`]: Self::select
  #[inline]
  #[must_use]
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

  /// Lanewise selection.
  ///
  /// For each lane this returns `t` where `self` is all ones and `f` where
  /// `self` is all zeros.
  ///
  /// This function assumes `self` is a mask, meaning each lane is either all
  /// zeros or all ones. For bitwise selection use [`bitselect`].
  ///
  /// [`bitselect`]: Self::bitselect
  #[inline]
  #[must_use]
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

  /// Returns true for each positive element and false if it is zero or
  /// negative.
  #[inline]
  #[must_use]
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

  /// Returns true for each negative element and false if it is zero or
  /// positive.
  #[inline]
  #[must_use]
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

  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> i64 {
    let array: [i64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  #[must_use]
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
  #[must_use]
  pub fn unsigned_abs(self) -> u64x8 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // AVX512 might have this, unsure for now
        let arr: [i64; 8] = cast(self);
        cast(
          [
            arr[0].unsigned_abs(),
            arr[1].unsigned_abs(),
            arr[2].unsigned_abs(),
            arr[3].unsigned_abs(),
            arr[4].unsigned_abs(),
            arr[5].unsigned_abs(),
            arr[6].unsigned_abs(),
            arr[7].unsigned_abs(),
          ])
      } else {
        u64x8 {
          a : self.a.unsigned_abs(),
          b : self.b.unsigned_abs(),
        }
      }
    }
  }

  signed_fn_signum!();

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

  /// returns the bit mask for each high bit set in the vector with the lowest
  /// lane being the lowest bit
  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
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
  #[must_use]
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
  #[must_use]
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

  /// true if no high bits are set for any values of the vector
  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transpose matrix of 8x8 `i64` matrix. Currently not accelerated.
  #[must_use]
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

  #[inline]
  pub fn to_array(self) -> [i64; 8] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[i64; 8] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [i64; 8] {
    cast_mut(self)
  }

  #[inline]
  #[must_use]
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
  #[must_use]
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

  integer_fn_clamp!();

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        overflow.select(negative.select(Self::MIN, Self::MAX), result)
      } else {
        Self {
          a: self.a.saturating_add(rhs.a),
          b: self.b.saturating_add(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        overflow.select(negative.select(Self::MIN, Self::MAX), result)
      } else {
        Self {
          a: self.a.saturating_sub(rhs.a),
          b: self.b.saturating_sub(rhs.b),
        }
      }
    }
  }

  /// Lanewise saturating multiply.
  #[inline]
  #[must_use]
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

  integer_fn_saturating_div!([0, 1, 2, 3, 4, 5, 6, 7]);

  fn_blend!();
}

impl Not for i64x8 {
  type Output = Self;
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
}
