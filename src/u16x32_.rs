use super::*;

pick! {
  if #[cfg(target_feature="avx512bw")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct u16x32 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct u16x32 { pub(crate) a : u16x16, pub(crate) b : u16x16 }
  }
}

int_uint_consts!(u16, 32, u16x32, 512);

unsafe impl Zeroable for u16x32 {}
unsafe impl Pod for u16x32 {}

impl AlignTo for u16x32 {
  type Elem = u16;
}

impl Add for u16x32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: add_i16_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }
}

impl Sub for u16x32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: sub_i16_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }
}

impl Mul for u16x32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: mul_i16_keep_low_m512i(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }
}

integer_impl_div_rem!(
  u16,
  u16x32,
  [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ],
);

impl Shl for u16x32 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shl`)
  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Mask `rhs` to 15 to match `wrapping_shl`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i16_m512i(15));
        Self { avx512: shl_each_u16_m512i(self.avx512, rhs) }
      } else {
        let [self_a, self_b]: [u16x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x16; 2] = cast(rhs);

        cast([self_a << rhs_a, self_b << rhs_b])
      }
    }
  }
}

impl Shr for u16x32 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shr`)
  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        // Mask `rhs` to 15 to match `wrapping_shr`.
        let rhs = bitand_m512i(rhs.avx512, set_splat_i16_m512i(15));
        Self { avx512: shr_each_u16_m512i(self.avx512, rhs) }
      } else {
        let [self_a, self_b]: [u16x16; 2] = cast(self);
        let [rhs_a, rhs_b]: [u16x16; 2] = cast(rhs);

        cast([self_a >> rhs_a, self_b >> rhs_b])
      }
    }
  }
}

impl Add<u16> for u16x32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: u16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u16> for u16x32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: u16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u16> for u16x32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: u16) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u16x32> for u16 {
  type Output = u16x32;
  #[inline]
  fn add(self, rhs: u16x32) -> Self::Output {
    u16x32::splat(self).add(rhs)
  }
}

impl Sub<u16x32> for u16 {
  type Output = u16x32;
  #[inline]
  fn sub(self, rhs: u16x32) -> Self::Output {
    u16x32::splat(self).sub(rhs)
  }
}

impl Mul<u16x32> for u16 {
  type Output = u16x32;
  #[inline]
  fn mul(self, rhs: u16x32) -> Self::Output {
    u16x32::splat(self).mul(rhs)
  }
}

impl BitAnd for u16x32 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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

impl BitOr for u16x32 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512bw")] {
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

impl BitXor for u16x32 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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

macro_rules! impl_shl_t_for_u16x32 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u16x32 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            #[expect(clippy::suspicious_arithmetic_impl)]
            let shift = rhs as u16 & 15;
            Self { avx512: shl_all_u16_m512i(self.avx512, shift) }
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
impl_shl_t_for_u16x32!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u16x32 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u16x32 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512bw")] {
            // Use `rhs % 16` to perform wrapping shift and not unbounded shift.
            #[expect(clippy::suspicious_arithmetic_impl)]
            let shift = rhs as u16 & 15;
            Self { avx512: shr_all_u16_m512i(self.avx512, shift) }
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
impl_shr_t_for_u16x32!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

#[expect(deprecated)]
impl CmpEq for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_u16_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
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
impl CmpLt for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_u16_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
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
impl CmpGt for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_u16_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
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
impl CmpNe for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_u16_m512i::<{cmp_int_op!(Ne)}>(self.avx512, rhs.avx512) }
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
impl CmpLe for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_u16_m512i::<{cmp_int_op!(Le)}>(self.avx512, rhs.avx512) }
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
impl CmpGe for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: cmp_op_mask_u16_m512i::<{cmp_int_op!(Nlt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ge(rhs.a),
          b : self.b.simd_ge(rhs.b),
        }
      }
    }
  }
}

impl Not for u16x32 {
  type Output = Self;
  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: bitxor_m512i(self.avx512, set_splat_i16_m512i(-1)) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }
}

impl u16x32 {
  #[inline]
  #[must_use]
  pub const fn new(array: [u16; 32]) -> Self {
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
      if #[cfg(target_feature="avx512bw")] {
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

  /// Lanewise selection.
  ///
  /// For each lane of `self`:
  ///
  /// - If all bits are one, return the corresponding lane of `if_true`
  /// - If all bits are zero, return the corresponding lane of `if_false`
  ///
  /// This function assumes `self` is a mask, meaning each lane is either all
  /// zeros or all ones. For bitwise selection use [`bitselect`].
  ///
  /// [`bitselect`]: Self::bitselect
  #[inline]
  #[must_use]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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
  #[must_use]
  pub fn reduce_add(self) -> u16 {
    cast(i16x32::reduce_add(cast(self)))
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> u16 {
    let array: [u16x16; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: min_u16_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: max_u16_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: add_saturating_u16_m512i(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512bw")] {
        Self { avx512: sub_saturating_u16_m512i(self.avx512, rhs.avx512) }
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
    let [self_a, self_b]: [u16x16; 2] = cast(self);
    let [rhs_a, rhs_b]: [u16x16; 2] = cast(rhs);
    cast([self_a.saturating_mul(rhs_a), self_b.saturating_mul(rhs_b)])
  }

  integer_fn_saturating_div!([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);

  unsigned_fn_overflowing_add_sub!();

  /// Returns `self * rhs` and whether an overflow occured.
  ///
  /// Returns a tuple with:
  ///
  /// - The multiplication (returns the wrapped value if an overflow occured)
  /// - A mask indicating whether an overflow occured
  #[inline]
  #[must_use]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // x86 has no `_mm512_mul_epu16` intrinsic so there is no `avx512`
    // optimization.

    let [self_a, self_b] = cast::<u16x32, [u16x16; 2]>(self);
    let [rhs_a, rhs_b] = cast::<u16x32, [u16x16; 2]>(rhs);

    let result_a = self_a.overflowing_mul(rhs_a);
    let result_b = self_b.overflowing_mul(rhs_b);
    (cast([result_a.0, result_b.0]), cast([result_a.1, result_b.1]))
  }

  unsigned_fn_overflowing_div_rem!();

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> u32 {
    i16x32::to_bitmask(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    i16x32::any(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    i16x32::all(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transpose matrix of 32x32 `u16` matrix. Currently not accelerated.
  #[must_use]
  #[inline]
  pub fn transpose(data: [u16x32; 32]) -> [u16x32; 32] {
    cast(i16x32::transpose(cast(data)))
  }

  #[inline]
  pub fn to_array(self) -> [u16; 32] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[u16; 32] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [u16; 32] {
    cast_mut(self)
  }

  fn_blend!();
}
