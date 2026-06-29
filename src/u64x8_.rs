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

int_uint_consts!(u64, 8, u64x8, 512);

unsafe impl Zeroable for u64x8 {}
unsafe impl Pod for u64x8 {}

impl AlignTo for u64x8 {
  type Elem = u64;
}

impl Add for u64x8 {
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

impl Sub for u64x8 {
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

impl Mul for u64x8 {
  type Output = Self;
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
}

integer_impl_div_rem!(u64, u64x8, [0, 1, 2, 3, 4, 5, 6, 7]);

impl Add<u64> for u64x8 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: u64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u64> for u64x8 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: u64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u64> for u64x8 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: u64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u64x8> for u64 {
  type Output = u64x8;
  #[inline]
  fn add(self, rhs: u64x8) -> Self::Output {
    u64x8::splat(self).add(rhs)
  }
}

impl Sub<u64x8> for u64 {
  type Output = u64x8;
  #[inline]
  fn sub(self, rhs: u64x8) -> Self::Output {
    u64x8::splat(self).sub(rhs)
  }
}

impl Mul<u64x8> for u64 {
  type Output = u64x8;
  #[inline]
  fn mul(self, rhs: u64x8) -> Self::Output {
    u64x8::splat(self).mul(rhs)
  }
}

impl BitAnd for u64x8 {
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

impl BitOr for u64x8 {
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

impl BitXor for u64x8 {
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

macro_rules! impl_shl_t_for_u64x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u64x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
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
    })+
  };
}
impl_shl_t_for_u64x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u64x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u64x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
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
    })+
  };
}
impl_shr_t_for_u64x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl Shr for u64x8 {
  type Output = Self;

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
}

impl Shl for u64x8 {
  type Output = Self;

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
}

#[expect(deprecated)]
impl CmpEq for u64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGt for u64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLt for u64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpNe for u64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpLe for u64x8 {
  type Output = Self;
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
}

#[expect(deprecated)]
impl CmpGe for u64x8 {
  type Output = Self;
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
}

impl u64x8 {
  #[inline]
  #[must_use]
  pub const fn new(array: [u64; 8]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  simd_comparison_fns!();

  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_i8_m512i(f.avx512,t.avx512,movepi8_mask_m512i(self.avx512)) }
      } else {
        Self {
          a : self.a.blend(t.a, f.a),
          b : self.b.blend(t.b, f.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] + array[1]).reduce_add()
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    (array[0] * array[1]).reduce_mul()
  }

  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].max(array[1]).reduce_max()
  }

  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> u64 {
    let array: [u64x4; 2] = cast(self);
    array[0].min(array[1]).reduce_min()
  }

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> u32 {
    i64x8::to_bitmask(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    i64x8::any(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    i64x8::all(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Transpose matrix of 8x8 `u64` matrix. Currently not accelerated.
  #[must_use]
  #[inline]
  pub fn transpose(data: [u64x8; 8]) -> [u64x8; 8] {
    cast(i64x8::transpose(cast(data)))
  }

  #[inline]
  pub fn to_array(self) -> [u64; 8] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[u64; 8] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [u64; 8] {
    cast_mut(self)
  }

  #[inline]
  #[must_use]
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
  #[must_use]
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

  integer_fn_clamp!();

  #[inline]
  #[must_use]
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
  #[must_use]
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

impl Not for u64x8 {
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
