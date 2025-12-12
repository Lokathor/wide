use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i32x16 { pub(crate) avx512: m512i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(64))]
    pub struct i32x16 { pub(crate) a : i32x8, pub(crate) b : i32x8 }
  }
}

int_uint_consts!(i32, 16, i32x16, 512);

unsafe impl Zeroable for i32x16 {}
unsafe impl Pod for i32x16 {}

impl AlignTo for i32x16 {
  type Elem = i32;
}

impl Add for i32x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: add_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }
}

impl Sub for i32x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sub_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }
}

impl Mul for i32x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: mul_i32_keep_low_m512i(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }
}

impl Add<i32> for i32x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: i32) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i32> for i32x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: i32) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i32> for i32x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: i32) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i32x16> for i32 {
  type Output = i32x16;
  #[inline]
  fn add(self, rhs: i32x16) -> Self::Output {
    i32x16::splat(self).add(rhs)
  }
}

impl Sub<i32x16> for i32 {
  type Output = i32x16;
  #[inline]
  fn sub(self, rhs: i32x16) -> Self::Output {
    i32x16::splat(self).sub(rhs)
  }
}

impl Mul<i32x16> for i32 {
  type Output = i32x16;
  #[inline]
  fn mul(self, rhs: i32x16) -> Self::Output {
    i32x16::splat(self).mul(rhs)
  }
}

impl BitAnd for i32x16 {
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

impl BitOr for i32x16 {
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

impl BitXor for i32x16 {
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

macro_rules! impl_shl_t_for_i32x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i32x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u32);
            Self { avx512: shl_all_u32_m512i(self.avx512, shift) }
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
impl_shl_t_for_i32x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i32x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i32x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx512f")] {
            let shift = cast(rhs as u32);
            Self { avx512: shr_all_i32_m512i(self.avx512, shift) }
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
impl_shr_t_for_i32x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for i32x16 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    Self::simd_eq(self, rhs)
  }
}

impl CmpLt for i32x16 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    Self::simd_lt(self, rhs)
  }
}

impl CmpGt for i32x16 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    Self::simd_gt(self, rhs)
  }
}

impl i32x16 {
  #[inline]
  #[must_use]
  pub const fn new(array: [i32; 16]) -> Self {
    unsafe { core::mem::transmute(array) }
  }

  #[inline]
  #[must_use]
  pub fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Eq)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Nle)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_gt(rhs.a),
          b : self.b.simd_gt(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn simd_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_i32_m512i::<{cmp_int_op!(Lt)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : rhs.a.simd_gt(self.a),
          b : rhs.b.simd_gt(self.b),
        }
      }
    }
  }

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
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_i32_m512i(self.avx512, rhs.avx512) }
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
        Self { avx512: max_i32_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.max(rhs.a),
          b: self.b.max(rhs.b),
        }
      }
    }
  }
  
  #[inline]
  #[must_use]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512dq")] {
        movepi32_mask_m512i(self.avx512) as u32
      } else {
        self.a.to_bitmask() | (self.b.to_bitmask() << 8)
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [i32; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[i32; 16] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [i32; 16] {
    cast_mut(self)
  }

  #[inline]
  #[must_use]
  pub fn round_float(self) -> f32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        cast(convert_to_m512_from_i32_m512i(self.avx512))
      } else {
        f32x16 {
          a: self.a.round_float(),
          b: self.b.round_float(),
        }
      }
    }
  }
}

impl Not for i32x16 {
  type Output = Self;
  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512i(self.avx512, set_splat_i32_m512i(-1)) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }
}
