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
            let shift = cast(rhs as u16);
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
            let shift = cast(rhs as u16);
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

impl CmpEq for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    Self::simd_eq(self, rhs)
  }
}

impl CmpLt for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    Self::simd_lt(self, rhs)
  }
}

impl CmpGt for u16x32 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    Self::simd_gt(self, rhs)
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

  #[inline]
  #[must_use]
  pub fn simd_eq(self, rhs: Self) -> Self {
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

  #[inline]
  #[must_use]
  pub fn simd_gt(self, rhs: Self) -> Self {
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

  #[inline]
  #[must_use]
  pub fn simd_lt(self, rhs: Self) -> Self {
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

  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512bw")] {
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
  
  #[inline]
  #[must_use]
  pub fn to_bitmask(self) -> u32 {
    i16x32::to_bitmask(cast(self))
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
}
