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
            let shift = cast([rhs as u64, 0]);
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
            let shift = cast([rhs as u64, 0]);
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

impl CmpEq for i64x8 {
  type Output = Self;
  #[inline]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    Self::cmp_eq(self, rhs)
  }
}

impl i64x8 {
  #[inline]
  #[must_use]
  pub const fn new(array: [i64; 8]) -> Self {
    unsafe { core::mem::transmute(array) }
  }
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_eq_mask_i64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_eq(rhs.a),
          b : self.b.cmp_eq(rhs.b),
        }
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_gt_mask_i64_m512i(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_gt(rhs.a),
          b : self.b.cmp_gt(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    // lt is just gt the other way around
    rhs.cmp_gt(self)
  }

  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_i8_m512i(f.avx512,t.avx512,self.avx512) }
      } else {
        Self {
          a : self.a.blend(t.a, f.a),
          b : self.b.blend(t.b, f.b),
        }
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [i64; 8] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[i64; 8] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_array_mut(&mut self) -> &mut [i64; 8] {
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
