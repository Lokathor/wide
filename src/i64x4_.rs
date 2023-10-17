use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { a : i64x2, b : i64x2 }
  }
}

int_uint_consts!(i64, 4, i64x4, i64x4, i64a4, const_i64_as_i64x4, 256);

unsafe impl Zeroable for i64x4 {}
unsafe impl Pod for i64x4 {}

impl Add for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

impl Sub for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

impl Mul for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

impl Add<i64> for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: i64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i64> for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i64> for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i64x4> for i64 {
  type Output = i64x4;
  #[inline]
  #[must_use]
  fn add(self, rhs: i64x4) -> Self::Output {
    i64x4::splat(self).add(rhs)
  }
}

impl Sub<i64x4> for i64 {
  type Output = i64x4;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i64x4) -> Self::Output {
    i64x4::splat(self).sub(rhs)
  }
}

impl Mul<i64x4> for i64 {
  type Output = i64x4;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i64x4) -> Self::Output {
    i64x4::splat(self).mul(rhs)
  }
}

impl BitAnd for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

impl BitOr for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

impl BitXor for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
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
}

macro_rules! impl_shl_t_for_i64x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i64x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shl_all_u64_m256i(self.avx2, shift) }
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
impl_shl_t_for_i64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i64x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i64x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shr_all_u64_m256i(self.avx2, shift) }
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
impl_shr_t_for_i64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i64_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.cmp_eq(rhs.a),
          b : self.b.cmp_eq(rhs.b),
        }
      }
    }
  }
}

impl CmpGt for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i64_m256i(self.avx2, rhs.avx2) }
      } else {
        Self {
          a : self.a.cmp_gt(rhs.a),
          b : self.b.cmp_gt(rhs.b),
        }
      }
    }
  }
}

impl CmpLt for i64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: !(cmp_gt_mask_i64_m256i(self.avx2, rhs.avx2) ^ cmp_eq_mask_i64_m256i(self.avx2, rhs.avx2)) }
      } else {
        Self {
          a : self.a.cmp_lt(rhs.a),
          b : self.b.cmp_lt(rhs.b),
        }
      }
    }
  }
}

impl i64x4 {
  #[inline]
  #[must_use]
  pub fn new(array: [i64; 4]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(f.avx2,t.avx2,self.avx2) }
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
  pub fn round_float(self) -> f64x4 {
    let arr: [i64; 4] = cast(self);
    cast([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64])
  }

  #[inline]
  pub fn to_array(self) -> [i64; 4] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[i64; 4] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_array_mut(&mut self) -> &mut[i64; 4] {
    cast_mut(self)
  }
}

impl Not for i64x4 {
  type Output = Self;
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
}
