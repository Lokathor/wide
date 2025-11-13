use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { pub(crate) avx2: m256i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { pub(crate) a : u64x2, pub(crate) b : u64x2 }
  }
}

int_uint_consts!(u64, 4, u64x4, 256);

unsafe impl Zeroable for u64x4 {}
unsafe impl Pod for u64x4 {}

impl AlignTo for u64x4 {
  type Elem = u64;
}

impl Add for u64x4 {
  type Output = Self;
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
}

impl Sub for u64x4 {
  type Output = Self;
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
}

impl Mul for u64x4 {
  type Output = Self;
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
}

impl Add<u64> for u64x4 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: u64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u64> for u64x4 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: u64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u64> for u64x4 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: u64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u64x4> for u64 {
  type Output = u64x4;
  #[inline]
  fn add(self, rhs: u64x4) -> Self::Output {
    u64x4::splat(self).add(rhs)
  }
}

impl Sub<u64x4> for u64 {
  type Output = u64x4;
  #[inline]
  fn sub(self, rhs: u64x4) -> Self::Output {
    u64x4::splat(self).sub(rhs)
  }
}

impl Mul<u64x4> for u64 {
  type Output = u64x4;
  #[inline]
  fn mul(self, rhs: u64x4) -> Self::Output {
    u64x4::splat(self).mul(rhs)
  }
}

impl BitAnd for u64x4 {
  type Output = Self;
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
}

impl BitOr for u64x4 {
  type Output = Self;
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
}

impl BitXor for u64x4 {
  type Output = Self;
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
}

/// Shifts lanes by the corresponding lane.
///
/// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
/// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
/// of the type. (same as `wrapping_shl`)
impl Shl for u64x4 {
  type Output = Self;

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
}

macro_rules! impl_shl_t_for_u64x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u64x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
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
impl_shl_t_for_u64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

/// Shifts lanes by the corresponding lane.
///
/// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
/// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
/// of the type. (same as `wrapping_shr`)
impl Shr for u64x4 {
  type Output = Self;

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self { avx2: shr_each_u64_m256i(self.avx2, shift_by.avx2) }
      } else {
        Self {
          a : self.a.shr(rhs.a),
          b : self.b.shr(rhs.b),
        }
      }
    }
  }
}

macro_rules! impl_shr_t_for_u64x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u64x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
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
impl_shr_t_for_u64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for u64x4 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    Self::simd_eq(self, rhs)
  }
}

impl CmpGt for u64x4 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    Self::simd_gt(self, rhs)
  }
}

impl CmpLt for u64x4 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    // no lt, so just call gt with swapped args
    Self::simd_gt(rhs, self)
  }
}

impl u64x4 {
  #[inline]
  #[must_use]
  pub const fn new(array: [u64; 4]) -> Self {
    unsafe { core::mem::transmute(array) }
  }
  #[inline]
  #[must_use]
  pub fn simd_eq(self, rhs: Self) -> Self {
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
  #[must_use]
  pub fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // no unsigned gt than so inverting the high bit will get the correct result
        let highbit = u64x4::splat(1 << 63);
        Self { avx2: cmp_gt_mask_i64_m256i((self ^ highbit).avx2, (rhs ^ highbit).avx2) }
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
    // lt is just gt the other way around
    rhs.simd_gt(self)
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
  pub fn to_bitmask(self) -> u32 {
    i64x4::to_bitmask(cast(self))
  }

  #[inline]
  pub fn to_array(self) -> [u64; 4] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[u64; 4] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [u64; 4] {
    cast_mut(self)
  }

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    self.simd_lt(rhs).blend(self, rhs)
  }

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    self.simd_gt(rhs).blend(self, rhs)
  }

  #[inline]
  #[must_use]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let arr1: [u64; 4] = cast(self);
        let arr2: [u64; 4] = cast(rhs);
        cast([
          (arr1[0] as u128 * arr2[0] as u128 >> 64) as u64,
          (arr1[1] as u128 * arr2[1] as u128 >> 64) as u64,
          (arr1[2] as u128 * arr2[2] as u128 >> 64) as u64,
          (arr1[3] as u128 * arr2[3] as u128 >> 64) as u64,
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

impl Not for u64x4 {
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
