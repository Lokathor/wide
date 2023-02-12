use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i16x16 { avx2: m256i }
  } else if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i16x16 { pub(crate) sse0: m128i, pub(crate) sse1: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(C, align(32))]
    pub struct i16x16 { simd0: v128, simd1: v128 }

    impl Default for i16x16 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i16x16 {
      fn eq(&self, other: &Self) -> bool {
        !v128_any_true(v128_or(v128_xor(self.simd0, other.simd0), v128_xor(self.simd1, other.simd1)))
      }
    }

    impl Eq for i16x16 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i16x16 { arr: [i16;16] }
  }
}

int_uint_consts!(i16, 16, i16x16, i16x16, i16a16, const_i16_as_i16x16, 256);

unsafe impl Zeroable for i16x16 {}
unsafe impl Pod for i16x16 {}

impl Add for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self {
          sse0: add_i16_m128i(self.sse0, rhs.sse0),
          sse1: add_i16_m128i(self.sse1, rhs.sse1)
        }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_add(self.simd0, rhs.simd0),
          simd1: i16x8_add(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          self.arr[0].wrapping_add(rhs.arr[0]),
          self.arr[1].wrapping_add(rhs.arr[1]),
          self.arr[2].wrapping_add(rhs.arr[2]),
          self.arr[3].wrapping_add(rhs.arr[3]),
          self.arr[4].wrapping_add(rhs.arr[4]),
          self.arr[5].wrapping_add(rhs.arr[5]),
          self.arr[6].wrapping_add(rhs.arr[6]),
          self.arr[7].wrapping_add(rhs.arr[7]),
          self.arr[8].wrapping_add(rhs.arr[8]),
          self.arr[9].wrapping_add(rhs.arr[9]),
          self.arr[10].wrapping_add(rhs.arr[10]),
          self.arr[11].wrapping_add(rhs.arr[11]),
          self.arr[12].wrapping_add(rhs.arr[12]),
          self.arr[13].wrapping_add(rhs.arr[13]),
          self.arr[14].wrapping_add(rhs.arr[14]),
          self.arr[15].wrapping_add(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl Sub for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_i16_m128i(self.sse0, rhs.sse0),
          sse1: sub_i16_m128i(self.sse1, rhs.sse1)  }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_sub(self.simd0, rhs.simd0),
          simd1: i16x8_sub(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          self.arr[0].wrapping_sub(rhs.arr[0]),
          self.arr[1].wrapping_sub(rhs.arr[1]),
          self.arr[2].wrapping_sub(rhs.arr[2]),
          self.arr[3].wrapping_sub(rhs.arr[3]),
          self.arr[4].wrapping_sub(rhs.arr[4]),
          self.arr[5].wrapping_sub(rhs.arr[5]),
          self.arr[6].wrapping_sub(rhs.arr[6]),
          self.arr[7].wrapping_sub(rhs.arr[7]),
          self.arr[8].wrapping_sub(rhs.arr[8]),
          self.arr[9].wrapping_sub(rhs.arr[9]),
          self.arr[10].wrapping_sub(rhs.arr[10]),
          self.arr[11].wrapping_sub(rhs.arr[11]),
          self.arr[12].wrapping_sub(rhs.arr[12]),
          self.arr[13].wrapping_sub(rhs.arr[13]),
          self.arr[14].wrapping_sub(rhs.arr[14]),
          self.arr[15].wrapping_sub(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl Mul for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: mul_i16_keep_low_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: mul_i16_keep_low_m128i(self.sse0, rhs.sse0),
          sse1: mul_i16_keep_low_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_mul(self.simd0, rhs.simd0),
          simd1: i16x8_mul(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          self.arr[0].wrapping_mul(rhs.arr[0]),
          self.arr[1].wrapping_mul(rhs.arr[1]),
          self.arr[2].wrapping_mul(rhs.arr[2]),
          self.arr[3].wrapping_mul(rhs.arr[3]),
          self.arr[4].wrapping_mul(rhs.arr[4]),
          self.arr[5].wrapping_mul(rhs.arr[5]),
          self.arr[6].wrapping_mul(rhs.arr[6]),
          self.arr[7].wrapping_mul(rhs.arr[7]),
          self.arr[8].wrapping_mul(rhs.arr[8]),
          self.arr[9].wrapping_mul(rhs.arr[9]),
          self.arr[10].wrapping_mul(rhs.arr[10]),
          self.arr[11].wrapping_mul(rhs.arr[11]),
          self.arr[12].wrapping_mul(rhs.arr[12]),
          self.arr[13].wrapping_mul(rhs.arr[13]),
          self.arr[14].wrapping_mul(rhs.arr[14]),
          self.arr[15].wrapping_mul(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl Add<i16> for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: i16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i16> for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i16> for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i16) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i16x16> for i16 {
  type Output = i16x16;
  #[inline]
  #[must_use]
  fn add(self, rhs: i16x16) -> Self::Output {
    i16x16::splat(self).add(rhs)
  }
}

impl Sub<i16x16> for i16 {
  type Output = i16x16;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i16x16) -> Self::Output {
    i16x16::splat(self).sub(rhs)
  }
}

impl Mul<i16x16> for i16 {
  type Output = i16x16;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i16x16) -> Self::Output {
    i16x16::splat(self).mul(rhs)
  }
}

impl BitAnd for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitand_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitand_m128i(self.sse0, rhs.sse0),
          sse1: bitand_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: v128_and(self.simd0, rhs.simd0),
          simd1: v128_and(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          self.arr[0].bitand(rhs.arr[0]),
          self.arr[1].bitand(rhs.arr[1]),
          self.arr[2].bitand(rhs.arr[2]),
          self.arr[3].bitand(rhs.arr[3]),
          self.arr[4].bitand(rhs.arr[4]),
          self.arr[5].bitand(rhs.arr[5]),
          self.arr[6].bitand(rhs.arr[6]),
          self.arr[7].bitand(rhs.arr[7]),
          self.arr[8].bitand(rhs.arr[8]),
          self.arr[9].bitand(rhs.arr[9]),
          self.arr[10].bitand(rhs.arr[10]),
          self.arr[11].bitand(rhs.arr[11]),
          self.arr[12].bitand(rhs.arr[12]),
          self.arr[13].bitand(rhs.arr[13]),
          self.arr[14].bitand(rhs.arr[14]),
          self.arr[15].bitand(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl BitOr for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitor_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitor_m128i(self.sse0, rhs.sse0),
          sse1: bitor_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: v128_or(self.simd0, rhs.simd0),
          simd1: v128_or(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          self.arr[0].bitor(rhs.arr[0]),
          self.arr[1].bitor(rhs.arr[1]),
          self.arr[2].bitor(rhs.arr[2]),
          self.arr[3].bitor(rhs.arr[3]),
          self.arr[4].bitor(rhs.arr[4]),
          self.arr[5].bitor(rhs.arr[5]),
          self.arr[6].bitor(rhs.arr[6]),
          self.arr[7].bitor(rhs.arr[7]),
          self.arr[8].bitor(rhs.arr[8]),
          self.arr[9].bitor(rhs.arr[9]),
          self.arr[10].bitor(rhs.arr[10]),
          self.arr[11].bitor(rhs.arr[11]),
          self.arr[12].bitor(rhs.arr[12]),
          self.arr[13].bitor(rhs.arr[13]),
          self.arr[14].bitor(rhs.arr[14]),
          self.arr[15].bitor(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl BitXor for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitxor_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitxor_m128i(self.sse0, rhs.sse0),
          sse1: bitxor_m128i(self.sse1, rhs.sse1)  }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: v128_xor(self.simd0, rhs.simd0),
          simd1: v128_xor(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          self.arr[0].bitxor(rhs.arr[0]),
          self.arr[1].bitxor(rhs.arr[1]),
          self.arr[2].bitxor(rhs.arr[2]),
          self.arr[3].bitxor(rhs.arr[3]),
          self.arr[4].bitxor(rhs.arr[4]),
          self.arr[5].bitxor(rhs.arr[5]),
          self.arr[6].bitxor(rhs.arr[6]),
          self.arr[7].bitxor(rhs.arr[7]),
          self.arr[8].bitxor(rhs.arr[8]),
          self.arr[9].bitxor(rhs.arr[9]),
          self.arr[10].bitxor(rhs.arr[10]),
          self.arr[11].bitxor(rhs.arr[11]),
          self.arr[12].bitxor(rhs.arr[12]),
          self.arr[13].bitxor(rhs.arr[13]),
          self.arr[14].bitxor(rhs.arr[14]),
          self.arr[15].bitxor(rhs.arr[15]),
        ]}
      }
    }
  }
}

macro_rules! impl_shl_t_for_i16x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i16x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shl_all_u16_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self {
              sse0: shl_all_u16_m128i(self.sse0, shift),
              sse1: shl_all_u16_m128i(self.sse1, shift)
            }
          } else if #[cfg(target_feature="simd128")] {
            let u = rhs as u32;
            Self {
              simd0: i16x8_shl(self.simd0, u),
              simd1: i16x8_shl(self.simd1, u)
            }
          } else {
            let u = rhs as u64;
            Self { arr: [
              self.arr[0] << u,
              self.arr[1] << u,
              self.arr[2] << u,
              self.arr[3] << u,
              self.arr[4] << u,
              self.arr[5] << u,
              self.arr[6] << u,
              self.arr[7] << u,
              self.arr[8] << u,
              self.arr[9] << u,
              self.arr[10] << u,
              self.arr[11] << u,
              self.arr[12] << u,
              self.arr[13] << u,
              self.arr[14] << u,
              self.arr[15] << u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shl_t_for_i16x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i16x16 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i16x16 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shr_all_i16_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse0: shr_all_i16_m128i(self.sse0, shift),
              sse1: shr_all_i16_m128i(self.sse1, shift) }
          } else if #[cfg(target_feature="simd128")] {
            let u = rhs as u32;
            Self {
              simd0: i16x8_shr(self.simd0, u),
              simd1: i16x8_shr(self.simd1, u)
            }
          } else {
            let u = rhs as u64;
            Self { arr: [
              self.arr[0] >> u,
              self.arr[1] >> u,
              self.arr[2] >> u,
              self.arr[3] >> u,
              self.arr[4] >> u,
              self.arr[5] >> u,
              self.arr[6] >> u,
              self.arr[7] >> u,
              self.arr[8] >> u,
              self.arr[9] >> u,
              self.arr[10] >> u,
              self.arr[11] >> u,
              self.arr[12] >> u,
              self.arr[13] >> u,
              self.arr[14] >> u,
              self.arr[15] >> u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shr_t_for_i16x16!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_eq_mask_i16_m128i(self.sse0, rhs.sse0),
          sse1: cmp_eq_mask_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_eq(self.simd0, rhs.simd0),
          simd1: i16x8_eq(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] == rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] == rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] == rhs.arr[3] { -1 } else { 0 },
          if self.arr[4] == rhs.arr[4] { -1 } else { 0 },
          if self.arr[5] == rhs.arr[5] { -1 } else { 0 },
          if self.arr[6] == rhs.arr[6] { -1 } else { 0 },
          if self.arr[7] == rhs.arr[7] { -1 } else { 0 },
          if self.arr[8] == rhs.arr[8] { -1 } else { 0 },
          if self.arr[9] == rhs.arr[9] { -1 } else { 0 },
          if self.arr[10] == rhs.arr[10] { -1 } else { 0 },
          if self.arr[11] == rhs.arr[11] { -1 } else { 0 },
          if self.arr[12] == rhs.arr[12] { -1 } else { 0 },
          if self.arr[13] == rhs.arr[13] { -1 } else { 0 },
          if self.arr[14] == rhs.arr[14] { -1 } else { 0 },
          if self.arr[15] == rhs.arr[15] { -1 } else { 0 },
        ]}
      }
    }
  }
}

impl CmpGt for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_gt_mask_i16_m128i(self.sse0, rhs.sse0),
          sse1: cmp_gt_mask_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_gt(self.simd0, rhs.simd0),
          simd1: i16x8_gt(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] > rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] > rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] > rhs.arr[3] { -1 } else { 0 },
          if self.arr[4] > rhs.arr[4] { -1 } else { 0 },
          if self.arr[5] > rhs.arr[5] { -1 } else { 0 },
          if self.arr[6] > rhs.arr[6] { -1 } else { 0 },
          if self.arr[7] > rhs.arr[7] { -1 } else { 0 },
          if self.arr[8] > rhs.arr[8] { -1 } else { 0 },
          if self.arr[9] > rhs.arr[9] { -1 } else { 0 },
          if self.arr[10] > rhs.arr[10] { -1 } else { 0 },
          if self.arr[11] > rhs.arr[11] { -1 } else { 0 },
          if self.arr[12] > rhs.arr[12] { -1 } else { 0 },
          if self.arr[13] > rhs.arr[13] { -1 } else { 0 },
          if self.arr[14] > rhs.arr[14] { -1 } else { 0 },
          if self.arr[15] > rhs.arr[15] { -1 } else { 0 },
        ]}
      }
    }
  }
}

impl CmpLt for i16x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: !cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) ^ cmp_eq_mask_i16_m256i(self.avx2,rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_lt_mask_i16_m128i(self.sse0, rhs.sse0),
          sse1: cmp_lt_mask_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_lt(self.simd0, rhs.simd0),
          simd1: i16x8_lt(self.simd1, rhs.simd1)
        }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] < rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] < rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] < rhs.arr[3] { -1 } else { 0 },
          if self.arr[4] < rhs.arr[4] { -1 } else { 0 },
          if self.arr[5] < rhs.arr[5] { -1 } else { 0 },
          if self.arr[6] < rhs.arr[6] { -1 } else { 0 },
          if self.arr[7] < rhs.arr[7] { -1 } else { 0 },
          if self.arr[8] < rhs.arr[8] { -1 } else { 0 },
          if self.arr[9] < rhs.arr[9] { -1 } else { 0 },
          if self.arr[10] < rhs.arr[10] { -1 } else { 0 },
          if self.arr[11] < rhs.arr[11] { -1 } else { 0 },
          if self.arr[12] < rhs.arr[12] { -1 } else { 0 },
          if self.arr[13] < rhs.arr[13] { -1 } else { 0 },
          if self.arr[14] < rhs.arr[14] { -1 } else { 0 },
          if self.arr[15] < rhs.arr[15] { -1 } else { 0 },
        ]}
      }
    }
  }
}

impl i16x16 {
  #[inline]
  #[must_use]
  pub fn new(array: [i16; 16]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(f.avx2, t.avx2, self.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self {
          sse0: blend_varying_i8_m128i(f.sse0, t.sse0, self.sse0),
          sse1: blend_varying_i8_m128i(f.sse1, t.sse1, self.sse1)
        }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: v128_bitselect(t.simd0, f.simd0, self.simd0),
          simd1: v128_bitselect(t.simd1, f.simd1, self.simd1)
        }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: abs_i16_m256i(self.avx2) }
      } else if #[cfg(target_feature="ssse3")] {
        Self { sse0: abs_i16_m128i(self.sse0),
          sse1: abs_i16_m128i(self.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_abs(self.simd0),
          simd1: i16x8_abs(self.simd1)
        }
      } else {
        let arr: [i16; 16] = cast(self);
        cast([
          arr[0].wrapping_abs(),
          arr[1].wrapping_abs(),
          arr[2].wrapping_abs(),
          arr[3].wrapping_abs(),
          arr[4].wrapping_abs(),
          arr[5].wrapping_abs(),
          arr[6].wrapping_abs(),
          arr[7].wrapping_abs(),
          arr[8].wrapping_abs(),
          arr[9].wrapping_abs(),
          arr[10].wrapping_abs(),
          arr[11].wrapping_abs(),
          arr[12].wrapping_abs(),
          arr[13].wrapping_abs(),
          arr[14].wrapping_abs(),
          arr[15].wrapping_abs(),
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: max_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: max_i16_m128i(self.sse0, rhs.sse0),
          sse1: max_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_max(self.simd0, rhs.simd0),
          simd1: i16x8_max(self.simd1, rhs.simd1)
        }
      } else {
        self.cmp_lt(rhs).blend(rhs, self)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: min_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: min_i16_m128i(self.sse0, rhs.sse0),
          sse1: min_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self {
          simd0: i16x8_min(self.simd0, rhs.simd0),
          simd1: i16x8_min(self.simd1, rhs.simd1)
        }
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_saturating_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: add_saturating_i16_m128i(self.sse0, rhs.sse0), sse1: add_saturating_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i16x8_add_sat(self.simd0, rhs.simd0), simd1: i16x8_add_sat(self.simd1, rhs.simd1) }
      } else {
        Self { arr: [
          self.arr[0].saturating_add(rhs.arr[0]),
          self.arr[1].saturating_add(rhs.arr[1]),
          self.arr[2].saturating_add(rhs.arr[2]),
          self.arr[3].saturating_add(rhs.arr[3]),
          self.arr[4].saturating_add(rhs.arr[4]),
          self.arr[5].saturating_add(rhs.arr[5]),
          self.arr[6].saturating_add(rhs.arr[6]),
          self.arr[7].saturating_add(rhs.arr[7]),
          self.arr[8].saturating_add(rhs.arr[8]),
          self.arr[9].saturating_add(rhs.arr[9]),
          self.arr[10].saturating_add(rhs.arr[10]),
          self.arr[11].saturating_add(rhs.arr[11]),
          self.arr[12].saturating_add(rhs.arr[12]),
          self.arr[13].saturating_add(rhs.arr[13]),
          self.arr[14].saturating_add(rhs.arr[14]),
          self.arr[15].saturating_add(rhs.arr[15]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_saturating_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_saturating_i16_m128i(self.sse0, rhs.sse0), sse1: sub_saturating_i16_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i16x8_sub_sat(self.simd0, rhs.simd0), simd1: i16x8_sub_sat(self.simd1, rhs.simd1) }
      } else {
        Self { arr: [
          self.arr[0].saturating_sub(rhs.arr[0]),
          self.arr[1].saturating_sub(rhs.arr[1]),
          self.arr[2].saturating_sub(rhs.arr[2]),
          self.arr[3].saturating_sub(rhs.arr[3]),
          self.arr[4].saturating_sub(rhs.arr[4]),
          self.arr[5].saturating_sub(rhs.arr[5]),
          self.arr[6].saturating_sub(rhs.arr[6]),
          self.arr[7].saturating_sub(rhs.arr[7]),
          self.arr[8].saturating_sub(rhs.arr[8]),
          self.arr[9].saturating_sub(rhs.arr[9]),
          self.arr[10].saturating_sub(rhs.arr[10]),
          self.arr[11].saturating_sub(rhs.arr[11]),
          self.arr[12].saturating_sub(rhs.arr[12]),
          self.arr[13].saturating_sub(rhs.arr[13]),
          self.arr[14].saturating_sub(rhs.arr[14]),
          self.arr[15].saturating_sub(rhs.arr[15]),
        ]}
      }
    }
  }

  /// Multiply and scale equivilent to ((self * rhs) + 0x4000) >> 15 on each
  /// lane, effectively multiplying by a 16 bit fixed point number between -1
  /// and 1. This corresponds to the following instructions:
  /// - vqrdmulhq_n_s16 instruction on simd128
  /// - _mm256_mulhrs_epi16 on avx2
  /// - emulated via mul_i16_* on sse2
  #[inline]
  #[must_use]
  pub fn mul_scale_round(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: mul_i16_scale_round_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        // unfortunately mul_i16_scale_round_m128i only got added in sse3
        let hi0 = mul_i16_keep_high_m128i(self.sse0, rhs.sse0);
        let lo0 = mul_i16_keep_low_m128i(self.sse0, rhs.sse0);
        let mut v10 = unpack_low_i16_m128i(lo0, hi0);
        let mut v20 = unpack_high_i16_m128i(lo0, hi0);
        let a = set_splat_i32_m128i(0x4000);
        v10 = shr_imm_i32_m128i::<15>(add_i32_m128i(v10, a));
        v20 = shr_imm_i32_m128i::<15>(add_i32_m128i(v20, a));
        let s0 = pack_i32_to_i16_m128i(v10, v20);

        let hi1 = mul_i16_keep_high_m128i(self.sse1, rhs.sse1);
        let lo1 = mul_i16_keep_low_m128i(self.sse1, rhs.sse1);
        let mut v11 = unpack_low_i16_m128i(lo1, hi1);
        let mut v21 = unpack_high_i16_m128i(lo1, hi1);
        v11 = shr_imm_i32_m128i::<15>(add_i32_m128i(v11, a));
        v21 = shr_imm_i32_m128i::<15>(add_i32_m128i(v21, a));
        let s1 = pack_i32_to_i16_m128i(v11, v21);

        Self { sse0: s0, sse1: s1 }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: vqrdmulhq_n_s16(self.simd0, rhs.simd0), simd1: vqrdmulhq_n_s16(self.simd1, rhs.simd1) }
      } else {
        // compiler does a surprisingly good job of vectorizing this
        Self { arr: [
          ((i32::from(self.arr[0]) * i32::from(rhs.arr[0]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[1]) * i32::from(rhs.arr[1]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[2]) * i32::from(rhs.arr[2]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[3]) * i32::from(rhs.arr[3]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[4]) * i32::from(rhs.arr[4]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[5]) * i32::from(rhs.arr[5]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[6]) * i32::from(rhs.arr[6]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[7]) * i32::from(rhs.arr[7]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[8]) * i32::from(rhs.arr[8]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[9]) * i32::from(rhs.arr[9]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[10]) * i32::from(rhs.arr[10]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[11]) * i32::from(rhs.arr[11]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[12]) * i32::from(rhs.arr[12]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[13]) * i32::from(rhs.arr[13]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[14]) * i32::from(rhs.arr[14]) + 0x4000) >> 15) as i16,
          ((i32::from(self.arr[15]) * i32::from(rhs.arr[15]) + 0x4000) >> 15) as i16,
        ]}
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [i16; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[i16; 16] {
    cast_ref(self)
  }
}
