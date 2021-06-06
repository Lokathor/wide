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
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i16x16 { arr: [i16;16] }
  }
}

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
        let u: u64 = rhs as u64;
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([u, 0]);
            Self { avx2: shl_all_u16_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self {
              sse0: shl_all_u16_m128i(self.sse0, shift),
              sse1: shl_all_u16_m128i(self.sse1, shift)
            }
          } else {
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
        let u: u64 = rhs as u64;
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([u, 0]);
            Self { avx2: shr_all_u16_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse0: shr_all_i16_m128i(self.sse0, shift),
              sse1: shr_all_i16_m128i(self.sse1, shift) }
          } else {
            Self { arr: [
              self.arr[0] >> u,
              self.arr[1] >> u,
              self.arr[2] >> u,
              self.arr[3] >> u,
              self.arr[4] >> u,
              self.arr[5] >> u,
              self.arr[6] >> u,
              self.arr[7] >> u,
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
        //TODO: what about == ?
        Self { avx2: !cmp_gt_mask_i16_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_lt_mask_i16_m128i(self.sse0, rhs.sse0),
          sse1: cmp_lt_mask_i16_m128i(self.sse1, rhs.sse1) }
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
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i16_m256i(f.avx2, t.avx2, self.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_i16_m128i(f.sse, t.sse, self.sse) }
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
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }
}
