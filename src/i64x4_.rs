use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { avx2: m256i }
  } else if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { sse0: m128i, sse1: m128i }
} else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i64x4 { arr: [i64;4] }
  }
}

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
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: add_i64_m128i(self.sse0, rhs.sse0), sse1: add_i64_m128i(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0].wrapping_add(rhs.arr[0]),
          self.arr[1].wrapping_add(rhs.arr[1]),
          self.arr[2].wrapping_add(rhs.arr[2]),
          self.arr[3].wrapping_add(rhs.arr[3]),
        ]}
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
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_i64_m128i(self.sse0, rhs.sse0), sse1: sub_i64_m128i(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0].wrapping_sub(rhs.arr[0]),
          self.arr[1].wrapping_sub(rhs.arr[1]),
          self.arr[2].wrapping_sub(rhs.arr[2]),
          self.arr[3].wrapping_sub(rhs.arr[3]),
        ]}
      }
    }
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
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitand_m128i(self.sse0, rhs.sse0), sse1: bitand_m128i(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0].bitand(rhs.arr[0]),
          self.arr[1].bitand(rhs.arr[1]),
          self.arr[2].bitand(rhs.arr[2]),
          self.arr[3].bitand(rhs.arr[3]),
        ]}
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
          } else  if #[cfg(target_feature="sse2")] {
            Self { sse0: bitor_m128i(self.sse0, rhs.sse0) , sse1: bitor_m128i(self.sse1, rhs.sse1)}
          } else {
            Self { arr: [
              self.arr[0].bitor(rhs.arr[0]),
              self.arr[1].bitor(rhs.arr[1]),
              self.arr[2].bitor(rhs.arr[2]),
              self.arr[3].bitor(rhs.arr[3]),
            ]}
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
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitxor_m128i(self.sse0, rhs.sse0), sse1: bitxor_m128i(self.sse1, rhs.sse1) }
      } else {
        Self { arr: [
          self.arr[0].bitxor(rhs.arr[0]),
          self.arr[1].bitxor(rhs.arr[1]),
          self.arr[2].bitxor(rhs.arr[2]),
          self.arr[3].bitxor(rhs.arr[3]),
        ]}
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
        let u = rhs as u64;
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([u, 0]);
            Self { avx2: shl_all_u64_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse0: shl_all_u64_m128i(self.sse0, shift), sse1: shl_all_u64_m128i(self.sse1, shift) }
          } else {
            Self { arr: [
              self.arr[0] << u,
              self.arr[1] << u,
              self.arr[2] << u,
              self.arr[3] << u,

            ]}
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
        let u = rhs as u64;
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([u, 0]);
            Self { avx2: shr_all_u64_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse0: shr_all_u64_m128i(self.sse0, shift), sse1: shr_all_u64_m128i(self.sse1, shift) }
          } else {
            Self { arr: [
              self.arr[0] >> u,
              self.arr[1] >> u,
              self.arr[2] >> u,
              self.arr[3] >> u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shr_t_for_i64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl i64x4 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i64_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: cmp_eq_mask_i64_m128i(self.sse0, rhs.sse0),sse1: cmp_eq_mask_i64_m128i(self.sse1, rhs.sse1) }
      } else {
        let s: [i64;4] = cast(self);
        let r: [i64;4] = cast(rhs);
        cast([
          if s[0] == r[0] { -1_i64 } else { 0 },
          if s[1] == r[1] { -1_i64 } else { 0 },
          if s[2] == r[2] { -1_i64 } else { 0 },
          if s[3] == r[3] { -1_i64 } else { 0 },
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i64_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse4.2")] {
        Self { sse0: cmp_gt_mask_i64_m128i(self.sse0, rhs.sse0), sse1: cmp_gt_mask_i64_m128i(self.sse1, rhs.sse1) }
      } else {
        let s: [i64;4] = cast(self);
        let r: [i64;4] = cast(rhs);
        cast([
          if s[0] > r[0] { -1_i64 } else { 0 },
          if s[1] > r[1] { -1_i64 } else { 0 },
          if s[2] > r[2] { -1_i64 } else { 0 },
          if s[3] > r[3] { -1_i64 } else { 0 },
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(f.avx2,t.avx2,self.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: blend_varying_i8_m128i(f.sse0, t.sse0, self.sse0), sse1: blend_varying_i8_m128i(f.sse1, t.sse1, self.sse1) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
}

impl Not for i64x4 {
  type Output = Self;
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: self.avx2.not()  }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: self.sse0.not() , sse1: self.sse1.not() }
      } else {
        Self {
          arr: [
          !self.arr[0],
          !self.arr[1],
          !self.arr[2],
          !self.arr[3],
        ]}
      }
    }
  }
}
