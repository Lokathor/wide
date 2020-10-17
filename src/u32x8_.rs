use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u32x8 { avx2: m256i }
  } else if #[cfg(target_feature="sse")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u32x8 { sse0: m128i, sse1: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u32x8 { arr: [u32;8] }
  }
}

unsafe impl Zeroable for u32x8 {}
unsafe impl Pod for u32x8 {}

impl Add for u32x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_i32_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: add_i32_m128i(self.sse0, rhs.sse0), sse1: add_i32_m128i(self.sse1, rhs.sse1)}
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
        ]}
      }
    }
  }
}

impl Sub for u32x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_i32_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_i32_m128i(self.sse0, rhs.sse0), sse1: sub_i32_m128i(self.sse1, rhs.sse1)}
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
        ]}
      }
    }
  }
}

impl Mul for u32x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: mul_i32_keep_low_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: mul_i32_keep_low_m128i(self.sse0, rhs.sse0), sse1: mul_i32_keep_low_m128i(self.sse1, rhs.sse1)}
      } else {
        let arr1: [u32; 8] = cast(self);
        let arr2: [u32; 8] = cast(rhs);
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
      }
    }
  }
}

impl BitAnd for u32x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitand_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitand_m128i(self.sse0, rhs.sse0), sse1: bitand_m128i(self.sse1, rhs.sse1)}
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
        ]}
      }
    }
  }
}

impl BitOr for u32x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitor_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitor_m128i(self.sse0, rhs.sse0), sse1: bitor_m128i(self.sse1, rhs.sse1)}
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
        ]}
      }
    }
  }
}

impl BitXor for u32x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitxor_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitxor_m128i(self.sse0, rhs.sse0), sse1: bitxor_m128i(self.sse1, rhs.sse1)}
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
        ]}
      }
    }
  }
}

macro_rules! impl_shl_t_for_u32x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u32x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        let u = rhs as u64;
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([u, 0]);
            Self { avx2: shl_all_u32_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse0: shl_all_u32_m128i(self.sse0, shift), sse1: shl_all_u32_m128i(self.sse1, shift)}
          }  else {
            Self { arr: [
              self.arr[0] << u,
              self.arr[1] << u,
              self.arr[2] << u,
              self.arr[3] << u,
              self.arr[4] << u,
              self.arr[5] << u,
              self.arr[6] << u,
              self.arr[7] << u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shl_t_for_u32x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u32x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u32x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        let u = rhs as u64;
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([u, 0]);
            Self { avx2: shr_all_u32_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse0: shr_all_u32_m128i(self.sse0, shift), sse1: shr_all_u32_m128i(self.sse1, shift)}
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
            ]}
          }
        }
      }
    })+
  };
}

impl_shr_t_for_u32x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl u32x8 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i32_m256i(self.avx2, rhs.avx2 ) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_eq_mask_i32_m128i(self.sse0,rhs.sse0), sse1: cmp_eq_mask_i32_m128i(self.sse1,rhs.sse1), }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { u32::MAX } else { 0 },
          if self.arr[1] == rhs.arr[1] { u32::MAX } else { 0 },
          if self.arr[2] == rhs.arr[2] { u32::MAX } else { 0 },
          if self.arr[3] == rhs.arr[3] { u32::MAX } else { 0 },
          if self.arr[4] == rhs.arr[4] { u32::MAX } else { 0 },
          if self.arr[5] == rhs.arr[5] { u32::MAX } else { 0 },
          if self.arr[6] == rhs.arr[6] { u32::MAX } else { 0 },
          if self.arr[7] == rhs.arr[7] { u32::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_gt_mask_i32_m256i(self.avx2, rhs.avx2 ) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_gt_mask_i32_m128i(self.sse0,rhs.sse0), sse1: cmp_gt_mask_i32_m128i(self.sse1,rhs.sse1), }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { u32::MAX } else { 0 },
          if self.arr[1] > rhs.arr[1] { u32::MAX } else { 0 },
          if self.arr[2] > rhs.arr[2] { u32::MAX } else { 0 },
          if self.arr[3] > rhs.arr[3] { u32::MAX } else { 0 },
          if self.arr[4] > rhs.arr[4] { u32::MAX } else { 0 },
          if self.arr[5] > rhs.arr[5] { u32::MAX } else { 0 },
          if self.arr[6] > rhs.arr[6] { u32::MAX } else { 0 },
          if self.arr[7] > rhs.arr[7] { u32::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i32_m256i(self.avx2, rhs.avx2 ) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_lt_mask_i32_m128i(self.sse0,rhs.sse0), sse1: cmp_lt_mask_i32_m128i(self.sse1,rhs.sse1), }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { u32::MAX } else { 0 },
          if self.arr[1] < rhs.arr[1] { u32::MAX } else { 0 },
          if self.arr[2] < rhs.arr[2] { u32::MAX } else { 0 },
          if self.arr[3] < rhs.arr[3] { u32::MAX } else { 0 },
          if self.arr[4] < rhs.arr[4] { u32::MAX } else { 0 },
          if self.arr[5] < rhs.arr[5] { u32::MAX } else { 0 },
          if self.arr[6] < rhs.arr[6] { u32::MAX } else { 0 },
          if self.arr[7] < rhs.arr[7] { u32::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: blend_varying_i8_m256i(f.avx2, t.avx2, self.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: blend_varying_i8_m128i(f.sse0, t.sse0, self.sse0), sse1: blend_varying_i8_m128i(f.sse1, t.sse1, self.sse1)}
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: max_i32_m256i(self.avx2, rhs.avx2 ) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: max_i32_m128i(self.sse0, rhs.sse0), sse1: max_i32_m128i(self.sse1, rhs.sse1) }
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
        Self { avx2: max_i32_m256i(self.avx2, rhs.avx2 ) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: max_i32_m128i(self.sse0, rhs.sse0), sse1: max_i32_m128i(self.sse1, rhs.sse1) }
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }
}

impl Not for u32x8 {
  type Output = Self;
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: self.avx2.not()  }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: self.sse0.not(), sse1: self.sse1.not() }
      } else {
        Self { arr: [
          !self.arr[0],
          !self.arr[1],
          !self.arr[2],
          !self.arr[3],
          !self.arr[4],
          !self.arr[5],
          !self.arr[6],
          !self.arr[7],
        ]}
      }
    }
  }
}
