use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { avx2: m256i }
  } else if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { sse0: m128i, sse1: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(C, align(32))]
    pub struct u64x4 { simd0: v128, simd1: v128 }

    impl Default for u64x4 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u64x4 {
      fn eq(&self, other: &Self) -> bool {
        !v128_any_true(v128_or(v128_xor(self.simd0, other.simd0), v128_xor(self.simd1, other.simd1)))
      }
    }

    impl Eq for u64x4 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct u64x4 { arr: [u64;4] }
  }
}

int_uint_consts!(u64, 4, u64x4, u64x4, u64a4, const_u64_as_u64x4, 256);

unsafe impl Zeroable for u64x4 {}
unsafe impl Pod for u64x4 {}

impl Add for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: add_i64_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: add_i64_m128i(self.sse0, rhs.sse0), sse1: add_i64_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: u64x2_add(self.simd0, rhs.simd0), simd1: u64x2_add(self.simd1, rhs.simd1) }
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

impl Sub for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: sub_i64_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_i64_m128i(self.sse0, rhs.sse0), sse1: sub_i64_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: u64x2_sub(self.simd0, rhs.simd0), simd1: u64x2_sub(self.simd1, rhs.simd1) }
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

impl Mul for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="simd128")] {
        Self { simd0: u64x2_mul(self.simd0, rhs.simd0), simd1: u64x2_mul(self.simd1, rhs.simd1) }
      } else {
        let arr1: [u64; 4] = cast(self);
        let arr2: [u64; 4] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
          arr1[2].wrapping_mul(arr2[2]),
          arr1[3].wrapping_mul(arr2[3]),
        ])
      }
    }
  }
}

impl Add<u64> for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: u64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u64> for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u64> for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u64x4> for u64 {
  type Output = u64x4;
  #[inline]
  #[must_use]
  fn add(self, rhs: u64x4) -> Self::Output {
    u64x4::splat(self).add(rhs)
  }
}

impl Sub<u64x4> for u64 {
  type Output = u64x4;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u64x4) -> Self::Output {
    u64x4::splat(self).sub(rhs)
  }
}

impl Mul<u64x4> for u64 {
  type Output = u64x4;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u64x4) -> Self::Output {
    u64x4::splat(self).mul(rhs)
  }
}

impl BitAnd for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitand_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitand_m128i(self.sse0, rhs.sse0), sse1: bitand_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_and(self.simd0, rhs.simd0), simd1: v128_and(self.simd1, rhs.simd1) }
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

impl BitOr for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx2")] {
        Self { avx2: bitor_m256i(self.avx2, rhs.avx2) }
      } else  if #[cfg(target_feature="sse2")] {
        Self { sse0: bitor_m128i(self.sse0, rhs.sse0) , sse1: bitor_m128i(self.sse1, rhs.sse1)}
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_or(self.simd0, rhs.simd0), simd1: v128_or(self.simd1, rhs.simd1) }
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

impl BitXor for u64x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: bitxor_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitxor_m128i(self.sse0, rhs.sse0), sse1: bitxor_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_xor(self.simd0, rhs.simd0), simd1: v128_xor(self.simd1, rhs.simd1) }
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

macro_rules! impl_shl_t_for_u64x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u64x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shl_all_u64_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse0: shl_all_u64_m128i(self.sse0, shift), sse1: shl_all_u64_m128i(self.sse1, shift) }
          } else if #[cfg(target_feature="simd128")] {
            let u = rhs as u32;
            Self { simd0: u64x2_shl(self.simd0, u), simd1: u64x2_shl(self.simd1, u) }
          } else {
            let u = rhs as u64;
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
impl_shl_t_for_u64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u64x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u64x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="avx2")] {
            let shift = cast([rhs as u64, 0]);
            Self { avx2: shr_all_u64_m256i(self.avx2, shift) }
          } else if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse0: shr_all_u64_m128i(self.sse0, shift), sse1: shr_all_u64_m128i(self.sse1, shift) }
          } else if #[cfg(target_feature="simd128")] {
            let u = rhs as u32;
            Self { simd0: u64x2_shr(self.simd0, u), simd1: u64x2_shr(self.simd1, u) }
          } else {
            let u = rhs as u64;
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
impl_shr_t_for_u64x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl u64x4 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: cmp_eq_mask_i64_m256i(self.avx2, rhs.avx2) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: cmp_eq_mask_i64_m128i(self.sse0, rhs.sse0),sse1: cmp_eq_mask_i64_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: u64x2_eq(self.simd0, rhs.simd0), simd1: u64x2_eq(self.simd1, rhs.simd1) }
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
        // u64x2_gt on WASM is not a thing. https://github.com/WebAssembly/simd/pull/414
        let s: [u64;4] = cast(self);
        let r: [u64;4] = cast(rhs);
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
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_bitselect(t.simd0, f.simd0, self.simd0), simd1: v128_bitselect(t.simd1, f.simd1, self.simd1) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
}

impl Not for u64x4 {
  type Output = Self;
  fn not(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx2: self.avx2.not()  }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: self.sse0.not() , sse1: self.sse1.not() }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_not(self.simd0) , simd1: v128_not(self.simd1) }
      } else {
        Self { arr: [
          !self.arr[0],
          !self.arr[1],
          !self.arr[2],
          !self.arr[3],
        ]}
      }
    }
  }
}
