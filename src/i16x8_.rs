use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i16x8 { sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct i16x8 { simd: v128 }

    impl Default for i16x8 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i16x8 {
      fn eq(&self, other: &Self) -> bool {
        u16x8_all_true(i16x8_eq(self.simd, other.simd))
      }
    }

    impl Eq for i16x8 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i16x8 { arr: [i16;8] }
  }
}

int_uint_consts!(i16, 8, i16x8, i16x8, i16a8, const_i16_as_i16x8, 128);

unsafe impl Zeroable for i16x8 {}
unsafe impl Pod for i16x8 {}

impl Add for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_add(self.simd, rhs.simd) }
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

impl Sub for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_sub(self.simd, rhs.simd) }
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

impl Mul for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: mul_i16_keep_low_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_mul(self.simd, rhs.simd) }
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
        ]}
      }
    }
  }
}

impl Add<i16> for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: i16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i16> for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i16> for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i16) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<i16x8> for i16 {
  type Output = i16x8;
  #[inline]
  #[must_use]
  fn add(self, rhs: i16x8) -> Self::Output {
    i16x8::splat(self).add(rhs)
  }
}

impl Sub<i16x8> for i16 {
  type Output = i16x8;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i16x8) -> Self::Output {
    i16x8::splat(self).sub(rhs)
  }
}

impl Mul<i16x8> for i16 {
  type Output = i16x8;
  #[inline]
  #[must_use]
  fn mul(self, rhs: i16x8) -> Self::Output {
    i16x8::splat(self).mul(rhs)
  }
}

impl BitAnd for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitand_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_and(self.simd, rhs.simd) }
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

impl BitOr for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitor_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_or(self.simd, rhs.simd) }
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

impl BitXor for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_xor(self.simd, rhs.simd) }
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

macro_rules! impl_shl_t_for_i16x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i16x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shl_all_u16_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: i16x8_shl(self.simd, rhs as u32) }
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
            ]}
          }
        }
      }
    })+
  };
}
impl_shl_t_for_i16x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i16x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i16x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shr_all_i16_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: i16x8_shr(self.simd, rhs as u32) }
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
            ]}
          }
        }
      }
    })+
  };
}
impl_shr_t_for_i16x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_eq(self.simd, rhs.simd) }
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
        ]}
      }
    }
  }
}

impl CmpGt for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_gt_mask_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_gt(self.simd, rhs.simd) }
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
        ]}
      }
    }
  }
}

impl CmpLt for i16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_lt_mask_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_lt(self.simd, rhs.simd) }
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
        ]}
      }
    }
  }
}

impl i16x8 {
  #[inline]
  #[must_use]
  pub fn new(array: [i16; 8]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_i8_m128i(f.sse, t.sse, self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_bitselect(t.simd, f.simd, self.simd) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="ssse3")] {
        Self { sse: abs_i16_m128i(self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_abs(self.simd) }
      } else {
        let arr: [i16; 8] = cast(self);
        cast([
          arr[0].wrapping_abs(),
          arr[1].wrapping_abs(),
          arr[2].wrapping_abs(),
          arr[3].wrapping_abs(),
          arr[4].wrapping_abs(),
          arr[5].wrapping_abs(),
          arr[6].wrapping_abs(),
          arr[7].wrapping_abs(),
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: max_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_max(self.simd, rhs.simd) }
      } else {
        self.cmp_lt(rhs).blend(rhs, self)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: min_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_min(self.simd, rhs.simd) }
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }

  pub fn to_array(self) -> [i16; 8] {
    cast(self)
  }

  pub fn as_array_ref(&self) -> &[i16; 8] {
    cast_ref(self)
  }
}
