use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i8x16 { pub(crate) sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct i8x16 { pub(crate) simd: v128 }

    impl Default for i8x16 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i8x16 {
      fn eq(&self, other: &Self) -> bool {
        u8x16_all_true(i8x16_eq(self.simd, other.simd))
      }
    }

    impl Eq for i8x16 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i8x16 { pub(crate) arr: [i8;16] }
  }
}

int_uint_consts!(i8, 16, i8x16, i8x16, i8a16, const_i8_as_i8x16, 128);

unsafe impl Zeroable for i8x16 {}
unsafe impl Pod for i8x16 {}

impl Add for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_add(self.simd, rhs.simd) }
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

impl Sub for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_sub(self.simd, rhs.simd) }
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

impl Add<i8> for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: i8) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i8> for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i8) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Add<i8x16> for i8 {
  type Output = i8x16;
  #[inline]
  #[must_use]
  fn add(self, rhs: i8x16) -> Self::Output {
    i8x16::splat(self).add(rhs)
  }
}

impl Sub<i8x16> for i8 {
  type Output = i8x16;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i8x16) -> Self::Output {
    i8x16::splat(self).sub(rhs)
  }
}

impl BitAnd for i8x16 {
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

impl BitOr for i8x16 {
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

impl BitXor for i8x16 {
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

impl CmpEq for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_eq(self.simd, rhs.simd) }
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

impl CmpGt for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_gt_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_gt(self.simd, rhs.simd) }
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

impl CmpLt for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_lt_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_lt(self.simd, rhs.simd) }
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

impl i8x16 {
  #[inline]
  #[must_use]
  pub fn new(array: [i8; 16]) -> Self {
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
        Self { sse: abs_i8_m128i(self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_abs(self.simd) }
      } else {
        let arr: [i8; 16] = cast(self);
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
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: max_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_max(self.simd, rhs.simd) }
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
        Self { sse: min_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_min(self.simd, rhs.simd) }
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn move_mask(self) -> i32 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        move_mask_i8_m128i(self.sse)
      } else if #[cfg(target_feature="simd128")] {
        i8x16_bitmask(self.simd) as i32
      } else {
        ((self.arr[0] < 0) as i32) << 0 |
        ((self.arr[1] < 0) as i32) << 1 |
        ((self.arr[2] < 0) as i32) << 2 |
        ((self.arr[3] < 0) as i32) << 3 |
        ((self.arr[4] < 0) as i32) << 4 |
        ((self.arr[5] < 0) as i32) << 5 |
        ((self.arr[6] < 0) as i32) << 6 |
        ((self.arr[7] < 0) as i32) << 7 |
        ((self.arr[8] < 0) as i32) << 8 |
        ((self.arr[9] < 0) as i32) << 9 |
        ((self.arr[10] < 0) as i32) << 10 |
        ((self.arr[11] < 0) as i32) << 11 |
        ((self.arr[12] < 0) as i32) << 12 |
        ((self.arr[13] < 0) as i32) << 13 |
        ((self.arr[14] < 0) as i32) << 14 |
        ((self.arr[15] < 0) as i32) << 15
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="simd128")] {
        v128_any_true(self.simd)
      } else {
        self.move_mask() != 0
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="simd128")] {
        u8x16_all_true(self.simd)
      } else {
        // sixteen lanes
        self.move_mask() == 0b1111_1111_1111_1111
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  #[inline]
  pub fn to_array(self) -> [i8; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[i8; 16] {
    cast_ref(self)
  }

  /// Converts the first eight i8 elements within this struct to i16 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i16x8(self) -> i16x8 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        i16x8 { sse: convert_to_i16_m128i_from_lower8_i8_m128i(self.sse) }
      } else {
        i16x8::new([
          i16::from(self.arr[0]),
          i16::from(self.arr[1]),
          i16::from(self.arr[2]),
          i16::from(self.arr[3]),
          i16::from(self.arr[4]),
          i16::from(self.arr[5]),
          i16::from(self.arr[6]),
          i16::from(self.arr[7]),
        ])
      }
    }
  }

  /// Converts the i8 elements within this struct to i16 elements.
  #[inline]
  #[must_use]
  pub fn to_i16x16(self) -> i16x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i16x16 { avx2: convert_to_i16_m256i_from_i8_m128i(self.sse) }
      } else {
        i16x16::new([
          i16::from(self.arr[0]),
          i16::from(self.arr[1]),
          i16::from(self.arr[2]),
          i16::from(self.arr[3]),
          i16::from(self.arr[4]),
          i16::from(self.arr[5]),
          i16::from(self.arr[6]),
          i16::from(self.arr[7]),
          i16::from(self.arr[8]),
          i16::from(self.arr[9]),
          i16::from(self.arr[10]),
          i16::from(self.arr[11]),
          i16::from(self.arr[12]),
          i16::from(self.arr[13]),
          i16::from(self.arr[14]),
          i16::from(self.arr[15]),
        ])
      }
    }
  }

  /// Converts the first four i8 elements within this struct to i32 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i32x4(self) -> i32x4 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        i32x4 { sse: convert_to_i32_m128i_from_lower4_i8_m128i(self.sse) }
      } else {
        i32x4::new([
          i32::from(self.arr[0]),
          i32::from(self.arr[1]),
          i32::from(self.arr[2]),
          i32::from(self.arr[3]),
        ])
      }
    }
  }

  /// Converts the first eight i8 elements within this struct to i32 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i32x8(self) -> i32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i32x8 { avx2: convert_to_i32_m256i_from_lower8_i8_m128i(self.sse) }
      } else {
        i32x8::new([
          i32::from(self.arr[0]),
          i32::from(self.arr[1]),
          i32::from(self.arr[2]),
          i32::from(self.arr[3]),
          i32::from(self.arr[4]),
          i32::from(self.arr[5]),
          i32::from(self.arr[6]),
          i32::from(self.arr[7]),
        ])
      }
    }
  }

  /// Converts the first two i8 elements within this struct to i64 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i64x2(self) -> i64x2 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        i64x2 { sse: convert_to_i64_m128i_from_lower2_i8_m128i(self.sse) }
      } else {
        i64x2::new([
          i64::from(self.arr[0]),
          i64::from(self.arr[1]),
        ])
      }
    }
  }

  /// Converts the first four i8 elements within this struct to i64 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i64x4(self) -> i64x4 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i64x4 { avx2: convert_to_i64_m256i_from_lower4_i8_m128i(self.sse) }
      } else {
        i64x4::new([
          i64::from(self.arr[0]),
          i64::from(self.arr[1]),
          i64::from(self.arr[2]),
          i64::from(self.arr[3]),
        ])
      }
    }
  }
}
