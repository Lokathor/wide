use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u8x16 { pub(crate) sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct u8x16 { pub(crate) simd: v128 }

    impl Default for u8x16 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u8x16 {
      fn eq(&self, other: &Self) -> bool {
        u8x16_all_true(u8x16_eq(self.simd, other.simd))
      }
    }

    impl Eq for u8x16 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u8x16 { pub(crate) arr: [u8;16] }
  }
}

int_uint_consts!(u8, 16, u8x16, u8x16, u8a16, const_u8_as_u8x16, 128);

unsafe impl Zeroable for u8x16 {}
unsafe impl Pod for u8x16 {}

impl Add for u8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_add(self.simd, rhs.simd) }
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

impl Sub for u8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_sub(self.simd, rhs.simd) }
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

impl Add<u8> for u8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: u8) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u8> for u8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u8) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Add<u8x16> for u8 {
  type Output = u8x16;
  #[inline]
  #[must_use]
  fn add(self, rhs: u8x16) -> Self::Output {
    u8x16::splat(self).add(rhs)
  }
}

impl Sub<u8x16> for u8 {
  type Output = u8x16;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u8x16) -> Self::Output {
    u8x16::splat(self).sub(rhs)
  }
}

impl BitAnd for u8x16 {
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

impl BitOr for u8x16 {
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

impl BitXor for u8x16 {
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

impl u8x16 {
  #[inline]
  #[must_use]
  pub fn new(array: [u8; 16]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_eq(self.simd, rhs.simd) }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { u8::MAX } else { 0 },
          if self.arr[1] == rhs.arr[1] { u8::MAX } else { 0 },
          if self.arr[2] == rhs.arr[2] { u8::MAX } else { 0 },
          if self.arr[3] == rhs.arr[3] { u8::MAX } else { 0 },
          if self.arr[4] == rhs.arr[4] { u8::MAX } else { 0 },
          if self.arr[5] == rhs.arr[5] { u8::MAX } else { 0 },
          if self.arr[6] == rhs.arr[6] { u8::MAX } else { 0 },
          if self.arr[7] == rhs.arr[7] { u8::MAX } else { 0 },
          if self.arr[8] == rhs.arr[8] { u8::MAX } else { 0 },
          if self.arr[9] == rhs.arr[9] { u8::MAX } else { 0 },
          if self.arr[10] == rhs.arr[10] { u8::MAX } else { 0 },
          if self.arr[11] == rhs.arr[11] { u8::MAX } else { 0 },
          if self.arr[12] == rhs.arr[12] { u8::MAX } else { 0 },
          if self.arr[13] == rhs.arr[13] { u8::MAX } else { 0 },
          if self.arr[14] == rhs.arr[14] { u8::MAX } else { 0 },
          if self.arr[15] == rhs.arr[15] { u8::MAX } else { 0 },
        ]}
      }
    }
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
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: max_u8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_max(self.simd, rhs.simd) }
      } else {
        Self { arr: [
          self.arr[0].max(rhs.arr[0]),
          self.arr[1].max(rhs.arr[1]),
          self.arr[2].max(rhs.arr[2]),
          self.arr[3].max(rhs.arr[3]),
          self.arr[4].max(rhs.arr[4]),
          self.arr[5].max(rhs.arr[5]),
          self.arr[6].max(rhs.arr[6]),
          self.arr[7].max(rhs.arr[7]),
          self.arr[8].max(rhs.arr[8]),
          self.arr[9].max(rhs.arr[9]),
          self.arr[10].max(rhs.arr[10]),
          self.arr[11].max(rhs.arr[11]),
          self.arr[12].max(rhs.arr[12]),
          self.arr[13].max(rhs.arr[13]),
          self.arr[14].max(rhs.arr[14]),
          self.arr[15].max(rhs.arr[15]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: min_u8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_min(self.simd, rhs.simd) }
      } else {
        Self { arr: [
          self.arr[0].min(rhs.arr[0]),
          self.arr[1].min(rhs.arr[1]),
          self.arr[2].min(rhs.arr[2]),
          self.arr[3].min(rhs.arr[3]),
          self.arr[4].min(rhs.arr[4]),
          self.arr[5].min(rhs.arr[5]),
          self.arr[6].min(rhs.arr[6]),
          self.arr[7].min(rhs.arr[7]),
          self.arr[8].min(rhs.arr[8]),
          self.arr[9].min(rhs.arr[9]),
          self.arr[10].min(rhs.arr[10]),
          self.arr[11].min(rhs.arr[11]),
          self.arr[12].min(rhs.arr[12]),
          self.arr[13].min(rhs.arr[13]),
          self.arr[14].min(rhs.arr[14]),
          self.arr[15].min(rhs.arr[15]),
        ]}
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [u8; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[u8; 16] {
    cast_ref(self)
  }

  /// Converts the first eight u8 elements within this struct to u16 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_u16x8(self) -> u16x8 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        u16x8 { sse: convert_to_u16_m128i_from_lower8_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        u16x8::new([
          u16::from(arr[0]),
          u16::from(arr[1]),
          u16::from(arr[2]),
          u16::from(arr[3]),
          u16::from(arr[4]),
          u16::from(arr[5]),
          u16::from(arr[6]),
          u16::from(arr[7]),
        ])
      }
    }
  }

  /// Converts the first eight u8 elements within this struct to i16 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i16x8(self) -> i16x8 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        i16x8 { sse: convert_to_u16_m128i_from_lower8_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        i16x8::new([
          i16::from(arr[0]),
          i16::from(arr[1]),
          i16::from(arr[2]),
          i16::from(arr[3]),
          i16::from(arr[4]),
          i16::from(arr[5]),
          i16::from(arr[6]),
          i16::from(arr[7]),
        ])
      }
    }
  }

  /// Converts the u8 elements within this struct to i16 elements.
  #[inline]
  #[must_use]
  pub fn to_i16x16(self) -> i16x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i16x16 { avx2: convert_to_i16_m256i_from_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        i16x16::new([
          i16::from(arr[0]),
          i16::from(arr[1]),
          i16::from(arr[2]),
          i16::from(arr[3]),
          i16::from(arr[4]),
          i16::from(arr[5]),
          i16::from(arr[6]),
          i16::from(arr[7]),
          i16::from(arr[8]),
          i16::from(arr[9]),
          i16::from(arr[10]),
          i16::from(arr[11]),
          i16::from(arr[12]),
          i16::from(arr[13]),
          i16::from(arr[14]),
          i16::from(arr[15]),
        ])
      }
    }
  }

  /// Converts the first four u8 elements within this struct to u32 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_u32x4(self) -> u32x4 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        u32x4 { sse: convert_to_u32_m128i_from_lower4_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        u32x4::new([
          u32::from(arr[0]),
          u32::from(arr[1]),
          u32::from(arr[2]),
          u32::from(arr[3]),
        ])
      }
    }
  }

  /// Converts the first four u8 elements within this struct to i32 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i32x4(self) -> i32x4 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        i32x4 { sse: convert_to_u32_m128i_from_lower4_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        i32x4::new([
          i32::from(arr[0]),
          i32::from(arr[1]),
          i32::from(arr[2]),
          i32::from(arr[3]),
        ])
      }
    }
  }

  /// Converts the first eight u8 elements within this struct to u32 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_u32x8(self) -> u32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // This function is named wrong in `safe_arch`.
        // It calls `_mm256_cvtepu8_epi32`.
        u32x8 { avx2: convert_to_i16_m256i_from_lower8_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        u32x8::new([
          u32::from(arr[0]),
          u32::from(arr[1]),
          u32::from(arr[2]),
          u32::from(arr[3]),
          u32::from(arr[4]),
          u32::from(arr[5]),
          u32::from(arr[6]),
          u32::from(arr[7]),
        ])
      }
    }
  }

  /// Converts the first eight u8 elements within this struct to i32 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i32x8(self) -> i32x8 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // This function is named wrong in `safe_arch`.
        // It calls `_mm256_cvtepu8_epi32`.
        i32x8 { avx2: convert_to_i16_m256i_from_lower8_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        i32x8::new([
          i32::from(arr[0]),
          i32::from(arr[1]),
          i32::from(arr[2]),
          i32::from(arr[3]),
          i32::from(arr[4]),
          i32::from(arr[5]),
          i32::from(arr[6]),
          i32::from(arr[7]),
        ])
      }
    }
  }

  /// Converts the first two u8 elements within this struct to u64 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_u64x2(self) -> u64x2 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        u64x2 { sse: convert_to_u64_m128i_from_lower2_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        u64x2::new([
          u64::from(arr[0]),
          u64::from(arr[1]),
        ])
      }
    }
  }

  /// Converts the first two u8 elements within this struct to i64 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i64x2(self) -> i64x2 {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        i64x2 { sse: convert_to_u64_m128i_from_lower2_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        i64x2::new([
          i64::from(arr[0]),
          i64::from(arr[1]),
        ])
      }
    }
  }

  /// Converts the first four u8 elements within this struct to u64 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_u64x4(self) -> u64x4 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // This function is named wrong in `safe_arch`.
        // It calls `_mm256_cvtepu8_epi64`.
        u64x4 { avx2: convert_to_i16_m256i_from_lower4_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        u64x4::new([
          u64::from(arr[0]),
          u64::from(arr[1]),
          u64::from(arr[2]),
          u64::from(arr[3]),
        ])
      }
    }
  }

  /// Converts the first four u8 elements within this struct to i64 elements.
  ///
  /// The remaining elements will be discarded.
  #[inline]
  #[must_use]
  pub fn to_i64x4(self) -> i64x4 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // This function is named wrong in `safe_arch`.
        // It calls `_mm256_cvtepu8_epi64`.
        i64x4 { avx2: convert_to_i16_m256i_from_lower4_u8_m128i(self.sse) }
      } else {
        let arr = self.to_array();
        i64x4::new([
          i64::from(arr[0]),
          i64::from(arr[1]),
          i64::from(arr[2]),
          i64::from(arr[3]),
        ])
      }
    }
  }
}
