use super::*;

pick! {
  if #[cfg(target_feature="avx2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i8x32 { avx: m256i }
  } else if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i8x32 { sse0: m128i, sse1: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(C, align(32))]
    pub struct i8x32 { simd0: v128, simd1: v128 }

    impl Default for i8x32 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i8x32 {
      fn eq(&self, other: &Self) -> bool {
        !v128_any_true(v128_or(v128_xor(self.simd0, other.simd0), v128_xor(self.simd1, other.simd1)))
      }
    }

    impl Eq for i8x32 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(32))]
    pub struct i8x32 { arr: [i8;32] }
  }
}

int_uint_consts!(i8, 32, i8x32, i8x32, i8a32, const_i8_as_i8x32, 256);

unsafe impl Zeroable for i8x32 {}
unsafe impl Pod for i8x32 {}

impl Add for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: add_i8_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: add_i8_m128i(self.sse0, rhs.sse0), sse1: add_i8_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_add(self.simd0, rhs.simd0), simd1: i8x16_add(self.simd1, rhs.simd1) }
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
          self.arr[16].wrapping_add(rhs.arr[16]),
          self.arr[17].wrapping_add(rhs.arr[17]),
          self.arr[18].wrapping_add(rhs.arr[18]),
          self.arr[19].wrapping_add(rhs.arr[19]),
          self.arr[20].wrapping_add(rhs.arr[20]),
          self.arr[21].wrapping_add(rhs.arr[21]),
          self.arr[22].wrapping_add(rhs.arr[22]),
          self.arr[23].wrapping_add(rhs.arr[23]),
          self.arr[24].wrapping_add(rhs.arr[24]),
          self.arr[25].wrapping_add(rhs.arr[25]),
          self.arr[26].wrapping_add(rhs.arr[26]),
          self.arr[27].wrapping_add(rhs.arr[27]),
          self.arr[28].wrapping_add(rhs.arr[28]),
          self.arr[29].wrapping_add(rhs.arr[29]),
          self.arr[30].wrapping_add(rhs.arr[30]),
          self.arr[31].wrapping_add(rhs.arr[31]),
        ]}
      }
    }
  }
}

impl Sub for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: sub_i8_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: sub_i8_m128i(self.sse0, rhs.sse0), sse1: sub_i8_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_sub(self.simd0, rhs.simd0), simd1: i8x16_sub(self.simd1, rhs.simd1) }
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
          self.arr[16].wrapping_sub(rhs.arr[16]),
          self.arr[17].wrapping_sub(rhs.arr[17]),
          self.arr[18].wrapping_sub(rhs.arr[18]),
          self.arr[19].wrapping_sub(rhs.arr[19]),
          self.arr[20].wrapping_sub(rhs.arr[20]),
          self.arr[21].wrapping_sub(rhs.arr[21]),
          self.arr[22].wrapping_sub(rhs.arr[22]),
          self.arr[23].wrapping_sub(rhs.arr[23]),
          self.arr[24].wrapping_sub(rhs.arr[24]),
          self.arr[25].wrapping_sub(rhs.arr[25]),
          self.arr[26].wrapping_sub(rhs.arr[26]),
          self.arr[27].wrapping_sub(rhs.arr[27]),
          self.arr[28].wrapping_sub(rhs.arr[28]),
          self.arr[29].wrapping_sub(rhs.arr[29]),
          self.arr[30].wrapping_sub(rhs.arr[30]),
          self.arr[31].wrapping_sub(rhs.arr[31]),
          ]}
      }
    }
  }
}

impl Add<i8> for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: i8) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i8> for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i8) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Add<i8x32> for i8 {
  type Output = i8x32;
  #[inline]
  #[must_use]
  fn add(self, rhs: i8x32) -> Self::Output {
    i8x32::splat(self).add(rhs)
  }
}

impl Sub<i8x32> for i8 {
  type Output = i8x32;
  #[inline]
  #[must_use]
  fn sub(self, rhs: i8x32) -> Self::Output {
    i8x32::splat(self).sub(rhs)
  }
}

impl BitAnd for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
          Self { avx : bitand_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitand_m128i(self.sse0, rhs.sse0),  sse1: bitand_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_and(self.simd0, rhs.simd0), simd1: v128_and(self.simd1, rhs.simd1) }
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
          self.arr[16].bitand(rhs.arr[16]),
          self.arr[17].bitand(rhs.arr[17]),
          self.arr[18].bitand(rhs.arr[18]),
          self.arr[19].bitand(rhs.arr[19]),
          self.arr[20].bitand(rhs.arr[20]),
          self.arr[21].bitand(rhs.arr[21]),
          self.arr[22].bitand(rhs.arr[22]),
          self.arr[23].bitand(rhs.arr[23]),
          self.arr[24].bitand(rhs.arr[24]),
          self.arr[25].bitand(rhs.arr[25]),
          self.arr[26].bitand(rhs.arr[26]),
          self.arr[27].bitand(rhs.arr[27]),
          self.arr[28].bitand(rhs.arr[28]),
          self.arr[29].bitand(rhs.arr[29]),
          self.arr[30].bitand(rhs.arr[30]),
          self.arr[31].bitand(rhs.arr[31]),
          ]}
      }
    }
  }
}

impl BitOr for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : bitor_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitor_m128i(self.sse0, rhs.sse0),  sse1: bitor_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_or(self.simd0, rhs.simd0), simd1: v128_or(self.simd1, rhs.simd1) }
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
          self.arr[16].bitor(rhs.arr[16]),
          self.arr[17].bitor(rhs.arr[17]),
          self.arr[18].bitor(rhs.arr[18]),
          self.arr[19].bitor(rhs.arr[19]),
          self.arr[20].bitor(rhs.arr[20]),
          self.arr[21].bitor(rhs.arr[21]),
          self.arr[22].bitor(rhs.arr[22]),
          self.arr[23].bitor(rhs.arr[23]),
          self.arr[24].bitor(rhs.arr[24]),
          self.arr[25].bitor(rhs.arr[25]),
          self.arr[26].bitor(rhs.arr[26]),
          self.arr[27].bitor(rhs.arr[27]),
          self.arr[28].bitor(rhs.arr[28]),
          self.arr[29].bitor(rhs.arr[29]),
          self.arr[30].bitor(rhs.arr[30]),
          self.arr[31].bitor(rhs.arr[31]),
        ]}
      }
    }
  }
}

impl BitXor for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : bitxor_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: bitxor_m128i(self.sse0, rhs.sse0),  sse1: bitxor_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_xor(self.simd0, rhs.simd0), simd1: v128_xor(self.simd1, rhs.simd1) }
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
          self.arr[16].bitxor(rhs.arr[16]),
          self.arr[17].bitxor(rhs.arr[17]),
          self.arr[18].bitxor(rhs.arr[18]),
          self.arr[19].bitxor(rhs.arr[19]),
          self.arr[20].bitxor(rhs.arr[20]),
          self.arr[21].bitxor(rhs.arr[21]),
          self.arr[22].bitxor(rhs.arr[22]),
          self.arr[23].bitxor(rhs.arr[23]),
          self.arr[24].bitxor(rhs.arr[24]),
          self.arr[25].bitxor(rhs.arr[25]),
          self.arr[26].bitxor(rhs.arr[26]),
          self.arr[27].bitxor(rhs.arr[27]),
          self.arr[28].bitxor(rhs.arr[28]),
          self.arr[29].bitxor(rhs.arr[29]),
          self.arr[30].bitxor(rhs.arr[30]),
          self.arr[31].bitxor(rhs.arr[31]),
        ]}
      }
    }
  }
}

impl CmpEq for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : cmp_eq_mask_i8_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_eq_mask_i8_m128i(self.sse0, rhs.sse0),  sse1: cmp_eq_mask_i8_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_eq(self.simd0, rhs.simd0), simd1: i8x16_eq(self.simd1, rhs.simd1) }
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
          if self.arr[16] == rhs.arr[16] { -1 } else { 0 },
          if self.arr[17] == rhs.arr[17] { -1 } else { 0 },
          if self.arr[18] == rhs.arr[18] { -1 } else { 0 },
          if self.arr[19] == rhs.arr[19] { -1 } else { 0 },
          if self.arr[20] == rhs.arr[20] { -1 } else { 0 },
          if self.arr[21] == rhs.arr[21] { -1 } else { 0 },
          if self.arr[22] == rhs.arr[22] { -1 } else { 0 },
          if self.arr[23] == rhs.arr[23] { -1 } else { 0 },
          if self.arr[24] == rhs.arr[24] { -1 } else { 0 },
          if self.arr[25] == rhs.arr[25] { -1 } else { 0 },
          if self.arr[26] == rhs.arr[26] { -1 } else { 0 },
          if self.arr[27] == rhs.arr[27] { -1 } else { 0 },
          if self.arr[28] == rhs.arr[28] { -1 } else { 0 },
          if self.arr[29] == rhs.arr[29] { -1 } else { 0 },
          if self.arr[30] == rhs.arr[30] { -1 } else { 0 },
          if self.arr[31] == rhs.arr[31] { -1 } else { 0 },
          ]}
      }
    }
  }
}

impl CmpGt for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : cmp_gt_mask_i8_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_gt_mask_i8_m128i(self.sse0, rhs.sse0),  sse1: cmp_gt_mask_i8_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_gt(self.simd0, rhs.simd0), simd1: i8x16_gt(self.simd1, rhs.simd1) }
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
          if self.arr[16] > rhs.arr[16] { -1 } else { 0 },
          if self.arr[17] > rhs.arr[17] { -1 } else { 0 },
          if self.arr[18] > rhs.arr[18] { -1 } else { 0 },
          if self.arr[19] > rhs.arr[19] { -1 } else { 0 },
          if self.arr[20] > rhs.arr[20] { -1 } else { 0 },
          if self.arr[21] > rhs.arr[21] { -1 } else { 0 },
          if self.arr[22] > rhs.arr[22] { -1 } else { 0 },
          if self.arr[23] > rhs.arr[23] { -1 } else { 0 },
          if self.arr[24] > rhs.arr[24] { -1 } else { 0 },
          if self.arr[25] > rhs.arr[25] { -1 } else { 0 },
          if self.arr[26] > rhs.arr[26] { -1 } else { 0 },
          if self.arr[27] > rhs.arr[27] { -1 } else { 0 },
          if self.arr[28] > rhs.arr[28] { -1 } else { 0 },
          if self.arr[29] > rhs.arr[29] { -1 } else { 0 },
          if self.arr[30] > rhs.arr[30] { -1 } else { 0 },
          if self.arr[31] > rhs.arr[31] { -1 } else { 0 },
        ]}
      }
    }
  }
}

impl CmpLt for i8x32 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx : !(cmp_gt_mask_i8_m256i(self.avx,rhs.avx) ^ cmp_eq_mask_i8_m256i(self.avx,rhs.avx)) }
      } else if #[cfg(target_feature="sse2")] {
        Self { sse0: cmp_lt_mask_i8_m128i(self.sse0, rhs.sse0),  sse1: cmp_lt_mask_i8_m128i(self.sse1, rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_lt(self.simd0, rhs.simd0), simd1: i8x16_lt(self.simd1, rhs.simd1) }
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
          if self.arr[16] < rhs.arr[16] { -1 } else { 0 },
          if self.arr[17] < rhs.arr[17] { -1 } else { 0 },
          if self.arr[18] < rhs.arr[18] { -1 } else { 0 },
          if self.arr[19] < rhs.arr[19] { -1 } else { 0 },
          if self.arr[20] < rhs.arr[20] { -1 } else { 0 },
          if self.arr[21] < rhs.arr[21] { -1 } else { 0 },
          if self.arr[22] < rhs.arr[22] { -1 } else { 0 },
          if self.arr[23] < rhs.arr[23] { -1 } else { 0 },
          if self.arr[24] < rhs.arr[24] { -1 } else { 0 },
          if self.arr[25] < rhs.arr[25] { -1 } else { 0 },
          if self.arr[26] < rhs.arr[26] { -1 } else { 0 },
          if self.arr[27] < rhs.arr[27] { -1 } else { 0 },
          if self.arr[28] < rhs.arr[28] { -1 } else { 0 },
          if self.arr[29] < rhs.arr[29] { -1 } else { 0 },
          if self.arr[30] < rhs.arr[30] { -1 } else { 0 },
          if self.arr[31] < rhs.arr[31] { -1 } else { 0 },
        ]}
      }
    }
  }
}

impl i8x32 {
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: blend_varying_i8_m256i(f.avx, t.avx, self.avx) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: blend_varying_i8_m128i(f.sse0, t.sse0, self.sse0),  sse1: blend_varying_i8_m128i(f.sse1, t.sse1, self.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: v128_bitselect(t.simd0, f.simd0, self.simd0), simd1: v128_bitselect(t.simd1, f.simd1, self.simd1) }
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
        Self { avx: abs_i8_m256i(self.avx) }
      } else if #[cfg(target_feature="ssse3")] {
        Self { sse0: abs_i8_m128i(self.sse0), sse1: abs_i8_m128i(self.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_abs(self.simd0), simd1: i8x16_abs(self.simd1) }
      } else {
        let arr: [i8; 32] = cast(self);
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
          arr[16].wrapping_abs(),
          arr[17].wrapping_abs(),
          arr[18].wrapping_abs(),
          arr[19].wrapping_abs(),
          arr[20].wrapping_abs(),
          arr[21].wrapping_abs(),
          arr[22].wrapping_abs(),
          arr[23].wrapping_abs(),
          arr[24].wrapping_abs(),
          arr[25].wrapping_abs(),
          arr[26].wrapping_abs(),
          arr[27].wrapping_abs(),
          arr[28].wrapping_abs(),
          arr[29].wrapping_abs(),
          arr[30].wrapping_abs(),
          arr[31].wrapping_abs(),
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx2")] {
        Self { avx: max_i8_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: max_i8_m128i(self.sse0,rhs.sse0), sse1: max_i8_m128i(self.sse1,rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_max(self.simd0,rhs.simd0), simd1: i8x16_max(self.simd1,rhs.simd1) }
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
        Self { avx: min_i8_m256i(self.avx,rhs.avx) }
      } else if #[cfg(target_feature="sse4.1")] {
        Self { sse0: min_i8_m128i(self.sse0,rhs.sse0), sse1: min_i8_m128i(self.sse1,rhs.sse1) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd0: i8x16_min(self.simd0,rhs.simd0), simd1: i8x16_min(self.simd1,rhs.simd1) }
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }

  pub fn to_array(self) -> [i8; 32] {
    cast(self)
  }

  pub fn as_array_ref(&self) -> &[i8; 32] {
    cast_ref(self)
  }
}
