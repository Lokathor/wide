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
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    use core::arch::aarch64::*;
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct i16x8 { neon : int16x8_t }

    impl Default for i16x8 {
      #[inline]
      #[must_use]
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i16x8 {
      #[inline]
      #[must_use]
      fn eq(&self, other: &Self) -> bool {
        unsafe { vminvq_u16(vceqq_s16(self.neon, other.neon))==u16::MAX }
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_s16(self.neon, rhs.neon) } }
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vsubq_s16(self.neon, rhs.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmulq_s16(self.neon, rhs.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vandq_s16(self.neon, rhs.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vorrq_s16(self.neon, rhs.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: veorq_s16(self.neon, rhs.neon) }}
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
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_s16(self.neon, vmovq_n_s16(rhs as i16)) }}
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
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_s16(self.neon, vmovq_n_s16( -(rhs as i16))) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s16_u16(vceqq_s16(self.neon, rhs.neon)) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s16_u16(vcgtq_s16(self.neon, rhs.neon)) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s16_u16(vcltq_s16(self.neon, rhs.neon)) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vbslq_s16(vreinterpretq_u16_s16(self.neon), t.neon, f.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vabsq_s16(self.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmaxq_s16(self.neon, rhs.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vminq_s16(self.neon, rhs.neon) }}
      } else {
        self.cmp_lt(rhs).blend(self, rhs)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_saturating_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_add_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vqaddq_s16(self.neon, rhs.neon) }}
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
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_saturating_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_sub_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqsubq_s16(self.neon, rhs.neon) } }
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
        ]}
      }
    }
  }

  /// Multiply and scale equivilent to ((self * rhs) + 0x4000) >> 15 on each
  /// lane, effectively multiplying by a 16 bit fixed point number between -1
  /// and 1. This corresponds to the following instructions:
  /// - vqrdmulhq_n_s16 instruction on neon
  /// - i16x8_q15mulr_sat on simd128
  /// - _mm_mulhrs_epi16 on ssse3
  /// - emulated via mul_i16_* on sse2
  #[inline]
  #[must_use]
  pub fn mul_scale_round(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="ssse3")] {
        Self { sse:  mul_i16_scale_round_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="sse2")] {
        // unfortunately mul_i16_scale_round_m128i only got added in sse3
        let hi = mul_i16_keep_high_m128i(self.sse, rhs.sse);
        let lo = mul_i16_keep_low_m128i(self.sse, rhs.sse);
        let mut v1 = unpack_low_i16_m128i(lo, hi);
        let mut v2 = unpack_high_i16_m128i(lo, hi);
        let a = set_splat_i32_m128i(0x4000);
        v1 = shr_imm_i32_m128i::<15>(add_i32_m128i(v1, a));
        v2 = shr_imm_i32_m128i::<15>(add_i32_m128i(v2, a));
        let s = pack_i32_to_i16_m128i(v1, v2);
        Self { sse: s }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i16x8_q15mulr_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqrdmulhq_s16(self.neon, rhs.neon) } }
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
        ]}
      }
    }
  }

  #[must_use]
  #[inline]
  pub fn transpose(data: [i16x8; 8]) -> [i16x8; 8] {
    pick! {
      if #[cfg(target_feature="sse2")] {
        let a1 = unpack_low_i16_m128i(data[0].sse, data[1].sse);
        let a2 = unpack_high_i16_m128i(data[0].sse, data[1].sse);
        let a3 = unpack_low_i16_m128i(data[2].sse, data[3].sse);
        let a4 = unpack_high_i16_m128i(data[2].sse, data[3].sse);
        let a5 = unpack_low_i16_m128i(data[4].sse, data[5].sse);
        let a6 = unpack_high_i16_m128i(data[4].sse, data[5].sse);
        let a7 = unpack_low_i16_m128i(data[6].sse, data[7].sse);
        let a8 = unpack_high_i16_m128i(data[6].sse, data[7].sse);

        let b1 = unpack_low_i32_m128i(a1, a3);
        let b2 = unpack_high_i32_m128i(a1, a3);
        let b3 = unpack_low_i32_m128i(a2, a4);
        let b4 = unpack_high_i32_m128i(a2, a4);
        let b5 = unpack_low_i32_m128i(a5, a7);
        let b6 = unpack_high_i32_m128i(a5, a7);
        let b7 = unpack_low_i32_m128i(a6, a8);
        let b8 = unpack_high_i32_m128i(a6, a8);

        [
          i16x8 { sse: unpack_low_i64_m128i(b1, b5) },
          i16x8 { sse: unpack_high_i64_m128i(b1, b5) },
          i16x8 { sse: unpack_low_i64_m128i(b2, b6) },
          i16x8 { sse: unpack_high_i64_m128i(b2, b6) },
          i16x8 { sse: unpack_low_i64_m128i(b3, b7) },
          i16x8 { sse: unpack_high_i64_m128i(b3, b7) },
          i16x8 { sse: unpack_low_i64_m128i(b4, b8) },
          i16x8 { sse: unpack_high_i64_m128i(b4, b8) } ,
        ]
     } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          let a1 = vtrnq_s16(data[0].neon, data[1].neon);
          let a2 = vtrnq_s16(data[2].neon, data[3].neon);
          let a3 = vtrnq_s16(data[4].neon, data[5].neon);
          let a4 = vtrnq_s16(data[6].neon, data[7].neon);

          let b1 = vtrnq_s32(vreinterpretq_s32_s16(a1.0), vreinterpretq_s32_s16(a2.0));
          let b2 = vtrnq_s32(vreinterpretq_s32_s16(a1.1), vreinterpretq_s32_s16(a2.1));
          let b3 = vtrnq_s32(vreinterpretq_s32_s16(a3.0), vreinterpretq_s32_s16(a4.0));
          let b4 = vtrnq_s32(vreinterpretq_s32_s16(a3.1), vreinterpretq_s32_s16(a4.1));

          [
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b1.0), vget_low_s32(b3.0))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b2.0), vget_low_s32(b4.0))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b1.1), vget_low_s32(b3.1))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b2.1), vget_low_s32(b4.1))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b1.0), vget_high_s32(b3.0))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b2.0), vget_high_s32(b4.0))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b1.1), vget_high_s32(b3.1))) },
            i16x8 { neon: vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b2.1), vget_high_s32(b4.1))) },
          ]
        }
      } else if #[cfg(target_feature="simd128")] {
        #[inline] fn lo_i16(a : v128, b : v128) -> v128 { i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a,b) }
        #[inline] fn hi_i16(a : v128, b : v128) -> v128 { i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a,b) }
        #[inline] fn lo_i32(a : v128, b : v128) -> v128 { i32x4_shuffle::<0, 4, 1, 5>(a,b) }
        #[inline] fn hi_i32(a : v128, b : v128) -> v128 { i32x4_shuffle::<2, 6, 3, 7>(a,b) }
        #[inline] fn lo_i64(a : v128, b : v128) -> v128 { i64x2_shuffle::<0, 2>(a,b) }
        #[inline] fn hi_i64(a : v128, b : v128) -> v128 { i64x2_shuffle::<1, 3>(a,b) }

        let a1 = lo_i16(data[0].simd, data[1].simd);
        let a2 = hi_i16(data[0].simd, data[1].simd);
        let a3 = lo_i16(data[2].simd, data[3].simd);
        let a4 = hi_i16(data[2].simd, data[3].simd);
        let a5 = lo_i16(data[4].simd, data[5].simd);
        let a6 = hi_i16(data[4].simd, data[5].simd);
        let a7 = lo_i16(data[6].simd, data[7].simd);
        let a8 = hi_i16(data[6].simd, data[7].simd);

        let b1 = lo_i32(a1, a3);
        let b2 = hi_i32(a1, a3);
        let b3 = lo_i32(a2, a4);
        let b4 = hi_i32(a2, a4);
        let b5 = lo_i32(a5, a7);
        let b6 = hi_i32(a5, a7);
        let b7 = lo_i32(a6, a8);
        let b8 = hi_i32(a6, a8);

        [
          i16x8 { simd: lo_i64(b1, b5) },
          i16x8 { simd: hi_i64(b1, b5) },
          i16x8 { simd: lo_i64(b2, b6) },
          i16x8 { simd: hi_i64(b2, b6) },
          i16x8 { simd: lo_i64(b3, b7) },
          i16x8 { simd: hi_i64(b3, b7) },
          i16x8 { simd: lo_i64(b4, b8) },
          i16x8 { simd: hi_i64(b4, b8) } ,
        ]

      } else {
        #[inline(always)]
        fn transpose_column(data: &[i16x8; 8], index: usize) -> i16x8 {
          i16x8::new([
            data[0].as_array_ref()[index],
            data[1].as_array_ref()[index],
            data[2].as_array_ref()[index],
            data[3].as_array_ref()[index],
            data[4].as_array_ref()[index],
            data[5].as_array_ref()[index],
            data[6].as_array_ref()[index],
            data[7].as_array_ref()[index],
          ])
        }

        [
          transpose_column(&data, 0),
          transpose_column(&data, 1),
          transpose_column(&data, 2),
          transpose_column(&data, 3),
          transpose_column(&data, 4),
          transpose_column(&data, 5),
          transpose_column(&data, 6),
          transpose_column(&data, 7),
        ]
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [i16; 8] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[i16; 8] {
    cast_ref(self)
  }
}
