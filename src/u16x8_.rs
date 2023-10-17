use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u16x8 { sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct u16x8 { simd: v128 }

    impl Default for u16x8 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u16x8 {
      fn eq(&self, other: &Self) -> bool {
        u16x8_all_true(u16x8_eq(self.simd, other.simd))
      }
    }

    impl Eq for u16x8 { }
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
      use core::arch::aarch64::*;
      #[repr(C)]
      #[derive(Copy, Clone)]
      pub struct u16x8 { neon : uint16x8_t }

      impl Default for u16x8 {
        #[inline]
        #[must_use]
        fn default() -> Self {
          Self::splat(0)
        }
      }

      impl PartialEq for u16x8 {
        #[inline]
        #[must_use]
        fn eq(&self, other: &Self) -> bool {
          unsafe { vminvq_u16(vceqq_u16(self.neon, other.neon))==u16::MAX }
        }
      }

      impl Eq for u16x8 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u16x8 { arr: [u16;8] }
  }
}

int_uint_consts!(u16, 8, u16x8, u16x8, u16a8, const_u16_as_u16x8, 128);

unsafe impl Zeroable for u16x8 {}
unsafe impl Pod for u16x8 {}

impl Add for u16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_add(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_u16(self.neon, rhs.neon) } }
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

impl Sub for u16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_sub(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vsubq_u16(self.neon, rhs.neon) }}
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

impl Mul for u16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: mul_i16_keep_low_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_mul(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmulq_u16(self.neon, rhs.neon) }}
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

impl Add<u16> for u16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: u16) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u16> for u16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u16) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u16> for u16x8 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u16) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u16x8> for u16 {
  type Output = u16x8;
  #[inline]
  #[must_use]
  fn add(self, rhs: u16x8) -> Self::Output {
    u16x8::splat(self).add(rhs)
  }
}

impl Sub<u16x8> for u16 {
  type Output = u16x8;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u16x8) -> Self::Output {
    u16x8::splat(self).sub(rhs)
  }
}

impl Mul<u16x8> for u16 {
  type Output = u16x8;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u16x8) -> Self::Output {
    u16x8::splat(self).mul(rhs)
  }
}

impl BitAnd for u16x8 {
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
        unsafe {Self { neon: vandq_u16(self.neon, rhs.neon) }}
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

impl BitOr for u16x8 {
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
        unsafe {Self { neon: vorrq_u16(self.neon, rhs.neon) }}
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

impl BitXor for u16x8 {
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
        unsafe {Self { neon: veorq_u16(self.neon, rhs.neon) }}
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

macro_rules! impl_shl_t_for_u16x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u16x8 {
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
            Self { simd: u16x8_shl(self.simd, rhs as u32) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_u16(self.neon, vmovq_n_s16(rhs as i16)) }}
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
impl_shl_t_for_u16x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u16x8 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u16x8 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shr_all_u16_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: u16x8_shr(self.simd, rhs as u32) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_u16(self.neon, vmovq_n_s16( -(rhs as i16))) }}
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
impl_shr_t_for_u16x8!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl u16x8 {
  #[inline]
  #[must_use]
  pub fn new(array: [u16; 8]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_eq(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vceqq_u16(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { u16::MAX } else { 0 },
          if self.arr[1] == rhs.arr[1] { u16::MAX } else { 0 },
          if self.arr[2] == rhs.arr[2] { u16::MAX } else { 0 },
          if self.arr[3] == rhs.arr[3] { u16::MAX } else { 0 },
          if self.arr[4] == rhs.arr[4] { u16::MAX } else { 0 },
          if self.arr[5] == rhs.arr[5] { u16::MAX } else { 0 },
          if self.arr[6] == rhs.arr[6] { u16::MAX } else { 0 },
          if self.arr[7] == rhs.arr[7] { u16::MAX } else { 0 },
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vbslq_u16(self.neon, t.neon, f.neon) }}
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: max_u8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_max(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmaxq_u16(self.neon, rhs.neon) }}
      } else {
        let arr: [u16; 8] = cast(self);
        let rhs: [u16; 8] = cast(rhs);
        cast([
          arr[0].max(rhs[0]),
          arr[1].max(rhs[1]),
          arr[2].max(rhs[2]),
          arr[3].max(rhs[3]),
          arr[4].max(rhs[4]),
          arr[5].max(rhs[5]),
          arr[6].max(rhs[6]),
          arr[7].max(rhs[7]),
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: min_u8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_min(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vminq_u16(self.neon, rhs.neon) }}
      } else {
        let arr: [u16; 8] = cast(self);
        let rhs: [u16; 8] = cast(rhs);
        cast([
          arr[0].min(rhs[0]),
          arr[1].min(rhs[1]),
          arr[2].min(rhs[2]),
          arr[3].min(rhs[3]),
          arr[4].min(rhs[4]),
          arr[5].min(rhs[5]),
          arr[6].min(rhs[6]),
          arr[7].min(rhs[7]),
        ])
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_saturating_u16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_add_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vqaddq_u16(self.neon, rhs.neon) }}
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
        Self { sse: sub_saturating_u16_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u16x8_sub_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vqsubq_u16(self.neon, rhs.neon) }}
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

  #[inline]
  pub fn to_array(self) -> [u16; 8] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[u16; 8] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_array_mut(&mut self) -> &mut[u16; 8] {
    cast_mut(self)
  }
}
