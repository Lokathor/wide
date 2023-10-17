use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u32x4 { sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct u32x4 { simd: v128 }

    impl Default for u32x4 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u32x4 {
      fn eq(&self, other: &Self) -> bool {
        u32x4_all_true(u32x4_eq(self.simd, other.simd))
      }
    }

    impl Eq for u32x4 { }
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    use core::arch::aarch64::*;
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct u32x4 { neon : uint32x4_t }

    impl Default for u32x4 {
      #[inline]
      #[must_use]
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u32x4 {
      #[inline]
      #[must_use]
      fn eq(&self, other: &Self) -> bool {
        unsafe { vminvq_u32(vceqq_u32(self.neon, other.neon))==u32::MAX }
      }
    }

    impl Eq for u32x4 { }
} else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u32x4 { arr: [u32;4] }
  }
}

int_uint_consts!(u32, 4, u32x4, u32x4, u32a4, const_u32_as_u32x4, 128);

unsafe impl Zeroable for u32x4 {}
unsafe impl Pod for u32x4 {}

impl Add for u32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i32_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_add(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_u32(self.neon, rhs.neon) } }
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

impl Sub for u32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i32_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_sub(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vsubq_u32(self.neon, rhs.neon) }}
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

impl Mul for u32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: mul_32_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_mul(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmulq_u32(self.neon, rhs.neon) }}
      } else {
        let arr1: [u32; 4] = cast(self);
        let arr2: [u32; 4] = cast(rhs);
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

impl Add<u32> for u32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: u32) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u32> for u32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u32) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u32> for u32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u32) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u32x4> for u32 {
  type Output = u32x4;
  #[inline]
  #[must_use]
  fn add(self, rhs: u32x4) -> Self::Output {
    u32x4::splat(self).add(rhs)
  }
}

impl Sub<u32x4> for u32 {
  type Output = u32x4;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u32x4) -> Self::Output {
    u32x4::splat(self).sub(rhs)
  }
}

impl Mul<u32x4> for u32 {
  type Output = u32x4;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u32x4) -> Self::Output {
    u32x4::splat(self).mul(rhs)
  }
}

impl BitAnd for u32x4 {
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
        unsafe {Self { neon: vandq_u32(self.neon, rhs.neon) }}
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

impl BitOr for u32x4 {
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
        unsafe {Self { neon: vorrq_u32(self.neon, rhs.neon) }}
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

impl BitXor for u32x4 {
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
        unsafe {Self { neon: veorq_u32(self.neon, rhs.neon) }}
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

macro_rules! impl_shl_t_for_u32x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u32x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shl_all_u32_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: u32x4_shl(self.simd, rhs as u32) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_u32(self.neon, vmovq_n_s32(rhs as i32)) }}
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
impl_shl_t_for_u32x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u32x4 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u32x4 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shr_all_u32_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: u32x4_shr(self.simd, rhs as u32) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_u32(self.neon, vmovq_n_s32( -(rhs as i32))) }}
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
impl_shr_t_for_u32x4!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl u32x4 {
  #[inline]
  #[must_use]
  pub fn new(array: [u32; 4]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i32_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_eq(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vceqq_u32(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { u32::MAX } else { 0 },
          if self.arr[1] == rhs.arr[1] { u32::MAX } else { 0 },
          if self.arr[2] == rhs.arr[2] { u32::MAX } else { 0 },
          if self.arr[3] == rhs.arr[3] { u32::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_gt_mask_i32_m128i(self.sse,rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_gt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcgtq_u32(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { u32::MAX } else { 0 },
          if self.arr[1] > rhs.arr[1] { u32::MAX } else { 0 },
          if self.arr[2] > rhs.arr[2] { u32::MAX } else { 0 },
          if self.arr[3] > rhs.arr[3] { u32::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_lt_mask_i32_m128i(self.sse,rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_lt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcltq_u32(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { u32::MAX } else { 0 },
          if self.arr[1] < rhs.arr[1] { u32::MAX } else { 0 },
          if self.arr[2] < rhs.arr[2] { u32::MAX } else { 0 },
          if self.arr[3] < rhs.arr[3] { u32::MAX } else { 0 },
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
        unsafe {Self { neon: vbslq_u32(self.neon, t.neon, f.neon) }}
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
        Self { sse: max_u32_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_max(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmaxq_u32(self.neon, rhs.neon) }}
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmaxq_u16(self.neon, rhs.neon) }}
      } else {
        let arr: [u32; 4] = cast(self);
        let rhs: [u32; 4] = cast(rhs);
        cast([
          arr[0].max(rhs[0]),
          arr[1].max(rhs[1]),
          arr[2].max(rhs[2]),
          arr[3].max(rhs[3]),
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: min_u32_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u32x4_min(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vminq_u32(self.neon, rhs.neon) }}
      } else {
        let arr: [u32; 4] = cast(self);
        let rhs: [u32; 4] = cast(rhs);
        cast([
          arr[0].min(rhs[0]),
          arr[1].min(rhs[1]),
          arr[2].min(rhs[2]),
          arr[3].min(rhs[3]),
        ])
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [u32; 4] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[u32; 4] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_array_mut(&mut self) -> &mut[u32; 4] {
    cast_mut(self)
  }
}
