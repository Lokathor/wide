use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u64x2 { sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct u64x2 { simd: v128 }

    impl Default for u64x2 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u64x2 {
      fn eq(&self, other: &Self) -> bool {
        u64x2_all_true(u64x2_eq(self.simd, other.simd))
      }
    }

    impl Eq for u64x2 { }
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    use core::arch::aarch64::*;
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct u64x2 { neon : uint64x2_t }

    impl Default for u64x2 {
      #[inline]
      #[must_use]
      fn default() -> Self {
        unsafe { Self { neon: vdupq_n_u64(0)} }
      }
    }

    impl PartialEq for u64x2 {
      #[inline]
      #[must_use]
      fn eq(&self, other: &Self) -> bool {
        unsafe {
          vgetq_lane_u64(self.neon,0) == vgetq_lane_u64(other.neon,0) && vgetq_lane_u64(self.neon,1) == vgetq_lane_u64(other.neon,1)
        }
      }
    }

    impl Eq for u64x2 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u64x2 { arr: [u64;2] }
  }
}

int_uint_consts!(u64, 2, u64x2, u64x2, u64a2, const_u64_as_u64x2, 128);

unsafe impl Zeroable for u64x2 {}
unsafe impl Pod for u64x2 {}

impl Add for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_add(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_u64(self.neon, rhs.neon) } }
      } else {
        Self { arr: [
          self.arr[0].wrapping_add(rhs.arr[0]),
          self.arr[1].wrapping_add(rhs.arr[1]),
        ]}
      }
    }
  }
}

impl Sub for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_sub(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vsubq_u64(self.neon, rhs.neon) } }
      } else {
        Self { arr: [
          self.arr[0].wrapping_sub(rhs.arr[0]),
          self.arr[1].wrapping_sub(rhs.arr[1]),
        ]}
      }
    }
  }
}

//we should try to implement this on sse2
impl Mul for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_mul(self.simd, rhs.simd) }
      } else {
        let arr1: [u64; 2] = cast(self);
        let arr2: [u64; 2] = cast(rhs);
        cast([
          arr1[0].wrapping_mul(arr2[0]),
          arr1[1].wrapping_mul(arr2[1]),
        ])
      }
    }
  }
}

impl Add<u64> for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: u64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u64> for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u64> for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Add<u64x2> for u64 {
  type Output = u64x2;
  #[inline]
  #[must_use]
  fn add(self, rhs: u64x2) -> Self::Output {
    u64x2::splat(self).add(rhs)
  }
}

impl Sub<u64x2> for u64 {
  type Output = u64x2;
  #[inline]
  #[must_use]
  fn sub(self, rhs: u64x2) -> Self::Output {
    u64x2::splat(self).sub(rhs)
  }
}

impl Mul<u64x2> for u64 {
  type Output = u64x2;
  #[inline]
  #[must_use]
  fn mul(self, rhs: u64x2) -> Self::Output {
    u64x2::splat(self).mul(rhs)
  }
}

impl BitAnd for u64x2 {
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
        unsafe {Self { neon: vandq_u64(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          self.arr[0].bitand(rhs.arr[0]),
          self.arr[1].bitand(rhs.arr[1]),
        ]}
      }
    }
  }
}

impl BitOr for u64x2 {
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
        unsafe {Self { neon: vorrq_u64(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          self.arr[0].bitor(rhs.arr[0]),
          self.arr[1].bitor(rhs.arr[1]),
        ]}
      }
    }
  }
}

impl BitXor for u64x2 {
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
        unsafe {Self { neon: veorq_u64(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          self.arr[0].bitxor(rhs.arr[0]),
          self.arr[1].bitxor(rhs.arr[1]),
        ]}
      }
    }
  }
}

macro_rules! impl_shl_t_for_u64x2 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for u64x2 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shl_all_u64_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: u64x2_shl(self.simd, rhs as u32) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_u64(self.neon, vmovq_n_s64(rhs as i64)) }}
          } else {
            let u = rhs as u64;
            Self { arr: [
              self.arr[0] << u,
              self.arr[1] << u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shl_t_for_u64x2!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_u64x2 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for u64x2 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([rhs as u64, 0]);
            Self { sse: shr_all_u64_m128i(self.sse, shift) }
          } else if #[cfg(target_feature="simd128")] {
            Self { simd: u64x2_shr(self.simd, rhs as u32) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            unsafe {Self { neon: vshlq_u64(self.neon, vmovq_n_s64(-(rhs as i64))) }}
          } else {
            let u = rhs as u64;
            Self { arr: [
              self.arr[0] >> u,
              self.arr[1] >> u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shr_t_for_u64x2!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl u64x2 {
  #[inline]
  #[must_use]
  pub fn new(array: [u64; 2]) -> Self {
    Self::from(array)
  }
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: cmp_eq_mask_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_eq(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vceqq_u64(self.neon, rhs.neon) } }
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] == r[0] { -1_i64 } else { 0 },
          if s[1] == r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        Self { sse: cmp_gt_mask_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcgtq_u64(self.neon, rhs.neon) }}
      } else {
        // u64x2_gt on WASM is not a thing. https://github.com/WebAssembly/simd/pull/414
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] > r[0] { -1_i64 } else { 0 },
          if s[1] > r[1] { -1_i64 } else { 0 },
        ])
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
        unsafe {Self { neon: vbslq_u64(self.neon, t.neon, f.neon) }}
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }

  #[inline]
  pub fn to_array(self) -> [u64; 2] {
    cast(self)
  }

  #[inline]
  pub fn as_array_ref(&self) -> &[u64; 2] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_array_mut(&mut self) -> &mut [u64; 2] {
    cast_mut(self)
  }
}
