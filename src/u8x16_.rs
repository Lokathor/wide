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
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    use core::arch::aarch64::*;
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct u8x16 { pub(crate) neon : uint8x16_t }

    impl Default for u8x16 {
      #[inline]
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for u8x16 {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        unsafe { vminvq_u8(vceqq_u8(self.neon, other.neon))==u8::MAX }
      }
    }

    impl Eq for u8x16 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u8x16 { pub(crate) arr: [u8;16] }
  }
}

int_uint_consts!(u8, 16, u8x16, 128);

unsafe impl Zeroable for u8x16 {}
unsafe impl Pod for u8x16 {}

impl AlignTo for u8x16 {
  type Elem = u8;
}

impl Add for u8x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_add(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_u8(self.neon, rhs.neon) } }
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
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_sub(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vsubq_u8(self.neon, rhs.neon) }}
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

impl Mul for u8x16 {
  type Output = Self;

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    // For x86 and wasm, this technically can be done explicitly by converting
    // to `i16` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vmulq_u8(self.neon, rhs.neon) } }
      } else {
        let self_array: [u8; 16] = cast(self);
        let rhs_array: [u8; 16] = cast(rhs);

        Self::new([
          self_array[0].wrapping_mul(rhs_array[0]),
          self_array[1].wrapping_mul(rhs_array[1]),
          self_array[2].wrapping_mul(rhs_array[2]),
          self_array[3].wrapping_mul(rhs_array[3]),
          self_array[4].wrapping_mul(rhs_array[4]),
          self_array[5].wrapping_mul(rhs_array[5]),
          self_array[6].wrapping_mul(rhs_array[6]),
          self_array[7].wrapping_mul(rhs_array[7]),
          self_array[8].wrapping_mul(rhs_array[8]),
          self_array[9].wrapping_mul(rhs_array[9]),
          self_array[10].wrapping_mul(rhs_array[10]),
          self_array[11].wrapping_mul(rhs_array[11]),
          self_array[12].wrapping_mul(rhs_array[12]),
          self_array[13].wrapping_mul(rhs_array[13]),
          self_array[14].wrapping_mul(rhs_array[14]),
          self_array[15].wrapping_mul(rhs_array[15]),
        ])
      }
    }
  }
}

integer_impl_div_rem!(
  u8,
  u8x16,
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);

impl Shl for u8x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shl`)
  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Mask `rhs` to 7 to match `wrapping_shl`.
          let shift_by = vreinterpretq_s8_u8(vandq_u8(rhs.neon, vmovq_n_u8(7)));
          Self { neon: vshlq_u8(self.neon, shift_by) }
        }
      } else {
        let self_array: [u8; 16] = cast(self);
        let rhs_array: [u8; 16] = cast(rhs);

        Self::new([
          self_array[0].wrapping_shl(rhs_array[0] as u32),
          self_array[1].wrapping_shl(rhs_array[1] as u32),
          self_array[2].wrapping_shl(rhs_array[2] as u32),
          self_array[3].wrapping_shl(rhs_array[3] as u32),
          self_array[4].wrapping_shl(rhs_array[4] as u32),
          self_array[5].wrapping_shl(rhs_array[5] as u32),
          self_array[6].wrapping_shl(rhs_array[6] as u32),
          self_array[7].wrapping_shl(rhs_array[7] as u32),
          self_array[8].wrapping_shl(rhs_array[8] as u32),
          self_array[9].wrapping_shl(rhs_array[9] as u32),
          self_array[10].wrapping_shl(rhs_array[10] as u32),
          self_array[11].wrapping_shl(rhs_array[11] as u32),
          self_array[12].wrapping_shl(rhs_array[12] as u32),
          self_array[13].wrapping_shl(rhs_array[13] as u32),
          self_array[14].wrapping_shl(rhs_array[14] as u32),
          self_array[15].wrapping_shl(rhs_array[15] as u32),
        ])
      }
    }
  }
}

impl Shr for u8x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shr`)
  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `u16` or `u32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
          // there is no shift-right intrinsic.
          let neg_rhs = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(rhs.neon, vmovq_n_u8(7))));
          Self { neon: vshlq_u8(self.neon, neg_rhs) }
        }
      } else {
        let self_array: [u8; 16] = cast(self);
        let rhs_array: [u8; 16] = cast(rhs);

        Self::new([
          self_array[0].wrapping_shr(rhs_array[0] as u32),
          self_array[1].wrapping_shr(rhs_array[1] as u32),
          self_array[2].wrapping_shr(rhs_array[2] as u32),
          self_array[3].wrapping_shr(rhs_array[3] as u32),
          self_array[4].wrapping_shr(rhs_array[4] as u32),
          self_array[5].wrapping_shr(rhs_array[5] as u32),
          self_array[6].wrapping_shr(rhs_array[6] as u32),
          self_array[7].wrapping_shr(rhs_array[7] as u32),
          self_array[8].wrapping_shr(rhs_array[8] as u32),
          self_array[9].wrapping_shr(rhs_array[9] as u32),
          self_array[10].wrapping_shr(rhs_array[10] as u32),
          self_array[11].wrapping_shr(rhs_array[11] as u32),
          self_array[12].wrapping_shr(rhs_array[12] as u32),
          self_array[13].wrapping_shr(rhs_array[13] as u32),
          self_array[14].wrapping_shr(rhs_array[14] as u32),
          self_array[15].wrapping_shr(rhs_array[15] as u32),
        ])
      }
    }
  }
}

impl Add<u8> for u8x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: u8) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<u8> for u8x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: u8) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<u8> for u8x16 {
  type Output = Self;

  #[inline]
  fn mul(self, rhs: u8) -> Self::Output {
    self * Self::splat(rhs)
  }
}

macro_rules! impl_shl_scalar {
  ($Rhs:ident) => {
    impl Shl<$Rhs> for u8x16 {
      type Output = Self;

      /// Shifts all lanes by a uniform value.
      ///
      /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
      /// of the type. (same as `wrapping_shl`)
      #[inline]
      fn shl(self, rhs: $Rhs) -> Self::Output {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shl`.
            Self { simd: u8x16_shl(self.simd, rhs as u32 & 7) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shl`.
            unsafe { Self { neon: vshlq_u8(self.neon, vmovq_n_s8(rhs as i8 & 7)) } }
          } else {
            let self_array = self.to_array();
            let rhs = rhs as u32;

            cast([
              self_array[0].wrapping_shl(rhs),
              self_array[1].wrapping_shl(rhs),
              self_array[2].wrapping_shl(rhs),
              self_array[3].wrapping_shl(rhs),
              self_array[4].wrapping_shl(rhs),
              self_array[5].wrapping_shl(rhs),
              self_array[6].wrapping_shl(rhs),
              self_array[7].wrapping_shl(rhs),
              self_array[8].wrapping_shl(rhs),
              self_array[9].wrapping_shl(rhs),
              self_array[10].wrapping_shl(rhs),
              self_array[11].wrapping_shl(rhs),
              self_array[12].wrapping_shl(rhs),
              self_array[13].wrapping_shl(rhs),
              self_array[14].wrapping_shl(rhs),
              self_array[15].wrapping_shl(rhs),
            ])
          }
        }
      }
    }
  };
}
impl_shl_scalar!(i8);
impl_shl_scalar!(u8);
impl_shl_scalar!(i16);
impl_shl_scalar!(u16);
impl_shl_scalar!(i32);
impl_shl_scalar!(u32);
impl_shl_scalar!(i64);
impl_shl_scalar!(u64);
impl_shl_scalar!(i128);
impl_shl_scalar!(u128);

macro_rules! impl_shr_scalar {
  ($Rhs:ident) => {
    impl Shr<$Rhs> for u8x16 {
      type Output = Self;

      /// Shifts all lanes by a uniform value.
      ///
      /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
      /// of the type. (same as `wrapping_shr`)
      #[inline]
      fn shr(self, rhs: $Rhs) -> Self::Output {
        // For x86, this technically can be done explicitly by converting
        // to `u16` or `u32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shr`.
            Self { simd: u8x16_shr(self.simd, rhs as u32 & 7) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
            // there is no shift-right intrinsic.
            unsafe { Self { neon: vshlq_u8(self.neon, vmovq_n_s8(-(rhs as i8 & 7))) } }
          } else {
            let self_array = self.to_array();
            let rhs = rhs as u32;

            cast([
              self_array[0].wrapping_shr(rhs),
              self_array[1].wrapping_shr(rhs),
              self_array[2].wrapping_shr(rhs),
              self_array[3].wrapping_shr(rhs),
              self_array[4].wrapping_shr(rhs),
              self_array[5].wrapping_shr(rhs),
              self_array[6].wrapping_shr(rhs),
              self_array[7].wrapping_shr(rhs),
              self_array[8].wrapping_shr(rhs),
              self_array[9].wrapping_shr(rhs),
              self_array[10].wrapping_shr(rhs),
              self_array[11].wrapping_shr(rhs),
              self_array[12].wrapping_shr(rhs),
              self_array[13].wrapping_shr(rhs),
              self_array[14].wrapping_shr(rhs),
              self_array[15].wrapping_shr(rhs),
            ])
          }
        }
      }
    }
  };
}
impl_shr_scalar!(i8);
impl_shr_scalar!(u8);
impl_shr_scalar!(i16);
impl_shr_scalar!(u16);
impl_shr_scalar!(i32);
impl_shr_scalar!(u32);
impl_shr_scalar!(i64);
impl_shr_scalar!(u64);
impl_shr_scalar!(i128);
impl_shr_scalar!(u128);

impl Add<u8x16> for u8 {
  type Output = u8x16;
  #[inline]
  fn add(self, rhs: u8x16) -> Self::Output {
    u8x16::splat(self).add(rhs)
  }
}

impl Sub<u8x16> for u8 {
  type Output = u8x16;
  #[inline]
  fn sub(self, rhs: u8x16) -> Self::Output {
    u8x16::splat(self).sub(rhs)
  }
}

impl Mul<u8x16> for u8 {
  type Output = u8x16;

  #[inline]
  fn mul(self, rhs: u8x16) -> Self::Output {
    u8x16::splat(self) * rhs
  }
}

impl BitAnd for u8x16 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitand_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_and(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vandq_u8(self.neon, rhs.neon) }}
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
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitor_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_or(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vorrq_u8(self.neon, rhs.neon) }}
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
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_xor(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: veorq_u8(self.neon, rhs.neon) }}
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

impl CmpEq for u8x16 {
  type Output = Self;
  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    Self::simd_eq(self, rhs)
  }
}

impl CmpNe for u8x16 {
  type Output = Self;
  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_eq(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_ne(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        Self { arr: [
          if self.arr[0] != rhs.arr[0] { u8::MAX } else { 0 },
          if self.arr[1] != rhs.arr[1] { u8::MAX } else { 0 },
          if self.arr[2] != rhs.arr[2] { u8::MAX } else { 0 },
          if self.arr[3] != rhs.arr[3] { u8::MAX } else { 0 },
          if self.arr[4] != rhs.arr[4] { u8::MAX } else { 0 },
          if self.arr[5] != rhs.arr[5] { u8::MAX } else { 0 },
          if self.arr[6] != rhs.arr[6] { u8::MAX } else { 0 },
          if self.arr[7] != rhs.arr[7] { u8::MAX } else { 0 },
          if self.arr[8] != rhs.arr[8] { u8::MAX } else { 0 },
          if self.arr[9] != rhs.arr[9] { u8::MAX } else { 0 },
          if self.arr[10] != rhs.arr[10] { u8::MAX } else { 0 },
          if self.arr[11] != rhs.arr[11] { u8::MAX } else { 0 },
          if self.arr[12] != rhs.arr[12] { u8::MAX } else { 0 },
          if self.arr[13] != rhs.arr[13] { u8::MAX } else { 0 },
          if self.arr[14] != rhs.arr[14] { u8::MAX } else { 0 },
          if self.arr[15] != rhs.arr[15] { u8::MAX } else { 0 },
        ]}
      }
    }
  }
}

impl CmpLt for u8x16 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    Self::simd_lt(self, rhs)
  }
}

impl CmpLe for u8x16 {
  type Output = Self;
  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    Self::simd_le(self, rhs)
  }
}

impl CmpGe for u8x16 {
  type Output = Self;
  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    Self::simd_ge(self, rhs)
  }
}

impl CmpGt for u8x16 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    Self::simd_gt(self, rhs)
  }
}

impl u8x16 {
  #[inline]
  #[must_use]
  pub const fn new(array: [u8; 16]) -> Self {
    unsafe { core::mem::transmute(array) }
  }
  #[inline]
  #[must_use]
  pub fn simd_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_eq(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vceqq_u8(self.neon, rhs.neon) }}
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
  pub fn simd_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).sse;
        let rhs_i8 = rhs.bitxor(offset).sse;
        Self { sse: cmp_lt_mask_i8_m128i(self_i8, rhs_i8) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_lt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcltq_u8(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { u8::MAX } else { 0 },
          if self.arr[1] < rhs.arr[1] { u8::MAX } else { 0 },
          if self.arr[2] < rhs.arr[2] { u8::MAX } else { 0 },
          if self.arr[3] < rhs.arr[3] { u8::MAX } else { 0 },
          if self.arr[4] < rhs.arr[4] { u8::MAX } else { 0 },
          if self.arr[5] < rhs.arr[5] { u8::MAX } else { 0 },
          if self.arr[6] < rhs.arr[6] { u8::MAX } else { 0 },
          if self.arr[7] < rhs.arr[7] { u8::MAX } else { 0 },
          if self.arr[8] < rhs.arr[8] { u8::MAX } else { 0 },
          if self.arr[9] < rhs.arr[9] { u8::MAX } else { 0 },
          if self.arr[10] < rhs.arr[10] { u8::MAX } else { 0 },
          if self.arr[11] < rhs.arr[11] { u8::MAX } else { 0 },
          if self.arr[12] < rhs.arr[12] { u8::MAX } else { 0 },
          if self.arr[13] < rhs.arr[13] { u8::MAX } else { 0 },
          if self.arr[14] < rhs.arr[14] { u8::MAX } else { 0 },
          if self.arr[15] < rhs.arr[15] { u8::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn simd_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).sse;
        let rhs_i8 = rhs.bitxor(offset).sse;
        // a <= b  is equivalent to  !(b < a)  or  !(a > b)
        let gt_mask = u8x16 { sse: cmp_gt_mask_i8_m128i(self_i8, rhs_i8) };
        Self { sse: gt_mask.bitxor(u8x16::splat(0xFF)).sse }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_le(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcleq_u8(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] <= rhs.arr[0] { u8::MAX } else { 0 },
          if self.arr[1] <= rhs.arr[1] { u8::MAX } else { 0 },
          if self.arr[2] <= rhs.arr[2] { u8::MAX } else { 0 },
          if self.arr[3] <= rhs.arr[3] { u8::MAX } else { 0 },
          if self.arr[4] <= rhs.arr[4] { u8::MAX } else { 0 },
          if self.arr[5] <= rhs.arr[5] { u8::MAX } else { 0 },
          if self.arr[6] <= rhs.arr[6] { u8::MAX } else { 0 },
          if self.arr[7] <= rhs.arr[7] { u8::MAX } else { 0 },
          if self.arr[8] <= rhs.arr[8] { u8::MAX } else { 0 },
          if self.arr[9] <= rhs.arr[9] { u8::MAX } else { 0 },
          if self.arr[10] <= rhs.arr[10] { u8::MAX } else { 0 },
          if self.arr[11] <= rhs.arr[11] { u8::MAX } else { 0 },
          if self.arr[12] <= rhs.arr[12] { u8::MAX } else { 0 },
          if self.arr[13] <= rhs.arr[13] { u8::MAX } else { 0 },
          if self.arr[14] <= rhs.arr[14] { u8::MAX } else { 0 },
          if self.arr[15] <= rhs.arr[15] { u8::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn simd_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).sse;
        let rhs_i8 = rhs.bitxor(offset).sse;
        // a >= b  is equivalent to  !(b > a)  or  !(a < b)
        let lt_mask = u8x16 { sse: cmp_lt_mask_i8_m128i(self_i8, rhs_i8) };
        Self { sse: lt_mask.bitxor(u8x16::splat(0xFF)).sse }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_ge(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcgeq_u8(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] >= rhs.arr[0] { u8::MAX } else { 0 },
          if self.arr[1] >= rhs.arr[1] { u8::MAX } else { 0 },
          if self.arr[2] >= rhs.arr[2] { u8::MAX } else { 0 },
          if self.arr[3] >= rhs.arr[3] { u8::MAX } else { 0 },
          if self.arr[4] >= rhs.arr[4] { u8::MAX } else { 0 },
          if self.arr[5] >= rhs.arr[5] { u8::MAX } else { 0 },
          if self.arr[6] >= rhs.arr[6] { u8::MAX } else { 0 },
          if self.arr[7] >= rhs.arr[7] { u8::MAX } else { 0 },
          if self.arr[8] >= rhs.arr[8] { u8::MAX } else { 0 },
          if self.arr[9] >= rhs.arr[9] { u8::MAX } else { 0 },
          if self.arr[10] >= rhs.arr[10] { u8::MAX } else { 0 },
          if self.arr[11] >= rhs.arr[11] { u8::MAX } else { 0 },
          if self.arr[12] >= rhs.arr[12] { u8::MAX } else { 0 },
          if self.arr[13] >= rhs.arr[13] { u8::MAX } else { 0 },
          if self.arr[14] >= rhs.arr[14] { u8::MAX } else { 0 },
          if self.arr[15] >= rhs.arr[15] { u8::MAX } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn simd_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Convert from u8 to i8.
        let offset = Self::splat(0x80);
        let self_i8 = self.bitxor(offset).sse;
        let rhs_i8 = rhs.bitxor(offset).sse;
        Self { sse: cmp_gt_mask_i8_m128i(self_i8, rhs_i8) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_gt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcgtq_u8(self.neon, rhs.neon) }}
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { u8::MAX } else { 0 },
          if self.arr[1] > rhs.arr[1] { u8::MAX } else { 0 },
          if self.arr[2] > rhs.arr[2] { u8::MAX } else { 0 },
          if self.arr[3] > rhs.arr[3] { u8::MAX } else { 0 },
          if self.arr[4] > rhs.arr[4] { u8::MAX } else { 0 },
          if self.arr[5] > rhs.arr[5] { u8::MAX } else { 0 },
          if self.arr[6] > rhs.arr[6] { u8::MAX } else { 0 },
          if self.arr[7] > rhs.arr[7] { u8::MAX } else { 0 },
          if self.arr[8] > rhs.arr[8] { u8::MAX } else { 0 },
          if self.arr[9] > rhs.arr[9] { u8::MAX } else { 0 },
          if self.arr[10] > rhs.arr[10] { u8::MAX } else { 0 },
          if self.arr[11] > rhs.arr[11] { u8::MAX } else { 0 },
          if self.arr[12] > rhs.arr[12] { u8::MAX } else { 0 },
          if self.arr[13] > rhs.arr[13] { u8::MAX } else { 0 },
          if self.arr[14] > rhs.arr[14] { u8::MAX } else { 0 },
          if self.arr[15] > rhs.arr[15] { u8::MAX } else { 0 },
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
        unsafe {Self { neon: vbslq_u8(self.neon, t.neon, f.neon) }}
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> u8 {
    cast(i8x16::reduce_add(cast(self)))
  }

  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> u8 {
    #[allow(dead_code)]
    const SHUFFLE_1: [i8; 16] =
      [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_2: [i8; 16] =
      [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_3: [i8; 16] =
      [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_4: [i8; 16] =
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    pick! {
      if #[cfg(target_feature="ssse3")] {
        let rhs = shuffle_av_i8z_all_m128i(self.sse, m128i::from(SHUFFLE_1));
        let max = max_u8_m128i(self.sse, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_2));
        let max = max_u8_m128i(max, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_3));
        let max = max_u8_m128i(max, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_4));
        let max = max_u8_m128i(max, rhs);
        get_i32_from_m128i_s(max) as u8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.simd, self.simd);
        let max = u8x16_max(self.simd, rhs);
        let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = u8x16_max(max, rhs);
        let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = u8x16_max(max, rhs);
        let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = u8x16_max(max, rhs);
        u8x16_extract_lane::<0>(max)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `uint8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_u8(self.neon, core::mem::transmute(SHUFFLE_1));
          let max = vmaxq_u8(self.neon, rhs);
          let rhs = vqtbl1q_u8(max, core::mem::transmute(SHUFFLE_2));
          let max = vmaxq_u8(max, rhs);
          let rhs = vqtbl1q_u8(max, core::mem::transmute(SHUFFLE_3));
          let max = vmaxq_u8(max, rhs);
          let rhs = vqtbl1q_u8(max, core::mem::transmute(SHUFFLE_4));
          let max = vmaxq_u8(max, rhs);
          vgetq_lane_u8(max, 0)
        }
      } else {
        let array: [u8; 16] = cast(self);
        array.into_iter().reduce(u8::max).unwrap()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> u8 {
    #[allow(dead_code)]
    const SHUFFLE_1: [i8; 16] =
      [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_2: [i8; 16] =
      [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_3: [i8; 16] =
      [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[allow(dead_code)]
    const SHUFFLE_4: [i8; 16] =
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    pick! {
      if #[cfg(target_feature="ssse3")] {
        let rhs = shuffle_av_i8z_all_m128i(self.sse, m128i::from(SHUFFLE_1));
        let min = min_u8_m128i(self.sse, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_2));
        let min = min_u8_m128i(min, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_3));
        let min = min_u8_m128i(min, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_4));
        let min = min_u8_m128i(min, rhs);
        get_i32_from_m128i_s(min) as u8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.simd, self.simd);
        let min = u8x16_min(self.simd, rhs);
        let rhs = u8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = u8x16_min(min, rhs);
        let rhs = u8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = u8x16_min(min, rhs);
        let rhs = u8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = u8x16_min(min, rhs);
        u8x16_extract_lane::<0>(min)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `uint8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_u8(self.neon, core::mem::transmute(SHUFFLE_1));
          let min = vminq_u8(self.neon, rhs);
          let rhs = vqtbl1q_u8(min, core::mem::transmute(SHUFFLE_2));
          let min = vminq_u8(min, rhs);
          let rhs = vqtbl1q_u8(min, core::mem::transmute(SHUFFLE_3));
          let min = vminq_u8(min, rhs);
          let rhs = vqtbl1q_u8(min, core::mem::transmute(SHUFFLE_4));
          let min = vminq_u8(min, rhs);
          vgetq_lane_u8(min, 0)
        }
      } else {
        let array: [u8; 16] = cast(self);
        array.into_iter().reduce(u8::min).unwrap()
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmaxq_u8(self.neon, rhs.neon) }}
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vminq_u8(self.neon, rhs.neon) }}
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

  integer_fn_clamp!();

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_saturating_u8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_add_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vqaddq_u8(self.neon, rhs.neon) }}
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
          self.arr[8].saturating_add(rhs.arr[8]),
          self.arr[9].saturating_add(rhs.arr[9]),
          self.arr[10].saturating_add(rhs.arr[10]),
          self.arr[11].saturating_add(rhs.arr[11]),
          self.arr[12].saturating_add(rhs.arr[12]),
          self.arr[13].saturating_add(rhs.arr[13]),
          self.arr[14].saturating_add(rhs.arr[14]),
          self.arr[15].saturating_add(rhs.arr[15]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_saturating_u8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u8x16_sub_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqsubq_u8(self.neon, rhs.neon) } }
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
          self.arr[8].saturating_sub(rhs.arr[8]),
          self.arr[9].saturating_sub(rhs.arr[9]),
          self.arr[10].saturating_sub(rhs.arr[10]),
          self.arr[11].saturating_sub(rhs.arr[11]),
          self.arr[12].saturating_sub(rhs.arr[12]),
          self.arr[13].saturating_sub(rhs.arr[13]),
          self.arr[14].saturating_sub(rhs.arr[14]),
          self.arr[15].saturating_sub(rhs.arr[15]),
        ]}
      }
    }
  }

  /// Lanewise saturating multiply.
  #[inline]
  #[must_use]
  pub fn saturating_mul(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          let low_wide_mul = vreinterpretq_u8_u16(
            vmull_u8(vget_low_u8(self.neon), vget_low_u8(rhs.neon)),
          );
          let high_wide_mul = vreinterpretq_u8_u16(
            vmull_u8(vget_high_u8(self.neon), vget_high_u8(rhs.neon)),
          );
          let low = Self { neon: vtrn1q_u8(low_wide_mul, high_wide_mul) };
          let high = Self { neon: vtrn2q_u8(low_wide_mul, high_wide_mul) };

          let no_overflow = high.simd_eq(Self::ZERO);
          no_overflow.blend(low, Self::MAX)
        }
      } else {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([
          self_array[0].saturating_mul(rhs_array[0]),
          self_array[1].saturating_mul(rhs_array[1]),
          self_array[2].saturating_mul(rhs_array[2]),
          self_array[3].saturating_mul(rhs_array[3]),
          self_array[4].saturating_mul(rhs_array[4]),
          self_array[5].saturating_mul(rhs_array[5]),
          self_array[6].saturating_mul(rhs_array[6]),
          self_array[7].saturating_mul(rhs_array[7]),
          self_array[8].saturating_mul(rhs_array[8]),
          self_array[9].saturating_mul(rhs_array[9]),
          self_array[10].saturating_mul(rhs_array[10]),
          self_array[11].saturating_mul(rhs_array[11]),
          self_array[12].saturating_mul(rhs_array[12]),
          self_array[13].saturating_mul(rhs_array[13]),
          self_array[14].saturating_mul(rhs_array[14]),
          self_array[15].saturating_mul(rhs_array[15]),
        ])
      }
    }
  }

  integer_fn_saturating_div!([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  ]);

  /// Unpack and interleave low lanes of two `u8x16`
  #[inline]
  #[must_use]
  pub fn unpack_low(lhs: u8x16, rhs: u8x16) -> u8x16 {
    pick! {
        if #[cfg(target_feature = "sse2")] {
            u8x16 { sse: unpack_low_i8_m128i(lhs.sse, rhs.sse) }
        } else if #[cfg(target_feature = "simd128")] {
          u8x16 { simd: u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(lhs.simd, rhs.simd) }
        } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vget_low_u8(lhs.neon) };
            let rhs = unsafe { vget_low_u8(rhs.neon) };

            let zipped = unsafe { vzip_u8(lhs, rhs) };
            u8x16 { neon: unsafe { vcombine_u8(zipped.0, zipped.1) } }
        } else {
            u8x16::new([
                lhs.as_array()[0], rhs.as_array()[0],
                lhs.as_array()[1], rhs.as_array()[1],
                lhs.as_array()[2], rhs.as_array()[2],
                lhs.as_array()[3], rhs.as_array()[3],
                lhs.as_array()[4], rhs.as_array()[4],
                lhs.as_array()[5], rhs.as_array()[5],
                lhs.as_array()[6], rhs.as_array()[6],
                lhs.as_array()[7], rhs.as_array()[7],
            ])
        }
    }
  }

  /// Unpack and interleave high lanes of two `u8x16`
  #[inline]
  #[must_use]
  pub fn unpack_high(lhs: u8x16, rhs: u8x16) -> u8x16 {
    pick! {
        if #[cfg(target_feature = "sse2")] {
            u8x16 { sse: unpack_high_i8_m128i(lhs.sse, rhs.sse) }
        } else if #[cfg(target_feature = "simd128")] {
            u8x16 { simd: u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(lhs.simd, rhs.simd) }
        } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vget_high_u8(lhs.neon) };
            let rhs = unsafe { vget_high_u8(rhs.neon) };

            let zipped = unsafe { vzip_u8(lhs, rhs) };
            u8x16 { neon: unsafe { vcombine_u8(zipped.0, zipped.1) } }
        } else {
            u8x16::new([
                lhs.as_array()[8], rhs.as_array()[8],
                lhs.as_array()[9], rhs.as_array()[9],
                lhs.as_array()[10], rhs.as_array()[10],
                lhs.as_array()[11], rhs.as_array()[11],
                lhs.as_array()[12], rhs.as_array()[12],
                lhs.as_array()[13], rhs.as_array()[13],
                lhs.as_array()[14], rhs.as_array()[14],
                lhs.as_array()[15], rhs.as_array()[15],
            ])
        }
    }
  }

  /// Pack and saturate two `i16x8` to `u8x16`
  #[inline]
  #[must_use]
  pub fn narrow_i16x8(lhs: i16x8, rhs: i16x8) -> Self {
    pick! {
        if #[cfg(target_feature = "sse2")] {
            u8x16 { sse: pack_i16_to_u8_m128i(lhs.sse, rhs.sse) }
        } else if #[cfg(target_feature = "simd128")] {
            u8x16 { simd: u8x16_narrow_i16x8(lhs.simd, rhs.simd) }
        } else if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
            let lhs = unsafe { vqmovun_s16(lhs.neon) };
            let rhs = unsafe { vqmovun_s16(rhs.neon) };
            u8x16 { neon: unsafe { vcombine_u8(lhs, rhs) } }
        } else {
            fn clamp(a: i16) -> u8 {
                  if a < u8::MIN as i16 {
                      u8::MIN
                  } else if a > u8::MAX as i16 {
                      u8::MAX
                  } else {
                      a as u8
                  }
            }

            Self { arr: [
                clamp(lhs.as_array()[0]),
                clamp(lhs.as_array()[1]),
                clamp(lhs.as_array()[2]),
                clamp(lhs.as_array()[3]),
                clamp(lhs.as_array()[4]),
                clamp(lhs.as_array()[5]),
                clamp(lhs.as_array()[6]),
                clamp(lhs.as_array()[7]),
                clamp(rhs.as_array()[0]),
                clamp(rhs.as_array()[1]),
                clamp(rhs.as_array()[2]),
                clamp(rhs.as_array()[3]),
                clamp(rhs.as_array()[4]),
                clamp(rhs.as_array()[5]),
                clamp(rhs.as_array()[6]),
                clamp(rhs.as_array()[7]),
            ]}
        }
    }
  }

  /// Returns a new vector where each element is based on the index values in
  /// `rhs`.
  ///
  /// * Index values in the range `[0, 15]` select the i-th element of `self`.
  /// * Index values that are out of range will cause that output lane to be
  ///   `0`.
  #[inline]
  pub fn swizzle(self, rhs: i8x16) -> i8x16 {
    cast(i8x16::swizzle(cast(self), rhs))
  }

  /// Works like [`swizzle`](Self::swizzle) with the following additional
  /// details
  ///
  /// * Indices in the range `[0, 15]` will select the i-th element of `self`.
  /// * If the high bit of any index is set (meaning that the index is
  ///   negative), then the corresponding output lane is guaranteed to be zero.
  /// * Otherwise the output lane is either `0` or `self[rhs[i] % 16]`,
  ///   depending on the implementation.
  #[inline]
  pub fn swizzle_relaxed(self, rhs: u8x16) -> u8x16 {
    cast(i8x16::swizzle_relaxed(cast(self), cast(rhs)))
  }

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> u32 {
    i8x16::to_bitmask(cast(self)) as u32
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    i8x16::any(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    i8x16::all(cast(self))
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    i8x16::none(cast(self))
  }

  /// Transpose matrix of 16x16 `u8` matrix. Currently not accelerated.
  #[must_use]
  #[inline]
  pub fn transpose(data: [u8x16; 16]) -> [u8x16; 16] {
    cast(i8x16::transpose(cast(data)))
  }

  #[inline]
  pub fn to_array(self) -> [u8; 16] {
    cast(self)
  }

  #[inline]
  pub fn as_array(&self) -> &[u8; 16] {
    cast_ref(self)
  }

  #[inline]
  pub fn as_mut_array(&mut self) -> &mut [u8; 16] {
    cast_mut(self)
  }
}
