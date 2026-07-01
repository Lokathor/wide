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
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    use core::arch::aarch64::*;
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct i8x16 { pub(crate) neon : int8x16_t }

    impl Default for i8x16 {
      #[inline]
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i8x16 {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        unsafe { vminvq_u8(vceqq_s8(self.neon, other.neon))==u8::MAX }
      }
    }

    impl Eq for i8x16 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i8x16 { arr: [i8;16] }
  }
}

impl_simd! {
  T = i8,
  N = 16,
  Simd = i8x16,

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_eq(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s8_u8(vceqq_s8(self.neon, rhs.neon)) }}
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

  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_eq(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_ne(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        Self { arr: [
          if self.arr[0] != rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] != rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] != rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] != rhs.arr[3] { -1 } else { 0 },
          if self.arr[4] != rhs.arr[4] { -1 } else { 0 },
          if self.arr[5] != rhs.arr[5] { -1 } else { 0 },
          if self.arr[6] != rhs.arr[6] { -1 } else { 0 },
          if self.arr[7] != rhs.arr[7] { -1 } else { 0 },
          if self.arr[8] != rhs.arr[8] { -1 } else { 0 },
          if self.arr[9] != rhs.arr[9] { -1 } else { 0 },
          if self.arr[10] != rhs.arr[10] { -1 } else { 0 },
          if self.arr[11] != rhs.arr[11] { -1 } else { 0 },
          if self.arr[12] != rhs.arr[12] { -1 } else { 0 },
          if self.arr[13] != rhs.arr[13] { -1 } else { 0 },
          if self.arr[14] != rhs.arr[14] { -1 } else { 0 },
          if self.arr[15] != rhs.arr[15] { -1 } else { 0 },
        ]}
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_lt_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_lt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s8_u8(vcltq_s8(self.neon, rhs.neon)) }}
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

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_gt_mask_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_gt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s8_u8(vcgtq_s8(self.neon, rhs.neon)) }}
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

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_gt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_le(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_gt(rhs)
      } else {
        Self { arr: [
          if self.arr[0] <= rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] <= rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] <= rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] <= rhs.arr[3] { -1 } else { 0 },
          if self.arr[4] <= rhs.arr[4] { -1 } else { 0 },
          if self.arr[5] <= rhs.arr[5] { -1 } else { 0 },
          if self.arr[6] <= rhs.arr[6] { -1 } else { 0 },
          if self.arr[7] <= rhs.arr[7] { -1 } else { 0 },
          if self.arr[8] <= rhs.arr[8] { -1 } else { 0 },
          if self.arr[9] <= rhs.arr[9] { -1 } else { 0 },
          if self.arr[10] <= rhs.arr[10] { -1 } else { 0 },
          if self.arr[11] <= rhs.arr[11] { -1 } else { 0 },
          if self.arr[12] <= rhs.arr[12] { -1 } else { 0 },
          if self.arr[13] <= rhs.arr[13] { -1 } else { 0 },
          if self.arr[14] <= rhs.arr[14] { -1 } else { 0 },
          if self.arr[15] <= rhs.arr[15] { -1 } else { 0 },
        ]}
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        !self.simd_lt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_ge(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_lt(rhs)
      } else {
        Self { arr: [
          if self.arr[0] >= rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] >= rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] >= rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] >= rhs.arr[3] { -1 } else { 0 },
          if self.arr[4] >= rhs.arr[4] { -1 } else { 0 },
          if self.arr[5] >= rhs.arr[5] { -1 } else { 0 },
          if self.arr[6] >= rhs.arr[6] { -1 } else { 0 },
          if self.arr[7] >= rhs.arr[7] { -1 } else { 0 },
          if self.arr[8] >= rhs.arr[8] { -1 } else { 0 },
          if self.arr[9] >= rhs.arr[9] { -1 } else { 0 },
          if self.arr[10] >= rhs.arr[10] { -1 } else { 0 },
          if self.arr[11] >= rhs.arr[11] { -1 } else { 0 },
          if self.arr[12] >= rhs.arr[12] { -1 } else { 0 },
          if self.arr[13] >= rhs.arr[13] { -1 } else { 0 },
          if self.arr[14] >= rhs.arr[14] { -1 } else { 0 },
          if self.arr[15] >= rhs.arr[15] { -1 } else { 0 },
        ]}
      }
    }
  }
}

int_uint_consts!(i8, 16, i8x16, 128);

unsafe impl Zeroable for i8x16 {}
unsafe impl Pod for i8x16 {}

impl AlignTo for i8x16 {
  type Elem = i8;
}

impl Add for i8x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_add(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_s8(self.neon, rhs.neon) } }
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
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_sub(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vsubq_s8(self.neon, rhs.neon) }}
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

impl Mul for i8x16 {
  type Output = Self;

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    // For x86 and wasm, this technically can be done explicitly by converting
    // to `i16` then converting back after multiplication, but that may not
    // actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vmulq_s8(self.neon, rhs.neon) } }
      } else {
        let self_array: [i8; 16] = cast(self);
        let rhs_array: [i8; 16] = cast(rhs);

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
  i8,
  i8x16,
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);

impl Shl for i8x16 {
  type Output = Self;

  /// Shifts lanes by the corresponding lane.
  ///
  /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
  /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
  /// of the type. (same as `wrapping_shl`)
  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `i16` or `i32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Mask `rhs` to 7 to match `wrapping_shl`.
          let shift_by = vandq_s8(rhs.neon, vmovq_n_s8(7));
          Self { neon: vshlq_s8(self.neon, shift_by) }
        }
      } else {
        let self_array: [i8; 16] = cast(self);
        let rhs_array: [i8; 16] = cast(rhs);

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

impl Shr for i8x16 {
  type Output = Self;

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    // For x86, this technically can be done explicitly by converting
    // to `i16` or `i32` then converting back after multiplication, but that may
    // not actually be faster than auto-vectorization.
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
          // there is no shift-right intrinsic.
          let neg_rhs = vnegq_s8(vandq_s8(rhs.neon, vmovq_n_s8(7)));
          Self { neon: vshlq_s8(self.neon, neg_rhs) }
        }
      } else {
        let self_array: [i8; 16] = cast(self);
        let rhs_array: [i8; 16] = cast(rhs);

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

impl Add<i8> for i8x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: i8) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<i8> for i8x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: i8) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<i8> for i8x16 {
  type Output = Self;

  #[inline]
  fn mul(self, rhs: i8) -> Self::Output {
    self * Self::splat(rhs)
  }
}

macro_rules! impl_shl_scalar {
  ($Rhs:ident) => {
    impl Shl<$Rhs> for i8x16 {
      type Output = Self;

      /// Shifts all lanes by a uniform value.
      ///
      /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
      /// of the type. (same as `wrapping_shl`)
      #[inline]
      fn shl(self, rhs: $Rhs) -> Self::Output {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shl`.
            Self { simd: i8x16_shl(self.simd, rhs as u32 & 7) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shl`.
            unsafe { Self { neon: vshlq_s8(self.neon, vmovq_n_s8(rhs as i8 & 7)) } }
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
    impl Shr<$Rhs> for i8x16 {
      type Output = Self;

      /// Shifts all lanes by a uniform value.
      ///
      /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the bitwidth
      /// of the type. (same as `wrapping_shr`)
      #[inline]
      fn shr(self, rhs: $Rhs) -> Self::Output {
        // For x86, this technically can be done explicitly by converting
        // to `i16` or `i32` then converting back after multiplication, but that
        // may not actually be faster than auto-vectorization.
        pick! {
          if #[cfg(target_feature="simd128")] {
            // Mask `rhs` to 7 to match `wrapping_shr`.
            Self { simd: i8x16_shr(self.simd, rhs as u32 & 7) }
          } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
            // Mask `rhs` to 7 to match `wrapping_shr`, and negate it because
            // there is no shift-right intrinsic.
            unsafe { Self { neon: vshlq_s8(self.neon, vmovq_n_s8(-(rhs as i8 & 7))) } }
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

impl Add<i8x16> for i8 {
  type Output = i8x16;
  #[inline]
  fn add(self, rhs: i8x16) -> Self::Output {
    i8x16::splat(self).add(rhs)
  }
}

impl Sub<i8x16> for i8 {
  type Output = i8x16;
  #[inline]
  fn sub(self, rhs: i8x16) -> Self::Output {
    i8x16::splat(self).sub(rhs)
  }
}

impl Mul<i8x16> for i8 {
  type Output = i8x16;

  #[inline]
  fn mul(self, rhs: i8x16) -> Self::Output {
    i8x16::splat(self) * rhs
  }
}

impl BitAnd for i8x16 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitand_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_and(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vandq_s8(self.neon, rhs.neon) }}
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
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitor_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_or(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vorrq_s8(self.neon, rhs.neon) }}
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
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_xor(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: veorq_s8(self.neon, rhs.neon) }}
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

impl i8x16 {
  /// converts `i16` to `i8`, saturating values that are too large
  #[inline]
  #[must_use]
  pub fn from_i16x16_saturate(v: i16x16) -> i8x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        i8x16 { sse: pack_i16_to_i8_m128i( extract_m128i_from_m256i::<0>(v.avx2), extract_m128i_from_m256i::<1>(v.avx2))  }
      } else if #[cfg(target_feature="sse2")] {
        i8x16 { sse: pack_i16_to_i8_m128i( v.a.sse, v.b.sse ) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        use core::arch::aarch64::*;

        unsafe {
          i8x16 { neon: vcombine_s8(vqmovn_s16(v.a.neon), vqmovn_s16(v.b.neon)) }
        }
      } else if #[cfg(target_feature="simd128")] {
        use core::arch::wasm32::*;

        i8x16 { simd: i8x16_narrow_i16x8(v.a.simd, v.b.simd) }
      } else {
        fn clamp(a : i16) -> i8 {
            if a < i8::MIN as i16 {
              i8::MIN
            }
            else if a > i8::MAX as i16 {
              i8::MAX
            } else {
                a as i8
            }
        }

        i8x16::new([
          clamp(v.as_array()[0]),
          clamp(v.as_array()[1]),
          clamp(v.as_array()[2]),
          clamp(v.as_array()[3]),
          clamp(v.as_array()[4]),
          clamp(v.as_array()[5]),
          clamp(v.as_array()[6]),
          clamp(v.as_array()[7]),
          clamp(v.as_array()[8]),
          clamp(v.as_array()[9]),
          clamp(v.as_array()[10]),
          clamp(v.as_array()[11]),
          clamp(v.as_array()[12]),
          clamp(v.as_array()[13]),
          clamp(v.as_array()[14]),
          clamp(v.as_array()[15]),
        ])
      }
    }
  }

  /// converts `i16` to `i8`, truncating the upper bits if they are set
  #[inline]
  #[must_use]
  pub fn from_i16x16_truncate(v: i16x16) -> i8x16 {
    pick! {
      if #[cfg(target_feature="avx2")] {
        let a = v.avx2.bitand(set_splat_i16_m256i(0xff));
        i8x16 { sse: pack_i16_to_u8_m128i( extract_m128i_from_m256i::<0>(a), extract_m128i_from_m256i::<1>(a))  }
      } else if #[cfg(target_feature="sse2")] {
        let mask = set_splat_i16_m128i(0xff);
        i8x16 { sse: pack_i16_to_u8_m128i( v.a.sse.bitand(mask), v.b.sse.bitand(mask) ) }
      } else {
        // no super good intrinsics on other platforms... plain old codegen does a reasonable job
        i8x16::new([
          v.as_array()[0] as i8,
          v.as_array()[1] as i8,
          v.as_array()[2] as i8,
          v.as_array()[3] as i8,
          v.as_array()[4] as i8,
          v.as_array()[5] as i8,
          v.as_array()[6] as i8,
          v.as_array()[7] as i8,
          v.as_array()[8] as i8,
          v.as_array()[9] as i8,
          v.as_array()[10] as i8,
          v.as_array()[11] as i8,
          v.as_array()[12] as i8,
          v.as_array()[13] as i8,
          v.as_array()[14] as i8,
          v.as_array()[15] as i8,
        ])
      }
    }
  }

  /// Bitwise selection.
  ///
  /// For each bit of `self`:
  ///
  /// - If the bit is one, return the corresponding bit of `if_one`
  /// - If the bit is zero, return the corresponding bit of `if_zero`
  ///
  /// If you know `self` is a mask, meaning each lane is either all zeros or all
  /// ones, consider using [`select`] which is faster.
  ///
  /// [`select`]: Self::select
  #[inline]
  #[must_use]
  pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self {
          sse: bitor_m128i(
            bitand_m128i(if_one.sse, self.sse),
            bitandnot_m128i(self.sse, if_zero.sse),
          ),
        }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_bitselect(if_one.simd, if_zero.simd, self.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vbslq_s8(vreinterpretq_u8_s8(self.neon), if_one.neon, if_zero.neon) }}
      } else {
        generic_bit_blend(self, if_one, if_zero)
      }
    }
  }

  /// Lanewise selection.
  ///
  /// For each lane of `self`:
  ///
  /// - If all bits are one, return the corresponding lane of `if_true`
  /// - If all bits are zero, return the corresponding lane of `if_false`
  ///
  /// This function assumes `self` is a mask, meaning each lane is either all
  /// zeros or all ones. For bitwise selection use [`bitselect`].
  ///
  /// [`bitselect`]: Self::bitselect
  #[inline]
  #[must_use]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_i8_m128i(if_false.sse, if_true.sse, self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_bitselect(if_true.simd, if_false.simd, self.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vbslq_s8(vreinterpretq_u8_s8(self.neon), if_true.neon, if_false.neon) }}
      } else {
        generic_bit_blend(self, if_true, if_false)
      }
    }
  }

  /// Returns true for each positive element and false if it is zero or
  /// negative.
  #[inline]
  #[must_use]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        Self { neon: unsafe { vreinterpretq_s8_u8(vcgtzq_s8(self.neon)) } }
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  /// Returns true for each negative element and false if it is zero or
  /// positive.
  #[inline]
  #[must_use]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        Self { neon: unsafe { vreinterpretq_s8_u8(vcltzq_s8(self.neon)) } }
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> i8 {
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
        let sum = add_i8_m128i(self.sse, rhs);
        let rhs = shuffle_av_i8z_all_m128i(sum, m128i::from(SHUFFLE_2));
        let sum = add_i8_m128i(sum, rhs);
        let rhs = shuffle_av_i8z_all_m128i(sum, m128i::from(SHUFFLE_3));
        let sum = add_i8_m128i(sum, rhs);
        let rhs = shuffle_av_i8z_all_m128i(sum, m128i::from(SHUFFLE_4));
        let sum = add_i8_m128i(sum, rhs);
        get_i32_from_m128i_s(sum) as i8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.simd, self.simd);
        let sum = i8x16_add(self.simd, rhs);
        let rhs = i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
        let sum = i8x16_add(sum, rhs);
        let rhs = i8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
        let sum = i8x16_add(sum, rhs);
        let rhs = i8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(sum, sum);
        let sum = i8x16_add(sum, rhs);
        i8x16_extract_lane::<0>(sum)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `int8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_s8(self.neon, core::mem::transmute(SHUFFLE_1));
          let sum = vaddq_s8(self.neon, rhs);
          let rhs = vqtbl1q_s8(sum, core::mem::transmute(SHUFFLE_2));
          let sum = vaddq_s8(sum, rhs);
          let rhs = vqtbl1q_s8(sum, core::mem::transmute(SHUFFLE_3));
          let sum = vaddq_s8(sum, rhs);
          let rhs = vqtbl1q_s8(sum, core::mem::transmute(SHUFFLE_4));
          let sum = vaddq_s8(sum, rhs);
          vgetq_lane_s8(sum, 0)
        }
      } else {
        let array: [i8; 16] = cast(self);
        array.into_iter().reduce(i8::wrapping_add).unwrap()
      }
    }
  }

  /// Reducing multiply. Returns the product of the elements of the vector.
  #[inline]
  #[must_use]
  pub fn reduce_mul(self) -> i8 {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        const HIGH_64: [u8; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0];
        const HIGH_32: [u8; 16] = [4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0];
        const HIGH_16: [u8; 16] = [2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        const HIGH_8: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        unsafe {
          // Use `transmute` instead of `cast` because `int8x16_t` does not
          // implement `bytemuck::Pod`.
          let high_64 = vqtbl1q_s8(self.neon, core::mem::transmute(HIGH_64));
          let reduce_64 = vmulq_s8(self.neon, high_64);
          let high_32 = vqtbl1q_s8(reduce_64, core::mem::transmute(HIGH_32));
          let reduce_32 = vmulq_s8(reduce_64, high_32);
          let high_16 = vqtbl1q_s8(reduce_32, core::mem::transmute(HIGH_16));
          let reduce_16 = vmulq_s8(reduce_32, high_16);
          let high_8 = vqtbl1q_s8(reduce_16, core::mem::transmute(HIGH_8));
          let reduce_8 = vmulq_s8(reduce_16, high_8);
          vgetq_lane_s8::<0>(reduce_8)
        }
      } else {
        self.to_array().into_iter().reduce(i8::wrapping_mul).unwrap()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_max(self) -> i8 {
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
      if #[cfg(all(target_feature="ssse3", target_feature="sse4.1"))] {
        let rhs = shuffle_av_i8z_all_m128i(self.sse, m128i::from(SHUFFLE_1));
        let max = max_i8_m128i(self.sse, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_2));
        let max = max_i8_m128i(max, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_3));
        let max = max_i8_m128i(max, rhs);
        let rhs = shuffle_av_i8z_all_m128i(max, m128i::from(SHUFFLE_4));
        let max = max_i8_m128i(max, rhs);
        get_i32_from_m128i_s(max) as i8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.simd, self.simd);
        let max = i8x16_max(self.simd, rhs);
        let rhs = i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = i8x16_max(max, rhs);
        let rhs = i8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = i8x16_max(max, rhs);
        let rhs = i8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(max, max);
        let max = i8x16_max(max, rhs);
        i8x16_extract_lane::<0>(max)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `int8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_s8(self.neon, core::mem::transmute(SHUFFLE_1));
          let max = vmaxq_s8(self.neon, rhs);
          let rhs = vqtbl1q_s8(max, core::mem::transmute(SHUFFLE_2));
          let max = vmaxq_s8(max, rhs);
          let rhs = vqtbl1q_s8(max, core::mem::transmute(SHUFFLE_3));
          let max = vmaxq_s8(max, rhs);
          let rhs = vqtbl1q_s8(max, core::mem::transmute(SHUFFLE_4));
          let max = vmaxq_s8(max, rhs);
          vgetq_lane_s8(max, 0)
        }
      } else {
        let array: [i8; 16] = cast(self);
        array.into_iter().reduce(i8::max).unwrap()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn reduce_min(self) -> i8 {
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
      if #[cfg(all(target_feature="ssse3", target_feature="sse4.1"))] {
        let rhs = shuffle_av_i8z_all_m128i(self.sse, m128i::from(SHUFFLE_1));
        let min = min_i8_m128i(self.sse, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_2));
        let min = min_i8_m128i(min, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_3));
        let min = min_i8_m128i(min, rhs);
        let rhs = shuffle_av_i8z_all_m128i(min, m128i::from(SHUFFLE_4));
        let min = min_i8_m128i(min, rhs);
        get_i32_from_m128i_s(min) as i8
      } else if #[cfg(target_feature="simd128")] {
        let rhs = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(self.simd, self.simd);
        let min = i8x16_min(self.simd, rhs);
        let rhs = i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = i8x16_min(min, rhs);
        let rhs = i8x16_shuffle::<2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = i8x16_min(min, rhs);
        let rhs = i8x16_shuffle::<1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(min, min);
        let min = i8x16_min(min, rhs);
        i8x16_extract_lane::<0>(min)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {
          // Use `transmute` instead of `cast` because `int8x16_t` does not
          // implement `bytemuck::Pod`.
          let rhs = vqtbl1q_s8(self.neon, core::mem::transmute(SHUFFLE_1));
          let min = vminq_s8(self.neon, rhs);
          let rhs = vqtbl1q_s8(min, core::mem::transmute(SHUFFLE_2));
          let min = vminq_s8(min, rhs);
          let rhs = vqtbl1q_s8(min, core::mem::transmute(SHUFFLE_3));
          let min = vminq_s8(min, rhs);
          let rhs = vqtbl1q_s8(min, core::mem::transmute(SHUFFLE_4));
          let min = vminq_s8(min, rhs);
          vgetq_lane_s8(min, 0)
        }
      } else {
        let array: [i8; 16] = cast(self);
        array.into_iter().reduce(i8::min).unwrap()
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vabsq_s8(self.neon) }}
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
  pub fn unsigned_abs(self) -> u8x16 {
    pick! {
      if #[cfg(target_feature="ssse3")] {
        u8x16 { sse: abs_i8_m128i(self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        u8x16 { simd: i8x16_abs(self.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { u8x16 { neon: vreinterpretq_u8_s8(vabsq_s8(self.neon)) }}
      } else {
        let arr: [i8; 16] = cast(self);
        cast(
          [
            arr[0].unsigned_abs(),
            arr[1].unsigned_abs(),
            arr[2].unsigned_abs(),
            arr[3].unsigned_abs(),
            arr[4].unsigned_abs(),
            arr[5].unsigned_abs(),
            arr[6].unsigned_abs(),
            arr[7].unsigned_abs(),
            arr[8].unsigned_abs(),
            arr[9].unsigned_abs(),
            arr[10].unsigned_abs(),
            arr[11].unsigned_abs(),
            arr[12].unsigned_abs(),
            arr[13].unsigned_abs(),
            arr[14].unsigned_abs(),
            arr[15].unsigned_abs(),
            ])
      }
    }
  }

  signed_fn_signum!();

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: max_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_max(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vmaxq_s8(self.neon, rhs.neon) }}
      } else {
        self.simd_lt(rhs).select(rhs, self)
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
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vminq_s8(self.neon, rhs.neon) }}
      } else {
        self.simd_lt(rhs).select(self, rhs)
      }
    }
  }

  integer_fn_clamp!();

  #[inline]
  #[must_use]
  pub fn from_slice_unaligned(input: &[i8]) -> Self {
    assert!(input.len() >= 16);

    pick! {
      if #[cfg(target_feature="sse2")] {
        unsafe { Self { sse: load_unaligned_m128i( &*(input.as_ptr() as * const [u8;16]) ) } }
      } else if #[cfg(target_feature="simd128")] {
        unsafe { Self { simd: v128_load(input.as_ptr() as *const v128 ) } }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vld1q_s8( input.as_ptr() as *const i8 ) } }
      } else {
        // 2018 edition doesn't have try_into
        unsafe { Self::new( *(input.as_ptr() as * const [i8;16]) ) }
      }
    }
  }

  #[inline]
  #[must_use]
  #[doc(alias("movemask", "move_mask"))]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="sse2")] {
        move_mask_i8_m128i(self.sse) as u32
      } else if #[cfg(target_feature="simd128")] {
        i8x16_bitmask(self.simd) as u32
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe
        {
          // set all to 1 if top bit is set, else 0
          let masked = vcltq_s8(self.neon, vdupq_n_s8(0));

          // select the right bit out of each lane
          let selectbit : uint8x16_t = core::mem::transmute([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]);
          let out = vandq_u8(masked, selectbit);

          // interleave the lanes so that a 16-bit sum accumulates the bits in the right order
          let table : uint8x16_t = core::mem::transmute([0u8, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]);
          let r = vqtbl1q_u8(out, table);

          // horizontally add the 16-bit lanes
          vaddvq_u16(vreinterpretq_u16_u8(r)) as u32
        }
       } else {
        ((self.arr[0] < 0) as u32) |
        ((self.arr[1] < 0) as u32) << 1 |
        ((self.arr[2] < 0) as u32) << 2 |
        ((self.arr[3] < 0) as u32) << 3 |
        ((self.arr[4] < 0) as u32) << 4 |
        ((self.arr[5] < 0) as u32) << 5 |
        ((self.arr[6] < 0) as u32) << 6 |
        ((self.arr[7] < 0) as u32) << 7 |
        ((self.arr[8] < 0) as u32) << 8 |
        ((self.arr[9] < 0) as u32) << 9 |
        ((self.arr[10] < 0) as u32) << 10 |
        ((self.arr[11] < 0) as u32) << 11 |
        ((self.arr[12] < 0) as u32) << 12 |
        ((self.arr[13] < 0) as u32) << 13 |
        ((self.arr[14] < 0) as u32) << 14 |
        ((self.arr[15] < 0) as u32) << 15
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="sse2")] {
        move_mask_i8_m128i(self.sse) != 0
      } else if #[cfg(target_feature="simd128")] {
        u8x16_bitmask(self.simd) != 0
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        unsafe {
          vminvq_s8(self.neon) < 0
        }
      } else {
        let v : [u64;2] = cast(self);
        ((v[0] | v[1]) & 0x8080808080808080) != 0
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="sse2")] {
        move_mask_i8_m128i(self.sse) == 0b1111_1111_1111_1111
      } else if #[cfg(target_feature="simd128")] {
        u8x16_bitmask(self.simd) == 0b1111_1111_1111_1111
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        unsafe {
          vmaxvq_s8(self.neon) < 0
        }
      } else {
        let v : [u64;2] = cast(self);
        (v[0] & v[1] & 0x8080808080808080) == 0x8080808080808080
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
    pick! {
      if #[cfg(target_feature="ssse3")] {
        Self { sse: shuffle_av_i8z_all_m128i(self.sse, add_saturating_u8_m128i(rhs.sse, set_splat_i8_m128i(0x70))) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_swizzle(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        unsafe { Self { neon: vqtbl1q_s8(self.neon, vreinterpretq_u8_s8(rhs.neon)) } }
      } else {
        let idxs = rhs.to_array();
        let arr = self.to_array();
        let mut out = [0i8;16];
        for i in 0..16 {
          let idx = idxs[i] as usize;
          if idx >= 16 {
            out[i] = 0;
          } else {
            out[i] = arr[idx];
          }
        }
        Self::new(out)
      }
    }
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
  pub fn swizzle_relaxed(self, rhs: i8x16) -> i8x16 {
    pick! {
      if #[cfg(target_feature="ssse3")] {
        Self { sse: shuffle_av_i8z_all_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_swizzle(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        unsafe { Self { neon: vqtbl1q_s8(self.neon, vreinterpretq_u8_s8(rhs.neon)) } }
      } else {
        let idxs = rhs.to_array();
        let arr = self.to_array();
        let mut out = [0i8;16];
        for i in 0..16 {
          let idx = idxs[i] as usize;
          if idx >= 16 {
            out[i] = 0;
          } else {
            out[i] = arr[idx];
          }
        }
        Self::new(out)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  #[inline]
  #[must_use]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_saturating_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_add_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vqaddq_s8(self.neon, rhs.neon) }}
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
        Self { sse: sub_saturating_i8_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i8x16_sub_sat(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqsubq_s8(self.neon, rhs.neon) } }
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
          let low_wide_mul = vreinterpretq_s8_s16(
            vmull_s8(vget_low_s8(self.neon), vget_low_s8(rhs.neon)),
          );
          let high_wide_mul = vreinterpretq_s8_s16(
            vmull_s8(vget_high_s8(self.neon), vget_high_s8(rhs.neon)),
          );
          let low_high = vuzpq_s8(low_wide_mul, high_wide_mul);
          let low = Self { neon: low_high.0 };
          let high = Self { neon: low_high.1 };

          let no_overflow = high.simd_eq(low.is_negative());
          let limit = Self::MAX ^ (self ^ rhs).is_negative();
          no_overflow.select(low, limit)
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

  signed_fn_overflowing_add_sub!();

  /// Returns `self * rhs` and whether an overflow occured.
  ///
  /// Returns a tuple with:
  ///
  /// - The multiplication (returns the wrapped value if an overflow occured)
  /// - A mask indicating whether an overflow occured
  #[inline]
  #[must_use]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          let low_wide_mul = vreinterpretq_s8_s16(
            vmull_s8(vget_low_s8(self.neon), vget_low_s8(rhs.neon)),
          );
          let high_wide_mul = vreinterpretq_s8_s16(
            vmull_s8(vget_high_s8(self.neon), vget_high_s8(rhs.neon)),
          );
          let low_high = vuzpq_s8(low_wide_mul, high_wide_mul);
          let low = Self { neon: low_high.0 };
          let high = Self { neon: low_high.1 };

          let overflow = high.simd_ne(low.is_negative());

          (low, overflow)
        }
      } else {
        // TODO(perf): This implementation looks quite bad. Is there a better
        // one?

        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        let widening_mul = cast::<[i16; 16], [[i8; 2]; 16]>([
          (self_array[0] as i16).wrapping_mul(rhs_array[0] as i16),
          (self_array[1] as i16).wrapping_mul(rhs_array[1] as i16),
          (self_array[2] as i16).wrapping_mul(rhs_array[2] as i16),
          (self_array[3] as i16).wrapping_mul(rhs_array[3] as i16),
          (self_array[4] as i16).wrapping_mul(rhs_array[4] as i16),
          (self_array[5] as i16).wrapping_mul(rhs_array[5] as i16),
          (self_array[6] as i16).wrapping_mul(rhs_array[6] as i16),
          (self_array[7] as i16).wrapping_mul(rhs_array[7] as i16),
          (self_array[8] as i16).wrapping_mul(rhs_array[8] as i16),
          (self_array[9] as i16).wrapping_mul(rhs_array[9] as i16),
          (self_array[10] as i16).wrapping_mul(rhs_array[10] as i16),
          (self_array[11] as i16).wrapping_mul(rhs_array[11] as i16),
          (self_array[12] as i16).wrapping_mul(rhs_array[12] as i16),
          (self_array[13] as i16).wrapping_mul(rhs_array[13] as i16),
          (self_array[14] as i16).wrapping_mul(rhs_array[14] as i16),
          (self_array[15] as i16).wrapping_mul(rhs_array[15] as i16),
        ]);
        let low = Self::new([
          widening_mul[0][0],
          widening_mul[1][0],
          widening_mul[2][0],
          widening_mul[3][0],
          widening_mul[4][0],
          widening_mul[5][0],
          widening_mul[6][0],
          widening_mul[7][0],
          widening_mul[8][0],
          widening_mul[9][0],
          widening_mul[10][0],
          widening_mul[11][0],
          widening_mul[12][0],
          widening_mul[13][0],
          widening_mul[14][0],
          widening_mul[15][0],
        ]);
        let high = Self::new([
          widening_mul[0][1],
          widening_mul[1][1],
          widening_mul[2][1],
          widening_mul[3][1],
          widening_mul[4][1],
          widening_mul[5][1],
          widening_mul[6][1],
          widening_mul[7][1],
          widening_mul[8][1],
          widening_mul[9][1],
          widening_mul[10][1],
          widening_mul[11][1],
          widening_mul[12][1],
          widening_mul[13][1],
          widening_mul[14][1],
          widening_mul[15][1],
        ]);

        let overflow = high.simd_ne(low.is_negative());

        (low, overflow)
      }
    }
  }

  signed_fn_overflowing_div_rem!();

  /// Transpose matrix of 16x16 `i8` matrix. Currently not accelerated.
  #[must_use]
  #[inline]
  pub fn transpose(data: [i8x16; 16]) -> [i8x16; 16] {
    // Can this be optimized?

    #[inline(always)]
    fn transpose_column(data: &[i8x16; 16], index: usize) -> i8x16 {
      i8x16::new([
        data[0].as_array()[index],
        data[1].as_array()[index],
        data[2].as_array()[index],
        data[3].as_array()[index],
        data[4].as_array()[index],
        data[5].as_array()[index],
        data[6].as_array()[index],
        data[7].as_array()[index],
        data[8].as_array()[index],
        data[9].as_array()[index],
        data[10].as_array()[index],
        data[11].as_array()[index],
        data[12].as_array()[index],
        data[13].as_array()[index],
        data[14].as_array()[index],
        data[15].as_array()[index],
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
      transpose_column(&data, 8),
      transpose_column(&data, 9),
      transpose_column(&data, 10),
      transpose_column(&data, 11),
      transpose_column(&data, 12),
      transpose_column(&data, 13),
      transpose_column(&data, 14),
      transpose_column(&data, 15),
    ]
  }

  fn_blend!();
}
