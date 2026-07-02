use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u64x2 { pub(crate) sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct u64x2 { pub(crate) simd: v128 }

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
    pub struct u64x2 { pub(crate) neon : uint64x2_t }

    impl Default for u64x2 {
      #[inline]
      fn default() -> Self {
        unsafe { Self { neon: vdupq_n_u64(0)} }
      }
    }

    impl PartialEq for u64x2 {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        unsafe {
          vgetq_lane_u64(self.neon,0) == vgetq_lane_u64(other.neon,0) &&
          vgetq_lane_u64(self.neon,1) == vgetq_lane_u64(other.neon,1)
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

impl_simd! {
  T = u64,
  N = 2,
  Simd = u64x2,

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
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
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_eq(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_ne(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] != r[0] { -1_i64 } else { 0 },
          if s[1] != r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    // lt is just gt the other way around
    rhs.simd_gt(self)
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        // no unsigned gt so inverting the high bit will get the correct result
        let highbit = u64x2::splat(1 << 63);
        Self { sse: cmp_gt_mask_i64_m128i((self ^ highbit).sse, (rhs ^ highbit).sse) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vcgtq_u64(self.neon, rhs.neon) }}
      } else {
        // u64x2_gt on WASM is not a thing. https://github.com/WebAssembly/simd/pull/414
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] > r[0] { u64::MAX } else { 0 },
          if s[1] > r[1] { u64::MAX } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_gt(rhs)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_gt(rhs)
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] <= r[0] { -1_i64 } else { 0 },
          if s[1] <= r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_lt(rhs)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_lt(rhs)
      } else {
        let s: [u64;2] = cast(self);
        let r: [u64;2] = cast(rhs);
        cast([
          if s[0] >= r[0] { -1_i64 } else { 0 },
          if s[1] >= r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
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
        unsafe {Self { neon: vbslq_u64(self.neon, if_one.neon, if_zero.neon) }}
      } else {
        generic_bit_blend(self, if_one, if_zero)
      }
    }
  }

  #[inline]
  pub fn select(self, if_true: Self, if_false: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_i8_m128i(if_false.sse, if_true.sse, self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: v128_bitselect(if_true.simd, if_false.simd, self.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vbslq_u64(self.neon, if_true.neon, if_false.neon) }}
      } else {
        generic_bit_blend(self, if_true, if_false)
      }
    }
  }

  #[inline]
  pub fn to_bitmask(self) -> u32 {
    i64x2::to_bitmask(cast(self))
  }

  #[inline]
  pub fn any(self) -> bool {
    i64x2::any(cast(self))
  }

  #[inline]
  pub fn all(self) -> bool {
    i64x2::all(cast(self))
  }

  /// Transpose matrix of 2x2 `u64` matrix.
  #[inline]
  pub fn transpose(data: [u64x2; 2]) -> [u64x2; 2] {
    cast(i64x2::transpose(cast(data)))
  }
}

impl_simd_uint! {
  T = u64,
  N = 2,
  Simd = u64x2,
  [0, 1],

  #[inline]
  fn not(self) -> Self::Output {
    self ^ cast::<u128, u64x2>(u128::MAX)
  }

  #[inline]
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

  #[inline]
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

  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    //we should try to implement this on sse2
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

  #[inline]
  fn shl(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self { sse: shl_each_u64_m128i(self.sse, shift_by.sse) }
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // mask the shift count to 63 to have same behavior on all platforms
          let shift_by = vreinterpretq_s64_u64(vandq_u64(rhs.neon, vmovq_n_u64(63)));
          Self { neon: vshlq_u64(self.neon, shift_by) }
        }
      } else {
        let arr: [u64; 2] = cast(self);
        let rhs: [u64; 2] = cast(rhs);
        cast([
          arr[0].wrapping_shl(rhs[0] as u32),
          arr[1].wrapping_shl(rhs[1] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shl(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self { sse: shl_all_u64_m128i(self.sse, shift) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_shl(self.simd, rhs) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe {Self { neon: vshlq_u64(self.neon, vmovq_n_s64(rhs as i64 & 63)) }}
      } else {
        Self { arr: [
          self.arr[0].wrapping_shl(rhs),
          self.arr[1].wrapping_shl(rhs),
        ]}
      }
    }
  }

  #[inline]
  fn shr(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // mask the shift count to 63 to have same behavior on all platforms
        let shift_by = rhs & Self::splat(63);
        Self { sse: shr_each_u64_m128i(self.sse, shift_by.sse) }
      } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // mask the shift count to 63 to have same behavior on all platforms
          // no right shift, have to pass negative value to left shift on neon
          let shift_by = vnegq_s64(vreinterpretq_s64_u64(vandq_u64(rhs.neon, vmovq_n_u64(63))));
          Self { neon: vshlq_u64(self.neon, shift_by) }
        }
      } else {
        let arr: [u64; 2] = cast(self);
        let rhs: [u64; 2] = cast(rhs);
        cast([
          arr[0].wrapping_shr(rhs[0] as u32),
          arr[1].wrapping_shr(rhs[1] as u32),
        ])
      }
    }
  }

  #[inline]
  fn shr(self, rhs: u32) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        let shift = cast([rhs as u64 & 63, 0]);
        Self { sse: shr_all_u64_m128i(self.sse, shift) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: u64x2_shr(self.simd, rhs) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe {Self { neon: vshlq_u64(self.neon, vmovq_n_s64(-(rhs as i64 & 63))) }}
      } else {
        Self { arr: [
          self.arr[0].wrapping_shr(rhs),
          self.arr[1].wrapping_shr(rhs),
        ]}
      }
    }
  }

  #[inline]
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

  #[inline]
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

  #[inline]
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

  #[inline]
  pub fn max(self, rhs: Self) -> Self {
    self.simd_gt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn min(self, rhs: Self) -> Self {
    self.simd_lt(rhs).select(self, rhs)
  }

  #[inline]
  pub fn reduce_add(self) -> u64 {
    cast(i64x2::reduce_add(cast(self)))
  }


  #[inline]
  pub fn reduce_mul(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].wrapping_mul(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.neon, 0).wrapping_mul(vgetq_lane_u64(self.neon, 1)) }
      } else {
        self.arr[0].wrapping_mul(self.arr[1])
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].max(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.neon, 0).max(vgetq_lane_u64(self.neon, 1)) }
      } else {
        self.arr[0].max(self.arr[1])
      }
    }
  }

  #[inline]
  pub fn reduce_min(self) -> u64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [u64; 2] = cast(self);
        array[0].min(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_u64(self.neon, 0).min(vgetq_lane_u64(self.neon, 1)) }
      } else {
        self.arr[0].min(self.arr[1])
      }
    }
  }

  #[inline]
  pub fn saturating_add(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let result = self + rhs;
        let overflow = result.simd_lt(self);
        // Return `MAX` (all bits set) if overflow occurs.
        result | overflow
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqaddq_u64(self.neon, rhs.neon) } }
      } else {
        Self {
          arr: [
            self.arr[0].saturating_add(rhs.arr[0]),
            self.arr[1].saturating_add(rhs.arr[1]),
          ],
        }
      }
    }
  }

  #[inline]
  pub fn saturating_sub(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let result = self - rhs;
        let no_overflow = result.simd_le(self);
        // Return `0` (no bits set) if overflow occurs.
        result & no_overflow
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqsubq_u64(self.neon, rhs.neon) } }
      } else {
        Self {
          arr: [
            self.arr[0].saturating_sub(rhs.arr[0]),
            self.arr[1].saturating_sub(rhs.arr[1]),
          ],
        }
      }
    }
  }

  #[inline]
  pub fn saturating_mul(self, rhs: Self) -> Self {
    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    Self::new([
      self_array[0].saturating_mul(rhs_array[0]),
      self_array[1].saturating_mul(rhs_array[1]),
    ])
  }

  #[inline]
  pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
    // TODO(perf): This implementation looks quite bad. Is there a better
    // one?

    let self_array = self.to_array();
    let rhs_array = rhs.to_array();

    let result = [
      self_array[0].overflowing_mul(rhs_array[0]),
      self_array[1].overflowing_mul(rhs_array[1]),
    ];
    (
      Self::new([result[0].0, result[1].0]),
      Self::new([-(result[0].1 as i64) as u64, -(result[1].1 as i64) as u64]),
    )
  }
}

impl AlignTo for u64x2 {
  type Elem = u64;
}

impl u64x2 {
  #[inline]
  #[must_use]
  pub fn mul_keep_high(self, rhs: Self) -> Self {
    let arr1: [u64; 2] = cast(self);
    let arr2: [u64; 2] = cast(rhs);
    cast([
      ((arr1[0] as u128 * arr2[0] as u128) >> 64) as u64,
      ((arr1[1] as u128 * arr2[1] as u128) >> 64) as u64,
    ])
  }
}
