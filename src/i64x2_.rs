use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i64x2 { pub(crate) sse: m128i }
  } else if #[cfg(target_feature="simd128")] {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct i64x2 { pub(crate) simd: v128 }

    impl Default for i64x2 {
      fn default() -> Self {
        Self::splat(0)
      }
    }

    impl PartialEq for i64x2 {
      fn eq(&self, other: &Self) -> bool {
        u64x2_all_true(i64x2_eq(self.simd, other.simd))
      }
    }

    impl Eq for i64x2 { }
  } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
    use core::arch::aarch64::*;
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct i64x2 { pub(crate) neon : int64x2_t }

    impl Default for i64x2 {
      #[inline]
      fn default() -> Self {
        unsafe { Self { neon: vdupq_n_s64(0)} }
      }
    }

    impl PartialEq for i64x2 {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        unsafe {
          vgetq_lane_s64(self.neon,0) == vgetq_lane_s64(other.neon,0) && vgetq_lane_s64(self.neon,1) == vgetq_lane_s64(other.neon,1)
        }
      }
    }

    impl Eq for i64x2 { }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i64x2 { arr: [i64;2] }
  }
}

impl_simd! {
  unsafe {
    T = i64,
    N = 2,
    Simd = i64x2,
  }

  #[inline]
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: cmp_eq_mask_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_eq(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s64_u64(vceqq_s64(self.neon, rhs.neon)) }}
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
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
        Self { simd: i64x2_ne(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_eq(rhs)
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] != r[0] { -1_i64 } else { 0 },
          if s[1] != r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        // only has gt, so flip arguments around to get lt
        Self { sse: cmp_gt_mask_i64_m128i( rhs.sse, self.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_lt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s64_u64(vcltq_s64(self.neon, rhs.neon)) }}
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] < r[0] { -1_i64 } else { 0 },
          if s[1] < r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        Self { sse: cmp_gt_mask_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_gt(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vreinterpretq_s64_u64(vcgtq_s64(self.neon, rhs.neon)) }}
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] > r[0] { -1_i64 } else { 0 },
          if s[1] > r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }

  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        !self.simd_gt(rhs)
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_le(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_gt(rhs)
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
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
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_ge(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        !self.simd_lt(rhs)
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
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
        unsafe {Self { neon: vbslq_s64(vreinterpretq_u64_s64(self.neon), if_one.neon, if_zero.neon) }}
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
        unsafe {Self { neon: vbslq_s64(vreinterpretq_u64_s64(self.neon), if_true.neon, if_false.neon) }}
      } else {
        generic_bit_blend(self, if_true, if_false)
      }
    }
  }

  /// returns the bit mask for each high bit set in the vector with the lowest
  /// lane being the lowest bit
  #[inline]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="sse")] {
        // use f64 move_mask since it is the same size as i64
        move_mask_m128d(cast(self.sse)) as u32
      } else if #[cfg(target_feature="simd128")] {
        i64x2_bitmask(self.simd) as u32
      } else {
        // nothing amazingly efficient for neon
        let arr: [u64; 2] = cast(self);
        (arr[0] >> 63 | ((arr[1] >> 62) & 2)) as u32
      }
    }
  }

  /// true if any high bits are set for any value in the vector
  #[inline]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="sse")] {
        // use f64 move_mask since it is the same size as i64
        move_mask_m128d(cast(self.sse)) != 0
      } else if #[cfg(target_feature="simd128")] {
        i64x2_bitmask(self.simd) != 0
      } else {
        let v : [u64;2] = cast(self);
        ((v[0] | v[1]) & 0x8000000000000000) != 0
      }
    }
  }

  /// true if all high bits are set for every value in the vector
  #[inline]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx2")] {
        // use f64 move_mask since it is the same size as i64
        move_mask_m128d(cast(self.sse)) == 0b11
      }  else if #[cfg(target_feature="simd128")] {
        i64x2_bitmask(self.simd) == 0b11
      } else {
        let v : [u64;2] = cast(self);
        ((v[0] & v[1]) & 0x8000000000000000) == 0x8000000000000000
      }
    }
  }

  ///
  /// This function is accelerated on multiple target architectures.
  #[inline]
  pub fn transpose(data: [i64x2; 2]) -> [i64x2; 2] {
    pick! {
      if #[cfg(any(
        target_feature="sse2",
        all(target_feature="neon",target_arch="aarch64"),
        target_feature="simd128",
      ))] {
        [data[0].unpack_lo(data[1]), data[0].unpack_hi(data[1])]
      } else {
        let [x, y, z, w]: [i64; 4] = cast(data);
        cast([x, z, y, w])
      }
    }
  }
}

impl_simd_int! {
  unsafe {
    T = i64,
    N = 2,
    Simd = i64x2,
    UnsignedSimd = u64x2,
    [0, 1],
  }

  #[inline]
  fn not(self) -> Self::Output {
    self ^ cast::<u128, i64x2>(u128::MAX)
  }

  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i64_m128i(self.sse, rhs.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_add(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vaddq_s64(self.neon, rhs.neon) } }
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
        Self { simd: i64x2_sub(self.simd, rhs.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vsubq_s64(self.neon, rhs.neon) } }
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
        Self { simd: i64x2_mul(self.simd, rhs.simd) }
      } else {
        let arr1: [i64; 2] = cast(self);
        let arr2: [i64; 2] = cast(rhs);
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
          let shift_by = vandq_s64(rhs.neon, vmovq_n_s64(63));
          Self { neon: vshlq_s64(self.neon, shift_by) }
        }
      } else {
        let arr: [i64; 2] = cast(self);
        let rhs: [i64; 2] = cast(rhs);
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
        Self { simd: i64x2_shl(self.simd, rhs) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        // Use `rhs % 64` to perform wrapping shift and not unbounded shift.
        #[expect(clippy::suspicious_arithmetic_impl)]
        unsafe {Self { neon: vshlq_s64(self.neon, vmovq_n_s64(rhs as i64 & 63)) }}
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
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        unsafe {
          // mask the shift count to 63 to have same behavior on all platforms
          // no right shift, have to pass negative value to left shift on neon
          let shift_by = vnegq_s64(vandq_s64(rhs.neon, vmovq_n_s64(63)));
          Self { neon: vshlq_s64(self.neon, shift_by) }
        }
      } else {
        let arr: [i64; 2] = cast(self);
        let rhs: [i64; 2] = cast(rhs);
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
      if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_shr(self.simd, rhs) }
      } else {
        let arr: [i64; 2] = cast(self);
        cast([
          arr[0].wrapping_shr(rhs),
          arr[1].wrapping_shr(rhs),
        ])
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
        unsafe {Self { neon: vandq_s64(self.neon, rhs.neon) }}
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
        unsafe {Self { neon: vorrq_s64(self.neon, rhs.neon) }}
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
        unsafe {Self { neon: veorq_s64(self.neon, rhs.neon) }}
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
  pub fn reduce_add(self) -> i64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [i64; 2] = cast(self);
        array[0].wrapping_add(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_s64(self.neon, 0).wrapping_add(vgetq_lane_s64(self.neon, 1)) }
      } else {
        self.arr[0].wrapping_add(self.arr[1])
      }
    }
  }

  #[inline]
  pub fn reduce_mul(self) -> i64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [i64; 2] = cast(self);
        array[0].wrapping_mul(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_s64(self.neon, 0).wrapping_mul(vgetq_lane_s64(self.neon, 1)) }
      } else {
        self.arr[0].wrapping_mul(self.arr[1])
      }
    }
  }

  #[inline]
  pub fn reduce_max(self) -> i64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [i64; 2] = cast(self);
        array[0].max(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_s64(self.neon, 0).max(vgetq_lane_s64(self.neon, 1)) }
      } else {
        self.arr[0].max(self.arr[1])
      }
    }
  }

  #[inline]
  pub fn reduce_min(self) -> i64 {
    pick! {
      if #[cfg(any(target_feature="sse2", target_feature="simd128"))] {
        let array: [i64; 2] = cast(self);
        array[0].min(array[1])
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { vgetq_lane_s64(self.neon, 0).min(vgetq_lane_s64(self.neon, 1)) }
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
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqaddq_s64(self.neon, rhs.neon) } }
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
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();
        let negative = self.is_negative();

        // If overflow occurs return `MAX` if positive or `MIN` if negative.
        overflow.select(Self::MAX ^ negative, result)
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe { Self { neon: vqsubq_s64(self.neon, rhs.neon) } }
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
      Self::new([-(result[0].1 as i64), -(result[1].1 as i64)]),
    )
  }

  #[inline]
  pub fn abs(self) -> Self {
    pick! {
      // x86 doesn't have this builtin
      if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_abs(self.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))]{
        unsafe {Self { neon: vabsq_s64(self.neon) }}
      } else {
        let arr: [i64; 2] = cast(self);
        cast(
          [
            arr[0].wrapping_abs(),
            arr[1].wrapping_abs(),
          ])
      }
    }
  }

  #[inline]
  pub fn is_positive(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        Self { neon: unsafe { vreinterpretq_s64_u64(vcgtzq_s64(self.neon)) } }
      } else {
        self.simd_gt(Self::ZERO)
      }
    }
  }

  #[inline]
  pub fn is_negative(self) -> Self {
    pick! {
      if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        Self { neon: unsafe { vreinterpretq_s64_u64(vcltzq_s64(self.neon)) } }
      } else {
        self.simd_lt(Self::ZERO)
      }
    }
  }
}

/// The following functionality exists only for [`i64x2`], or only for
/// particular types inconsistently.
impl i64x2 {
  #[inline]
  #[must_use]
  pub fn round_float(self) -> f64x2 {
    let arr: [i64; 2] = cast(self);
    cast([arr[0] as f64, arr[1] as f64])
  }

  // Sometimes used for `transpose`.
  #[must_use]
  #[inline]
  #[allow(dead_code)]
  pub(crate) fn unpack_lo(self, b: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: unpack_low_i64_m128i(self.sse, b.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_shuffle::<0, 2>(self.simd, b.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        Self { neon: unsafe { vzip1q_s64(self.neon, b.neon) } }
      } else {
        Self::new([self.as_array()[0], b.as_array()[0]])
      }
    }
  }

  // Sometimes used for `transpose`.
  #[must_use]
  #[inline]
  #[allow(dead_code)]
  pub(crate) fn unpack_hi(self, b: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: unpack_high_i64_m128i(self.sse, b.sse) }
      } else if #[cfg(target_feature="simd128")] {
        Self { simd: i64x2_shuffle::<1, 3>(self.simd, b.simd) }
      } else if #[cfg(all(target_feature="neon",target_arch="aarch64"))] {
        Self { neon: unsafe { vzip2q_s64(self.neon, b.neon) } }
      } else {
        Self::new([self.as_array()[1], b.as_array()[1]])
      }
    }
  }
}
