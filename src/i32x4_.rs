use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i32x4 { sse: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i32x4 { arr: [i32;4] }
  }
}

unsafe impl Zeroable for i32x4 {}
unsafe impl Pod for i32x4 {}

impl Add for i32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i32_m128i(self.sse, rhs.sse) }
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

impl Sub for i32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i32_m128i(self.sse, rhs.sse) }
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

impl Mul for i32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: mul_i32_keep_low_m128i(self.sse, rhs.sse) }
      } else {
        let arr1: [i32; 4] = cast(self);
        let arr2: [i32; 4] = cast(rhs);
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

impl BitAnd for i32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitand_m128i(self.sse, rhs.sse) }
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

impl BitOr for i32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitor_m128i(self.sse, rhs.sse) }
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

impl BitXor for i32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128i(self.sse, rhs.sse) }
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

impl<I: Into<u64>> Shl<I> for i32x4 {
  type Output = Self;
  /// Shifts all lanes by the value given.
  #[inline]
  #[must_use]
  fn shl(self, rhs: I) -> Self::Output {
    let u = rhs.into();
    pick! {
      if #[cfg(target_feature="sse2")] {
        let shift = cast([u, 0]);
        Self { sse: shl_all_u32_m128i(self.sse, shift) }
      } else {
        Self { arr: [
          self.arr[0] << u,
          self.arr[1] << u,
          self.arr[2] << u,
          self.arr[3] << u,
        ]}
      }
    }
  }
}

impl<I: Into<u64>> Shr<I> for i32x4 {
  type Output = Self;
  /// Shifts all lanes by the value given.
  #[inline]
  #[must_use]
  fn shr(self, rhs: I) -> Self::Output {
    let u = rhs.into();
    pick! {
      if #[cfg(target_feature="sse2")] {
        let shift = cast([u, 0]);
        Self { sse: shr_all_i32_m128i(self.sse, shift) }
      } else {
        Self { arr: [
          self.arr[0] >> u,
          self.arr[1] >> u,
          self.arr[2] >> u,
          self.arr[3] >> u,
        ]}
      }
    }
  }
}

impl i32x4 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_eq_mask_i32_m128i(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] == rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] == rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] == rhs.arr[3] { -1 } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_gt_mask_i32_m128i(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] > rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] > rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] > rhs.arr[3] { -1 } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: cmp_lt_mask_i32_m128i(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { -1 } else { 0 },
          if self.arr[1] < rhs.arr[1] { -1 } else { 0 },
          if self.arr[2] < rhs.arr[2] { -1 } else { 0 },
          if self.arr[3] < rhs.arr[3] { -1 } else { 0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_i8_m128i(t, f, mask) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
}
