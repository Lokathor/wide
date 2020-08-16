use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u16x8 { sse: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u16x8 { arr: [u16;8] }
  }
}

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

impl BitAnd for u16x8 {
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

impl<I: Into<u64>> Shl<I> for u16x8 {
  type Output = Self;
  /// Shifts all lanes by the value given.
  #[inline]
  #[must_use]
  fn shl(self, rhs: I) -> Self::Output {
    let u: u64 = rhs.into();
    pick! {
      if #[cfg(target_feature="sse2")] {
        let shift = cast([u, 0]);
        Self { sse: shl_all_u16_m128i(self.sse, shift) }
      } else {
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
}

impl<I: Into<u64>> Shr<I> for u16x8 {
  type Output = Self;
  /// Shifts all lanes by the value given.
  #[inline]
  #[must_use]
  fn shr(self, rhs: I) -> Self::Output {
    let u: u64 = rhs.into();
    pick! {
      if #[cfg(target_feature="sse2")] {
        let shift = cast([u, 0]);
        Self { sse: shr_all_u16_m128i(self.sse, shift) }
      } else {
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
}

impl u16x8 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_i16_m128i(self.sse, rhs.sse) }
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
}
