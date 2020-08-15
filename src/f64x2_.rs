use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f64x2 { sse: m128d }
  } else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f64x2 { arr: [f64;2] }
  }
}

unsafe impl Zeroable for f64x2 {}
unsafe impl Pod for f64x2 {}

impl Add for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] + rhs.arr[0],
          self.arr[1] + rhs.arr[1],
        ]}
      }
    }
  }
}

impl Sub for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] - rhs.arr[0],
          self.arr[1] - rhs.arr[1],
        ]}
      }
    }
  }
}

impl Mul for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: mul_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] * rhs.arr[0],
          self.arr[1] * rhs.arr[1],
        ]}
      }
    }
  }
}

impl Div for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn div(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: div_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] / rhs.arr[0],
          self.arr[1] / rhs.arr[1],
        ]}
      }
    }
  }
}

impl BitAnd for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitand_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() & rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() & rhs.arr[1].to_bits()),
        ]}
      }
    }
  }
}

impl BitOr for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitor_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() | rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() | rhs.arr[1].to_bits()),
        ]}
      }
    }
  }
}

impl BitXor for f64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          f64::from_bits(self.arr[0].to_bits() ^ rhs.arr[0].to_bits()),
          f64::from_bits(self.arr[1].to_bits() ^ rhs.arr[1].to_bits()),
        ]}
      }
    }
  }
}

impl f64x2 {
  #[inline]
  #[must_use]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_eq_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] == rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] == rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_ne(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_neq_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] != rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] != rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_ge(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_ge_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] >= rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] >= rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_gt_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] > rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] > rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_le(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_le_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] <= rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] <= rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn cmp_lt(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_lt_mask_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          if self.arr[0] < rhs.arr[0] { f64::from_bits(u64::MAX) } else { 0.0 },
          if self.arr[1] < rhs.arr[1] { f64::from_bits(u64::MAX) } else { 0.0 },
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_m128d(t, f, mask) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
}
