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
        Self { sse: blend_varying_m128d(f.sse, t.sse, self.sse) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    let non_sign_bits = f64x2::from(f64::from_bits(i64::MAX as u64));
    self & non_sign_bits
  }
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: max_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].max(rhs.arr[0]),
          self.arr[1].max(rhs.arr[1]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: min_m128d(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].min(rhs.arr[0]),
          self.arr[1].min(rhs.arr[1]),
        ]}
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn is_nan(self) -> Self {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: cmp_unord_mask_m128d(self.sse, self.sse) }
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
  pub fn is_finite(self) -> Self {
    let shifted_exp_mask = u64x2::from(0xFFE0000000000000);
    let u: u64x2 = cast(self);
    let shift_u = u << 1_u64;
    let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
    cast(out)
  }
}
