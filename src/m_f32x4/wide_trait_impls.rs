use super::*;

impl Clone for f32x4 {
  #[inline(always)]
  fn clone(&self) -> Self {
    *self
  }
}
impl Copy for f32x4 {}
impl Default for f32x4 {
  #[inline(always)]
  fn default() -> Self {
    Self::zeroed()
  }
}
unsafe impl Zeroable for f32x4 {}
unsafe impl Pod for f32x4 {}

impl From<f32> for f32x4 {
  #[inline]
  fn from(val: f32) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: m128::splat(val) }
    } else {
      Self::new(val,val,val,val)
    }}
  }
}

impl Add for f32x4 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.add(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] + rhs.arr[0],
        self.arr[1] + rhs.arr[1],
        self.arr[2] + rhs.arr[2],
        self.arr[3] + rhs.arr[3],
      ] }
    }}
  }
}

impl Add<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: &Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.add(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] + rhs.arr[0],
        self.arr[1] + rhs.arr[1],
        self.arr[2] + rhs.arr[2],
        self.arr[3] + rhs.arr[3],
      ] }
    }}
  }
}

impl Div for f32x4 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.div(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] / rhs.arr[0],
        self.arr[1] / rhs.arr[1],
        self.arr[2] / rhs.arr[2],
        self.arr[3] / rhs.arr[3],
      ] }
    }}
  }
}

impl Div<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: &Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.div(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] / rhs.arr[0],
        self.arr[1] / rhs.arr[1],
        self.arr[2] / rhs.arr[2],
        self.arr[3] / rhs.arr[3],
      ] }
    }}
  }
}

impl Mul for f32x4 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.mul(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] * rhs.arr[0],
        self.arr[1] * rhs.arr[1],
        self.arr[2] * rhs.arr[2],
        self.arr[3] * rhs.arr[3],
      ] }
    }}
  }
}

impl Mul<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: &Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.mul(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] * rhs.arr[0],
        self.arr[1] * rhs.arr[1],
        self.arr[2] * rhs.arr[2],
        self.arr[3] * rhs.arr[3],
      ] }
    }}
  }
}

impl Sub for f32x4 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.sub(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] - rhs.arr[0],
        self.arr[1] - rhs.arr[1],
        self.arr[2] - rhs.arr[2],
        self.arr[3] - rhs.arr[3],
      ] }
    }}
  }
}

impl Sub<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: &Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.sub(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0] - rhs.arr[0],
        self.arr[1] - rhs.arr[1],
        self.arr[2] - rhs.arr[2],
        self.arr[3] - rhs.arr[3],
      ] }
    }}
  }
}

impl BitAnd for f32x4 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.bitand(rhs.sse) }
    } else {
      Self { arr: [
        f32::from_bits(self.arr[0].to_bits() & rhs.arr[0].to_bits()),
        f32::from_bits(self.arr[1].to_bits() & rhs.arr[1].to_bits()),
        f32::from_bits(self.arr[2].to_bits() & rhs.arr[2].to_bits()),
        f32::from_bits(self.arr[3].to_bits() & rhs.arr[3].to_bits()),
      ] }
    }}
  }
}

impl BitAnd<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: &Self) -> Self {
    self & *rhs
  }
}

impl BitOr for f32x4 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.bitor(rhs.sse) }
    } else {
      Self { arr: [
        f32::from_bits(self.arr[0].to_bits() | rhs.arr[0].to_bits()),
        f32::from_bits(self.arr[1].to_bits() | rhs.arr[1].to_bits()),
        f32::from_bits(self.arr[2].to_bits() | rhs.arr[2].to_bits()),
        f32::from_bits(self.arr[3].to_bits() | rhs.arr[3].to_bits()),
      ] }
    }}
  }
}

impl BitOr<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: &Self) -> Self {
    self | *rhs
  }
}

impl BitXor for f32x4 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.bitxor(rhs.sse) }
    } else {
      Self { arr: [
        f32::from_bits(self.arr[0].to_bits() ^ rhs.arr[0].to_bits()),
        f32::from_bits(self.arr[1].to_bits() ^ rhs.arr[1].to_bits()),
        f32::from_bits(self.arr[2].to_bits() ^ rhs.arr[2].to_bits()),
        f32::from_bits(self.arr[3].to_bits() ^ rhs.arr[3].to_bits()),
      ] }
    }}
  }
}

impl BitXor<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: &Self) -> Self {
    self ^ *rhs
  }
}

impl Not for f32x4 {
  type Output = Self;
  /// Bitwise negation
  #[inline(always)]
  fn not(self) -> Self {
    self ^ Self::ALL_BITS_ACTIVE
  }
}

impl core::iter::Sum for f32x4 {
  #[inline]
  fn sum<I: Iterator<Item = f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(0.0, 0.0, 0.0, 0.0);
    for i in iter {
      total += i;
    }
    total
  }
}
impl<'a> core::iter::Sum<&'a f32x4> for f32x4 {
  #[inline]
  fn sum<I: Iterator<Item = &'a f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(0.0, 0.0, 0.0, 0.0);
    for i in iter {
      total += i;
    }
    total
  }
}

impl core::iter::Product for f32x4 {
  #[inline]
  fn product<I: Iterator<Item = f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(1.0, 1.0, 1.0, 1.0);
    for i in iter {
      total *= i;
    }
    total
  }
}
impl<'a> core::iter::Product<&'a f32x4> for f32x4 {
  #[inline]
  fn product<I: Iterator<Item = &'a f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(1.0, 1.0, 1.0, 1.0);
    for i in iter {
      total *= i;
    }
    total
  }
}

impl Neg for f32x4 {
  type Output = f32x4;
  #[inline]
  fn neg(self) -> f32x4 {
    f32x4::new(0.0, 0.0, 0.0, 0.0) - self
  }
}
impl Neg for &'_ f32x4 {
  type Output = f32x4;
  #[inline]
  fn neg(self) -> f32x4 {
    f32x4::new(0.0, 0.0, 0.0, 0.0) - self
  }
}

impl AddAssign for f32x4 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs
  }
}

impl AddAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn add_assign(&mut self, rhs: &Self) {
    *self = *self + rhs
  }
}

impl DivAssign for f32x4 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    *self = *self / rhs
  }
}

impl DivAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn div_assign(&mut self, rhs: &Self) {
    *self = *self / rhs
  }
}

impl MulAssign for f32x4 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs
  }
}

impl MulAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn mul_assign(&mut self, rhs: &Self) {
    *self = *self * rhs
  }
}

impl RemAssign for f32x4 {
  #[inline]
  fn rem_assign(&mut self, rhs: Self) {
    *self = *self % rhs
  }
}

impl RemAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn rem_assign(&mut self, rhs: &Self) {
    *self = *self % rhs
  }
}

impl SubAssign for f32x4 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs
  }
}

impl SubAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn sub_assign(&mut self, rhs: &Self) {
    *self = *self - rhs
  }
}

impl BitAndAssign for f32x4 {
  #[inline]
  fn bitand_assign(&mut self, rhs: Self) {
    *self = *self & rhs
  }
}

impl BitAndAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn bitand_assign(&mut self, rhs: &Self) {
    *self = *self & rhs
  }
}

impl BitOrAssign for f32x4 {
  #[inline]
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs
  }
}

impl BitOrAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn bitor_assign(&mut self, rhs: &Self) {
    *self = *self | rhs
  }
}

impl BitXorAssign for f32x4 {
  #[inline]
  fn bitxor_assign(&mut self, rhs: Self) {
    *self = *self ^ rhs
  }
}

impl BitXorAssign<&'_ f32x4> for f32x4 {
  #[inline]
  fn bitxor_assign(&mut self, rhs: &Self) {
    *self = *self ^ rhs
  }
}
