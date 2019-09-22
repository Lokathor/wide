use super::*;

/*

Method TODO:
* partial_eq / partial_ord replacement

*/

cfg_if! {
  if #[cfg(target_feature="sse")] {
    #[repr(C, align(16))]
    pub struct f32x4 {
      sse: m128
    }
  } else {
    #[repr(C, align(16))]
    pub struct f32x4 {
      arr: [f32; 4]
    }
  }
}

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

impl Index<usize> for f32x4 {
  type Output = f32;
  #[inline(always)]
  fn index(&self, index: usize) -> &f32 {
    let r: &[f32; 4] = cast_ref(self);
    &r[index]
  }
}
impl IndexMut<usize> for f32x4 {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut f32 {
    let r: &mut [f32; 4] = cast_mut(self);
    &mut r[index]
  }
}

#[test]
fn declaration_tests() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<f32x4>(), 16);
  assert_eq!(align_of::<f32x4>(), 16);
}

/// Consts
impl f32x4 {
  pub const EPSILON: [f32; 4] = [
    std::f32::EPSILON,
    std::f32::EPSILON,
    std::f32::EPSILON,
    std::f32::EPSILON,
  ];
  pub const INFINITY: [f32; 4] = [
    std::f32::INFINITY,
    std::f32::INFINITY,
    std::f32::INFINITY,
    std::f32::INFINITY,
  ];
  pub const MAX: [f32; 4] = [std::f32::MAX, std::f32::MAX, std::f32::MAX, std::f32::MAX];
  pub const MIN: [f32; 4] = [std::f32::MIN, std::f32::MIN, std::f32::MIN, std::f32::MIN];
  pub const MIN_POSITIVE: [f32; 4] = [
    std::f32::MIN_POSITIVE,
    std::f32::MIN_POSITIVE,
    std::f32::MIN_POSITIVE,
    std::f32::MIN_POSITIVE,
  ];
  pub const NAN: [f32; 4] = [std::f32::NAN, std::f32::NAN, std::f32::NAN, std::f32::NAN];
  pub const NEG_INFINITY: [f32; 4] = [
    std::f32::NEG_INFINITY,
    std::f32::NEG_INFINITY,
    std::f32::NEG_INFINITY,
    std::f32::NEG_INFINITY,
  ];
  pub const DIGITS: [u32; 4] = [
    std::f32::DIGITS,
    std::f32::DIGITS,
    std::f32::DIGITS,
    std::f32::DIGITS,
  ];
  pub const MANTISSA_DIGITS: [u32; 4] = [
    std::f32::MANTISSA_DIGITS,
    std::f32::MANTISSA_DIGITS,
    std::f32::MANTISSA_DIGITS,
    std::f32::MANTISSA_DIGITS,
  ];
  pub const RADIX: [u32; 4] = [
    std::f32::RADIX,
    std::f32::RADIX,
    std::f32::RADIX,
    std::f32::RADIX,
  ];
  pub const MAX_10_EXP: [i32; 4] = [
    std::f32::MAX_10_EXP,
    std::f32::MAX_10_EXP,
    std::f32::MAX_10_EXP,
    std::f32::MAX_10_EXP,
  ];
  pub const MAX_EXP: [i32; 4] = [
    std::f32::MAX_EXP,
    std::f32::MAX_EXP,
    std::f32::MAX_EXP,
    std::f32::MAX_EXP,
  ];
  pub const MIN_10_EXP: [i32; 4] = [
    std::f32::MIN_10_EXP,
    std::f32::MIN_10_EXP,
    std::f32::MIN_10_EXP,
    std::f32::MIN_10_EXP,
  ];
  pub const MIN_EXP: [i32; 4] = [
    std::f32::MIN_EXP,
    std::f32::MIN_EXP,
    std::f32::MIN_EXP,
    std::f32::MIN_EXP,
  ];
  pub const E: [f32; 4] = [
    std::f32::consts::E,
    std::f32::consts::E,
    std::f32::consts::E,
    std::f32::consts::E,
  ];
  pub const FRAC_1_PI: [f32; 4] = [
    std::f32::consts::FRAC_1_PI,
    std::f32::consts::FRAC_1_PI,
    std::f32::consts::FRAC_1_PI,
    std::f32::consts::FRAC_1_PI,
  ];
  pub const FRAC_2_PI: [f32; 4] = [
    std::f32::consts::FRAC_2_PI,
    std::f32::consts::FRAC_2_PI,
    std::f32::consts::FRAC_2_PI,
    std::f32::consts::FRAC_2_PI,
  ];
  pub const FRAC_2_SQRT_PI: [f32; 4] = [
    std::f32::consts::FRAC_2_SQRT_PI,
    std::f32::consts::FRAC_2_SQRT_PI,
    std::f32::consts::FRAC_2_SQRT_PI,
    std::f32::consts::FRAC_2_SQRT_PI,
  ];
  pub const FRAC_1_SQRT_2: [f32; 4] = [
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
  ];
  pub const FRAC_PI_2: [f32; 4] = [
    std::f32::consts::FRAC_PI_2,
    std::f32::consts::FRAC_PI_2,
    std::f32::consts::FRAC_PI_2,
    std::f32::consts::FRAC_PI_2,
  ];
  pub const FRAC_PI_3: [f32; 4] = [
    std::f32::consts::FRAC_PI_3,
    std::f32::consts::FRAC_PI_3,
    std::f32::consts::FRAC_PI_3,
    std::f32::consts::FRAC_PI_3,
  ];
  pub const FRAC_PI_4: [f32; 4] = [
    std::f32::consts::FRAC_PI_4,
    std::f32::consts::FRAC_PI_4,
    std::f32::consts::FRAC_PI_4,
    std::f32::consts::FRAC_PI_4,
  ];
  pub const FRAC_PI_6: [f32; 4] = [
    std::f32::consts::FRAC_PI_6,
    std::f32::consts::FRAC_PI_6,
    std::f32::consts::FRAC_PI_6,
    std::f32::consts::FRAC_PI_6,
  ];
  pub const FRAC_PI_8: [f32; 4] = [
    std::f32::consts::FRAC_PI_8,
    std::f32::consts::FRAC_PI_8,
    std::f32::consts::FRAC_PI_8,
    std::f32::consts::FRAC_PI_8,
  ];
  pub const LN_2: [f32; 4] = [
    std::f32::consts::LN_2,
    std::f32::consts::LN_2,
    std::f32::consts::LN_2,
    std::f32::consts::LN_2,
  ];
  pub const LN_10: [f32; 4] = [
    std::f32::consts::LN_10,
    std::f32::consts::LN_10,
    std::f32::consts::LN_10,
    std::f32::consts::LN_10,
  ];
  pub const LOG2_E: [f32; 4] = [
    std::f32::consts::LOG2_E,
    std::f32::consts::LOG2_E,
    std::f32::consts::LOG2_E,
    std::f32::consts::LOG2_E,
  ];
  pub const LOG10_E: [f32; 4] = [
    std::f32::consts::LOG10_E,
    std::f32::consts::LOG10_E,
    std::f32::consts::LOG10_E,
    std::f32::consts::LOG10_E,
  ];
  pub const PI: [f32; 4] = [
    std::f32::consts::PI,
    std::f32::consts::PI,
    std::f32::consts::PI,
    std::f32::consts::PI,
  ];
  pub const SQRT_2: [f32; 4] = [
    std::f32::consts::SQRT_2,
    std::f32::consts::SQRT_2,
    std::f32::consts::SQRT_2,
    std::f32::consts::SQRT_2,
  ];
}

impl f32x4 {
  #[inline(always)]
  pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
    cfg_block! {if #[cfg(target_feature="sse")] {
      Self { sse: m128::set(a,b,c,d) }
    } else {
      Self { arr: [a,b,c,d] }
    }}
  }
}

impl Add for f32x4 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    cfg_block! {if #[cfg(target_feature="sse")] {
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
    cfg_block! {if #[cfg(target_feature="sse")] {
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
    cfg_block! {if #[cfg(target_feature="sse")] {
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
    cfg_block! {if #[cfg(target_feature="sse")] {
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
    cfg_block! {if #[cfg(target_feature="sse")] {
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
    cfg_block! {if #[cfg(target_feature="sse")] {
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

impl Rem for f32x4 {
  type Output = Self;
  #[inline]
  fn rem(self, rhs: Self) -> Self {
    cfg_block! {if #[cfg(target_feature="sse")] {
      let arr1: [f32; 4] = cast(self.sse);
      let arr2: [f32; 4] = cast(rhs.sse);
      Self {
        sse: cast([
        arr1[0] % arr2[0],
        arr1[1] % arr2[1],
        arr1[2] % arr2[2],
        arr1[3] % arr2[3],
      ])
      }
    } else {
      Self { arr: [
        self.arr[0] % rhs.arr[0],
        self.arr[1] % rhs.arr[1],
        self.arr[2] % rhs.arr[2],
        self.arr[3] % rhs.arr[3],
      ] }
    }}
  }
}

impl Rem<&'_ f32x4> for f32x4 {
  type Output = Self;
  #[inline]
  fn rem(self, rhs: &Self) -> Self {
    cfg_block! {if #[cfg(target_feature="sse")] {
      let arr1: [f32; 4] = cast(self.sse);
      let arr2: [f32; 4] = cast(rhs.sse);
      Self {
        sse: cast([
        arr1[0] % arr2[0],
        arr1[1] % arr2[1],
        arr1[2] % arr2[2],
        arr1[3] % arr2[3],
      ])
      }
    } else {
      Self { arr: [
        self.arr[0] % rhs.arr[0],
        self.arr[1] % rhs.arr[1],
        self.arr[2] % rhs.arr[2],
        self.arr[3] % rhs.arr[3],
      ] }
    }}
  }
}

impl Sub for f32x4 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self {
    cfg_block! {if #[cfg(target_feature="sse")] {
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
    cfg_block! {if #[cfg(target_feature="sse")] {
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

impl Debug for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Debug::fmt(&self[0], f)?;
    Debug::fmt(&self[1], f)?;
    Debug::fmt(&self[2], f)?;
    Debug::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl Display for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    Display::fmt(&self[0], f)?;
    Display::fmt(&self[1], f)?;
    Display::fmt(&self[2], f)?;
    Display::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl LowerExp for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    LowerExp::fmt(&self[0], f)?;
    LowerExp::fmt(&self[1], f)?;
    LowerExp::fmt(&self[2], f)?;
    LowerExp::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl UpperExp for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    write!(f, "f32x4(")?;
    UpperExp::fmt(&self[0], f)?;
    UpperExp::fmt(&self[1], f)?;
    UpperExp::fmt(&self[2], f)?;
    UpperExp::fmt(&self[3], f)?;
    write!(f, ")")
  }
}

impl Neg for f32x4 {
  type Output = f32x4;
  #[inline]
  fn neg(self) -> f32x4 {
    cfg_block! {if #[cfg(target_feature="sse")] {
      Self { sse: self.sse.neg() }
    } else {
      Self { arr: [
        -self.arr[0],
        -self.arr[1],
        -self.arr[2],
        -self.arr[3],
      ] }
    }}
  }
}
impl Neg for &'_ f32x4 {
  type Output = f32x4;
  #[inline]
  fn neg(self) -> f32x4 {
    cfg_block! {if #[cfg(target_feature="sse")] {
      f32x4 { sse: self.sse.neg() }
    } else {
      f32x4 { arr: [
        -self.arr[0],
        -self.arr[1],
        -self.arr[2],
        -self.arr[3],
      ] }
    }}
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

impl core::iter::Sum for f32x4 {
  #[inline]
  fn sum<I: Iterator<Item = f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(0.0, 0.0, 0.0, 0.0);
    for i in iter {
      total *= i;
    }
    total
  }
}
impl<'a> core::iter::Sum<&'a f32x4> for f32x4 {
  #[inline]
  fn sum<I: Iterator<Item = &'a f32x4>>(iter: I) -> f32x4 {
    let mut total = f32x4::new(0.0, 0.0, 0.0, 0.0);
    for i in iter {
      total *= i;
    }
    total
  }
}

impl AsRef<[f32; 4]> for f32x4 {
  #[inline(always)]
  fn as_ref(&self) -> &[f32; 4] {
    cast_ref(self)
  }
}
impl AsMut<[f32; 4]> for f32x4 {
  #[inline(always)]
  fn as_mut(&mut self) -> &mut [f32; 4] {
    cast_mut(self)
  }
}

impl From<[f32; 4]> for f32x4 {
  #[inline(always)]
  fn from(arr: [f32; 4]) -> Self {
    cast(arr)
  }
}
impl From<(f32, f32, f32, f32)> for f32x4 {
  #[inline(always)]
  fn from((a, b, c, d): (f32, f32, f32, f32)) -> Self {
    Self::new(a, b, c, d)
  }
}

impl From<[i8; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [i8; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}

impl From<[u8; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [u8; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}

impl From<[i16; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [i16; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}

impl From<[u16; 4]> for f32x4 {
  #[inline]
  fn from([a, b, c, d]: [u16; 4]) -> Self {
    Self::new(f32::from(a), f32::from(b), f32::from(c), f32::from(d))
  }
}
