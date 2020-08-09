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

impl core::fmt::Debug for f64x2 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f64; 2] = cast(*self);
    write!(f, "({},{})", a[0], a[1])
  }
}

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
