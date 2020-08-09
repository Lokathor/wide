use super::*;

pick! {
  if #[cfg(target_feature="sse")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f32x4 { sse: m128 }
  } else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(16))]
    pub struct f32x4 { arr: [f32;4] }
  }
}

unsafe impl Zeroable for f32x4 {}
unsafe impl Pod for f32x4 {}

impl core::fmt::Debug for f32x4 {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [f32; 4] = cast(*self);
    write!(f, "({},{},{},{})", a[0], a[1], a[2], a[3])
  }
}

impl Add for f32x4 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse")] {
        Self { sse: add_m128(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0] + rhs.arr[0],
          self.arr[1] + rhs.arr[1],
          self.arr[2] + rhs.arr[2],
          self.arr[3] + rhs.arr[3],
        ]}
      }
    }
  }
}
