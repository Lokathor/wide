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

impl core::fmt::Debug for i32x4 {
  #[rustfmt::skip]
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [i32; 4] = cast(*self);
    write!(
      f,
      "({},{},{},{})",
      a[0], a[1], a[2], a[3],
    )
  }
}

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
