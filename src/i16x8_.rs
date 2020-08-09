use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i16x8 { sse: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i16x8 { arr: [i16;8] }
  }
}

unsafe impl Zeroable for i16x8 {}
unsafe impl Pod for i16x8 {}

impl core::fmt::Debug for i16x8 {
  #[rustfmt::skip]
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [i16; 8] = cast(*self);
    write!(
      f,
      "({},{},{},{},{},{},{},{})",
      a[0], a[1], a[2], a[3],
      a[4], a[5], a[6], a[7],
    )
  }
}

impl Add for i16x8 {
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
