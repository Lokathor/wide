use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u64x2 { sse: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct u64x2 { arr: [u64;2] }
  }
}

unsafe impl Zeroable for u64x2 {}
unsafe impl Pod for u64x2 {}

impl core::fmt::Debug for u64x2 {
  #[rustfmt::skip]
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    let a: [u64; 2] = cast(*self);
    write!(
      f,
      "({},{})",
      a[0], a[1],
    )
  }
}

impl Add for u64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i64_m128i(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].wrapping_add(rhs.arr[0]),
          self.arr[1].wrapping_add(rhs.arr[1]),
        ]}
      }
    }
  }
}
