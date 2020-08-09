use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i8x16 { sse: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i8x16 { arr: [i8;16] }
  }
}

unsafe impl Zeroable for i8x16 {}
unsafe impl Pod for i8x16 {}

impl Add for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: add_i8_m128i(self.sse, rhs.sse) }
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
          self.arr[8].wrapping_add(rhs.arr[8]),
          self.arr[9].wrapping_add(rhs.arr[9]),
          self.arr[10].wrapping_add(rhs.arr[10]),
          self.arr[11].wrapping_add(rhs.arr[11]),
          self.arr[12].wrapping_add(rhs.arr[12]),
          self.arr[13].wrapping_add(rhs.arr[13]),
          self.arr[14].wrapping_add(rhs.arr[14]),
          self.arr[15].wrapping_add(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl Sub for i8x16 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i8_m128i(self.sse, rhs.sse) }
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
          self.arr[8].wrapping_sub(rhs.arr[8]),
          self.arr[9].wrapping_sub(rhs.arr[9]),
          self.arr[10].wrapping_sub(rhs.arr[10]),
          self.arr[11].wrapping_sub(rhs.arr[11]),
          self.arr[12].wrapping_sub(rhs.arr[12]),
          self.arr[13].wrapping_sub(rhs.arr[13]),
          self.arr[14].wrapping_sub(rhs.arr[14]),
          self.arr[15].wrapping_sub(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl BitAnd for i8x16 {
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
          self.arr[8].bitand(rhs.arr[8]),
          self.arr[9].bitand(rhs.arr[9]),
          self.arr[10].bitand(rhs.arr[10]),
          self.arr[11].bitand(rhs.arr[11]),
          self.arr[12].bitand(rhs.arr[12]),
          self.arr[13].bitand(rhs.arr[13]),
          self.arr[14].bitand(rhs.arr[14]),
          self.arr[15].bitand(rhs.arr[15]),
        ]}
      }
    }
  }
}

impl BitOr for i8x16 {
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
          self.arr[8].bitor(rhs.arr[8]),
          self.arr[9].bitor(rhs.arr[9]),
          self.arr[10].bitor(rhs.arr[10]),
          self.arr[11].bitor(rhs.arr[11]),
          self.arr[12].bitor(rhs.arr[12]),
          self.arr[13].bitor(rhs.arr[13]),
          self.arr[14].bitor(rhs.arr[14]),
          self.arr[15].bitor(rhs.arr[15]),
        ]}
      }
    }
  }
}
