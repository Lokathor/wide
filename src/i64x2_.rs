use super::*;

pick! {
  if #[cfg(target_feature="sse2")] {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i64x2 { sse: m128i }
  } else {
    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    #[repr(C, align(16))]
    pub struct i64x2 { arr: [i64;2] }
  }
}

int_uint_consts!(i64, 2, i64x2, i64x2, i64a2, const_i64_as_i64x2, 128);

unsafe impl Zeroable for i64x2 {}
unsafe impl Pod for i64x2 {}

impl Add for i64x2 {
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

impl Sub for i64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: sub_i64_m128i(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].wrapping_sub(rhs.arr[0]),
          self.arr[1].wrapping_sub(rhs.arr[1]),
        ]}
      }
    }
  }
}

impl BitAnd for i64x2 {
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
        ]}
      }
    }
  }
}

impl BitOr for i64x2 {
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
        ]}
      }
    }
  }
}

impl BitXor for i64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse2")] {
        Self { sse: bitxor_m128i(self.sse, rhs.sse) }
      } else {
        Self { arr: [
          self.arr[0].bitxor(rhs.arr[0]),
          self.arr[1].bitxor(rhs.arr[1]),
        ]}
      }
    }
  }
}

macro_rules! impl_shl_t_for_i64x2 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shl<$shift_type> for i64x2 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shl(self, rhs: $shift_type) -> Self::Output {
        let u = rhs as u64;
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse: shl_all_u64_m128i(self.sse, shift) }
          } else {
            Self { arr: [
              self.arr[0] << u,
              self.arr[1] << u,
            ]}
          }
        }
      }
    })+
  };
}
impl_shl_t_for_i64x2!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_shr_t_for_i64x2 {
  ($($shift_type:ty),+ $(,)?) => {
    $(impl Shr<$shift_type> for i64x2 {
      type Output = Self;
      /// Shifts all lanes by the value given.
      #[inline]
      #[must_use]
      fn shr(self, rhs: $shift_type) -> Self::Output {
        let u = rhs as u64;
        pick! {
          if #[cfg(target_feature="sse2")] {
            let shift = cast([u, 0]);
            Self { sse: shr_all_u64_m128i(self.sse, shift) }
          } else {
            Self { arr: [
              self.arr[0] >> u,
              self.arr[1] >> u,
            ]}
          }
        }
      }
    })+
  };
}

impl_shr_t_for_i64x2!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl CmpEq for i64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: cmp_eq_mask_i64_m128i(self.sse, rhs.sse) }
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] == r[0] { -1_i64 } else { 0 },
          if s[1] == r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }
}

impl CmpGt for i64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        Self { sse: cmp_gt_mask_i64_m128i(self.sse, rhs.sse) }
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] > r[0] { -1_i64 } else { 0 },
          if s[1] > r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }
}

impl CmpLt for i64x2 {
  type Output = Self;
  #[inline]
  #[must_use]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="sse4.2")] {
        Self { sse: !cmp_gt_mask_i64_m128i(self.sse, rhs.sse) }
      } else {
        let s: [i64;2] = cast(self);
        let r: [i64;2] = cast(rhs);
        cast([
          if s[0] < r[0] { -1_i64 } else { 0 },
          if s[1] < r[1] { -1_i64 } else { 0 },
        ])
      }
    }
  }
}

impl i64x2 {
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="sse4.1")] {
        Self { sse: blend_varying_i8_m128i(f.sse, t.sse, self.sse) }
      } else {
        generic_bit_blend(self, t, f)
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn round_float(self) -> f64x2 {
    let arr: [i64; 2] = cast(self);
    cast([arr[0] as f64, arr[1] as f64])
  }
}
