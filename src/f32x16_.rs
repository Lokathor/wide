use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(64))]
    pub struct f32x16 { pub(crate) avx512: m512 }
  } else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(64))]
    pub struct f32x16 { pub(crate) a : f32x8, pub(crate) b : f32x8 }
  }
}

unsafe impl Zeroable for f32x16 {}
unsafe impl Pod for f32x16 {}

impl Add for f32x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: add_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }
}

impl Sub for f32x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sub_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }
}

impl Mul for f32x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: mul_m512(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }
}

impl Div for f32x16 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: div_m512(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.div(rhs.a), b: self.b.div(rhs.b) }
      }
    }
  }
}

impl Add<f32> for f32x16 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: f32) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<f32> for f32x16 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: f32) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<f32> for f32x16 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Div<f32> for f32x16 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: f32) -> Self::Output {
    self.div(Self::splat(rhs))
  }
}

impl Add<f32x16> for f32 {
  type Output = f32x16;
  #[inline]
  fn add(self, rhs: f32x16) -> Self::Output {
    f32x16::splat(self).add(rhs)
  }
}

impl Sub<f32x16> for f32 {
  type Output = f32x16;
  #[inline]
  fn sub(self, rhs: f32x16) -> Self::Output {
    f32x16::splat(self).sub(rhs)
  }
}

impl Mul<f32x16> for f32 {
  type Output = f32x16;
  #[inline]
  fn mul(self, rhs: f32x16) -> Self::Output {
    f32x16::splat(self).mul(rhs)
  }
}

impl Div<f32x16> for f32 {
  type Output = f32x16;
  #[inline]
  fn div(self, rhs: f32x16) -> Self::Output {
    f32x16::splat(self).div(rhs)
  }
}

impl BitAnd for f32x16 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitand_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitand(rhs.a),
          b : self.b.bitand(rhs.b),
        }
      }
    }
  }
}

impl BitOr for f32x16 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitor_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitor(rhs.a),
          b : self.b.bitor(rhs.b),
        }
      }
    }
  }
}

impl BitXor for f32x16 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitxor(rhs.a),
          b : self.b.bitxor(rhs.b),
        }
      }
    }
  }
}

impl CmpEq for f32x16 {
  type Output = Self;
  #[inline]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(EqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_eq(rhs.a),
          b : self.b.cmp_eq(rhs.b),
        }
      }
    }
  }
}

impl CmpGt for f32x16 {
  type Output = Self;
  #[inline]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(GreaterThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_gt(rhs.a),
          b : self.b.cmp_gt(rhs.b),
        }
      }
    }
  }
}

impl CmpGe for f32x16 {
  type Output = Self;
  #[inline]
  fn cmp_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(GreaterEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_ge(rhs.a),
          b : self.b.cmp_ge(rhs.b),
        }
      }
    }
  }
}

impl CmpLt for f32x16 {
  type Output = Self;
  #[inline]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(LessThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_lt(rhs.a),
          b : self.b.cmp_lt(rhs.b),
        }
      }
    }
  }
}

impl CmpLe for f32x16 {
  type Output = Self;
  #[inline]
  fn cmp_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(LessEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_le(rhs.a),
          b : self.b.cmp_le(rhs.b),
        }
      }
    }
  }
}

impl CmpNe for f32x16 {
  type Output = Self;
  #[inline]
  fn cmp_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(NotEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_ne(rhs.a),
          b : self.b.cmp_ne(rhs.b),
        }
      }
    }
  }
}

impl f32x16 {
  #[inline]
  #[must_use]
  pub fn new(array: [f32; 16]) -> Self {
    Self::from(array)
  }

  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_m512(f.avx512, t.avx512, movepi32_mask_m512(self.avx512)) }
      } else {
        Self {
          a : self.a.blend(t.a, f.a),
          b : self.b.blend(t.b, f.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.max(rhs.a),
          b: self.b.max(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.min(rhs.a),
          b: self.b.min(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn is_nan(self) -> Self {
    pick! {
      if #[cfg(target_feature = "avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(Unordered)}>(self.avx512, self.avx512) }
      } else {
        Self {
          a: self.a.is_nan(),
          b: self.b.is_nan(),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn is_finite(self) -> Self {
    let shifted_exp_mask = u32x16::splat(0x7F800000);
    let u: u32x16 = cast(self);
    let shift_u = u << 1_u32;
    let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
    cast(out)
  }

  #[inline]
  #[must_use]
  pub fn round(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: round_m512::<{round_op!(Nearest)}>(self.avx512) }
      } else {
        Self {
          a: self.a.round(),
          b: self.b.round(),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn round_int(self) -> i32x16 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        cast(convert_to_i32_m512i_from_m512(self.avx512))
      } else {
        i32x16 {
          a: self.a.round_int(),
          b: self.b.round_int(),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn to_array(self) -> [f32; 16] {
    cast(self)
  }

  #[inline]
  #[must_use]
  pub fn as_array_ref(&self) -> &[f32; 16] {
    cast_ref(self)
  }

  #[inline]
  #[must_use]
  pub fn as_array_mut(&mut self) -> &mut [f32; 16] {
    cast_mut(self)
  }

  #[inline]
  #[must_use]
  pub fn mul_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_add_m512(self.avx512, m.avx512, a.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        (self * m) + a
      } else {
        Self {
          a: self.a.mul_add(m.a, a.a),
          b: self.b.mul_add(m.b, a.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_sub(self, m: Self, s: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_sub_m512(self.avx512, m.avx512, s.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        (self * m) - s
      } else {
        Self {
          a: self.a.mul_sub(m.a, s.a),
          b: self.b.mul_sub(m.b, s.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_neg_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_neg_add_m512(self.avx512, m.avx512, a.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        a - (self * m)
      } else {
        Self {
          a: self.a.mul_neg_add(m.a, a.a),
          b: self.b.mul_neg_add(m.b, a.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn mul_neg_sub(self, m: Self, s: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_neg_sub_m512(self.avx512, m.avx512, s.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        -(self * m) - s
      } else {
        Self {
          a: self.a.mul_neg_sub(m.a, s.a),
          b: self.b.mul_neg_sub(m.b, s.b),
        }
      }
    }
  }
}

impl Not for f32x16 {
  type Output = Self;
  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512(self.avx512, set_splat_m512(f32::from_bits(u32::MAX))) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }
}
