use super::*;

pick! {
  if #[cfg(target_feature="avx512f")] {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(64))]
    pub struct f64x8 { pub(crate) avx512: m512d }
  } else {
    #[derive(Default, Clone, Copy, PartialEq)]
    #[repr(C, align(64))]
    pub struct f64x8 { pub(crate) a : f64x4, pub(crate) b : f64x4 }
  }
}

unsafe impl Zeroable for f64x8 {}
unsafe impl Pod for f64x8 {}

impl Add for f64x8 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: add_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.add(rhs.a),
          b : self.b.add(rhs.b),
        }
      }
    }
  }
}

impl Sub for f64x8 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sub_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.sub(rhs.a),
          b : self.b.sub(rhs.b),
        }
      }
    }
  }
}

impl Mul for f64x8 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: mul_m512d(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.mul(rhs.a), b: self.b.mul(rhs.b) }
      }
    }
  }
}

impl Div for f64x8 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: div_m512d(self.avx512, rhs.avx512) }
      } else {
        Self { a: self.a.div(rhs.a), b: self.b.div(rhs.b) }
      }
    }
  }
}

impl Add<f64> for f64x8 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: f64) -> Self::Output {
    self.add(Self::splat(rhs))
  }
}

impl Sub<f64> for f64x8 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: f64) -> Self::Output {
    self.sub(Self::splat(rhs))
  }
}

impl Mul<f64> for f64x8 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f64) -> Self::Output {
    self.mul(Self::splat(rhs))
  }
}

impl Div<f64> for f64x8 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: f64) -> Self::Output {
    self.div(Self::splat(rhs))
  }
}

impl Add<f64x8> for f64 {
  type Output = f64x8;
  #[inline]
  fn add(self, rhs: f64x8) -> Self::Output {
    f64x8::splat(self).add(rhs)
  }
}

impl Sub<f64x8> for f64 {
  type Output = f64x8;
  #[inline]
  fn sub(self, rhs: f64x8) -> Self::Output {
    f64x8::splat(self).sub(rhs)
  }
}

impl Mul<f64x8> for f64 {
  type Output = f64x8;
  #[inline]
  fn mul(self, rhs: f64x8) -> Self::Output {
    f64x8::splat(self).mul(rhs)
  }
}

impl Div<f64x8> for f64 {
  type Output = f64x8;
  #[inline]
  fn div(self, rhs: f64x8) -> Self::Output {
    f64x8::splat(self).div(rhs)
  }
}

impl BitAnd for f64x8 {
  type Output = Self;
  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitand_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitand(rhs.a),
          b : self.b.bitand(rhs.b),
        }
      }
    }
  }
}

impl BitOr for f64x8 {
  type Output = Self;
  #[inline]
  fn bitor(self, rhs: Self) -> Self::Output {
    pick! {
    if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitor_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitor(rhs.a),
          b : self.b.bitor(rhs.b),
        }
      }
    }
  }
}

impl BitXor for f64x8 {
  type Output = Self;
  #[inline]
  fn bitxor(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.bitxor(rhs.a),
          b : self.b.bitxor(rhs.b),
        }
      }
    }
  }
}

impl CmpEq for f64x8 {
  type Output = Self;
  #[inline]
  fn cmp_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(EqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_eq(rhs.a),
          b : self.b.cmp_eq(rhs.b),
        }
      }
    }
  }
}

impl CmpGt for f64x8 {
  type Output = Self;
  #[inline]
  fn cmp_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(GreaterThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_gt(rhs.a),
          b : self.b.cmp_gt(rhs.b),
        }
      }
    }
  }
}

impl CmpGe for f64x8 {
  type Output = Self;
  #[inline]
  fn cmp_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(GreaterEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_ge(rhs.a),
          b : self.b.cmp_ge(rhs.b),
        }
      }
    }
  }
}

impl CmpLt for f64x8 {
  type Output = Self;
  #[inline]
  fn cmp_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(LessThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_lt(rhs.a),
          b : self.b.cmp_lt(rhs.b),
        }
      }
    }
  }
}

impl CmpLe for f64x8 {
  type Output = Self;
  #[inline]
  fn cmp_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(LessEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_le(rhs.a),
          b : self.b.cmp_le(rhs.b),
        }
      }
    }
  }
}

impl CmpNe for f64x8 {
  type Output = Self;
  #[inline]
  fn cmp_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(NotEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.cmp_ne(rhs.a),
          b : self.b.cmp_ne(rhs.b),
        }
      }
    }
  }
}

impl f64x8 {
  #[inline]
  #[must_use]
  pub fn new(array: [f64; 8]) -> Self {
    Self::from(array)
  }
  
  #[inline]
  #[must_use]
  pub fn blend(self, t: Self, f: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: blend_varying_m512d(f.avx512, t.avx512, movepi64_mask_m512d(self.avx512)) }
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
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let non_sign_bits = f64x8::from(f64::from_bits(i64::MAX as u64));
        self & non_sign_bits
      } else {
        Self {
          a: self.a.abs(),
          b: self.b.abs(),
        }
      }
    }
  }
  
  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_m512d(self.avx512, rhs.avx512) }
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
        Self { avx512: min_m512d(self.avx512, rhs.avx512) }
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
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(Unordered)}>(self.avx512, self.avx512) }
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
    let shifted_exp_mask = u64x8::splat(0x7FF0000000000000);
    let u: u64x8 = cast(self);
    let shift_u = u << 1_u64;
    let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
    cast(out)
  }
  
  #[inline]
  #[must_use]
  pub fn round(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: round_m512d::<{round_op!(Nearest)}>(self.avx512) }
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
  pub fn round_int(self) -> i64x8 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        cast(convert_to_i64_m512i_from_m512d(self.avx512))
      } else {
        i64x8 {
          a: self.a.round_int(),
          b: self.b.round_int(),
        }
      }
    }
  }
  
  #[inline]
  #[must_use]
  pub fn to_array(self) -> [f64; 8] {
    cast(self)
  }
  
  #[inline]
  #[must_use]
  pub fn as_array_ref(&self) -> &[f64; 8] {
    cast_ref(self)
  }
  
  #[inline]
  #[must_use]
  pub fn as_array_mut(&mut self) -> &mut [f64; 8] {
    cast_mut(self)
  }
}

impl Not for f64x8 {
  type Output = Self;
  #[inline]
  fn not(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512d(self.avx512, set_splat_m512d(f64::from_bits(u64::MAX))) }
      } else {
        Self {
          a : self.a.not(),
          b : self.b.not(),
        }
      }
    }
  }
}