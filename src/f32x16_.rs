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

impl AlignTo for f32x16 {
  type Elem = f32;
}

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

impl Neg for f32x16 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512(self.avx512, Self::splat(-0.0).avx512) }
      } else {
        Self {
          a : self.a.neg(),
          b : self.b.neg(),
        }
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
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(EqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }
}

impl CmpGt for f32x16 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(GreaterThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_gt(rhs.a),
          b : self.b.simd_gt(rhs.b),
        }
      }
    }
  }
}

impl CmpGe for f32x16 {
  type Output = Self;
  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(GreaterEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ge(rhs.a),
          b : self.b.simd_ge(rhs.b),
        }
      }
    }
  }
}

impl CmpLt for f32x16 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(LessThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_lt(rhs.a),
          b : self.b.simd_lt(rhs.b),
        }
      }
    }
  }
}

impl CmpLe for f32x16 {
  type Output = Self;
  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(LessEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_le(rhs.a),
          b : self.b.simd_le(rhs.b),
        }
      }
    }
  }
}

impl CmpNe for f32x16 {
  type Output = Self;
  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512::<{cmp_op!(NotEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ne(rhs.a),
          b : self.b.simd_ne(rhs.b),
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
        // max_m512 seems to do rhs < self ? self : rhs. So if there's any NaN
        // involved, it chooses rhs, so we need to specifically check rhs for
        // NaN.
        rhs.is_nan().blend(self, Self { avx512: max_m512(self.avx512, rhs.avx512) })
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
        // min_m512 seems to do rhs > self ? self : rhs. So if there's any NaN
        // involved, it chooses rhs, so we need to specifically check rhs for
        // NaN.
        rhs.is_nan().blend(self, Self { avx512: min_m512(self.avx512, rhs.avx512) })
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
    let out = !(shift_u & shifted_exp_mask).simd_eq(shifted_exp_mask);
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
  pub fn as_array(&self) -> &[f32; 16] {
    cast_ref(self)
  }

  #[inline]
  #[must_use]
  pub fn as_mut_array(&mut self) -> &mut [f32; 16] {
    cast_mut(self)
  }

  /// Performs a multiply-add operation: `self * m + a`
  ///
  /// When hardware FMA support is available, this computes the result with a
  /// single rounding operation. Without FMA support, it falls back to separate
  /// multiply and add operations with two roundings.
  ///
  /// # Platform-specific behavior
  /// - On `x86`/`x86_64` with AVX-512F+FMA: Uses 512-bit `vfmadd` (single
  ///   rounding, best accuracy)
  /// - On `x86`/`x86_64` with AVX-512F only: Uses `(self * m) + a` (two
  ///   roundings)
  /// - Other platforms: Delegates to [`f32x8`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([1.0; 16]);
  /// let b = f32x16::from([2.0; 16]);
  /// let c = f32x16::from([10.0; 16]);
  ///
  /// let result = a.mul_add(b, c);
  ///
  /// let expected = f32x16::from([12.0; 16]);
  /// assert_eq!(result, expected);
  /// ```
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

  /// Performs a multiply-subtract operation: `self * m - s`
  ///
  /// When hardware FMA support is available, this computes the result with a
  /// single rounding operation. Without FMA support, it falls back to separate
  /// multiply and subtract operations with two roundings.
  ///
  /// # Platform-specific behavior
  /// - On `x86`/`x86_64` with AVX-512F+FMA: Uses 512-bit `vfmsub` (single
  ///   rounding, best accuracy)
  /// - On `x86`/`x86_64` with AVX-512F only: Uses `(self * m) - s` (two
  ///   roundings)
  /// - Other platforms: Delegates to [`f32x8`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([10.0; 16]);
  /// let b = f32x16::from([3.0; 16]);
  /// let c = f32x16::from([5.0; 16]);
  ///
  /// let result = a.mul_sub(b, c);
  ///
  /// let expected = f32x16::from([25.0; 16]);
  /// assert_eq!(result, expected);
  /// ```
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

  /// Performs a negative multiply-add operation: `a - (self * m)`
  ///
  /// When hardware FMA support is available, this computes the result with a
  /// single rounding operation. Without FMA support, it falls back to separate
  /// operations with two roundings.
  ///
  /// # Platform-specific behavior
  /// - On `x86`/`x86_64` with AVX-512F+FMA: Uses 512-bit `vfnmadd` (single
  ///   rounding, best accuracy)
  /// - On `x86`/`x86_64` with AVX-512F only: Uses `a - (self * m)` (two
  ///   roundings)
  /// - Other platforms: Delegates to [`f32x8`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([4.0; 16]);
  /// let b = f32x16::from([2.0; 16]);
  /// let c = f32x16::from([10.0; 16]);
  ///
  /// let result = a.mul_neg_add(b, c);
  ///
  /// let expected = f32x16::from([2.0; 16]);
  /// assert_eq!(result, expected);
  /// ```
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

  /// Performs a negative multiply-subtract operation: `-(self * m) - s`
  ///
  /// When hardware FMA support is available, this computes the result with a
  /// single rounding operation. Without FMA support, it falls back to separate
  /// operations with two roundings.
  ///
  /// # Platform-specific behavior
  /// - On `x86`/`x86_64` with AVX-512F+FMA: Uses 512-bit `vfnmsub` (single
  ///   rounding, best accuracy)
  /// - On `x86`/`x86_64` with AVX-512F only: Uses `-(self * m) - s` (two
  ///   roundings)
  /// - Other platforms: Delegates to [`f32x8`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([4.0; 16]);
  /// let b = f32x16::from([2.0; 16]);
  /// let c = f32x16::from([1.0; 16]);
  ///
  /// let result = a.mul_neg_sub(b, c);
  ///
  /// let expected = f32x16::from([-9.0; 16]);
  /// assert_eq!(result, expected);
  /// ```
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
