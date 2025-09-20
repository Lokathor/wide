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

  /// Lanewise absolute value, treating each lane as a signed value.
  ///
  /// This is implemented by clearing the sign bit for each lane.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0,
  ///   14.0, -15.0, 16.0,
  /// ]);
  /// let result = a.abs();
  /// let expected = f32x16::from([
  ///   1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
  ///   15.0, 16.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn abs(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        let non_sign_bits = f32x16::from(f32::from_bits(i32::MAX as u32));
        self & non_sign_bits
      } else {
        Self {
          a: self.a.abs(),
          b: self.b.abs(),
        }
      }
    }
  }

  /// Lanewise square root.
  ///
  /// # Platform-specific behavior
  /// - On `x86/x86_64` with AVX-512F: Uses 512-bit `vsqrtps` (hardware square
  ///   root)
  /// - Other platforms: Delegates to [`f32x8`] (inherits its sqrt behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
  ///   169.0, 196.0, 225.0, 256.0,
  /// ]);
  /// let result = a.sqrt();
  /// let expected = f32x16::from([
  ///   1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
  ///   15.0, 16.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn sqrt(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sqrt_m512(self.avx512) }
      } else {
        Self {
          a: self.a.sqrt(),
          b: self.b.sqrt(),
        }
      }
    }
  }

  /// Lanewise floor (round towards negative infinity).
  ///
  /// # Platform-specific behavior
  /// - On `x86/x86_64` with AVX-512F: Uses 512-bit rounding with down mode
  /// - Other platforms: Delegates to [`f32x8`] (inherits its floor behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.1, 2.9, -1.1, -2.9, 3.5, -3.5, 0.0, -0.0, 1.1, 2.9, -1.1, -2.9, 3.5,
  ///   -3.5, 0.0, -0.0,
  /// ]);
  /// let result = a.floor();
  /// let expected = f32x16::from([
  ///   1.0, 2.0, -2.0, -3.0, 3.0, -4.0, 0.0, -0.0, 1.0, 2.0, -2.0, -3.0, 3.0,
  ///   -4.0, 0.0, -0.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn floor(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: round_m512::<{round_op!(NegInf)}>(self.avx512) }
      } else {
        Self {
          a: self.a.floor(),
          b: self.b.floor(),
        }
      }
    }
  }

  /// Lanewise ceiling (round towards positive infinity).
  ///
  /// # Platform-specific behavior
  /// - On `x86/x86_64` with AVX-512F: Uses 512-bit rounding with up mode
  /// - Other platforms: Delegates to [`f32x8`] (inherits its ceil behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.1, 2.9, -1.1, -2.9, 3.5, -3.5, 0.0, -0.0, 1.1, 2.9, -1.1, -2.9, 3.5,
  ///   -3.5, 0.0, -0.0,
  /// ]);
  /// let result = a.ceil();
  /// let expected = f32x16::from([
  ///   2.0, 3.0, -1.0, -2.0, 4.0, -3.0, 0.0, -0.0, 2.0, 3.0, -1.0, -2.0, 4.0,
  ///   -3.0, 0.0, -0.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn ceil(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: round_m512::<{round_op!(PosInf)}>(self.avx512) }
      } else {
        Self {
          a: self.a.ceil(),
          b: self.b.ceil(),
        }
      }
    }
  }

  /// Calculates the lanewise maximum of both vectors. This is a faster
  /// implementation than `max`, but it doesn't specify any behavior if NaNs are
  /// involved.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0, 1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0,
  ///   8.0,
  /// ]);
  /// let b = f32x16::from([
  ///   4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0,
  ///   0.0,
  /// ]);
  /// let result = a.fast_max(b);
  /// let expected = f32x16::from([
  ///   4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 8.0, 4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0,
  ///   8.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn fast_max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.fast_max(rhs.a),
          b: self.b.fast_max(rhs.b),
        }
      }
    }
  }

  /// Calculates the lanewise minimum of both vectors. This is a faster
  /// implementation than `min`, but it doesn't specify any behavior if NaNs are
  /// involved.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0, 1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0,
  ///   8.0,
  /// ]);
  /// let b = f32x16::from([
  ///   4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0,
  ///   0.0,
  /// ]);
  /// let result = a.fast_min(b);
  /// let expected = f32x16::from([
  ///   1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0,
  ///   0.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn fast_min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_m512(self.avx512, rhs.avx512) }
      } else {
        Self {
          a: self.a.fast_min(rhs.a),
          b: self.b.fast_min(rhs.b),
        }
      }
    }
  }

  /// Gathers the sign bits of each lane into a single integer.
  ///
  /// The output has bit `i` set if lane `i` had a negative sign.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   -1.0, 0.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0,
  ///   14.0, -15.0, 16.0,
  /// ]);
  /// let mask = a.move_mask();
  /// assert_eq!(mask, 0b0101010101010101);
  ///
  /// // Test with all negative
  /// let all_neg = f32x16::from([-1.0; 16]);
  /// assert_eq!(all_neg.move_mask(), 0xFFFF);
  ///
  /// // Test with all positive
  /// let all_pos = f32x16::from([1.0; 16]);
  /// assert_eq!(all_pos.move_mask(), 0x0000);
  /// ```
  #[inline]
  #[must_use]
  pub fn move_mask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi32_mask_m512(self.avx512) as u32
      } else {
        (self.b.move_mask() << 8) | self.a.move_mask()
      }
    }
  }

  /// True if any lane has a negative sign bit.
  ///
  /// This checks if any lane is negative (sign bit set).
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([0.0; 16]);
  /// assert!(!a.any());
  ///
  /// let b = f32x16::from([
  ///   -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ///   0.0,
  /// ]);
  /// assert!(b.any());
  /// ```
  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi32_mask_m512(self.avx512) != 0
      } else {
        self.a.any() || self.b.any()
      }
    }
  }

  /// True if all lanes have a negative sign bit.
  ///
  /// This checks if all lanes are negative (sign bit set).
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([-1.0; 16]);
  /// assert!(a.all());
  ///
  /// let b = f32x16::from([
  ///   -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
  ///   -1.0, -1.0, -1.0, -1.0,
  /// ]);
  /// assert!(!b.all());
  /// ```
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi32_mask_m512(self.avx512) == 0b1111111111111111
      } else {
        self.a.all() && self.b.all()
      }
    }
  }

  /// True if no lanes have a negative sign bit.
  ///
  /// This is equivalent to `!self.any()` and checks if all lanes are
  /// non-negative.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([0.0; 16]);
  /// assert!(a.none());
  ///
  /// let b = f32x16::from([
  ///   -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ///   0.0,
  /// ]);
  /// assert!(!b.none());
  /// ```
  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  /// Horizontal addition of all lanes into a single value.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
  ///   15.0, 16.0,
  /// ]);
  /// let sum = a.reduce_add();
  /// assert_eq!(sum, 136.0);
  /// ```
  #[inline]
  #[must_use]
  pub fn reduce_add(self) -> f32 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // For AVX-512, we can use horizontal add operations
        // First, extract two 256-bit halves and add them
        let hi_half = extract_m256_from_m512::<1>(self.avx512);
        let lo_half = extract_m256_from_m512::<0>(self.avx512);
        let sum_half = add_m256(lo_half, hi_half);

        // Now use f32x8's reduce_add on the sum
        let f32x8_sum = f32x8 { avx: sum_half };
        f32x8_sum.reduce_add()
      } else {
        self.a.reduce_add() + self.b.reduce_add()
      }
    }
  }

  /// Lanewise check for infinity values.
  ///
  /// Returns a mask where each lane is all 1s if the corresponding input lane
  /// is positive or negative infinity, and all 0s otherwise.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   f32::INFINITY,
  ///   1.0,
  ///   f32::NEG_INFINITY,
  ///   0.0,
  ///   f32::INFINITY,
  ///   2.0,
  ///   f32::NEG_INFINITY,
  ///   3.0,
  ///   f32::INFINITY,
  ///   4.0,
  ///   f32::NEG_INFINITY,
  ///   5.0,
  ///   f32::INFINITY,
  ///   6.0,
  ///   f32::NEG_INFINITY,
  ///   7.0,
  /// ]);
  /// let result = a.is_inf();
  /// // Check that infinities are detected as true, finite values as false
  /// assert!(result.any());
  /// ```
  #[inline]
  #[must_use]
  pub fn is_inf(self) -> Self {
    let shifted_inf = u32x16::splat(0xFF000000);
    let u: u32x16 = cast(self);
    let shift_u = u << 1_u32;
    let out = (shift_u).cmp_eq(shifted_inf);
    cast(out)
  }

  /// Flip the sign of `self` lanes based on the sign of `signs` lanes.
  ///
  /// For each lane, if `signs` lane is negative, flips the sign of `self` lane.
  /// Otherwise keeps `self` lane unchanged.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let a = f32x16::from([
  ///   1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0,
  ///   -14.0, 15.0, -16.0,
  /// ]);
  /// let signs = f32x16::from([
  ///   1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
  ///   -1.0, 1.0, -1.0,
  /// ]);
  /// let result = a.flip_signs(signs);
  /// let expected = f32x16::from([
  ///   1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
  ///   15.0, 16.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn flip_signs(self, signs: Self) -> Self {
    self ^ (signs & Self::from(-0.0))
  }

  /// Copy the sign from the `sign` lanes to the `self` lanes.
  ///
  /// Each lane uses the magnitude from `self` and the sign from `sign`.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let magnitude = f32x16::from([
  ///   1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
  ///   15.0, 16.0,
  /// ]);
  /// let signs = f32x16::from([
  ///   1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
  ///   -1.0, 1.0, -1.0,
  /// ]);
  /// let result = magnitude.copysign(signs);
  /// let expected = f32x16::from([
  ///   1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0,
  ///   -14.0, 15.0, -16.0,
  /// ]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn copysign(self, sign: Self) -> Self {
    let magnitude_mask = Self::from(f32::from_bits(u32::MAX >> 1));
    (self & magnitude_mask) | (sign & Self::from(-0.0))
  }

  /// Convert radians to degrees.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// use core::f32::consts::PI;
  /// let radians = f32x16::from([
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  /// ]);
  /// let degrees = radians.to_degrees();
  /// let expected = f32x16::from([
  ///   0.0, 90.0, 180.0, 360.0, 0.0, 90.0, 180.0, 360.0, 0.0, 90.0, 180.0, 360.0,
  ///   0.0, 90.0, 180.0, 360.0,
  /// ]);
  /// assert_eq!(degrees, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn to_degrees(self) -> Self {
    const RAD_TO_DEG_RATIO: f32 = 180.0_f32 / core::f32::consts::PI;
    self * Self::splat(RAD_TO_DEG_RATIO)
  }

  /// Convert degrees to radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// use core::f32::consts::PI;
  /// let degrees = f32x16::from([
  ///   0.0, 90.0, 180.0, 360.0, 0.0, 90.0, 180.0, 360.0, 0.0, 90.0, 180.0, 360.0,
  ///   0.0, 90.0, 180.0, 360.0,
  /// ]);
  /// let radians = degrees.to_radians();
  /// let expected = f32x16::from([
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  ///   0.0,
  ///   PI / 2.0,
  ///   PI,
  ///   2.0 * PI,
  /// ]);
  /// // Use approximate comparison due to floating point precision
  /// let diff: [f32; 16] = (radians - expected).abs().to_array();
  /// for &d in &diff {
  ///   assert!(d < 0.000001);
  /// }
  /// ```
  #[inline]
  #[must_use]
  pub fn to_radians(self) -> Self {
    const DEG_TO_RAD_RATIO: f32 = core::f32::consts::PI / 180.0_f32;
    self * Self::splat(DEG_TO_RAD_RATIO)
  }

  /// Lanewise sine and cosine of the input in radians.
  ///
  /// Returns `(sin, cos)` tuple.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let pi = core::f32::consts::PI;
  /// let input = f32x16::from([
  ///   0.0,
  ///   pi / 6.0,
  ///   pi / 4.0,
  ///   pi / 3.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   2.0 * pi,
  ///   0.0,
  ///   pi / 6.0,
  ///   pi / 4.0,
  ///   pi / 3.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   2.0 * pi,
  /// ]);
  /// let (sin_vals, cos_vals) = input.sin_cos();
  ///
  /// // Check a few key values
  /// let sin_arr: [f32; 16] = sin_vals.to_array();
  /// let cos_arr: [f32; 16] = cos_vals.to_array();
  /// assert!((sin_arr[0] - 0.0).abs() < 0.001); // sin(0) = 0
  /// assert!((cos_arr[0] - 1.0).abs() < 0.001); // cos(0) = 1
  /// ```
  #[inline]
  #[must_use]
  pub fn sin_cos(self) -> (Self, Self) {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // For AVX-512, extract to two 256-bit vectors, compute separately, then combine
        let lo_256 = extract_m256_from_m512::<0>(self.avx512);
        let hi_256 = extract_m256_from_m512::<1>(self.avx512);
        let lo_f32x8 = f32x8 { avx: lo_256 };
        let hi_f32x8 = f32x8 { avx: hi_256 };
        let (sin_lo, cos_lo) = lo_f32x8.sin_cos();
        let (sin_hi, cos_hi) = hi_f32x8.sin_cos();
        let sin_512 = cast_m256_to_m512(sin_lo.avx);
        let sin_result = insert_m256_to_m512::<1>(sin_512, sin_hi.avx);
        let cos_512 = cast_m256_to_m512(cos_lo.avx);
        let cos_result = insert_m256_to_m512::<1>(cos_512, cos_hi.avx);
        (Self { avx512: sin_result }, Self { avx512: cos_result })
      } else {
        let (sin_a, cos_a) = self.a.sin_cos();
        let (sin_b, cos_b) = self.b.sin_cos();
        (Self { a: sin_a, b: sin_b }, Self { a: cos_a, b: cos_b })
      }
    }
  }

  /// Lanewise sine of the input in radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let pi = core::f32::consts::PI;
  /// let input = f32x16::from([
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  /// ]);
  /// let result = input.sin();
  /// let arr: [f32; 16] = result.to_array();
  /// assert!((arr[0] - 0.0).abs() < 0.001); // sin(0) = 0
  /// assert!((arr[1] - 1.0).abs() < 0.001); // sin(π/2) = 1
  /// ```
  #[inline]
  #[must_use]
  pub fn sin(self) -> Self {
    let (s, _) = self.sin_cos();
    s
  }

  /// Lanewise cosine of the input in radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let pi = core::f32::consts::PI;
  /// let input = f32x16::from([
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  ///   0.0,
  ///   pi / 2.0,
  ///   pi,
  ///   3.0 * pi / 2.0,
  /// ]);
  /// let result = input.cos();
  /// let arr: [f32; 16] = result.to_array();
  /// assert!((arr[0] - 1.0).abs() < 0.001); // cos(0) = 1
  /// assert!((arr[2] + 1.0).abs() < 0.001); // cos(π) = -1
  /// ```
  #[inline]
  #[must_use]
  pub fn cos(self) -> Self {
    let (_, c) = self.sin_cos();
    c
  }

  /// Lanewise tangent of the input in radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let pi = core::f32::consts::PI;
  /// let input = f32x16::from([
  ///   0.0,
  ///   pi / 4.0,
  ///   -pi / 4.0,
  ///   pi / 6.0,
  ///   0.0,
  ///   pi / 4.0,
  ///   -pi / 4.0,
  ///   pi / 6.0,
  ///   0.0,
  ///   pi / 4.0,
  ///   -pi / 4.0,
  ///   pi / 6.0,
  ///   0.0,
  ///   pi / 4.0,
  ///   -pi / 4.0,
  ///   pi / 6.0,
  /// ]);
  /// let result = input.tan();
  /// let arr: [f32; 16] = result.to_array();
  /// assert!((arr[0] - 0.0).abs() < 0.001); // tan(0) = 0
  /// assert!((arr[1] - 1.0).abs() < 0.001); // tan(π/4) = 1
  /// ```
  #[inline]
  #[must_use]
  pub fn tan(self) -> Self {
    let (s, c) = self.sin_cos();
    s / c
  }

  /// Lanewise arcsine (inverse sine) in radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let input = f32x16::from([
  ///   0.0, 0.5, 1.0, -0.5, -1.0, 0.707, -0.707, 0.866, 0.0, 0.5, 1.0, -0.5, -1.0,
  ///   0.707, -0.707, 0.866,
  /// ]);
  /// let result = input.asin();
  /// let arr: [f32; 16] = result.to_array();
  /// assert!((arr[0] - 0.0).abs() < 0.001); // asin(0) = 0
  /// ```
  #[inline]
  #[must_use]
  pub fn asin(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Extract to two 256-bit vectors, compute separately, then combine
        let lo_256 = extract_m256_from_m512::<0>(self.avx512);
        let hi_256 = extract_m256_from_m512::<1>(self.avx512);
        let lo_f32x8 = f32x8 { avx: lo_256 };
        let hi_f32x8 = f32x8 { avx: hi_256 };
        let asin_lo = lo_f32x8.asin();
        let asin_hi = hi_f32x8.asin();
        let result_512 = cast_m256_to_m512(asin_lo.avx);
        let result = insert_m256_to_m512::<1>(result_512, asin_hi.avx);
        Self { avx512: result }
      } else {
        Self { a: self.a.asin(), b: self.b.asin() }
      }
    }
  }

  /// Lanewise arccosine (inverse cosine) in radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let input = f32x16::from([
  ///   1.0, 0.5, 0.0, -0.5, -1.0, 0.707, -0.707, 0.866, 1.0, 0.5, 0.0, -0.5, -1.0,
  ///   0.707, -0.707, 0.866,
  /// ]);
  /// let result = input.acos();
  /// let arr: [f32; 16] = result.to_array();
  /// assert!((arr[0] - 0.0).abs() < 0.001); // acos(1) = 0
  /// ```
  #[inline]
  #[must_use]
  pub fn acos(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Extract to two 256-bit vectors, compute separately, then combine
        let lo_256 = extract_m256_from_m512::<0>(self.avx512);
        let hi_256 = extract_m256_from_m512::<1>(self.avx512);
        let lo_f32x8 = f32x8 { avx: lo_256 };
        let hi_f32x8 = f32x8 { avx: hi_256 };
        let acos_lo = lo_f32x8.acos();
        let acos_hi = hi_f32x8.acos();
        let result_512 = cast_m256_to_m512(acos_lo.avx);
        let result = insert_m256_to_m512::<1>(result_512, acos_hi.avx);
        Self { avx512: result }
      } else {
        Self { a: self.a.acos(), b: self.b.acos() }
      }
    }
  }

  /// Lanewise arctangent (inverse tangent) in radians.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let input = f32x16::from([
  ///   0.0, 1.0, -1.0, 0.577, -0.577, 1.732, -1.732, 0.0, 0.0, 1.0, -1.0, 0.577,
  ///   -0.577, 1.732, -1.732, 0.0,
  /// ]);
  /// let result = input.atan();
  /// let arr: [f32; 16] = result.to_array();
  /// assert!((arr[0] - 0.0).abs() < 0.001); // atan(0) = 0
  /// ```
  #[inline]
  #[must_use]
  pub fn atan(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Extract to two 256-bit vectors, compute separately, then combine
        let lo_256 = extract_m256_from_m512::<0>(self.avx512);
        let hi_256 = extract_m256_from_m512::<1>(self.avx512);
        let lo_f32x8 = f32x8 { avx: lo_256 };
        let hi_f32x8 = f32x8 { avx: hi_256 };
        let atan_lo = lo_f32x8.atan();
        let atan_hi = hi_f32x8.atan();
        let result_512 = cast_m256_to_m512(atan_lo.avx);
        let result = insert_m256_to_m512::<1>(result_512, atan_hi.avx);
        Self { avx512: result }
      } else {
        Self { a: self.a.atan(), b: self.b.atan() }
      }
    }
  }

  /// Lanewise two-argument arctangent in radians.
  ///
  /// `y.atan2(x)` computes the angle from the positive x-axis to the point (x,
  /// y).
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let y = f32x16::from([
  ///   1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0,
  ///   1.0, -1.0,
  /// ]);
  /// let x = f32x16::from([
  ///   1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
  ///   0.0, 0.0,
  /// ]);
  /// let result = y.atan2(x);
  /// let arr: [f32; 16] = result.to_array();
  /// let pi = core::f32::consts::PI;
  /// assert!((arr[0] - pi / 4.0).abs() < 0.001); // atan2(1, 1) = π/4
  /// ```
  #[inline]
  #[must_use]
  pub fn atan2(self, x: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Extract to two 256-bit vectors, compute separately, then combine
        let lo_256 = extract_m256_from_m512::<0>(self.avx512);
        let hi_256 = extract_m256_from_m512::<1>(self.avx512);
        let x_lo_256 = extract_m256_from_m512::<0>(x.avx512);
        let x_hi_256 = extract_m256_from_m512::<1>(x.avx512);
        let lo_f32x8 = f32x8 { avx: lo_256 };
        let hi_f32x8 = f32x8 { avx: hi_256 };
        let x_lo_f32x8 = f32x8 { avx: x_lo_256 };
        let x_hi_f32x8 = f32x8 { avx: x_hi_256 };
        let atan2_lo = lo_f32x8.atan2(x_lo_f32x8);
        let atan2_hi = hi_f32x8.atan2(x_hi_f32x8);
        let result_512 = cast_m256_to_m512(atan2_lo.avx);
        let result = insert_m256_to_m512::<1>(result_512, atan2_hi.avx);
        Self { avx512: result }
      } else {
        Self { a: self.a.atan2(x.a), b: self.b.atan2(x.b) }
      }
    }
  }

  /// Lanewise arcsine and arccosine of the input.
  ///
  /// Returns `(asin, acos)` tuple.
  ///
  /// # Examples
  /// ```
  /// # use wide::f32x16;
  /// let input = f32x16::from([
  ///   0.0, 0.5, 1.0, -0.5, -1.0, 0.707, -0.707, 0.866, 0.0, 0.5, 1.0, -0.5, -1.0,
  ///   0.707, -0.707, 0.866,
  /// ]);
  /// let (asin_vals, acos_vals) = input.asin_acos();
  /// let asin_arr: [f32; 16] = asin_vals.to_array();
  /// let acos_arr: [f32; 16] = acos_vals.to_array();
  /// assert!((asin_arr[0] - 0.0).abs() < 0.001);
  /// ```
  #[inline]
  #[must_use]
  pub fn asin_acos(self) -> (Self, Self) {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // Extract to two 256-bit vectors, compute separately, then combine
        let lo_256 = extract_m256_from_m512::<0>(self.avx512);
        let hi_256 = extract_m256_from_m512::<1>(self.avx512);
        let lo_f32x8 = f32x8 { avx: lo_256 };
        let hi_f32x8 = f32x8 { avx: hi_256 };
        let (asin_lo, acos_lo) = lo_f32x8.asin_acos();
        let (asin_hi, acos_hi) = hi_f32x8.asin_acos();
        let asin_512 = cast_m256_to_m512(asin_lo.avx);
        let asin_result = insert_m256_to_m512::<1>(asin_512, asin_hi.avx);
        let acos_512 = cast_m256_to_m512(acos_lo.avx);
        let acos_result = insert_m256_to_m512::<1>(acos_512, acos_hi.avx);
        (Self { avx512: asin_result }, Self { avx512: acos_result })
      } else {
        let (asin_a, acos_a) = self.a.asin_acos();
        let (asin_b, acos_b) = self.b.asin_acos();
        (Self { a: asin_a, b: asin_b }, Self { a: acos_a, b: acos_b })
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
