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

macro_rules! const_f64_as_f64x8 {
  ($i:ident, $f:expr) => {
    #[allow(non_upper_case_globals)]
    pub const $i: f64x8 = f64x8::new([$f; 8]);
  };
}

unsafe impl Zeroable for f64x8 {}
unsafe impl Pod for f64x8 {}

impl AlignTo for f64x8 {
  type Elem = f64;
}

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

impl Neg for f64x8 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: bitxor_m512d(self.avx512, Self::splat(-0.0).avx512) }
      } else {
        Self {
          a : self.a.neg(),
          b : self.b.neg(),
        }
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
  fn simd_eq(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(EqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_eq(rhs.a),
          b : self.b.simd_eq(rhs.b),
        }
      }
    }
  }
}

impl CmpGt for f64x8 {
  type Output = Self;
  #[inline]
  fn simd_gt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(GreaterThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_gt(rhs.a),
          b : self.b.simd_gt(rhs.b),
        }
      }
    }
  }
}

impl CmpGe for f64x8 {
  type Output = Self;
  #[inline]
  fn simd_ge(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(GreaterEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ge(rhs.a),
          b : self.b.simd_ge(rhs.b),
        }
      }
    }
  }
}

impl CmpLt for f64x8 {
  type Output = Self;
  #[inline]
  fn simd_lt(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(LessThanOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_lt(rhs.a),
          b : self.b.simd_lt(rhs.b),
        }
      }
    }
  }
}

impl CmpLe for f64x8 {
  type Output = Self;
  #[inline]
  fn simd_le(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(LessEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_le(rhs.a),
          b : self.b.simd_le(rhs.b),
        }
      }
    }
  }
}

impl CmpNe for f64x8 {
  type Output = Self;
  #[inline]
  fn simd_ne(self, rhs: Self) -> Self::Output {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: cmp_op_mask_m512d::<{cmp_op!(NotEqualOrdered)}>(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.simd_ne(rhs.a),
          b : self.b.simd_ne(rhs.b),
        }
      }
    }
  }
}

impl f64x8 {
  const_f64_as_f64x8!(ONE, 1.0);
  const_f64_as_f64x8!(ZERO, 0.0);
  const_f64_as_f64x8!(HALF, 0.5);
  const_f64_as_f64x8!(E, core::f64::consts::E);
  const_f64_as_f64x8!(FRAC_1_PI, core::f64::consts::FRAC_1_PI);
  const_f64_as_f64x8!(FRAC_2_PI, core::f64::consts::FRAC_2_PI);
  const_f64_as_f64x8!(FRAC_2_SQRT_PI, core::f64::consts::FRAC_2_SQRT_PI);
  const_f64_as_f64x8!(FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2);
  const_f64_as_f64x8!(FRAC_PI_2, core::f64::consts::FRAC_PI_2);
  const_f64_as_f64x8!(FRAC_PI_3, core::f64::consts::FRAC_PI_3);
  const_f64_as_f64x8!(FRAC_PI_4, core::f64::consts::FRAC_PI_4);
  const_f64_as_f64x8!(FRAC_PI_6, core::f64::consts::FRAC_PI_6);
  const_f64_as_f64x8!(FRAC_PI_8, core::f64::consts::FRAC_PI_8);
  const_f64_as_f64x8!(LN_2, core::f64::consts::LN_2);
  const_f64_as_f64x8!(LN_10, core::f64::consts::LN_10);
  const_f64_as_f64x8!(LOG2_E, core::f64::consts::LOG2_E);
  const_f64_as_f64x8!(LOG10_E, core::f64::consts::LOG10_E);
  const_f64_as_f64x8!(LOG10_2, core::f64::consts::LOG10_2);
  const_f64_as_f64x8!(LOG2_10, core::f64::consts::LOG2_10);
  const_f64_as_f64x8!(PI, core::f64::consts::PI);
  const_f64_as_f64x8!(SQRT_2, core::f64::consts::SQRT_2);
  const_f64_as_f64x8!(TAU, core::f64::consts::TAU);

  #[inline]
  #[must_use]
  pub const fn new(array: [f64; 8]) -> Self {
    unsafe { core::mem::transmute(array) }
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
  pub fn floor(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: round_m512d::<{round_op!(NegInf)}>(self.avx512) }
      } else {
        Self {
          a : self.a.floor(),
          b : self.b.floor(),
        }
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn ceil(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: round_m512d::<{round_op!(PosInf)}>(self.avx512) }
      } else {
        Self {
          a : self.a.ceil(),
          b : self.b.ceil(),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn fast_max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: max_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.fast_max(rhs.a),
          b : self.b.fast_max(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn max(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        rhs.is_nan().blend(self, Self { avx512: max_m512d(self.avx512, rhs.avx512) })
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
  pub fn fast_min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: min_m512d(self.avx512, rhs.avx512) }
      } else {
        Self {
          a : self.a.fast_min(rhs.a),
          b : self.b.fast_min(rhs.b),
        }
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn min(self, rhs: Self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        rhs.is_nan().blend(self, Self { avx512: min_m512d(self.avx512, rhs.avx512) })
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
    let shifted_exp_mask = u64x8::splat(0xFFE0000000000000);
    let u: u64x8 = cast(self);
    let shift_u = u << 1_u64;
    let out = !(shift_u & shifted_exp_mask).simd_eq(shifted_exp_mask);
    cast(out)
  }

  #[inline]
  #[must_use]
  pub fn is_inf(self) -> Self {
    let shifted_inf = u64x8::from(0xFFE0000000000000);
    let u: u64x8 = cast(self);
    let shift_u = u << 1_u64;
    let out = (shift_u).simd_eq(shifted_inf);
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
    let rounded: [f64; 8] = cast(self.round());
    cast([
      rounded[0] as i64,
      rounded[1] as i64,
      rounded[2] as i64,
      rounded[3] as i64,
      rounded[4] as i64,
      rounded[5] as i64,
      rounded[6] as i64,
      rounded[7] as i64,
    ])
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
  /// - Other platforms: Delegates to [`f64x4`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f64x8;
  /// let a = f64x8::from([1.0; 8]);
  /// let b = f64x8::from([2.0; 8]);
  /// let c = f64x8::from([10.0; 8]);
  ///
  /// let result = a.mul_add(b, c);
  ///
  /// let expected = f64x8::from([12.0; 8]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn mul_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_add_m512d(self.avx512, m.avx512, a.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        (self * m) + a
      } else {
        Self {
          a : self.a.mul_add(m.a, a.a),
          b : self.b.mul_add(m.b, a.b),
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
  /// - Other platforms: Delegates to [`f64x4`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f64x8;
  /// let a = f64x8::from([10.0; 8]);
  /// let b = f64x8::from([3.0; 8]);
  /// let c = f64x8::from([5.0; 8]);
  ///
  /// let result = a.mul_sub(b, c);
  ///
  /// let expected = f64x8::from([25.0; 8]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn mul_sub(self, m: Self, s: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_sub_m512d(self.avx512, m.avx512, s.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        (self * m) - s
      } else {
        Self {
          a : self.a.mul_sub(m.a, s.a),
          b : self.b.mul_sub(m.b, s.b),
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
  /// - Other platforms: Delegates to [`f64x4`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f64x8;
  /// let a = f64x8::from([4.0; 8]);
  /// let b = f64x8::from([2.0; 8]);
  /// let c = f64x8::from([10.0; 8]);
  ///
  /// let result = a.mul_neg_add(b, c);
  ///
  /// let expected = f64x8::from([2.0; 8]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn mul_neg_add(self, m: Self, a: Self) -> Self {
    pick! {
      if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
        Self { avx512: fused_mul_neg_add_m512d(self.avx512, m.avx512, a.avx512) }
      } else if #[cfg(target_feature="avx512f")] {
        // still want to use 512 bit ops
        a - (self * m)
      } else {
        Self {
          a : self.a.mul_neg_add(m.a, a.a),
          b : self.b.mul_neg_add(m.b, a.b),
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
  /// - Other platforms: Delegates to [`f64x4`] (inherits its FMA behavior)
  ///
  /// # Examples
  /// ```
  /// # use wide::f64x8;
  /// let a = f64x8::from([4.0; 8]);
  /// let b = f64x8::from([2.0; 8]);
  /// let c = f64x8::from([1.0; 8]);
  ///
  /// let result = a.mul_neg_sub(b, c);
  ///
  /// let expected = f64x8::from([-9.0; 8]);
  /// assert_eq!(result, expected);
  /// ```
  #[inline]
  #[must_use]
  pub fn mul_neg_sub(self, m: Self, s: Self) -> Self {
    pick! {
       if #[cfg(all(target_feature="avx512f",target_feature="fma"))] {
         Self { avx512: fused_mul_neg_sub_m512d(self.avx512, m.avx512, s.avx512) }
        } else if #[cfg(target_feature="avx512f")] {
          // still want to use 512 bit ops
          -(self * m) - s
        } else {
         Self {
           a : self.a.mul_neg_sub(m.a, s.a),
           b : self.b.mul_neg_sub(m.b, s.b),
         }
       }
    }
  }

  #[inline]
  #[must_use]
  pub fn flip_signs(self, signs: Self) -> Self {
    self ^ (signs & Self::from(-0.0))
  }

  #[inline]
  #[must_use]
  pub fn copysign(self, sign: Self) -> Self {
    let magnitude_mask = Self::from(f64::from_bits(u64::MAX >> 1));
    (self & magnitude_mask) | (sign & Self::from(-0.0))
  }

  #[inline]
  pub fn asin_acos(self) -> (Self, Self) {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x8!(R4asin, 2.967721961301243206100E-3);
    const_f64_as_f64x8!(R3asin, -5.634242780008963776856E-1);
    const_f64_as_f64x8!(R2asin, 6.968710824104713396794E0);
    const_f64_as_f64x8!(R1asin, -2.556901049652824852289E1);
    const_f64_as_f64x8!(R0asin, 2.853665548261061424989E1);

    const_f64_as_f64x8!(S3asin, -2.194779531642920639778E1);
    const_f64_as_f64x8!(S2asin, 1.470656354026814941758E2);
    const_f64_as_f64x8!(S1asin, -3.838770957603691357202E2);
    const_f64_as_f64x8!(S0asin, 3.424398657913078477438E2);

    const_f64_as_f64x8!(P5asin, 4.253011369004428248960E-3);
    const_f64_as_f64x8!(P4asin, -6.019598008014123785661E-1);
    const_f64_as_f64x8!(P3asin, 5.444622390564711410273E0);
    const_f64_as_f64x8!(P2asin, -1.626247967210700244449E1);
    const_f64_as_f64x8!(P1asin, 1.956261983317594739197E1);
    const_f64_as_f64x8!(P0asin, -8.198089802484824371615E0);

    const_f64_as_f64x8!(Q4asin, -1.474091372988853791896E1);
    const_f64_as_f64x8!(Q3asin, 7.049610280856842141659E1);
    const_f64_as_f64x8!(Q2asin, -1.471791292232726029859E2);
    const_f64_as_f64x8!(Q1asin, 1.395105614657485689735E2);
    const_f64_as_f64x8!(Q0asin, -4.918853881490881290097E1);

    let xa = self.abs();

    let big = xa.simd_ge(f64x8::splat(0.625));

    let x1 = big.blend(f64x8::splat(1.0) - xa, xa * xa);

    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x2 * x2;
    let x5 = x4 * x1;

    let do_big = big.any();
    let do_small = !big.all();

    let mut rx = f64x8::default();
    let mut sx = f64x8::default();
    let mut px = f64x8::default();
    let mut qx = f64x8::default();

    if do_big {
      rx = x3.mul_add(R3asin, x2 * R2asin)
        + x4.mul_add(R4asin, x1.mul_add(R1asin, R0asin));
      sx =
        x3.mul_add(S3asin, x4) + x2.mul_add(S2asin, x1.mul_add(S1asin, S0asin));
    }

    if do_small {
      px = x3.mul_add(P3asin, P0asin)
        + x4.mul_add(P4asin, x1 * P1asin)
        + x5.mul_add(P5asin, x2 * P2asin);
      qx = x4.mul_add(Q4asin, x5)
        + x3.mul_add(Q3asin, x1 * Q1asin)
        + x2.mul_add(Q2asin, Q0asin);
    };

    let vx = big.blend(rx, px);
    let wx = big.blend(sx, qx);

    let y1 = vx / wx * x1;

    let mut z1 = f64x8::default();
    let mut z2 = f64x8::default();
    if do_big {
      let xb = (x1 + x1).sqrt();
      z1 = xb.mul_add(y1, xb);
    }

    if do_small {
      z2 = xa.mul_add(y1, xa);
    }

    // asin
    let z3 = f64x8::FRAC_PI_2 - z1;
    let asin = big.blend(z3, z2);
    let asin = asin.flip_signs(self);

    // acos
    let z3 = self.simd_lt(f64x8::ZERO).blend(f64x8::PI - z1, z1);
    let z4 = f64x8::FRAC_PI_2 - z2.flip_signs(self);
    let acos = big.blend(z3, z4);

    (asin, acos)
  }

  #[inline]
  pub fn acos(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x8!(R4asin, 2.967721961301243206100E-3);
    const_f64_as_f64x8!(R3asin, -5.634242780008963776856E-1);
    const_f64_as_f64x8!(R2asin, 6.968710824104713396794E0);
    const_f64_as_f64x8!(R1asin, -2.556901049652824852289E1);
    const_f64_as_f64x8!(R0asin, 2.853665548261061424989E1);

    const_f64_as_f64x8!(S3asin, -2.194779531642920639778E1);
    const_f64_as_f64x8!(S2asin, 1.470656354026814941758E2);
    const_f64_as_f64x8!(S1asin, -3.838770957603691357202E2);
    const_f64_as_f64x8!(S0asin, 3.424398657913078477438E2);

    const_f64_as_f64x8!(P5asin, 4.253011369004428248960E-3);
    const_f64_as_f64x8!(P4asin, -6.019598008014123785661E-1);
    const_f64_as_f64x8!(P3asin, 5.444622390564711410273E0);
    const_f64_as_f64x8!(P2asin, -1.626247967210700244449E1);
    const_f64_as_f64x8!(P1asin, 1.956261983317594739197E1);
    const_f64_as_f64x8!(P0asin, -8.198089802484824371615E0);

    const_f64_as_f64x8!(Q4asin, -1.474091372988853791896E1);
    const_f64_as_f64x8!(Q3asin, 7.049610280856842141659E1);
    const_f64_as_f64x8!(Q2asin, -1.471791292232726029859E2);
    const_f64_as_f64x8!(Q1asin, 1.395105614657485689735E2);
    const_f64_as_f64x8!(Q0asin, -4.918853881490881290097E1);

    let xa = self.abs();

    let big = xa.simd_ge(f64x8::splat(0.625));

    let x1 = big.blend(f64x8::splat(1.0) - xa, xa * xa);

    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x2 * x2;
    let x5 = x4 * x1;

    let do_big = big.any();
    let do_small = !big.all();

    let mut rx = f64x8::default();
    let mut sx = f64x8::default();
    let mut px = f64x8::default();
    let mut qx = f64x8::default();

    if do_big {
      rx = x3.mul_add(R3asin, x2 * R2asin)
        + x4.mul_add(R4asin, x1.mul_add(R1asin, R0asin));
      sx =
        x3.mul_add(S3asin, x4) + x2.mul_add(S2asin, x1.mul_add(S1asin, S0asin));
    }
    if do_small {
      px = x3.mul_add(P3asin, P0asin)
        + x4.mul_add(P4asin, x1 * P1asin)
        + x5.mul_add(P5asin, x2 * P2asin);
      qx = x4.mul_add(Q4asin, x5)
        + x3.mul_add(Q3asin, x1 * Q1asin)
        + x2.mul_add(Q2asin, Q0asin);
    };

    let vx = big.blend(rx, px);
    let wx = big.blend(sx, qx);

    let y1 = vx / wx * x1;

    let mut z1 = f64x8::default();
    let mut z2 = f64x8::default();
    if do_big {
      let xb = (x1 + x1).sqrt();
      z1 = xb.mul_add(y1, xb);
    }

    if do_small {
      z2 = xa.mul_add(y1, xa);
    }

    // acos
    let z3 = self.simd_lt(f64x8::ZERO).blend(f64x8::PI - z1, z1);
    let z4 = f64x8::FRAC_PI_2 - z2.flip_signs(self);
    let acos = big.blend(z3, z4);

    acos
  }
  #[inline]
  #[must_use]
  pub fn asin(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x8!(R4asin, 2.967721961301243206100E-3);
    const_f64_as_f64x8!(R3asin, -5.634242780008963776856E-1);
    const_f64_as_f64x8!(R2asin, 6.968710824104713396794E0);
    const_f64_as_f64x8!(R1asin, -2.556901049652824852289E1);
    const_f64_as_f64x8!(R0asin, 2.853665548261061424989E1);

    const_f64_as_f64x8!(S3asin, -2.194779531642920639778E1);
    const_f64_as_f64x8!(S2asin, 1.470656354026814941758E2);
    const_f64_as_f64x8!(S1asin, -3.838770957603691357202E2);
    const_f64_as_f64x8!(S0asin, 3.424398657913078477438E2);

    const_f64_as_f64x8!(P5asin, 4.253011369004428248960E-3);
    const_f64_as_f64x8!(P4asin, -6.019598008014123785661E-1);
    const_f64_as_f64x8!(P3asin, 5.444622390564711410273E0);
    const_f64_as_f64x8!(P2asin, -1.626247967210700244449E1);
    const_f64_as_f64x8!(P1asin, 1.956261983317594739197E1);
    const_f64_as_f64x8!(P0asin, -8.198089802484824371615E0);

    const_f64_as_f64x8!(Q4asin, -1.474091372988853791896E1);
    const_f64_as_f64x8!(Q3asin, 7.049610280856842141659E1);
    const_f64_as_f64x8!(Q2asin, -1.471791292232726029859E2);
    const_f64_as_f64x8!(Q1asin, 1.395105614657485689735E2);
    const_f64_as_f64x8!(Q0asin, -4.918853881490881290097E1);

    let xa = self.abs();

    let big = xa.simd_ge(f64x8::splat(0.625));

    let x1 = big.blend(f64x8::splat(1.0) - xa, xa * xa);

    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x2 * x2;
    let x5 = x4 * x1;

    let do_big = big.any();
    let do_small = !big.all();

    let mut rx = f64x8::default();
    let mut sx = f64x8::default();
    let mut px = f64x8::default();
    let mut qx = f64x8::default();

    if do_big {
      rx = x3.mul_add(R3asin, x2 * R2asin)
        + x4.mul_add(R4asin, x1.mul_add(R1asin, R0asin));
      sx =
        x3.mul_add(S3asin, x4) + x2.mul_add(S2asin, x1.mul_add(S1asin, S0asin));
    }
    if do_small {
      px = x3.mul_add(P3asin, P0asin)
        + x4.mul_add(P4asin, x1 * P1asin)
        + x5.mul_add(P5asin, x2 * P2asin);
      qx = x4.mul_add(Q4asin, x5)
        + x3.mul_add(Q3asin, x1 * Q1asin)
        + x2.mul_add(Q2asin, Q0asin);
    };

    let vx = big.blend(rx, px);
    let wx = big.blend(sx, qx);

    let y1 = vx / wx * x1;

    let mut z1 = f64x8::default();
    let mut z2 = f64x8::default();
    if do_big {
      let xb = (x1 + x1).sqrt();
      z1 = xb.mul_add(y1, xb);
    }

    if do_small {
      z2 = xa.mul_add(y1, xa);
    }

    // asin
    let z3 = f64x8::FRAC_PI_2 - z1;
    let asin = big.blend(z3, z2);
    let asin = asin.flip_signs(self);

    asin
  }

  #[inline]
  pub fn atan(self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x8!(MORE_BITS, 6.123233995736765886130E-17);
    const_f64_as_f64x8!(MORE_BITS_O2, 6.123233995736765886130E-17 * 0.5);
    const_f64_as_f64x8!(T3PO8, core::f64::consts::SQRT_2 + 1.0);

    const_f64_as_f64x8!(P4atan, -8.750608600031904122785E-1);
    const_f64_as_f64x8!(P3atan, -1.615753718733365076637E1);
    const_f64_as_f64x8!(P2atan, -7.500855792314704667340E1);
    const_f64_as_f64x8!(P1atan, -1.228866684490136173410E2);
    const_f64_as_f64x8!(P0atan, -6.485021904942025371773E1);

    const_f64_as_f64x8!(Q4atan, 2.485846490142306297962E1);
    const_f64_as_f64x8!(Q3atan, 1.650270098316988542046E2);
    const_f64_as_f64x8!(Q2atan, 4.328810604912902668951E2);
    const_f64_as_f64x8!(Q1atan, 4.853903996359136964868E2);
    const_f64_as_f64x8!(Q0atan, 1.945506571482613964425E2);

    let t = self.abs();

    // small:  t < 0.66
    // medium: t <= t <= 2.4142 (1+sqrt(2))
    // big:    t > 2.4142
    let notbig = t.simd_le(T3PO8);
    let notsmal = t.simd_ge(Self::splat(0.66));

    let mut s = notbig.blend(Self::FRAC_PI_4, Self::FRAC_PI_2);
    s = notsmal & s;
    let mut fac = notbig.blend(MORE_BITS_O2, MORE_BITS);
    fac = notsmal & fac;

    // small:  z = t / 1.0;
    // medium: z = (t-1.0) / (t+1.0);
    // big:    z = -1.0 / t;
    let mut a = notbig & t;
    a = notsmal.blend(a - Self::ONE, a);
    let mut b = notbig & Self::ONE;
    b = notsmal.blend(b + t, b);
    let z = a / b;

    let zz = z * z;

    let px = polynomial_4!(zz, P0atan, P1atan, P2atan, P3atan, P4atan);
    let qx = polynomial_5n!(zz, Q0atan, Q1atan, Q2atan, Q3atan, Q4atan);

    let mut re = (px / qx).mul_add(z * zz, z);
    re += s + fac;

    // get sign bit
    re = (self.sign_bit()).blend(-re, re);

    re
  }

  #[inline]
  pub fn atan2(self, x: Self) -> Self {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
    const_f64_as_f64x8!(MORE_BITS, 6.123233995736765886130E-17);
    const_f64_as_f64x8!(MORE_BITS_O2, 6.123233995736765886130E-17 * 0.5);
    const_f64_as_f64x8!(T3PO8, core::f64::consts::SQRT_2 + 1.0);

    const_f64_as_f64x8!(P4atan, -8.750608600031904122785E-1);
    const_f64_as_f64x8!(P3atan, -1.615753718733365076637E1);
    const_f64_as_f64x8!(P2atan, -7.500855792314704667340E1);
    const_f64_as_f64x8!(P1atan, -1.228866684490136173410E2);
    const_f64_as_f64x8!(P0atan, -6.485021904942025371773E1);

    const_f64_as_f64x8!(Q4atan, 2.485846490142306297962E1);
    const_f64_as_f64x8!(Q3atan, 1.650270098316988542046E2);
    const_f64_as_f64x8!(Q2atan, 4.328810604912902668951E2);
    const_f64_as_f64x8!(Q1atan, 4.853903996359136964868E2);
    const_f64_as_f64x8!(Q0atan, 1.945506571482613964425E2);

    let y = self;

    // move in first octant
    let x1 = x.abs();
    let y1 = y.abs();
    let swapxy = y1.simd_gt(x1);
    // swap x and y if y1 > x1
    let mut x2 = swapxy.blend(y1, x1);
    let mut y2 = swapxy.blend(x1, y1);

    // check for special case: x and y are both +/- INF
    let both_infinite = x.is_inf() & y.is_inf();
    if both_infinite.any() {
      let minus_one = -Self::ONE;
      x2 = both_infinite.blend(x2 & minus_one, x2);
      y2 = both_infinite.blend(y2 & minus_one, y2);
    }

    // x = y = 0 gives NAN here
    let t = y2 / x2;

    // small:  t < 0.66
    // medium: t <= t <= 2.4142 (1+sqrt(2))
    // big:    t > 2.4142
    let notbig = t.simd_le(T3PO8);
    let notsmal = t.simd_ge(Self::splat(0.66));

    let mut s = notbig.blend(Self::FRAC_PI_4, Self::FRAC_PI_2);
    s = notsmal & s;
    let mut fac = notbig.blend(MORE_BITS_O2, MORE_BITS);
    fac = notsmal & fac;

    // small:  z = t / 1.0;
    // medium: z = (t-1.0) / (t+1.0);
    // big:    z = -1.0 / t;
    let mut a = notbig & t;
    a = notsmal.blend(a - Self::ONE, a);
    let mut b = notbig & Self::ONE;
    b = notsmal.blend(b + t, b);
    let z = a / b;

    let zz = z * z;

    let px = polynomial_4!(zz, P0atan, P1atan, P2atan, P3atan, P4atan);
    let qx = polynomial_5n!(zz, Q0atan, Q1atan, Q2atan, Q3atan, Q4atan);

    let mut re = (px / qx).mul_add(z * zz, z);
    re += s + fac;

    // move back in place
    re = swapxy.blend(Self::FRAC_PI_2 - re, re);
    re = ((x | y).simd_eq(Self::ZERO)).blend(Self::ZERO, re);
    re = (x.sign_bit()).blend(Self::PI - re, re);

    // get sign bit
    re = (y.sign_bit()).blend(-re, re);

    re
  }

  #[inline]
  #[must_use]
  pub fn sin_cos(self) -> (Self, Self) {
    // Based on the Agner Fog "vector class library":
    // https://github.com/vectorclass/version2/blob/master/vectormath_trig.h

    const_f64_as_f64x8!(P0sin, -1.66666666666666307295E-1);
    const_f64_as_f64x8!(P1sin, 8.33333333332211858878E-3);
    const_f64_as_f64x8!(P2sin, -1.98412698295895385996E-4);
    const_f64_as_f64x8!(P3sin, 2.75573136213857245213E-6);
    const_f64_as_f64x8!(P4sin, -2.50507477628578072866E-8);
    const_f64_as_f64x8!(P5sin, 1.58962301576546568060E-10);

    const_f64_as_f64x8!(P0cos, 4.16666666666665929218E-2);
    const_f64_as_f64x8!(P1cos, -1.38888888888730564116E-3);
    const_f64_as_f64x8!(P2cos, 2.48015872888517045348E-5);
    const_f64_as_f64x8!(P3cos, -2.75573141792967388112E-7);
    const_f64_as_f64x8!(P4cos, 2.08757008419747316778E-9);
    const_f64_as_f64x8!(P5cos, -1.13585365213876817300E-11);

    const_f64_as_f64x8!(DP1, 7.853981554508209228515625E-1 * 2.);
    const_f64_as_f64x8!(DP2, 7.94662735614792836714E-9 * 2.);
    const_f64_as_f64x8!(DP3, 3.06161699786838294307E-17 * 2.);

    const_f64_as_f64x8!(TWO_OVER_PI, 2.0 / core::f64::consts::PI);

    let xa = self.abs();

    let y = (xa * TWO_OVER_PI).round();
    let q = y.round_int();

    let x = y.mul_neg_add(DP3, y.mul_neg_add(DP2, y.mul_neg_add(DP1, xa)));

    let x2 = x * x;
    let mut s = polynomial_5!(x2, P0sin, P1sin, P2sin, P3sin, P4sin, P5sin);
    let mut c = polynomial_5!(x2, P0cos, P1cos, P2cos, P3cos, P4cos, P5cos);
    s = (x * x2).mul_add(s, x);
    c =
      (x2 * x2).mul_add(c, x2.mul_neg_add(f64x8::from(0.5), f64x8::from(1.0)));

    let swap = !((q & i64x8::from(1)).simd_eq(i64x8::from(0)));

    let mut overflow: f64x8 = cast(q.simd_gt(i64x8::from(0x80000000000000)));
    overflow &= xa.is_finite();
    s = overflow.blend(f64x8::from(0.0), s);
    c = overflow.blend(f64x8::from(1.0), c);

    // calc sin
    let mut sin1 = cast::<_, f64x8>(swap).blend(c, s);
    let sign_sin: i64x8 = (q << 62) ^ cast::<_, i64x8>(self);
    sin1 = sin1.flip_signs(cast(sign_sin));

    // calc cos
    let mut cos1 = cast::<_, f64x8>(swap).blend(s, c);
    let sign_cos: i64x8 = ((q + i64x8::from(1)) & i64x8::from(2)) << 62;
    cos1 ^= cast::<_, f64x8>(sign_cos);

    (sin1, cos1)
  }
  #[inline]
  #[must_use]
  pub fn sin(self) -> Self {
    let (s, _) = self.sin_cos();
    s
  }
  #[inline]
  #[must_use]
  pub fn cos(self) -> Self {
    let (_, c) = self.sin_cos();
    c
  }
  #[inline]
  #[must_use]
  pub fn tan(self) -> Self {
    let (s, c) = self.sin_cos();
    s / c
  }
  #[inline]
  #[must_use]
  pub fn to_degrees(self) -> Self {
    const_f64_as_f64x8!(RAD_TO_DEG_RATIO, 180.0_f64 / core::f64::consts::PI);
    self * RAD_TO_DEG_RATIO
  }
  #[inline]
  #[must_use]
  pub fn to_radians(self) -> Self {
    const_f64_as_f64x8!(DEG_TO_RAD_RATIO, core::f64::consts::PI / 180.0_f64);
    self * DEG_TO_RAD_RATIO
  }
  #[inline]
  #[must_use]
  pub fn sqrt(self) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: sqrt_m512d(self.avx512) }
      } else {
        Self {
          a : self.a.sqrt(),
          b : self.b.sqrt(),
        }
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn to_bitmask(self) -> u32 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi64_mask_m512d(self.avx512) as u32
      } else {
        (self.b.to_bitmask() << 4) | self.a.to_bitmask()
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn any(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi64_mask_m512d(self.avx512) != 0
      } else {
        self.a.any() || self.b.any()
      }
    }
  }
  #[inline]
  #[must_use]
  pub fn all(self) -> bool {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        movepi64_mask_m512d(self.avx512) == 0b11111111
      } else {
        self.a.all() && self.b.all()
      }
    }
  }

  #[inline]
  #[must_use]
  pub fn none(self) -> bool {
    !self.any()
  }

  #[inline]
  fn vm_pow2n(self) -> Self {
    const_f64_as_f64x8!(pow2_52, 4503599627370496.0);
    const_f64_as_f64x8!(bias, 1023.0);
    let a = self + (bias + pow2_52);
    let c = cast::<_, i64x8>(a) << 52;
    cast::<_, f64x8>(c)
  }

  /// Calculate the exponent of a packed `f64x8`
  #[inline]
  #[must_use]
  pub fn exp(self) -> Self {
    const_f64_as_f64x8!(P2, 1.0 / 2.0);
    const_f64_as_f64x8!(P3, 1.0 / 6.0);
    const_f64_as_f64x8!(P4, 1. / 24.);
    const_f64_as_f64x8!(P5, 1. / 120.);
    const_f64_as_f64x8!(P6, 1. / 720.);
    const_f64_as_f64x8!(P7, 1. / 5040.);
    const_f64_as_f64x8!(P8, 1. / 40320.);
    const_f64_as_f64x8!(P9, 1. / 362880.);
    const_f64_as_f64x8!(P10, 1. / 3628800.);
    const_f64_as_f64x8!(P11, 1. / 39916800.);
    const_f64_as_f64x8!(P12, 1. / 479001600.);
    const_f64_as_f64x8!(P13, 1. / 6227020800.);
    const_f64_as_f64x8!(LN2D_HI, 0.693145751953125);
    const_f64_as_f64x8!(LN2D_LO, 1.42860682030941723212E-6);
    let max_x = f64x8::from(708.39);
    let r = (self * Self::LOG2_E).round();
    let x = r.mul_neg_add(LN2D_HI, self);
    let x = r.mul_neg_add(LN2D_LO, x);
    let z =
      polynomial_13!(x, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13);
    let n2 = Self::vm_pow2n(r);
    let z = (z + Self::ONE) * n2;
    // check for overflow
    let in_range = self.abs().simd_lt(max_x);
    let in_range = in_range & self.is_finite();
    in_range.blend(z, Self::ZERO)
  }

  #[inline]
  fn exponent(self) -> f64x8 {
    const_f64_as_f64x8!(pow2_52, 4503599627370496.0);
    const_f64_as_f64x8!(bias, 1023.0);
    let a = cast::<_, u64x8>(self);
    let b = a >> 52;
    let c = b | cast::<_, u64x8>(pow2_52);
    let d = cast::<_, f64x8>(c);
    let e = d - (pow2_52 + bias);
    e
  }

  #[inline]
  fn fraction_2(self) -> Self {
    let t1 = cast::<_, u64x8>(self);
    let t2 = cast::<_, u64x8>(
      (t1 & u64x8::from(0x000FFFFFFFFFFFFF)) | u64x8::from(0x3FE0000000000000),
    );
    cast::<_, f64x8>(t2)
  }
  #[inline]
  fn is_zero_or_subnormal(self) -> Self {
    let t = cast::<_, i64x8>(self);
    let t = t & i64x8::splat(0x7FF0000000000000);
    i64x8::round_float(t.simd_eq(i64x8::splat(0)))
  }
  #[inline]
  fn infinity() -> Self {
    cast::<_, f64x8>(i64x8::splat(0x7FF0000000000000))
  }
  #[inline]
  fn nan_log() -> Self {
    cast::<_, f64x8>(i64x8::splat(0x7FF8000000000000 | 0x101 << 29))
  }
  #[inline]
  fn nan_pow() -> Self {
    cast::<_, f64x8>(i64x8::splat(0x7FF8000000000000 | 0x101 << 29))
  }
  #[inline]
  fn sign_bit(self) -> Self {
    let t1 = cast::<_, i64x8>(self);
    let t2 = t1 >> 63;
    !cast::<_, f64x8>(t2).simd_eq(f64x8::ZERO)
  }

  #[inline]
  pub fn reduce_add(self) -> f64 {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        // From https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
        let lo = cast_to_m256d_from_m512d(self.avx512);
        let hi = extract_m256d_from_m512d::<1>(self.avx512);
        let v  = add_m256d(lo, hi);                // [a0+a4, a1+a5, a2+a6, a3+a7]
        let t  = add_horizontal_m256d(v, v);       // [s01, s23, s01, s23]
        let lo = cast_to_m128d_from_m256d(t);      // s01
        let hi = extract_m128d_from_m256d::<1>(t); // s23
        let s  = add_m128d(lo, hi);                // [sum, ...]
        get_f64_from_m128d_s(s)
      } else {
        self.a.reduce_add() + self.b.reduce_add()
      }
    }
  }

  /// Natural log (ln(x))
  #[inline]
  #[must_use]
  pub fn ln(self) -> Self {
    const_f64_as_f64x8!(HALF, 0.5);
    const_f64_as_f64x8!(P0, 7.70838733755885391666E0);
    const_f64_as_f64x8!(P1, 1.79368678507819816313E1);
    const_f64_as_f64x8!(P2, 1.44989225341610930846E1);
    const_f64_as_f64x8!(P3, 4.70579119878881725854E0);
    const_f64_as_f64x8!(P4, 4.97494994976747001425E-1);
    const_f64_as_f64x8!(P5, 1.01875663804580931796E-4);

    const_f64_as_f64x8!(Q0, 2.31251620126765340583E1);
    const_f64_as_f64x8!(Q1, 7.11544750618563894466E1);
    const_f64_as_f64x8!(Q2, 8.29875266912776603211E1);
    const_f64_as_f64x8!(Q3, 4.52279145837532221105E1);
    const_f64_as_f64x8!(Q4, 1.12873587189167450590E1);
    const_f64_as_f64x8!(LN2F_HI, 0.693359375);
    const_f64_as_f64x8!(LN2F_LO, -2.12194440e-4);
    const_f64_as_f64x8!(VM_SQRT2, 1.414213562373095048801);
    const_f64_as_f64x8!(VM_SMALLEST_NORMAL, 1.17549435E-38);

    let x1 = self;
    let x = Self::fraction_2(x1);
    let e = Self::exponent(x1);
    let mask = x.simd_gt(VM_SQRT2 * HALF);
    let x = (!mask).blend(x + x, x);
    let fe = mask.blend(e + Self::ONE, e);
    let x = x - Self::ONE;
    let px = polynomial_5!(x, P0, P1, P2, P3, P4, P5);
    let x2 = x * x;
    let px = x2 * x * px;
    let qx = polynomial_5n!(x, Q0, Q1, Q2, Q3, Q4);
    let res = px / qx;
    let res = fe.mul_add(LN2F_LO, res);
    let res = res + x2.mul_neg_add(HALF, x);
    let res = fe.mul_add(LN2F_HI, res);
    let overflow = !self.is_finite();
    let underflow = x1.simd_lt(VM_SMALLEST_NORMAL);
    let mask = overflow | underflow;
    if !mask.any() {
      res
    } else {
      let is_zero = self.is_zero_or_subnormal();
      let res = underflow.blend(Self::nan_log(), res);
      let res = is_zero.blend(Self::infinity(), res);
      let res = overflow.blend(self, res);
      res
    }
  }

  #[inline]
  #[must_use]
  pub fn log2(self) -> Self {
    Self::ln(self) * Self::LOG2_E
  }
  #[inline]
  #[must_use]
  pub fn log10(self) -> Self {
    Self::ln(self) * Self::LOG10_E
  }

  #[inline]
  #[must_use]
  pub fn pow_f64x8(self, y: Self) -> Self {
    const_f64_as_f64x8!(ln2d_hi, 0.693145751953125);
    const_f64_as_f64x8!(ln2d_lo, 1.42860682030941723212E-6);
    const_f64_as_f64x8!(P0log, 2.0039553499201281259648E1);
    const_f64_as_f64x8!(P1log, 5.7112963590585538103336E1);
    const_f64_as_f64x8!(P2log, 6.0949667980987787057556E1);
    const_f64_as_f64x8!(P3log, 2.9911919328553073277375E1);
    const_f64_as_f64x8!(P4log, 6.5787325942061044846969E0);
    const_f64_as_f64x8!(P5log, 4.9854102823193375972212E-1);
    const_f64_as_f64x8!(P6log, 4.5270000862445199635215E-5);
    const_f64_as_f64x8!(Q0log, 6.0118660497603843919306E1);
    const_f64_as_f64x8!(Q1log, 2.1642788614495947685003E2);
    const_f64_as_f64x8!(Q2log, 3.0909872225312059774938E2);
    const_f64_as_f64x8!(Q3log, 2.2176239823732856465394E2);
    const_f64_as_f64x8!(Q4log, 8.3047565967967209469434E1);
    const_f64_as_f64x8!(Q5log, 1.5062909083469192043167E1);

    // Taylor expansion constants
    const_f64_as_f64x8!(p2, 1.0 / 2.0); // coefficients for Taylor expansion of exp
    const_f64_as_f64x8!(p3, 1.0 / 6.0);
    const_f64_as_f64x8!(p4, 1.0 / 24.0);
    const_f64_as_f64x8!(p5, 1.0 / 120.0);
    const_f64_as_f64x8!(p6, 1.0 / 720.0);
    const_f64_as_f64x8!(p7, 1.0 / 5040.0);
    const_f64_as_f64x8!(p8, 1.0 / 40320.0);
    const_f64_as_f64x8!(p9, 1.0 / 362880.0);
    const_f64_as_f64x8!(p10, 1.0 / 3628800.0);
    const_f64_as_f64x8!(p11, 1.0 / 39916800.0);
    const_f64_as_f64x8!(p12, 1.0 / 479001600.0);
    const_f64_as_f64x8!(p13, 1.0 / 6227020800.0);

    let x1 = self.abs();
    let x = x1.fraction_2();
    let mask = x.simd_gt(f64x8::SQRT_2 * f64x8::HALF);
    let x = (!mask).blend(x + x, x);
    let x = x - f64x8::ONE;
    let x2 = x * x;
    let px = polynomial_6!(x, P0log, P1log, P2log, P3log, P4log, P5log, P6log);
    let px = px * x * x2;
    let qx = polynomial_6n!(x, Q0log, Q1log, Q2log, Q3log, Q4log, Q5log);
    let lg1 = px / qx;

    let ef = x1.exponent();
    let ef = mask.blend(ef + f64x8::ONE, ef);
    let e1 = (ef * y).round();
    let yr = ef.mul_sub(y, e1);

    let lg = f64x8::HALF.mul_neg_add(x2, x) + lg1;
    let x2err = (f64x8::HALF * x).mul_sub(x, f64x8::HALF * x2);
    let lg_err = f64x8::HALF.mul_add(x2, lg - x) - lg1;

    let e2 = (lg * y * f64x8::LOG2_E).round();
    let v = lg.mul_sub(y, e2 * ln2d_hi);
    let v = e2.mul_neg_add(ln2d_lo, v);
    let v = v - (lg_err + x2err).mul_sub(y, yr * f64x8::LN_2);

    let x = v;
    let e3 = (x * f64x8::LOG2_E).round();
    let x = e3.mul_neg_add(f64x8::LN_2, x);
    let z =
      polynomial_13m!(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13)
        + f64x8::ONE;
    let ee = e1 + e2 + e3;
    let ei = cast::<_, i64x8>(ee.round_int());
    let ej = cast::<_, i64x8>(ei + (cast::<_, i64x8>(z) >> 52));

    let overflow = cast::<_, f64x8>(!ej.simd_lt(i64x8::splat(0x07FF)))
      | ee.simd_gt(f64x8::splat(3000.0));
    let underflow = cast::<_, f64x8>(!ej.simd_gt(i64x8::splat(0x000)))
      | ee.simd_lt(f64x8::splat(-3000.0));

    // Add exponent by integer addition
    let z = cast::<_, f64x8>(cast::<_, i64x8>(z) + (ei << 52));

    // Check for overflow/underflow
    let z = if (overflow | underflow).any() {
      let z = underflow.blend(f64x8::ZERO, z);
      overflow.blend(Self::infinity(), z)
    } else {
      z
    };

    // Check for self == 0
    let x_zero = self.is_zero_or_subnormal();
    let z = x_zero.blend(
      y.simd_lt(f64x8::ZERO).blend(
        Self::infinity(),
        y.simd_eq(f64x8::ZERO).blend(f64x8::ONE, f64x8::ZERO),
      ),
      z,
    );

    let x_sign = self.sign_bit();

    let z = if x_sign.any() {
      // Y into an integer
      let yi = y.simd_eq(y.round());
      // Is y odd?
      let y_odd = cast::<_, i64x8>(y.round_int() << 63).round_float();
      let z1 =
        yi.blend(z | y_odd, self.simd_eq(Self::ZERO).blend(z, Self::nan_pow()));
      x_sign.blend(z1, z)
    } else {
      z
    };

    let x_finite = self.is_finite();
    let y_finite = y.is_finite();
    let e_finite = ee.is_finite();

    if (x_finite & y_finite & (e_finite | x_zero)).all() {
      return z;
    }

    (self.is_nan() | y.is_nan()).blend(self + y, z)
  }

  #[inline]
  pub fn powf(self, y: f64) -> Self {
    Self::pow_f64x8(self, f64x8::splat(y))
  }

  #[inline]
  #[must_use]
  pub fn to_array(self) -> [f64; 8] {
    cast(self)
  }

  #[inline]
  #[must_use]
  pub fn as_array(&self) -> &[f64; 8] {
    cast_ref(self)
  }

  #[inline]
  #[must_use]
  pub fn as_mut_array(&mut self) -> &mut [f64; 8] {
    cast_mut(self)
  }

  #[inline]
  pub fn from_i32x8(v: i32x8) -> Self {
    pick! {
      if #[cfg(target_feature="avx512f")] {
        Self { avx512: convert_to_m512d_from_i32_m256i(v.avx2) }
      } else {
        Self::new([
          v.as_array()[0] as f64,
          v.as_array()[1] as f64,
          v.as_array()[2] as f64,
          v.as_array()[3] as f64,
          v.as_array()[4] as f64,
          v.as_array()[5] as f64,
          v.as_array()[6] as f64,
          v.as_array()[7] as f64,
        ])
      }
    }
  }
}

impl From<i32x8> for f64x8 {
  #[inline]
  fn from(v: i32x8) -> Self {
    Self::from_i32x8(v)
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
