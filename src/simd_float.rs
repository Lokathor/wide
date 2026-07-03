macro_rules! impl_simd_float {
  (
    // SAFETY: The contents of this macro assume that:
    //
    // - `T` implements `Pod`
    // - `Pod` can be implemented for `Simd`
    // - `size_of::<Simd>()` is `size_of::<T>() * N`
    // - `align_of::<Simd>()` is `size_of::<Simd>()`
    unsafe {
      T = $T:ident,
      N = $N:literal,
      Simd = $Simd:ident,
      UnsignedT = $UnsignedT:ident,
    }

    $fn_neg:item
    $fn_not:item
    $fn_add:item
    $fn_sub:item
    $fn_mul:item
    $fn_div:item
    $fn_rem:item
    $fn_bitand:item
    $fn_bitor:item
    $fn_bitxor:item
    $fn_reduce_add:item
    $fn_reduce_mul:item
    $fn_is_nan:item
    $fn_is_inf:item
    $fn_is_finite:item
    $fn_is_sign_positive:item
    $fn_is_sign_negative:item
    $fn_recip:item
    $fn_recip_sqrt:item
    $fn_max:item
    $fn_fast_max:item
    $fn_min:item
    $fn_fast_min:item
    $fn_clamp:item
    $fn_fast_clamp:item
    $fn_abs:item
    $fn_floor:item
    $fn_ceil:item
    $fn_round:item
    $fn_round_int:item
    $fn_fast_round_int:item
    $fn_round_ties_even:item
    $fn_trunc:item
    $fn_trunc_int:item
    $fn_fast_trunc_int:item
    $fn_mul_add:item
    $fn_mul_sub:item
    $fn_mul_neg_add:item
    $fn_mul_neg_sub:item
    $fn_pow_simd:item
    $fn_powf:item
    $fn_sqrt:item
    $fn_exp:item
    $fn_exp2:item
    $fn_ln:item
    $fn_cbrt:item
    $fn_asin:item
    $fn_acos:item
    $fn_atan:item
    $fn_atan2:item
    $fn_sin_cos:item
    $fn_asin_acos:item
    $fn_exp_m1:item
    $fn_ln_1p:item
    $fn_sinh:item
    $fn_cosh:item
    $fn_tanh:item
  ) => {
    impl_unary_operator!($Simd, Neg, neg, $fn_neg);
    impl_unary_operator!($Simd, Not, not, $fn_not);

    impl_binary_operator!($T, $Simd, Add, add, AddAssign, add_assign, $fn_add);
    impl_binary_operator!($T, $Simd, Sub, sub, SubAssign, sub_assign, $fn_sub);
    impl_binary_operator!($T, $Simd, Mul, mul, MulAssign, mul_assign, $fn_mul);
    impl_binary_operator!($T, $Simd, Div, div, DivAssign, div_assign, $fn_div);
    impl_binary_operator!($T, $Simd, Rem, rem, RemAssign, rem_assign, $fn_rem);
    impl_binary_operator!(
      $T,
      $Simd,
      BitAnd,
      bitand,
      BitAndAssign,
      bitand_assign,
      $fn_bitand
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitOr,
      bitor,
      BitOrAssign,
      bitor_assign,
      $fn_bitor
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitXor,
      bitxor,
      BitXorAssign,
      bitxor_assign,
      $fn_bitxor
    );

    impl<Rhs> core::iter::Sum<Rhs> for $Simd
    where
      $Simd: AddAssign<Rhs>,
    {
      #[inline]
      fn sum<I: Iterator<Item = Rhs>>(iter: I) -> Self {
        let mut total = Self::zeroed();
        for val in iter {
          total += val;
        }
        total
      }
    }

    impl<Rhs> core::iter::Product<Rhs> for $Simd
    where
      $Simd: MulAssign<Rhs>,
    {
      #[inline]
      fn product<I: Iterator<Item = Rhs>>(iter: I) -> Self {
        let mut total = Self::from(1.0);
        for val in iter {
          total *= val;
        }
        total
      }
    }

    macro_rules! impl_formatting_trait {
      ($Trait:path) => {
        impl $Trait for $Simd {
          #[allow(clippy::missing_inline_in_public_items)]
          fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            write!(f, "(")?;
            for (i, x) in self.to_array().iter().enumerate() {
              if i > 0 {
                write!(f, ", ")?;
              }
              <$UnsignedT as $Trait>::fmt(&x.to_bits(), f)?;
            }
            write!(f, ")")
          }
        }
      }
    }
    impl_formatting_trait!(core::fmt::Binary);
    impl_formatting_trait!(core::fmt::LowerHex);
    impl_formatting_trait!(core::fmt::Octal);
    impl_formatting_trait!(core::fmt::UpperHex);

    /// The following functionality exists for all SIMD vectors of floats.
    impl $Simd {
      /// A SIMD vector with all elements set to `1.0`.
      pub const ONE: Self = Self::splat(1.0);

      /// A SIMD vector with all elements set to `0.5`.
      pub const HALF: Self = Self::splat(0.5);

      /// A SIMD vector with all elements set to `0.0`.
      pub const ZERO: Self = Self::splat(0.0);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::EPSILON`].")]
      pub const EPSILON: Self = Self::splat($T::EPSILON);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::MIN`].")]
      pub const MIN: Self = Self::splat($T::MIN);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::MIN_POSITIVE`].")]
      pub const MIN_POSITIVE: Self = Self::splat($T::MIN_POSITIVE);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::MAX`].")]
      pub const MAX: Self = Self::splat($T::MAX);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::NAN`].")]
      pub const NAN: Self = Self::splat($T::NAN);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::INFINITY`].")]
      pub const INFINITY: Self = Self::splat($T::INFINITY);

      #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::NEG_INFINITY`].")]
      pub const NEG_INFINITY: Self = Self::splat($T::NEG_INFINITY);

      /// A SIMD vector with all elements set to [Euler's number (e)].
      ///
      #[doc = concat!("[Euler's number (e)]: core::", stringify!($T), "::consts::E")]
      pub const E: Self = Self::splat(core::$T::consts::E);

      /// A SIMD vector with all elements set to [1/π].
      ///
      #[doc = concat!("[1/π]: core::", stringify!($T), "::consts::FRAC_1_PI")]
      pub const FRAC_1_PI: Self = Self::splat(core::$T::consts::FRAC_1_PI);

      /// A SIMD vector with all elements set to [2/π].
      ///
      #[doc = concat!("[2/π]: core::", stringify!($T), "::consts::FRAC_2_PI")]
      pub const FRAC_2_PI: Self = Self::splat(core::$T::consts::FRAC_2_PI);

      /// A SIMD vector with all elements set to [2/sqrt(π)].
      ///
      #[doc = concat!("[2/sqrt(π)]: core::", stringify!($T), "::consts::FRAC_2_SQRT_PI")]
      pub const FRAC_2_SQRT_PI: Self =
        Self::splat(core::$T::consts::FRAC_2_SQRT_PI);

      /// A SIMD vector with all elements set to [1/sqrt(2)].
      ///
      #[doc = concat!("[1/sqrt(2)]: core::", stringify!($T), "::consts::FRAC_1_SQRT_2")]
      pub const FRAC_1_SQRT_2: Self =
        Self::splat(core::$T::consts::FRAC_1_SQRT_2);

      /// A SIMD vector with all elements set to [π/2].
      ///
      #[doc = concat!("[π/2]: core::", stringify!($T), "::consts::FRAC_PI_2")]
      pub const FRAC_PI_2: Self = Self::splat(core::$T::consts::FRAC_PI_2);

      /// A SIMD vector with all elements set to [π/3].
      ///
      #[doc = concat!("[π/3]: core::", stringify!($T), "::consts::FRAC_PI_3")]
      pub const FRAC_PI_3: Self = Self::splat(core::$T::consts::FRAC_PI_3);

      /// A SIMD vector with all elements set to [π/4].
      ///
      #[doc = concat!("[π/4]: core::", stringify!($T), "::consts::FRAC_PI_4")]
      pub const FRAC_PI_4: Self = Self::splat(core::$T::consts::FRAC_PI_4);

      /// A SIMD vector with all elements set to [π/6].
      ///
      #[doc = concat!("[π/6]: core::", stringify!($T), "::consts::FRAC_PI_6")]
      pub const FRAC_PI_6: Self = Self::splat(core::$T::consts::FRAC_PI_6);

      /// A SIMD vector with all elements set to [π/8].
      ///
      #[doc = concat!("[π/8]: core::", stringify!($T), "::consts::FRAC_PI_8")]
      pub const FRAC_PI_8: Self = Self::splat(core::$T::consts::FRAC_PI_8);

      /// A SIMD vector with all elements set to [ln(2)].
      ///
      #[doc = concat!("[ln(2)]: core::", stringify!($T), "::consts::LN_2")]
      pub const LN_2: Self = Self::splat(core::$T::consts::LN_2);

      /// A SIMD vector with all elements set to [ln(10)].
      ///
      #[doc = concat!("[ln(10)]: core::", stringify!($T), "::consts::LN_10")]
      pub const LN_10: Self = Self::splat(core::$T::consts::LN_10);

      /// A SIMD vector with all elements set to [log<sub>2</sub>(e)].
      ///
      #[doc = concat!("[log<sub>2</sub>(e)]: core::", stringify!($T), "::consts::LOG2_E")]
      pub const LOG2_E: Self = Self::splat(core::$T::consts::LOG2_E);

      /// A SIMD vector with all elements set to [log<sub>10</sub>(e)].
      ///
      #[doc = concat!("[log<sub>10</sub>(e)]: core::", stringify!($T), "::consts::LOG10_E")]
      pub const LOG10_E: Self = Self::splat(core::$T::consts::LOG10_E);

      /// A SIMD vector with all elements set to [log<sub>10</sub>(2)].
      ///
      #[doc = concat!("[log<sub>10</sub>(2)]: core::", stringify!($T), "::consts::LOG10_2")]
      pub const LOG10_2: Self = Self::splat(core::$T::consts::LOG10_2);

      /// A SIMD vector with all elements set to [log<sub>2</sub>(10)].
      ///
      #[doc = concat!("[log<sub>2</sub>(10)]: core::", stringify!($T), "::consts::LOG2_10")]
      pub const LOG2_10: Self = Self::splat(core::$T::consts::LOG2_10);

      /// A SIMD vector with all elements set to [Archimedes’ constant (π)].
      ///
      #[doc = concat!("[Archimedes’ constant (π)]: core::", stringify!($T), "::consts::PI")]
      pub const PI: Self = Self::splat(core::$T::consts::PI);

      /// A SIMD vector with all elements set to [sqrt(2)].
      ///
      #[doc = concat!("[sqrt(2)]: core::", stringify!($T), "::consts::SQRT_2")]
      pub const SQRT_2: Self = Self::splat(core::$T::consts::SQRT_2);

      /// A SIMD vector with all elements set to [the full circle constant (τ)].
      ///
      /// Equal to 2π.
      ///
      #[doc = concat!("[the full circle constant (τ)]: core::", stringify!($T), "::consts::TAU")]
      pub const TAU: Self = Self::splat(core::$T::consts::TAU);

      /// horizontal add of all the elements of the vector
      #[must_use]
      $fn_reduce_add

      /// horizontal multiplication of all the elements of the vector
      #[must_use]
      $fn_reduce_mul

      #[must_use]
      $fn_is_nan

      #[must_use]
      $fn_is_inf

      #[must_use]
      $fn_is_finite

      /// Returns true for each element if it has a positive sign, including `+0.0`,
      /// `NaN`s with positive sign bit and positive infinity.
      #[must_use]
      $fn_is_sign_positive

      /// Returns true for each element if it has a negative sign, including `-0.0`,
      /// `NaN`s with negative sign bit and negative infinity.
      #[must_use]
      $fn_is_sign_negative

      #[must_use]
      $fn_recip

      #[must_use]
      $fn_recip_sqrt

      #[inline]
      #[must_use]
      pub fn to_degrees(self) -> Self {
        const RAD_TO_DEG_RATIO: $Simd = $Simd::splat(180.0 / core::$T::consts::PI);
        self * RAD_TO_DEG_RATIO
      }

      #[inline]
      #[must_use]
      pub fn to_radians(self) -> Self {
        const DEG_TO_RAD_RATIO: $Simd = $Simd::splat(core::$T::consts::PI / 180.0);
        self * DEG_TO_RAD_RATIO
      }

      /// Calculates the lanewise maximum of both vectors. If either lane is
      /// NaN, the other lane gets chosen. Use `fast_max` for a faster
      /// implementation that doesn't handle NaNs.
      #[must_use]
      $fn_max

      /// Calculates the lanewise maximum of both vectors. This is a faster
      /// implementation than `max`, but it doesn't specify any behavior if NaNs
      /// are involved.
      #[must_use]
      $fn_fast_max

      /// Calculates the lanewise minimum of both vectors. If either lane is
      /// NaN, the other lane gets chosen. Use `fast_min` for a faster
      /// implementation that doesn't handle NaNs.
      #[must_use]
      $fn_min

      /// Calculates the lanewise minimum of both vectors. This is a faster
      /// implementation than `min`, but it doesn't specify any behavior if NaNs
      /// are involved.
      #[must_use]
      $fn_fast_min

      #[inline]
      #[must_use]
      pub fn midpoint(self, other: Self) -> Self {
        (self + other) * 0.5
      }

      /// Restrict a value to a certain interval unless it is NaN.
      ///
      /// If `self`, `min` or `max` are NaN, the result is NaN.  If `min > max`,
      /// the result is `min` since `max(min)` dominates.
      #[must_use]
      $fn_clamp

      /// Restrict a value to a certain interval unless it is NaN.
      ///
      /// If `self` is NaN, the result is NaN.  If `min > max`, the result is
      /// `min` since `max(min)` dominates. If `min` or `max` are NaN, the
      /// result is unspecified.
      #[must_use]
      $fn_fast_clamp

      #[must_use]
      $fn_abs

      #[inline]
      #[must_use]
      pub fn signum(self) -> Self {
        let result = Self::ONE | self & -Self::ZERO;

        self.is_nan().select(self, result)
      }

      #[inline]
      #[must_use]
      pub fn copysign(self, sign: Self) -> Self {
        let magnitude_mask = Self::from($T::from_bits($UnsignedT::MAX >> 1));
        (self & magnitude_mask) | (sign & Self::from(-0.0))
      }

      #[inline]
      #[must_use]
      pub fn flip_signs(self, signs: Self) -> Self {
        self ^ (signs & Self::from(-0.0))
      }

      #[must_use]
      $fn_floor

      #[must_use]
      $fn_ceil

      /// Returns the nearest integers to `self`. If a value is half-way between
      /// two integers, round away from `0.0`.
      ///
      /// This function always returns the precise result.
      ///
      /// For most targets [`round`] is slower than [`round_ties_even`]. If you
      /// do not care about the difference, consider using that instead.
      ///
      /// [`round`]: Self::round
      /// [`round_ties_even`]: Self::round_ties_even
      #[must_use]
      $fn_round

      /// Rounds each lane into an integer. This saturates out of range values
      /// and turns NaNs into 0. Use `fast_round_int` for a faster
      /// implementation that doesn't handle out of range values or NaNs.
      #[must_use]
      $fn_round_int

      /// Rounds each lane into an integer. This is a faster implementation than
      /// `round_int`, but it doesn't handle out of range values or NaNs. For
      /// those values you get implementation defined behavior.
      #[must_use]
      $fn_fast_round_int

      /// Returns the nearest integers to `self`. Rounds half-way cases to the
      /// number with an even least significant digit.
      ///
      /// This function always returns the precise result.
      #[must_use]
      $fn_round_ties_even

      #[must_use]
      $fn_trunc

      /// Truncates each lane into an integer. This saturates out of range
      /// values and turns NaNs into 0. Use `fast_trunc_int` for a faster
      /// implementation that doesn't handle out of range values or NaNs.
      #[must_use]
      $fn_trunc_int

      /// Truncates each lane into an integer. This is a faster implementation
      /// than `trunc_int`, but it doesn't handle out of range values or NaNs.
      /// For those values you get implementation defined behavior.
      #[must_use]
      $fn_fast_trunc_int

      #[inline]
      #[must_use]
      pub fn fract(self) -> Self {
        self - self.trunc()
      }

      /// Performs a multiply-add operation: `self * m + a`
      ///
      /// When hardware FMA support is available, this computes the result with
      /// a single rounding operation. Without FMA support, it falls back to
      /// separate multiply and add operations with two roundings.
      #[must_use]
      $fn_mul_add

      /// Performs a multiply-subtract operation: `self * m - s`
      ///
      /// When hardware FMA support is available, this computes the result with
      /// a single rounding operation. Without FMA support, it falls back to
      /// separate multiply and subtract operations with two roundings.
      #[must_use]
      $fn_mul_sub

      /// Performs a negative multiply-add operation: `a - (self * m)`
      ///
      /// When hardware FMA support is available, this computes the result with
      /// a single rounding operation. Without FMA support, it falls back to
      /// separate operations with two roundings.
      #[must_use]
      $fn_mul_neg_add

      /// Performs a negative multiply-subtract operation: `-(self * m) - s`
      ///
      /// When hardware FMA support is available, this computes the result with
      /// a single rounding operation. Without FMA support, it falls back to
      /// separate operations with two roundings.
      #[must_use]
      $fn_mul_neg_sub

      #[inline]
      #[must_use]
      pub fn div_euclid(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        (self % rhs)
          .simd_lt(Self::ZERO)
          .select(rhs.simd_gt(Self::ZERO).select(q - Self::ONE, q + Self::ONE), q)
      }

      #[inline]
      #[must_use]
      pub fn rem_euclid(self, rhs: Self) -> Self {
        let r = self % rhs;
        r.simd_lt(Self::ZERO).select(r + rhs.abs(), r)
      }

      #[must_use]
      $fn_pow_simd

      #[must_use]
      $fn_powf

      #[must_use]
      $fn_sqrt

      #[must_use]
      $fn_exp

      /// Returns `2^self`.
      #[must_use]
      $fn_exp2

      /// Natural log (ln(x))
      #[must_use]
      $fn_ln

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

      /// Calculates the cube root: `self^(1/3)`.
      #[must_use]
      $fn_cbrt

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

      #[must_use]
      $fn_asin

      #[must_use]
      $fn_acos

      #[must_use]
      $fn_atan

      #[must_use]
      $fn_atan2

      #[must_use]
      $fn_sin_cos

      #[must_use]
      $fn_asin_acos

      /// Calculate `e^self - 1` for each lane. Accurate even for very small
      /// values.
      #[must_use]
      $fn_exp_m1

      /// Calculate `ln(1 + self)` for each lane. Accurate even for very small
      /// values.
      #[must_use]
      $fn_ln_1p

      /// Calculates hyperbolic sine: `(e^self - e^(-self))/2`.
      #[must_use]
      $fn_sinh

      /// Calculates hyperbolic cosine: `(e^self + e^(-self))/2`.
      #[must_use]
      $fn_cosh

      /// Calculates hyperbolic tangent: `sinh(self)/cosh(self)`.
      #[must_use]
      $fn_tanh
    }
  };
}
