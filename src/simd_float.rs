macro_rules! impl_simd_float {
  (
    T = $T:ident,
    N = $N:literal,
    Simd = $Simd:ident,
    UnsignedT = $UnsignedT:ident,

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
  ) => {
    impl $Simd {
      pub const ONE: Self = Self::splat(1.0);
      pub const HALF: Self = Self::splat(0.5);
      pub const ZERO: Self = Self::splat(0.0);
      pub const EPSILON: Self = Self::splat($T::EPSILON);
      pub const MIN: Self = Self::splat($T::MIN);
      pub const MIN_POSITIVE: Self = Self::splat($T::MIN_POSITIVE);
      pub const MAX: Self = Self::splat($T::MAX);
      pub const NAN: Self = Self::splat($T::NAN);
      pub const INFINITY: Self = Self::splat($T::INFINITY);
      pub const NEG_INFINITY: Self = Self::splat($T::NEG_INFINITY);
      pub const E: Self = Self::splat(core::$T::consts::E);
      pub const FRAC_1_PI: Self = Self::splat(core::$T::consts::FRAC_1_PI);
      pub const FRAC_2_PI: Self = Self::splat(core::$T::consts::FRAC_2_PI);
      pub const FRAC_2_SQRT_PI: Self =
        Self::splat(core::$T::consts::FRAC_2_SQRT_PI);
      pub const FRAC_1_SQRT_2: Self =
        Self::splat(core::$T::consts::FRAC_1_SQRT_2);
      pub const FRAC_PI_2: Self = Self::splat(core::$T::consts::FRAC_PI_2);
      pub const FRAC_PI_3: Self = Self::splat(core::$T::consts::FRAC_PI_3);
      pub const FRAC_PI_4: Self = Self::splat(core::$T::consts::FRAC_PI_4);
      pub const FRAC_PI_6: Self = Self::splat(core::$T::consts::FRAC_PI_6);
      pub const FRAC_PI_8: Self = Self::splat(core::$T::consts::FRAC_PI_8);
      pub const LN_2: Self = Self::splat(core::$T::consts::LN_2);
      pub const LN_10: Self = Self::splat(core::$T::consts::LN_10);
      pub const LOG2_E: Self = Self::splat(core::$T::consts::LOG2_E);
      pub const LOG10_E: Self = Self::splat(core::$T::consts::LOG10_E);
      pub const LOG10_2: Self = Self::splat(core::$T::consts::LOG10_2);
      pub const LOG2_10: Self = Self::splat(core::$T::consts::LOG2_10);
      pub const PI: Self = Self::splat(core::$T::consts::PI);
      pub const SQRT_2: Self = Self::splat(core::$T::consts::SQRT_2);
      pub const TAU: Self = Self::splat(core::$T::consts::TAU);

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
    }
  };
}
