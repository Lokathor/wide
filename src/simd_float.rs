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
    }
  };
}
