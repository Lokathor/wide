macro_rules! impl_simd_float {
  (
    T = $T:ident,
    N = $N:literal,
    Simd = $Simd:ident,
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
    }
  };
}
