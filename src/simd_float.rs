//! Defines APIs applicable to SIMD vectors of floats.

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

use bytemuck::Zeroable;

use crate::{
    Select, SimdElement, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i32x4, i32x8, i32x16, i64x2,
    i64x4, i64x8,
    simd::{MaskOps, SimdAssociatedTypes, SupportSimdOf},
    u32x4, u32x8, u32x16, u64x2, u64x4, u64x8,
};

/// A sealed trait for all primitive floating-point types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveFloat`, update the documentation of this trait to explain its
/// intended usage.
#[expect(private_bounds)]
pub trait PrimitiveFloat: SimdElement + MaskOps {}

/// Defines the internal implementations of functions applicable to SIMD vectors
/// of floats.
pub(crate) trait SimdFloatBackend: Sized {
    fn neg_backend(self) -> Self;

    fn not_backend(self) -> Self;

    fn add_backend(self, rhs: Self) -> Self;

    fn sub_backend(self, rhs: Self) -> Self;

    fn mul_backend(self, rhs: Self) -> Self;

    fn div_backend(self, rhs: Self) -> Self;

    fn rem_backend(self, rhs: Self) -> Self;

    fn bitand_backend(self, rhs: Self) -> Self;

    fn bitor_backend(self, rhs: Self) -> Self;

    fn bitxor_backend(self, rhs: Self) -> Self;

    fn reduce_add_backend(self) -> Self::T
    where
        Self: SimdAssociatedTypes;

    fn reduce_mul_backend(self) -> Self::T
    where
        Self: SimdAssociatedTypes;

    fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self;

    #[expect(clippy::wrong_self_convention)]
    fn is_nan_backend(self) -> Self;

    #[expect(clippy::wrong_self_convention)]
    fn is_infinite_backend(self) -> Self;

    #[expect(clippy::wrong_self_convention)]
    fn is_finite_backend(self) -> Self;

    #[expect(clippy::wrong_self_convention)]
    fn is_sign_positive_backend(self) -> Self;

    fn recip_backend(self) -> Self;

    fn recip_sqrt_backend(self) -> Self;

    fn max_backend(self, rhs: Self) -> Self;

    fn fast_max_backend(self, rhs: Self) -> Self;

    fn min_backend(self, rhs: Self) -> Self;

    fn fast_min_backend(self, rhs: Self) -> Self;

    fn clamp_backend(self, min: Self, max: Self) -> Self;

    fn fast_clamp_backend(self, min: Self, max: Self) -> Self;

    fn reduce_max_backend(self) -> Self::T
    where
        Self: SimdAssociatedTypes;

    fn fast_reduce_max_backend(self) -> Self::T
    where
        Self: SimdAssociatedTypes;

    fn reduce_min_backend(self) -> Self::T
    where
        Self: SimdAssociatedTypes;

    fn fast_reduce_min_backend(self) -> Self::T
    where
        Self: SimdAssociatedTypes;

    fn abs_backend(self) -> Self;

    fn floor_backend(self) -> Self;

    fn ceil_backend(self) -> Self;

    fn round_backend(self) -> Self;

    fn round_int_backend(self) -> Self::SignedSimd
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::SignedT>;

    fn fast_round_int_backend(self) -> Self::SignedSimd
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::SignedT>;

    fn round_ties_even_backend(self) -> Self;

    fn trunc_backend(self) -> Self;

    fn trunc_int_backend(self) -> Self::SignedSimd
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::SignedT>;

    fn fast_trunc_int_backend(self) -> Self::SignedSimd
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::SignedT>;

    fn fast_mul_add_backend(self, a: Self, b: Self) -> Self;

    fn fast_mul_sub_backend(self, a: Self, b: Self) -> Self;

    fn fast_mul_neg_add_backend(self, a: Self, b: Self) -> Self;

    fn fast_mul_neg_sub_backend(self, a: Self, b: Self) -> Self;

    fn powf_simd_backend(self, n: Self) -> Self;

    fn sqrt_backend(self) -> Self;

    fn exp_backend(self) -> Self;

    fn exp2_backend(self) -> Self;

    fn ln_backend(self) -> Self;

    fn cbrt_backend(self) -> Self;

    fn asin_backend(self) -> Self;

    fn acos_backend(self) -> Self;

    fn atan_backend(self) -> Self;

    fn atan2_backend(self, x: Self) -> Self;

    fn sin_cos_backend(self) -> (Self, Self);

    fn asin_acos_backend(self) -> (Self, Self);

    fn exp_m1_backend(self) -> Self;

    fn ln_1p_backend(self) -> Self;

    fn sinh_backend(self) -> Self;

    fn cosh_backend(self) -> Self;

    fn tanh_backend(self) -> Self;
}

impl PrimitiveFloat for f32 {}
impl PrimitiveFloat for f64 {}

macro_rules! impl_simd_float {
  (
      T = $T:ident,
      Simd = $Simd:ident,
      SignedSimd = $SignedSimd:ident,
      UnsignedT = $UnsignedT:ident,
      UnsignedSimd = $UnsignedSimd:ident,
  ) => {
    impl_unary_operator!(
      $Simd,
      Neg,
      neg,
      neg_backend,
      /// Computes negation for each element of `self`.
      ///
      /// This always returns the precise result, simply flipping the sign-bit.
    );
    impl_unary_operator!(
      $Simd,
      Not,
      not,
      not_backend,
      /// Computes bitwise NOT for each element of `self`.
      ///
      /// This computes bitwise NOT, even though the `!` operator is not
      /// implemented for Rust's primitive floats.
    );

    impl_binary_operator!(
      $T,
      $Simd,
      Add,
      add,
      AddAssign,
      add_assign,
      add_backend,
      /// Computes addition for each element of `self` and the corresponding
      /// element of `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes addition for each element of `self` and the scalar `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes addition for the scalar `self` and each element of `rhs`,
      /// returning a SIMD vector.
      ///
      /// This always returns the precise result.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      Sub,
      sub,
      SubAssign,
      sub_assign,
      sub_backend,
      /// Computes subtraction for each element of `self` and the corresponding
      /// element of `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes subtraction for each element of `self` and the scalar `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes subtraction for the scalar `self` and each element of `rhs`,
      /// returning a SIMD vector.
      ///
      /// This always returns the precise result.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      Mul,
      mul,
      MulAssign,
      mul_assign,
      mul_backend,
      /// Computes multiplication for each element of `self` and the
      /// corresponding element of `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes multiplication for each element of `self` and the scalar
      /// `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes multiplication for the scalar `self` and each element of
      /// `rhs`, returning a SIMD vector.
      ///
      /// This always returns the precise result.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      Div,
      div,
      DivAssign,
      div_assign,
      div_backend,
      /// Computes division for each element of `self` and the corresponding
      /// element of `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes division for each element of `self` and the scalar `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes division for the scalar `self` and each element of
      /// `rhs`, returning a SIMD vector.
      ///
      /// This always returns the precise result.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      Rem,
      rem,
      RemAssign,
      rem_assign,
      rem_backend,
      /// Computes the remainder for each element of `self` and the
      /// corresponding element of `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes the remainder for each element of `self` and the scalar
      /// `rhs`.
      ///
      /// This always returns the precise result.
      ,
      /// Computes the remainder for the scalar `self` and each element of
      /// `rhs`, returning a SIMD vector.
      ///
      /// This always returns the precise result.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitAnd,
      bitand,
      BitAndAssign,
      bitand_assign,
      bitand_backend,
      /// Computes bitwise AND for each element of `self` and the corresponding
      /// element of `rhs`.
      ///
      /// This computes bitwise AND, even though the `&` operator is not
      /// implemented for Rust's primitive floats.
      ,
      /// Computes bitwise AND for each element of `self` and the scalar `rhs`.
      ///
      /// This computes bitwise AND, even though the `&` operator is not
      /// implemented for Rust's primitive floats.
      ,
      /// Computes bitwise AND for the scalar `self` and each element of `rhs`,
      /// returning a SIMD vector.
      ///
      /// This computes bitwise AND, even though the `&` operator is not
      /// implemented for Rust's primitive floats.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitOr,
      bitor,
      BitOrAssign,
      bitor_assign,
      bitor_backend,
      /// Computes bitwise OR for each element of `self` and the corresponding
      /// element of `rhs`.
      ///
      /// This computes bitwise OR, even though the `|` operator is not
      /// implemented for Rust's primitive floats.
      ,
      /// Computes bitwise OR for each element of `self` and the scalar `rhs`.
      ///
      /// This computes bitwise OR, even though the `|` operator is not
      /// implemented for Rust's primitive floats.
      ,
      /// Computes bitwise OR for the scalar `self` and each element of `rhs`,
      /// returning a SIMD vector.
      ///
      /// This computes bitwise OR, even though the `|` operator is not
      /// implemented for Rust's primitive floats.
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitXor,
      bitxor,
      BitXorAssign,
      bitxor_assign,
      bitxor_backend,
      /// Computes bitwise XOR for each element of `self` and the corresponding
      /// element of `rhs`.
      ///
      /// This computes bitwise XOR, even though the `^` operator is not
      /// implemented for Rust's primitive floats.
      ,
      /// Computes bitwise XOR for each element of `self` and the scalar `rhs`.
      ///
      /// This computes bitwise XOR, even though the `^` operator is not
      /// implemented for Rust's primitive floats.
      ,
      /// Computes bitwise XOR for the scalar `self` and each element of `rhs`,
      /// returning a SIMD vector.
      ///
      /// This computes bitwise XOR, even though the `^` operator is not
      /// implemented for Rust's primitive floats.
    );

    impl<Rhs> core::iter::Sum<Rhs> for $Simd
    where
      $Simd: AddAssign<Rhs>,
    {
      /// Computes the sum of multiple SIMD vectors for each lane.
      ///
      /// The order of addition is not specified.
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
      /// Computes the product of multiple SIMD vectors for each lane.
      ///
      /// The order of multiplication is not specified.
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

    impl Select<$Simd> for $UnsignedSimd {
      #[inline]
      fn select(self, if_true: $Simd, if_false: $Simd) -> $Simd {
        $Simd::from_bits(self).select(if_true, if_false)
      }
    }

    impl Select<$UnsignedSimd> for $Simd {
      #[inline]
      fn select(self, if_true: $UnsignedSimd, if_false: $UnsignedSimd) -> $UnsignedSimd {
        self.to_bits().select(if_true, if_false)
      }
    }

    impl Select<$Simd> for $SignedSimd {
      #[inline]
      fn select(self, if_true: $Simd, if_false: $Simd) -> $Simd {
        $Simd::from_bits(self.cast_unsigned()).select(if_true, if_false)
      }
    }

    impl Select<$SignedSimd> for $Simd {
      #[inline]
      fn select(self, if_true: $SignedSimd, if_false: $SignedSimd) -> $SignedSimd {
        self.to_bits().cast_signed().select(if_true, if_false)
      }
    }

    /// The following functionality exists for all SIMD vectors of floats.
    impl $Simd {
      /// A SIMD vector with all elements set to `1.0`.
      pub const ONE: Self = Self::splat(1.0);

      /// A SIMD vector with all elements set to `0.5`.
      pub const HALF: Self = Self::splat(0.5);

      /// A SIMD vector with all elements set to `0.0`.
      pub const ZERO: Self = Self::splat(0.0);

      /// A SIMD vector with all elements set to [`T::EPSILON`](f32::EPSILON).
      pub const EPSILON: Self = Self::splat($T::EPSILON);

      /// A SIMD vector with all elements set to [`T::MIN`](f32::MIN).
      pub const MIN: Self = Self::splat($T::MIN);

      /// A SIMD vector with all elements set to
      /// [`T::MIN_POSITIVE`](f32::MIN_POSITIVE).
      pub const MIN_POSITIVE: Self = Self::splat($T::MIN_POSITIVE);

      /// A SIMD vector with all elements set to [`T::MAX`](f32::MAX).
      pub const MAX: Self = Self::splat($T::MAX);

      /// A SIMD vector with all elements set to [`T::NAN`](f32::NAN).
      pub const NAN: Self = Self::splat($T::NAN);

      /// A SIMD vector with all elements set to [`T::INFINITY`](f32::INFINITY).
      pub const INFINITY: Self = Self::splat($T::INFINITY);

      /// A SIMD vector with all elements set to
      /// [`T::NEG_INFINITY`](f32::NEG_INFINITY).
      pub const NEG_INFINITY: Self = Self::splat($T::NEG_INFINITY);

      /// A SIMD vector with all elements set to
      /// [Euler's number (e)](core::f32::consts::E).
      pub const E: Self = Self::splat(core::$T::consts::E);

      /// A SIMD vector with all elements set to
      /// [1/π](core::f32::consts::FRAC_1_PI).
      pub const FRAC_1_PI: Self = Self::splat(core::$T::consts::FRAC_1_PI);

      /// A SIMD vector with all elements set to
      /// [2/π](core::f32::consts::FRAC_2_PI).
      pub const FRAC_2_PI: Self = Self::splat(core::$T::consts::FRAC_2_PI);

      /// A SIMD vector with all elements set to
      /// [2/sqrt(π)](core::f32::consts::FRAC_2_SQRT_PI).
      pub const FRAC_2_SQRT_PI: Self =
        Self::splat(core::$T::consts::FRAC_2_SQRT_PI);

      /// A SIMD vector with all elements set to
      /// [1/sqrt(2)](core::f32::consts::FRAC_1_SQRT_2).
      pub const FRAC_1_SQRT_2: Self =
        Self::splat(core::$T::consts::FRAC_1_SQRT_2);

      /// A SIMD vector with all elements set to
      /// [π/2](core::f32::consts::FRAC_PI_2).
      pub const FRAC_PI_2: Self = Self::splat(core::$T::consts::FRAC_PI_2);

      /// A SIMD vector with all elements set to
      /// [π/3](core::f32::consts::FRAC_PI_3).
      pub const FRAC_PI_3: Self = Self::splat(core::$T::consts::FRAC_PI_3);

      /// A SIMD vector with all elements set to
      /// [π/4](core::f32::consts::FRAC_PI_4).
      pub const FRAC_PI_4: Self = Self::splat(core::$T::consts::FRAC_PI_4);

      /// A SIMD vector with all elements set to
      /// [π/6](core::f32::consts::FRAC_PI_6).
      pub const FRAC_PI_6: Self = Self::splat(core::$T::consts::FRAC_PI_6);

      /// A SIMD vector with all elements set to
      /// [π/8](core::f32::consts::FRAC_PI_8).
      pub const FRAC_PI_8: Self = Self::splat(core::$T::consts::FRAC_PI_8);

      /// A SIMD vector with all elements set to
      /// [ln(2)](core::f32::consts::LN_2).
      pub const LN_2: Self = Self::splat(core::$T::consts::LN_2);

      /// A SIMD vector with all elements set to
      /// [ln(10)](core::f32::consts::LN_10).
      pub const LN_10: Self = Self::splat(core::$T::consts::LN_10);

      /// A SIMD vector with all elements set to
      /// [log<sub>2</sub>(e)](core::f32::consts::LOG2_E).
      pub const LOG2_E: Self = Self::splat(core::$T::consts::LOG2_E);

      /// A SIMD vector with all elements set to
      /// [log<sub>10</sub>(e)](core::f32::consts::LOG10_E).
      pub const LOG10_E: Self = Self::splat(core::$T::consts::LOG10_E);

      /// A SIMD vector with all elements set to
      /// [log<sub>10</sub>(2)](core::f32::consts::LOG10_2).
      pub const LOG10_2: Self = Self::splat(core::$T::consts::LOG10_2);

      /// A SIMD vector with all elements set to
      /// [log<sub>2</sub>(10)](core::f32::consts::LOG2_10).
      pub const LOG2_10: Self = Self::splat(core::$T::consts::LOG2_10);

      /// A SIMD vector with all elements set to
      /// [Archimedes’ constant (π)](core::f32::consts::PI).
      pub const PI: Self = Self::splat(core::$T::consts::PI);

      /// A SIMD vector with all elements set to
      /// [sqrt(2)](core::f32::consts::SQRT_2).
      pub const SQRT_2: Self = Self::splat(core::$T::consts::SQRT_2);

      /// A SIMD vector with all elements set to
      /// [the full circle constant (τ)](core::f32::consts::TAU).
      ///
      /// Equal to 2π.
      pub const TAU: Self = Self::splat(core::$T::consts::TAU);

      /// Returns a [mask] that checks if each element is NaN.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[must_use]
      pub fn is_nan(self) -> Self {
        self.is_nan_backend()
      }

      /// Returns a [mask] that checks if each element is infinity (either
      /// positive or negative).
      ///
      /// [mask]: crate#masks
      #[inline]
      #[must_use]
      pub fn is_infinite(self) -> Self {
        self.is_infinite_backend()
      }

      /// Returns a [mask] that checks if each element is neither infinite nor
      /// NaN.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[must_use]
      pub fn is_finite(self) -> Self {
        self.is_finite_backend()
      }

      /// Returns a [mask] that checks if each element has a positive sign,
      /// including `+0.0`, NaNs with positive sign bit and positive infinity.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[must_use]
      pub fn is_sign_positive(self) -> Self {
        self.is_sign_positive_backend()
      }

      /// Returns a [mask] that checks if each element has a negative sign,
      /// including `-0.0`, NaNs with negative sign bit and negative infinity.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[must_use]
      pub fn is_sign_negative(self) -> Self {
        // Both float `is_sign_negative` and signed integer `is_negative` check
        // whether the most significant bit is set.
        Self::from_bits(self.to_bits().cast_signed().is_negative().cast_unsigned())
      }

      /// Approximates the reciprocal (inverse) of a number, `1/x`.
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      ///
      /// To compute the reciprocal deterministically, use `1.0 / x`.
      #[inline]
      #[must_use]
      pub fn recip(self) -> Self {
        self.recip_backend()
      }

      /// Approximates the square root of the reciprocal (inverse) of a number,
      /// `sqrt(1/x)`.
      ///
      /// # Unspecified precision
      ///
      /// The precision of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn recip_sqrt(self) -> Self {
        self.recip_sqrt_backend()
      }

      /// Converts radians to degrees.
      ///
      /// # Unspecified precision
      ///
      /// This currently performs a deterministic multiplication by a constant.
      /// However, this is not made as a guarantee, because the equivalent `std`
      /// float function has unspecified precision.
      ///
      /// The precision of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn to_degrees(self) -> Self {
        const RAD_TO_DEG_RATIO: $Simd = $Simd::splat(180.0 / core::$T::consts::PI);
        self * RAD_TO_DEG_RATIO
      }

      /// Converts degrees to radians.
      ///
      /// # Unspecified precision
      ///
      /// This currently performs a deterministic multiplication by a constant.
      /// However, this is not made as a guarantee, because the equivalent `std`
      /// float function has unspecified precision.
      ///
      /// The precision of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn to_radians(self) -> Self {
        const DEG_TO_RAD_RATIO: $Simd = $Simd::splat(core::$T::consts::PI / 180.0);
        self * DEG_TO_RAD_RATIO
      }

      /// Returns the maximum between each element of `self` and the
      /// corresponding element of `other`, ignoring NaN.
      ///
      /// For each lane, if exactly one of the arguments is NaN, then the other
      /// argument is returned. If both arguments are NaN, the return value is
      /// NaN. If the inputs compare equal (such as for the case of `+0.0` and
      /// `-0.0`), either input may be returned non-deterministically.
      ///
      /// See [`fast_max`] for a faster variant that does not handle NaNs.
      ///
      /// [`fast_max`]: Self::fast_max
      #[inline]
      #[must_use]
      pub fn max(self, other: Self) -> Self {
        self.max_backend(other)
      }

      /// Returns the maximum between each element of `self` and the
      /// corresponding element of `other`, not specifying behavior for NaNs.
      ///
      /// For each lane, if both arguments are NaN, the return value is NaN. If
      /// the inputs compare equal (such as for the case of `+0.0` and `-0.0`),
      /// or if exactly one of the arguments is NaN, either input may be
      /// returned non-deterministically.
      ///
      /// See [`max`] for a slower variant that does handle NaNs.
      ///
      /// [`max`]: Self::max
      #[inline]
      #[must_use]
      pub fn fast_max(self, other: Self) -> Self {
        self.fast_max_backend(other)
      }

      /// Returns the minimum between each element of `self` and the
      /// corresponding element of `other`, ignoring NaN.
      ///
      /// For each lane, if exactly one of the arguments is NaN, then the other
      /// argument is returned. If both arguments are NaN, the return value is
      /// NaN. If the inputs compare equal (such as for the case of `+0.0` and
      /// `-0.0`), either input may be returned non-deterministically.
      ///
      /// See [`fast_min`] for a faster variant that does not handle NaNs.
      ///
      /// [`fast_min`]: Self::fast_min
      #[inline]
      #[must_use]
      pub fn min(self, other: Self) -> Self {
        self.min_backend(other)
      }

      /// Returns the minimum between each element of `self` and the
      /// corresponding element of `other`, not specifying behavior for NaNs.
      ///
      /// For each lane, if both arguments are NaN, the return value is NaN. If
      /// the inputs compare equal (such as for the case of `+0.0` and `-0.0`),
      /// or if exactly one of the arguments is NaN, either input may be
      /// returned non-deterministically.
      ///
      /// See [`min`] for a slower variant that does handle NaNs.
      ///
      /// [`min`]: Self::min
      #[inline]
      #[must_use]
      pub fn fast_min(self, other: Self) -> Self {
        self.fast_min_backend(other)
      }

      /// Reducing addition. Returns the sum of the vector's elements.
      ///
      /// Equivalent to `self[0] + self[1] + ...`.
      ///
      /// # Non-deterministic
      ///
      /// If this performs multiple additions, their order is non-deterministic.
      /// This could also perform one sum operation with higher precision than
      /// expected. This can vary by platform, version, and could even differ
      /// within the same execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn reduce_add(self) -> $T {
        self.reduce_add_backend()
      }

      /// Reducing multiplication. Returns the product of the vector's
      /// elements.
      ///
      /// Equivalent to `self[0] * self[1] * ...`.
      ///
      /// # Non-deterministic
      ///
      /// If this performs multiple multiplications, their order is
      /// non-deterministic. This could also perform one product operation with
      /// higher precision than expected. This can vary by platform, version,
      /// and could even differ within the same execution from one invocation to
      /// the next.
      #[inline]
      #[must_use]
      pub fn reduce_mul(self) -> $T {
        self.reduce_mul_backend()
      }

      /// Returns the maximum between all elements of `self`.
      ///
      /// If only some elements are NaN, then a non-NaN element is returned. If
      /// all elements are NaN, the return value is NaN. If elements compare
      /// equal (such as for the case of `+0.0` and `-0.0`), either may be
      /// returned non-deterministically.
      ///
      /// See [`fast_reduce_max`] for a faster variant that does not handle
      /// NaNs.
      ///
      /// [`fast_reduce_max`]: Self::fast_reduce_max
      #[inline]
      #[must_use]
      pub fn reduce_max(self) -> $T {
        self.reduce_max_backend()
      }

      /// Returns the maximum between all elements of `self`.
      ///
      /// Unlike [`reduce_max`], this function does not handle NaNs. If any
      /// element is NaN, any element may be returned. If elements compare equal
      /// (such as for the case of `+0.0` and `-0.0`), either may be returned
      /// non-deterministically.
      ///
      /// [`reduce_max`]: Self::reduce_max
      #[inline]
      #[must_use]
      pub fn fast_reduce_max(self) -> $T {
        self.fast_reduce_max_backend()
      }

      /// Returns the minimum between all elements of `self`.
      ///
      /// If only some elements are NaN, then a non-NaN element is returned. If
      /// all elements are NaN, the return value is NaN. If elements compare
      /// equal (such as for the case of `+0.0` and `-0.0`), either may be
      /// returned non-deterministically.
      ///
      /// See [`fast_reduce_min`] for a faster variant that does not handle
      /// NaNs.
      ///
      /// [`fast_reduce_min`]: Self::fast_reduce_min
      #[inline]
      #[must_use]
      pub fn reduce_min(self) -> $T {
        self.reduce_min_backend()
      }

      /// Returns the minimum between all elements of `self`.
      ///
      /// Unlike [`reduce_min`], this function does not handle NaNs. If any
      /// element is NaN, any element may be returned. If elements compare equal
      /// (such as for the case of `+0.0` and `-0.0`), either may be returned
      /// non-deterministically.
      ///
      /// [`reduce_min`]: Self::reduce_min
      #[inline]
      #[must_use]
      pub fn fast_reduce_min(self) -> $T {
        self.fast_reduce_min_backend()
      }

      /// Calculates the midpoint (average) between `self` and `other`.
      ///
      /// This returns NaN when *either* argument is NaN or if a combination of
      /// +inf and -inf is provided as arguments.
      ///
      /// This currently returns a less precise result than
      /// [`T::midpoint`](f32::midpoint) in order to gain performance, but this
      /// may change in a future version.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn midpoint(self, other: Self) -> Self {
        (self + other) * 0.5
      }

      /// Raw transmutation to unsigned integer vector.
      ///
      /// Note that this function preserves the *bitwise* value, and not the
      /// numeric value.
      #[inline]
      #[must_use]
      pub const fn to_bits(self) -> $UnsignedSimd {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$Simd, $UnsignedSimd>(self) }
      }

      /// Raw transmutation from unsigned integer vector.
      ///
      /// Note that this function preserves the *bitwise* value, and not the
      /// numeric value.
      #[inline]
      #[must_use]
      pub const fn from_bits(bits: $UnsignedSimd) -> Self {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$UnsignedSimd, $Simd>(bits) }
      }

      /// Creates a SIMD vector from a native-endian byte array.
      ///
      /// The bytes are interpreted as the raw memory representation of the
      /// vector. This is equivalent to the `from_ne_bytes` functions on
      /// primitive float types, but operates on the entire vector at once.
      ///
      /// This is the inverse of [`to_ne_bytes`].
      ///
      /// [`to_ne_bytes`]: Self::to_ne_bytes
      #[inline]
      #[must_use]
      pub const fn from_ne_bytes(bytes: [u8; size_of::<$Simd>()]) -> Self {
        // SAFETY: `$Simd` accepts all bit-patterns and only contains
        // initialized memory, and the byte array has the same size.
        unsafe { core::mem::transmute::<[u8; size_of::<$Simd>()], $Simd>(bytes) }
      }

      /// Returns the raw memory representation of `self` as a native-endian
      /// byte array.
      ///
      /// This is the inverse of [`from_ne_bytes`].
      ///
      /// [`from_ne_bytes`]: Self::from_ne_bytes
      #[inline]
      #[must_use]
      pub const fn to_ne_bytes(self) -> [u8; size_of::<$Simd>()] {
        // SAFETY: `$Simd` accepts all bit-patterns and only contains
        // initialized memory, and the byte array has the same size.
        unsafe { core::mem::transmute::<$Simd, [u8; size_of::<$Simd>()]>(self) }
      }

      /// Bitwise selection.
      ///
      /// For each bit of `self`:
      ///
      /// - If the bit is one, return the corresponding bit of `if_one`
      /// - If the bit is zero, return the corresponding bit of `if_zero`
      ///
      /// This computes a bitwise operation, even for NaNs, and even though
      /// bitwise operators are not implemented for Rust's primitive floats.
      ///
      /// If you know `self` is a [mask], meaning each element is either all
      /// zeros or all ones, consider using [`select`] which is faster.
      ///
      /// [mask]: crate#masks
      /// [`select`]: Self::select
      #[inline]
      #[must_use]
      pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
        self.bitselect_backend(if_one, if_zero)
      }

      /// Restrict a value to a certain interval unless it is NaN.
      ///
      /// If `self`, `min` or `max` are NaN, the result is NaN. If `min > max`,
      /// the result is `min`. If inputs compare equal (such as for the case of
      /// `+0.0` and `-0.0`), either input may be returned
      /// non-deterministically.
      ///
      /// See [`fast_clamp`] for a faster variant that does not handle `min` or
      /// `max` being NaN.
      ///
      /// [`fast_clamp`]: Self::fast_clamp
      #[inline]
      #[must_use]
      pub fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp_backend(min, max)
      }

      /// Restrict a value to a certain interval unless it is NaN.
      ///
      /// If `self` is NaN, the result is NaN. If `min > max`, the result is
      /// `min`. If inputs compare equal (such as for the case of
      /// `+0.0` and `-0.0`), or if `min` or `max` are NaN, any input may be
      /// returned non-deterministically.
      ///
      /// See [`clamp`] for a slower variant that also handles `min` or `max`
      /// being NaN.
      ///
      /// [`clamp`]: Self::clamp
      #[inline]
      #[must_use]
      pub fn fast_clamp(self, min: Self, max: Self) -> Self {
        self.fast_clamp_backend(min, max)
      }

      /// Computes the absolute value of `self`.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn abs(self) -> Self {
        self.abs_backend()
      }

      /// Returns numbers representing the sign of each element.
      ///
      /// - `1.0` if the element is positive, `+0.0`, or `INFINITY`
      /// - `-1.0` if the element is negative, `-0.0`, or `NEG_INFINITY`
      /// - NaN if the element is NaN
      ///
      /// This matches the behavior of [`f32::signum`].
      ///
      /// [`f32::signum`]: https://doc.rust-lang.org/std/primitive.f32.html#method.signum
      #[inline]
      #[must_use]
      pub fn signum(self) -> Self {
        Self::ONE | self & -Self::ZERO | self.is_nan()
      }

      /// Returns numbers composed of the magnitudes of `self` and the signs of
      /// `sign`.
      ///
      /// Equal to `self` if the sign of `self` and `sign` are the same,
      /// otherwise equal to `-self`. Even if `self` or `sign` are NaN, the
      /// result is the exact bit pattern of `self` with the sign bit of `sign`.
      #[inline]
      #[must_use]
      pub fn copysign(self, sign: Self) -> Self {
        Self::splat(-0.0).bitselect(sign, self)
      }

      /// Flips the sign of `self` based on the sign of `sign`.
      ///
      /// If `sign` has a positive sign, the result is `self`. If `sign` has a
      /// negative sign, the result is `-self`. Even if `self` or `sign` are
      /// NaN, the result is the exact bit pattern of `self` with a sign flipped
      /// based on the sign bit of `sign`.
      #[inline]
      #[must_use]
      pub fn flipsign(self, sign: Self) -> Self {
        self ^ (sign & Self::from(-0.0))
      }

      /// Returns the largest integer less than or equal to each input element.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn floor(self) -> Self {
        self.floor_backend()
      }

      /// Returns the smallest integer greater than or equal to each input
      /// element.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn ceil(self) -> Self {
        self.ceil_backend()
      }

      /// Returns the nearest integer to each input element. If a value is
      /// half-way between two integers, round away from `0.0`.
      ///
      /// This function always returns the precise result.
      ///
      /// For most targets architectures, [`round`] is slower than
      /// [`round_ties_even`]. If you do not care about the difference, consider
      /// using that instead.
      ///
      /// [`round`]: Self::round
      /// [`round_ties_even`]: Self::round_ties_even
      #[inline]
      #[must_use]
      pub fn round(self) -> Self {
        self.round_backend()
      }

      /// Returns the nearest integer to each input element.
      ///
      /// The result for values half-way between two integers is currently not
      /// specified.
      ///
      /// This saturates out of range values and turns NaNs to `0`. See
      /// [`fast_round_int`] for a faster variant that does not handle out of
      /// range values or NaNs.
      ///
      /// [`fast_round_int`]: Self::fast_round_int
      #[inline]
      #[must_use]
      pub fn round_int(self) -> $SignedSimd {
        self.round_int_backend()
      }

      /// Returns the nearest integer to each input element.
      ///
      /// The result for values half-way between two integers is currently not
      /// specified.
      ///
      /// This function does not handle out of range values or NaNs. See
      /// [`round_int`] for a slower variant that does handle out of range
      /// values and NaNs.
      ///
      /// [`round_int`]: Self::round_int
      #[inline]
      #[must_use]
      pub fn fast_round_int(self) -> $SignedSimd {
        self.fast_round_int_backend()
      }

      /// Returns the nearest integer to each input element. Rounds half-way
      /// cases to the number with an even least significant digit.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn round_ties_even(self) -> Self {
        self.round_ties_even_backend()
      }

      /// Returns the integer part of each input element. This means that
      /// non-integer numbers are always truncated towards zero.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn trunc(self) -> Self {
        self.trunc_backend()
      }

      /// Returns the integer part of each input element. This means that
      /// non-integer numbers are always truncated towards zero.
      ///
      /// This saturates out of range values and turns NaNs to `0`. See
      /// [`fast_trunc_int`] for a faster variant that does not handle out of
      /// range values or NaNs.
      ///
      /// [`fast_trunc_int`]: Self::fast_trunc_int
      #[inline]
      #[must_use]
      pub fn trunc_int(self) -> $SignedSimd {
        self.trunc_int_backend()
      }

      /// Returns the integer part of each input element. This means that
      /// non-integer numbers are always truncated towards zero.
      ///
      /// This function does not handle out of range values or NaNs. See
      /// [`trunc_int`] for a slower variant that does handle out of range
      /// values and NaNs.
      ///
      /// [`trunc_int`]: Self::trunc_int
      #[inline]
      #[must_use]
      pub fn fast_trunc_int(self) -> $SignedSimd {
        self.fast_trunc_int_backend()
      }

      /// Returns the fractional part of each input element.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn fract(self) -> Self {
        self - self.trunc()
      }

      /// Fused multiply-add. Computes `(self * a) + b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[inline]
      #[must_use]
      pub fn fast_mul_add(self, a: Self, b: Self) -> Self {
        self.fast_mul_add_backend(a, b)
      }

      /// Fused multiply-sub. Computes `(self * a) - b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[inline]
      #[must_use]
      pub fn fast_mul_sub(self, a: Self, b: Self) -> Self {
        self.fast_mul_sub_backend(a, b)
      }

      /// Fused multiply-negate-add. Computes `-(self * a) + b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[inline]
      #[must_use]
      pub fn fast_mul_neg_add(self, a: Self, b: Self) -> Self {
        self.fast_mul_neg_add_backend(a, b)
      }

      /// Fused multiply-negate-sub. Computes `-(self * a) - b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      #[inline]
      #[must_use]
      pub fn fast_mul_neg_sub(self, a: Self, b: Self) -> Self {
        self.fast_mul_neg_sub_backend(a, b)
      }

      /// Calculates Euclidean division, the matching function for
      /// [`rem_euclid`].
      ///
      /// This computes the integer `n` such that
      /// `self = n * rhs + self.rem_euclid(rhs)`. In other words, the result is
      /// `self / rhs` rounded to the integer `n` such that `self >= n * rhs`.
      ///
      /// This function is not guaranteed to exactly match
      /// [`T::div_euclid`](f32::div_euclid).
      ///
      /// [`rem_euclid`]: Self::rem_euclid
      #[inline]
      #[must_use]
      pub fn div_euclid(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        (self % rhs)
          .simd_lt(Self::ZERO)
          .select(rhs.simd_gt(Self::ZERO).select(q - Self::ONE, q + Self::ONE), q)
      }

      /// Calculates the least nonnegative remainder of `self` when divided by
      /// `rhs`.
      ///
      /// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
      /// most cases. However, due to a floating point round-off error it can
      /// result in `r == rhs.abs()`, violating the mathematical definition, if
      /// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
      /// This result is not an element of the function's codomain, but it is the
      /// closest floating point number in the real numbers and thus fulfills the
      /// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
      /// approximately.
      ///
      /// This function is not guaranteed to exactly match
      /// [`T::rem_euclid`](f32::rem_euclid).
      #[inline]
      #[must_use]
      pub fn rem_euclid(self, rhs: Self) -> Self {
        let r = self % rhs;
        r.simd_lt(Self::ZERO).select(r + rhs.abs(), r)
      }

      /// Raises each element of the number `self` to the corresponding element
      /// of the floating point power `n`.
      ///
      /// This function cannot be named simply `powf`, because a now deprecated
      /// function already uses that name.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn powf_simd(self, n: Self) -> Self {
        self.powf_simd_backend(n)
      }

      /// Returns the square root of a number for each input element.
      ///
      /// Returns NaN if `self` is a negative number other than `-0.0`.
      ///
      /// This function always returns the precise result.
      #[inline]
      #[must_use]
      pub fn sqrt(self) -> Self {
        self.sqrt_backend()
      }

      /// Returns `e^(self)`, (the exponential function) for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn exp(self) -> Self {
        self.exp_backend()
      }

      /// Returns `2^(self)` for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn exp2(self) -> Self {
        self.exp2_backend()
      }

      /// Returns the natural logarithm of a number for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn ln(self) -> Self {
        self.ln_backend()
      }

      /// Returns the base 2 logarithm of a number for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn log2(self) -> Self {
        Self::ln(self) * Self::LOG2_E
      }

      /// Returns the base 10 logarithm of a number for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn log10(self) -> Self {
        Self::ln(self) * Self::LOG10_E
      }

      /// Returns the cube root of a number for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn cbrt(self) -> Self {
        self.cbrt_backend()
      }

      /// Computes the sine of a number (in radians) for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn sin(self) -> Self {
        let (s, _) = self.sin_cos();
        s
      }

      /// Computes the cosine of a number (in radians) for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn cos(self) -> Self {
        let (_, c) = self.sin_cos();
        c
      }

      /// Computes the tangent of a number (in radians) for each input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s / c
      }

      /// Computes the arcsine of a number for each input element. Return value
      /// is in radians in the range [-pi/2, pi/2] or NaN if the number is
      /// outside the range [-1, 1].
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn asin(self) -> Self {
        self.asin_backend()
      }

      /// Computes the arccosine of a number for each input element. Return
      /// value is in radians in the range [0, pi] or NaN if the number is
      /// outside the range [-1, 1].
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn acos(self) -> Self {
        self.acos_backend()
      }

      /// Computes the arctangent of a number for each input element. Return
      /// value is in radians in the range [-pi/2, pi/2].
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn atan(self) -> Self {
        self.atan_backend()
      }

      /// Computes the four quadrant arctangent of each element of `self` (`y`)
      /// and the corresponding element of `other` (`x`) in radians.
      ///
      /// | `x`     | `y`     | Piecewise Definition | Range         |
      /// |---------|---------|----------------------|---------------|
      /// | `>= +0` | `>= +0` | `arctan(y/x)`        | `[+0, +pi/2]` |
      /// | `>= +0` | `<= -0` | `arctan(y/x)`        | `[-pi/2, -0]` |
      /// | `<= -0` | `>= +0` | `arctan(y/x) + pi`   | `[+pi/2, +pi]`|
      /// | `<= -0` | `<= -0` | `arctan(y/x) - pi`   | `[-pi, -pi/2]`|
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn atan2(self, other: Self) -> Self {
        self.atan2_backend(other)
      }

      /// Simultaneously computes the sine and cosine of a number `x` for each
      /// input element. Returns `(sin(x), cos(x))`.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn sin_cos(self) -> (Self, Self) {
        self.sin_cos_backend()
      }

      /// Simultaneously computes the arcsine and arccosine of a number `x` for
      /// each input element. Returns `(asin(x), acos(x))`.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn asin_acos(self) -> (Self, Self) {
        self.asin_acos_backend()
      }

      /// Returns `e^(self) - 1` for each input element in a way that is
      /// accurate even if a number is close to zero.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn exp_m1(self) -> Self {
        self.exp_m1_backend()
      }

      /// Returns `ln(1+n)` (natural logarithm) for each input element more
      /// accurately than if the operations were performed separately.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn ln_1p(self) -> Self {
        self.ln_1p_backend()
      }

      /// Returns the hyperbolic sine (`(e^self - e^(-self))/2`) for each input
      /// element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn sinh(self) -> Self {
        self.sinh_backend()
      }

      /// Returns the hyperbolic cosine (`(e^self + e^(-self))/2`) for each
      /// input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn cosh(self) -> Self {
        self.cosh_backend()
      }

      /// Returns the hyperbolic tangent (`sinh(self)/cosh(self)`) for each
      /// input element.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[inline]
      #[must_use]
      pub fn tanh(self) -> Self {
        self.tanh_backend()
      }

      /// Returns a [mask] that checks if each element is infinity (either
      /// positive or negative).
      ///
      /// This function has been deprecated and renamed to [`is_infinite`].
      ///
      /// [mask]: crate#masks
      /// [`is_infinite`]: Self::is_infinite
      #[inline]
      #[must_use]
      #[deprecated(since = "1.8.0", note = "renamed to `is_infinite`")]
      pub fn is_inf(self) -> Self {
        self.is_infinite()
      }

      /// Flips the sign of `self` based on the sign of `sign`.
      ///
      /// If `sign` has a positive sign, the result is `self`. If `sign` has a
      /// negative sign, the result is `-self`. Even if `self` or `sign` are
      /// NaN, the result is the exact bit pattern of `self` with a sign flipped
      /// based on the sign bit of `sign`.
      ///
      /// This function has been deprecated and renamed to [`flipsign`].
      ///
      /// [`flipsign`]: Self::flipsign
      #[inline]
      #[must_use]
      #[deprecated(since = "1.8.0", note = "renamed to `flipsign`")]
      pub fn flip_signs(self, sign: Self) -> Self {
        self.flipsign(sign)
      }

      /// Fused multiply-add. Computes `(self * a) + b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      ///
      /// This function has been deprecated and renamed to [`fast_mul_add`]. The
      /// name `mul_add` was problematic because the `std` function `mul_add`
      /// has different behavior than this function.
      ///
      /// [`fast_mul_add`]: Self::fast_mul_add
      #[inline]
      #[must_use]
      #[deprecated(since = "1.8.0", note = "renamed to `fast_mul_add`")]
      pub fn mul_add(self, a: Self, b: Self) -> Self {
        self.fast_mul_add(a, b)
      }

      /// Fused multiply-sub. Computes `(self * a) - b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      ///
      /// This function has been deprecated and renamed to [`fast_mul_sub`]. The
      /// name `mul_sub` was problematic because the `std` function `mul_add`
      /// has different behavior than this library's `mul_add` function.
      ///
      /// [`fast_mul_sub`]: Self::fast_mul_sub
      #[inline]
      #[must_use]
      #[deprecated(since = "1.8.0", note = "renamed to `fast_mul_sub`")]
      pub fn mul_sub(self, a: Self, b: Self) -> Self {
        self.fast_mul_sub(a, b)
      }

      /// Fused multiply-negate-add. Computes `-(self * a) + b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      ///
      /// This function has been deprecated and renamed to [`fast_mul_neg_add`].
      /// The name `mul_neg_add` was problematic because the `std` function
      /// `mul_add` has different behavior than this library's `mul_add`
      /// function.
      ///
      /// [`fast_mul_neg_add`]: Self::fast_mul_neg_add
      #[inline]
      #[must_use]
      #[deprecated(since = "1.8.0", note = "renamed to `fast_mul_neg_add`")]
      pub fn mul_neg_add(self, a: Self, b: Self) -> Self {
        self.fast_mul_neg_add(a, b)
      }

      /// Fused multiply-negate-sub. Computes `-(self * a) - b`.
      ///
      /// If there is hardware FMA support, this computes the result with only
      /// one rounding error. If not, this falls back to separate multiply and
      /// add operations, resulting in two rounding errors.
      ///
      /// This function has been deprecated and renamed to [`fast_mul_neg_sub`].
      /// The name `mul_neg_sub` was problematic because the `std` function
      /// `mul_add` has different behavior than this library's `mul_add`
      /// function.
      ///
      /// [`fast_mul_neg_sub`]: Self::fast_mul_neg_sub
      #[inline]
      #[must_use]
      #[deprecated(since = "1.8.0", note = "renamed to `fast_mul_neg_sub`")]
      pub fn mul_neg_sub(self, a: Self, b: Self) -> Self {
        self.fast_mul_neg_sub(a, b)
      }

      /// Raises each element of the number `self` to the scalar floating point
      /// power `n`.
      ///
      /// This function has been deprecated because it raises all elements of
      /// `x` to the same power, even though that brings no performance benefit.
      /// Use `x.powf_simd(splat(n))` instead.
      ///
      /// # Non-deterministic
      ///
      /// The exact result of this function is non-deterministic. This means it
      /// can vary by platform, version, and could even differ within the same
      /// execution from one invocation to the next.
      #[deprecated(since = "1.6.0", note = "use `x.powf_simd(splat(n))` instead")]
      #[inline]
      #[must_use]
      pub fn powf(self, n: $T) -> Self {
        self.powf_simd(Self::splat(n))
      }
    }
  };
}
impl_simd_float!(
    T = f32,
    Simd = f32x4,
    SignedSimd = i32x4,
    UnsignedT = u32,
    UnsignedSimd = u32x4,
);
impl_simd_float!(
    T = f32,
    Simd = f32x8,
    SignedSimd = i32x8,
    UnsignedT = u32,
    UnsignedSimd = u32x8,
);
impl_simd_float!(
    T = f32,
    Simd = f32x16,
    SignedSimd = i32x16,
    UnsignedT = u32,
    UnsignedSimd = u32x16,
);
impl_simd_float!(
    T = f64,
    Simd = f64x2,
    SignedSimd = i64x2,
    UnsignedT = u64,
    UnsignedSimd = u64x2,
);
impl_simd_float!(
    T = f64,
    Simd = f64x4,
    SignedSimd = i64x4,
    UnsignedT = u64,
    UnsignedSimd = u64x4,
);
impl_simd_float!(
    T = f64,
    Simd = f64x8,
    SignedSimd = i64x8,
    UnsignedT = u64,
    UnsignedSimd = u64x8,
);
