//! Defines APIs applicable to SIMD vectors of floats.

use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Sub};

use crate::{
    PrimitiveFloat, Simd, SimdElement, SupportsLaneCount, primitive_traits::NotPrimitiveFloat,
    simd::SimdAssociatedTypes,
};

/*
These commented-out lines contain implementations of traits that should have
separate implementations for SIMD integers and SIMD floats.

Rust currently does not allow two blanket trait implementations based on
disjoint traits. Because of this, these traits have one shared implementation
for both floats and integers (see `simd_float_or_integer.rs`).

If Rust gains support for two blanket trait implementations based on disjoint
traits, these float-only implementations can be uncommented.

impl_unary_operator!(
    T: PrimitiveFloat,
    Neg,
    neg,
    neg_backend,
    /// Computes negation for each element of `self`.
    ///
    /// This always returns the precise result, simply flipping the sign-bit.
);
impl_unary_operator!(
    T: PrimitiveFloat,
    Not,
    not,
    not_backend,
    /// Computes bitwise NOT for each element of `self`.
    ///
    /// This computes bitwise NOT, even though the `!` operator is not
    /// implemented for Rust's primitive floats.
);

impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);
impl_binary_operator!(
    T: PrimitiveFloat,
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
);

impl<T, const N: usize, Rhs> core::iter::Sum<Rhs> for Simd<T, N>
where
    T: PrimitiveFloat + SupportsLaneCount<N>,
    Simd<T, N>: AddAssign<Rhs>,
{
    /// Computes the sum of multiple SIMD vectors for each lane.
    ///
    /// The order of addition is not specified.
    #[inline]
    fn sum<I: Iterator<Item = Rhs>>(iter: I) -> Self {
        let mut total = Self::ZERO;
        for val in iter {
            total += val;
        }
        total
    }
}

impl<T, const N: usize, Rhs> core::iter::Product<Rhs> for Simd<T, N>
where
    T: PrimitiveFloat + SupportsLaneCount<N>,
    Simd<T, N>: MulAssign<Rhs>,
{
    /// Computes the product of multiple SIMD vectors for each lane.
    ///
    /// The order of multiplication is not specified.
    #[inline]
    fn product<I: Iterator<Item = Rhs>>(iter: I) -> Self {
        let mut total = Self::ONE;
        for val in iter {
            total *= val;
        }
        total
    }
}

macro_rules! impl_formatting_trait {
    ($Trait:path) => {
        impl<T, const N: usize> $Trait for Simd<T, N>
        where
            T: PrimitiveFloat + SupportsLaneCount<N>,
        {
            #[allow(clippy::missing_inline_in_public_items)]
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "(")?;
                for (i, x) in self.to_bits().to_array().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    <T::Unsigned as $Trait>::fmt(x, f)?;
                }
                write!(f, ")")
            }
        }
    };
}
impl_formatting_trait!(core::fmt::Binary);
impl_formatting_trait!(core::fmt::LowerHex);
impl_formatting_trait!(core::fmt::Octal);
impl_formatting_trait!(core::fmt::UpperHex);
*/

impl<T, const N: usize> Simd<T, N>
where
    T: PrimitiveFloat + SupportsLaneCount<N>,
{
    /// The size of this SIMD vector in bits.
    pub const BITS: u16 = (size_of::<Self>() * 8) as u16;

    /// A SIMD vector with all elements set to `1.0`.
    pub const ONE: Self = Self::splat(T::ONE);

    /// A SIMD vector with all elements set to `0.5`.
    pub const HALF: Self = Self::splat(T::HALF);

    /// A SIMD vector with all elements set to `0.0`.
    pub const ZERO: Self = Self::splat(T::ZERO);

    /// A SIMD vector with all elements set to [`T::EPSILON`](f32::EPSILON).
    pub const EPSILON: Self = Self::splat(T::EPSILON);

    /// A SIMD vector with all elements set to [`T::MIN`](f32::MIN).
    pub const MIN: Self = Self::splat(T::MIN);

    /// A SIMD vector with all elements set to
    /// [`T::MIN_POSITIVE`](f32::MIN_POSITIVE).
    pub const MIN_POSITIVE: Self = Self::splat(T::MIN_POSITIVE);

    /// A SIMD vector with all elements set to [`T::MAX`](f32::MAX).
    pub const MAX: Self = Self::splat(T::MAX);

    /// A SIMD vector with all elements set to [`T::NAN`](f32::NAN).
    pub const NAN: Self = Self::splat(T::NAN);

    /// A SIMD vector with all elements set to [`T::INFINITY`](f32::INFINITY).
    pub const INFINITY: Self = Self::splat(T::INFINITY);

    /// A SIMD vector with all elements set to
    /// [`T::NEG_INFINITY`](f32::NEG_INFINITY).
    pub const NEG_INFINITY: Self = Self::splat(T::NEG_INFINITY);

    /// A SIMD vector with all elements set to
    /// [Euler's number (e)](core::f32::consts::E).
    pub const E: Self = Self::splat(T::E);

    /// A SIMD vector with all elements set to
    /// [1/π](core::f32::consts::FRAC_1_PI).
    pub const FRAC_1_PI: Self = Self::splat(T::FRAC_1_PI);

    /// A SIMD vector with all elements set to
    /// [2/π](core::f32::consts::FRAC_2_PI).
    pub const FRAC_2_PI: Self = Self::splat(T::FRAC_2_PI);

    /// A SIMD vector with all elements set to
    /// [2/sqrt(π)](core::f32::consts::FRAC_2_SQRT_PI).
    pub const FRAC_2_SQRT_PI: Self = Self::splat(T::FRAC_2_SQRT_PI);

    /// A SIMD vector with all elements set to
    /// [1/sqrt(2)](core::f32::consts::FRAC_1_SQRT_2).
    pub const FRAC_1_SQRT_2: Self = Self::splat(T::FRAC_1_SQRT_2);

    /// A SIMD vector with all elements set to
    /// [π/2](core::f32::consts::FRAC_PI_2).
    pub const FRAC_PI_2: Self = Self::splat(T::FRAC_PI_2);

    /// A SIMD vector with all elements set to
    /// [π/3](core::f32::consts::FRAC_PI_3).
    pub const FRAC_PI_3: Self = Self::splat(T::FRAC_PI_3);

    /// A SIMD vector with all elements set to
    /// [π/4](core::f32::consts::FRAC_PI_4).
    pub const FRAC_PI_4: Self = Self::splat(T::FRAC_PI_4);

    /// A SIMD vector with all elements set to
    /// [π/6](core::f32::consts::FRAC_PI_6).
    pub const FRAC_PI_6: Self = Self::splat(T::FRAC_PI_6);

    /// A SIMD vector with all elements set to
    /// [π/8](core::f32::consts::FRAC_PI_8).
    pub const FRAC_PI_8: Self = Self::splat(T::FRAC_PI_8);

    /// A SIMD vector with all elements set to
    /// [ln(2)](core::f32::consts::LN_2).
    pub const LN_2: Self = Self::splat(T::LN_2);

    /// A SIMD vector with all elements set to
    /// [ln(10)](core::f32::consts::LN_10).
    pub const LN_10: Self = Self::splat(T::LN_10);

    /// A SIMD vector with all elements set to
    /// [log<sub>2</sub>(e)](core::f32::consts::LOG2_E).
    pub const LOG2_E: Self = Self::splat(T::LOG2_E);

    /// A SIMD vector with all elements set to
    /// [log<sub>10</sub>(e)](core::f32::consts::LOG10_E).
    pub const LOG10_E: Self = Self::splat(T::LOG10_E);

    /// A SIMD vector with all elements set to
    /// [log<sub>10</sub>(2)](core::f32::consts::LOG10_2).
    pub const LOG10_2: Self = Self::splat(T::LOG10_2);

    /// A SIMD vector with all elements set to
    /// [log<sub>2</sub>(10)](core::f32::consts::LOG2_10).
    pub const LOG2_10: Self = Self::splat(T::LOG2_10);

    /// A SIMD vector with all elements set to
    /// [Archimedes’ constant (π)](core::f32::consts::PI).
    pub const PI: Self = Self::splat(T::PI);

    /// A SIMD vector with all elements set to
    /// [sqrt(2)](core::f32::consts::SQRT_2).
    pub const SQRT_2: Self = Self::splat(T::SQRT_2);

    /// A SIMD vector with all elements set to
    /// [the full circle constant (τ)](core::f32::consts::TAU).
    ///
    /// Equal to 2π.
    pub const TAU: Self = Self::splat(T::TAU);

    /// Returns a [mask] that checks if each element is NaN.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> Self {
        T::is_nan_backend(self)
    }

    /// Returns a [mask] that checks if each element is infinity (either
    /// positive or negative).
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn is_infinite(self) -> Self {
        T::is_infinite_backend(self)
    }

    /// Returns a [mask] that checks if each element is neither infinite nor
    /// NaN.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> Self {
        T::is_finite_backend(self)
    }

    /// Returns a [mask] that checks if each element has a positive sign,
    /// including `+0.0`, NaNs with positive sign bit and positive infinity.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn is_sign_positive(self) -> Self {
        T::is_sign_positive_backend(self)
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
        T::recip_backend(self)
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
        T::recip_sqrt_backend(self)
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
        self * const { Self::splat(T::RADIANS_TO_DEGREES_RATIO) }
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
        self * const { Self::splat(T::DEGREES_TO_RADIANS_RATIO) }
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
        <T as TSimdFloatBackend<T, N>>::max_backend(self, other)
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
        T::fast_max_backend(self, other)
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
        <T as TSimdFloatBackend<T, N>>::min_backend(self, other)
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
        T::fast_min_backend(self, other)
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
    pub fn reduce_add(self) -> T {
        <T as TSimdFloatBackend<T, N>>::reduce_add_backend(self)
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
    pub fn reduce_mul(self) -> T {
        <T as TSimdFloatBackend<T, N>>::reduce_mul_backend(self)
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
    pub fn reduce_max(self) -> T {
        <T as TSimdFloatBackend<T, N>>::reduce_max_backend(self)
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
    pub fn fast_reduce_max(self) -> T {
        T::fast_reduce_max_backend(self)
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
    pub fn reduce_min(self) -> T {
        <T as TSimdFloatBackend<T, N>>::reduce_min_backend(self)
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
    pub fn fast_reduce_min(self) -> T {
        T::fast_reduce_min_backend(self)
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
        (self + other) * T::HALF
    }

    /// Raw transmutation to unsigned integer vector.
    ///
    /// Note that this function preserves the *bitwise* value, and not the
    /// numeric value.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> Simd<T::Unsigned, N> {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory. They have the same size since `T` and
        // `T::Unsigned` must have the same size.
        unsafe { core::mem::transmute_copy::<Simd<T, N>, Simd<T::Unsigned, N>>(&self) }
    }

    /// Raw transmutation from unsigned integer vector.
    ///
    /// Note that this function preserves the *bitwise* value, and not the
    /// numeric value.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: Simd<T::Unsigned, N>) -> Self {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory. They have the same size since `T` and
        // `T::Unsigned` must have the same size.
        unsafe { core::mem::transmute_copy::<Simd<T::Unsigned, N>, Simd<T, N>>(&bits) }
    }

    /// Creates a SIMD vector from a native-endian byte array.
    ///
    /// Note that this takes `[u8; size_of::<Self>()]`, but it is
    /// written as `Self::Bytes` due to language limitations.
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
    pub const fn from_ne_bytes(bytes: <Self as SimdAssociatedTypes>::Bytes) -> Self {
        // SAFETY: `Self` accepts all bit-patterns and only contains
        // initialized memory, and the byte array has the same size.
        unsafe { core::mem::transmute_copy::<<Self as SimdAssociatedTypes>::Bytes, Self>(&bytes) }
    }

    /// Returns the raw memory representation of `self` as a native-endian
    /// byte array.
    ///
    /// Note that this returns `[u8; size_of::<Self>()]`, but it is
    /// written as `Self::Bytes` due to language limitations.
    ///
    /// This is the inverse of [`from_ne_bytes`].
    ///
    /// [`from_ne_bytes`]: Self::from_ne_bytes
    #[inline]
    #[must_use]
    pub const fn to_ne_bytes(self) -> <Self as SimdAssociatedTypes>::Bytes {
        // SAFETY: `Self` accepts all bit-patterns and only contains
        // initialized memory, and the byte array has the same size.
        unsafe { core::mem::transmute_copy::<Self, <Self as SimdAssociatedTypes>::Bytes>(&self) }
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
        <T as TSimdFloatBackend<T, N>>::bitselect_backend(self, if_one, if_zero)
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
        <T as TSimdFloatBackend<T, N>>::clamp_backend(self, min, max)
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
        T::fast_clamp_backend(self, min, max)
    }

    /// Computes the absolute value of `self`.
    ///
    /// This function always returns the precise result.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        <T as TSimdFloatBackend<T, N>>::abs_backend(self)
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
        Self::splat(T::NEG_ZERO).bitselect(sign, self)
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
        self ^ (sign & Self::from(T::NEG_ZERO))
    }

    /// Returns the largest integer less than or equal to each input element.
    ///
    /// This function always returns the precise result.
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        T::floor_backend(self)
    }

    /// Returns the smallest integer greater than or equal to each input
    /// element.
    ///
    /// This function always returns the precise result.
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        T::ceil_backend(self)
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
        T::round_backend(self)
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
    pub fn round_int(self) -> Simd<T::Signed, N> {
        T::round_int_backend(self)
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
    pub fn fast_round_int(self) -> Simd<T::Signed, N> {
        T::fast_round_int_backend(self)
    }

    /// Returns the nearest integer to each input element. Rounds half-way
    /// cases to the number with an even least significant digit.
    ///
    /// This function always returns the precise result.
    #[inline]
    #[must_use]
    pub fn round_ties_even(self) -> Self {
        T::round_ties_even_backend(self)
    }

    /// Returns the integer part of each input element. This means that
    /// non-integer numbers are always truncated towards zero.
    ///
    /// This function always returns the precise result.
    #[inline]
    #[must_use]
    pub fn trunc(self) -> Self {
        T::trunc_backend(self)
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
    pub fn trunc_int(self) -> Simd<T::Signed, N> {
        T::trunc_int_backend(self)
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
    pub fn fast_trunc_int(self) -> Simd<T::Signed, N> {
        T::fast_trunc_int_backend(self)
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
        T::fast_mul_add_backend(self, a, b)
    }

    /// Fused multiply-sub. Computes `(self * a) - b`.
    ///
    /// If there is hardware FMA support, this computes the result with only
    /// one rounding error. If not, this falls back to separate multiply and
    /// add operations, resulting in two rounding errors.
    #[inline]
    #[must_use]
    pub fn fast_mul_sub(self, a: Self, b: Self) -> Self {
        T::fast_mul_sub_backend(self, a, b)
    }

    /// Fused multiply-negate-add. Computes `-(self * a) + b`.
    ///
    /// If there is hardware FMA support, this computes the result with only
    /// one rounding error. If not, this falls back to separate multiply and
    /// add operations, resulting in two rounding errors.
    #[inline]
    #[must_use]
    pub fn fast_mul_neg_add(self, a: Self, b: Self) -> Self {
        T::fast_mul_neg_add_backend(self, a, b)
    }

    /// Fused multiply-negate-sub. Computes `-(self * a) - b`.
    ///
    /// If there is hardware FMA support, this computes the result with only
    /// one rounding error. If not, this falls back to separate multiply and
    /// add operations, resulting in two rounding errors.
    #[inline]
    #[must_use]
    pub fn fast_mul_neg_sub(self, a: Self, b: Self) -> Self {
        T::fast_mul_neg_sub_backend(self, a, b)
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
        (self % rhs).simd_lt(Self::ZERO).select(
            rhs.simd_gt(Self::ZERO).select(q - Self::ONE, q + Self::ONE),
            q,
        )
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
        T::powf_simd_backend(self, n)
    }

    /// Returns the square root of a number for each input element.
    ///
    /// Returns NaN if `self` is a negative number other than `-0.0`.
    ///
    /// This function always returns the precise result.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        T::sqrt_backend(self)
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
        T::exp_backend(self)
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
        T::exp2_backend(self)
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
        T::ln_backend(self)
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
        T::cbrt_backend(self)
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
        T::asin_backend(self)
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
        T::acos_backend(self)
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
        T::atan_backend(self)
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
        T::atan2_backend(self, other)
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
        T::sin_cos_backend(self)
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
        T::asin_acos_backend(self)
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
        T::exp_m1_backend(self)
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
        T::ln_1p_backend(self)
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
        T::sinh_backend(self)
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
        T::cosh_backend(self)
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
        T::tanh_backend(self)
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
    pub fn powf(self, n: T) -> Self {
        self.powf_simd(Self::splat(n))
    }
}

macro_rules! float {
    ($T:ident) => {
        impl_binary_operator_for_scalar!(
            $T,
            Add,
            add,
            AddAssign,
            add_assign,
            $T::add_backend,
            /// Computes addition for the scalar `self` and each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This always returns the precise result.
        );
        impl_binary_operator_for_scalar!(
            $T,
            Sub,
            sub,
            SubAssign,
            sub_assign,
            $T::sub_backend,
            /// Computes subtraction for the scalar `self` and each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This always returns the precise result.
        );
        impl_binary_operator_for_scalar!(
            $T,
            Mul,
            mul,
            MulAssign,
            mul_assign,
            $T::mul_backend,
            /// Computes multiplication for the scalar `self` and each element of
            /// `rhs`, returning a SIMD vector.
            ///
            /// This always returns the precise result.
        );
        impl_binary_operator_for_scalar!(
            $T,
            Div,
            div,
            DivAssign,
            div_assign,
            $T::div_backend,
            /// Computes division for the scalar `self` and each element of
            /// `rhs`, returning a SIMD vector.
            ///
            /// This always returns the precise result.
        );
        impl_binary_operator_for_scalar!(
            $T,
            Rem,
            rem,
            RemAssign,
            rem_assign,
            $T::rem_backend,
            /// Computes the remainder for the scalar `self` and each element of
            /// `rhs`, returning a SIMD vector.
            ///
            /// This always returns the precise result.
        );
        impl_binary_operator_for_scalar!(
            $T,
            BitAnd,
            bitand,
            BitAndAssign,
            bitand_assign,
            $T::bitand_backend,
            /// Computes bitwise AND for the scalar `self` and each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This computes bitwise AND, even though the `&` operator is not
            /// implemented for Rust's primitive floats.
        );
        impl_binary_operator_for_scalar!(
            $T,
            BitOr,
            bitor,
            BitOrAssign,
            bitor_assign,
            $T::bitor_backend,
            /// Computes bitwise OR for the scalar `self` and each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This computes bitwise OR, even though the `|` operator is not
            /// implemented for Rust's primitive floats.
        );
        impl_binary_operator_for_scalar!(
            $T,
            BitXor,
            bitxor,
            BitXorAssign,
            bitxor_assign,
            $T::bitxor_backend,
            /// Computes bitwise XOR for the scalar `self` and each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This computes bitwise XOR, even though the `^` operator is not
            /// implemented for Rust's primitive floats.
        );
    };
}
float!(f32);
float!(f64);

generate_t_backend_trait! {
#[generate_t_backend_trait(TSimdFloatBackend)]
/// Defines the internal implementations of functions applicable to SIMD vectors
/// of floats.
pub(crate) trait SimdFloatBackend<T, const N: usize>: Sized
where
    T: SimdElement + SupportsLaneCount<N>,
{
    fn neg_backend(self) -> Simd<T, N>;

    fn not_backend(self) -> Simd<T, N>;

    fn add_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn sub_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn mul_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn div_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn rem_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn bitand_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn bitor_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn bitxor_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn reduce_add_backend(self) -> T;

    fn reduce_mul_backend(self) -> T;

    fn bitselect_backend(self, if_one: Simd<T, N>, if_zero: Simd<T, N>) -> Simd<T, N>;

    #[expect(clippy::wrong_self_convention)]
    fn is_nan_backend(self) -> Simd<T, N>;

    #[expect(clippy::wrong_self_convention)]
    fn is_infinite_backend(self) -> Simd<T, N>;

    #[expect(clippy::wrong_self_convention)]
    fn is_finite_backend(self) -> Simd<T, N>;

    #[expect(clippy::wrong_self_convention)]
    fn is_sign_positive_backend(self) -> Simd<T, N>;

    fn recip_backend(self) -> Simd<T, N>;

    fn recip_sqrt_backend(self) -> Simd<T, N>;

    fn max_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn fast_max_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn min_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn fast_min_backend(self, rhs: Simd<T, N>) -> Simd<T, N>;

    fn clamp_backend(self, min: Simd<T, N>, max: Simd<T, N>) -> Simd<T, N>;

    fn fast_clamp_backend(self, min: Simd<T, N>, max: Simd<T, N>) -> Simd<T, N>;

    fn reduce_max_backend(self) -> T;

    fn fast_reduce_max_backend(self) -> T;

    fn reduce_min_backend(self) -> T;

    fn fast_reduce_min_backend(self) -> T;

    fn abs_backend(self) -> Simd<T, N>;

    fn floor_backend(self) -> Simd<T, N>;

    fn ceil_backend(self) -> Simd<T, N>;

    fn round_backend(self) -> Simd<T, N>;

    fn round_int_backend(self) -> Simd<T::Signed, N>;

    fn fast_round_int_backend(self) -> Simd<T::Signed, N>;

    fn round_ties_even_backend(self) -> Simd<T, N>;

    fn trunc_backend(self) -> Simd<T, N>;

    fn trunc_int_backend(self) -> Simd<T::Signed, N>;

    fn fast_trunc_int_backend(self) -> Simd<T::Signed, N>;

    fn fast_mul_add_backend(self, a: Simd<T, N>, b: Simd<T, N>) -> Simd<T, N>;

    fn fast_mul_sub_backend(self, a: Simd<T, N>, b: Simd<T, N>) -> Simd<T, N>;

    fn fast_mul_neg_add_backend(self, a: Simd<T, N>, b: Simd<T, N>) -> Simd<T, N>;

    fn fast_mul_neg_sub_backend(self, a: Simd<T, N>, b: Simd<T, N>) -> Simd<T, N>;

    fn powf_simd_backend(self, n: Simd<T, N>) -> Simd<T, N>;

    fn sqrt_backend(self) -> Simd<T, N>;

    fn exp_backend(self) -> Simd<T, N>;

    fn exp2_backend(self) -> Simd<T, N>;

    fn ln_backend(self) -> Simd<T, N>;

    fn cbrt_backend(self) -> Simd<T, N>;

    fn asin_backend(self) -> Simd<T, N>;

    fn acos_backend(self) -> Simd<T, N>;

    fn atan_backend(self) -> Simd<T, N>;

    fn atan2_backend(self, x: Simd<T, N>) -> Simd<T, N>;

    fn sin_cos_backend(self) -> (Simd<T, N>, Simd<T, N>);

    fn asin_acos_backend(self) -> (Simd<T, N>, Simd<T, N>);

    fn exp_m1_backend(self) -> Simd<T, N>;

    fn ln_1p_backend(self) -> Simd<T, N>;

    fn sinh_backend(self) -> Simd<T, N>;

    fn cosh_backend(self) -> Simd<T, N>;

    fn tanh_backend(self) -> Simd<T, N>;
}
}

/// In order for the generic SIMD float API to work, [`SimdFloatBackend`] is
/// used as a super-trait of [`SupportedSimd`].
///
/// This is incorrect logically, since [`SimdFloatBackend`] is only applicable
/// to SIMD vectors of floats. However this seems to be the only way to do it on
/// stable Rust.
///
/// This means that non-floats do need some implementation of
/// [`SimdFloatBackend`]. This block is a dummy implementation for all
/// non-floats that fills functions with [`unreachable!`].
impl<T, const N: usize> SimdFloatBackend<T, N> for Simd<T, N>
where
    T: SimdElement + NotPrimitiveFloat + SupportsLaneCount<N>,
{
    fn abs_backend(self) -> Self {
        unreachable!()
    }
    fn acos_backend(self) -> Self {
        unreachable!()
    }
    fn add_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn asin_acos_backend(self) -> (Self, Self) {
        unreachable!()
    }
    fn asin_backend(self) -> Self {
        unreachable!()
    }
    fn atan2_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn atan_backend(self) -> Self {
        unreachable!()
    }
    fn bitand_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn bitor_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn bitselect_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn bitxor_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn cbrt_backend(self) -> Self {
        unreachable!()
    }
    fn ceil_backend(self) -> Self {
        unreachable!()
    }
    fn clamp_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn cosh_backend(self) -> Self {
        unreachable!()
    }
    fn div_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn exp2_backend(self) -> Self {
        unreachable!()
    }
    fn exp_backend(self) -> Self {
        unreachable!()
    }
    fn exp_m1_backend(self) -> Self {
        unreachable!()
    }
    fn fast_clamp_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_max_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_min_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_mul_add_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_mul_neg_add_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_mul_neg_sub_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_mul_sub_backend(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    fn fast_reduce_max_backend(self) -> T {
        unreachable!()
    }
    fn fast_reduce_min_backend(self) -> T {
        unreachable!()
    }
    fn fast_round_int_backend(self) -> Simd<T::Signed, N> {
        unreachable!()
    }
    fn fast_trunc_int_backend(self) -> Simd<T::Signed, N> {
        unreachable!()
    }
    fn floor_backend(self) -> Self {
        unreachable!()
    }
    fn is_finite_backend(self) -> Self {
        unreachable!()
    }
    fn is_infinite_backend(self) -> Self {
        unreachable!()
    }
    fn is_nan_backend(self) -> Self {
        unreachable!()
    }
    fn is_sign_positive_backend(self) -> Self {
        unreachable!()
    }
    fn ln_1p_backend(self) -> Self {
        unreachable!()
    }
    fn ln_backend(self) -> Self {
        unreachable!()
    }
    fn max_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn min_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn mul_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn neg_backend(self) -> Self {
        unreachable!()
    }
    fn not_backend(self) -> Self {
        unreachable!()
    }
    fn powf_simd_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn recip_backend(self) -> Self {
        unreachable!()
    }
    fn recip_sqrt_backend(self) -> Self {
        unreachable!()
    }
    fn reduce_add_backend(self) -> T {
        unreachable!()
    }
    fn reduce_max_backend(self) -> T {
        unreachable!()
    }
    fn reduce_min_backend(self) -> T {
        unreachable!()
    }
    fn reduce_mul_backend(self) -> T {
        unreachable!()
    }
    fn rem_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn round_backend(self) -> Self {
        unreachable!()
    }
    fn round_int_backend(self) -> Simd<T::Signed, N> {
        unreachable!()
    }
    fn round_ties_even_backend(self) -> Self {
        unreachable!()
    }
    fn sin_cos_backend(self) -> (Self, Self) {
        unreachable!()
    }
    fn sinh_backend(self) -> Self {
        unreachable!()
    }
    fn sqrt_backend(self) -> Self {
        unreachable!()
    }
    fn sub_backend(self, _: Self) -> Self {
        unreachable!()
    }
    fn tanh_backend(self) -> Self {
        unreachable!()
    }
    fn trunc_backend(self) -> Self {
        unreachable!()
    }
    fn trunc_int_backend(self) -> Simd<T::Signed, N> {
        unreachable!()
    }
}
