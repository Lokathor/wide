//! Defines primitive traits used to expose primitive-specific SIMD
//! functionality.
//!
//! It would be appropriate to define these traits in `simd_float.rs`,
//! `simd_integer.rs`, `simd_signed.rs` and `simd_unsigned.rs`, but
//! due to Rust type-system limitations, the definitions of these traits are
//! tightly coupled and easier to do in one module. In the future, these traits
//! may be moved back to the `simd_*.rs` modules.

use crate::{
    MaskOps, SimdElement,
    utils::{PrimitiveFloatUtils, PrimitiveIntegerUtils},
};

/// A sealed trait for all primitive floating-point types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveFloat`, update the documentation of this trait to explain its
/// intended usage.
#[expect(private_bounds)]
pub trait PrimitiveFloat:
    SimdElement
    + MaskOps
    + PartialOrd
    + PrimitiveFloatOrInteger
    + FloatIntegerDisjointMarker<Marker = f32>
    + PrimitiveFloatUtils
{
}

/// A sealed trait for all primitive integer types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveInteger`, update the documentation of this trait to explain its
/// intended usage.
#[expect(private_bounds)]
pub trait PrimitiveInteger:
    SimdElement
    + MaskOps
    + Ord
    + PrimitiveFloatOrInteger
    + FloatIntegerDisjointMarker<Marker = u32>
    + PrimitiveIntegerUtils
{
}

/// A sealed trait for primitive integer types that have a double-width variant.
///
/// This is used for [`Simd::widening_mul`](crate::Simd::widening_mul).
pub trait HasDoubleWidth: PrimitiveInteger {
    /// The primitive integer with double the width of `Self` and the same
    /// signedness.
    type DoubleWidth: PrimitiveInteger;
}

/// A sealed trait for all primitive signed integer types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveSigned`, update the documentation of this trait to explain its
/// intended usage.
#[expect(private_bounds)]
pub trait PrimitiveSigned:
    PrimitiveInteger<Signed = Self> + SignedUnsignedDisjointMarker<Marker = i32>
{
}

/// A sealed trait for all primitive unsigned integer types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveSigned`, update the documentation of this trait to explain its
/// intended usage.
#[expect(private_bounds)]
pub trait PrimitiveUnsigned:
    PrimitiveInteger<Unsigned = Self> + SignedUnsignedDisjointMarker<Marker = u32>
{
}

/// An internal trait that represents `PrimitiveFloat OR PrimitiveInteger`.
///
/// This is used to implement operator traits, that due to Rust language
/// limitations (not allowing two disjoint blanket implementations), are
/// implemented with one generic implementation for [`PrimitiveFloatOrInteger`]
/// rather than two separate implementations for [`PrimitiveFloat`] and
/// [`PrimitiveInteger`].
pub(crate) trait PrimitiveFloatOrInteger: SimdElement {
    /// `true` for [`PrimitiveFloat`] types and `false` for [`PrimitiveInteger`]
    /// types.
    ///
    /// This is used to select between the backend functions of
    /// [`SimdFloatBackend`](crate::simd_float::SimdFloatBackend) and
    /// [`SimdIntegerBackend`](crate::simd_integer::SimdIntegerBackend).
    const IS_FLOAT: bool;

    /// Numeric `0`.
    const ZERO: Self;

    /// Numeric `1`.
    const ONE: Self;

    /// Converts floats to unsigned integers via `to_bits` and keeps integers
    /// unchanged (returns the original types).
    ///
    /// See `impl_formatting_traits` in `simd_float_or_integer.rs`.
    fn fmt_cast(
        self,
    ) -> impl core::fmt::Binary + core::fmt::LowerHex + core::fmt::Octal + core::fmt::UpperHex;
}

/// A marker trait representing `!PrimitiveFloat` used for an internal
/// type-system hack in `simd_float.rs`.
pub(crate) trait NotPrimitiveFloat {}

/// A marker trait representing `!PrimitiveInteger` used for an internal
/// type-system hack in `simd_integer.rs`.
pub(crate) trait NotPrimitiveInteger {}

/// A marker trait representing `!PrimitiveSigned` used for an internal
/// type-system hack in `simd_signed.rs`.
pub(crate) trait NotPrimitiveSigned {}

/// A marker trait representing `!PrimitiveUnsigned` used for an internal
/// type-system hack in `simd_unsigned.rs`.
pub(crate) trait NotPrimitiveUnsigned {}

/// A trait used to tell the compiler that [`PrimitiveFloat`] and
/// [`PrimitiveInteger`] are disjoint.
///
/// This makes the compiler deny `T: PrimitiveFloat + PrimitiveInteger`
/// statements. In the future, the type system may use this to allow coexisting
/// blanket trait implementations over `T: PrimitiveFloat` and
/// `T: PrimitiveInteger`, which will be very useful for this library.
trait FloatIntegerDisjointMarker {
    /// Use [`f32`] to mark floats and [`u32`] to mark integers.
    type Marker;
}

/// A trait used to tell the compiler that [`PrimitiveSigned`] and
/// [`PrimitiveUnsigned`] are disjoint.
///
/// This makes the compiler deny `T: PrimitiveSigned + PrimitiveUnsigned`
/// statements. In the future, the type system may use this to allow coexisting
/// blanket trait implementations over `T: PrimitiveSigned` and
/// `T: PrimitiveUnsigned`.
trait SignedUnsignedDisjointMarker {
    /// Use [`i32`] to mark signed and [`u32`] to mark unsigned.
    type Marker;
}

macro_rules! float {
    ($T:ident) => {
        impl PrimitiveFloat for $T {}

        impl PrimitiveFloatOrInteger for $T {
            const IS_FLOAT: bool = true;

            const ZERO: Self = 0.0;

            const ONE: Self = 1.0;

            #[inline]
            fn fmt_cast(
                self,
            ) -> impl core::fmt::Binary + core::fmt::LowerHex + core::fmt::Octal + core::fmt::UpperHex {
                self.to_bits()
            }
        }

        impl NotPrimitiveInteger for $T {}

        impl NotPrimitiveSigned for $T {}

        impl NotPrimitiveUnsigned for $T {}

        impl FloatIntegerDisjointMarker for $T {
            type Marker = f32;
        }
    };
}
float!(f32);
float!(f64);

macro_rules! integer {
    ($T:ident) => {
        impl PrimitiveInteger for $T {}

        impl PrimitiveFloatOrInteger for $T {
            const IS_FLOAT: bool = false;

            const ZERO: Self = 0;

            const ONE: Self = 1;

            #[inline]
            fn fmt_cast(
                self,
            ) -> impl core::fmt::Binary + core::fmt::LowerHex + core::fmt::Octal + core::fmt::UpperHex {
                self
            }
        }

        impl NotPrimitiveFloat for $T {}

        impl FloatIntegerDisjointMarker for $T {
            type Marker = u32;
        }
    };
}
integer!(i8);
integer!(i16);
integer!(i32);
integer!(i64);
integer!(i128);
integer!(isize);
integer!(u8);
integer!(u16);
integer!(u32);
integer!(u64);
integer!(u128);
integer!(usize);

impl HasDoubleWidth for i8 {
    type DoubleWidth = i16;
}
impl HasDoubleWidth for i16 {
    type DoubleWidth = i32;
}
impl HasDoubleWidth for i32 {
    type DoubleWidth = i64;
}
impl HasDoubleWidth for i64 {
    type DoubleWidth = i128;
}
impl HasDoubleWidth for u8 {
    type DoubleWidth = u16;
}
impl HasDoubleWidth for u16 {
    type DoubleWidth = u32;
}
impl HasDoubleWidth for u32 {
    type DoubleWidth = u64;
}
impl HasDoubleWidth for u64 {
    type DoubleWidth = u128;
}

macro_rules! signed {
    ($T:ident) => {
        impl PrimitiveSigned for $T {}

        impl NotPrimitiveUnsigned for $T {}

        impl SignedUnsignedDisjointMarker for $T {
            type Marker = i32;
        }
    };
}
signed!(i8);
signed!(i16);
signed!(i32);
signed!(i64);
signed!(i128);
signed!(isize);

macro_rules! unsigned {
    ($T:ident) => {
        impl PrimitiveUnsigned for $T {}

        impl NotPrimitiveSigned for $T {}

        impl SignedUnsignedDisjointMarker for $T {
            type Marker = u32;
        }
    };
}
unsigned!(u8);
unsigned!(u16);
unsigned!(u32);
unsigned!(u64);
unsigned!(u128);
unsigned!(usize);
