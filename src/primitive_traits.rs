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
/// Implemented for [`f32`] and [`f64`].
///
/// This trait is used to access SIMD float functionality in contexts generic
/// over `T`:
///
/// ```rust
/// # use wide::{Simd, SimdElement, SupportedSimd, PrimitiveFloat};
/// #
/// fn example<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveFloat,
///     Simd<T, N>: SupportedSimd,
///     Simd<T::Unsigned, N>: SupportedSimd,
///     Simd<T::Signed, N>: SupportedSimd,
/// {
///     // Use SIMD float functionality
///     x.sin().abs().acos() + x
/// }
/// ```
///
/// This trait does not aim to expose scalar float functionality, only to enable
/// access to [`Simd<T, N>`] functionality. For scalar functionality, consider
/// combining this trait with the [`num-primitive`] crate.
///
/// [`Simd<T, N>`]: crate::Simd
/// [`num-primitive`]: https://crates.io/crates/num-primitive
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
/// Implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`i128`], [`isize`],
/// [`u8`], [`u16`], [`u32`], [`u64`], [`u128`] and [`usize`].
///
/// This trait is used to access SIMD integer functionality in contexts generic
/// over `T`:
///
/// ```rust
/// # use wide::{Simd, SimdElement, SupportedSimd, PrimitiveInteger};
/// #
/// fn example<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveInteger,
///     Simd<T, N>: SupportedSimd,
///     Simd<T::Unsigned, N>: SupportedSimd,
///     Simd<T::Signed, N>: SupportedSimd,
/// {
///     // Use SIMD integer functionality
///     x.saturating_add(x) * x
/// }
/// ```
///
/// This trait does not aim to expose scalar integer functionality, only to
/// enable access to [`Simd<T, N>`] functionality. For scalar functionality,
/// consider combining this trait with the [`num-primitive`] crate.
///
/// # Important limitation
///
/// Due to Rust language limitations, SIMD integer items whose name conflicts
/// with a SIMD float item's name (e.g., `reduce_add`) cannot be used in generic
/// contexts.
///
/// In documentation, such items are still shown as generic over `T`, even
/// though in reality those items are duplicated for each concrete `T` type.
/// This is done in order to avoid duplicated documentation, which is hard to
/// navigate.
///
/// [`Simd<T, N>`]: crate::Simd
/// [`num-primitive`]: https://crates.io/crates/num-primitive
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
/// Implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`u8`], [`u16`], [`u32`]
/// and [`u64`].
///
/// This is used for [`Simd::widening_mul`](crate::Simd::widening_mul).
pub trait HasDoubleWidth: PrimitiveInteger {
    /// The primitive integer with double the width of `Self` and the same
    /// signedness.
    type DoubleWidth: PrimitiveInteger;
}

/// A sealed trait for all primitive signed integer types.
///
/// Implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`i128`] and [`isize`].
///
/// This trait is used to access SIMD signed integer functionality in contexts
/// generic over `T`:
///
/// ```rust
/// # use wide::{Simd, SimdElement, SupportedSimd, PrimitiveSigned};
/// #
/// fn example<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveSigned,
///     Simd<T, N>: SupportedSimd,
///     Simd<T::Unsigned, N>: SupportedSimd,
///     Simd<T::Signed, N>: SupportedSimd,
/// {
///     // Use SIMD signed integer functionality
///     x.saturating_abs().add_unsigned(x.cast_unsigned())
/// }
/// ```
///
/// This trait does not aim to expose scalar signed integer functionality, only
/// to enable access to [`Simd<T, N>`] functionality. For scalar functionality,
/// consider combining this trait with the [`num-primitive`] crate.
///
/// # Important limitation
///
/// Due to Rust language limitations, SIMD signed integer items whose name
/// conflicts with a SIMD float item's name (e.g., `abs`) cannot be used in
/// generic contexts.
///
/// In documentation, such items are still shown as generic over `T`, even
/// though in reality those items are duplicated for each concrete `T` type.
/// This is done in order to avoid duplicated documentation, which is hard to
/// navigate.
///
/// [`Simd<T, N>`]: crate::Simd
/// [`num-primitive`]: https://crates.io/crates/num-primitive
#[expect(private_bounds)]
pub trait PrimitiveSigned:
    PrimitiveInteger<Signed = Self> + SignedUnsignedDisjointMarker<Marker = i32>
{
}

/// A sealed trait for all primitive unsigned integer types.
///
/// Implemented for [`u8`], [`u16`], [`u32`], [`u64`], [`u128`] and [`usize`].
///
/// This trait is used to access SIMD unsigned integer functionality in contexts
/// generic over `T`:
///
/// ```rust
/// # use wide::{Simd, SimdElement, SupportedSimd, PrimitiveUnsigned};
/// #
/// fn example<T, const N: usize>(x: Simd<T, N>) -> Simd<T, N>
/// where
///     T: PrimitiveUnsigned,
///     Simd<T, N>: SupportedSimd,
///     Simd<T::Unsigned, N>: SupportedSimd,
///     Simd<T::Signed, N>: SupportedSimd,
/// {
///     // Use SIMD signed integer functionality
///     x.saturating_add_signed(x.cast_signed())
/// }
/// ```
///
/// This trait does not aim to expose scalar unsigned integer functionality,
/// only to enable access to [`Simd<T, N>`] functionality. For scalar
/// functionality, consider combining this trait with the [`num-primitive`]
/// crate.
///
/// [`Simd<T, N>`]: crate::Simd
/// [`num-primitive`]: https://crates.io/crates/num-primitive
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
