//! Defines primitive traits used to expose primitive-specific SIMD
//! functionality.
//!
//! It would be appropriate to define these traits in `simd_float.rs`,
//! `simd_integer.rs`, `simd_signed.rs` and `simd_unsigned.rs`, but
//! due to Rust type-system limitations, the definitions of these traits are
//! tightly coupled and easier to do in one module. In the future, these traits
//! may be moved back to the `simd_*.rs` modules.

use crate::{MaskOps, SimdElement};

/// A sealed trait for all primitive floating-point types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveFloat`, update the documentation of this trait to explain its
/// intended usage.
pub trait PrimitiveFloat: SimdElement + MaskOps {}

/// A sealed trait for all primitive integer types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveInteger`, update the documentation of this trait to explain its
/// intended usage.
pub trait PrimitiveInteger: SimdElement + MaskOps {}

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
pub trait PrimitiveSigned: PrimitiveInteger<Signed = Self> {}

/// A sealed trait for all primitive unsigned integer types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveSigned`, update the documentation of this trait to explain its
/// intended usage.
pub trait PrimitiveUnsigned: PrimitiveInteger<Unsigned = Self> {}

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

impl PrimitiveFloat for f32 {}
impl PrimitiveFloat for f64 {}

impl PrimitiveInteger for i8 {}
impl PrimitiveInteger for i16 {}
impl PrimitiveInteger for i32 {}
impl PrimitiveInteger for i64 {}
impl PrimitiveInteger for i128 {}
impl PrimitiveInteger for isize {}
impl PrimitiveInteger for u8 {}
impl PrimitiveInteger for u16 {}
impl PrimitiveInteger for u32 {}
impl PrimitiveInteger for u64 {}
impl PrimitiveInteger for u128 {}
impl PrimitiveInteger for usize {}

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

impl PrimitiveSigned for i8 {}
impl PrimitiveSigned for i16 {}
impl PrimitiveSigned for i32 {}
impl PrimitiveSigned for i64 {}
impl PrimitiveSigned for i128 {}
impl PrimitiveSigned for isize {}

impl PrimitiveUnsigned for u8 {}
impl PrimitiveUnsigned for u16 {}
impl PrimitiveUnsigned for u32 {}
impl PrimitiveUnsigned for u64 {}
impl PrimitiveUnsigned for u128 {}
impl PrimitiveUnsigned for usize {}

impl NotPrimitiveFloat for i8 {}
impl NotPrimitiveFloat for i16 {}
impl NotPrimitiveFloat for i32 {}
impl NotPrimitiveFloat for i64 {}
impl NotPrimitiveFloat for i128 {}
impl NotPrimitiveFloat for isize {}
impl NotPrimitiveFloat for u8 {}
impl NotPrimitiveFloat for u16 {}
impl NotPrimitiveFloat for u32 {}
impl NotPrimitiveFloat for u64 {}
impl NotPrimitiveFloat for u128 {}
impl NotPrimitiveFloat for usize {}

impl NotPrimitiveInteger for f32 {}
impl NotPrimitiveInteger for f64 {}

impl NotPrimitiveSigned for f32 {}
impl NotPrimitiveSigned for f64 {}
impl NotPrimitiveSigned for u8 {}
impl NotPrimitiveSigned for u16 {}
impl NotPrimitiveSigned for u32 {}
impl NotPrimitiveSigned for u64 {}
impl NotPrimitiveSigned for u128 {}
impl NotPrimitiveSigned for usize {}

impl NotPrimitiveUnsigned for f32 {}
impl NotPrimitiveUnsigned for f64 {}
impl NotPrimitiveUnsigned for i8 {}
impl NotPrimitiveUnsigned for i16 {}
impl NotPrimitiveUnsigned for i32 {}
impl NotPrimitiveUnsigned for i64 {}
impl NotPrimitiveUnsigned for i128 {}
impl NotPrimitiveUnsigned for isize {}
