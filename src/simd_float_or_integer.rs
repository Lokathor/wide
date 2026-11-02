//! A module containing trait implementations that, due to Rust language
//! limitations (not allowing two blanket implementations based on disjoint
//! traits), apply for both SIMD floats and SIMD integers via the
//! [`PrimitiveFloatOrInteger`] trait.
//!
//! Each trait implementation here is meant to be two separate trait implementations:
//!
//! - One in `simd_float.rs` with `T: PrimitiveFloat`
//! - One in `simd_integer.rs` with `T: PrimitiveInteger`
//!
//! If Rust gains support for two blanket implementations based on disjoint
//! traits, this module can be removed and the implementations can be
//! separated.

use core::ops::{Neg, Not};

use crate::{
    Simd, SupportedSimd, primitive_traits::PrimitiveFloatOrInteger, simd_float::SimdFloatBackend,
    simd_integer::SimdIntegerBackend,
};

impl_unary_operator!(
    T: PrimitiveFloatOrInteger,
    Neg,
    neg,
    neg_backend,
    /// Computes negation for each element of `self`.
    ///
    /// For floats, this always returns the precise result, simply flipping the
    /// sign-bit. For integers, this computes wrapping negation.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);
impl_unary_operator!(
    T: PrimitiveFloatOrInteger,
    Not,
    not,
    not_backend,
    /// Computes bitwise NOT for each element of `self`.
    ///
    /// For floats, this computes bitwise NOT, even though the `!` operator is
    /// not implemented for Rust's primitive floats.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);

/// Implementations of operator traits do not know whether to pick backend
/// functions from [`SimdFloatBackend`] or [`SimdIntegerBackend`], so these
/// functions wrap the backend functions, calling the correct one.
///
/// TODO(generics): Should we worry that these `if` statements won't get
/// optimized away?
#[expect(private_bounds)]
impl<T, const N: usize> Simd<T, N>
where
    T: PrimitiveFloatOrInteger,
    Simd<T, N>: SupportedSimd,
{
    #[inline]
    fn neg_backend(self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::neg_backend(self)
        } else {
            SimdIntegerBackend::neg_backend(self)
        }
    }

    #[inline]
    fn not_backend(self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::not_backend(self)
        } else {
            SimdIntegerBackend::not_backend(self)
        }
    }
}
