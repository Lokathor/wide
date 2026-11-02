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

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

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

impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    Add,
    add,
    AddAssign,
    add_assign,
    add_backend,
    /// Computes addition for each element of `self` and the corresponding
    /// element of `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping addition.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ,
    /// Computes addition for each element of `self` and the scalar `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping addition.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    Sub,
    sub,
    SubAssign,
    sub_assign,
    sub_backend,
    /// Computes subtraction for each element of `self` and the corresponding
    /// element of `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping subtraction.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ,
    /// Computes subtraction for each element of `self` and the scalar `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping subtraction.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    Mul,
    mul,
    MulAssign,
    mul_assign,
    mul_backend,
    /// Computes multiplication for each element of `self` and the
    /// corresponding element of `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping multiplication.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ,
    /// Computes multiplication for each element of `self` and the scalar
    /// `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping multiplication.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    Div,
    div,
    DivAssign,
    div_assign,
    div_backend,
    /// Computes division for each element of `self` and the corresponding
    /// element of `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping division.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ///
    /// # Panics
    ///
    /// For integers, this panics if any element of `rhs` is zero.
    ,
    /// Computes division for each element of `self` and the scalar `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes wrapping division.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ///
    /// # Panics
    ///
    /// For integers, this panics if any element of `rhs` is zero.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    Rem,
    rem,
    RemAssign,
    rem_assign,
    rem_backend,
    /// Computes the remainder for each element of `self` and the
    /// corresponding element of `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes `wrapping_rem`.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ///
    /// # Panics
    ///
    /// For integers, this panics if any element of `rhs` is zero.
    ,
    /// Computes the remainder for each element of `self` and the scalar
    /// `rhs`.
    ///
    /// For floats, this always returns the precise result. For integers, this
    /// computes `wrapping_rem`.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ///
    /// # Panics
    ///
    /// For integers, this panics if any element of `rhs` is zero.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    bitand_backend,
    /// Computes bitwise AND for each element of `self` and the corresponding
    /// element of `rhs`.
    ///
    /// For floats, this computes bitwise AND, even though the `&` operator is not
    /// implemented for Rust's primitive floats.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ,
    /// Computes bitwise AND for each element of `self` and the scalar `rhs`.
    ///
    /// For floats, this computes bitwise AND, even though the `&` operator is not
    /// implemented for Rust's primitive floats.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    BitOr,
    bitor,
    BitOrAssign,
    bitor_assign,
    bitor_backend,
    /// Computes bitwise OR for each element of `self` and the corresponding
    /// element of `rhs`.
    ///
    /// For floats, this computes bitwise OR, even though the `|` operator is not
    /// implemented for Rust's primitive floats.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ,
    /// Computes bitwise OR for each element of `self` and the scalar `rhs`.
    ///
    /// For floats, this computes bitwise OR, even though the `|` operator is not
    /// implemented for Rust's primitive floats.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
);
impl_binary_operator!(
    T: PrimitiveFloatOrInteger,
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    bitxor_backend,
    /// Computes bitwise XOR for each element of `self` and the corresponding
    /// element of `rhs`.
    ///
    /// For floats, this computes bitwise XOR, even though the `^` operator is not
    /// implemented for Rust's primitive floats.
    ///
    /// This is one of multiple trait implementations that should be separate
    /// for SIMD floats and SIMD integers, but cannot because Rust currently
    /// does not allow two blanket trait implementations based on disjoint
    /// traits.
    ,
    /// Computes bitwise XOR for each element of `self` and the scalar `rhs`.
    ///
    /// For floats, this computes bitwise XOR, even though the `^` operator is not
    /// implemented for Rust's primitive floats.
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

    #[inline]
    fn add_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::add_backend(self, rhs)
        } else {
            SimdIntegerBackend::add_backend(self, rhs)
        }
    }

    #[inline]
    fn sub_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::sub_backend(self, rhs)
        } else {
            SimdIntegerBackend::sub_backend(self, rhs)
        }
    }

    #[inline]
    fn mul_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::mul_backend(self, rhs)
        } else {
            SimdIntegerBackend::mul_backend(self, rhs)
        }
    }

    #[inline]
    #[track_caller]
    fn div_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::div_backend(self, rhs)
        } else {
            SimdIntegerBackend::div_backend(self, rhs)
        }
    }

    #[inline]
    #[track_caller]
    fn rem_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::rem_backend(self, rhs)
        } else {
            SimdIntegerBackend::rem_backend(self, rhs)
        }
    }

    #[inline]
    fn bitand_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::bitand_backend(self, rhs)
        } else {
            SimdIntegerBackend::bitand_backend(self, rhs)
        }
    }

    #[inline]
    fn bitor_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::bitor_backend(self, rhs)
        } else {
            SimdIntegerBackend::bitor_backend(self, rhs)
        }
    }

    #[inline]
    fn bitxor_backend(self, rhs: Self) -> Self {
        if T::IS_FLOAT {
            SimdFloatBackend::bitxor_backend(self, rhs)
        } else {
            SimdIntegerBackend::bitxor_backend(self, rhs)
        }
    }
}
