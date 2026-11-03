//! Defines APIs applicable to SIMD vectors of unsigned integers.

use crate::{
    PrimitiveUnsigned, Simd, SimdElement, SupportedSimd, primitive_traits::NotPrimitiveUnsigned, simd::{SimdAssociatedTypes, SupportSimdOf},
};

impl<T, const N: usize> Simd<T, N>
where
    T: PrimitiveUnsigned,
    Simd<T, N>: SupportedSimd,
    // TODO(generics): Try to make this bound implicit
    Simd<T::Signed, N>: SupportedSimd,
{
            /// Returns the bit patterns of `self` reinterpreted as signed integers of
            /// the same size.
            #[inline]
            #[must_use]
            pub const fn cast_signed(self) -> Simd<T::Signed, N> {
                // SAFETY: Both types accept all bit-patterns and only contain
                // initialized memory.
                unsafe { core::mem::transmute_copy::<Simd<T, N>, Simd<T::Signed, N>>(&self) }
            }

            /// Wrapping (modular) addition with a signed `rhs`. Computes
            /// `self + rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn add_signed(self, rhs: Simd<T::Signed, N>) -> Self {
                self + rhs.cast_unsigned()
            }

            /// Wrapping (modular) subtraction with a signed `rhs`. Computes
            /// `self - rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn sub_signed(self, rhs: Simd<T::Signed, N>) -> Self {
                self - rhs.cast_unsigned()
            }

            /// Saturating addition with a signed `rhs`. Computes `self + rhs`,
            /// saturating at the numeric bounds instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_add_signed(self, rhs: Simd<T::Signed, N>) -> Self {
                // NEON has a direct intrinsic for this operation
                self.saturating_add_signed_backend(rhs)
            }

            /// Saturating subtraction with a signed `rhs`. Computes
            /// `self - rhs`, saturating at the numeric bounds instead of
            /// overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_sub_signed(self, rhs: Simd<T::Signed, N>) -> Self {
                // Unlike for `saturating_add_signed`, NEON does not have a
                // direct intrinsic for this operation
                let (result, overflow) = self.overflowing_sub_signed(rhs);
                overflow.select(rhs.is_negative().cast_unsigned(), result)
            }

            /// Returns `self + rhs` with a signed `rhs` and whether an overflow
            /// occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The addition (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_add_signed(self, rhs: Simd<T::Signed, N>) -> (Self, Self) {
                let result = self.add_signed(rhs);
                let overflow = result.simd_lt(self) ^ rhs.is_negative().cast_unsigned();

                (result, overflow)
            }

            /// Returns `self - rhs` with a signed `rhs` and whether an overflow
            /// occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The subtraction (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_sub_signed(self, rhs: Simd<T::Signed, N>) -> (Self, Self) {
                let result = self.sub_signed(rhs);
                let overflow = result.simd_gt(self) ^ rhs.is_negative().cast_unsigned();

                (result, overflow)
            }
}

/// Defines the internal implementations of functions applicable to SIMD vectors
/// of unsigned integers.
pub(crate) trait SimdUnsignedBackend {
    fn saturating_add_signed_backend(self, rhs: Self::SignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::SignedT>;
}

/// In order for the generic SIMD unsigned integer API to work,
/// [`SimdUnsignedBackend`] is used as a super-trait of [`SupportedSimd`].
///
/// This is incorrect logically, since [`SimdUnsignedBackend`] is only
/// applicable to SIMD vectors of unsigned integers. However this seems to be
/// the only way to do it on stable Rust.
///
/// This means that non-unsigned-integers do need some implementation of
/// [`SimdUnsignedBackend`]. This block is a dummy implementation for all
/// non-unsigned-integers that fills functions with [`unreachable!`].
impl<T, const N: usize> SimdUnsignedBackend for Simd<T, N>
where
    T: SimdElement + NotPrimitiveUnsigned,
    Simd<T, N>: SupportedSimd,
{
    fn saturating_add_signed_backend(self, _: <Self as SimdAssociatedTypes>::SignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<<Self as SimdAssociatedTypes>::SignedT>,
    {
        unreachable!()
    }
}
