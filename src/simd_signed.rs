//! Defines APIs applicable to SIMD vectors of signed integers.

use crate::{
    PrimitiveSigned, Simd, SimdElement, SupportedSimd, i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4, i32x8, i32x16, i64x2, i64x4, i64x8, primitive_traits::NotPrimitiveSigned, simd::{SimdAssociatedTypes, SupportSimdOf},
};

macro_rules! impl_simd_signed {
    (
        T = $T:ident,
        N = $N:literal,
        Simd = $Simd:ident,
        UnsignedSimd = $UnsignedSimd:ident,
    ) => {
        /// The following functionality exists for all SIMD vectors of signed
        /// integers.
        impl $Simd {
            /// Returns the absolute value of each input element.
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                self.abs_backend()
            }

            /// Returns numbers representing the sign of each element.
            ///
            /// - `0` if the element is zero
            /// - `1` if the element is positive
            /// - `-1` if the element is negative
            #[inline]
            #[must_use]
            pub fn signum(self) -> Self {
                // Flip signs because the result for true in `is_positive/negative` is
                // `-1` (all bits set).
                self.is_negative() - self.is_positive()
            }
        }
    };
}
impl_simd_signed!(T = i8, N = 16, Simd = i8x16, UnsignedSimd = u8x16,);
impl_simd_signed!(T = i8, N = 32, Simd = i8x32, UnsignedSimd = u8x32,);
impl_simd_signed!(T = i8, N = 64, Simd = i8x64, UnsignedSimd = u8x64,);
impl_simd_signed!(T = i16, N = 8, Simd = i16x8, UnsignedSimd = u16x8,);
impl_simd_signed!(T = i16, N = 16, Simd = i16x16, UnsignedSimd = u16x16,);
impl_simd_signed!(T = i16, N = 32, Simd = i16x32, UnsignedSimd = u16x32,);
impl_simd_signed!(T = i32, N = 4, Simd = i32x4, UnsignedSimd = u32x4,);
impl_simd_signed!(T = i32, N = 8, Simd = i32x8, UnsignedSimd = u32x8,);
impl_simd_signed!(T = i32, N = 16, Simd = i32x16, UnsignedSimd = u32x16,);
impl_simd_signed!(T = i64, N = 2, Simd = i64x2, UnsignedSimd = u64x2,);
impl_simd_signed!(T = i64, N = 4, Simd = i64x4, UnsignedSimd = u64x4,);
impl_simd_signed!(T = i64, N = 8, Simd = i64x8, UnsignedSimd = u64x8,);

impl<T, const N: usize> Simd<T, N>
where
    T: PrimitiveSigned,
    Simd<T, N>: SupportedSimd,
    // TODO(generics): Try make this implicit bound
    Simd<T::Unsigned, N>: SupportedSimd,
{
            /// Returns the bit patterns of `self` reinterpreted as unsigned integers
            /// of the same size.
            #[inline]
            #[must_use]
            pub const fn cast_unsigned(self) -> Simd<T::Unsigned, N> {
                // SAFETY: Both types accept all bit-patterns and only contain
                // initialized memory.
                unsafe { core::mem::transmute_copy::<Simd<T, N>, Simd<T::Unsigned, N>>(&self) }
            }

            /// Computes the absolute value of each input element, returned as an
            /// unsigned integer in order to avoid wrapping.
            #[inline]
            #[must_use]
            pub fn unsigned_abs(self) -> Simd<T::Unsigned, N> {
                // Use `abs_backend` and not `abs` because `abs` is not
                // available in contexts generic over `T`
                self.abs_backend().cast_unsigned()
            }

            /// Returns a [mask] that is true for each positive element, and false if
            /// it is zero or negative.
            ///
            /// [mask]: crate#masks
            #[inline]
            #[must_use]
            pub fn is_positive(self) -> Self {
                self.is_positive_backend()
            }

            /// Returns a [mask] that is true for each negative element, and false if
            /// it is zero or positive.
            ///
            /// [mask]: crate#masks
            #[inline]
            #[must_use]
            pub fn is_negative(self) -> Self {
                self.is_negative_backend()
            }

            /// Wrapping (modular) addition with an unsigned `rhs`. Computes
            /// `self + rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn add_unsigned(self, rhs: Simd<T::Unsigned, N>) -> Self {
                self + rhs.cast_signed()
            }

            /// Wrapping (modular) subtraction with an unsigned `rhs`. Computes
            /// `self - rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn sub_unsigned(self, rhs: Simd<T::Unsigned, N>) -> Self {
                self - rhs.cast_signed()
            }

            /// Saturating integer negation. Computes `-self`, returning `MAX`
            /// if `self == MIN` instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_neg(self) -> Self {
                self.saturating_neg_backend()
            }

            /// Saturating absolute value. Computes `self.abs()`, returning
            /// `MAX` if `self == MIN` instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_abs(self) -> Self {
                self.saturating_abs_backend()
            }

            /// Saturating addition with an unsigned `rhs`. Computes
            /// `self + rhs`, saturating at the numeric bounds instead of
            /// overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_add_unsigned(self, rhs: Simd<T::Unsigned, N>) -> Self {
                // NEON has a direct intrinsic for this operation
                self.saturating_add_unsigned_backend(rhs)
            }

            /// Saturating subtraction with an unsigned `rhs`. Computes
            /// `self - rhs`, saturating at the numeric bounds instead of
            /// overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_sub_unsigned(self, rhs: Simd<T::Unsigned, N>) -> Self {
                // Unlike for `saturating_add_unsigned`, NEON does not have a
                // direct intrinsic for this operation
                let (result, overflow) = self.overflowing_sub_unsigned(rhs);
                overflow.select(const { Self::splat(T::MIN) }, result)
            }

            /// Computes the absolute value of self.
            ///
            /// Returns a tuple with:
            ///
            /// - The absolute value (returns the wrapped value if an overflow
            ///   occured)
            /// - A mask indicating whether an overflow occured (only occurs
            ///   when `self == MIN`)
            #[inline]
            #[must_use]
            pub fn overflowing_abs(self) -> (Self, Self) {
                // Use `abs_backend` and not `abs` because `abs` is not
                // available in contexts generic over `T`
                (self.abs_backend(), self.simd_eq(const { Self::splat(T::MIN) }))
            }

            /// Returns `self + rhs` with an unsigned `rhs` and whether an overflow
            /// occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The addition (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_add_unsigned(self, rhs: Simd<T::Unsigned, N>) -> (Self, Self) {
                pick! {
                    // On NEON, `saturating_add_unsigned` is implement using a
                    // direct intrinsic. It should be faster to call the
                    // intrinsics for wrapping and saturating addition, then
                    // compare the results, rather than to use the fallback
                    // overflow check.
                    if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                        let result = self.add_unsigned(rhs);
                        let saturating_result = self.saturating_add_unsigned(rhs);
                        (result, result.simd_ne(saturating_result))
                    } else {
                        let rhs = rhs.cast_signed();
                        let result = self + rhs;
                        let overflow = ((!(self ^ rhs) & (self ^ result)) ^ rhs).is_negative();

                        (result, overflow)
                    }
                }
            }

            /// Returns `self - rhs` with an unsigned `rhs` and whether an overflow
            /// occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The subtraction (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_sub_unsigned(self, rhs: Simd<T::Unsigned, N>) -> (Self, Self) {
                let rhs = rhs.cast_signed();
                let result = self - rhs;
                let overflow = (((self ^ rhs) & (self ^ result)) ^ rhs).is_negative();

                (result, overflow)
            }
}

/// Defines the internal implementations of functions applicable to SIMD vectors
/// of signed integers.
pub(crate) trait SimdSignedBackend {
    fn abs_backend(self) -> Self;

    #[expect(clippy::wrong_self_convention)]
    fn is_positive_backend(self) -> Self;

    #[expect(clippy::wrong_self_convention)]
    fn is_negative_backend(self) -> Self;

    fn saturating_neg_backend(self) -> Self;

    fn saturating_abs_backend(self) -> Self;

    fn saturating_add_unsigned_backend(self, rhs: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;
}

/// In order for the generic SIMD signed integer API to work,
/// [`SimdSignedBackend`] is used as a super-trait of [`SupportedSimd`].
///
/// This is incorrect logically, since [`SimdSignedBackend`] is only applicable
/// to SIMD vectors of signed integers. However this seems to be the only way to
/// do it on stable Rust.
///
/// This means that non-signed-integers do need some implementation of
/// [`SimdSignedBackend`]. This block is a dummy implementation for all
/// non-signed-integers that fills functions with [`unreachable!`].
impl<T, const N: usize> SimdSignedBackend for Simd<T, N>
where
    T: SimdElement + NotPrimitiveSigned,
    Simd<T, N>: SupportedSimd,
{
    fn abs_backend(self) -> Self {
        unreachable!()
    }
    fn is_negative_backend(self) -> Self {
        unreachable!()
    }
    fn is_positive_backend(self) -> Self {
        unreachable!()
    }
    fn saturating_abs_backend(self) -> Self {
        unreachable!()
    }
    fn saturating_add_unsigned_backend(self, _: <Self as SimdAssociatedTypes>::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<<Self as SimdAssociatedTypes>::UnsignedT>,
    {
        unreachable!()
    }
    fn saturating_neg_backend(self) -> Self {
        unreachable!()
    }
}
