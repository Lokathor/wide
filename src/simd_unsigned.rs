//! Defines APIs applicable to SIMD vectors of unsigned integers.

use crate::{
    PrimitiveInteger, Simd, SimdElement, SupportedSimd, i8x16, i8x32, i8x64, i16x8, i16x16, i16x32,
    i32x4, i32x8, i32x16, i64x2, i64x4, i64x8,
    simd::{SimdAssociatedTypes, SupportSimdOf},
    u8x16, u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8, u32x16, u64x2, u64x4, u64x8,
};

/// A sealed trait for all primitive unsigned integer types.
///
/// TODO(generics): Once appropriate SIMD functionality becomes generic over
/// `T: PrimitiveSigned`, update the documentation of this trait to explain its
/// intended usage.
pub trait PrimitiveUnsigned: PrimitiveInteger<Unsigned = Self> {}

/// Defines the internal implementations of functions applicable to SIMD vectors
/// of unsigned integers.
pub(crate) trait SimdUnsignedBackend {
    fn saturating_add_signed_backend(self, rhs: Self::SignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::SignedT>;
}

/// A marker trait representing `!PrimitiveUnsigned` used for an internal
/// type-system hack.
trait NotPrimitiveUnsigned {}

macro_rules! impl_primitive_unsigned {
    ($T:ident) => {
        impl PrimitiveUnsigned for $T {}
    };
}
impl_primitive_unsigned!(u8);
impl_primitive_unsigned!(u16);
impl_primitive_unsigned!(u32);
impl_primitive_unsigned!(u64);
impl_primitive_unsigned!(u128);
impl_primitive_unsigned!(usize);

impl NotPrimitiveUnsigned for f32 {}
impl NotPrimitiveUnsigned for f64 {}
impl NotPrimitiveUnsigned for i8 {}
impl NotPrimitiveUnsigned for i16 {}
impl NotPrimitiveUnsigned for i32 {}
impl NotPrimitiveUnsigned for i64 {}

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

macro_rules! impl_simd_unsigned {
    (
        T = $T:ident,
        N = $N:literal,
        Simd = $Simd:ident,
        SignedSimd = $SignedSimd:ident,
    ) => {
        /// The following functionality exists for all SIMD vectors of unsigned
        /// integers.
        impl $Simd {
            /// Returns the bit patterns of `self` reinterpreted as signed integers of
            /// the same size.
            #[inline]
            #[must_use]
            pub const fn cast_signed(self) -> $SignedSimd {
                // SAFETY: Both types accept all bit-patterns and only contain
                // initialized memory.
                unsafe { core::mem::transmute::<$Simd, $SignedSimd>(self) }
            }

            /// Wrapping (modular) addition with a signed `rhs`. Computes
            /// `self + rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn add_signed(self, rhs: $SignedSimd) -> Self {
                self + rhs.cast_unsigned()
            }

            /// Wrapping (modular) subtraction with a signed `rhs`. Computes
            /// `self - rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn sub_signed(self, rhs: $SignedSimd) -> Self {
                self - rhs.cast_unsigned()
            }

            /// Saturating addition with a signed `rhs`. Computes `self + rhs`,
            /// saturating at the numeric bounds instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_add_signed(self, rhs: $SignedSimd) -> Self {
                // NEON has a direct intrinsic for this operation
                self.saturating_add_signed_backend(rhs)
            }

            /// Saturating subtraction with a signed `rhs`. Computes
            /// `self - rhs`, saturating at the numeric bounds instead of
            /// overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_sub_signed(self, rhs: $SignedSimd) -> Self {
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
            pub fn overflowing_add_signed(self, rhs: $SignedSimd) -> (Self, Self) {
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
            pub fn overflowing_sub_signed(self, rhs: $SignedSimd) -> (Self, Self) {
                let result = self.sub_signed(rhs);
                let overflow = result.simd_gt(self) ^ rhs.is_negative().cast_unsigned();

                (result, overflow)
            }
        }
    };
}
impl_simd_unsigned!(T = u8, N = 16, Simd = u8x16, SignedSimd = i8x16,);
impl_simd_unsigned!(T = u8, N = 32, Simd = u8x32, SignedSimd = i8x32,);
impl_simd_unsigned!(T = u8, N = 64, Simd = u8x64, SignedSimd = i8x64,);
impl_simd_unsigned!(T = u16, N = 8, Simd = u16x8, SignedSimd = i16x8,);
impl_simd_unsigned!(T = u16, N = 16, Simd = u16x16, SignedSimd = i16x16,);
impl_simd_unsigned!(T = u16, N = 32, Simd = u16x32, SignedSimd = i16x32,);
impl_simd_unsigned!(T = u32, N = 4, Simd = u32x4, SignedSimd = i32x4,);
impl_simd_unsigned!(T = u32, N = 8, Simd = u32x8, SignedSimd = i32x8,);
impl_simd_unsigned!(T = u32, N = 16, Simd = u32x16, SignedSimd = i32x16,);
impl_simd_unsigned!(T = u64, N = 2, Simd = u64x2, SignedSimd = i64x2,);
impl_simd_unsigned!(T = u64, N = 4, Simd = u64x4, SignedSimd = i64x4,);
impl_simd_unsigned!(T = u64, N = 8, Simd = u64x8, SignedSimd = i64x8,);
