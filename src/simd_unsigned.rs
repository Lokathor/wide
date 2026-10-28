//! Defines APIs applicable to SIMD vectors of unsigned integers.

use crate::{
    i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4, i32x8, i32x16, i64x2, i64x4, i64x8, u8x16,
    u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8, u32x16, u64x2, u64x4, u64x8,
};

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
