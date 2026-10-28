//! Defines APIs applicable to SIMD vectors of signed integers.

use bytemuck::cast;

use crate::{
    Select, i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4, i32x8, i32x16, i64x2, i64x4, i64x8,
    u8x16, u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8, u32x16, u64x2, u64x4, u64x8,
};

macro_rules! impl_simd_signed {
    (
        T = $T:ident,
        N = $N:literal,
        Simd = $Simd:ident,
        UnsignedSimd = $UnsignedSimd:ident,
    ) => {
        impl Select<$Simd> for $UnsignedSimd {
            #[inline]
            fn select(self, if_true: $Simd, if_false: $Simd) -> $Simd {
                self.cast_signed().select(if_true, if_false)
            }
        }

        impl Select<$UnsignedSimd> for $Simd {
            #[inline]
            fn select(self, if_true: $UnsignedSimd, if_false: $UnsignedSimd) -> $UnsignedSimd {
                self.cast_unsigned().select(if_true, if_false)
            }
        }

        /// The following functionality exists for all SIMD vectors of signed
        /// integers.
        impl $Simd {
            /// Returns the bit patterns of `self` reinterpreted as unsigned integers
            /// of the same size.
            #[inline]
            #[must_use]
            pub const fn cast_unsigned(self) -> $UnsignedSimd {
                // SAFETY: Both types accept all bit-patterns and only contain
                // initialized memory.
                unsafe { core::mem::transmute::<$Simd, $UnsignedSimd>(self) }
            }

            /// Computes the absolute value of each input element, returned as an
            /// unsigned integer in order to avoid wrapping.
            #[inline]
            #[must_use]
            pub fn unsigned_abs(self) -> $UnsignedSimd {
                cast::<$Simd, $UnsignedSimd>(self.abs())
            }

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
            pub fn add_unsigned(self, rhs: $UnsignedSimd) -> Self {
                self + rhs.cast_signed()
            }

            /// Wrapping (modular) subtraction with an unsigned `rhs`. Computes
            /// `self - rhs`, wrapping around at the boundary of the type.
            #[inline]
            #[must_use]
            pub fn sub_unsigned(self, rhs: $UnsignedSimd) -> Self {
                self - rhs.cast_signed()
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
            pub fn overflowing_add_unsigned(self, rhs: $UnsignedSimd) -> (Self, Self) {
                let rhs = rhs.cast_signed();
                let result = self + rhs;
                let overflow = ((!(self ^ rhs) & (self ^ result)) ^ rhs).is_negative();

                (result, overflow)
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
            pub fn overflowing_sub_unsigned(self, rhs: $UnsignedSimd) -> (Self, Self) {
                let rhs = rhs.cast_signed();
                let result = self - rhs;
                let overflow = (((self ^ rhs) & (self ^ result)) ^ rhs).is_negative();

                (result, overflow)
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
