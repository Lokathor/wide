//! Defines APIs applicable to SIMD vectors of integers (both signed and
//! unsigned).

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use bytemuck::{Zeroable, cast};

use crate::{
    i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4, i32x8, i32x16, i64x2, i64x4, i64x8, u8x16,
    u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8, u32x16, u64x2, u64x4, u64x8,
};

macro_rules! impl_simd_integer {
    (
        T = $T:ident,
        N = $N:literal,
        Simd = $Simd:ident,
        autofill = $autofill:ident,
        UnsignedSimd = $UnsignedSimd:ident,
        SignedSimd = $SignedSimd:ident,
        WiderSimd = { $($WiderSimd:ident)? },
        T_BITS = $T_BITS:literal,
        T_BITS_MUL_2 = $T_BITS_MUL_2:literal,
        INDICES = [$($INDEX:literal),* $(,)?],
    ) => {
        impl Eq for $Simd {}

        impl_unary_operator!(
            $Simd,
            Not,
            not,
            not_backend,
            /// Computes bitwise NOT for each element of `self`.
        );

        impl_binary_operator!(
            $T,
            $Simd,
            Add,
            add,
            AddAssign,
            add_assign,
            add_backend,
            /// Computes wrapping addition for each element of `self` and the
            /// corresponding element of `rhs`.
            ,
            /// Computes wrapping addition for each element of `self` and the
            /// scalar `rhs`.
            ,
            /// Computes wrapping addition for the scalar `self` and each
            /// element of `rhs`, returning a SIMD vector.
        );
        impl_binary_operator!(
            $T,
            $Simd,
            Sub,
            sub,
            SubAssign,
            sub_assign,
            sub_backend,
            /// Computes wrapping subtraction for each element of `self` and the
            /// corresponding element of `rhs`.
            ,
            /// Computes wrapping subtraction for each element of `self` and the
            /// scalar `rhs`.
            ,
            /// Computes wrapping subtraction for the scalar `self` and each
            /// element of `rhs`, returning a SIMD vector.
        );
        impl_binary_operator!(
            $T,
            $Simd,
            Mul,
            mul,
            MulAssign,
            mul_assign,
            mul_backend,
            /// Computes wrapping multiplication for each element of `self` and
            /// the corresponding element of `rhs`.
            ,
            /// Computes wrapping multiplication for each element of `self` and
            /// the scalar `rhs`.
            ,
            /// Computes wrapping multiplication for the scalar `self` and each
            /// element of `rhs`, returning a SIMD vector.
        );
        impl_binary_operator!(
            $T,
            $Simd,
            Div,
            div,
            DivAssign,
            div_assign,
            div_backend,
            /// Computes wrapping division for each element of `self` and the
            /// corresponding element of `rhs`.
            ///
            /// Note that because division has no hardware support, this operation
            /// is very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if any element of `rhs` is zero.
            ,
            /// Computes wrapping division for each element of `self` and the
            /// scalar `rhs`.
            ///
            /// Note that because division has no hardware support, this operation
            /// is very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if `rhs` is zero.
            ,
            /// Computes wrapping division for the scalar `self` and each
            /// element of `rhs`, returning a SIMD vector.
            ///
            /// Note that because division has no hardware support, this operation
            /// is very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if any element of `rhs` is zero.
        );
        impl_binary_operator!(
            $T,
            $Simd,
            Rem,
            rem,
            RemAssign,
            rem_assign,
            rem_backend,
            /// Computes the wrapping remainder of each element of `self`
            /// divided by the corresponding element of `rhs`.
            ///
            /// Note that because division has no hardware support, this operation
            /// is very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if any element of `rhs` is zero.
            ,
            /// Computes the wrapping remainder of each element of `self`
            /// divided by the scalar `rhs`.
            ///
            /// Note that because division has no hardware support, this operation
            /// is very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if `rhs` is zero.
            ,
            /// Computes the wrapping remainder of the scalar `self` divided by
            /// each element of `rhs`, returning a SIMD vector.
            ///
            /// Note that because division has no hardware support, this operation
            /// is very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if any element of `rhs` is zero.
        );
        impl_shift_operator!(
            $T,
            $Simd,
            $UnsignedSimd,
            $SignedSimd,
            Shl,
            shl,
            ShlAssign,
            shl_assign,
            shl_backend,
            shl_scalar_backend,
            /// Shifts left each element of `self` by the corresponding element
            /// of `rhs`.
            ///
            /// This operator behaves like [`wrapping_shl`].
            ///
            /// Note that on most targets, this operator is slower than
            /// [`unbounded_shl`], so consider using that instead.
            ///
            #[doc = concat!("[`wrapping_shl`]: ", stringify!($T), "::wrapping_shl")]
            #[doc = concat!("[`unbounded_shl`]: ", stringify!($Simd), "::unbounded_shl")]
            ,
            /// Shifts left each element of `self` by the scalar `rhs`.
            ///
            /// This operator behaves like [`wrapping_shl`].
            ///
            /// On most targets, shifting by a scalar is faster than shifting by
            /// SIMD vector. Note that on most targets, this operator is slower than
            /// [`unbounded_shl_scalar`], so consider using that instead.
            ///
            #[doc = concat!("[`wrapping_shl`]: ", stringify!($T), "::wrapping_shl")]
            #[doc = concat!("[`unbounded_shl_scalar`]: ", stringify!($Simd), "::unbounded_shl_scalar")]
            ,
            /// Shifts left the scalar `self` by each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This operator behaves like [`wrapping_shl`].
            ///
            /// Note that on most targets, this operator is slower than
            /// [`unbounded_shl`], so consider using that instead.
            ///
            #[doc = concat!("[`wrapping_shl`]: ", stringify!($T), "::wrapping_shl")]
            #[doc = concat!("[`unbounded_shl`]: ", stringify!($Simd), "::unbounded_shl")]
        );
        impl_shift_operator!(
            $T,
            $Simd,
            $UnsignedSimd,
            $SignedSimd,
            Shr,
            shr,
            ShrAssign,
            shr_assign,
            shr_backend,
            shr_scalar_backend,
            /// Shifts right each element of `self` by the corresponding element of
            /// `rhs`.
            ///
            /// This operator behaves like [`wrapping_shr`].
            ///
            /// Note that on most targets, this operator is slower than
            /// [`unbounded_shr`], so consider using that instead.
            ///
            #[doc = concat!("[`wrapping_shr`]: ", stringify!($T), "::wrapping_shr")]
            #[doc = concat!("[`unbounded_shr`]: ", stringify!($Simd), "::unbounded_shr")]
            ,
            /// Shifts right each element of `self` by the scalar `rhs`.
            ///
            /// This operator behaves like [`wrapping_shr`].
            ///
            /// On most targets, shifting by a scalar is faster than shifting by
            /// a SIMD vector. Note that on most targets, this operator is
            /// slower than [`unbounded_shr`], so consider using that instead.
            ///
            #[doc = concat!("[`wrapping_shr`]: ", stringify!($T), "::wrapping_shr")]
            #[doc = concat!("[`unbounded_shr`]: ", stringify!($Simd), "::unbounded_shr")],
            /// Shifts right the scalar `self` by each element of `rhs`,
            /// returning a SIMD vector.
            ///
            /// This operator behaves like [`wrapping_shr`].
            ///
            /// Note that on most targets, this operator is slower than
            /// [`unbounded_shr`], so consider using that instead.
            ///
            #[doc = concat!("[`wrapping_shr`]: ", stringify!($T), "::wrapping_shr")]
            #[doc = concat!("[`unbounded_shr`]: ", stringify!($Simd), "::unbounded_shr")]
        );
        impl_binary_operator!(
            $T,
            $Simd,
            BitAnd,
            bitand,
            BitAndAssign,
            bitand_assign,
            bitand_backend,
            /// Computes bitwise AND for each element of `self` and the
            /// corresponding element of `rhs`.
            ,
            /// Computes bitwise AND for each element of `self` and the
            /// scalar `rhs`.
            ,
            /// Computes bitwise AND for the scalar `self` and each element of
            /// `rhs`, returning a SIMD vector.
        );
        impl_binary_operator!(
            $T,
            $Simd,
            BitOr,
            bitor,
            BitOrAssign,
            bitor_assign,
            bitor_backend,
            /// Computes bitwise OR for each element of `self` and the
            /// corresponding element of `rhs`.
            ,
            /// Computes bitwise OR for each element of `self` and the
            /// scalar `rhs`.
            ,
            /// Computes bitwise OR for the scalar `self` and each element of
            /// `rhs`, returning a SIMD vector.
        );
        impl_binary_operator!(
            $T,
            $Simd,
            BitXor,
            bitxor,
            BitXorAssign,
            bitxor_assign,
            bitxor_backend,
            /// Computes bitwise XOR for each element of `self` and the
            /// corresponding element of `rhs`.
            ,
            /// Computes bitwise XOR for each element of `self` and the
            /// scalar `rhs`.
            ,
            /// Computes bitwise XOR for the scalar `self` and each element of
            /// `rhs`, returning a SIMD vector.
        );

        impl<Rhs> core::iter::Sum<Rhs> for $Simd
        where
            $Simd: AddAssign<Rhs>,
        {
            #[inline]
            fn sum<I: Iterator<Item = Rhs>>(iter: I) -> Self {
                let mut total = Self::zeroed();
                for val in iter {
                total += val;
                }
                total
            }
        }

        impl<Rhs> core::iter::Product<Rhs> for $Simd
        where
            $Simd: MulAssign<Rhs>,
        {
            #[inline]
            fn product<I: Iterator<Item = Rhs>>(iter: I) -> Self {
                let mut total = Self::from(1);
                for val in iter {
                total *= val;
                }
                total
            }
        }

        macro_rules! impl_formatting_trait {
            ($Trait:path) => {
                impl $Trait for $Simd {
                    #[allow(clippy::missing_inline_in_public_items)]
                    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                        write!(f, "(")?;
                        for (i, x) in self.to_array().iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            <$T as $Trait>::fmt(x, f)?;
                        }
                        write!(f, ")")
                    }
                }
            }
        }
        impl_formatting_trait!(core::fmt::Binary);
        impl_formatting_trait!(core::fmt::LowerHex);
        impl_formatting_trait!(core::fmt::Octal);
        impl_formatting_trait!(core::fmt::UpperHex);

        /// The following functionality exists for all SIMD vectors of integers.
        impl $Simd {
            /// A SIMD vector with all elements set to `1`.
            pub const ONE: Self = Self::splat(1);

            /// A SIMD vector with all elements set to `0`.
            pub const ZERO: Self = Self::splat(0);

            #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::MAX`].")]
            pub const MAX: Self = Self::splat($T::MAX);

            #[doc = concat!("A SIMD vector with all elements set to [`", stringify!($T) ,"::MIN`].")]
            pub const MIN: Self = Self::splat($T::MIN);

            /// The number of elements in this SIMD vector.
            pub const LANES: u16 = $N;

            /// The size of this SIMD vector in bits.
            pub const BITS: u16 = (size_of::<Self>() * 8) as u16;

            /// Returns the maximum between each element of `self` and the
            /// corresponding element of `other`.
            #[inline]
            #[must_use]
            pub fn max(self, other: Self) -> Self {
                self.max_backend(other)
            }

            /// Returns the minimum between each element of `self` and the
            /// corresponding element of `other`.
            #[inline]
            #[must_use]
            pub fn min(self, other: Self) -> Self {
                self.min_backend(other)
            }

            /// Clamps each element of `self` between the corresponding elements
            /// of `min` and `max`.
            ///
            /// If `min > max`, the result is unspecified. Consider manually
            /// checking for that case.
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                self.max(min).min(max)
            }

            /// Reducing addition. Returns the sum of the vector's elements.
            ///
            /// Equivalent to `self[0] + self[1] + ...`.
            #[inline]
            #[must_use]
            pub fn reduce_add(self) -> $T {
                self.reduce_add_backend()
            }

            /// Reducing multiplication. Returns the product of the vector's
            /// elements.
            ///
            /// Equivalent to `self[0] * self[1] * ...`.
            #[inline]
            #[must_use]
            pub fn reduce_mul(self) -> $T {
                self.reduce_mul_backend()
            }

            /// Reducing maximum. Returns the maximum of the vector's elements.
            ///
            /// Equivalent to `self[0].max(self[1].max(...))`.
            #[inline]
            #[must_use]
            pub fn reduce_max(self) -> $T {
                self.reduce_max_backend()
            }

            /// Reducing minimum. Returns the minimum of the vector's elements.
            ///
            /// Equivalent to `self[0].min(self[1].min(...))`.
            #[inline]
            #[must_use]
            pub fn reduce_min(self) -> $T {
                self.reduce_min_backend()
            }

            /// Shifts left each element of `self` by the corresponding element
            /// of `rhs`, without bounding `rhs`.
            ///
            #[doc = concat!("If `rhs` is larger than or equal to the number of bits in [`", stringify!($T), "`],")]
            /// the entire value is shifted out, and `0` is returned.
            ///
            /// This is different from the standard operator, which behaves like
            /// [`wrapping_shl`]. For most targets, `unbounded_shl` is faster than the
            /// standard operator.
            ///
            /// If you intend to shift all elements by the same value, consider using
            /// [`unbounded_shl_scalar`] which is faster.
            ///
            #[doc = concat!("[`wrapping_shl`]: ", stringify!($T), "::wrapping_shl")]
            /// [`unbounded_shl_scalar`]: Self::unbounded_shl_scalar
            #[inline]
            #[must_use]
            pub fn unbounded_shl(self, rhs: $UnsignedSimd) -> Self {
                self.unbounded_shl_backend(rhs)
            }

            /// Shifts left each element of `self` by the uniform scalar `rhs`,
            /// without bounding `rhs`.
            ///
            #[doc = concat!("If `rhs` is larger than or equal to the number of bits in [`", stringify!($T), "`],")]
            /// the entire value is shifted out, and `0` is returned.
            ///
            /// This is different from the standard operator, which behaves like
            /// [`wrapping_shl`]. For most targets, `unbounded_shl_scalar` is
            /// faster than the standard operator.
            ///
            /// This function is faster than `self.unbounded_shl(splat(rhs))`
            /// because it has special hardware support.
            ///
            #[doc = concat!("[`wrapping_shl`]: ", stringify!($T), "::wrapping_shl")]
            #[inline]
            #[must_use]
            pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
                self.unbounded_shl_scalar_backend(rhs)
            }

            /// Shifts right each element of `self` by the corresponding element
            /// of `rhs`, without bounding `rhs`.
            ///
            #[doc = concat!("If `rhs` is larger than or equal to the number of bits in [`", stringify!($T), "`],")]
            /// the entire value is shifted out. For unsigned integers, this
            /// yields `0`. For signed integers, this yields `0` for a positive
            /// number and `-1` for a negative number.
            ///
            /// This is different from the standard operator, which behaves like
            /// [`wrapping_shr`]. For most targets, `unbounded_shr` is faster than the
            /// standard operator.
            ///
            /// If you intend to shift all elements by the same value, consider using
            /// [`unbounded_shr_scalar`] which is faster.
            ///
            #[doc = concat!("[`wrapping_shr`]: ", stringify!($T), "::wrapping_shr")]
            /// [`unbounded_shr_scalar`]: Self::unbounded_shr_scalar
            #[inline]
            #[must_use]
            pub fn unbounded_shr(self, rhs: $UnsignedSimd) -> Self {
                self.unbounded_shr_backend(rhs)
            }

            /// Shifts right each element of `self` by the uniform scalar `rhs`,
            /// without bounding `rhs`.
            ///
            #[doc = concat!("If `rhs` is larger than or equal to the number of bits in [`", stringify!($T), "`],")]
            /// the entire value is shifted out. For unsigned integers, this
            /// yields `0`. For signed integers, this yields `0` for a positive
            /// number and `-1` for a negative number.
            ///
            /// This is different from the standard operator, which behaves like
            /// [`wrapping_shr`]. For most targets, `unbounded_shr_scalar` is faster
            /// than the standard operator.
            ///
            /// This function is faster than `self.unbounded_shr(splat(rhs))` because
            /// it has special hardware support.
            ///
            #[doc = concat!("[`wrapping_shr`]: ", stringify!($T), "::wrapping_shr")]
            #[inline]
            #[must_use]
            pub fn unbounded_shr_scalar(self, rhs: u32) -> Self {
                self.unbounded_shr_scalar_backend(rhs)
            }

            /// Saturating integer addition. Computes `self + rhs`, saturating at the
            /// numeric bounds instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_add(self, rhs: Self) -> Self {
                self.saturating_add_backend(rhs)
            }

            /// Saturating integer subtraction. Computes `self - rhs`, saturating at
            /// the numeric bounds instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_sub(self, rhs: Self) -> Self {
                self.saturating_sub_backend(rhs)
            }

            /// Saturating integer multiplication. Computes `self * rhs`, saturating
            /// at the numeric bounds instead of overflowing.
            #[inline]
            #[must_use]
            pub fn saturating_mul(self, rhs: Self) -> Self {
                self.saturating_mul_backend(rhs)
            }

            /// Saturating integer division. Computes `self / rhs`, saturating at the
            /// numeric bounds instead of overflowing.
            ///
            /// Note that for unsigned integers, overflow never occurs and the
            /// operation is identical to regular division.
            ///
            /// Note that because division has no hardware support, this operation is
            /// very slow and should be avoided if possible.
            ///
            /// # Panics
            ///
            /// Panics if any element of `rhs` is zero.
            #[inline]
            #[must_use]
            pub fn saturating_div(self, rhs: Self) -> Self {
                let self_array = self.to_array();
                let rhs_array = rhs.to_array();

                Self::from_array([$(self_array[$INDEX].saturating_div(rhs_array[$INDEX])),*])
            }

            /// Returns `self + rhs` and whether an overflow occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The addition (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_add(self, rhs: Self) -> (Self, Self) {
                self.overflowing_add_backend(rhs)
            }

            /// Returns `self - rhs` and whether an overflow occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The subtraction (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_sub(self, rhs: Self) -> (Self, Self) {
                self.overflowing_sub_backend(rhs)
            }

            /// Returns `self * rhs` and whether an overflow occured.
            ///
            /// Returns a tuple with:
            ///
            /// - The multiplication (returns the wrapped value if an overflow occured)
            /// - A mask indicating whether an overflow occured
            #[inline]
            #[must_use]
            pub fn overflowing_mul(self, rhs: Self) -> (Self, Self) {
                self.overflowing_mul_backend(rhs)
            }

            /// Returns `self / rhs` and whether an overflow occured.
            ///
            /// Note that for unsigned integers, overflow never occurs thus the
            /// second value is always `false`.
            ///
            /// Returns a tuple with:
            ///
            /// - The division (returns `self` if an overflow occured)
            /// - A mask indicating whether an overflow occured (always false)
            ///
            /// Note that because division has no hardware support, this operation is
            /// very slow and should be avoided if possible.
            #[inline]
            #[must_use]
            pub fn overflowing_div(self, rhs: Self) -> (Self, Self) {
                self.overflowing_div_backend(rhs)
            }

            /// Returns `self % rhs` and whether an overflow occured.
            ///
            /// Note that for unsigned integers, overflow never occurs thus the
            /// second value is always `false`.
            ///
            /// Returns a tuple with:
            ///
            /// - The remainder (returns zero if an overflow occured)
            /// - A mask indicating whether an overflow occured (always false)
            ///
            /// Note that because division has no hardware support, this operation is
            /// very slow and should be avoided if possible.
            #[inline]
            #[must_use]
            pub fn overflowing_rem(self, rhs: Self) -> (Self, Self) {
                self.overflowing_rem_backend(rhs)
            }

            $(
                /// Widening multiplication. Computes `self * rhs`, widening to a SIMD
                /// vector of a larger integer type.
                ///
                /// The returned value is always exact and can never overflow.
                ///
                /// This function is different from [`mul_keep_low_high`], which returns
                /// two seperate SIMD vectors for low and high parts, instead of a
                /// single SIMD vector of a larger integer type. Also note that while
                /// [`mul_keep_low_high`] exists for all types, `widening_mul` does not
                /// exist for types with no wider variant (e.g., for `i32x16` because
                /// there is no `i64x16`).
                ///
                /// [`mul_keep_low_high`]: Self::mul_keep_low_high
                #[inline]
                #[must_use]
                pub fn widening_mul(self, rhs: Self) -> $WiderSimd {
                    self.widening_mul_backend(rhs)
                }
            )?

            #[doc = concat!(
                "Computes `self * rhs`, producing intermediate ",
                $T_BITS_MUL_2,
                "-bit integers, then returns their low ",
                $T_BITS,
                "-bit parts and high ",
                $T_BITS,
                "-bit parts in two seperate SIMD vectors."
            )]
            ///
            /// This function is different from `widening_mul`, which returns a single
            /// SIMD vector of a larger integer type, instead of two seperate SIMD
            /// vectors for low and high parts. Also note that while
            /// `mul_keep_low_high` exists for all types, `widening_mul` does not
            /// exist for types with no wider variant (e.g., for `i32x16` because
            /// there is no `i64x16`).
            #[inline]
            #[must_use]
            pub fn mul_keep_low_high(self, rhs: Self) -> ($UnsignedSimd, Self) {
                self.mul_keep_low_high_backend(rhs)
            }

            #[doc = concat!(
                "Computes `self * rhs`, producing intermediate ",
                $T_BITS_MUL_2,
                "-bit integers, then returns their high ",
                $T_BITS,
                "-bit parts."
            )]
            #[inline]
            #[must_use]
            pub fn mul_keep_high(self, rhs: Self) -> Self {
                self.mul_keep_high_backend(rhs)
            }

            /// Creates a SIMD vector from a native-endian byte array.
            ///
            /// The bytes are interpreted as the raw memory representation of the
            /// vector. This is equivalent to the `from_ne_bytes` functions on
            /// primitive integer types, but operates on the entire vector at once.
            ///
            /// This is the inverse of [`to_ne_bytes`].
            ///
            /// [`to_ne_bytes`]: Self::to_ne_bytes
            #[inline]
            #[must_use]
            pub const fn from_ne_bytes(bytes: [u8; size_of::<$Simd>()]) -> Self {
                // SAFETY: `$Simd` accepts all bit-patterns and only contains
                // initialized memory, and the byte array has the same size.
                unsafe { core::mem::transmute::<[u8; size_of::<$Simd>()], $Simd>(bytes) }
            }

            /// Returns the raw memory representation of `self` as a native-endian
            /// byte array.
            ///
            /// This is the inverse of [`from_ne_bytes`].
            ///
            /// [`from_ne_bytes`]: Self::from_ne_bytes
            #[inline]
            #[must_use]
            pub const fn to_ne_bytes(self) -> [u8; size_of::<$Simd>()] {
                // SAFETY: `$Simd` accepts all bit-patterns and only contains
                // initialized memory, and the byte array has the same size.
                unsafe { core::mem::transmute::<$Simd, [u8; size_of::<$Simd>()]>(self) }
            }

            /// Bitwise selection.
            ///
            /// For each bit of `self`:
            ///
            /// - If the bit is one, return the corresponding bit of `if_one`
            /// - If the bit is zero, return the corresponding bit of `if_zero`
            ///
            /// If you know `self` is a [mask], meaning each element is either
            /// all zeros or all ones, consider using [`select`] which is
            /// faster.
            ///
            /// [mask]: crate#masks
            /// [`select`]: Self::select
            #[inline]
            #[must_use]
            pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
                self.bitselect_backend(if_one, if_zero)
            }

            #[inline]
            #[track_caller]
            fn div_backend(self, rhs: Self) -> Self {
                let self_array = self.to_array();
                let rhs_array = rhs.to_array();

                Self::from_array([$(self_array[$INDEX].wrapping_div(rhs_array[$INDEX])),*])
            }

            #[inline]
            #[track_caller]
            fn rem_backend(self, rhs: Self) -> Self {
                let self_array = self.to_array();
                let rhs_array = rhs.to_array();

                Self::from_array([$(self_array[$INDEX].wrapping_rem(rhs_array[$INDEX])),*])
            }

            // Invoke a macro that automatically fills some of this API's
            // backend functions. There is a separate macro for signed integers
            // and unsigned integers.
            //
            // This is used for functions that do not have the same
            // implementation for all SIMD integer types, but do have the same
            // implementation for all SIMD signed integer types, or for all SIMD
            // unsigned integer types.
            $autofill!(
                T = $T,
                Simd = $Simd,
                UnsignedSimd = $UnsignedSimd,
            );
        }
    };
}

/// Automatically fills SIMD integer backend functions that have the same
/// implementation for all SIMD signed integer types.
macro_rules! simd_integer_backend_signed_autofill {
    (
        T = $T:ident,
        Simd = $Simd:ident,
        UnsignedSimd = $UnsignedSimd:ident,
    ) => {
        #[inline]
        fn not_backend(self) -> Self {
            cast::<$UnsignedSimd, $Simd>(!cast::<$Simd, $UnsignedSimd>(self))
        }

        #[inline]
        fn add_backend(self, rhs: Self) -> Self {
            // Wrapping addition is the same for signed and unsigned integers.
            cast::<$UnsignedSimd, $Simd>(
                cast::<$Simd, $UnsignedSimd>(self) + cast::<$Simd, $UnsignedSimd>(rhs),
            )
        }

        #[inline]
        fn sub_backend(self, rhs: Self) -> Self {
            // Wrapping subtraction is the same for signed and unsigned integers.
            cast::<$UnsignedSimd, $Simd>(
                cast::<$Simd, $UnsignedSimd>(self) - cast::<$Simd, $UnsignedSimd>(rhs),
            )
        }

        #[inline]
        fn mul_backend(self, rhs: Self) -> Self {
            // Wrapping multiplication is the same for signed and unsigned integers.
            cast::<$UnsignedSimd, $Simd>(
                cast::<$Simd, $UnsignedSimd>(self) * cast::<$Simd, $UnsignedSimd>(rhs),
            )
        }

        #[inline]
        fn bitand_backend(self, rhs: Self) -> Self {
            cast::<$UnsignedSimd, $Simd>(
                cast::<$Simd, $UnsignedSimd>(self) & cast::<$Simd, $UnsignedSimd>(rhs),
            )
        }

        #[inline]
        fn bitor_backend(self, rhs: Self) -> Self {
            cast::<$UnsignedSimd, $Simd>(
                cast::<$Simd, $UnsignedSimd>(self) | cast::<$Simd, $UnsignedSimd>(rhs),
            )
        }

        #[inline]
        fn bitxor_backend(self, rhs: Self) -> Self {
            cast::<$UnsignedSimd, $Simd>(
                cast::<$Simd, $UnsignedSimd>(self) ^ cast::<$Simd, $UnsignedSimd>(rhs),
            )
        }

        #[inline]
        fn shl_backend(self, rhs: $UnsignedSimd) -> Self {
            cast(cast::<$Simd, $UnsignedSimd>(self) << rhs)
        }

        #[inline]
        fn shl_scalar_backend(self, rhs: u32) -> Self {
            cast(cast::<$Simd, $UnsignedSimd>(self) << rhs)
        }

        #[inline]
        fn reduce_add_backend(self) -> $T {
            // Wrapping addition is the same for signed and unsigned integers.
            cast::<$Simd, $UnsignedSimd>(self)
                .reduce_add()
                .cast_signed()
        }

        #[inline]
        fn reduce_mul_backend(self) -> $T {
            // Wrapping multiplication is the same for signed and unsigned integers.
            cast::<$Simd, $UnsignedSimd>(self)
                .reduce_mul()
                .cast_signed()
        }

        #[inline]
        fn unbounded_shl_backend(self, rhs: $UnsignedSimd) -> Self {
            // Shift left is the same for unsigned and signed integers.
            cast(cast::<$Simd, $UnsignedSimd>(self).unbounded_shl(rhs))
        }

        #[inline]
        fn unbounded_shl_scalar_backend(self, rhs: u32) -> Self {
            // Shift left is the same for unsigned and signed integers.
            cast(cast::<$Simd, $UnsignedSimd>(self).unbounded_shl_scalar(rhs))
        }

        #[inline]
        fn saturating_mul_backend(self, rhs: Self) -> Self {
            let (result, overflow) = self.overflowing_mul(rhs);
            let limit = Self::MAX ^ (self ^ rhs).is_negative();
            overflow.select(limit, result)
        }

        #[inline]
        fn overflowing_add_backend(self, rhs: Self) -> (Self, Self) {
            let result = self + rhs;
            let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();

            (result, overflow)
        }

        #[inline]
        fn overflowing_sub_backend(self, rhs: Self) -> (Self, Self) {
            let result = self - rhs;
            let overflow = ((self ^ rhs) & (self ^ result)).is_negative();

            (result, overflow)
        }

        #[inline]
        fn overflowing_div_backend(self, rhs: Self) -> (Self, Self) {
            // The second field is equivalent to
            // `self.simd_eq(Self::MIN) & rhs.simd_eq(-1)` but may be cheaper.
            (self / rhs, ((self ^ Self::MAX) & rhs).simd_eq(!Self::ZERO))
        }

        #[inline]
        fn overflowing_rem_backend(self, rhs: Self) -> (Self, Self) {
            // The second field is equivalent to
            // `self.simd_eq(Self::MIN) & rhs.simd_eq(-1)` but may be cheaper.
            (self % rhs, ((self ^ Self::MAX) & rhs).simd_eq(!Self::ZERO))
        }

        #[inline]
        fn bitselect_backend(self, if_one: Self, if_zero: Self) -> Self {
            self.cast_unsigned()
                .bitselect(if_one.cast_unsigned(), if_zero.cast_unsigned())
                .cast_signed()
        }
    };
}

/// Automatically fills SIMD integer backend functions that have the same
/// implementation for all SIMD unsigned integer types.
macro_rules! simd_integer_backend_unsigned_autofill {
    (
        T = $T:ident,
        Simd = $Simd:ident,
        UnsignedSimd = $UnsignedSimd:ident,
    ) => {
        #[inline]
        fn saturating_mul_backend(self, rhs: Self) -> Self {
            let (low, high) = self.mul_keep_low_high(rhs);
            low | high.simd_ne(Self::ZERO)
        }

        #[inline]
        fn overflowing_add_backend(self, rhs: Self) -> (Self, Self) {
            let result = self + rhs;
            let overflow = result.simd_lt(self);

            (result, overflow)
        }

        #[inline]
        fn overflowing_sub_backend(self, rhs: Self) -> (Self, Self) {
            let result = self - rhs;
            let overflow = result.simd_gt(self);

            (result, overflow)
        }

        #[inline]
        fn overflowing_div_backend(self, rhs: Self) -> (Self, Self) {
            (self / rhs, Self::ZERO)
        }

        #[inline]
        fn overflowing_rem_backend(self, rhs: Self) -> (Self, Self) {
            (self % rhs, Self::ZERO)
        }
    };
}

impl_simd_integer!(
    T = i8,
    N = 16,
    Simd = i8x16,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u8x16,
    SignedSimd = i8x16,
    WiderSimd = { i16x16 },
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);
impl_simd_integer!(
    T = i8,
    N = 32,
    Simd = i8x32,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u8x32,
    SignedSimd = i8x32,
    WiderSimd = { i16x32 },
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    INDICES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ],
);
impl_simd_integer!(
    T = i8,
    N = 64,
    Simd = i8x64,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u8x64,
    SignedSimd = i8x64,
    WiderSimd = { /*i16x64*/ },
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    INDICES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
);
impl_simd_integer!(
    T = i16,
    N = 8,
    Simd = i16x8,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u16x8,
    SignedSimd = i16x8,
    WiderSimd = { i32x8 },
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7],
);
impl_simd_integer!(
    T = i16,
    N = 16,
    Simd = i16x16,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u16x16,
    SignedSimd = i16x16,
    WiderSimd = { i32x16 },
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);
impl_simd_integer!(
    T = i16,
    N = 32,
    Simd = i16x32,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u16x32,
    SignedSimd = i16x32,
    WiderSimd = { /*i32x32*/ },
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    INDICES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ],
);
impl_simd_integer!(
    T = i32,
    N = 4,
    Simd = i32x4,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u32x4,
    SignedSimd = i32x4,
    WiderSimd = { i64x4 },
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    INDICES = [0, 1, 2, 3],
);
impl_simd_integer!(
    T = i32,
    N = 8,
    Simd = i32x8,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u32x8,
    SignedSimd = i32x8,
    WiderSimd = { i64x8 },
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7],
);
impl_simd_integer!(
    T = i32,
    N = 16,
    Simd = i32x16,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u32x16,
    SignedSimd = i32x16,
    WiderSimd = { /*i64x16*/ },
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);
impl_simd_integer!(
    T = i64,
    N = 2,
    Simd = i64x2,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u64x2,
    SignedSimd = i64x2,
    WiderSimd = { /*i128x2*/ },
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    INDICES = [0, 1],
);
impl_simd_integer!(
    T = i64,
    N = 4,
    Simd = i64x4,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u64x4,
    SignedSimd = i64x4,
    WiderSimd = { /*i128x4*/ },
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    INDICES = [0, 1, 2, 3],
);
impl_simd_integer!(
    T = i64,
    N = 8,
    Simd = i64x8,
    autofill = simd_integer_backend_signed_autofill,
    UnsignedSimd = u64x8,
    SignedSimd = i64x8,
    WiderSimd = { /*i128x8*/ },
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7],
);
impl_simd_integer!(
    T = u8,
    N = 16,
    Simd = u8x16,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u8x16,
    SignedSimd = i8x16,
    WiderSimd = { u16x16 },
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);
impl_simd_integer!(
    T = u8,
    N = 32,
    Simd = u8x32,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u8x32,
    SignedSimd = i8x32,
    WiderSimd = { u16x32 },
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    INDICES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ],
);
impl_simd_integer!(
    T = u8,
    N = 64,
    Simd = u8x64,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u8x64,
    SignedSimd = i8x64,
    WiderSimd = { /*u16x64*/ },
    T_BITS = 8,
    T_BITS_MUL_2 = 16,
    INDICES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
);
impl_simd_integer!(
    T = u16,
    N = 8,
    Simd = u16x8,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u16x8,
    SignedSimd = i16x8,
    WiderSimd = { u32x8 },
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7],
);
impl_simd_integer!(
    T = u16,
    N = 16,
    Simd = u16x16,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u16x16,
    SignedSimd = i16x16,
    WiderSimd = { u32x16 },
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);
impl_simd_integer!(
    T = u16,
    N = 32,
    Simd = u16x32,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u16x32,
    SignedSimd = i16x32,
    WiderSimd = { /*u32x32*/ },
    T_BITS = 16,
    T_BITS_MUL_2 = 32,
    INDICES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ],
);
impl_simd_integer!(
    T = u32,
    N = 4,
    Simd = u32x4,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u32x4,
    SignedSimd = i32x4,
    WiderSimd = { u64x4 },
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    INDICES = [0, 1, 2, 3],
);
impl_simd_integer!(
    T = u32,
    N = 8,
    Simd = u32x8,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u32x8,
    SignedSimd = i32x8,
    WiderSimd = { u64x8 },
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7],
);
impl_simd_integer!(
    T = u32,
    N = 16,
    Simd = u32x16,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u32x16,
    SignedSimd = i32x16,
    WiderSimd = { /*u64x16*/ },
    T_BITS = 32,
    T_BITS_MUL_2 = 64,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
);
impl_simd_integer!(
    T = u64,
    N = 2,
    Simd = u64x2,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u64x2,
    SignedSimd = i64x2,
    WiderSimd = { /*u128x2*/ },
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    INDICES = [0, 1],
);
impl_simd_integer!(
    T = u64,
    N = 4,
    Simd = u64x4,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u64x4,
    SignedSimd = i64x4,
    WiderSimd = { /*u128x4*/ },
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    INDICES = [0, 1, 2, 3],
);
impl_simd_integer!(
    T = u64,
    N = 8,
    Simd = u64x8,
    autofill = simd_integer_backend_unsigned_autofill,
    UnsignedSimd = u64x8,
    SignedSimd = i64x8,
    WiderSimd = { /*u128x8*/ },
    T_BITS = 64,
    T_BITS_MUL_2 = 128,
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7],
);
