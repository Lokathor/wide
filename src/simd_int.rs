macro_rules! impl_simd_int {
  (
    // SAFETY: The contents of this macro assume that:
    //
    // - `T` implements `Pod`
    // - `Pod` can be implemented for `Simd`
    // - `size_of::<Simd>()` is `size_of::<T>() * N`
    // - `align_of::<Simd>()` is `size_of::<Simd>()`
    unsafe {
      T = $T:ident,
      N = $N:literal,
      Simd = $Simd:ident,
      UnsignedSimd = $UnsignedSimd:ident,
      [$($index:literal),* $(,)?],
    }

    $fn_not:item
    $fn_add:item
    $fn_sub:item
    $fn_mul:item
    $fn_shl:item
    $fn_shl_u32:item
    $fn_shr:item
    $fn_shr_u32:item
    $fn_bitand:item
    $fn_bitor:item
    $fn_bitxor:item
    $fn_max:item
    $fn_min:item
    $fn_reduce_add:item
    $fn_reduce_mul:item
    $fn_reduce_max:item
    $fn_reduce_min:item
    $fn_saturating_add:item
    $fn_saturating_sub:item
    $fn_saturating_mul:item
    $fn_overflowing_mul:item
    $fn_abs:item
    $fn_is_positive:item
    $fn_is_negative:item
  ) => {
    impl_unary_operator!(
      $Simd,
      Neg,
      neg,
      #[inline]
      fn neg(self) -> Self::Output {
        Self::default() - self
      }
    );
    impl_unary_operator!($Simd, Not, not, $fn_not);

    impl_binary_operator!($T, $Simd, Add, add, AddAssign, add_assign, $fn_add);
    impl_binary_operator!($T, $Simd, Sub, sub, SubAssign, sub_assign, $fn_sub);
    impl_binary_operator!($T, $Simd, Mul, mul, MulAssign, mul_assign, $fn_mul);
    impl_binary_operator!(
        $T,
        $Simd,
        Div,
        div,
        DivAssign,
        div_assign,
        #[inline]
        fn div(self, rhs: Self) -> Self::Output {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::new([$(self_array[$index].wrapping_div(rhs_array[$index])),*])
        },
        /// Lanewise divide.
        ///
        /// Note that because division has no hardware support, this operation
        /// is very slow and should be avoided if possible.
        ,
        /// Lanewise divide.
        ///
        /// Note that because division has no hardware support, this operation
        /// is very slow and should be avoided if possible.
        ,
        /// Lanewise divide.
        ///
        /// Note that because division has no hardware support, this operation
        /// is very slow and should be avoided if possible.
    );
    impl_binary_operator!(
        $T,
        $Simd,
        Rem,
        rem,
        RemAssign,
        rem_assign,
        #[inline]
        fn rem(self, rhs: Self) -> Self::Output {
            let self_array = self.to_array();
            let rhs_array = rhs.to_array();

            Self::new([$(self_array[$index].wrapping_rem(rhs_array[$index])),*])
        },
        /// Lanewise remainder.
        ///
        /// Note that because division has no hardware support, this operation
        /// is very slow and should be avoided if possible.
        ,
        /// Lanewise remainder.
        ///
        /// Note that because division has no hardware support, this operation
        /// is very slow and should be avoided if possible.
        ,
        /// Lanewise remainder.
        ///
        /// Note that because division has no hardware support, this operation
        /// is very slow and should be avoided if possible.
    );
    impl_shift_operator!(
      $T,
      $Simd,
      Shl,
      shl,
      ShlAssign,
      shl_assign,
      $fn_shl,
      $fn_shl_u32,
      /// Shifts lanes by the corresponding lane.
      ///
      /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the
      /// bitwidth of the type. (same as `wrapping_shl`)
      ,
      /// Shifts all lanes by the value given.
      ///
      /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the
      /// bitwidth of the type. (same as `wrapping_shl`)
      ,
      /// Shifts the same value by each lane, returning a SIMD type.
      ///
      /// Bitwise shift-left; yields `self << mask(rhs)`, where mask removes any
      /// high-order bits of `rhs` that would cause the shift to exceed the
      /// bitwidth of the type. (same as `wrapping_shl`)
    );
    impl_shift_operator!(
      $T,
      $Simd,
      Shr,
      shr,
      ShrAssign,
      shr_assign,
      $fn_shr,
      $fn_shr_u32,
      /// Shifts each lane individually.
      ///
      /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes
      /// any high-order bits of `rhs` that would cause the shift to exceed the
      /// bitwidth of the type. (same as `wrapping_shr`)
      ,
      /// Shifts all lanes by the value given.
      ///
      /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes
      /// any high-order bits of `rhs` that would cause the shift to exceed the
      /// bitwidth of the type. (same as `wrapping_shr`)
      ,
      /// Shifts the same value by each lane, returning a SIMD type.
      ///
      /// Bitwise shift-right; yields `self >> mask(rhs)`, where mask removes
      /// any high-order bits of `rhs` that would cause the shift to exceed the
      /// bitwidth of the type. (same as `wrapping_shr`)
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitAnd,
      bitand,
      BitAndAssign,
      bitand_assign,
      $fn_bitand
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitOr,
      bitor,
      BitOrAssign,
      bitor_assign,
      $fn_bitor
    );
    impl_binary_operator!(
      $T,
      $Simd,
      BitXor,
      bitxor,
      BitXorAssign,
      bitxor_assign,
      $fn_bitxor
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

    impl $Simd {
      pub const ONE: Self = Self::splat(1);
      pub const ZERO: Self = Self::splat(0);
      pub const MAX: Self = Self::splat($T::MAX);
      pub const MIN: Self = Self::splat($T::MIN);

      /// The number of lanes in this SIMD vector.
      pub const LANES: u16 = $N;

      /// The size of this SIMD vector in bits.
      pub const BITS: u16 = $N;

      #[must_use]
      $fn_max

      #[must_use]
      $fn_min

      /// Restrict each element to a certain interval.
      ///
      /// If `min > max`, the result is unspeficied. Consider manually checking
      /// for that case.
      #[inline]
      #[must_use]
      pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
      }

      /// horizontal add of all the elements of the vector
      #[must_use]
      $fn_reduce_add

      /// Reducing multiply. Returns the product of the elements of the vector.
      #[must_use]
      $fn_reduce_mul

      /// horizontal max of all the elements of the vector
      #[must_use]
      $fn_reduce_max

      /// horizontal min of all the elements of the vector
      #[must_use]
      $fn_reduce_min

      #[must_use]
      $fn_saturating_add

      #[must_use]
      $fn_saturating_sub

      /// Lanewise saturating multiply.
      #[must_use]
      $fn_saturating_mul

      /// Lanewise saturating divide.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      #[must_use]
      pub fn saturating_div(self, rhs: Self) -> Self {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([$(self_array[$index].saturating_div(rhs_array[$index])),*])
      }

      #[inline]
      #[must_use]
      pub fn unsigned_abs(self) -> $UnsignedSimd {
        cast::<$Simd, $UnsignedSimd>(self.abs())
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
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();

        (result, overflow)
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
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();

        (result, overflow)
      }

      /// Returns `self * rhs` and whether an overflow occured.
      ///
      /// Returns a tuple with:
      ///
      /// - The multiplication (returns the wrapped value if an overflow
      ///   occured)
      /// - A mask indicating whether an overflow occured
      #[must_use]
      $fn_overflowing_mul

      /// Returns `self / rhs` and whether an overflow occured.
      ///
      /// Returns a tuple with:
      ///
      /// - The division (returns `self` if an overflow occured)
      /// - A mask indicating whether an overflow occured
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      #[must_use]
      pub fn overflowing_div(self, rhs: Self) -> (Self, Self) {
        // The second field is equivalent to
        // `self.simd_eq(Self::MIN) & rhs.simd_eq(-1)` but may be cheaper.
        (self / rhs, ((self ^ Self::MAX) & rhs).simd_eq(!Self::ZERO))
      }

      /// Returns `self % rhs` and whether an overflow occured.
      ///
      /// Returns a tuple with:
      ///
      /// - The remainder (returns zero if an overflow occured)
      /// - A mask indicating whether an overflow occured
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      #[must_use]
      pub fn overflowing_rem(self, rhs: Self) -> (Self, Self) {
        // The second field is equivalent to
        // `self.simd_eq(Self::MIN) & rhs.simd_eq(-1)` but may be cheaper.
        (self % rhs, ((self ^ Self::MAX) & rhs).simd_eq(!Self::ZERO))
      }

      #[must_use]
      $fn_abs

      /// Returns numbers representing the sign of each element.
      ///
      /// - `0` if the number is zero
      /// - `1` if the number is positive
      /// - `-1` if the number is negative
      #[inline]
      #[must_use]
      pub fn signum(self) -> Self {
        // Flip signs because the result for true in `is_positive/negative` is
        // `-1` (all bits set).
        self.is_negative() - self.is_positive()
      }

      /// Returns true for each positive element and false if it is zero or
      /// negative.
      #[must_use]
      $fn_is_positive

      /// Returns true for each negative element and false if it is zero or
      /// positive.
      #[must_use]
      $fn_is_negative
    }
  };
}
