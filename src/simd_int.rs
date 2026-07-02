macro_rules! impl_simd_int {
  (
    T = $T:ident,
    N = $N:literal,
    Simd = $Simd:ident,
    [$($index:literal),* $(,)?],

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

    impl $Simd {
      pub const ONE: Self = Self::splat(1);
      pub const ZERO: Self = Self::splat(0);
      pub const MAX: Self = Self::splat($T::MAX);
      pub const MIN: Self = Self::splat($T::MIN);

      /// The number of lanes in this SIMD vector.
      pub const LANES: u16 = $N;

      /// The size of this SIMD vector in bits.
      pub const BITS: u16 = $N;
    }
  };
}
