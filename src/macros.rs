macro_rules! int_uint_consts {
  ($type:ty, $lanes:expr, $simd:ty, $bits:expr) => {
    // ensure the size of the SIMD type is the same as the size of the array and
    // number of bits is OK
    const _: () = assert!(
      core::mem::size_of::<$simd>() == core::mem::size_of::<[$type; $lanes]>()
        && core::mem::size_of::<$simd>() * 8 == $bits as usize
    );

    impl $simd {
      pub const ONE: $simd = <$simd>::new([1; $lanes]);
      pub const ZERO: $simd = <$simd>::new([0; $lanes]);
      pub const MAX: $simd = <$simd>::new([<$type>::MAX; $lanes]);
      pub const MIN: $simd = <$simd>::new([<$type>::MIN; $lanes]);

      /// The number of lanes in this SIMD vector.
      pub const LANES: u16 = $lanes;

      /// The size of this SIMD vector in bits.
      pub const BITS: u16 = $bits;
    }
  };
}

macro_rules! simd_comparison_fns {
  () => {
    /// Test if each element is equal to the corresponding element in `other`.
    #[inline]
    #[expect(deprecated)]
    pub fn simd_eq<Rhs>(self, other: Rhs) -> <Self as CmpEq<Rhs>>::Output
    where
      Self: CmpEq<Rhs>,
    {
      CmpEq::simd_eq(self, other)
    }

    /// Test if each element is not equal to the corresponding element in
    /// `other`.
    #[inline]
    #[expect(deprecated)]
    pub fn simd_ne<Rhs>(self, other: Rhs) -> <Self as CmpNe<Rhs>>::Output
    where
      Self: CmpNe<Rhs>,
    {
      CmpNe::simd_ne(self, other)
    }

    /// Test if each element is less than the corresponding element in `other`.
    #[inline]
    #[expect(deprecated)]
    pub fn simd_lt<Rhs>(self, other: Rhs) -> <Self as CmpLt<Rhs>>::Output
    where
      Self: CmpLt<Rhs>,
    {
      CmpLt::simd_lt(self, other)
    }

    /// Test if each element is greater than the corresponding element in
    /// `other`.
    #[inline]
    #[expect(deprecated)]
    pub fn simd_gt<Rhs>(self, other: Rhs) -> <Self as CmpGt<Rhs>>::Output
    where
      Self: CmpGt<Rhs>,
    {
      CmpGt::simd_gt(self, other)
    }

    /// Test if each element is less than or equal to the corresponding element
    /// in `other`.
    #[inline]
    #[expect(deprecated)]
    pub fn simd_le<Rhs>(self, other: Rhs) -> <Self as CmpLe<Rhs>>::Output
    where
      Self: CmpLe<Rhs>,
    {
      CmpLe::simd_le(self, other)
    }

    /// Test if each element is greater than or equal to the corresponding
    /// element in `other`.
    #[inline]
    #[expect(deprecated)]
    pub fn simd_ge<Rhs>(self, other: Rhs) -> <Self as CmpGe<Rhs>>::Output
    where
      Self: CmpGe<Rhs>,
    {
      CmpGe::simd_ge(self, other)
    }
  };
}
