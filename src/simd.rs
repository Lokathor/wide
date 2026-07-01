macro_rules! impl_simd {
  (
    T = $T:ident,
    N = $N:literal,
    Simd = $Simd:ident,

    $fn_simd_eq:item
    $fn_simd_ne:item
    $fn_simd_lt:item
    $fn_simd_gt:item
    $fn_simd_le:item
    $fn_simd_ge:item
  ) => {
    impl From<[$T; $N]> for $Simd {
      #[inline]
      fn from(arr: [$T; $N]) -> Self {
        Self::new(arr)
      }
    }

    impl From<$Simd> for [$T; $N] {
      #[inline]
      fn from(simd: $Simd) -> Self {
        simd.to_array()
      }
    }

    impl From<$T> for $Simd {
      /// Splats the single value given across all lanes.
      #[inline]
      fn from(elem: $T) -> Self {
        Self::splat(elem)
      }
    }

    #[expect(deprecated)]
    impl CmpEq for $Simd {
      type Output = Self;

      $fn_simd_eq
    }

    #[expect(deprecated)]
    impl CmpEq<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_eq(self, rhs: $T) -> Self::Output {
        self.simd_eq(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpNe for $Simd {
      type Output = Self;

      $fn_simd_ne
    }

    #[expect(deprecated)]
    impl CmpNe<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_ne(self, rhs: $T) -> Self::Output {
        self.simd_ne(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpLt for $Simd {
      type Output = Self;

      $fn_simd_lt
    }

    #[expect(deprecated)]
    impl CmpLt<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_lt(self, rhs: $T) -> Self::Output {
        self.simd_lt(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpGt for $Simd {
      type Output = Self;

      $fn_simd_gt
    }

    #[expect(deprecated)]
    impl CmpGt<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_gt(self, rhs: $T) -> Self::Output {
        self.simd_gt(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpLe for $Simd {
      type Output = Self;

      $fn_simd_le
    }

    #[expect(deprecated)]
    impl CmpLe<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_le(self, rhs: $T) -> Self::Output {
        self.simd_le(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpGe for $Simd {
      type Output = Self;

      $fn_simd_ge
    }

    #[expect(deprecated)]
    impl CmpGe<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_ge(self, rhs: $T) -> Self::Output {
        self.simd_ge(Self::splat(rhs))
      }
    }

    impl $Simd {
      #[inline]
      #[must_use]
      pub const fn new(array: [$T; $N]) -> Self {
        unsafe { core::mem::transmute::<[$T; $N], $Simd>(array) }
      }

      #[inline]
      #[must_use]
      pub const fn splat(elem: $T) -> Self {
        unsafe { core::mem::transmute::<[$T; $N], $Simd>([elem; $N]) }
      }

      #[inline]
      #[must_use]
      pub fn to_array(self) -> [$T; $N] {
        cast(self)
      }

      #[inline]
      #[must_use]
      pub fn as_array(&self) -> &[$T; $N] {
        cast_ref(self)
      }

      #[inline]
      #[must_use]
      pub fn as_mut_array(&mut self) -> &mut [$T; $N] {
        cast_mut(self)
      }

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
    }
  };
}
