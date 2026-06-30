macro_rules! impl_simd {
  (
    T = $T:ident,
    N = $N:literal,
    Simd = $Simd:ident,
  ) => {
    impl From<$T> for $Simd {
      /// Splats the single value given across all lanes.
      #[inline]
      fn from(elem: $T) -> Self {
        Self::splat(elem)
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
    }
  };
}
