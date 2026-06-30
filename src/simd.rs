macro_rules! impl_simd {
  (
    T = $T:ident,
    N = $N:literal,
    Simd = $Simd:ident,
  ) => {
    impl $Simd {
      #[inline]
      #[must_use]
      pub const fn new(array: [$T; $N]) -> Self {
        unsafe { core::mem::transmute::<[$T; $N], $Simd>(array) }
      }
    }
  };
}
