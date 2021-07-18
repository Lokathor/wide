macro_rules! impl_nonfloat_consts_inner_impl {
  ($type:ty, $simd:ty, $macro_name:ident) => {
    impl $simd {
      $macro_name!(ONE, 1);
      $macro_name!(ZERO, 0);
      $macro_name!(MAX, <$type>::MAX);
      $macro_name!(MIN, <$type>::MIN);
    }
  };
}

macro_rules! impl_nonfloat_consts {
  ($type:ty, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 128) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack128bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    impl_nonfloat_consts_inner_impl!($type, $simd_type, $macro_name);
  };
  ($type:ty, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 256) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack256bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    impl_nonfloat_consts_inner_impl!($type, $simd_type, $macro_name);
  };
}
