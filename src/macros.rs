macro_rules! int_uint_consts_inner {
  ($type:ty, $lanes:expr, $simd:ty, $macro_name:ident, $bits:expr) => {
    impl $simd {
      $macro_name!(ONE, 1);
      $macro_name!(ZERO, 0);
      $macro_name!(MAX, <$type>::MAX);
      $macro_name!(MIN, <$type>::MIN);

      /// The number of lanes in this SIMD vector.
      pub const LANES: u16 = $lanes;

      /// The size of this SIMD vector in bits.
      pub const BITS: u16 = $bits;
    }
  };
}

macro_rules! int_uint_consts {
  ($type:ty, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 128) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack128bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    int_uint_consts_inner!($type, $lanes, $simd_type, $macro_name, 128);
  };
  ($type:ty, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 256) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack256bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    int_uint_consts_inner!($type, $lanes, $simd_type, $macro_name, 256);
  };
}
