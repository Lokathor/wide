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

macro_rules! float_consts_inner {
  (f32, $simd:ty, $macro_name:ident) => {
    impl $simd {
      $macro_name!(ONE, 1.0);
      $macro_name!(ZERO, 0.0);
      $macro_name!(HALF, 0.5);
      $macro_name!(E, core::f32::consts::E);
      $macro_name!(FRAC_1_PI, core::f32::consts::FRAC_1_PI);
      $macro_name!(FRAC_2_PI, core::f32::consts::FRAC_2_PI);
      $macro_name!(FRAC_2_SQRT_PI, core::f32::consts::FRAC_2_SQRT_PI);
      $macro_name!(FRAC_1_SQRT_2, core::f32::consts::FRAC_1_SQRT_2);
      $macro_name!(FRAC_PI_2, core::f32::consts::FRAC_PI_2);
      $macro_name!(FRAC_PI_3, core::f32::consts::FRAC_PI_3);
      $macro_name!(FRAC_PI_4, core::f32::consts::FRAC_PI_4);
      $macro_name!(FRAC_PI_6, core::f32::consts::FRAC_PI_6);
      $macro_name!(FRAC_PI_8, core::f32::consts::FRAC_PI_8);
      $macro_name!(LN_2, core::f32::consts::LN_2);
      $macro_name!(LN_10, core::f32::consts::LN_10);
      $macro_name!(LOG2_E, core::f32::consts::LOG2_E);
      $macro_name!(LOG10_E, core::f32::consts::LOG10_E);
      $macro_name!(LOG10_2, core::f32::consts::LOG10_2);
      $macro_name!(LOG2_10, core::f32::consts::LOG2_10);
      $macro_name!(PI, core::f32::consts::PI);
      $macro_name!(SQRT_2, core::f32::consts::SQRT_2);
      $macro_name!(TAU, 6.283_185_307_179_586_476_925_286_766_559_005_77_f32);
    }
  };
  (f64, $simd:ty, $macro_name:ident) => {
    impl $simd {
      $macro_name!(ONE, 1.0);
      $macro_name!(ZERO, 0.0);
      $macro_name!(HALF, 0.5);
      $macro_name!(E, core::f64::consts::E);
      $macro_name!(FRAC_1_PI, core::f64::consts::FRAC_1_PI);
      $macro_name!(FRAC_2_PI, core::f64::consts::FRAC_2_PI);
      $macro_name!(FRAC_2_SQRT_PI, core::f64::consts::FRAC_2_SQRT_PI);
      $macro_name!(FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2);
      $macro_name!(FRAC_PI_2, core::f64::consts::FRAC_PI_2);
      $macro_name!(FRAC_PI_3, core::f64::consts::FRAC_PI_3);
      $macro_name!(FRAC_PI_4, core::f64::consts::FRAC_PI_4);
      $macro_name!(FRAC_PI_6, core::f64::consts::FRAC_PI_6);
      $macro_name!(FRAC_PI_8, core::f64::consts::FRAC_PI_8);
      $macro_name!(LN_2, core::f64::consts::LN_2);
      $macro_name!(LN_10, core::f64::consts::LN_10);
      $macro_name!(LOG2_E, core::f64::consts::LOG2_E);
      $macro_name!(LOG10_E, core::f64::consts::LOG10_E);
      $macro_name!(LOG10_2, core::f64::consts::LOG10_2);
      $macro_name!(LOG2_10, core::f64::consts::LOG2_10);
      $macro_name!(PI, core::f64::consts::PI);
      $macro_name!(SQRT_2, core::f64::consts::SQRT_2);
      $macro_name!(TAU, 6.283_185_307_179_586_476_925_286_766_559_005_77_f64);
    }
  };
}

macro_rules! float_consts {
  (f32, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 128) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack128bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    float_consts_inner!(f32, $simd_type, $macro_name);
  };
  (f32, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 256) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack256bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    float_consts_inner!(f32, $simd_type, $macro_name);
  };
  (f64, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 128) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack128bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    float_consts_inner!(f64, $simd_type, $macro_name);
  };
  (f64, $lanes:expr, $simd_type:ty, $simd_ident:ident, $aligned:ident, $macro_name:ident, 256) => {
    macro_rules! $macro_name {
      ($i: ident, $f: expr) => {
        pub const $i: $simd_type = unsafe {
          ConstUnionHack256bit { $aligned: [$f; $lanes] }.$simd_ident
        };
      };
    }

    float_consts_inner!(f64, $simd_type, $macro_name);
  };
}
