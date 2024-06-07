#![allow(clippy::approx_constant)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::assertions_on_constants)]

mod t_f32x4;
mod t_f32x8;
mod t_f64x2;
mod t_f64x4;
mod t_i16x16;
mod t_i16x8;
mod t_i32x4;
mod t_i32x8;
mod t_i64x2;
mod t_i64x4;
mod t_i8x16;
mod t_i8x32;
mod t_u16x16;
mod t_u16x8;
mod t_u32x4;
mod t_u32x8;
mod t_u64x2;
mod t_u64x4;
mod t_u8x16;
mod t_usefulness;

/// Uses a simple linear congruential
/// generator to generate random values to avoid pulling in a pile of
/// dependencies.
fn lcg(var: &mut u64) -> u64 {
  const A: u64 = 2862933555777941757;
  const B: u64 = 3037000493;

  *var = var.wrapping_mul(A).wrapping_add(B);

  // generate more ff and 00 than we normally would to test edge cases
  match *var & 0xf {
    1 => 0,
    2 => u64::MAX,
    3 => 0b0101010101010101010101010101010101010101010101010101010101010101,
    _ => *var,
  }
}

/// Test a vector operation against a scalar operation for random values to make
/// sure that the behavior is the same. This allows for easier for correctness
/// for various values of the vector.
fn test_random_vector_vs_scalar<
  V,
  VR,
  T,
  TR,
  FnVec: Fn(V, V) -> VR,
  FnScalar: Fn(T, T) -> TR,
  const N: usize,
>(
  vector_fn: FnVec,
  scalar_fn: FnScalar,
) where
  V: Copy + From<[T; N]>,
  T: Copy + Default + std::fmt::Debug + GenSample,
  TR: Copy + PartialEq + std::fmt::Debug + Default + GenSample,
  VR: Copy + Into<[TR; N]>,
{
  let mut a_arr = [T::default(); N];
  let mut b_arr: [T; N] = [T::default(); N];

  // simple linear congruential generator
  let mut var = 1;

  // do 100 iterations
  for _i in 0..100 {
    for i in 0..N {
      a_arr[i] = T::get_sample(lcg(&mut var));
      b_arr[i] = T::get_sample(lcg(&mut var));
    }

    let mut expected_arr: [TR; N] = [TR::default(); N];
    for i in 0..N {
      expected_arr[i] = scalar_fn(a_arr[i], b_arr[i]);
    }

    let expected_vec_arr: [TR; N] =
      vector_fn(V::from(a_arr), V::from(b_arr)).into();

    for i in 0..N {
      assert!(
        expected_arr[i].binary_eq(expected_vec_arr[i]),
        "scalar = {:?}\nvec = {:?}\na = {:?}\nb = {:?}",
        expected_arr,
        expected_vec_arr,
        a_arr,
        b_arr
      );
    }
  }
}

/// Test a vector operation against a scalar operation for random values to make
/// sure that the behavior is the same. This allows for easier for correctness
/// for various values of the vector.
fn test_random_vector_vs_scalar_reduce<
  V,
  T,
  TR,
  FnVec: Fn(V) -> TR,
  FnScalar: Fn(TR, T, usize) -> TR,
  const N: usize,
>(
  vector_fn: FnVec,
  acc: TR,
  scalar_fn: FnScalar,
) where
  V: From<[T; N]> + Into<[T; N]> + Copy + std::fmt::Debug,
  T: Copy + PartialEq + std::fmt::Debug + Default + GenSample,
  TR: Copy + PartialEq + std::fmt::Debug + Default,
{
  let mut a_arr = [T::default(); N];

  // simple linear congruential generator
  let mut var = 1;

  // do 100 iterations
  for _i in 0..100 {
    for i in 0..N {
      a_arr[i] = T::get_sample(lcg(&mut var));
    }

    let expected_scalar =
      a_arr.iter().enumerate().fold(acc, |acc, (i, &v)| scalar_fn(acc, v, i));

    let expected_vec = vector_fn(V::from(a_arr));
    assert_eq!(expected_scalar, expected_vec);
  }
}

/// trait to reduce a 64 bit pseudo-random number to a random sample value
trait GenSample
where
  Self: PartialEq + Copy,
{
  fn get_sample(v: u64) -> Self;
  fn binary_eq(self, b: Self) -> bool {
    self == b
  }
}

impl GenSample for u64 {
  fn get_sample(v: u64) -> Self {
    v
  }
}

impl GenSample for u32 {
  fn get_sample(v: u64) -> Self {
    v as u32
  }
}

impl GenSample for u16 {
  fn get_sample(v: u64) -> Self {
    v as u16
  }
}

impl GenSample for u8 {
  fn get_sample(v: u64) -> Self {
    v as u8
  }
}

impl GenSample for i64 {
  fn get_sample(v: u64) -> Self {
    v as i64
  }
}

impl GenSample for i32 {
  fn get_sample(v: u64) -> Self {
    v as i32
  }
}

impl GenSample for i16 {
  fn get_sample(v: u64) -> Self {
    v as i16
  }
}

impl GenSample for i8 {
  fn get_sample(v: u64) -> Self {
    v as i8
  }
}

impl GenSample for f32 {
  fn get_sample(v: u64) -> Self {
    match v & 31 {
      0 => f32::NAN,
      1 => f32::INFINITY,
      2 => f32::NEG_INFINITY,
      _ => (v as i64 as f32).sqrt(),
    }
  }

  /// floating points Nan always fails equality so we need to special case it
  fn binary_eq(self, b: Self) -> bool {
    if self.is_nan() {
      b.is_nan()
    } else if self.is_infinite() {
      b.is_infinite() && self.is_sign_positive() == b.is_sign_positive()
    } else {
      self == b
    }
  }
}

impl GenSample for f64 {
  fn get_sample(v: u64) -> Self {
    match v & 31 {
      0 => f64::NAN,
      1 => f64::INFINITY,
      2 => f64::NEG_INFINITY,
      _ => (v as i64 as f64).sqrt(),
    }
  }

  /// floating points Nan always fails equality so we need to special case it
  fn binary_eq(self, b: Self) -> bool {
    if self.is_nan() {
      b.is_nan()
    } else if self.is_infinite() {
      b.is_infinite() && self.is_sign_positive() == b.is_sign_positive()
    } else {
      self == b
    }
  }
}
