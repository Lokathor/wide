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

/// Test a vector operation against a scalar operation for random values to make sure that the behavior is the same.
/// This allows for easier for correctness for various values of the vector.
/// Uses a simple linear congruential generator to generate random values to avoid pulling in a pile of dependencies.
fn test_random_vector_vs_scalar<
  V,
  T,
  FnVec: Fn(V, V) -> V,
  FnScalar: Fn(T, T) -> T,
  const N: usize,
>(
  vector_fn: FnVec,
  scalar_fn: FnScalar,
) where
  V: From<[T; N]> + Into<[T; N]> + Copy + std::fmt::Debug,
  T: Copy + PartialEq + std::fmt::Debug + Default + GenSample,
{
  let mut a_arr = [T::default(); N];
  let mut b_arr: [T; N] = [T::default(); N];

  // simple linear congruential generator
  const A: u64 = 2862933555777941757;
  const B: u64 = 3037000493;
  let mut var = 1;

  // do 100 iterations
  for _i in 0..100 {
    for i in 0..N {
      a_arr[i] = T::get_sample(var);
      var = var.wrapping_mul(A).wrapping_add(B);
      b_arr[i] = T::get_sample(var);
      var = var.wrapping_mul(A).wrapping_add(B);
    }

    let mut expected_arr: [T; N] = [T::default(); N];
    for i in 0..N {
      expected_arr[i] = scalar_fn(a_arr[i], b_arr[i]);
    }

    let expected_vec = vector_fn(V::from(a_arr), V::from(b_arr));
    assert_eq!(expected_arr, expected_vec.into());
  }
}

/// trait to reduce a 64 bit pseudo-random number to a random sample value
trait GenSample {
  fn get_sample(v: u64) -> Self;
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
