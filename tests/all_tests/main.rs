#![allow(clippy::approx_constant)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::nonminimal_bool)]
#![allow(unused_imports)]
#![allow(clippy::precedence)]
#![allow(clippy::eq_op)]
#![allow(clippy::identity_op)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_closure)]

use core::fmt;
use std::{num::Wrapping, ops::ShlAssign};

use wide::AlignTo;

mod t_f32x16;
mod t_f32x4;
mod t_f32x8;
mod t_f64x2;
mod t_f64x4;
mod t_f64x8;
mod t_i16x16;
mod t_i16x32;
mod t_i16x8;
mod t_i32x16;
mod t_i32x4;
mod t_i32x8;
mod t_i64x2;
mod t_i64x4;
mod t_i64x8;
mod t_i8x16;
mod t_i8x32;
mod t_u16x16;
mod t_u16x32;
mod t_u16x8;
mod t_u32x16;
mod t_u32x4;
mod t_u32x8;
mod t_u64x2;
mod t_u64x4;
mod t_u64x8;
mod t_u8x16;
mod t_u8x32;
mod t_usefulness;

/// Generates the next pseudo-random number.
/// Definitely non-cryptographic, just used for generating random test values.
fn next_rand_u64(state: &mut u64) -> u64 {
  // Constants for the LCG
  const A: u64 = 6364136223846793005;
  const C: u64 = 1442695040888963407;

  // Update the state and calculate the next number (rotate to avoid lack of
  // randomness in low bits)
  *state = state.wrapping_mul(A).wrapping_add(C).rotate_left(31);

  *state
}

const RNG_SEED: u64 = 0x123456789abcdef0;

/// Generate a pseudo-random value for a type that implements GenSample.
fn gen_random<T: GenSample>(rng: &mut u64) -> T {
  let r = next_rand_u64(rng);

  // generate special values more often than random chance to test edge cases
  let next = match r & 0xf {
    0 => 0,
    1 => 1,
    2 => u64::MAX,
    _ => next_rand_u64(rng),
  };

  T::get_sample(next)
}

/// Test a vector operation against a pure scalar implementation for random
/// values to make sure that the behavior is the same. This allows for easier
/// for correctness for various values of the vector.
#[track_caller]
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

  // use a fixed seed for reproducibility
  let mut rng = RNG_SEED;

  // do 100 iterations
  for _i in 0..100 {
    for i in 0..N {
      a_arr[i] = gen_random(&mut rng);
      b_arr[i] = gen_random(&mut rng);
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
        "scalar = {:?}\nvec = {:?}\na = {:?}\nb = {:?} caller={:?}",
        expected_arr,
        expected_vec_arr,
        a_arr,
        b_arr,
        std::panic::Location::caller()
      );
    }
  }
}

/// Test a vector reduce operations that generate a scalar from a vector
/// against a pure scalar implementation for random values to make
/// sure that the behavior is the same. This allows for easier for correctness
/// for various values of the vector.
///
/// The scalar operation uses the same construction as the Rust fold function
/// which takes an accumulator and returns the accumulator after applying the
/// operation.
#[track_caller]
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

  // use a fixed seed for reproducibility
  let mut rng = RNG_SEED;

  // do 100 iterations
  for _i in 0..100 {
    for i in 0..N {
      a_arr[i] = gen_random(&mut rng);
    }

    let mut expected_scalar = acc;
    for i in 0..N {
      expected_scalar = scalar_fn(expected_scalar, a_arr[i], i);
    }

    let expected_vec = vector_fn(V::from(a_arr));
    assert_eq!(
      expected_scalar,
      expected_vec,
      "scalar = {:?} vec = {:?} source = {:?} caller={:?}",
      expected_scalar,
      expected_vec,
      a_arr,
      std::panic::Location::caller()
    );
  }
}

/// Trait to implement various basic tests for SIMD vector types.
///
/// This is a bit of a weird use of traits, but it allows us to easily implement
/// test methods that can be called as associated functions on the vector types
/// themselves, e.g. `f32x4::test_basic_traits()`.
///
/// This has a couple nice features:
/// - The values for V, T and N are inferred from the type that the trait is
///   implemented for, so we don't have to specify them manually.
/// - The associated functions can take advantage of the basic traits that are
///   already required for the type at the trait level, and not have to
///   re-specify them for each function.
trait TestBasicTraits<V, T, const N: usize>
where
  V: Copy
    + Clone
    + From<[T; N]>
    + Into<[T; N]>
    + PartialEq
    + Default
    + std::fmt::Debug
    + AlignTo<Elem = T>,
  T: Copy
    + Clone
    + PartialEq
    + Default
    + std::fmt::Debug
    + GenSample
    + PartialOrd,
{
  /// tests the traits of integer SIMD types
  fn test_basic_traits_int()
  where
    V: std::ops::Add<Output = V>
      + std::ops::Sub<Output = V>
      + std::ops::BitXor<Output = V>
      + std::ops::BitOr<Output = V>
      + std::ops::BitAnd<Output = V>
      + std::ops::Not<Output = V>
      + std::ops::Neg<Output = V>
      + wide::CmpEq<Output = V>,
    T: Copy
      + std::ops::BitXor<Output = T>
      + std::ops::BitOr<Output = T>
      + std::ops::BitAnd<Output = T>
      + std::ops::Not<Output = T>,
    Wrapping<T>: std::ops::Add<Output = Wrapping<T>>
      + std::ops::Sub<Output = Wrapping<T>>
      + std::ops::Neg<Output = Wrapping<T>>,
  {
    // test add
    test_random_vector_vs_scalar(
      |a: V, b| a + b,
      |a, b| (Wrapping::<T>(a) + Wrapping::<T>(b)).0,
    );

    // test sub
    test_random_vector_vs_scalar(
      |a: V, b| a - b,
      |a, b| (Wrapping::<T>(a) - Wrapping::<T>(b)).0,
    );

    // test neg
    test_random_vector_vs_scalar(|a: V, _b| -a, |a, _b| (-Wrapping::<T>(a)).0);

    test_random_vector_vs_scalar(|a: V, b| a ^ b, |a, b| a ^ b);

    test_random_vector_vs_scalar(|a: V, b| a & b, |a, b| a & b);

    test_random_vector_vs_scalar(|a: V, b| a | b, |a, b| a | b);

    // test not
    let a = V::from([T::default(); N]);
    let b = V::from([!T::default(); N]);

    assert!(a != b);
    assert!(a == a);
    assert!(b == a.not());
  }

  fn test_wrapping_mul_for_int()
  where
    V: std::ops::Mul<Output = V>,
    T: std::ops::Mul<Output = T>,
    Wrapping<T>: std::ops::Mul<Output = Wrapping<T>>,
  {
    // test mul
    test_random_vector_vs_scalar(
      |a: V, b| a * b,
      |a, b| (Wrapping::<T>(a) * Wrapping::<T>(b)).0,
    );
  }

  fn test_shl_shr()
  where
    V: std::ops::Shl<u32, Output = V> + std::ops::Shr<u32, Output = V>,
    T: std::ops::Shl<u32, Output = T> + std::ops::Shr<u32, Output = T>,
  {
    // test shl
    test_random_vector_vs_scalar(|a: V, _b| a << 3, |a, _b| a << 3);

    // test shr
    test_random_vector_vs_scalar(|a: V, _b| a >> 3, |a, _b| a >> 3);
  }

  /// tests the basic traits according to floating point operations
  fn test_basic_traits_float()
  where
    V: std::ops::Add<Output = V>
      + std::ops::Sub<Output = V>
      + std::ops::Mul<Output = V>
      + std::ops::Div<Output = V>
      + std::ops::Neg<Output = V>
      + wide::CmpEq<Output = V>,
    T: std::ops::Add<Output = T>
      + std::ops::Sub<Output = T>
      + std::ops::Neg<Output = T>
      + std::ops::Mul<Output = T>
      + std::ops::Div<Output = T>,
  {
    // test add
    test_random_vector_vs_scalar(|a: V, b| a + b, |a, b| a + b);

    // test sub
    test_random_vector_vs_scalar(|a: V, b| a - b, |a, b| a - b);

    // test mul
    test_random_vector_vs_scalar(|a: V, b| a * b, |a, b| a * b);

    // test div (rust floating point does not panic on div by zero)
    test_random_vector_vs_scalar(|a: V, b| a / b, |a, b| a / b);

    // test neg
    test_random_vector_vs_scalar(|a: V, _b| -a, |a, _b| -a);
  }

  /// tests the traits of SIMD comparison operations
  fn test_basic_traits_simd_cmp()
  where
    V: wide::CmpGt<Output = V>
      + wide::CmpLt<Output = V>
      + wide::CmpEq<Output = V>,
  {
    test_random_vector_vs_scalar(
      |a: V, b| a.simd_eq(b),
      |a, b| if a == b { T::NOT } else { T::default() },
    );

    // test gt
    test_random_vector_vs_scalar(
      |a: V, b| a.simd_gt(b),
      |a, b| if a > b { T::NOT } else { T::default() },
    );

    test_random_vector_vs_scalar(
      |a: V, b| a.simd_lt(b),
      |a, b| if a < b { T::NOT } else { T::default() },
    );
  }

  /// tests the traits of SIMD comparison operations ge and le
  fn test_basic_traits_simd_cmp_ge_le()
  where
    V: wide::CmpGe<Output = V> + wide::CmpLe<Output = V>,
  {
    // test greater or equal
    test_random_vector_vs_scalar(
      |a: V, b| a.simd_ge(b),
      |a, b| if a >= b { T::NOT } else { T::default() },
    );

    // test less than or equal
    test_random_vector_vs_scalar(
      |a: V, b| a.simd_le(b),
      |a, b| if a <= b { T::NOT } else { T::default() },
    );
  }

  fn test_basic_traits_aligned_to() {
    // test AlignTo
    let mut rng = 0x123456789abcdef0;
    let mut my_slice = [T::default(); 57];
    for i in 0..my_slice.len() {
      my_slice[i] = gen_random(&mut rng);
    }

    for i in 0..57 {
      let (head, body, tail) = V::simd_align_to(&my_slice[i..]);

      assert_eq!(head.len() + body.len() * N + tail.len(), my_slice.len() - i);

      for j in 0..head.len() {
        assert!(head[j] == my_slice[i + j]);
      }
      for j in 0..body.len() {
        let vec_arr: [T; N] = body[j].into();
        for k in 0..N {
          assert!(vec_arr[k] == my_slice[i + head.len() + j * N + k]);
        }
      }

      for j in 0..tail.len() {
        assert!(tail[j] == my_slice[i + head.len() + body.len() * N + j]);
      }

      // assert that mutable version returned the same thing
      let h = head.to_vec();
      let b = body.to_vec();
      let t = tail.to_vec();

      let (head_mut, body_mut, tail_mut) =
        V::simd_align_to_mut(&mut my_slice[i..]);
      assert_eq!(head_mut, h);
      assert_eq!(body_mut, b);
      assert_eq!(tail_mut, t);
    }
  }
}

/// implement blanket trait to allow calling test functions as associated
/// functions
impl<V, T, const N: usize> TestBasicTraits<V, T, N> for V
where
  V: Copy
    + From<[T; N]>
    + Into<[T; N]>
    + Default
    + std::fmt::Debug
    + PartialEq
    + AlignTo<Elem = T>,
  T: Copy + Default + std::fmt::Debug + GenSample + PartialEq + PartialOrd,
{
}

/// trait to reduce a 64 bit pseudo-random number to a random sample value
trait GenSample
where
  Self: PartialEq + Copy + Default,
{
  const NOT: Self;

  fn get_sample(v: u64) -> Self;
  fn binary_eq(self, b: Self) -> bool {
    self == b
  }
}

impl GenSample for u64 {
  const NOT: Self = u64::MAX;
  fn get_sample(v: u64) -> Self {
    v
  }
}

impl GenSample for u32 {
  const NOT: Self = u32::MAX;
  fn get_sample(v: u64) -> Self {
    v as u32
  }
}

impl GenSample for u16 {
  const NOT: Self = u16::MAX;
  fn get_sample(v: u64) -> Self {
    v as u16
  }
}

impl GenSample for u8 {
  const NOT: Self = u8::MAX;
  fn get_sample(v: u64) -> Self {
    v as u8
  }
}

impl GenSample for i64 {
  const NOT: Self = -1;

  fn get_sample(v: u64) -> Self {
    v as i64
  }
}

impl GenSample for i32 {
  const NOT: Self = -1;
  fn get_sample(v: u64) -> Self {
    v as i32
  }
}

impl GenSample for i16 {
  const NOT: Self = -1;
  fn get_sample(v: u64) -> Self {
    v as i16
  }
}

impl GenSample for i8 {
  const NOT: Self = -1;
  fn get_sample(v: u64) -> Self {
    v as i8
  }
}

impl GenSample for f32 {
  const NOT: Self = f32::from_bits(u32::MAX);
  fn get_sample(v: u64) -> Self {
    // generate special float values more often than random
    // chance to test edge cases
    let m = (v >> 8) & 15;

    match m {
      1 => f32::NAN,
      2 => f32::INFINITY,
      3 => f32::NEG_INFINITY,
      _ => ((v as i64) as f32) / 7.0,
    }
  }

  /// floating points Nan always fails equality so we need to special case it
  fn binary_eq(self, b: Self) -> bool {
    const MAX_REL_DIFF: f32 = 0.000001;

    if self == b {
      return true;
    } else if self.is_nan() {
      b.is_nan()
    } else if self.is_infinite() {
      b.is_infinite() && self.is_sign_positive() == b.is_sign_positive()
    } else if (self - b).abs() < MAX_REL_DIFF {
      // return true if the difference is very small in absolute terms
      return true;
    } else {
      // the error could be large in absolute terms, but small in relative terms
      // if both numbers are large
      let denominator = self.abs().max(b.abs());

      // one or both are zero, but not equal
      if denominator == 0.0 {
        return false;
      }

      (self - b).abs() / denominator < MAX_REL_DIFF
    }
  }
}

impl GenSample for f64 {
  const NOT: Self = f64::from_bits(u64::MAX);

  // generate special float values more often than random
  // chance to test edge cases
  fn get_sample(v: u64) -> Self {
    let m = (v >> 8) & 15;

    match m {
      1 => f64::NAN,
      2 => f64::INFINITY,
      3 => f64::NEG_INFINITY,
      _ => ((v as i64) as f64) / 7.0,
    }
  }

  /// floating points Nan always fails equality so we need to special case it
  fn binary_eq(self, b: Self) -> bool {
    const MAX_REL_DIFF: f64 = 0.000001;

    if self == b {
      return true;
    } else if self.is_nan() {
      b.is_nan()
    } else if self.is_infinite() {
      b.is_infinite() && self.is_sign_positive() == b.is_sign_positive()
    } else if (self - b).abs() < MAX_REL_DIFF {
      // return true if the difference is very small in absolute terms
      return true;
    } else {
      // the error could be large in absolute terms, but small in relative terms
      // if both numbers are large
      let denominator = self.abs().max(b.abs());

      // one or both are zero, but not equal
      if denominator == 0.0 {
        return false;
      }

      (self - b).abs() / denominator < MAX_REL_DIFF
    }
  }
}

/// defines tests per type of SIMD vector. This allows us to easily generate
/// the same tests for multiple types without copy/pasting code.
#[macro_export]
macro_rules! generate_basic_traits_test {
  ($simd_type:ident, $elem_type:ident) => {
    #[test]
    fn basic_traits() {
      use crate::TestBasicTraits;

      crate::basic_traits_tests_for!($elem_type, $simd_type);
    }
  };
}

#[macro_export]
macro_rules! basic_traits_tests_for {
  (f32, $T:ident) => {
    $T::test_basic_traits_float();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_simd_cmp_ge_le();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
    crate::test_random_vector_vs_scalar(|a: $T, _| a.round(), |a, _| a.round());
  };

  (f64, $T:ident) => {
    $T::test_basic_traits_float();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_simd_cmp_ge_le();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
    crate::test_random_vector_vs_scalar(|a: $T, _| a.round(), |a, _| a.round());
    crate::test_random_vector_vs_scalar(|a: $T, _| a.floor(), |a, _| a.floor());
    crate::test_random_vector_vs_scalar(|a: $T, _| a.ceil(), |a, _| a.ceil());
  };

  (i8, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
  };

  (u8, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
  };

  (i16, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_wrapping_mul_for_int();
    $T::test_shl_shr();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, _b| a.abs(), |a, _b| a.abs());

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));

    crate::test_random_vector_vs_scalar(
      |a: $T, b| a.saturating_add(b),
      |a, b| a.saturating_add(b),
    );
    crate::test_random_vector_vs_scalar(
      |a: $T, b| a.saturating_sub(b),
      |a, b| a.saturating_sub(b),
    );
  };

  (u16, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_wrapping_mul_for_int();
    $T::test_shl_shr();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
    crate::test_random_vector_vs_scalar(
      |a: $T, b| a.saturating_add(b),
      |a, b| a.saturating_add(b),
    );
    crate::test_random_vector_vs_scalar(
      |a: $T, b| a.saturating_sub(b),
      |a, b| a.saturating_sub(b),
    );
  };

  (i32, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_wrapping_mul_for_int();
    $T::test_shl_shr();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
  };

  (u32, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_wrapping_mul_for_int();
    $T::test_shl_shr();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_aligned_to();

    crate::test_random_vector_vs_scalar(|a: $T, b| a.max(b), |a, b| a.max(b));
    crate::test_random_vector_vs_scalar(|a: $T, b| a.min(b), |a, b| a.min(b));
  };

  (i64, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_wrapping_mul_for_int();
    $T::test_shl_shr();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_aligned_to();
  };

  (u64, $T:ident) => {
    $T::test_basic_traits_int();
    $T::test_wrapping_mul_for_int();
    $T::test_shl_shr();
    $T::test_basic_traits_simd_cmp();
    $T::test_basic_traits_aligned_to();
  };

  ($other:ident, $T:ident) => {
    compile_error!(concat!("Unsupported element type: ", stringify!($other)));
  };
}
