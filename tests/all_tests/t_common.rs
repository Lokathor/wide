use wide::SimdType;

/// Performs a binary operation both in scalar and vector form and compares the results.
/// This makes it less error prone to test the binary operations and makes it easier to add new tests.
pub fn test_binary_op<
  T: SimdType<V, N> + Default + PartialEq + std::fmt::Debug + Copy,
  V: Copy,
  FnVector: Fn(T, T) -> T,
  FnScalar: Fn(V, V) -> V,
  const N: usize,
>(
  a: T,
  b: T,
  fn_scalar: FnScalar,
  fn_vector: FnVector,
) {
  let mut expected = T::default();
  for i in 0..N {
    expected.as_mut_array()[i] = fn_scalar(a.as_array()[i], b.as_array()[i]);
  }

  let actual = fn_vector(a, b);

  // assert equality for manually calculated result
  assert_eq!(expected, actual);

  // assert equality using the binary_op method as well
  assert_eq!(expected, a.binary_op(b, fn_scalar))
}

pub fn test_unary_op<
  T: SimdType<V, N> + Default + PartialEq + std::fmt::Debug + Copy,
  V: Copy,
  FnVector: Fn(T) -> T,
  FnScalar: Fn(V) -> V,
  const N: usize,
>(
  a: T,
  fn_scalar: FnScalar,
  fn_vector: FnVector,
) {
  let mut expected = T::default();
  for i in 0..N {
    expected.as_mut_array()[i] = fn_scalar(a.as_array()[i]);
  }

  let actual = fn_vector(a);

  // assert equality for manually calculated result
  assert_eq!(expected, actual);

  // assert equality using the unary_op method as well
  assert_eq!(expected, a.unary_op(fn_scalar))
}
