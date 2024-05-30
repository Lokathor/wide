use wide::SimdType;

/// Performs a binary operation both in scalar and vector form and compares the
/// results. This makes it less error prone to test the binary operations and
/// makes it easier to add new tests.
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
  let expected = T::from_fn(|i| fn_scalar(a.as_array()[i], b.as_array()[i]));

  let actual = fn_vector(a, b);

  // assert equality for manually calculated result
  assert_eq!(expected, actual, "scalar={:?} vector={:?}", expected, actual);
}

pub fn test_unary_op<
  T: SimdType<V, N> + PartialEq + std::fmt::Debug + Copy,
  V: Copy + PartialEq + std::fmt::Debug,
  FnVector: Fn(T) -> T,
  FnScalar: Fn(V) -> V,
  const N: usize,
>(
  a: T,
  fn_scalar: FnScalar,
  fn_vector: FnVector,
) {
  let expected = T::from_fn(|i| fn_scalar(a.as_array()[i]));
  // ensure that the elements got put in the right place
  for i in 0..N {
    assert_eq!(expected.as_array()[i], fn_scalar(a.as_array()[i]));
  }

  let actual = fn_vector(a);

  // assert equality for manually calculated result
  assert_eq!(expected, actual, "scalar={:?} vector={:?}", expected, actual);
}
