use std::num::Wrapping;
use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u64x2>(), 16);
  assert_eq!(core::mem::align_of::<u64x2>(), 16);
}

crate::generate_basic_traits_test!(u64x2, u64);

#[test]
fn impl_add_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = u64x2::from([1, 2]);
  let expected = u64x2::from([u64::MAX, u64::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u64x2() {
  let a = u64x2::from([1, 0]);
  let b = u64x2::from([1, 1]);
  let expected = u64x2::from([0, u64::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_u64x2() {
  let a = u64x2::from([u64::MIN + 1, u64::MAX]);
  let b = u64x2::from([2, 2]);
  let expected = u64x2::from([2, (Wrapping(u64::MAX) * Wrapping(2)).0]);
  let actual = a * b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u64x2, b| a * b,
    |a, b| a.wrapping_mul(b),
  );
}

#[test]
fn impl_bitand_for_u64x2() {
  let a = u64x2::from([1, 1]);
  let b = u64x2::from([0, 1]);
  let expected = u64x2::from([0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u64x2() {
  let a = u64x2::from([1, 1]);
  let b = u64x2::from([0, 1]);
  let expected = u64x2::from([1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u64x2() {
  let a = u64x2::from([1, 1]);
  let b = u64x2::from([0, 1]);
  let expected = u64x2::from([1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_each_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX]);
  let shift = u64x2::from([2, 65 /* test masking behavior */]);
  let expected = u64x2::from([(u64::MAX - 1) << 2, u64::MAX << 1]);
  let actual = a << shift;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u64x2, b| a << b,
    |a, b| a.wrapping_shl(b as u32),
  );
}

#[test]
fn impl_shl_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = 2;
  let expected = u64x2::from([(u64::MAX - 1) << 2, (u64::MAX - 1) << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_each_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX]);
  let shift = u64x2::from([2, 65 /* test masking behavior */]);
  let expected = u64x2::from([(u64::MAX - 1) >> 2, u64::MAX >> 1]);
  let actual = a >> shift;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u64x2, b| a >> b,
    |a, b| a.wrapping_shr(b as u32),
  );
}

#[test]
fn impl_shr_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = 2;
  let expected = u64x2::from([(u64::MAX - 1) >> 2, (u64::MAX - 1) >> 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x2_blend() {
  let use_t: u64 = u64::MAX;
  let t = u64x2::from([1, 2]);
  let f = u64x2::from([17, 18]);
  let mask = u64x2::from([use_t, 0]);
  let expected = u64x2::from([1, 18]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x2_cmp_eq() {
  let a = u64x2::from([1_u64, 4]);
  let b = u64x2::from([3_u64, 4]);
  let expected = u64x2::from([0, u64::MAX]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x2_cmp_gt() {
  let a = u64x2::from([1_u64, 4]);
  let b = u64x2::from([3_u64, 4]);
  let expected = u64x2::from([0, 0]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u64x2, b| a.simd_gt(b),
    |a, b| if a > b { u64::MAX } else { 0 },
  );
}

#[test]
fn impl_u64x2_cmp_lt() {
  let a = u64x2::from([3_u64, 4]);
  let b = u64x2::from([1_u64, 4]);
  let expected = u64x2::from([0, 0]);
  let actual = a.simd_lt(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u64x2, b| a.simd_lt(b),
    |a, b| if a < b { u64::MAX } else { 0 },
  );
}

#[cfg(feature = "serde")]
#[test]
fn impl_u64x2_ser_de_roundtrip() {
  let serialized =
    bincode::serialize(&u64x2::ZERO).expect("serialization failed");
  let deserialized =
    bincode::deserialize(&serialized).expect("deserializaion failed");
  assert_eq!(u64x2::ZERO, deserialized);
}
