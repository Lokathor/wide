use std::num::Wrapping;
use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u64x2>(), 16);
  assert_eq!(core::mem::align_of::<u64x2>(), 16);
}

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
fn impl_shl_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = 2;
  let expected = u64x2::from([(u64::MAX - 1) << 2, (u64::MAX - 1) << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
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
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x2_cmp_gt() {
  let a = u64x2::from([1_u64, 4]);
  let b = u64x2::from([3_u64, 4]);
  let expected = u64x2::from([0, 0]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}
