use std::num::Wrapping;
use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i64x4>(), 32);
  assert_eq!(core::mem::align_of::<i64x4>(), 32);
}

#[test]
fn impl_add_for_i64x4() {
  let a = i64x4::from([i64::MAX - 1, i64::MAX - 1, 6, 9]);
  let b = i64x4::from([1, 2, 3, 4]);
  let expected = i64x4::from([i64::MAX, i64::MIN, 9, 13]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i64x4() {
  let a = i64x4::from([1, 0, 9, 12]);
  let b = i64x4::from([1, 1, 3, 3]);
  let expected = i64x4::from([0, -1, 6, 9]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i64x4() {
  let a = i64x4::from([i64::MIN + 1, 24, 5402, i64::MAX]);
  let b = i64x4::from([1, -26, -5402, 2]);
  let expected = i64x4::from([
    i64::MIN + 1,
    24 * -26,
    5402 * -5402,
    (Wrapping(i64::MAX) * Wrapping(2)).0,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i64x4() {
  let a = i64x4::from([1, 1, 0, 0]);
  let b = i64x4::from([0, 1, 0, 1]);
  let expected = i64x4::from([0, 1, 0, 0]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i64x4() {
  let a = i64x4::from([1, 1, 0, 0]);
  let b = i64x4::from([0, 1, 0, 1]);
  let expected = i64x4::from([1, 1, 0, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i64x4() {
  let a = i64x4::from([1, 1, 1, 0]);
  let b = i64x4::from([0, 1, 0, 1]);
  let expected = i64x4::from([1, 0, 1, 1]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i64x4() {
  let a = i64x4::from([i64::MAX - 1, i64::MAX - 1, 65535, 0]);
  let b = 2;
  let expected =
    i64x4::from([(i64::MAX - 1) << 2, (i64::MAX - 1) << 2, 65535 << 2, 0 << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i64x4() {
  let a = i64x4::from([i64::MAX - 1, i64::MAX - 1, 65535, 0]);
  let b = 2;
  let expected =
    i64x4::from([(i64::MAX - 1) >> 2, (i64::MAX - 1) >> 2, 65535 >> 2, 0 >> 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x4_blend() {
  let use_t: i64 = i64::MAX;
  let t = i64x4::from([1, 2, 3, 4]);
  let f = i64x4::from([17, 18, 21, 45]);
  let mask = i64x4::from([use_t, 0, 0, use_t]);
  let expected = i64x4::from([1, 18, 21, 4]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x4_cmp_eq() {
  let a = i64x4::from([1_i64, 4, i64::MAX, 5]);
  let b = i64x4::from([3_i64, 4, i64::MAX, 1]);
  let expected = i64x4::from([0, -1, -1, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}
