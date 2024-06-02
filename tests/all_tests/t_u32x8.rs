use std::num::Wrapping;
use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u32x8>(), 32);
  assert_eq!(core::mem::align_of::<u32x8>(), 32);
}

#[test]
fn impl_add_for_u32x8() {
  let a = u32x8::from([1, 2, u32::MAX - 1, u32::MAX - 1, 31, 72, 13, 53]);
  let b = u32x8::from([17, 18, 1, 2, 12, 12, 634, 15]);
  let expected = u32x8::from([18, 20, u32::MAX, u32::MIN, 43, 84, 647, 68]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u32x8() {
  let a = u32x8::from([9001, 2, 1, 0, 12, 1, 9, 10]);
  let b = u32x8::from([17, 18, 1, 1, 15, 1, 2, 5]);
  let expected =
    u32x8::from([8984, 4294967280, 0, u32::MAX, 4294967293, 0, 7, 5]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_u32x8() {
  let a = u32x8::from([1, 2, u32::MIN + 1, u32::MAX, 123, u32::MIN, 9, 3802]);
  let b = u32x8::from([17, 18, 1, 32, 456, 4, 190, 100]);
  let expected = u32x8::from([
    17,
    36,
    1,
    (Wrapping(u32::MAX) * Wrapping(32)).0,
    123 * 456,
    0,
    190 * 9,
    380200,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x8, b| a * b,
    |a, b| a.wrapping_mul(b),
  );
}

#[test]
fn impl_bitand_for_u32x8() {
  let a = u32x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x8::from([0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u32x8::from([0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x8, b| a | b, |a, b| a | b);
}

#[test]
fn impl_bitor_for_u32x8() {
  let a = u32x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x8::from([0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u32x8::from([0, 1, 1, 1, 1, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u32x8() {
  let a = u32x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x8::from([0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u32x8::from([0, 1, 1, 0, 1, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_u32x8() {
  let a =
    u32x8::from([1, 2, u32::MAX - 1, i32::MAX as u32 - 1, 128, 255, 590, 5667]);
  let b = 2;
  let expected = u32x8::from([
    1 << 2,
    2 << 2,
    (u32::MAX - 1) << 2,
    (i32::MAX as u32 - 1) << 2,
    128 << 2,
    255 << 2,
    590 << 2,
    5667 << 2,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_u32x8() {
  let a =
    u32x8::from([1, 2, u32::MAX - 1, i32::MAX as u32 - 1, 128, 255, 590, 5667]);
  let b = 2;
  let expected = u32x8::from([
    1 >> 2,
    2 >> 2,
    (u32::MAX - 1) >> 2,
    (i32::MAX as u32 - 1) >> 2,
    128 >> 2,
    255 >> 2,
    590 >> 2,
    5667 >> 2,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x8_cmp_eq() {
  let a = u32x8::from([1, 2, 3, 4, 2, 1, 8, 2]);
  let b = u32x8::from([2_u32; 8]);
  let expected = u32x8::from([0, u32::MAX, 0, 0, u32::MAX, 0, 0, u32::MAX]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x8_cmp_gt() {
  let a = u32x8::from([1, 2, u32::MAX, 4, 1, 2, 8, 10]);
  let b = u32x8::from([5, 5, 5, 5, 5, 5, 5, 5]);
  let expected = u32x8::from([0, 0, u32::MAX, 0, 0, 0, u32::MAX, u32::MAX]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x8, b| a.cmp_gt(b),
    |a, b| if a > b { u32::MAX } else { 0 },
  );
}

#[test]
fn impl_u32x8_cmp_lt() {
  let a = u32x8::from([5, 5, 5, 5, 5, 5, 5, 5]);
  let b = u32x8::from([1, 2, u32::MAX, 4, 1, 2, 8, 10]);
  let expected = u32x8::from([0, 0, u32::MAX, 0, 0, 0, u32::MAX, u32::MAX]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x8, b| a.cmp_lt(b),
    |a, b| if a < b { u32::MAX } else { 0 },
  );
}

#[test]
fn impl_u32x8_blend() {
  let use_t: u32 = u32::MAX;
  let t = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let f = u32x8::from([17, 18, 19, 20, 25, 30, 50, 90]);
  let mask = u32x8::from([use_t, 0, use_t, 0, 0, 0, 0, use_t]);
  let expected = u32x8::from([1, 18, 3, 20, 25, 30, 50, 8]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x8_max() {
  let a = u32x8::from([1, 2, 1, 0, 6, 0, 12, u32::MAX]);
  let b = u32x8::from([17, 0, 1, 1, 19, 0, 0, 1000]);
  let expected = u32x8::from([17, 2, 1, 1, 19, 0, 12, u32::MAX]);
  let actual = a.max(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x8, b| a.max(b), |a, b| a.max(b));
}

#[test]
fn impl_u32x8_min() {
  let a = u32x8::from([1, 2, 1, 0, 6, 0, 12, u32::MAX]);
  let b = u32x8::from([17, 0, 1, 1, 19, 0, 0, 1000]);
  let expected = u32x8::from([1, 0, 1, 0, 6, 0, 0, 1000]);
  let actual = a.min(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x8, b| a.min(b), |a, b| a.min(b));
}
