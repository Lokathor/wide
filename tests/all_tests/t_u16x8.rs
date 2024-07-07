use std::num::Wrapping;
use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u16x8>(), 16);
  assert_eq!(core::mem::align_of::<u16x8>(), 16);
}

#[test]
fn impl_add_for_u16x8() {
  let a = u16x8::from([1, 2, 3, 4, 5, 6, u16::MAX - 1, u16::MAX - 1]);
  let b = u16x8::from([17, 18, 19, 20, 21, 22, 1, 2]);
  let expected = u16x8::from([18, 20, 22, 24, 26, 28, u16::MAX, 0]);
  let actual = a + b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u16x8, b| a + b,
    |a, b| a.wrapping_add(b),
  );
}

#[test]
fn impl_sub_for_u16x8() {
  let a = u16x8::from([1468, 220, 3, 4456, 5, 6897, 1, 0]);
  let b = u16x8::from([17, 180, 192, 200, 121, 22, 1, 1]);
  let expected = u16x8::from([1451, 40, 65347, 4256, 65420, 6875, 0, u16::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u16x8, b| a - b,
    |a, b| a.wrapping_sub(b),
  );
}

#[test]
fn impl_saturating_add_for_u16x8() {
  let a = u16x8::from([1, 2, 3, 4, 5, 6, u16::MAX - 1, u16::MAX - 1]);
  let b = u16x8::from([17, 18, 19, 20, 21, 22, 1, 2]);
  let expected = u16x8::from([18, 20, 22, 24, 26, 28, u16::MAX, u16::MAX]);
  let actual = a.saturating_add(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u16x8, b| a.saturating_add(b),
    |a, b| a.saturating_add(b),
  );
}

#[test]
fn impl_saturating_sub_for_u16x8() {
  let a = u16x8::from([1468, 220, 3, 4456, 5, 6897, 1, 0]);
  let b = u16x8::from([17, 180, 192, 200, 121, 22, 1, 1]);
  let expected = u16x8::from([1451, 40, 0, 4256, 0, 6875, 0, 0]);
  let actual = a.saturating_sub(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u16x8, b| a.saturating_sub(b),
    |a, b| a.saturating_sub(b),
  );
}

#[test]
fn impl_mul_for_u16x8() {
  let a = u16x8::from([1, 2, u16::MAX, 4, 5, 6, u16::MIN + 1, u16::MIN]);
  let b = u16x8::from([17, 18, 190, 20, 21, 22, 1, 1]);
  let expected = u16x8::from([
    17,
    36,
    (Wrapping(u16::MAX) * Wrapping(190)).0,
    80,
    105,
    132,
    u16::MIN + 1,
    u16::MIN,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u16x8, b| a * b,
    |a, b| a.wrapping_mul(b),
  );
}

#[test]
fn impl_bitand_for_u8x16() {
  let a = u8x16::from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
  let b = u8x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = u8x16::from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, b| a & b, |a, b| a & b);
}

#[test]
fn impl_bitor_for_u8x16() {
  let a = u8x16::from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
  let b = u8x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = u8x16::from([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, b| a | b, |a, b| a | b);
}

#[test]
fn impl_bitxor_for_u8x16() {
  let a = u8x16::from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
  let b = u8x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = u8x16::from([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, b| a ^ b, |a, b| a ^ b);
}

#[test]
fn impl_shl_for_u16x8() {
  let a = u16x8::from([1, 2, 3, 4, 5, 6, u16::MAX - 1, u16::MAX - 1]);
  let b = 2;
  let expected = u16x8::from([
    1 << 2,
    2 << 2,
    3 << 2,
    4 << 2,
    5 << 2,
    6 << 2,
    (u16::MAX - 1) << 2,
    (u16::MAX - 1) << 2,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, _b| a << 3, |a, _b| a << 3);
}

#[test]
fn impl_shr_for_u16x8() {
  let a = u16x8::from([1, 2, 3, 4, 5, 6, u16::MAX - 1, u16::MAX - 1]);
  let b = 2;
  let expected = u16x8::from([
    1 >> 2,
    2 >> 2,
    3 >> 2,
    4 >> 2,
    5 >> 2,
    6 >> 2,
    (u16::MAX - 1) >> 2,
    (u16::MAX - 1) >> 2,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, _b| a >> 3, |a, _b| a >> 3);
}

#[test]
fn impl_u16x8_cmp_eq() {
  let a = u16x8::from([1, 2, 3, 4, 1, 2, 3, 4]);
  let b = u16x8::from([2_u16; 8]);
  let expected = u16x8::from([0, u16::MAX, 0, 0, 0, u16::MAX, 0, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u16x8_blend() {
  let use_t: u16 = u16::MAX;
  let t = u16x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let f = u16x8::from([17, 18, 19, 20, 21, 22, 23, 24]);
  let mask = u16x8::from([use_t, 0, use_t, 0, use_t, 0, use_t, 0]);
  let expected = u16x8::from([1, 18, 3, 20, 5, 22, 7, 24]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u16x8_max() {
  let a = u16x8::from([1, 37001, 3, 4, 5, 6, 7, 8]);
  let b = u16x8::from([37000, 37000, 19, 20, 2, 2, 2, 24]);
  let expected = u16x8::from([37000, 37001, 19, 20, 5, 6, 7, 24]);
  let actual = a.max(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, b| a.max(b), |a, b| a.max(b));
}

#[test]
fn impl_u16x8_min() {
  let a = u16x8::from([1, 37001, 3, 4, 5, 6, 7, 8]);
  let b = u16x8::from([37000, 37000, 19, 20, 2, 2, 2, 24]);
  let expected = u16x8::from([1, 37000, 3, 4, 2, 2, 2, 8]);
  let actual = a.min(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u16x8, b| a.min(b), |a, b| a.min(b));
}

#[test]
fn impl_u16x8_from_u8x16_low() {
  let a =
    u8x16::from([255, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255, 128]);
  let expected = u16x8::from([255, 2, 3, 4, 5, 6, 7, 8]);
  let actual = u16x8::from_u8x16_low(a);
  assert_eq!(expected, actual);
}
#[test]
fn impl_u16x8_from_u8x16_high() {
  let a =
    u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255, 128]);
  let expected = u16x8::from([9, 10, 11, 12, 13, 14, 255, 128]);
  let actual = u16x8::from_u8x16_high(a);
  assert_eq!(expected, actual);
}
