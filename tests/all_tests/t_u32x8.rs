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
}

#[test]
fn impl_bitand_for_u32x8() {
  let a = u32x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x8::from([0, 1, 0, 1, 0, 1, 1, 1]);

  crate::t_common::test_binary_op(a, b, |a, b| a & b, |a, b| a & b);
}

#[test]
fn impl_bitor_for_u32x8() {
  let a = u32x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x8::from([0, 1, 0, 1, 0, 1, 1, 1]);

  crate::t_common::test_binary_op(a, b, |a, b| a | b, |a, b| a | b);
}

#[test]
fn impl_bitxor_for_u32x8() {
  let a = u32x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x8::from([0, 1, 0, 1, 0, 1, 1, 1]);

  crate::t_common::test_binary_op(a, b, |a, b| a ^ b, |a, b| a ^ b);
}

#[test]
fn impl_shl_for_u32x8() {
  let a =
    u32x8::from([1, 2, u32::MAX - 1, i32::MAX as u32 - 1, 128, 255, 590, 5667]);
  let b = 2;

  crate::t_common::test_unary_op(a, |a| a << b, |a| a << b);
}

#[test]
fn impl_shr_for_u32x8() {
  let a =
    u32x8::from([1, 2, u32::MAX - 1, i32::MAX as u32 - 1, 128, 255, 590, 5667]);
  let b = 2;

  crate::t_common::test_unary_op(a, |a| a >> b, |a| a >> b);
}

#[test]
fn impl_u32x8_cmp_eq() {
  let a = u32x8::from([1, 2, 3, 4, 2, 1, 8, 2]);
  let b = u32x8::from([2_u32; 8]);

  crate::t_common::test_binary_op(
    a,
    b,
    |a, b| if a == b { u32::MAX } else { 0 },
    |a, b| a.cmp_eq(b),
  );
}

#[test]
fn impl_u32x8_cmp_gt() {
  let a = u32x8::from([1, 2, 9, 4, 1, 2, 8, 10]);
  let b = u32x8::from([5_u32; 8]);

  crate::t_common::test_binary_op(
    a,
    b,
    |a, b| if a > b { u32::MAX } else { 0 },
    |a, b| a.cmp_gt(b),
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
  let a = u32x8::from([1, 2, u32::MAX, i32::MAX as u32, 6, 8, 12, 9]);
  let b = u32x8::from([17, 18, 1, 1, 19, 0, 1, u32::MAX]);

  crate::t_common::test_binary_op(a, b, |a, b| a.max(b), |a, b| a.max(b));
}

#[test]
fn impl_u32x8_min() {
  let a = u32x8::from([1, 2, u32::MAX, i32::MAX as u32, 6, 8, 12, 9]);
  let b = u32x8::from([17, 18, 1, 1, 19, 0, 1, u32::MAX]);

  crate::t_common::test_binary_op(a, b, |a, b| a.min(b), |a, b| a.min(b));
}

#[test]
fn impl_u32x8_shr_all() {
  let a = u32x8::from([15313, 52322, u32::MAX, 4, 1322, 5, 2552352, 2123]);
  let shift =
    u32x8::from([1, 2, 3, 4, 5, 6, 33 /* test masking behavior */, 31]);

  crate::t_common::test_binary_op(
    a,
    shift,
    |a, b| a.wrapping_shr(b),
    |a, b| a >> b,
  );
}

#[test]
fn impl_u32x8_shl_all() {
  let a = u32x8::from([15313, 52322, u32::MAX, 4, 1322, 5, 2552352, 2123]);
  let shift =
    u32x8::from([1, 2, 3, 4, 5, 6, 33 /* test masking behavior */, 31]);

  crate::t_common::test_binary_op(
    a,
    shift,
    |a, b| a.wrapping_shl(b),
    |a, b| a << b,
  );
}

#[test]
fn impl_u32x8_not() {
  let a = u32x8::from([15313, 52322, u32::MAX, 4, 1322, 5, 2552352, 2123]);

  crate::t_common::test_unary_op(a, |a| !a, |a| !a);
}
