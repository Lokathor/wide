use std::num::Wrapping;
use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u32x4>(), 16);
  assert_eq!(core::mem::align_of::<u32x4>(), 16);
}

#[test]
fn impl_add_for_u32x4() {
  let a = u32x4::from([1, 2, u32::MAX - 1, u32::MAX - 1]);
  let b = u32x4::from([17, 18, 1, 2]);
  let expected = u32x4::from([18, 20, u32::MAX, u32::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u32x4() {
  let a = u32x4::from([9001, 2, 1, 0]);
  let b = u32x4::from([17, 18, 1, 1]);
  let expected = u32x4::from([8984, 4294967280, 0, u32::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_u32x4() {
  let a = u32x4::from([1, 2, u32::MIN + 1, u32::MAX]);
  let b = u32x4::from([17, 18, 1, 32]);
  let expected =
    u32x4::from([17, 36, 1, (Wrapping(u32::MAX) * Wrapping(32)).0]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_u32x4() {
  let a = u32x4::from([0, 0, 1, 1]);
  let b = u32x4::from([0, 1, 0, 1]);
  let expected = u32x4::from([0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u32x4() {
  let a = u32x4::from([0, 0, 1, 1]);
  let b = u32x4::from([0, 1, 0, 1]);
  let expected = u32x4::from([0, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u32x4() {
  let a = u32x4::from([0, 0, 1, 1]);
  let b = u32x4::from([0, 1, 0, 1]);
  let expected = u32x4::from([0, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_u32x4() {
  let a = u32x4::from([1, 2, u32::MAX - 1, u32::MAX - 1]);
  let b = 2;
  let expected =
    u32x4::from([1 << 2, 2 << 2, (u32::MAX - 1) << 2, (u32::MAX - 1) << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_u32x4() {
  let a = u32x4::from([1, 2, u32::MAX - 1, u32::MAX - 1]);
  let b = 2;
  let expected =
    u32x4::from([1 >> 2, 2 >> 2, (u32::MAX - 1) >> 2, (u32::MAX - 1) >> 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x4_cmp_eq() {
  let a = u32x4::from([1, 2, 3, 4]);
  let b = u32x4::from([2_u32; 4]);
  let expected = u32x4::from([0, u32::MAX, 0, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x4_cmp_gt() {
  let a = u32x4::from([1, 2, 3, 4]);
  let b = u32x4::from([2_u32; 4]);
  let expected = u32x4::from([0, 0, u32::MAX, u32::MAX]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x4_cmp_lt() {
  let a = u32x4::from([1, 2, 3, 4]);
  let b = u32x4::from([2_u32; 4]);
  let expected = u32x4::from([u32::MAX, 0, 0, 0]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);

  let expected = u32x4::from([0, 0, 0, 0]);
  let actual = a.cmp_lt(a);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x4_blend() {
  let use_t: u32 = u32::MAX;
  let t = u32x4::from([1, 2, 3, 4]);
  let f = u32x4::from([17, 18, 19, 20]);
  let mask = u32x4::from([use_t, 0, use_t, 0]);
  let expected = u32x4::from([1, 18, 3, 20]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x4_max() {
  let a = u32x4::from([0, 2, 3, 4]);
  let b = u32x4::from([17, 1, 0, 20]);
  let expected = u32x4::from([17, 2, 3, 20]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x4_min() {
  let a = u32x4::from([0, 2, 3, 4]);
  let b = u32x4::from([17, 1, 0, 20]);
  let expected = u32x4::from([0, 1, 0, 4]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}
