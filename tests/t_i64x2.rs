use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i64x2>(), 16);
  assert_eq!(core::mem::align_of::<i64x2>(), 16);
}

#[test]
fn impl_add_for_i64x2() {
  let a = i64x2::from([i64::MAX - 1, i64::MAX - 1]);
  let b = i64x2::from([1, 2]);
  let expected = i64x2::from([i64::MAX, i64::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i64x2() {
  let a = i64x2::from([i64::MIN + 1, i64::MIN]);
  let b = i64x2::from([1, 1]);
  let expected = i64x2::from([i64::MIN, i64::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i64x2() {
  let a = i64x2::from([i64::MIN + 1, 24]);
  let b = i64x2::from([1, -26]);
  let expected = i64x2::from([i64::MIN + 1, 24 * -26]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i64x2() {
  let a = i64x2::from([1, 1]);
  let b = i64x2::from([0, 1]);
  let expected = i64x2::from([0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i64x2() {
  let a = i64x2::from([1, 1]);
  let b = i64x2::from([0, 1]);
  let expected = i64x2::from([1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i64x2() {
  let a = i64x2::from([1, 1]);
  let b = i64x2::from([0, 1]);
  let expected = i64x2::from([1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i64x2() {
  let a = i64x2::from([i64::MAX - 1, i64::MAX - 1]);
  let b = 2;
  let expected = i64x2::from([(i64::MAX - 1) << 2, (i64::MAX - 1) << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x2_blend() {
  let use_t: i64 = -1;
  let t = i64x2::from([1, 2]);
  let f = i64x2::from([17, 18]);
  let mask = i64x2::from([use_t, 0]);
  let expected = i64x2::from([1, 18]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x2_cmp_eq() {
  let a = i64x2::from([1_i64, 4]);
  let b = i64x2::from([3_i64, 4]);
  let expected = i64x2::from([0, -1]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x2_cmp_gt() {
  let a = i64x2::from([3_i64, 4]);
  let b = i64x2::from([1_i64, 4]);
  let expected = i64x2::from([-1, 0]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}
