use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i16x8>(), 16);
  assert_eq!(core::mem::align_of::<i16x8>(), 16);
}

#[test]
fn impl_add_for_i16x8() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MAX - 1, i16::MAX - 1]);
  let b = i16x8::from([17, 18, 19, 20, 21, 22, 1, 2]);
  let expected = i16x8::from([18, 20, 22, 24, 26, 28, i16::MAX, i16::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i16x8() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
  let b = i16x8::from([17, -18, 190, -20, 21, -22, 1, 1]);
  let expected = i16x8::from([-16, 20, -187, 24, -16, 28, i16::MIN, i16::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_add_saturating_for_i16x8() {
  let a = i16x8::from([i16::MAX, i16::MIN, 3, 4, -1, -2, -3, -4]);
  let b = i16x8::from([i16::MAX, i16::MIN, 7, 8, -15, -26, -37, 48]);
  let expected = i16x8::from([i16::MAX, i16::MIN, 10, 12, -16, -28, -40, 44]);
  let actual = a.saturating_add(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_scale_i16x8() {
  let a = i16x8::from([100, 200, 300, 400, 500, -600, 700, -800]);
  let b = i16x8::from([900, 1000, 1100, 1200, 1300, -1400, -1500, 1600]);
  let actual = a.mul_scale_round(b);
  let expected = i16x8::from([3, 6, 10, 15, 20, 26, -32, -39]);
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_saturating_for_i16x8() {
  let a = i16x8::from([1, 2, 3, 4, 5, i16::MIN, i16::MIN + 1, i16::MAX]);
  let b = i16x8::from([17, -18, 190, -20, 21, -1, 1, -1]);
  let expected =
    i16x8::from([-16, 20, -187, 24, -16, i16::MIN + 1, i16::MIN, i16::MAX]);
  let actual = a.saturating_sub(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i16x8() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
  let b = i16x8::from([17, -18, 190, -20, 21, -22, 1, 1]);
  let expected =
    i16x8::from([17, -36, 570, -80, 105, -132, i16::MIN + 1, i16::MIN]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i16x8() {
  let a = i16x8::from([0, 0, 1, 1, 0, 0, 1, 1]);
  let b = i16x8::from([0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = i16x8::from([0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i16x8() {
  let a = i16x8::from([0, 0, 1, 1, 0, 0, 1, 1]);
  let b = i16x8::from([0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = i16x8::from([0, 1, 1, 1, 0, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i16x8() {
  let a = i16x8::from([0, 0, 1, 1, 0, 0, 1, 1]);
  let b = i16x8::from([0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = i16x8::from([0, 1, 1, 0, 0, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i16x8() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
  let b = 2;
  let expected = i16x8::from([
    1 << 2,
    2 << 2,
    3 << 2,
    4 << 2,
    5 << 2,
    6 << 2,
    (i16::MIN + 1) << 2,
    i16::MIN << 2,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i16x8() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
  let b = 2;
  let expected = i16x8::from([
    1 >> 2,
    2 >> 2,
    3 >> 2,
    4 >> 2,
    5 >> 2,
    6 >> 2,
    (i16::MIN + 1) >> 2,
    i16::MIN >> 2,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_cmp_eq() {
  let a = i16x8::from([1, 2, 3, 4, 1, 2, 3, 4]);
  let b = i16x8::from([2_i16; 8]);
  let expected = i16x8::from([0, -1, 0, 0, 0, -1, 0, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_cmp_gt() {
  let a = i16x8::from([1, 2, 3, 4, 1, 2, 3, 4]);
  let b = i16x8::from([2_i16; 8]);
  let expected = i16x8::from([0, 0, -1, -1, 0, 0, -1, -1]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_cmp_lt() {
  let a = i16x8::from([1, 2, 3, 4, 1, 2, 3, 4]);
  let b = i16x8::from([2_i16; 8]);
  let expected = i16x8::from([-1, 0, 0, 0, -1, 0, 0, 0]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);

  let expected = i16x8::from([0, 0, 0, 0, 0, 0, 0, 0]);
  let actual = a.cmp_lt(a);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_blend() {
  let use_t: i16 = -1;
  let t = i16x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let f = i16x8::from([17, 18, 19, 20, 21, 22, 23, 24]);
  let mask = i16x8::from([use_t, 0, use_t, 0, use_t, 0, use_t, 0]);
  let expected = i16x8::from([1, 18, 3, 20, 5, 22, 7, 24]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_abs() {
  let a = i16x8::from([1, -2, 3, -4, 5, -6, -7, i16::MIN]);
  let expected = i16x8::from([1, 2, 3, 4, 5, 6, 7, i16::MIN]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_max() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
  let b = i16x8::from([17, -18, 190, -20, 21, -22, 1, 1]);
  let expected = i16x8::from([17, 2, 190, 4, 21, 6, 1, 1]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x8_min() {
  let a = i16x8::from([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
  let b = i16x8::from([17, -18, 190, -20, 21, -22, 1, 1]);
  let expected = i16x8::from([1, -18, 3, -20, 5, -22, i16::MIN + 1, i16::MIN]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}