use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i8x16>(), 16);
  assert_eq!(core::mem::align_of::<i8x16>(), 16);
}

#[test]
fn impl_add_for_i8x16() {
  let a =
    i8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 126, 127]);
  let b =
    i8x16::from([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 1]);
  let expected = i8x16::from([
    18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 127, -128,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i8x16() {
  let a = i8x16::from([
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    i8::MIN + 1,
    i8::MIN,
  ]);
  let b =
    i8x16::from([17, 27, -1, 20, 21, -8, 23, 0, 1, 2, -9, 28, 64, 30, 1, 1]);
  let expected = i8x16::from([
    -16,
    -25,
    4,
    -16,
    -16,
    14,
    -16,
    8,
    8,
    8,
    20,
    -16,
    -51,
    -16,
    i8::MIN,
    i8::MAX,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_add_for_i8x16() {
  let a =
    i8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 126, 127]);
  let b =
    i8x16::from([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 1]);
  let expected = i8x16::from([
    18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 127, 127,
  ]);
  let actual = a.saturating_add(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_sub_for_i8x16() {
  let a = i8x16::from([
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    i8::MIN + 1,
    i8::MIN,
  ]);
  let b =
    i8x16::from([17, 27, -1, 20, 21, -8, 23, 0, 1, 2, -9, 28, 64, 30, 1, 1]);
  let expected = i8x16::from([
    -16,
    -25,
    4,
    -16,
    -16,
    14,
    -16,
    8,
    8,
    8,
    20,
    -16,
    -51,
    -16,
    i8::MIN,
    i8::MIN,
  ]);
  let actual = a.saturating_sub(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i8x16() {
  let a = i8x16::from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
  let b = i8x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = i8x16::from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i8x16() {
  let a = i8x16::from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
  let b = i8x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = i8x16::from([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i8x16() {
  let a = i8x16::from([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
  let b = i8x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let expected = i8x16::from([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_cmp_eq() {
  let a = i8x16::from([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
  let b = i8x16::from([2_i8; 16]);
  let expected =
    i8x16::from([0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_cmp_gt() {
  let a = i8x16::from([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
  let b = i8x16::from([2_i8; 16]);
  let expected =
    i8x16::from([0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_cmp_lt() {
  let a = i8x16::from([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
  let b = i8x16::from([2_i8; 16]);
  let expected =
    i8x16::from([-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);

  let expected = i8x16::from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  let actual = a.cmp_lt(a);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_blend() {
  let use_t: i8 = -1;
  let t =
    i8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 126, 127]);
  let f =
    i8x16::from([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 1]);
  let mask = i8x16::from([
    use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0,
    use_t, 0,
  ]);
  let expected =
    i8x16::from([1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 126, 1]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_abs() {
  let a = i8x16::from([
    -1,
    2,
    -3,
    4,
    5,
    -6,
    7,
    8,
    9,
    -10,
    -11,
    12,
    13,
    -14,
    -126,
    i8::MIN,
  ]);
  let expected =
    i8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 126, i8::MIN]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_max() {
  let a =
    i8x16::from([10, 2, -3, 4, 5, -6, 7, 8, 9, 7, -11, 12, 13, 6, 55, i8::MIN]);
  let b = i8x16::from([
    -1,
    2,
    -3,
    4,
    5,
    -6,
    7,
    8,
    9,
    -10,
    -11,
    12,
    13,
    -14,
    -126,
    i8::MIN + 1,
  ]);
  let expected =
    i8x16::from([10, 2, -3, 4, 5, -6, 7, 8, 9, 7, -11, 12, 13, 6, 55, -127]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_min() {
  let a =
    i8x16::from([10, 2, -3, 4, 5, -6, 7, 8, 9, 7, -11, 12, 13, 6, 55, i8::MIN]);
  let b = i8x16::from([
    -1,
    2,
    -3,
    4,
    5,
    -6,
    7,
    8,
    9,
    -10,
    -11,
    12,
    13,
    -14,
    -126,
    i8::MIN + 1,
  ]);
  let expected = i8x16::from([
    -1,
    2,
    -3,
    4,
    5,
    -6,
    7,
    8,
    9,
    -10,
    -11,
    12,
    13,
    -14,
    -126,
    i8::MIN,
  ]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i8x16_widen() {
  let a = i8x16::from([
    10,
    2,
    -3,
    4,
    5,
    -6,
    7,
    8,
    9,
    7,
    i8::MAX,
    12,
    13,
    6,
    55,
    i8::MIN,
  ]);

  let actual = a.convert_to_i16();

  let expected = i16x16::from([
    10,
    2,
    -3,
    4,
    5,
    -6,
    7,
    8,
    9,
    7,
    i8::MAX as i16,
    12,
    13,
    6,
    55,
    i8::MIN as i16,
  ]);

  assert_eq!(expected, actual);
}
