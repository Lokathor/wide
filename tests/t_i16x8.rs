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
  let b = 2_u32;
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
