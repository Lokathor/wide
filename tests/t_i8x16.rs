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
