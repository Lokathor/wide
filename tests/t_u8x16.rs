use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u8x16>(), 16);
  assert_eq!(core::mem::align_of::<u8x16>(), 16);
}

#[test]
fn impl_add_for_u8x16() {
  let a =
    u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 250, 250]);
  let b =
    u8x16::from([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 5, 6]);
  let expected = u8x16::from([
    18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 255, 0,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u8x16() {
  let a = u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 0]);
  let b =
    u8x16::from([170, 18, 10, 200, 241, 2, 93, 4, 12, 8, 27, 28, 29, 30, 1, 1]);
  let expected = u8x16::from([
    87,
    240,
    249,
    60,
    20,
    4,
    170,
    4,
    253,
    2,
    240,
    240,
    240,
    240,
    0,
    u8::MAX,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}
