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
