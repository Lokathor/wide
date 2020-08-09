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
