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
