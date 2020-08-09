use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u16x8>(), 16);
  assert_eq!(core::mem::align_of::<u16x8>(), 16);
}

#[test]
fn impl_add_for_u16x8() {
  let a = u16x8::from([1, 2, 3, 4, 5, 6, u16::MAX - 1, u16::MAX - 1]);
  let b = u16x8::from([17, 18, 19, 20, 21, 22, 1, 2]);
  let expected = u16x8::from([18, 20, 22, 24, 26, 28, u16::MAX, 0]);
  let actual = a + b;
  assert_eq!(expected, actual);
}
