use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u32x4>(), 16);
  assert_eq!(core::mem::align_of::<u32x4>(), 16);
}

#[test]
fn impl_add_for_u32x4() {
  let a = u32x4::from([1, 2, u32::MAX - 1, u32::MAX - 1]);
  let b = u32x4::from([17, 18, 1, 2]);
  let expected = u32x4::from([18, 20, u32::MAX, u32::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}
