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

#[test]
fn impl_sub_for_u32x4() {
  let a = u32x4::from([9001, 2, 1, 0]);
  let b = u32x4::from([17, 18, 1, 1]);
  let expected = u32x4::from([8984, 4294967280, 0, u32::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_u32x4() {
  let a = u32x4::from([0, 0, 1, 1]);
  let b = u32x4::from([0, 1, 0, 1]);
  let expected = u32x4::from([0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u32x4() {
  let a = u32x4::from([0, 0, 1, 1]);
  let b = u32x4::from([0, 1, 0, 1]);
  let expected = u32x4::from([0, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}
