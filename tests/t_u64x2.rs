use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u64x2>(), 16);
  assert_eq!(core::mem::align_of::<u64x2>(), 16);
}

#[test]
fn impl_add_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = u64x2::from([1, 2]);
  let expected = u64x2::from([u64::MAX, u64::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u64x2() {
  let a = u64x2::from([1, 0]);
  let b = u64x2::from([1, 1]);
  let expected = u64x2::from([0, u64::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_u64x2() {
  let a = u64x2::from([1, 1]);
  let b = u64x2::from([0, 1]);
  let expected = u64x2::from([0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u64x2() {
  let a = u64x2::from([1, 1]);
  let b = u64x2::from([0, 1]);
  let expected = u64x2::from([1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u64x2() {
  let a = u64x2::from([1, 1]);
  let b = u64x2::from([0, 1]);
  let expected = u64x2::from([1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = 2_u32;
  let expected = u64x2::from([(u64::MAX - 1) << 2, (u64::MAX - 1) << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_u64x2() {
  let a = u64x2::from([u64::MAX - 1, u64::MAX - 1]);
  let b = 2_u32;
  let expected = u64x2::from([(u64::MAX - 1) >> 2, (u64::MAX - 1) >> 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}