use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i64x2>(), 16);
  assert_eq!(core::mem::align_of::<i64x2>(), 16);
}

#[test]
fn impl_add_for_i64x2() {
  let a = i64x2::from([i64::MAX - 1, i64::MAX - 1]);
  let b = i64x2::from([1, 2]);
  let expected = i64x2::from([i64::MAX, i64::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i64x2() {
  let a = i64x2::from([i64::MIN + 1, i64::MIN]);
  let b = i64x2::from([1, 1]);
  let expected = i64x2::from([i64::MIN, i64::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}
