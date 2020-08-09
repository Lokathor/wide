use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i32x4>(), 16);
  assert_eq!(core::mem::align_of::<i32x4>(), 16);
}

#[test]
fn impl_add_for_i32x4() {
  let a = i32x4::from([1, 2, i32::MAX - 1, i32::MAX - 1]);
  let b = i32x4::from([17, 18, 1, 2]);
  let expected = i32x4::from([18, 20, i32::MAX, i32::MIN]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i32x4() {
  let a = i32x4::from([1, 2, i32::MIN + 1, i32::MIN]);
  let b = i32x4::from([17, -18, 1, 1]);
  let expected = i32x4::from([-16, 20, i32::MIN, i32::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i32x4() {
  let a = i32x4::from([1, 2, i32::MIN + 1, i32::MIN]);
  let b = i32x4::from([17, -18, 1, 1]);
  let expected = i32x4::from([17, -36, i32::MIN + 1, i32::MIN]);
  let actual = a * b;
  assert_eq!(expected, actual);
}
