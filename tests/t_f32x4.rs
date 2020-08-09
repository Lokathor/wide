use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<f32x4>(), 16);
  assert_eq!(core::mem::align_of::<f32x4>(), 16);
}

#[test]
fn impl_debug_for_f32x4() {
  let expected = "(1.0, 2.0, 3.0, 4.0)";
  let actual = format!("{:?}", f32x4::from([1.0, 2.0, 3.0, 4.0]));
  assert_eq!(expected, actual);

  let expected = "(1.000, 2.000, 3.000, 4.000)";
  let actual = format!("{:.3?}", f32x4::from([1.0, 2.0, 3.0, 4.0]));
  assert_eq!(expected, actual);
}

#[test]
fn impl_add_for_f32x4() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([5.0, 6.0, 7.0, 8.0]);
  let expected = f32x4::from([6.0, 8.0, 10.0, 12.0]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_f32x4() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([5.0, 7.0, 17.0, 1.0]);
  let expected = f32x4::from([-4.0, -5.0, -14.0, 3.0]);
  let actual = a - b;
  assert_eq!(expected, actual);
}
