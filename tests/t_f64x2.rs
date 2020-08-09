use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<f64x2>(), 16);
  assert_eq!(core::mem::align_of::<f64x2>(), 16);
}

#[test]
fn impl_add_for_f64x2() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([5.0, 6.0]);
  let expected = f64x2::from([6.0, 8.0]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_f64x2() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([5.0, -10.0]);
  let expected = f64x2::from([-4.0, 12.0]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_f64x2() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([5.0, -10.0]);
  let expected = f64x2::from([5.0, -20.0]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_div_for_f64x2() {
  let a = f64x2::from([50.0, 2.0]);
  let b = f64x2::from([5.0, -10.0]);
  let expected = f64x2::from([10.0, -0.2]);
  let actual = a / b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_f64x2() {
  let a = f64x2::from([0.0, 1.0]);
  let b = f64x2::from([1.0, 1.0]);
  let expected = f64x2::from([0.0, 1.0]);
  let actual = a & b;
  assert_eq!(expected, actual);
}
