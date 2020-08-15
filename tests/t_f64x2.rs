use wide::*;

use bytemuck::*;

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

#[test]
fn impl_bitor_for_f64x2() {
  let a = f64x2::from([0.0, 1.0]);
  let b = f64x2::from([1.0, 1.0]);
  let expected = f64x2::from([1.0, 1.0]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_f64x2() {
  let a = f64x2::from([0.0, 1.0]);
  let b = f64x2::from([1.0, 1.0]);
  let expected = f64x2::from([1.0, 0.0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_cmp_eq() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [0, -1];
  let actual: [i64; 2] = cast(a.cmp_eq(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_cmp_ne() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [-1, 0];
  let actual: [i64; 2] = cast(a.cmp_ne(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_cmp_ge() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [0, -1];
  let actual: [i64; 2] = cast(a.cmp_ge(b));
  assert_eq!(expected, actual);
  //
  let a = f64x2::from([3.0, 4.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [-1, -1];
  let actual: [i64; 2] = cast(a.cmp_ge(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_cmp_gt() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [0, 0];
  let actual: [i64; 2] = cast(a.cmp_gt(b));
  assert_eq!(expected, actual);
  //
  let a = f64x2::from([3.0, 4.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [-1, -1];
  let actual: [i64; 2] = cast(a.cmp_gt(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_cmp_le() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [-1, -1];
  let actual: [i64; 2] = cast(a.cmp_le(b));
  assert_eq!(expected, actual);
  //
  let a = f64x2::from([3.0, 4.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [0, 0];
  let actual: [i64; 2] = cast(a.cmp_le(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_cmp_lt() {
  let a = f64x2::from([1.0, 2.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [-1, 0];
  let actual: [i64; 2] = cast(a.cmp_lt(b));
  assert_eq!(expected, actual);
  //
  let a = f64x2::from([3.0, 4.0]);
  let b = f64x2::from([2.0, 2.0]);
  let expected: [i64; 2] = [0, 0];
  let actual: [i64; 2] = cast(a.cmp_lt(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_blend() {
  let use_t: f64 = f64::from_bits(u64::MAX);
  let t = f64x2::from([1.0, 2.0]);
  let f = f64x2::from([5.0, 6.0]);
  let mask = f64x2::from([use_t, 0.0]);
  let expected = f64x2::from([1.0, 6.0]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f64x2_abs() {
  let a = f64x2::from([-1.0, 2.0]);
  let expected = f64x2::from([1.0, 2.0]);
  let actual = a.abs();
  assert_eq!(expected, actual);
  //
  let a = f64x2::from([-3.5, f64::NEG_INFINITY]);
  let expected = f64x2::from([3.5, f64::INFINITY]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}
