use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i64x8>(), 64);
  assert_eq!(core::mem::align_of::<i64x8>(), 64);
}

crate::generate_basic_traits_test!(i64x8, i64);

#[test]
fn impl_add_for_i64x8() {
  let a =
    i64x8::from([1, 2, i64::MAX - 1, i64::MAX - 1, 100, -50, i64::MIN + 1, 0]);
  let b = i64x8::from([17, 18, 1, 2, 200, -30, -1, i64::MAX]);
  let expected =
    i64x8::from([18, 20, i64::MAX, i64::MIN, 300, -80, i64::MIN, i64::MAX]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i64x8() {
  let a =
    i64x8::from([1, 2, i64::MIN + 1, i64::MIN + 1, 100, -50, i64::MAX - 1, 0]);
  let b = i64x8::from([17, 18, 1, 2, 200, -30, -1, i64::MAX]);
  let expected =
    i64x8::from([-16, -16, i64::MIN, i64::MAX, -100, -20, i64::MAX, -i64::MAX]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i64x8() {
  let a = i64x8::from([1, 2, 3, 4, -5, -6, -7, 0]);
  let b = i64x8::from([17, -18, 19, -20, 21, -22, -23, i64::MAX]);
  let expected = i64x8::from([17, -36, 57, -80, -105, 132, 161, 0]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i64x8() {
  let a = i64x8::from([0, 0, 1, 1, 0, 1, 1, -1]);
  let b = i64x8::from([0, 1, 0, 1, 1, 0, 1, i64::MAX]);
  let expected = i64x8::from([0, 0, 0, 1, 0, 0, 1, i64::MAX]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i64x8() {
  let a = i64x8::from([0, 0, 1, 1, 0, 1, 1, 0]);
  let b = i64x8::from([0, 1, 0, 1, 1, 0, 1, 0]);
  let expected = i64x8::from([0, 1, 1, 1, 1, 1, 1, 0]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i64x8() {
  let a = i64x8::from([0, 0, 1, 1, 0, 1, 1, 0]);
  let b = i64x8::from([0, 1, 0, 1, 1, 0, 1, 0]);
  let expected = i64x8::from([0, 1, 1, 0, 1, 1, 0, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i64x8() {
  let a = i64x8::from([1, 2, 3, 4, -1, -2, -3, i64::MAX]);
  let b = 1;
  let expected = i64x8::from([2, 4, 6, 8, -2, -4, -6, -2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i64x8() {
  let a = i64x8::from([2, 4, 6, 8, 7, i64::MAX / 2, 65536, i64::MAX]);
  let b = 1;
  let expected =
    i64x8::from([1, 2, 3, 4, 3, i64::MAX / 4, 32768, i64::MAX / 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x8_cmp_eq() {
  let a = i64x8::from([1, 2, 3, 4, -1, -2, -3, i64::MIN]);
  let b = i64x8::from([2, 2, 2, 2, -1, -1, -1, i64::MIN]);
  let expected = i64x8::from([0, -1, 0, 0, -1, 0, 0, -1]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x8_cmp_gt() {
  let a = i64x8::from([1, 2, 3, 4, -1, -2, -3, i64::MIN]);
  let b = i64x8::from([0, 2, 2, 5, -2, -1, -4, i64::MAX]);
  let expected = i64x8::from([-1, 0, -1, 0, -1, 0, -1, 0]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x8_cmp_lt() {
  let a = i64x8::from([1, 2, 3, 4, -1, -2, -3, i64::MIN]);
  let b = i64x8::from([0, 2, 4, 3, 0, -3, -2, i64::MAX]);
  let expected = i64x8::from([0, 0, -1, 0, -1, 0, -1, -1]);
  let actual = a.simd_lt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x8_blend() {
  let use_t = i64x8::from([0, -1, 0, -1, 0, -1, 0, -1]);
  let t = i64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let f = i64x8::from([9, 10, 11, 12, 13, 14, 15, 16]);
  let expected = i64x8::from([9, 2, 11, 4, 13, 6, 15, 8]);
  let actual = use_t.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x8_to_array() {
  let a = i64x8::from([1, 2, 3, 4, -5, -6, -7, i64::MIN]);
  let expected = [1, 2, 3, 4, -5, -6, -7, i64::MIN];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}

#[test]
fn impl_i64x8_new() {
  let a = i64x8::new([1, 2, 3, 4, -5, -6, -7, i64::MIN]);
  let expected = [1, 2, 3, 4, -5, -6, -7, i64::MIN];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}
