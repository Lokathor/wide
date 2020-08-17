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

#[test]
fn impl_bitand_for_i32x4() {
  let a = i32x4::from([0, 0, 1, 1]);
  let b = i32x4::from([0, 1, 0, 1]);
  let expected = i32x4::from([0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i32x4() {
  let a = i32x4::from([0, 0, 1, 1]);
  let b = i32x4::from([0, 1, 0, 1]);
  let expected = i32x4::from([0, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i32x4() {
  let a = i32x4::from([0, 0, 1, 1]);
  let b = i32x4::from([0, 1, 0, 1]);
  let expected = i32x4::from([0, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i32x4() {
  let a = i32x4::from([1, 2, i32::MAX - 1, i32::MAX - 1]);
  let b = 2;
  let expected =
    i32x4::from([1 << 2, 2 << 2, (i32::MAX - 1) << 2, (i32::MAX - 1) << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i32x4() {
  let a = i32x4::from([1, 2, i32::MAX - 1, i32::MAX - 1]);
  let b = 2;
  let expected =
    i32x4::from([1 >> 2, 2 >> 2, (i32::MAX - 1) >> 2, (i32::MAX - 1) >> 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_cmp_eq() {
  let a = i32x4::from([1, 2, 3, 4]);
  let b = i32x4::from([2_i32; 4]);
  let expected = i32x4::from([0, -1, 0, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_cmp_gt() {
  let a = i32x4::from([1, 2, 3, 4]);
  let b = i32x4::from([2_i32; 4]);
  let expected = i32x4::from([0, 0, -1, -1]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_cmp_lt() {
  let a = i32x4::from([1, 2, 3, 4]);
  let b = i32x4::from([2_i32; 4]);
  let expected = i32x4::from([-1, 0, 0, 0]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_blend() {
  let use_t: i32 = -1;
  let t = i32x4::from([1, 2, 3, 4]);
  let f = i32x4::from([17, 18, 19, 20]);
  let mask = i32x4::from([use_t, 0, use_t, 0]);
  let expected = i32x4::from([1, 18, 3, 20]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_abs() {
  let a = i32x4::from([-1, 2, -3, i32::MIN]);
  let expected = i32x4::from([1, 2, 3, i32::MIN]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_max() {
  let a = i32x4::from([1, 2, i32::MIN + 1, i32::MIN]);
  let b = i32x4::from([17, -18, 1, 1]);
  let expected = i32x4::from([17, 2, 1, 1]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_min() {
  let a = i32x4::from([1, 2, i32::MIN + 1, i32::MIN]);
  let b = i32x4::from([17, -18, 1, 1]);
  let expected = i32x4::from([1, -18, i32::MIN + 1, i32::MIN]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x4_round_float() {
  let a = i32x4::from([-1, 30, i32::MIN, i32::MAX]);
  let expected = f32x4::from([-1.0, 30.0, i32::MIN as f32, i32::MAX as f32]);
  let actual = a.round_float();
  assert_eq!(expected, actual);
}
