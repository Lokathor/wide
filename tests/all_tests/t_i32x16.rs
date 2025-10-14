use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i32x16>(), 64);
  assert_eq!(core::mem::align_of::<i32x16>(), 64);
}

#[test]
fn basic_traits() {
  type T = i32x16;
  use crate::TestBasicTraits;

  T::test_basic_traits_int();
  T::test_wrapping_mul_for_int();
  T::test_basic_traits_simd_cmp();
  T::test_basic_traits_aligned_to();
}

#[test]
fn impl_add_for_i32x16() {
  let a = i32x16::from([
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    i32::MAX - 1,
    14,
    15,
    16,
  ]);
  let b = i32x16::from([
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 1, 30, 31, 32,
  ]);
  let expected = i32x16::from([
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    36,
    38,
    40,
    i32::MAX,
    44,
    46,
    48,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i32x16() {
  let a = i32x16::from([
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    i32::MIN + 1,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
  ]);
  let b = i32x16::from([
    17, 18, 19, 20, 21, 22, 23, 24, 1, 26, 27, 28, 29, 30, 31, 32,
  ]);
  let expected = i32x16::from([
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    i32::MIN,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i32x16() {
  let a =
    i32x16::from([1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12, 13, 14, 15, 16]);
  let b = i32x16::from([
    17, -18, 19, -20, 21, -22, 23, -24, -25, 26, -27, 28, -29, 30, -31, 32,
  ]);
  let expected = i32x16::from([
    17, -36, 57, -80, -105, 132, -161, 192, -225, 260, -297, 336, -377, 420,
    -465, 512,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i32x16() {
  let a = i32x16::from([0; 16]);
  let b = i32x16::from([0; 16]);
  let expected = i32x16::from([0; 16]);
  let actual = a & b;
  assert_eq!(expected, actual);

  let a = i32x16::from([!0; 16]);
  let b = i32x16::from([0; 16]);
  let expected = i32x16::from([0; 16]);
  let actual = a & b;
  assert_eq!(expected, actual);

  let a = i32x16::from([!0; 16]);
  let b = i32x16::from([!0; 16]);
  let expected = i32x16::from([!0; 16]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i32x16() {
  let a = i32x16::from([0; 16]);
  let b = i32x16::from([0; 16]);
  let expected = i32x16::from([0; 16]);
  let actual = a | b;
  assert_eq!(expected, actual);

  let a = i32x16::from([!0; 16]);
  let b = i32x16::from([0; 16]);
  let expected = i32x16::from([!0; 16]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i32x16() {
  let a = i32x16::from([0; 16]);
  let b = i32x16::from([0; 16]);
  let expected = i32x16::from([0; 16]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  let a = i32x16::from([!0; 16]);
  let b = i32x16::from([0; 16]);
  let expected = i32x16::from([!0; 16]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  let a = i32x16::from([!0; 16]);
  let b = i32x16::from([!0; 16]);
  let expected = i32x16::from([0; 16]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i32x16() {
  let a = i32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]);
  let b = 1;
  let expected =
    i32x16::from([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i32x16() {
  let a =
    i32x16::from([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -2]);
  let b = 1;
  let expected =
    i32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_eq_for_i32x16() {
  let a = i32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  let b = i32x16::from([0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]);
  let expected =
    i32x16::from([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_gt_for_i32x16() {
  let a =
    i32x16::from([0, 1, 2, 3, -4, -3, -2, -1, 8, 9, 10, 11, 12, 13, 14, 15]);
  let b = i32x16::from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  let expected =
    i32x16::from([0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_lt_for_i32x16() {
  let a =
    i32x16::from([0, 1, 2, 3, -4, -3, -2, -1, 8, 9, 10, 11, 12, 13, 14, 15]);
  let b = i32x16::from([100; 16]);
  let expected = i32x16::from([-1; 16]);
  let actual = a.simd_lt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_blend_for_i32x16() {
  let use_t: i32x16 =
    i32x16::from([0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]);
  let t = i32x16::from([1; 16]);
  let f = i32x16::from([0; 16]);
  let expected = i32x16::from([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  let actual = use_t.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_min_for_i32x16() {
  let a = i32x16::from([
    0,
    1,
    2,
    3,
    -4,
    -3,
    -2,
    -1,
    i32::MIN,
    i32::MAX,
    100,
    -100,
    0,
    0,
    0,
    0,
  ]);
  let b = i32x16::from([
    15,
    14,
    13,
    12,
    -5,
    -6,
    -7,
    -8,
    i32::MAX,
    i32::MIN,
    -100,
    100,
    1,
    -1,
    0,
    0,
  ]);
  let expected = i32x16::from([
    0,
    1,
    2,
    3,
    -5,
    -6,
    -7,
    -8,
    i32::MIN,
    i32::MIN,
    -100,
    -100,
    0,
    -1,
    0,
    0,
  ]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_max_for_i32x16() {
  let a = i32x16::from([
    0,
    1,
    2,
    3,
    -4,
    -3,
    -2,
    -1,
    i32::MIN,
    i32::MAX,
    100,
    -100,
    0,
    0,
    0,
    0,
  ]);
  let b = i32x16::from([
    15,
    14,
    13,
    12,
    -5,
    -6,
    -7,
    -8,
    i32::MAX,
    i32::MIN,
    -100,
    100,
    1,
    -1,
    0,
    0,
  ]);
  let expected = i32x16::from([
    15,
    14,
    13,
    12,
    -4,
    -3,
    -2,
    -1,
    i32::MAX,
    i32::MAX,
    100,
    100,
    1,
    0,
    0,
    0,
  ]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_round_float_for_i32x16() {
  let a =
    i32x16::from([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]);
  let expected = f32x16::from([
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
    -2.0, -1.0,
  ]);
  let actual = a.round_float();
  assert_eq!(expected, actual);
}

#[test]
fn impl_i32x16_new() {
  let a = i32x16::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  let expected = [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}

#[test]
fn impl_basic_for_i32x16() {
  crate::test_random_vector_vs_scalar(
    |a: i32x16, b| a + b,
    |a, b| a.wrapping_add(b),
  );
  crate::test_random_vector_vs_scalar(
    |a: i32x16, b| a - b,
    |a, b| a.wrapping_sub(b),
  );
  crate::test_random_vector_vs_scalar(
    |a: i32x16, b| a * b,
    |a, b| a.wrapping_mul(b),
  );
  crate::test_random_vector_vs_scalar(|a: i32x16, b| a & b, |a, b| a & b);
  crate::test_random_vector_vs_scalar(|a: i32x16, b| a | b, |a, b| a | b);
  crate::test_random_vector_vs_scalar(|a: i32x16, b| a ^ b, |a, b| a ^ b);
  crate::test_random_vector_vs_scalar(|a: i32x16, b| a.min(b), |a, b| a.min(b));
  crate::test_random_vector_vs_scalar(|a: i32x16, b| a.max(b), |a, b| a.max(b));
}

#[test]
fn impl_shift_for_i32x16() {
  for shift in 0..32 {
    crate::test_random_vector_vs_scalar(
      |a: i32x16, _b| a << shift,
      |a, _b| a.wrapping_shl(shift),
    );
    crate::test_random_vector_vs_scalar(
      |a: i32x16, _b| a >> shift,
      |a, _b| a.wrapping_shr(shift),
    );
  }
}

#[test]
fn impl_cmp_for_i32x16() {
  crate::test_random_vector_vs_scalar(
    |a: i32x16, b| a.simd_eq(b),
    |a, b| if a == b { -1 } else { 0 },
  );
  crate::test_random_vector_vs_scalar(
    |a: i32x16, b| a.simd_gt(b),
    |a, b| if a > b { -1 } else { 0 },
  );
  crate::test_random_vector_vs_scalar(
    |a: i32x16, b| a.simd_lt(b),
    |a, b| if a < b { -1 } else { 0 },
  );
}
