use wide::*;

use bytemuck::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<f32x16>(), 64);
  assert_eq!(core::mem::align_of::<f32x16>(), 64);
}

crate::generate_basic_traits_test!(f32x16, f32);

#[test]
fn impl_debug_for_f32x16() {
  let expected = "(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)";
  let actual = format!(
    "{:?}",
    f32x16::from([
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
      14.0, 15.0, 16.0
    ])
  );
  assert_eq!(expected, actual);

  let expected = "(1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000, 10.000, 11.000, 12.000, 13.000, 14.000, 15.000, 16.000)";
  let actual = format!(
    "{:.3?}",
    f32x16::from([
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
      14.0, 15.0, 16.0
    ])
  );
  assert_eq!(expected, actual);
}

#[test]
fn impl_add_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let b = f32x16::from([
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0,
  ]);
  let expected = f32x16::from([
    6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
    32.0, 34.0, 36.0,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_add_const_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let expected = f32x16::from([
    6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 20.0, 21.0,
  ]);
  let actual = a + 5.0;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_const_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let expected = f32x16::from([
    -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0,
  ]);
  let actual = a - 2.0;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_const_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let expected = f32x16::from([
    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0,
    28.0, 30.0, 32.0,
  ]);
  let actual = a * 2.0;
  assert_eq!(expected, actual);
}

#[test]
fn impl_div_const_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let expected = f32x16::from([
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
    8.0,
  ]);
  let actual = a / 2.0;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let b = f32x16::from([
    5.0, 7.0, 17.0, 1.0, 1.0, 9.0, 2.0, 6.0, 3.0, 5.0, 8.0, 10.0, 15.0, 11.0,
    20.0, 14.0,
  ]);
  let expected = f32x16::from([
    -4.0, -5.0, -14.0, 3.0, 4.0, -3.0, 5.0, 2.0, 6.0, 5.0, 3.0, 2.0, -2.0, 3.0,
    -5.0, 2.0,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_neg_for_f32x16() {
  let a = f32x16::from([
    1.0,
    -2.0,
    3.0,
    -4.0,
    5.0,
    -6.0,
    7.0,
    -8.0,
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    9.0,
    -10.0,
    11.0,
    -12.0,
  ]);
  let expected = f32x16::from([
    -1.0,
    2.0,
    -3.0,
    4.0,
    -5.0,
    6.0,
    -7.0,
    8.0,
    -0.0,
    0.0,
    f32::NEG_INFINITY,
    f32::INFINITY,
    -9.0,
    10.0,
    -11.0,
    12.0,
  ]);
  assert_eq!(-a, expected);

  // Verify that 0.0 and -0.0 are properly sign-flipped
  let zero = f32x16::splat(0.0);
  let neg_zero = -zero;
  let bits: [u32; 16] = cast(neg_zero);
  assert_eq!(bits, [0x80000000u32; 16]); // All should be -0.0

  let neg_zero_input = f32x16::splat(-0.0);
  let pos_zero = -neg_zero_input;
  let bits: [u32; 16] = cast(pos_zero);
  assert_eq!(bits, [0x00000000u32; 16]); // All should be 0.0
}

#[test]
fn impl_mul_for_f32x16() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let b = f32x16::from([
    5.0, 7.0, 17.0, 1.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0,
  ]);
  let expected = f32x16::from([
    5.0, 14.0, 51.0, 4.0, 25.0, 36.0, 49.0, 64.0, 18.0, 30.0, 44.0, 60.0, 78.0,
    98.0, 120.0, 144.0,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_div_for_f32x16() {
  let a = f32x16::from([
    4.0, 9.0, 10.0, 12.0, 5.0, 6.0, 7.0, 8.0, 18.0, 20.0, 15.0, 16.0, 21.0,
    24.0, 30.0, 32.0,
  ]);
  let b = f32x16::from([
    2.0, 2.0, 5.0, -3.0, 2.0, 1.5, 3.0, 2.5, 3.0, 4.0, 5.0, 8.0, 7.0, 6.0,
    10.0, 16.0,
  ]);
  let expected = f32x16::from([
    2.0, 4.5, 2.0, -4.0, 2.5, 4.0, 2.3333333, 3.2, 6.0, 5.0, 3.0, 2.0, 3.0,
    4.0, 3.0, 2.0,
  ]);
  let actual = a / b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_f32x16() {
  let a = f32x16::from([
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    1.0,
  ]);
  let b = f32x16::from([
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    1.0,
  ]);
  let expected = f32x16::from([
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    1.0,
  ]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_f32x16() {
  let a = f32x16::from([
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    1.0,
  ]);
  let b = f32x16::from([
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    1.0,
  ]);
  let expected = f32x16::from([
    0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
    1.0,
  ]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_f32x16() {
  let a = f32x16::from([
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    1.0,
  ]);
  let b = f32x16::from([
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    1.0,
  ]);
  let expected = f32x16::from([
    0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    0.0,
  ]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_cmp_eq() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 5.0, 2.0,
    6.0,
  ]);
  let b = f32x16::from([2.0; 16]);
  let expected: [i32; 16] =
    [0, -1, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0];
  let actual: [i32; 16] = cast(a.simd_eq(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_cmp_ne() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 5.0, 2.0,
    6.0,
  ]);
  let b = f32x16::from([2.0; 16]);
  let expected: [i32; 16] =
    [-1, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1];
  let actual: [i32; 16] = cast(a.simd_ne(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_cmp_ge() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 5.0, 2.0,
    6.0,
  ]);
  let b = f32x16::from([2.0; 16]);
  let expected: [i32; 16] =
    [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1];
  let actual: [i32; 16] = cast(a.simd_ge(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_cmp_gt() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 1.0, 7.0, 8.0, 3.0, 2.0, 9.0, 10.0, 3.0,
    11.0,
  ]);
  let b = f32x16::from([3.0; 16]);
  let expected: [i32; 16] =
    [0, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, -1];
  let actual: [i32; 16] = cast(a.simd_gt(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_cmp_le() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 4.0, 3.0, 7.0, 8.0, 4.0, 2.0, 9.0,
    10.0,
  ]);
  let b = f32x16::from([4.0; 16]);
  let expected: [i32; 16] =
    [-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0];
  let actual: [i32; 16] = cast(a.simd_le(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_cmp_lt() {
  let a = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 3.0, 7.0, 8.0, 2.0, 1.0, 3.0, 9.0,
    10.0,
  ]);
  let b = f32x16::from([3.0; 16]);
  let expected: [i32; 16] =
    [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0];
  let actual: [i32; 16] = cast(a.simd_lt(b));
  assert_eq!(expected, actual);

  let expected: [i32; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let actual: [i32; 16] = cast(a.simd_lt(a));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_blend() {
  let use_t: f32 = f32::from_bits(u32::MAX);
  let t = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let f = f32x16::from([
    5.0, 6.0, 7.0, 8.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
    30.0, 31.0, 32.0,
  ]);
  let mask = f32x16::from([
    use_t, 0.0, use_t, 0.0, 0.0, 0.0, 0.0, use_t, use_t, 0.0, use_t, 0.0, 0.0,
    use_t, 0.0, use_t,
  ]);
  let expected = f32x16::from([
    1.0, 6.0, 3.0, 8.0, 21.0, 22.0, 23.0, 8.0, 9.0, 26.0, 11.0, 28.0, 29.0,
    14.0, 31.0, 16.0,
  ]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_abs() {
  let a = f32x16::from([
    -1.0,
    2.0,
    -3.5,
    f32::NEG_INFINITY,
    6.0,
    15.0,
    -19.0,
    -9.0,
    4.5,
    -20.0,
    f32::INFINITY,
    5.0,
    -4.0,
    13.0,
    9.5,
    -3.0,
  ]);
  let expected = f32x16::from([
    1.0,
    2.0,
    3.5,
    f32::INFINITY,
    6.0,
    15.0,
    19.0,
    9.0,
    4.5,
    20.0,
    f32::INFINITY,
    5.0,
    4.0,
    13.0,
    9.5,
    3.0,
  ]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_signum() {
  let array = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    24.01,
    -24.01,
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::NAN,
    24.01,
    -24.01,
    f32::MAX,
    f32::MIN,
  ];

  let expected = f32x16::new(array.map(f32::signum));
  let actual = f32x16::new(array).signum();

  // Use bitwise equality to accept NaNs as equal.
  assert_eq!(expected ^ actual, f32x16::ZERO);
}

#[test]
fn impl_f32x16_floor() {
  let a = f32x16::from([
    -1.1,
    60.9,
    1.1,
    f32::INFINITY,
    96.6,
    -53.2,
    0.1,
    9.2,
    6.9,
    -3.4,
    85.3,
    -79.8,
    4.2,
    -6.4,
    7.3,
    -9.1,
  ]);
  let expected = f32x16::from([
    -2.0,
    60.0,
    1.0,
    f32::INFINITY,
    96.0,
    -54.0,
    0.0,
    9.0,
    6.0,
    -4.0,
    85.0,
    -80.0,
    4.0,
    -7.0,
    7.0,
    -10.0,
  ]);
  let actual = a.floor();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_ceil() {
  let a = f32x16::from([
    -1.1,
    60.9,
    1.1,
    f32::NEG_INFINITY,
    96.6,
    -53.2,
    0.1,
    9.2,
    6.9,
    -3.4,
    85.3,
    -79.8,
    4.2,
    -6.4,
    7.3,
    -9.1,
  ]);
  let expected = f32x16::from([
    -1.0,
    61.0,
    2.0,
    f32::NEG_INFINITY,
    97.0,
    -53.0,
    1.0,
    10.0,
    7.0,
    -3.0,
    86.0,
    -79.0,
    5.0,
    -6.0,
    8.0,
    -9.0,
  ]);
  let actual = a.ceil();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_fast_max() {
  let a = f32x16::from([
    1.0,
    5.0,
    3.0,
    0.0,
    6.0,
    -8.0,
    12.0,
    9.0,
    2.0,
    -3.0,
    f32::INFINITY,
    10.0,
    19.0,
    -5.0,
    -1.0,
    -9.0,
  ]);
  let b = f32x16::from([
    2.0,
    -3.0,
    f32::INFINITY,
    10.0,
    19.0,
    -5.0,
    -1.0,
    -9.0,
    1.0,
    5.0,
    3.0,
    0.0,
    6.0,
    -8.0,
    12.0,
    9.0,
  ]);
  let expected = f32x16::from([
    2.0,
    5.0,
    f32::INFINITY,
    10.0,
    19.0,
    -5.0,
    12.0,
    9.0,
    2.0,
    5.0,
    f32::INFINITY,
    10.0,
    19.0,
    -5.0,
    12.0,
    9.0,
  ]);
  let actual = a.fast_max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_max() {
  let a = f32x16::from([
    1.0,
    5.0,
    3.0,
    f32::NAN,
    6.0,
    -8.0,
    12.0,
    f32::NAN,
    0.0,
    -5.0,
    7.0,
    15.0,
    f32::NAN,
    -20.0,
    8.0,
    11.0,
  ]);
  let b = f32x16::from([
    2.0,
    -3.0,
    f32::INFINITY,
    10.0,
    19.0,
    f32::NAN,
    -1.0,
    -9.0,
    5.0,
    -2.0,
    f32::NAN,
    10.0,
    3.0,
    -15.0,
    9.0,
    f32::NAN,
  ]);
  let expected = f32x16::from([
    2.0,
    5.0,
    f32::INFINITY,
    10.0,
    19.0,
    -8.0,
    12.0,
    -9.0,
    5.0,
    -2.0,
    7.0,
    15.0,
    3.0,
    -15.0,
    9.0,
    11.0,
  ]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_fast_min() {
  let a = f32x16::from([
    1.0,
    5.0,
    3.0,
    f32::NEG_INFINITY,
    6.0,
    -8.0,
    12.0,
    9.0,
    2.0,
    -3.0,
    f32::INFINITY,
    10.0,
    19.0,
    -5.0,
    -1.0,
    -9.0,
  ]);
  let b = f32x16::from([
    2.0,
    -3.0,
    f32::INFINITY,
    10.0,
    19.0,
    -5.0,
    -1.0,
    -9.0,
    1.0,
    5.0,
    3.0,
    f32::NEG_INFINITY,
    6.0,
    -8.0,
    12.0,
    9.0,
  ]);
  let expected = f32x16::from([
    1.0,
    -3.0,
    3.0,
    f32::NEG_INFINITY,
    6.0,
    -8.0,
    -1.0,
    -9.0,
    1.0,
    -3.0,
    3.0,
    f32::NEG_INFINITY,
    6.0,
    -8.0,
    -1.0,
    -9.0,
  ]);
  let actual = a.fast_min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_min() {
  let a = f32x16::from([
    1.0,
    5.0,
    3.0,
    f32::NEG_INFINITY,
    6.0,
    -8.0,
    12.0,
    f32::NAN,
    0.0,
    -5.0,
    7.0,
    15.0,
    f32::NAN,
    -20.0,
    8.0,
    11.0,
  ]);
  let b = f32x16::from([
    2.0,
    -3.0,
    f32::INFINITY,
    10.0,
    19.0,
    f32::NAN,
    -1.0,
    -9.0,
    5.0,
    -2.0,
    f32::NAN,
    10.0,
    3.0,
    -15.0,
    9.0,
    f32::NAN,
  ]);
  let expected = f32x16::from([
    1.0,
    -3.0,
    3.0,
    f32::NEG_INFINITY,
    6.0,
    -8.0,
    -1.0,
    -9.0,
    0.0,
    -5.0,
    7.0,
    10.0,
    3.0,
    -20.0,
    8.0,
    11.0,
  ]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_fast_clamp() {
  let value = f32x16::new([
    5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0,
    10.0, 10.0, 0.0,
  ]);
  let min = f32x16::new([
    3.0, 11.0, 5.0, 0.0, 3.0, 11.0, 5.0, 0.0, 3.0, 11.0, 5.0, 0.0, 3.0, 11.0,
    5.0, 0.0,
  ]);
  let max = f32x16::new([
    8.0, 14.0, 9.0, 0.0, 8.0, 14.0, 9.0, 0.0, 8.0, 14.0, 9.0, 0.0, 8.0, 14.0,
    9.0, 0.0,
  ]);
  let expected = f32x16::new([
    5.0, 11.0, 9.0, 0.0, 5.0, 11.0, 9.0, 0.0, 5.0, 11.0, 9.0, 0.0, 5.0, 11.0,
    9.0, 0.0,
  ]);
  let actual = value.fast_clamp(min, max);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_clamp() {
  let value = f32x16::new([
    5.0,
    10.0,
    10.0,
    f32::NAN,
    5.0,
    10.0,
    10.0,
    f32::NAN,
    5.0,
    10.0,
    10.0,
    f32::NAN,
    5.0,
    10.0,
    10.0,
    f32::NAN,
  ]);
  let min = f32x16::new([
    3.0, 11.0, 5.0, 1.0, 3.0, 11.0, 5.0, 1.0, 3.0, 11.0, 5.0, 1.0, 3.0, 11.0,
    5.0, 1.0,
  ]);
  let max = f32x16::new([
    8.0, 14.0, 9.0, 3.0, 8.0, 14.0, 9.0, 3.0, 8.0, 14.0, 9.0, 3.0, 8.0, 14.0,
    9.0, 3.0,
  ]);
  let expected = f32x16::new([
    5.0,
    11.0,
    9.0,
    f32::NAN,
    5.0,
    11.0,
    9.0,
    f32::NAN,
    5.0,
    11.0,
    9.0,
    f32::NAN,
    5.0,
    11.0,
    9.0,
    f32::NAN,
  ]);
  let actual = value.clamp(min, max);
  // Use bitwise equality to accept NaNs as equal.
  assert_eq!(expected ^ actual, f32x16::ZERO);
}

#[test]
#[should_panic]
fn impl_f32x16_clamp_min_gt_max() {
  let value = f32x16::new([
    5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0,
    10.0, 10.0, 0.0,
  ]);
  let min = f32x16::new([
    10.0, 11.0, 5.0, 1.0, 10.0, 11.0, 5.0, 1.0, 10.0, 11.0, 5.0, 1.0, 10.0,
    11.0, 5.0, 1.0,
  ]);
  let max = f32x16::new([
    8.0, 14.0, 9.0, 3.0, 8.0, 14.0, 9.0, 3.0, 8.0, 14.0, 9.0, 3.0, 8.0, 14.0,
    9.0, 3.0,
  ]);
  let _ = value.clamp(min, max);
}

#[test]
#[should_panic]
fn impl_f32x16_clamp_nan_min() {
  let value = f32x16::new([
    5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0,
    10.0, 10.0, 0.0,
  ]);
  let min = f32x16::new([
    3.0,
    11.0,
    5.0,
    f32::NAN,
    3.0,
    11.0,
    5.0,
    f32::NAN,
    3.0,
    11.0,
    5.0,
    f32::NAN,
    3.0,
    11.0,
    5.0,
    f32::NAN,
  ]);
  let max = f32x16::new([
    8.0, 14.0, 9.0, 3.0, 8.0, 14.0, 9.0, 3.0, 8.0, 14.0, 9.0, 3.0, 8.0, 14.0,
    9.0, 3.0,
  ]);
  let _ = value.clamp(min, max);
}

#[test]
#[should_panic]
fn impl_f32x16_clamp_nan_max() {
  let value = f32x16::new([
    5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0, 10.0, 10.0, 0.0, 5.0,
    10.0, 10.0, 0.0,
  ]);
  let min = f32x16::new([
    3.0, 11.0, 5.0, 1.0, 3.0, 11.0, 5.0, 1.0, 3.0, 11.0, 5.0, 1.0, 3.0, 11.0,
    5.0, 1.0,
  ]);
  let max = f32x16::new([
    8.0,
    14.0,
    9.0,
    f32::NAN,
    8.0,
    14.0,
    9.0,
    f32::NAN,
    8.0,
    14.0,
    9.0,
    f32::NAN,
    8.0,
    14.0,
    9.0,
    f32::NAN,
  ]);
  let _ = value.clamp(min, max);
}

#[test]
fn impl_f32x16_midpoint() {
  let a: [f32; 16] = [
    5.2,
    -16349.0,
    3467890356635.1,
    2401.0,
    -21.0,
    -236456708943.0,
    2340894786738.2,
    -4235.0,
    -21.0,
    -236456708943.0,
    2340894786738.2,
    -4235.0,
    5.2,
    -16349.0,
    3467890356635.1,
    2401.0,
  ];
  let b: [f32; 16] = [
    -21.0,
    -236456708943.0,
    2340894786738.2,
    -4235.0,
    5.2,
    -16349.0,
    3467890356635.1,
    2401.0,
    5.2,
    -16349.0,
    3467890356635.1,
    2401.0,
    -21.0,
    -236456708943.0,
    2340894786738.2,
    -4235.0,
  ];

  let expected = f32x16::new([
    a[0].midpoint(b[0]),
    a[1].midpoint(b[1]),
    a[2].midpoint(b[2]),
    a[3].midpoint(b[3]),
    a[4].midpoint(b[4]),
    a[5].midpoint(b[5]),
    a[6].midpoint(b[6]),
    a[7].midpoint(b[7]),
    a[8].midpoint(b[8]),
    a[9].midpoint(b[9]),
    a[10].midpoint(b[10]),
    a[11].midpoint(b[11]),
    a[12].midpoint(b[12]),
    a[13].midpoint(b[13]),
    a[14].midpoint(b[14]),
    a[15].midpoint(b[15]),
  ]);
  let actual = f32x16::new(a).midpoint(f32x16::new(b));

  // Use bitwise equality to accept NaNs as equal.
  assert_eq!(expected ^ actual, f32x16::ZERO);
}

#[test]
fn impl_f32x16_is_nan() {
  let a = f32x16::from([
    0.0,
    f32::NAN,
    f32::NAN,
    0.0,
    0.0,
    0.0,
    f32::NAN,
    0.0,
    1.0,
    2.0,
    1000.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    -0.0,
    f32::NAN,
    -1.0,
  ]);
  let expected: [u32; 16] = [
    0,
    u32::MAX,
    u32::MAX,
    0,
    0,
    0,
    u32::MAX,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    u32::MAX,
    0,
  ];
  let actual: [u32; 16] = cast(a.is_nan());
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_is_finite() {
  let a = f32x16::from([
    f32::NAN,
    1.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    2.0,
    5.0,
    f32::INFINITY,
    9.0,
    -0.0,
    8.0,
    100.0,
    -50.0,
    f32::NEG_INFINITY,
    f32::NAN,
    4.0,
    5.0,
  ]);
  let expected: [u32; 16] = [
    0,
    u32::MAX,
    0,
    0,
    u32::MAX,
    u32::MAX,
    0,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    0,
    0,
    u32::MAX,
    u32::MAX,
  ];
  let actual: [u32; 16] = cast(a.is_finite());
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_is_inf() {
  let a = f32x16::from([
    f32::NAN,
    1.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    2.0,
    5.0,
    f32::INFINITY,
    9.0,
    -0.0,
    8.0,
    100.0,
    -50.0,
    f32::NEG_INFINITY,
    f32::NAN,
    4.0,
    5.0,
  ]);
  let expected: [u32; 16] = [
    0,
    0,
    u32::MAX,
    u32::MAX,
    0,
    0,
    u32::MAX,
    0,
    0,
    0,
    0,
    0,
    u32::MAX,
    0,
    0,
    0,
  ];
  let actual: [u32; 16] = cast(a.is_inf());
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_round() {
  let a = f32x16::from([
    1.1, 2.5, 3.7, 4.0, 7.2, 10.5, 12.7, 35.12, -1.1, -2.5, -3.7, -4.0, -7.2,
    -10.5, -12.7, -35.12,
  ]);
  let expected = f32x16::from([
    1.0, 2.0, 4.0, 4.0, 7.0, 10.0, 13.0, 35.0, -1.0, -2.0, -4.0, -4.0, -7.0,
    -10.0, -13.0, -35.0,
  ]);
  let actual = a.round();
  assert_eq!(expected, actual);
  //
  let a = f32x16::from([
    f32::INFINITY,
    f32::NEG_INFINITY,
    5.5,
    5.0,
    7.2,
    10.5,
    12.7,
    35.12,
    0.0,
    -0.0,
    0.4,
    -0.4,
    3.6,
    -3.6,
    4.5,
    -4.5,
  ]);
  let expected = f32x16::from([
    f32::INFINITY,
    f32::NEG_INFINITY,
    6.0,
    5.0,
    7.0,
    10.0,
    13.0,
    35.0,
    0.0,
    -0.0,
    0.0,
    -0.0,
    4.0,
    -4.0,
    4.0,
    -4.0,
  ]);
  let actual = a.round();
  assert_eq!(expected, actual);
  //
  let a = f32x16::from(f32::NAN);
  let expected: [u32; 16] = [u32::MAX; 16];
  let actual: [u32; 16] = cast(a.round().is_nan());
  assert_eq!(expected, actual);
  //
  let a = f32x16::from(-0.0);
  let expected = a;
  let actual = a.round();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_fast_round_int() {
  for (f, i) in [(1.0, 1), (1.1, 1), (-2.1, -2), (2.5, 2), (0.0, 0), (-0.0, 0)]
    .iter()
    .copied()
  {
    let a = f32x16::from(f);
    let expected = i32x16::from(i);
    let actual = a.fast_round_int();
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_f32x16_round_int() {
  for (f, i) in [
    (1.0, 1),
    (1.1, 1),
    (-2.1, -2),
    (2.5, 2),
    (0.0, 0),
    (-0.0, 0),
    (f32::NAN, 0),
    (f32::INFINITY, i32::MAX),
    (f32::NEG_INFINITY, i32::MIN),
  ]
  .iter()
  .copied()
  {
    let a = f32x16::from(f);
    let expected = i32x16::from(i);
    let actual = a.round_int();
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_f32x16_trunc() {
  for array in [
    [
      0.0, -0.0, 1.0, -1.0, 5.3, -5.3, 27.8, -27.8, 5.3, -5.3, 27.8, -27.8,
      2401.63, -2401.63, 4911111.2, -4911111.2,
    ],
    [
      2401.63,
      -2401.63,
      4911111.2,
      -4911111.2,
      18388608.0,
      18388608.0,
      f32::MAX,
      f32::MIN,
      f32::INFINITY,
      f32::NEG_INFINITY,
      f32::NAN,
      30.0,
      2401.63,
      -2401.63,
      4911111.2,
      -4911111.2,
    ],
  ] {
    let expected = f32x16::new(array.map(f32::trunc));
    let actual = f32x16::new(array).trunc();

    // Use bitwise equality to accept NaNs as equal.
    assert_eq!(expected ^ actual, f32x16::ZERO);
  }
}

#[test]
fn impl_f32x16_fast_trunc_int() {
  for (f, i) in [(1.0, 1), (1.1, 1), (-2.1, -2), (2.5, 2), (3.7, 3), (-0.0, 0)]
    .iter()
    .copied()
  {
    let a = f32x16::from(f);
    let expected = i32x16::from(i);
    let actual = a.fast_trunc_int();
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_f32x16_trunc_int() {
  for (f, i) in [
    (1.0, 1),
    (1.1, 1),
    (-2.1, -2),
    (2.5, 2),
    (3.7, 3),
    (-0.0, 0),
    (f32::NAN, 0),
    (f32::INFINITY, i32::MAX),
    (f32::NEG_INFINITY, i32::MIN),
  ]
  .iter()
  .copied()
  {
    let a = f32x16::from(f);
    let expected = i32x16::from(i);
    let actual = a.trunc_int();
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_f32x16_fract() {
  for array in [
    [
      0.0, -0.0, 1.0, -1.0, 5.3, -5.3, 27.8, -27.8, 5.3, -5.3, 27.8, -27.8,
      2401.63, -2401.63, 4911111.2, -4911111.2,
    ],
    [
      2401.63,
      -2401.63,
      4911111.2,
      -4911111.2,
      18388608.0,
      18388608.0,
      f32::MAX,
      f32::MIN,
      f32::INFINITY,
      f32::NEG_INFINITY,
      f32::NAN,
      30.0,
      2401.63,
      -2401.63,
      4911111.2,
      -4911111.2,
    ],
  ] {
    let expected = f32x16::new(array.map(f32::fract));
    let actual = f32x16::new(array).fract();

    // Use bitwise equality to accept NaNs as equal.
    assert_eq!(expected ^ actual, f32x16::ZERO);
  }
}

#[test]
fn impl_f32x16_mul_add() {
  let a = f32x16::from([
    2.0, 3.0, 4.0, 5.0, 6.7, 9.2, 11.5, 12.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0,
  ]);
  let b = f32x16::from([
    4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, 5.6, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0,
  ]);
  let c = f32x16::from([1.0; 16]);
  let expected: [f32; 16] = cast(f32x16::from([
    9.0, 16.0, 25.0, 36.0, 11.05, 82.88, 49.3, 69.32, 3.0, 7.0, 13.0, 21.0,
    31.0, 43.0, 57.0, 73.0,
  ]));
  let actual: [f32; 16] = cast(a.mul_add(b, c));
  for (act, exp) in actual.iter().zip(expected.iter()) {
    assert!((exp - act).abs() < 0.000001);
  }
}

#[test]
fn impl_f32x16_mul_neg_add() {
  let a = f32x16::from([
    2.0, 3.0, 4.0, 5.0, 6.7, 9.2, 11.5, 12.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0,
  ]);
  let b = f32x16::from([
    4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, -5.6, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0,
  ]);
  let c = f32x16::from([1.0; 16]);
  let expected: [f32; 16] = cast(f32x16::from([
    -7.0, -14.0, -23.0, -34.0, -9.05, -80.88, -47.3, 69.32, -1.0, -5.0, -11.0,
    -19.0, -29.0, -41.0, -55.0, -71.0,
  ]));
  let actual: [f32; 16] = cast(a.mul_neg_add(b, c));
  for (act, exp) in actual.iter().zip(expected.iter()) {
    assert!((exp - act).abs() < 0.00001);
  }
}

#[test]
fn impl_f32x16_mul_sub() {
  let a = f32x16::from([
    2.0, 3.0, 4.0, 5.0, 6.7, 9.2, 11.5, 12.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0,
  ]);
  let b = f32x16::from([
    4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, 5.6, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0,
  ]);
  let c = f32x16::from([1.0; 16]);
  let expected: [f32; 16] = cast(f32x16::from([
    7.0, 14.0, 23.0, 34.0, 9.05, 80.88, 47.3, 67.32, 1.0, 5.0, 11.0, 19.0,
    29.0, 41.0, 55.0, 71.0,
  ]));
  let actual: [f32; 16] = cast(a.mul_sub(b, c));
  for (act, exp) in actual.iter().zip(expected.iter()) {
    assert!((exp - act).abs() < 0.000001);
  }
}

#[test]
fn impl_f32x16_mul_neg_sub() {
  let a = f32x16::from([
    2.0, 3.0, 4.0, 5.0, 6.7, 9.2, 11.5, 12.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0,
  ]);
  let b = f32x16::from([
    4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, -5.6, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0,
  ]);
  let c = f32x16::from([1.0; 16]);
  let expected: [f32; 16] = cast(f32x16::from([
    -9.0, -16.0, -25.0, -36.0, -11.05, -82.88, -49.3, 67.32, -3.0, -7.0, -13.0,
    -21.0, -31.0, -43.0, -57.0, -73.0,
  ]));
  let actual: [f32; 16] = cast(a.mul_neg_sub(b, c));
  for (act, exp) in actual.iter().zip(expected.iter()) {
    assert!((exp - act).abs() < 0.00001);
  }
}

#[test]
fn impl_f32x16_flip_signs() {
  let a = f32x16::from([
    1.0, 1.0, -1.0, -1.0, 5.2, 6.7, -8.2, -12.5, 3.0, -6.4, 7.2, -24.01, 3.2,
    1.6, -0.8, 0.4,
  ]);
  let b = f32x16::from([
    2.0, -3.0, 4.0, -5.0, 5.2, 6.7, -8.2, -12.5, 3.3, -4.0, -5.5, 6.6, -6.9,
    5.4, 3.1, -6.0,
  ]);
  let expected = f32x16::from([
    1.0, -1.0, -1.0, 1.0, 5.2, 6.7, 8.2, 12.5, 3.0, 6.4, -7.2, -24.01, -3.2,
    1.6, -0.8, -0.4,
  ]);
  let actual = a.flip_signs(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_copysign() {
  let a = f32x16::from([
    1.0, 1.0, -1.0, -1.0, 5.2, 6.7, -8.2, -12.5, 3.0, -6.4, 7.2, -24.01, 3.2,
    1.6, -0.8, 0.4,
  ]);
  let b = f32x16::from([
    2.0, -3.0, 4.0, -5.0, 5.2, 6.7, -8.2, -12.5, 3.3, -4.0, -5.5, 6.6, -6.9,
    5.4, 3.1, -6.0,
  ]);
  let expected = f32x16::from([
    1.0, -1.0, 1.0, -1.0, 5.2, 6.7, -8.2, -12.5, 3.0, -6.4, -7.2, 24.01, -3.2,
    1.6, 0.8, -0.4,
  ]);
  let actual = a.copysign(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_asin_acos() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f32 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let (actual_asins, actual_acoses) = f32x16::from(origs).asin_acos();
    for i in 0..8 {
      let orig = origs[i];
      let check = |name: &str, vals: f32x16, expected: f32| {
        let actual_arr: [f32; 16] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 0.0000006,
          "Wanted {name}({orig}) to be {expected} but got {actual}",
          name = name,
          orig = orig,
          expected = expected,
          actual = actual
        );
      };
      check("asin", actual_asins, orig.asin());
      check("acos", actual_acoses, orig.acos());
    }
  }
}

#[test]
fn impl_f32x16_asin() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 4) as f32 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let actual_asins = f32x16::from(origs).asin();
    for i in 0..8 {
      let orig = origs[i];
      let check = |name: &str, vals: f32x16, expected: f32| {
        let actual_arr: [f32; 16] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 0.0000006,
          "Wanted {name}({orig}) to be {expected} but got {actual}",
          name = name,
          orig = orig,
          expected = expected,
          actual = actual
        );
      };
      check("asin", actual_asins, orig.asin());
    }
  }
}

#[test]
fn impl_f32x16_acos() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f32 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let actual_acoses = f32x16::from(origs).acos();
    for i in 0..8 {
      let orig = origs[i];
      let check = |name: &str, vals: f32x16, expected: f32| {
        let actual_arr: [f32; 16] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 0.0000006,
          "Wanted {name}({orig}) to be {expected} but got {actual}",
          name = name,
          orig = orig,
          expected = expected,
          actual = actual
        );
      };
      check("acos", actual_acoses, orig.acos());
    }
  }
}

#[test]
fn impl_f32x16_atan() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f32 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let actual_atans = f32x16::from(origs).atan();
    for i in 0..8 {
      let orig = origs[i];
      let check = |name: &str, vals: f32x16, expected: f32| {
        let actual_arr: [f32; 16] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 0.0000006,
          "Wanted {name}({orig}) to be {expected} but got {actual}",
          name = name,
          orig = orig,
          expected = expected,
          actual = actual
        );
      };
      check("atan", actual_atans, orig.atan());
    }
  }
}

#[test]
fn impl_f32x16_atan2() {
  let inc_y = 1.0 / 51.0 / 8.0;
  let inc_x = 1.0 / 2501.0 / 8.0;
  for y in -50..=50 {
    let base_y = (y * 8) as f32 * inc_y;
    let origs_y = [
      base_y,
      base_y + inc_y,
      base_y + 2.0 * inc_y,
      base_y + 3.0 * inc_y,
      base_y + 4.0 * inc_y,
      base_y + 5.0 * inc_y,
      base_y + 6.0 * inc_y,
      base_y + 7.0 * inc_y,
      base_y,
      base_y + inc_y,
      base_y + 2.0 * inc_y,
      base_y + 3.0 * inc_y,
      base_y + 4.0 * inc_y,
      base_y + 5.0 * inc_y,
      base_y + 6.0 * inc_y,
      base_y + 7.0 * inc_y,
    ];
    let actual_y = f32x16::from(origs_y);
    for x in -2500..=2500 {
      let base_x = (x * 8) as f32 * inc_x;
      let origs_x = [
        base_x,
        base_x + inc_x,
        base_x + 2.0 * inc_x,
        base_x + 3.0 * inc_x,
        base_x + 4.0 * inc_x,
        base_x + 5.0 * inc_x,
        base_x + 6.0 * inc_x,
        base_x + 7.0 * inc_x,
        base_x,
        base_x + inc_x,
        base_x + 2.0 * inc_x,
        base_x + 3.0 * inc_x,
        base_x + 4.0 * inc_x,
        base_x + 5.0 * inc_x,
        base_x + 6.0 * inc_x,
        base_x + 7.0 * inc_x,
      ];
      let actual_x = f32x16::from(origs_x);
      let actual_atan2s = actual_y.atan2(actual_x);
      for i in 0..8 {
        let orig_y = origs_y[i];
        let orig_x = origs_x[i];
        let check = |name: &str, vals: f32x16, expected: f32| {
          let actual_arr: [f32; 16] = cast(vals);
          let actual = actual_arr[i];
          assert!(
            (actual - expected).abs() < 0.0000006,
            "Wanted {name}({orig_y}, {orig_x}) to be {expected} but got {actual}",
            name = name,
            orig_y = orig_y,
            orig_x = orig_x,
            expected = expected,
            actual = actual
          );
        };
        check("atan2", actual_atan2s, orig_y.atan2(orig_x));
      }
    }
  }
}

#[test]
fn impl_f32x16_sin_cos() {
  for x in -2500..=2500 {
    let base = (x * 4) as f32;
    let angles = [
      base,
      base + 1.0,
      base + 2.0,
      base + 3.0,
      base + 4.0,
      base + 5.0,
      base + 6.0,
      base + 7.0,
      base,
      base + 1.0,
      base + 2.0,
      base + 3.0,
      base + 4.0,
      base + 5.0,
      base + 6.0,
      base + 7.0,
    ];
    let (actual_sins, actual_coses) = f32x16::from(angles).sin_cos();
    for i in 0..4 {
      let angle = angles[i];
      let check = |name: &str, vals: f32x16, expected: f32| {
        let actual_arr: [f32; 16] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 0.0000002,
          "Wanted {name}({angle}) to be {expected} but got {actual}",
          name = name,
          angle = angle,
          expected = expected,
          actual = actual
        );
      };
      check("sin", actual_sins, angle.sin());
      check("cos", actual_coses, angle.cos());
    }
  }
}

#[test]
fn impl_f32x16_to_degrees() {
  let pi = core::f32::consts::PI;
  let a = f32x16::from([
    0.0,
    pi / 2.0,
    pi,
    2.0 * pi,
    0.0,
    pi / 2.0,
    pi,
    2.0 * pi,
    pi / 2.0,
    0.0,
    pi,
    2.0 * pi,
    pi / 2.0,
    pi / 2.0,
    pi,
    -pi,
  ]);
  let expected = f32x16::from([
    0.0, 90.0, 180.0, 360.0, 0.0, 90.0, 180.0, 360.0, 90.0, 0.0, 180.0, 360.0,
    90.0, 90.0, 180.0, -180.0,
  ]);
  let actual = a.to_degrees();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_to_radians() {
  let pi = core::f32::consts::PI;
  let a = f32x16::from([
    0.0, 90.0, 180.0, 360.0, 0.0, 90.0, 180.0, 360.0, 90.0, 0.0, 180.0, 360.0,
    90.0, 90.0, 180.0, -180.0,
  ]);
  let expected = f32x16::from([
    0.0,
    pi / 2.0,
    pi,
    2.0 * pi,
    0.0,
    pi / 2.0,
    pi,
    2.0 * pi,
    pi / 2.0,
    0.0,
    pi,
    2.0 * pi,
    pi / 2.0,
    pi / 2.0,
    pi,
    -pi,
  ]);
  let actual = a.to_radians();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x16_recip() {
  {
    let expected = f32x16::from(0.0);
    let actual = f32x16::from(f32::INFINITY).recip();
    assert_eq!(expected, actual);
  }
  {
    let expected = f32x16::from(0.0);
    let actual = f32x16::from(-f32::INFINITY).recip();
    assert_eq!(expected, actual);
  }
  {
    let actual = f32x16::from(f32::NAN).recip();
    assert!(actual.is_nan().any());
  }
  {
    let expected = f32x16::from(f32::INFINITY);
    let actual = f32x16::from(0.0).recip();
    assert_eq!(expected, actual);
  }
  {
    let expected = f32x16::from(0.49987793);
    let actual = f32x16::from(2.0).recip();
    let diff: [f32; 16] = cast((actual - expected).abs());
    assert!(diff[0] < 0.001);
  }
  {
    let expected = f32x16::from(-0.08102417);
    let actual = f32x16::from(-12.34).recip();
    let diff: [f32; 16] = cast((actual - expected).abs());
    assert!(diff[0] < 0.001);
  }
}

#[test]
fn impl_f32x16_recip_sqrt() {
  {
    let expected = f32x16::from(0.0);
    let actual = f32x16::from(f32::INFINITY).recip_sqrt();
    assert_eq!(expected, actual);
  }
  {
    let actual = f32x16::from(-f32::INFINITY).recip_sqrt();
    assert!(actual.is_nan().any());
  }
  {
    let actual = f32x16::from(f32::NAN).recip_sqrt();
    assert!(actual.is_nan().any());
  }
  {
    let expected = f32x16::from(f32::INFINITY);
    let actual = f32x16::from(0.0).recip_sqrt();
    assert_eq!(expected, actual);
  }
  {
    let expected = f32x16::from(0.70703125);
    let actual = f32x16::from(2.0).recip_sqrt();
    let diff: [f32; 16] = cast((actual - expected).abs());
    assert!(diff[0] < 0.001);
  }
  {
    let actual = f32x16::from(-12.34).recip_sqrt();
    assert!(actual.is_nan().any());
  }
}

#[test]
fn impl_f32x16_sqrt() {
  for (f, e) in [
    (f32::INFINITY, f32::INFINITY),
    (0.0, 0.0),
    (-0.0, -0.0),
    (4.0, 2.0),
    (9.0, 3.0),
    (16.0, 4.0),
    (25.0, 5.0),
    (5000.0 * 5000.0, 5000.0),
  ]
  .iter()
  .copied()
  {
    let expected = f32x16::from(e);
    let actual = f32x16::from(f).sqrt();
    assert_eq!(expected, actual);
  }
  assert_eq!(
    cast::<_, i32x16>(f32x16::from(f32::NAN).sqrt().is_nan()),
    i32x16::from(-1)
  );
  assert_eq!(
    cast::<_, i32x16>(f32x16::from(f32::NEG_INFINITY).sqrt().is_nan()),
    i32x16::from(-1)
  );
  assert_eq!(
    cast::<_, i32x16>(f32x16::from(-1.0).sqrt().is_nan()),
    i32x16::from(-1)
  );
}

#[test]
fn impl_f32x16_exp() {
  for f in [(-2.0), (-1.0), (0.0), (1.0), (1.5), (2.0), (10.0)].iter().copied()
  {
    let expected = f32x16::from((f as f32).exp());
    let actual = f32x16::from(f).exp();
    let diff_from_std: [f32; 16] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.000000000000001);
  }
}

#[test]
fn impl_f32x16_exp2() {
  for x in [-2.0, -1.1, 0.0, 1.3, 1.5, 2.0, 10.4] {
    let _: f32 = x;
    let expected = f32x16::from(x.exp2());
    let actual = f32x16::from(x).exp2();
    let diff_from_std: [f32; 16] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 1e-12);
  }
}

#[test]
fn test_f32x16_to_bitmask() {
  let a = f32x16::from([
    -1.0, 0.0, -2.0, -3.0, -1.0, 0.0, -2.0, -3.0, 5.0, -6.0, 7.0, -8.0, 9.0,
    10.0, -10.0, -1.0,
  ]);
  let expected = 0b1100101011011101;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);
  //
  let a = f32x16::from([
    1.0, 0.0, 2.0, -3.0, 1.0, 0.0, 2.0, -3.0, 0.0, -0.0, 1.0, 2.0, -3.0, 3.0,
    -2.0, -1.0,
  ]);
  let expected = 0b1101001010001000;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);
}

#[test]
fn test_f32x16_any() {
  let a = f32x16::from([
    -1.0,
    0.0,
    -2.0,
    -3.0,
    2.0,
    -1.0,
    -2.0,
    f32::NAN,
    -1.0,
    0.0,
    -2.0,
    -3.0,
    2.0,
    -1.0,
    -2.0,
    -0.0,
  ])
  .is_nan();
  assert!(a.any());
  //
  let a = f32x16::from([
    1.0, 0.0, 2.0, 3.0, 2.0, 5.0, 6.7, 7.1, -1.0, 0.0, -2.0, -3.0, 2.0, -1.0,
    -2.0, 0.0,
  ])
  .is_nan();
  assert!(!a.any());
}

#[test]
fn test_f32x16_all() {
  let a = f32x16::from([f32::NAN; 16]).is_nan();
  assert!(a.all());
  //
  let a = f32x16::from([
    1.0,
    -0.0,
    2.0,
    3.0,
    4.0,
    9.0,
    7.2,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
  ])
  .is_nan();
  assert!(!a.all());
}

#[test]
fn test_f32x16_none() {
  let a = f32x16::from([
    1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0,
    3.0,
  ])
  .is_nan();
  assert!(a.none());
  //
  let a = f32x16::from([
    1.0,
    -0.0,
    2.0,
    3.0,
    1.0,
    -0.0,
    2.0,
    f32::NAN,
    1.0,
    0.0,
    2.0,
    3.0,
    1.0,
    0.0,
    2.0,
    3.0,
  ])
  .is_nan();
  assert!(!a.none());
}

#[test]
fn impl_f32x16_ln() {
  for f in [0.1, 0.5, 1.0, 2.718282, 10.0, 35.0, 1250.0].iter().copied() {
    let expected = f32x16::from((f as f32).ln());
    let actual = f32x16::from(f).ln();
    let diff_from_std: [f32; 16] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.0000001);
  }
}

#[test]
fn impl_f32x16_log2() {
  for f in [0.1, 0.5, 1.0, 2.718282, 10.0, 35.0, 1250.0].iter().copied() {
    let expected = f32x16::from((f as f32).log2());
    let actual = f32x16::from(f).log2();
    let diff_from_std: [f32; 16] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.000001);
  }
}

#[test]
fn impl_f32x16_log10() {
  for f in [0.1, 0.5, 1.0, 2.718282, 10.0, 35.0, 1250.0].iter().copied() {
    let expected = f32x16::from((f as f32).log10());
    let actual = f32x16::from(f).log10();
    let diff_from_std: [f32; 16] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.000001);
  }
}

#[test]
fn impl_f32x16_pow_f32x16() {
  let p = f32x16::from([
    29.0, 0.1, 0.5, 1.0, 2.718282, -0.2, -1.5, 3.4, 29.0, 0.1, 0.5, 1.0,
    2.718282, -0.2, -1.5, 3.4,
  ]);
  let f = f32x16::from([
    1.2, 2.0, 3.0, 1.5, 9.2, 6.1, 2.5, -4.5, 1.2, 2.0, 3.0, 1.5, 9.2, 6.1, 2.5,
    -4.5,
  ]);
  let res = f.pow_f32x16(p);

  let p: [f32; 16] = cast(p);
  let f: [f32; 16] = cast(f);
  let res: [f32; 16] = cast(res);
  for i in 0..p.len() {
    let expected = f[i].powf(p[i]);
    if !(expected.is_nan() && res[i].is_nan()) {
      assert!((expected - res[i]).abs() < 0.0001);
    }
  }
}

#[test]
fn impl_f32x16_powf() {
  for f in [0.1, 0.5, 1.0, 2.718282, 3.0, 4.0, 2.5, -1.0].iter().copied() {
    let expected = f32x16::splat(2.0 as f32).powf(f);
    let actual = f32x16::from(2.0_f32.powf(f));
    let diff_from_std: [f32; 16] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.000001);
  }
}

#[test]
fn impl_f32x16_reduce_add() {
  let p = f32x16::from([
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.008, -0.01,
    0.005, -0.001, 0.006, 0.004, -0.009, 0.007,
  ]);
  assert!((p.reduce_add() - 0.056) < 0.000000001);
}

#[test]
fn impl_f32x16_reduce_mul() {
  let value = f32x16::new([
    0.2, 0.3, 0.5, 0.7, 1.1, 1.3, 1.7, 1.9, 2.3, 2.9, 3.1, 3.7, 4.1, 4.3, 4.7,
    5.3,
  ]);
  let expected = 3258.9158;
  let actual = value.reduce_mul();
  let difference = (actual - expected).abs();
  assert!(difference < 1e-2);
}

#[test]
fn impl_f32x16_sum() {
  let mut p = Vec::with_capacity(250_000);
  for _ in 0..125_000 {
    p.push(f32x16::splat(0.001));
  }
  let now = std::time::Instant::now();
  let sum: f32 = p.iter().map(|x| x.reduce_add()).sum();
  let duration = now.elapsed().as_micros();
  println!("Time take {} {}us", sum, duration);

  let p = vec![0.001; 1_000_000];
  let now = std::time::Instant::now();
  let sum2: f32 = p.iter().sum();
  let duration = now.elapsed().as_micros();
  println!("Time take {} {}us", sum2, duration);
}

#[test]
fn impl_transpose_for_f32x16() {
  let a = [
    f32x16::new([
      1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12,
      1.13, 1.14, 1.15, 1.16,
    ]),
    f32x16::new([
      2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 2.10, 2.11, 2.12,
      2.13, 2.14, 2.15, 2.16,
    ]),
    f32x16::new([
      3.01, 3.02, 3.03, 3.04, 3.05, 3.06, 3.07, 3.08, 3.09, 3.10, 3.11, 3.12,
      3.13, 3.14, 3.15, 3.16,
    ]),
    f32x16::new([
      4.01, 4.02, 4.03, 4.04, 4.05, 4.06, 4.07, 4.08, 4.09, 4.10, 4.11, 4.12,
      4.13, 4.14, 4.15, 4.16,
    ]),
    f32x16::new([
      5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 5.09, 5.10, 5.11, 5.12,
      5.13, 5.14, 5.15, 5.16,
    ]),
    f32x16::new([
      6.01, 6.02, 6.03, 6.04, 6.05, 6.06, 6.07, 6.08, 6.09, 6.10, 6.11, 6.12,
      6.13, 6.14, 6.15, 6.16,
    ]),
    f32x16::new([
      7.01, 7.02, 7.03, 7.04, 7.05, 7.06, 7.07, 7.08, 7.09, 7.10, 7.11, 7.12,
      7.13, 7.14, 7.15, 7.16,
    ]),
    f32x16::new([
      8.01, 8.02, 8.03, 8.04, 8.05, 8.06, 8.07, 8.08, 8.09, 8.10, 8.11, 8.12,
      8.13, 8.14, 8.15, 8.16,
    ]),
    f32x16::new([
      9.01, 9.02, 9.03, 9.04, 9.05, 9.06, 9.07, 9.08, 9.09, 9.10, 9.11, 9.12,
      9.13, 9.14, 9.15, 9.16,
    ]),
    f32x16::new([
      10.01, 10.02, 10.03, 10.04, 10.05, 10.06, 10.07, 10.08, 10.09, 10.10,
      10.11, 10.12, 10.13, 10.14, 10.15, 10.16,
    ]),
    f32x16::new([
      11.01, 11.02, 11.03, 11.04, 11.05, 11.06, 11.07, 11.08, 11.09, 11.10,
      11.11, 11.12, 11.13, 11.14, 11.15, 11.16,
    ]),
    f32x16::new([
      12.01, 12.02, 12.03, 12.04, 12.05, 12.06, 12.07, 12.08, 12.09, 12.10,
      12.11, 12.12, 12.13, 12.14, 12.15, 12.16,
    ]),
    f32x16::new([
      13.01, 13.02, 13.03, 13.04, 13.05, 13.06, 13.07, 13.08, 13.09, 13.10,
      13.11, 13.12, 13.13, 13.14, 13.15, 13.16,
    ]),
    f32x16::new([
      14.01, 14.02, 14.03, 14.04, 14.05, 14.06, 14.07, 14.08, 14.09, 14.10,
      14.11, 14.12, 14.13, 14.14, 14.15, 14.16,
    ]),
    f32x16::new([
      15.01, 15.02, 15.03, 15.04, 15.05, 15.06, 15.07, 15.08, 15.09, 15.10,
      15.11, 15.12, 15.13, 15.14, 15.15, 15.16,
    ]),
    f32x16::new([
      16.01, 16.02, 16.03, 16.04, 16.05, 16.06, 16.07, 16.08, 16.09, 16.10,
      16.11, 16.12, 16.13, 16.14, 16.15, 16.16,
    ]),
  ];

  let result = f32x16::transpose(a);

  let expected = [
    f32x16::new([
      1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01, 10.01, 11.01,
      12.01, 13.01, 14.01, 15.01, 16.01,
    ]),
    f32x16::new([
      1.02, 2.02, 3.02, 4.02, 5.02, 6.02, 7.02, 8.02, 9.02, 10.02, 11.02,
      12.02, 13.02, 14.02, 15.02, 16.02,
    ]),
    f32x16::new([
      1.03, 2.03, 3.03, 4.03, 5.03, 6.03, 7.03, 8.03, 9.03, 10.03, 11.03,
      12.03, 13.03, 14.03, 15.03, 16.03,
    ]),
    f32x16::new([
      1.04, 2.04, 3.04, 4.04, 5.04, 6.04, 7.04, 8.04, 9.04, 10.04, 11.04,
      12.04, 13.04, 14.04, 15.04, 16.04,
    ]),
    f32x16::new([
      1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05, 9.05, 10.05, 11.05,
      12.05, 13.05, 14.05, 15.05, 16.05,
    ]),
    f32x16::new([
      1.06, 2.06, 3.06, 4.06, 5.06, 6.06, 7.06, 8.06, 9.06, 10.06, 11.06,
      12.06, 13.06, 14.06, 15.06, 16.06,
    ]),
    f32x16::new([
      1.07, 2.07, 3.07, 4.07, 5.07, 6.07, 7.07, 8.07, 9.07, 10.07, 11.07,
      12.07, 13.07, 14.07, 15.07, 16.07,
    ]),
    f32x16::new([
      1.08, 2.08, 3.08, 4.08, 5.08, 6.08, 7.08, 8.08, 9.08, 10.08, 11.08,
      12.08, 13.08, 14.08, 15.08, 16.08,
    ]),
    f32x16::new([
      1.09, 2.09, 3.09, 4.09, 5.09, 6.09, 7.09, 8.09, 9.09, 10.09, 11.09,
      12.09, 13.09, 14.09, 15.09, 16.09,
    ]),
    f32x16::new([
      1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10,
      12.10, 13.10, 14.10, 15.10, 16.10,
    ]),
    f32x16::new([
      1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11,
      12.11, 13.11, 14.11, 15.11, 16.11,
    ]),
    f32x16::new([
      1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12,
      12.12, 13.12, 14.12, 15.12, 16.12,
    ]),
    f32x16::new([
      1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13,
      12.13, 13.13, 14.13, 15.13, 16.13,
    ]),
    f32x16::new([
      1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14,
      12.14, 13.14, 14.14, 15.14, 16.14,
    ]),
    f32x16::new([
      1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15,
      12.15, 13.15, 14.15, 15.15, 16.15,
    ]),
    f32x16::new([
      1.16, 2.16, 3.16, 4.16, 5.16, 6.16, 7.16, 8.16, 9.16, 10.16, 11.16,
      12.16, 13.16, 14.16, 15.16, 16.16,
    ]),
  ];

  assert_eq!(result, expected);
}

#[test]
fn impl_f32x16_from_i32x16() {
  let i = i32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  let f = f32x16::from([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  assert_eq!(f32x16::from_i32x16(i), f);
}

#[cfg(feature = "serde")]
#[test]
fn impl_f32x16_ser_de_roundtrip() {
  let serialized =
    bincode::serialize(&f32x16::ZERO).expect("serialization failed");
  let deserialized =
    bincode::deserialize(&serialized).expect("deserializaion failed");
  assert_eq!(f32x16::ZERO, deserialized);
}
