use std::num::Wrapping;
use wide::*;

// FIXME: ensure all tests here have significant differences between the upper
// and lower halves of the operand registers. Many rn are just duplicated across
// halves

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u32x16>(), 64);
  assert_eq!(core::mem::align_of::<u32x16>(), 64);
}

#[test]
fn basic_traits() {
  crate::test_basic_traits::<u32x16, _, 16>();
}

#[test]
fn impl_add_for_u32x16() {
  let a = u32x16::from([
    1,
    2,
    u32::MAX - 1,
    u32::MAX - 1,
    31,
    72,
    13,
    53,
    500,
    516,
    u32::MAX / 2,
    777,
    1000,
    2020,
    9999,
    65536,
  ]);
  let b = u32x16::from([
    17,
    18,
    1,
    2,
    12,
    12,
    634,
    15,
    500,
    16,
    u32::MAX / 4,
    23,
    777,
    5,
    1,
    65536,
  ]);
  let expected = u32x16::from([
    18,
    20,
    u32::MAX,
    u32::MIN,
    43,
    84,
    647,
    68,
    1000,
    532,
    u32::MAX / 2 + u32::MAX / 4,
    800,
    1777,
    2025,
    10000,
    131072,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a + b,
    |a, b| a.wrapping_add(b),
  );
}

#[test]
fn impl_sub_for_u32x16() {
  let a =
    u32x16::from([9001, 2, 1, 0, 12, 1, 9, 10, 9001, 2, 1, 0, 12, 1, 9, 10]);
  let b = u32x16::from([17, 18, 1, 1, 15, 1, 2, 5, 17, 18, 1, 1, 15, 1, 2, 5]);
  let expected = u32x16::from([
    8984,
    4294967280,
    0,
    u32::MAX,
    4294967293,
    0,
    7,
    5,
    8984,
    4294967280,
    0,
    u32::MAX,
    4294967293,
    0,
    7,
    5,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a - b,
    |a, b| a.wrapping_sub(b),
  );
}

#[test]
fn impl_mul_for_u32x16() {
  let a = u32x16::from([
    1,
    2,
    u32::MIN + 1,
    u32::MAX,
    123,
    u32::MIN,
    9,
    3802,
    1,
    2,
    u32::MIN + 1,
    u32::MAX,
    123,
    u32::MIN,
    9,
    3802,
  ]);
  let b = u32x16::from([
    17, 18, 1, 32, 456, 4, 190, 100, 17, 18, 1, 32, 456, 4, 190, 100,
  ]);
  let expected = u32x16::from([
    17,
    36,
    1,
    (Wrapping(u32::MAX) * Wrapping(32)).0,
    123 * 456,
    0,
    190 * 9,
    380200,
    17,
    36,
    1,
    (Wrapping(u32::MAX) * Wrapping(32)).0,
    123 * 456,
    0,
    190 * 9,
    380200,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a * b,
    |a, b| a.wrapping_mul(b),
  );
}

#[test]
fn impl_bitand_for_u32x16() {
  let a = u32x16::from([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x16::from([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u32x16::from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, b| a | b, |a, b| a | b);
}

#[test]
fn impl_bitor_for_u32x16() {
  let a = u32x16::from([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x16::from([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u32x16::from([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, b| a & b, |a, b| a & b);
}

#[test]
fn impl_bitxor_for_u32x16() {
  let a = u32x16::from([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u32x16::from([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u32x16::from([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, b| a ^ b, |a, b| a ^ b);
}

#[test]
fn impl_shl_for_u32x16() {
  let a = u32x16::from([
    1,
    2,
    u32::MAX - 1,
    i32::MAX as u32 - 1,
    128,
    255,
    590,
    5667,
    1,
    2,
    u32::MAX - 1,
    i32::MAX as u32 - 1,
    128,
    255,
    590,
    5667,
  ]);
  let b = 2;
  let expected = u32x16::from([
    1 << 2,
    2 << 2,
    (u32::MAX - 1) << 2,
    (i32::MAX as u32 - 1) << 2,
    128 << 2,
    255 << 2,
    590 << 2,
    5667 << 2,
    1 << 2,
    2 << 2,
    (u32::MAX - 1) << 2,
    (i32::MAX as u32 - 1) << 2,
    128 << 2,
    255 << 2,
    590 << 2,
    5667 << 2,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, _b| a << 3, |a, _b| a << 3);
}

#[test]
fn impl_shr_for_u32x16() {
  let a = u32x16::from([
    1,
    2,
    u32::MAX - 1,
    i32::MAX as u32 - 1,
    128,
    255,
    590,
    5667,
    1,
    2,
    u32::MAX - 1,
    i32::MAX as u32 - 1,
    128,
    255,
    590,
    5667,
  ]);
  let b = 2;
  let expected = u32x16::from([
    1 >> 2,
    2 >> 2,
    (u32::MAX - 1) >> 2,
    (i32::MAX as u32 - 1) >> 2,
    128 >> 2,
    255 >> 2,
    590 >> 2,
    5667 >> 2,
    1 >> 2,
    2 >> 2,
    (u32::MAX - 1) >> 2,
    (i32::MAX as u32 - 1) >> 2,
    128 >> 2,
    255 >> 2,
    590 >> 2,
    5667 >> 2,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, _b| a >> 3, |a, _b| a >> 3);
}

#[test]
fn impl_u32x16_cmp_eq() {
  let a = u32x16::from([1, 2, 3, 4, 2, 1, 8, 2, 4, 3, 2, 1, 4, 2, 3, 1]);
  let b = u32x16::from([2_u32; 16]);
  let expected = u32x16::from([
    0,
    u32::MAX,
    0,
    0,
    u32::MAX,
    0,
    0,
    u32::MAX,
    0,
    0,
    u32::MAX,
    0,
    0,
    u32::MAX,
    0,
    0,
  ]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x16_cmp_gt() {
  let a = u32x16::from([
    1,
    2,
    u32::MAX,
    4,
    1,
    2,
    8,
    10,
    1,
    2,
    u32::MIN,
    4,
    1,
    2,
    8,
    10,
  ]);
  let b = u32x16::from([5_u32; 16]);
  let expected = u32x16::from([
    0,
    0,
    u32::MAX,
    0,
    0,
    0,
    u32::MAX,
    u32::MAX,
    0,
    0,
    0,
    0,
    0,
    0,
    u32::MAX,
    u32::MAX,
  ]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a.simd_gt(b),
    |a, b| if a > b { u32::MAX } else { 0 },
  );
}

#[test]
fn impl_u32x16_cmp_lt() {
  let a = u32x16::from([5_u32; 16]);
  let b = u32x16::from([
    1,
    2,
    u32::MAX,
    4,
    1,
    2,
    8,
    10,
    1,
    2,
    u32::MIN,
    4,
    1,
    2,
    8,
    10,
  ]);
  let expected = u32x16::from([
    0,
    0,
    u32::MAX,
    0,
    0,
    0,
    u32::MAX,
    u32::MAX,
    0,
    0,
    0,
    0,
    0,
    0,
    u32::MAX,
    u32::MAX,
  ]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a.cmp_lt(b),
    |a, b| if a < b { u32::MAX } else { 0 },
  );
}

#[test]
fn impl_u32x16_blend() {
  let use_t: u32 = u32::MAX;
  let t = u32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
  let f = u32x16::from([
    17, 18, 19, 20, 25, 30, 50, 90, 17, 18, 19, 20, 25, 30, 50, 90,
  ]);
  let mask = u32x16::from([
    use_t, 0, use_t, 0, 0, 0, 0, use_t, use_t, 0, use_t, 0, 0, 0, 0, use_t,
  ]);
  let expected =
    u32x16::from([1, 18, 3, 20, 25, 30, 50, 8, 1, 18, 3, 20, 25, 30, 50, 8]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u32x16_max() {
  let a = u32x16::from([
    1,
    2,
    1,
    0,
    6,
    0,
    12,
    u32::MAX,
    1,
    2,
    1,
    0,
    6,
    0,
    12,
    u32::MAX,
  ]);
  let b =
    u32x16::from([17, 0, 1, 1, 19, 0, 0, 1000, 17, 0, 1, 1, 19, 0, 0, 1000]);
  let expected = u32x16::from([
    17,
    2,
    1,
    1,
    19,
    0,
    12,
    u32::MAX,
    17,
    2,
    1,
    1,
    19,
    0,
    12,
    u32::MAX,
  ]);
  let actual = a.max(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, b| a.max(b), |a, b| a.max(b));
}

#[test]
fn impl_u32x16_min() {
  let a = u32x16::from([
    1,
    2,
    1,
    0,
    6,
    0,
    12,
    u32::MAX,
    1,
    2,
    1,
    0,
    6,
    0,
    12,
    u32::MAX,
  ]);
  let b =
    u32x16::from([17, 0, 1, 1, 19, 0, 0, 1000, 17, 0, 1, 1, 19, 0, 0, 1000]);
  let expected =
    u32x16::from([1, 0, 1, 0, 6, 0, 0, 1000, 1, 0, 1, 0, 6, 0, 0, 1000]);
  let actual = a.min(b);
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, b| a.min(b), |a, b| a.min(b));
}

#[test]
fn impl_u32x4_shr_each() {
  let a = u32x16::from([
    15313,
    52322,
    u32::MAX,
    4,
    10,
    20,
    30,
    40,
    15313,
    52322,
    u32::MAX,
    4,
    10,
    20,
    30,
    40,
  ]);
  let shift = u32x16::from([
    1, 30, 8, 33, /* test masking behavior */
    1, 2, 3, 4, 1, 30, 8, 33, 1, 2, 3, 4,
  ]);
  let expected = u32x16::from([
    7656, 0, 16777215, 2, 5, 5, 3, 2, 7656, 0, 16777215, 2, 5, 5, 3, 2,
  ]);
  let actual = a >> shift;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a >> b,
    |a, b| a.wrapping_shr(b),
  );
}

#[test]
fn impl_u32x16_shl_each() {
  let a = u32x16::from([
    15313,
    52322,
    u32::MAX,
    4,
    1,
    2,
    3,
    4,
    15313,
    52322,
    u32::MAX,
    4,
    1,
    2,
    3,
    4,
  ]);
  let shift = u32x16::from([
    1, 30, 8, 33, /* test masking behavior */
    1, 2, 3, 4, 1, 30, 8, 33, 1, 2, 3, 4,
  ]);
  let expected = u32x16::from([
    30626, 2147483648, 4294967040, 8, 2, 8, 24, 64, 30626, 2147483648,
    4294967040, 8, 2, 8, 24, 64,
  ]);
  let actual = a << shift;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| a << b,
    |a, b| a.wrapping_shl(b),
  );
}

#[test]
fn impl_u32x16_not() {
  let a = u32x16::from([
    15313,
    52322,
    u32::MAX,
    4,
    1,
    2,
    3,
    4,
    15313,
    52322,
    u32::MAX,
    4,
    1,
    2,
    3,
    4,
  ]);
  let expected = u32x16::from([
    4294951982, 4294914973, 0, 4294967291, 4294967294, 4294967293, 4294967292,
    4294967291, 4294951982, 4294914973, 0, 4294967291, 4294967294, 4294967293,
    4294967292, 4294967291,
  ]);
  let actual = !a;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u32x16, _b| !a, |a, _b| !a);
}

#[test]
fn impl_u32x16_from_u16x16() {
  let a = u16x16::from([
    1,
    2,
    3,
    4,
    5,
    i16::MAX as u16,
    u16::MAX - 1,
    u16::MAX,
    1,
    2,
    3,
    4,
    5,
    i16::MAX as u16,
    u16::MAX - 1,
    u16::MAX,
  ]);
  let actual = u32x16::from(a);
  let expected = u32x16::from([
    1,
    2,
    3,
    4,
    5,
    i16::MAX as u32,
    (u16::MAX - 1) as u32,
    u16::MAX as u32,
    1,
    2,
    3,
    4,
    5,
    i16::MAX as u32,
    (u16::MAX - 1) as u32,
    u16::MAX as u32,
  ]);

  assert_eq!(actual, expected);

  crate::test_random_vector_vs_scalar(
    |a: u16x16, _b| u32x16::from(a),
    |a, _b| a as u32,
  );
}

#[test]
fn test_u32x16_any() {
  let a = u32x16::from([
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
    0,
    0,
    0,
  ]);
  assert!(a.any());
  //
  let a = u32x16::from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert!(!a.any());
}

#[test]
fn test_u32x16_all() {
  let a = u32x16::from([
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
    0,
    0,
    0,
  ]);
  assert!(!a.all());
  //
  let a = u32x16::from([u32::MAX; 16]);
  assert!(a.all());
}

#[test]
fn test_u32x16_none() {
  let a = u32x16::from([
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
    0,
    0,
    0,
  ]);
  assert!(!a.none());
  //
  let a = u32x16::from([0; 16]);
  assert!(a.none());
}

#[test]
fn impl_u32x16_mul_keep_high() {
  crate::test_random_vector_vs_scalar(
    |a: u32x16, b| u32x16::mul_keep_high(a, b),
    |a, b| ((u64::from(a) * u64::from(b)) >> 32) as u32,
  );
}

#[cfg(feature = "serde")]
#[test]
fn impl_u32x16_ser_de_roundtrip() {
  let serialized =
    bincode::serialize(&u32x16::ZERO).expect("serialization failed");
  let deserialized =
    bincode::deserialize(&serialized).expect("deserializaion failed");
  assert_eq!(u32x16::ZERO, deserialized);
}
