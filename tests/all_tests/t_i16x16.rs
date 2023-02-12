use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i16x16>(), 32);
  assert_eq!(core::mem::align_of::<i16x16>(), 32);
}

#[test]
fn impl_add_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    15,
    20,
    5000,
    2990,
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    15,
    20,
    5000,
    2990,
  ]);
  let b = i16x16::from([
    17, 18, 1, 2, 20, 5, 900, 900, 17, 18, 1, 2, 20, 5, 900, 900,
  ]);
  let expected = i16x16::from([
    18,
    20,
    i16::MAX,
    i16::MIN,
    35,
    25,
    5900,
    3890,
    18,
    20,
    i16::MAX,
    i16::MIN,
    35,
    25,
    5900,
    3890,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    15,
    20,
    5000,
    2990,
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    15,
    20,
    5000,
    2990,
  ]);
  let b = i16x16::from([
    17, -18, 1, 1, 20, 5, 900, 900, 17, -18, 1, 1, 20, 5, 900, 900,
  ]);
  let expected = i16x16::from([
    -16,
    20,
    i16::MIN,
    i16::MAX,
    -5,
    15,
    4100,
    2090,
    -16,
    20,
    i16::MIN,
    i16::MAX,
    -5,
    15,
    4100,
    2090,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_add_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    15,
    20,
    5000,
    2990,
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    15,
    20,
    5000,
    2990,
  ]);
  let b = i16x16::from([
    17, 18, 1, 2, 20, 5, 900, 900, 17, 18, 1, 2, 20, 5, 900, 900,
  ]);
  let expected = i16x16::from([
    18,
    20,
    i16::MAX,
    i16::MAX,
    35,
    25,
    5900,
    3890,
    18,
    20,
    i16::MAX,
    i16::MAX,
    35,
    25,
    5900,
    3890,
  ]);
  let actual = a.saturating_add(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_sub_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    15,
    20,
    5000,
    2990,
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    15,
    20,
    5000,
    2990,
  ]);
  let b = i16x16::from([
    17, -18, 1, 1, 20, 5, 900, 900, 17, -18, 1, 1, 20, 5, 900, 900,
  ]);
  let expected = i16x16::from([
    -16,
    20,
    i16::MIN,
    i16::MIN,
    -5,
    15,
    4100,
    2090,
    -16,
    20,
    i16::MIN,
    i16::MIN,
    -5,
    15,
    4100,
    2090,
  ]);
  let actual = a.saturating_sub(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_scale_i16x16() {
  let a = i16x16::from([
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
    1400, 1500,
  ]);
  let b = i16x16::from([
    0, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2100, 2200, 2300,
  ]);
  let actual = a.mul_scale_round(b);
  let expected = i16x16::from([
    0, 3, 6, 10, 15, 20, 26, 32, 39, 47, 55, 64, 73, 83, 94, 105,
  ]);
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    2,
    3,
    4,
    5,
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    2,
    3,
    4,
    5,
  ]);
  let b =
    i16x16::from([17, -18, 1, 1, -1, -2, -6, 3, 17, -18, 1, 1, -1, -2, -6, 3]);
  let expected = i16x16::from([
    17,
    -36,
    i16::MIN + 1,
    i16::MIN,
    -2,
    -6,
    -24,
    15,
    17,
    -36,
    i16::MIN + 1,
    i16::MIN,
    -2,
    -6,
    -24,
    15,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i16x16() {
  let a = i16x16::from([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  let b = i16x16::from([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = i16x16::from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i16x16() {
  let a = i16x16::from([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  let b = i16x16::from([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = i16x16::from([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i16x16() {
  let a = i16x16::from([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  let b = i16x16::from([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = i16x16::from([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    128,
    255,
    590,
    5667,
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    128,
    255,
    590,
    5667,
  ]);
  let b = 2;
  let expected = i16x16::from([
    1 << 2,
    2 << 2,
    (i16::MAX - 1) << 2,
    (i16::MAX - 1) << 2,
    128 << 2,
    255 << 2,
    590 << 2,
    5667 << 2,
    1 << 2,
    2 << 2,
    (i16::MAX - 1) << 2,
    (i16::MAX - 1) << 2,
    128 << 2,
    255 << 2,
    590 << 2,
    5667 << 2,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i16x16() {
  let a = i16x16::from([
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    128,
    255,
    590,
    5667,
    1,
    2,
    i16::MAX - 1,
    i16::MAX - 1,
    128,
    255,
    590,
    5667,
  ]);
  let b = 2;
  let expected = i16x16::from([
    1 >> 2,
    2 >> 2,
    (i16::MAX - 1) >> 2,
    (i16::MAX - 1) >> 2,
    128 >> 2,
    255 >> 2,
    590 >> 2,
    5667 >> 2,
    1 >> 2,
    2 >> 2,
    (i16::MAX - 1) >> 2,
    (i16::MAX - 1) >> 2,
    128 >> 2,
    255 >> 2,
    590 >> 2,
    5667 >> 2,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_cmp_eq() {
  let a = i16x16::from([1, 2, 3, 4, 2, 1, 8, 2, 1, 2, 3, 4, 2, 1, 8, 2]);
  let b = i16x16::from([2_i16; 16]);
  let expected =
    i16x16::from([0, -1, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_cmp_gt() {
  let a = i16x16::from([1, 2, 9, 4, 1, 2, 8, 10, 1, 2, 9, 4, 1, 2, 8, 10]);
  let b = i16x16::from([5_i16; 16]);
  let expected =
    i16x16::from([0, 0, -1, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, -1, -1]);
  let actual = a.cmp_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_cmp_lt() {
  let a = i16x16::from([1, 2, 9, 4, 1, 2, 8, 10, 1, 2, 9, 4, 1, 2, 8, 10]);
  let b = i16x16::from([5_i16; 16]);
  let expected =
    i16x16::from([-1, -1, 0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0]);
  let actual = a.cmp_lt(b);
  assert_eq!(expected, actual);

  let expected = i16x16::from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  let actual = a.cmp_lt(a);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_blend() {
  let use_t: i16 = -1;
  let t = i16x16::from([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
  let f = i16x16::from([
    17, 18, 19, 20, 25, 30, 50, 90, 17, 18, 19, 20, 25, 30, 50, 90,
  ]);
  let mask = i16x16::from([
    use_t, 0, use_t, 0, 0, 0, 0, use_t, use_t, 0, use_t, 0, 0, 0, 0, use_t,
  ]);
  let expected =
    i16x16::from([1, 18, 3, 20, 25, 30, 50, 8, 1, 18, 3, 20, 25, 30, 50, 8]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_abs() {
  let a = i16x16::from([
    -1,
    2,
    -3,
    i16::MIN,
    6,
    -15,
    -19,
    9,
    -1,
    2,
    -3,
    i16::MIN,
    6,
    -15,
    -19,
    9,
  ]);
  let expected = i16x16::from([
    1,
    2,
    3,
    i16::MIN,
    6,
    15,
    19,
    9,
    1,
    2,
    3,
    i16::MIN,
    6,
    15,
    19,
    9,
  ]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_max() {
  let a = i16x16::from([
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    6,
    -8,
    12,
    9,
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    6,
    -8,
    12,
    9,
  ]);
  let b = i16x16::from([
    17, -18, 1, 1, 19, -5, -1, -9, 17, -18, 1, 1, 19, -5, -1, -9,
  ]);
  let expected =
    i16x16::from([17, 2, 1, 1, 19, -5, 12, 9, 17, 2, 1, 1, 19, -5, 12, 9]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x16_min() {
  let a = i16x16::from([
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    6,
    -8,
    12,
    9,
    1,
    2,
    i16::MIN + 1,
    i16::MIN,
    6,
    -8,
    12,
    9,
  ]);
  let b = i16x16::from([
    17, -18, 1, 1, 19, -5, -1, -9, 17, -18, 1, 1, 19, -5, -1, -9,
  ]);
  let expected = i16x16::from([
    1,
    -18,
    i16::MIN + 1,
    i16::MIN,
    6,
    -8,
    -1,
    -9,
    1,
    -18,
    i16::MIN + 1,
    i16::MIN,
    6,
    -8,
    -1,
    -9,
  ]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}
