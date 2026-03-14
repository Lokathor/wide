use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i16x32>(), 64);
  assert_eq!(core::mem::align_of::<i16x32>(), 64);
}

crate::generate_basic_traits_test!(i16x32, i16);

#[test]
fn impl_add_for_i16x32() {
  let a = i16x32::from([
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
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    i16::MAX - 1,
  ]);
  let b = i16x32::from([
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 1,
  ]);
  let expected = i16x32::from([
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
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    62,
    64,
    66,
    68,
    70,
    72,
    74,
    76,
    78,
    i16::MAX,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_i16x32() {
  let a = i16x32::from([
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
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    i16::MIN + 1,
    30,
    31,
    32,
  ]);
  let b = i16x32::from([
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 46, 47, 48,
  ]);
  let expected = i16x32::from([
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    -16,
    i16::MIN,
    -16,
    -16,
    -16,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_i16x32() {
  let a = i16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5,
    -6, -7, -8, 100, 200, 300, 400, 500, 600, 700, 800,
  ]);
  let b = i16x32::from([
    2, 2, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, 2, 2, 2, 2, 2, 2,
    2, 2, 10, 10, 10, 10, 10, 10, 10, 10,
  ]);
  let expected = i16x32::from([
    2, 4, 6, 8, 10, 12, 14, 16, -18, -20, -22, -24, -26, -28, -30, -32, -2, -4,
    -6, -8, -10, -12, -14, -16, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_i16x32() {
  let a = i16x32::from([0; 32]);
  let b = i16x32::from([0; 32]);
  let expected = i16x32::from([0; 32]);
  let actual = a & b;
  assert_eq!(expected, actual);

  let a = i16x32::from([!0; 32]);
  let b = i16x32::from([0; 32]);
  let expected = i16x32::from([0; 32]);
  let actual = a & b;
  assert_eq!(expected, actual);

  let a = i16x32::from([!0; 32]);
  let b = i16x32::from([!0; 32]);
  let expected = i16x32::from([!0; 32]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_i16x32() {
  let a = i16x32::from([0; 32]);
  let b = i16x32::from([0; 32]);
  let expected = i16x32::from([0; 32]);
  let actual = a | b;
  assert_eq!(expected, actual);

  let a = i16x32::from([!0; 32]);
  let b = i16x32::from([0; 32]);
  let expected = i16x32::from([!0; 32]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_i16x32() {
  let a = i16x32::from([0; 32]);
  let b = i16x32::from([0; 32]);
  let expected = i16x32::from([0; 32]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  let a = i16x32::from([!0; 32]);
  let b = i16x32::from([0; 32]);
  let expected = i16x32::from([!0; 32]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  let a = i16x32::from([!0; 32]);
  let b = i16x32::from([!0; 32]);
  let expected = i16x32::from([0; 32]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_i16x32() {
  let a = i16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5,
    -6, -7, -8, 255, 256, 257, 258, 259, 260, 261, 262,
  ]);
  let b = 1;
  let expected = i16x32::from([
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, -2, -4, -6, -8,
    -10, -12, -14, -16, 510, 512, 514, 516, 518, 520, 522, 524,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_i16x32() {
  let a = i16x32::from([
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, -2, -4, -6, -8,
    -10, -12, -14, -16, 510, 512, 514, 516, 518, 520, 522, 524,
  ]);
  let b = 1;
  let expected = i16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5,
    -6, -7, -8, 255, 256, 257, 258, 259, 260, 261, 262,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_eq_for_i16x32() {
  let a = i16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);
  let b = i16x32::from([
    0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20,
    20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30,
  ]);
  let expected = i16x32::from([
    -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
    0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
  ]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_gt_for_i16x32() {
  let a = i16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -15, -14, -13,
    -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
  ]);
  let b = i16x32::from([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
  ]);
  let expected = i16x32::from([
    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_lt_for_i16x32() {
  let a = i16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -15, -14, -13,
    -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
  ]);
  let b = i16x32::from([100; 32]);
  let expected = i16x32::from([-1; 32]);
  let actual = a.simd_lt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_blend_for_i16x32() {
  let use_t: i16x32 = i16x32::from([
    0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
    -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
  ]);
  let t = i16x32::from([1; 32]);
  let f = i16x32::from([0; 32]);
  let expected = i16x32::from([
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1,
  ]);
  let actual = use_t.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_min_for_i16x32() {
  let a = i16x32::from([
    0,
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
    13,
    14,
    15,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -10,
    -9,
    i16::MIN,
    i16::MAX,
    100,
    -100,
    0,
    0,
    0,
    0,
  ]);
  let b = i16x32::from([
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    0,
    -1,
    -2,
    -3,
    -4,
    -5,
    -6,
    -7,
    -8,
    i16::MAX,
    i16::MIN,
    -100,
    100,
    1,
    -1,
    0,
    0,
  ]);
  let expected = i16x32::from([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    0,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -10,
    -9,
    i16::MIN,
    i16::MIN,
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
fn impl_max_for_i16x32() {
  let a = i16x32::from([
    0,
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
    13,
    14,
    15,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -10,
    -9,
    i16::MIN,
    i16::MAX,
    100,
    -100,
    0,
    0,
    0,
    0,
  ]);
  let b = i16x32::from([
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    0,
    -1,
    -2,
    -3,
    -4,
    -5,
    -6,
    -7,
    -8,
    i16::MAX,
    i16::MIN,
    -100,
    100,
    1,
    -1,
    0,
    0,
  ]);
  let expected = i16x32::from([
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    -1,
    -2,
    -3,
    -4,
    -5,
    -6,
    -7,
    -8,
    i16::MAX,
    i16::MAX,
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
fn impl_saturating_add_for_i16x32() {
  let a = i16x32::from([
    i16::MAX,
    i16::MAX - 1,
    i16::MAX - 2,
    100,
    -100,
    i16::MIN + 2,
    i16::MIN + 1,
    i16::MIN,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
  ]);
  let b = i16x32::from([
    1, 2, 3, 200, -200, -3, -2, -1, 100, 100, 100, 100, 100, 100, 100, 100,
    -100, -100, -100, -100, -100, -100, -100, -100, 10000, 10000, 10000, 10000,
    10000, 10000, 10000, 10000,
  ]);
  let expected = i16x32::from([
    i16::MAX,
    i16::MAX,
    i16::MAX,
    300,
    -300,
    i16::MIN,
    i16::MIN,
    i16::MIN,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    -108,
    -107,
    -106,
    -105,
    -104,
    -103,
    -102,
    -101,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
  ]);
  let actual = a.saturating_add(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_sub_for_i16x32() {
  let a = i16x32::from([
    i16::MIN,
    i16::MIN + 1,
    i16::MIN + 2,
    100,
    -100,
    i16::MAX - 2,
    i16::MAX - 1,
    i16::MAX,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
  ]);
  let b = i16x32::from([
    1, 2, 3, -200, 200, -3, -2, -1, 100, 100, 100, 100, 100, 100, 100, 100,
    -100, -100, -100, -100, -100, -100, -100, -100, -10000, -10000, -10000,
    -10000, -10000, -10000, -10000, -10000,
  ]);
  let expected = i16x32::from([
    i16::MIN,
    i16::MIN,
    i16::MIN,
    300,
    -300,
    i16::MAX,
    i16::MAX,
    i16::MAX,
    -100,
    -99,
    -98,
    -97,
    -96,
    -95,
    -94,
    -93,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
  ]);
  let actual = a.saturating_sub(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x32_new() {
  let a = i16x32::new([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);
  let expected = [
    0i16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}

#[test]
fn test_i16x32_move_mask() {
  let a = i16x32::from([
    -1, 0, -2, -3, -1, 0, -2, -3,
    -1, 0, -2, -3, -1, 0, -2, -3,
    -1, 0, -2, -3, -1, 0, -2, -3,
    -1, 0, -2, -3, -1, 0, -2, -3,
  ]);
  let expected = 0b11011101110111011101110111011101;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);
  
  let a = i16x32::from([
    1, 0, 2, -3, 1, 0, 2, -3,
    1, 0, 2, -3, 1, 0, 2, -3,
    1, 0, 2, -3, 1, 0, 2, -3,
    1, 0, 2, -3, 1, 0, 2, -3,
  ]);
  let expected = 0b10001000100010001000100010001000;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);
}

#[test]
fn impl_dot_for_i16x32() {
  let a = i16x32::from([
    1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN,
    10, 20, 30, 40, 50, 60, i16::MAX - 1, i16::MAX,
    1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN,
    10, 20, 30, 40, 50, 60, i16::MAX - 1, i16::MAX,
  ]);
  let b = i16x32::from([
    17, -18, 190, -20, 21, -22, 3, 2,
    170, -180, 1900, -200, 210, -220, 30, 20,
    17, -18, 190, -20, 21, -22, 3, 2,
    170, -180, 1900, -200, 210, -220, 30, 20,
  ]);
  let expected = i32x16::from([
    -19, 490, -27, -163837,
    -1900, 49000, -2700, 1638320,
    -19, 490, -27, -163837,
    -1900, 49000, -2700, 1638320,
  ]);
  let actual = a.dot(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_i16x32_reduce_add() {
  let p = i16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
  ]);
  assert_eq!(p.reduce_add(), 528);
}

#[test]
fn impl_i16x32_reduce_min() {
  for i in 0..32 {
    let mut v = [i16::MAX; 32];
    v[i] = i16::MIN;
    let p = i16x32::from(v);
    assert_eq!(p.reduce_min(), i16::MIN);
  }
}

#[test]
fn impl_i16x32_reduce_max() {
  for i in 0..32 {
    let mut v = [i16::MIN; 32];
    v[i] = i16::MAX;
    let p = i16x32::from(v);
    assert_eq!(p.reduce_max(), i16::MAX);
  }
}