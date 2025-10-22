use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u16x32>(), 64);
  assert_eq!(core::mem::align_of::<u16x32>(), 64);
}

crate::generate_basic_traits_test!(u16x32, u16);

#[test]
fn impl_add_for_u16x32() {
  let a = u16x32::from([
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
    u16::MAX - 1,
  ]);
  let b = u16x32::from([
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 1,
  ]);
  let expected = u16x32::from([
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    u16::MAX,
  ]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u16x32() {
  let a = u16x32::from([
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 1,
  ]);
  let b = u16x32::from([
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 1,
  ]);
  let expected = u16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0,
  ]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_u16x32() {
  let a = u16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 100, 200, 300, 400, 500, 600, 700, 800,
  ]);
  let b = u16x32::from([
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100, 100, 100,
  ]);
  let expected = u16x32::from([
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
    180, 190, 200, 210, 220, 230, 240, 10000, 20000, 30000, 40000, 50000,
    60000, 4464, 14464,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_u16x32() {
  let a = u16x32::from([u16::MAX; 32]);
  let b = u16x32::from([0; 32]);
  let expected = u16x32::from([0; 32]);
  let actual = a & b;
  assert_eq!(expected, actual);

  let a = u16x32::from([u16::MAX; 32]);
  let b = u16x32::from([u16::MAX; 32]);
  let expected = u16x32::from([u16::MAX; 32]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u16x32() {
  let a = u16x32::from([u16::MAX; 32]);
  let b = u16x32::from([0; 32]);
  let expected = u16x32::from([u16::MAX; 32]);
  let actual = a | b;
  assert_eq!(expected, actual);

  let a = u16x32::from([0; 32]);
  let b = u16x32::from([0; 32]);
  let expected = u16x32::from([0; 32]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u16x32() {
  let a = u16x32::from([u16::MAX; 32]);
  let b = u16x32::from([u16::MAX; 32]);
  let expected = u16x32::from([0; 32]);
  let actual = a ^ b;
  assert_eq!(expected, actual);

  let a = u16x32::from([u16::MAX; 32]);
  let b = u16x32::from([0; 32]);
  let expected = u16x32::from([u16::MAX; 32]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_u16x32() {
  let a = u16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 255, 256, 257, 258, 259, 260, 261, 262,
  ]);
  let b = 1;
  let expected = u16x32::from([
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    42, 44, 46, 48, 510, 512, 514, 516, 518, 520, 522, 524,
  ]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_u16x32() {
  let a = u16x32::from([
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    42, 44, 46, 48, 510, 512, 514, 516, 518, 520, 522, 524,
  ]);
  let b = 1;
  let expected = u16x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 255, 256, 257, 258, 259, 260, 261, 262,
  ]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_eq_for_u16x32() {
  let a = u16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);
  let b = u16x32::from([
    0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20,
    20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30,
  ]);
  let expected = u16x32::from([
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
  ]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_gt_for_u16x32() {
  let a = u16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);
  let b = u16x32::from([
    100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0,
    100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0,
  ]);
  let expected = u16x32::from([
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
  ]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_cmp_lt_for_u16x32() {
  let a = u16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);
  let b = u16x32::from([100; 32]);
  let expected = u16x32::from([u16::MAX; 32]);
  let actual = a.simd_lt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_blend_for_u16x32() {
  let use_t: u16x32 = u16x32::from([
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
    0,
    u16::MAX,
  ]);
  let t = u16x32::from([1; 32]);
  let f = u16x32::from([0; 32]);
  let expected = u16x32::from([
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1,
  ]);
  let actual = use_t.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_min_for_u16x32() {
  let a = u16x32::from([
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
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    u16::MAX - 3,
    0,
    0,
    0,
    0,
  ]);
  let b = u16x32::from([
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
    800,
    700,
    600,
    500,
    400,
    300,
    200,
    100,
    0,
    1,
    2,
    3,
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    u16::MAX - 3,
  ]);
  let expected = u16x32::from([
    0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0, 100, 200, 300, 400, 400,
    300, 200, 100, 0, 1, 2, 3, 0, 0, 0, 0,
  ]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_max_for_u16x32() {
  let a = u16x32::from([
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
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    u16::MAX - 3,
    0,
    0,
    0,
    0,
  ]);
  let b = u16x32::from([
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
    800,
    700,
    600,
    500,
    400,
    300,
    200,
    100,
    0,
    1,
    2,
    3,
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    u16::MAX - 3,
  ]);
  let expected = u16x32::from([
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
    800,
    700,
    600,
    500,
    500,
    600,
    700,
    800,
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    u16::MAX - 3,
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    u16::MAX - 3,
  ]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_add_for_u16x32() {
  let a = u16x32::from([
    u16::MAX,
    u16::MAX - 1,
    u16::MAX - 2,
    100,
    200,
    0,
    1,
    2,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    20000,
    30000,
    40000,
    50000,
    60000,
    65000,
    65100,
    65200,
    65300,
    65400,
    65500,
    65510,
    65520,
    65530,
    65535,
  ]);
  let b = u16x32::from([
    1, 2, 3, 200, 100, 100, 100, 100, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    1000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000,
    1000, 1000, 100, 100, 100, 100,
  ]);
  let expected = u16x32::from([
    u16::MAX,
    u16::MAX,
    u16::MAX,
    300,
    300,
    100,
    101,
    102,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    20000,
    30000,
    40000,
    50000,
    60000,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
    u16::MAX,
  ]);
  let actual = a.saturating_add(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_saturating_sub_for_u16x32() {
  let a = u16x32::from([
    0, 1, 2, 300, 200, 100, 101, 102, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
    8000, 10000, 20000, 30000, 40000, 50000, 60000, 65000, 65100, 65200, 65300,
    65400, 65500, 65510, 65520, 65530, 65535,
  ]);
  let b = u16x32::from([
    1, 2, 3, 200, 100, 100, 100, 100, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    1000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000,
    1000, 1000, 100, 100, 100, 100,
  ]);
  let expected = u16x32::from([
    0, 0, 0, 100, 100, 0, 1, 2, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 0,
    10000, 20000, 30000, 40000, 50000, 64000, 64100, 64200, 64300, 64400,
    64500, 65410, 65420, 65430, 65435,
  ]);
  let actual = a.saturating_sub(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u16x32_new() {
  let a = u16x32::new([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ]);
  let expected = [
    0u16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  ];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}
