use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<i32x16>(), 64);
  assert_eq!(core::mem::align_of::<i32x16>(), 64);
}

crate::generate_basic_traits_test!(i32x16, i32);

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
fn impl_cmp_ne_for_i32x16() {
  let a = i32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  let b = i32x16::from([0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]);

  assert_eq!(a.simd_ne(b), !a.simd_eq(b));
}

#[test]
fn impl_cmp_ge_for_i32x16() {
  let a = i32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  let b = i32x16::from([0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]);

  assert_eq!(a.simd_ge(b), !a.simd_lt(b));
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
fn impl_cmp_le_for_i32x16() {
  let a = i32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  let b = i32x16::from([0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]);

  assert_eq!(a.simd_le(b), !a.simd_gt(b));
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
fn impl_i32x16_is_negative() {
  let value =
    i32x16::new([1, -1, 2, 3, -2, -5, 0, 6, 9, 1, -2, -3, -4, 0, -1, 1]);
  let expected =
    i32x16::new([0, -1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, -1, 0]);
  let actual = value.is_negative();
  assert_eq!(expected, actual);
}

#[test]
fn impl_abs_for_i32x16() {
  let a = i32x16::from([
    -1,
    2,
    -3,
    i32::MIN,
    6,
    -15,
    -19,
    9,
    -1,
    2,
    -3,
    i32::MIN,
    6,
    -15,
    -19,
    9,
  ]);
  let expected = i32x16::from([
    1,
    2,
    3,
    i32::MIN,
    6,
    15,
    19,
    9,
    1,
    2,
    3,
    i32::MIN,
    6,
    15,
    19,
    9,
  ]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_unsigned_abs_for_i32x16() {
  let a = i32x16::from([
    -1,
    2,
    -3,
    i32::MIN,
    6,
    -15,
    -19,
    9,
    -1,
    2,
    -3,
    i32::MIN,
    6,
    -15,
    -19,
    9,
  ]);
  let expected = u32x16::from([
    1,
    2,
    3,
    i32::MIN as u32,
    6,
    15,
    19,
    9,
    1,
    2,
    3,
    i32::MIN as u32,
    6,
    15,
    19,
    9,
  ]);
  let actual = a.unsigned_abs();
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
fn impl_i32x16_saturating_add() {
  for (value, rhs) in [
    (1, 2),
    (10, 20),
    (15, -10),
    (15, -20),
    (-15, 20),
    (-15, 10),
    (-15, -16),
    (0, 15),
    (0, -15),
    (15, 0),
    (-15, 0),
    (5, i32::MAX - 1),
    (-5, i32::MIN + 1),
    (i32::MAX - 1, 5),
    (i32::MIN + 1, -5),
    (0, i32::MAX),
    (0, i32::MIN),
    (i32::MAX, 0),
    (i32::MIN, 0),
  ] {
    let expected = i32x16::splat(value.saturating_add(rhs));
    let actual = i32x16::splat(value).saturating_add(i32x16::splat(rhs));
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_i32x16_saturating_sub() {
  for (value, rhs) in [
    (1, 2),
    (10, 20),
    (15, -10),
    (15, -20),
    (-15, 20),
    (-15, 10),
    (-15, -16),
    (0, 15),
    (0, -15),
    (15, 0),
    (-15, 0),
    (5, i32::MAX - 1),
    (-5, i32::MIN + 1),
    (i32::MAX - 1, 5),
    (i32::MIN + 1, -5),
    (0, i32::MAX),
    (0, i32::MIN),
    (i32::MAX, 0),
    (i32::MIN, 0),
  ] {
    let expected = i32x16::splat(value.saturating_sub(rhs));
    let actual = i32x16::splat(value).saturating_sub(i32x16::splat(rhs));
    assert_eq!(expected, actual);
  }
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

#[test]
fn test_i32x4_move_mask() {
  let a = i32x16::from([
    -1, 0, -2, -3,
    -1, 0, -2, -3,
    -1, 0, -2, -3,
    -1, 0, -2, -3,
  ]);
  let expected = 0b1101110111011101;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);
  //
  let a = i32x16::from([
    i32::MAX, 0, 2, -3,
    i32::MAX, 0, 2, -3,
    i32::MAX, 0, 2, -3,
    i32::MAX, 0, 2, -3,
  ]);
  let expected = 0b1000100010001000;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar_reduce(
    |a: i32x16| a.to_bitmask(),
    0_u32,
    |acc, a, idx| acc | if a < 0 { 1 << idx } else { 0 },
  );
}

#[test]
fn test_i32x16_any() {
  assert!(!i32x16::splat(0).any());
  assert!(i32x16::splat(!0).any());
  for i in 0..16 {
    let mut a = i32x16::splat(0);
    a.as_mut_array()[i] = !0;
    assert!(a.any());
  }
}

#[test]
fn test_i32x16_all() {
  assert!(!i32x16::splat(0).all());
  assert!(i32x16::splat(!0).all());
  for i in 0..16 {
    let mut a = i32x16::splat(!0);
    a.as_mut_array()[i] = 0;
    assert!(!a.all());
  }
}

#[test]
fn test_i32x16_none() {
  assert!(i32x16::splat(0).none());
  assert!(!i32x16::splat(!0).none());
  for i in 0..16 {
    let mut a = i32x16::splat(0);
    a.as_mut_array()[i] = !0;
    assert!(!a.none());
  }
}

#[test]
fn impl_i32x16_reduce_add() {
  let p = i32x16::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  ]);
  assert_eq!(p.reduce_add(), 136);
}

#[test]
fn impl_i32x16_reduce_min() {
  for i in 0..16 {
    let mut v = [i32::MAX; 16];
    v[i] = i32::MIN;
    let p = i32x16::from(v);
    assert_eq!(p.reduce_min(), i32::MIN);
  }
}

#[test]
fn impl_i32x16_reduce_max() {
  for i in 0..16 {
    let mut v = [i32::MIN; 16];
    v[i] = i32::MAX;
    let p = i32x16::from(v);
    assert_eq!(p.reduce_max(), i32::MAX);
  }
}

#[test]
fn impl_i32x16_transpose() {
  let data = std::array::from_fn(|i| {
    i32x16::new(std::array::from_fn(|j| (i * 100 + j) as i32))
  });
  let expected = std::array::from_fn(|i| {
    i32x16::new(std::array::from_fn(|j| (j * 100 + i) as i32))
  });
  let actual = i32x16::transpose(data);
  assert_eq!(expected, actual);
}
