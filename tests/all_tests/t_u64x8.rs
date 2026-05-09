use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u64x8>(), 64);
  assert_eq!(core::mem::align_of::<u64x8>(), 64);
}

crate::generate_basic_traits_test!(u64x8, u64);

#[test]
fn impl_add_for_u64x8() {
  let a = u64x8::from([0, 1, 2, 3, 4, 5, 6, u64::MAX - 1]);
  let b = u64x8::from([17, 18, 19, 20, 21, 22, 23, 1]);
  let expected = u64x8::from([17, 19, 21, 23, 25, 27, 29, u64::MAX]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u64x8() {
  let a = u64x8::from([17, 18, 19, 20, 21, 22, 23, 1]);
  let b = u64x8::from([17, 18, 19, 0, 2, 23, 24, 1]);
  let expected = u64x8::from([0, 0, 0, 20, 19, u64::MAX, u64::MAX, 0]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_u64x8() {
  let a = u64x8::from([17, 18, 19, 20, 21, 22, 1 << 32, 24]);
  let b = a;
  let expected = u64x8::from([
    17 * 17,
    18 * 18,
    19 * 19,
    20 * 20,
    21 * 21,
    22 * 22,
    0,
    24 * 24,
  ]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_u64x8() {
  let a = u64x8::from([0, 0, 1, 1, 1, 0, 0, 1]);
  let b = u64x8::from([0, 1, 0, 1, 0, 1, 1, 1]);
  let expected = u64x8::from([0, 0, 0, 1, 0, 0, 0, 1]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u64x8() {
  let a = u64x8::from([0, 0, 1, 1, 0, 1, 1, 0]);
  let b = u64x8::from([0, 1, 0, 1, 1, 0, 1, 0]);
  let expected = u64x8::from([0, 1, 1, 1, 1, 1, 1, 0]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u64x8() {
  let a = u64x8::from([0, 0, 1, 1, 0, 1, 1, 0]);
  let b = u64x8::from([0, 1, 0, 1, 1, 0, 1, 0]);
  let expected = u64x8::from([0, 1, 1, 0, 1, 1, 0, 0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_u64x8() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, u64::MAX]);
  let b = 1;
  let expected = u64x8::from([2, 4, 6, 8, 10, 12, 14, 0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1110]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_u64x8() {
  let a = u64x8::from([2, 4, 6, 8, 10, 12, 14, u64::MAX]);
  let b = 1;
  let expected = u64x8::from([1, 2, 3, 4, 5, 6, 7, 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_cmp_eq() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let b = u64x8::from([2_u64; 8]);
  let expected = u64x8::from([0, u64::MAX, 0, 0, 0, 0, 0, 0]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_cmp_ne() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let b = u64x8::from([2_u64; 8]);

  assert_eq!(a.simd_ne(b), !a.simd_eq(b));
}

#[test]
fn impl_u64x8_cmp_ge() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let b = u64x8::from([2_u64; 8]);

  assert_eq!(a.simd_ge(b), !a.simd_lt(b));
}

#[test]
fn impl_u64x8_cmp_gt() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let b = u64x8::from([2_u64; 8]);
  let expected = u64x8::from([
    0,
    0,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
  ]);
  let actual = a.simd_gt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_cmp_le() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let b = u64x8::from([2_u64; 8]);

  assert_eq!(a.simd_le(b), !a.simd_gt(b));
}

#[test]
fn impl_u64x8_cmp_lt() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let b = u64x8::from([2_u64; 8]);
  let expected = u64x8::from([u64::MAX, 0, 0, 0, 0, 0, 0, 0]);
  let actual = a.simd_lt(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_blend() {
  let use_t = u64x8::from([0, u64::MAX, 0, u64::MAX, 0, u64::MAX, 0, u64::MAX]);
  let t = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let f = u64x8::from([9, 10, 11, 12, 13, 14, 15, 16]);
  let expected = u64x8::from([9, 2, 11, 4, 13, 6, 15, 8]);
  let actual = use_t.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_reduce_add() {
  let value = u64x8::new([1, 2, 3, 5, 7, 11, 13, 17]);
  let expected = 59;
  let actual = value.reduce_add();
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_reduce_max() {
  for i in 0..8 {
    let mut value = u64x8::new([9, 10, 5, 1, 3, 4, 5, 6]);
    value.as_mut_array()[i] = u64::MAX - 1;

    let expected = u64::MAX - 1;
    let actual = value.reduce_max();
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_u64x8_reduce_min() {
  for i in 0..8 {
    let mut value = u64x8::new([9, u64::MAX - 1, 5, 6, u64::MAX - 1, 5, 5, 6]);
    value.as_mut_array()[i] = 1;

    let expected = 1;
    let actual = value.reduce_min();
    assert_eq!(expected, actual);
  }
}

#[test]
fn test_u64x8_any() {
  assert!(!u64x8::splat(0).any());
  assert!(u64x8::splat(!0).any());
  for i in 0..8 {
    let mut a = u64x8::splat(0);
    a.as_mut_array()[i] = !0;
    assert!(a.any());
  }
}

#[test]
fn test_u64x8_all() {
  assert!(!u64x8::splat(0).all());
  assert!(u64x8::splat(!0).all());
  for i in 0..8 {
    let mut a = u64x8::splat(!0);
    a.as_mut_array()[i] = 0;
    assert!(!a.all());
  }
}

#[test]
fn test_u64x8_none() {
  assert!(u64x8::splat(0).none());
  assert!(!u64x8::splat(!0).none());
  for i in 0..8 {
    let mut a = u64x8::splat(0);
    a.as_mut_array()[i] = !0;
    assert!(!a.none());
  }
}

#[test]
fn impl_u64x8_transpose() {
  let data = std::array::from_fn(|i| {
    u64x8::new(std::array::from_fn(|j| (i * 100 + j) as u64))
  });
  let expected = std::array::from_fn(|i| {
    u64x8::new(std::array::from_fn(|j| (j * 100 + i) as u64))
  });
  let actual = u64x8::transpose(data);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_to_array() {
  let a = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let expected = [1, 2, 3, 4, 5, 6, 7, 8];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x8_new() {
  let a = u64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
  let expected = [1, 2, 3, 4, 5, 6, 7, 8];
  let actual = a.to_array();
  assert_eq!(expected, actual);
}
