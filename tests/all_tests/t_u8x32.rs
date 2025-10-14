use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u8x32>(), 32);
  assert_eq!(core::mem::align_of::<u8x32>(), 32);
}

#[test]
fn basic_traits() {
  type T = u8x32;
  use crate::TestBasicTraits;

  T::test_basic_traits_int();
  T::test_basic_traits_aligned_to();
}

#[test]
fn impl_add_for_u8x32() {
  crate::test_random_vector_vs_scalar(
    |a: u8x32, b| a + b,
    |a, b| a.wrapping_add(b),
  );
}

#[test]
fn impl_sub_for_u8x32() {
  crate::test_random_vector_vs_scalar(
    |a: u8x32, b| a - b,
    |a, b| a.wrapping_sub(b),
  );
}

#[test]
fn impl_saturating_add_for_u8x32() {
  crate::test_random_vector_vs_scalar(
    |a: u8x32, b| a.saturating_add(b),
    |a, b| a.saturating_add(b),
  );
}

#[test]
fn impl_saturating_sub_for_u8x32() {
  crate::test_random_vector_vs_scalar(
    |a: u8x32, b| a.saturating_sub(b),
    |a, b| a.saturating_sub(b),
  );
}

#[test]
fn impl_bitand_for_u8x32() {
  let a = u8x32::from([
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1,
  ]);
  let b = u8x32::from([
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1,
  ]);
  let expected = u8x32::from([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1,
  ]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u8x32() {
  let a = u8x32::from([
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1,
  ]);
  let b = u8x32::from([
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1,
  ]);
  let expected = u8x32::from([
    0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 1, 1, 0, 1, 1, 1,
  ]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u8x32() {
  let a = u8x32::from([
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1,
  ]);
  let b = u8x32::from([
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1,
  ]);
  let expected = u8x32::from([
    0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
  ]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_u8x32_cmp_eq() {
  let a = u8x32::from([
    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
    2, 3, 4, 1, 2, 3, 4,
  ]);
  let b = u8x32::from([2_u8; 32]);
  let expected = u8x32::from([
    0, 0xff, 0, 0, 0, 0xff, 0, 0, 0, 0xff, 0, 0, 0, 0xff, 0, 0, 0, 0xff, 0, 0,
    0, 0xff, 0, 0, 0, 0xff, 0, 0, 0, 0xff, 0, 0,
  ]);
  let actual = a.simd_eq(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u8x32_not() {
  let a = u8x32::from([
    233, 90, 206, 251, 179, 93, 136, 194, 135, 57, 6, 243, 234, 196, 243, 49,
    44, 116, 195, 174, 208, 190, 94, 155, 233, 244, 133, 1, 76, 10, 180, 175,
  ]);
  let expected = u8x32::from([
    22, 165, 49, 4, 76, 162, 119, 61, 120, 198, 249, 12, 21, 59, 12, 206, 211,
    139, 60, 81, 47, 65, 161, 100, 22, 11, 122, 254, 179, 245, 75, 80,
  ]);
  let actual = !a;
  assert_eq!(expected, actual);

  crate::test_random_vector_vs_scalar(|a: u8x32, _b| !a, |a, _b| !a);
}

#[test]
fn impl_u8x32_blend() {
  let use_t: u8 = 0xff;
  let t = u8x32::from([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 126, 127, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 126, 127,
  ]);
  let f = u8x32::from([
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 1, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 1,
  ]);
  let mask = u8x32::from([
    use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0,
    use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0, use_t, 0,
    use_t, 0, use_t, 0,
  ]);
  let expected = u8x32::from([
    1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 126, 1, 1, 18, 3, 20, 5,
    22, 7, 24, 9, 26, 11, 28, 13, 30, 126, 1,
  ]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u8x32_max() {
  crate::test_random_vector_vs_scalar(|a: u8x32, b| a.max(b), |a, b| a.max(b));
}

#[test]
fn impl_u8x32_min() {
  crate::test_random_vector_vs_scalar(|a: u8x32, b| a.min(b), |a, b| a.min(b));
}

#[test]
fn test_u8x32_move_mask() {
  let a = u8x32::from([
    0xff, 0, 0x82, 0x83, 0xff, 0, 0x82, 0x83, 0xff, 0, 0xff, 0, 0xff, 0, 0xff,
    0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
    0xff, 0,
  ]);
  let expected = 0b01010101011111110101010111011101;
  let actual = a.to_bitmask();
  assert_eq!(expected, actual);
}

#[test]
fn test_u8x32_any() {
  let a = u8x32::from([
    0, 0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ]);
  assert!(a.any());

  let a = u8x32::from([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0xff, 0, 0, 0,
  ]);
  assert!(a.any());

  //
  let a = u8x32::from([0; 32]);
  assert!(!a.any());
}

#[test]
fn test_u8x32_all() {
  let a = u8x32::from([
    0, 0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ]);
  assert!(!a.all());
  //
  let a = u8x32::from([0xff; 32]);
  assert!(a.all());
}

#[test]
fn test_u8x32_none() {
  let a = u8x32::from([
    0, 0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ]);
  assert!(!a.none());

  let a = u8x32::from([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0xff, 0, 0, 0,
  ]);
  assert!(!a.none());

  //
  let a = u8x32::from([0; 32]);
  assert!(a.none());
}

#[cfg(feature = "serde")]
#[test]
fn impl_u8x32_ser_de_roundtrip() {
  let serialized =
    bincode::serialize(&u8x32::ZERO).expect("serialization failed");
  let deserialized =
    bincode::deserialize(&serialized).expect("deserializaion failed");
  assert_eq!(u8x32::ZERO, deserialized);
}
