use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u64x8>(), 64);
  assert_eq!(core::mem::align_of::<u64x8>(), 64);
}

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
  let actual = a.cmp_eq(b);
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
