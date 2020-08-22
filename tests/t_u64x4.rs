use wide::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<u64x4>(), 32);
  assert_eq!(core::mem::align_of::<u64x4>(), 32);
}

#[test]
fn impl_add_for_u64x4() {
  let a = u64x4::from([u64::MAX - 1, u64::MAX - 1, 6, 9]);
  let b = u64x4::from([1, 2, 3, 4]);
  let expected = u64x4::from([u64::MAX, u64::MIN, 9, 13]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_u64x4() {
  let a = u64x4::from([1, 0, 9, 12]);
  let b = u64x4::from([1, 1, 3, 3]);
  let expected = u64x4::from([0, u64::MAX, 6, 9]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_u64x4() {
  let a = u64x4::from([1, 1, 0, 0]);
  let b = u64x4::from([0, 1, 0, 1]);
  let expected = u64x4::from([0, 1, 0, 0]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_u64x4() {
  let a = u64x4::from([1, 1, 0, 0]);
  let b = u64x4::from([0, 1, 0, 1]);
  let expected = u64x4::from([1, 1, 0, 1]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_u64x4() {
  let a = u64x4::from([1, 1, 1, 0]);
  let b = u64x4::from([0, 1, 0, 1]);
  let expected = u64x4::from([1, 0, 1, 1]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shl_for_u64x4() {
  let a = u64x4::from([u64::MAX - 1, u64::MAX - 1, 65535, 0]);
  let b = 2;
  let expected =
    u64x4::from([(u64::MAX - 1) << 2, (u64::MAX - 1) << 2, 65535 << 2, 0 << 2]);
  let actual = a << b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_shr_for_u64x4() {
  let a = u64x4::from([u64::MAX - 1, u64::MAX - 1, 65535, 0]);
  let b = 2;
  let expected =
    u64x4::from([(u64::MAX - 1) >> 2, (u64::MAX - 1) >> 2, 65535 >> 2, 0 >> 2]);
  let actual = a >> b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x4_blend() {
  let use_t: u64 = u64::MAX;
  let t = u64x4::from([1, 2, 3, 4]);
  let f = u64x4::from([17, 18, 21, 45]);
  let mask = u64x4::from([use_t, 0, 0, use_t]);
  let expected = u64x4::from([1, 18, 21, 4]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_u64x4_cmp_eq() {
  let a = u64x4::from([1_u64, 4, u64::MAX, 5]);
  let b = u64x4::from([3_u64, 4, u64::MAX, 1]);
  let expected = u64x4::from([0, u64::MAX, u64::MAX, 0]);
  let actual = a.cmp_eq(b);
  assert_eq!(expected, actual);
}
