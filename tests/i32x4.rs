use wide::*;

#[test]
fn declaration_tests_i32x4() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<i32x4>(), 16);
  assert_eq!(align_of::<i32x4>(), 16);
}

#[test]
#[allow(non_snake_case)]
fn declaration_tests_ConstUnionHack_i32x4() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<ConstUnionHack_i32x4>(), size_of::<i32x4>());
  assert_eq!(align_of::<ConstUnionHack_i32x4>(), align_of::<i32x4>());
}

#[test]
fn shift_left_i32() {
  let nums = i32x4::new(1, 2, 3, 4);

  let shifted_one = nums << 1;
  assert_eq!(shifted_one[0], 2);
  assert_eq!(shifted_one[1], 4);
  assert_eq!(shifted_one[2], 6);
  assert_eq!(shifted_one[3], 8);

  let shifted_18 = nums << 18;
  assert_eq!(shifted_18[0], 262144);
  assert_eq!(shifted_18[1], 524288);
  assert_eq!(shifted_18[2], 786432);
  assert_eq!(shifted_18[3], 1048576);

  let shifted_32 = nums << 32;
  assert_eq!(shifted_32[0], 0);
  assert_eq!(shifted_32[1], 0);
  assert_eq!(shifted_32[2], 0);
  assert_eq!(shifted_32[3], 0);
}
