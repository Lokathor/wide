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
