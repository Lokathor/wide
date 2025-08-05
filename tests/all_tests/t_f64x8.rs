use core::f64;

use wide::*;

use bytemuck::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<f64x8>(), 64);
  assert_eq!(core::mem::align_of::<f64x8>(), 64);
}

// TODO: port existing f64x4 tests