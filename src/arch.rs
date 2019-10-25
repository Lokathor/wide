//! Architecture specific functionality.
//!
//! **THIS MODULE IS AN IMPLEMENTATION DETAIL OF THE CRATE.**
//!
//! The API here is not subject to the semver promise of the crate and if you
//! decide to directly use anything in here I am free to break your code.

use super::*;

/// Wrap the inner value to a minimum alignment of 2.
///
/// This is for alignment shenanigans, you're not expected to use it in a
/// struct, more just in function arguments and such.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(2))]
pub struct Align2<T>(pub T);

/// Wrap the inner value to a minimum alignment of 4.
///
/// This is for alignment shenanigans, you're not expected to use it in a
/// struct, more just in function arguments and such.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(4))]
pub struct Align4<T>(pub T);

/// Wrap the inner value to a minimum alignment of 8.
///
/// This is for alignment shenanigans, you're not expected to use it in a
/// struct, more just in function arguments and such.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(8))]
pub struct Align8<T>(pub T);

/// Wrap the inner value to a minimum alignment of 16.
///
/// This is for alignment shenanigans, you're not expected to use it in a
/// struct, more just in function arguments and such.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Align16<T>(pub T);

/// Wrap the inner value to a minimum alignment of 32.
///
/// This is for alignment shenanigans, you're not expected to use it in a
/// struct, more just in function arguments and such.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct Align32<T>(pub T);

unsafe impl<T> Zeroable for Align2<T> where T: Zeroable {}
unsafe impl<T> Zeroable for Align4<T> where T: Zeroable {}
unsafe impl<T> Zeroable for Align8<T> where T: Zeroable {}
unsafe impl<T> Zeroable for Align16<T> where T: Zeroable {}
unsafe impl<T> Zeroable for Align32<T> where T: Zeroable {}
//
unsafe impl Pod for Align2<[u8; 2]> {}
unsafe impl Pod for Align2<[i8; 2]> {}
//
unsafe impl Pod for Align4<[u8; 4]> {}
unsafe impl Pod for Align4<[i8; 4]> {}
unsafe impl Pod for Align4<[u16; 2]> {}
unsafe impl Pod for Align4<[i16; 2]> {}
//
unsafe impl Pod for Align8<[u8; 8]> {}
unsafe impl Pod for Align8<[i8; 8]> {}
unsafe impl Pod for Align8<[u16; 4]> {}
unsafe impl Pod for Align8<[i16; 4]> {}
unsafe impl Pod for Align8<[u32; 2]> {}
unsafe impl Pod for Align8<[i32; 2]> {}
unsafe impl Pod for Align8<[f32; 2]> {}
//
unsafe impl Pod for Align16<[u8; 16]> {}
unsafe impl Pod for Align16<[i8; 16]> {}
unsafe impl Pod for Align16<[u16; 8]> {}
unsafe impl Pod for Align16<[i16; 8]> {}
unsafe impl Pod for Align16<[u32; 4]> {}
unsafe impl Pod for Align16<[i32; 4]> {}
unsafe impl Pod for Align16<[f32; 4]> {}
unsafe impl Pod for Align16<[u64; 2]> {}
unsafe impl Pod for Align16<[i64; 2]> {}
unsafe impl Pod for Align16<[f64; 2]> {}
unsafe impl Pod for Align16<u128> {}
unsafe impl Pod for Align16<i128> {}
//
unsafe impl Pod for Align32<[u8; 32]> {}
unsafe impl Pod for Align32<[i8; 32]> {}
unsafe impl Pod for Align32<[u16; 16]> {}
unsafe impl Pod for Align32<[i16; 16]> {}
unsafe impl Pod for Align32<[u32; 8]> {}
unsafe impl Pod for Align32<[i32; 8]> {}
unsafe impl Pod for Align32<[f32; 8]> {}
unsafe impl Pod for Align32<[u64; 4]> {}
unsafe impl Pod for Align32<[i64; 4]> {}
unsafe impl Pod for Align32<[f64; 4]> {}
unsafe impl Pod for Align32<[u128; 2]> {}
unsafe impl Pod for Align32<[i128; 2]> {}
//
#[cfg(target_pointer_width = "32")]
unsafe impl Pod for Align8<[usize; 2]> {}
#[cfg(target_pointer_width = "32")]
unsafe impl Pod for Align16<[usize; 4]> {}
#[cfg(target_pointer_width = "32")]
unsafe impl Pod for Align32<[usize; 8]> {}
//
#[cfg(target_pointer_width = "64")]
unsafe impl Pod for Align16<[usize; 2]> {}
#[cfg(target_pointer_width = "64")]
unsafe impl Pod for Align32<[usize; 4]> {}

#[cfg(target_arch = "x86")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
