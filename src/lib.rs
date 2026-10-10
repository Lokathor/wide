#![no_std]
#![allow(non_camel_case_types)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::missing_inline_in_public_items)]
#![allow(clippy::eq_op)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::let_and_return)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::misrefactored_assign_op)]
#![allow(clippy::approx_constant)]
#![forbid(missing_docs)]

//! A crate to help you go wide.
//!
//! This crate provides SIMD-compatible data types.
//!
//! When possible, explicit SIMD is used with all the math operations here. As a
//! fallback, the fact that all the lengths of a fixed length array are doing
//! the same thing will often make LLVM notice that it should use SIMD
//! instructions to complete the task. In the worst case, the code just becomes
//! totally scalar (though the math is still correct, at least).
//!
//! # Masks
//!
//! SIMD vector masks are per-element booleans, and are represented by SIMD
//! vectors where each element's value in bits is either all zeros (`false`) or
//! all ones (`true`).
//!
//! SIMD versions of functions that regularly return `bool` return masks. For
//! example, [`f32::is_sign_positive`] returns `bool`, and
//! [`f32x4::is_sign_positive`] returns an [`f32x4`] that represents a mask.
//! The [`select`] method can be used to perform a per-element `if` statement
//! over a mask. For example, for this simple scalar code:
//!
//! ```
//! let x = 1.0_f32;
//!
//! let result = if x.is_sign_positive() {
//!     5.0
//! } else {
//!     3.0
//! };
//!
//! assert_eq!(result, 5.0);
//! ```
//!
//! This is the SIMD version:
//!
//! ```
//! # use wide::f32x4;
//! #
//! let x = f32x4::new([1.0, -1.0, -1.0, 1.0]);
//!
//! let result = x.is_sign_positive().select(
//!     f32x4::splat(5.0),
//!     f32x4::splat(3.0),
//! );
//!
//! assert_eq!(result, f32x4::new([5.0, 3.0, 3.0, 5.0]));
//! ```
//!
//! To select from two values of a downstream type, implement the [`Select`]
//! trait.
//!
//! # Shuffling
//!
//! Shuffling, also known as swizzling, creates a new SIMD vector by selecting
//! elements from one or more input vectors according to a set of indices.
//!
//! Single-input shuffling is performed using the [`shuffle`] method and its
//! variants. Multi-input shuffling is performed by placing the input vectors in
//! an array and calling the corresponding methods from [`ShuffleExt`].
//!
//! Currently, constant-index shuffling is only supported for specific types,
//! with the [`shuffle_consts`] method.
//!
//! Runtime-index based shuffling:
//!
//! ```
//! use wide::{f32x4, ShuffleExt, u32x4};
//!
//! let simd_a = f32x4::new([0.0, 1.0, 2.0, 3.0]);
//! let simd_b = f32x4::new([100.0, 101.0, 102.0, 103.0]);
//!
//! let reverse = simd_a.shuffle(u32x4::new([3, 2, 1, 0]));
//!
//! assert_eq!(reverse, f32x4::new([3.0, 2.0, 1.0, 0.0]));
//!
//! // Here, indices `0..4` map to `simd_a`, and indices `4..8` map to `simd_b`
//! let from_two_vectors = [simd_a, simd_b].shuffle(u32x4::new([2, 3, 4, 5]));
//!
//! assert_eq!(from_two_vectors, f32x4::new([2.0, 3.0, 100.0, 101.0]));
//! ```
//!
//! Constant-index based shuffling:
//!
//! ```
//! use wide::f32x4;
//!
//! let simd_a = f32x4::new([0.0, 1.0, 2.0, 3.0]);
//!
//! let reverse = simd_a.shuffle_consts::<3, 2, 1, 0>();
//!
//! assert_eq!(reverse, f32x4::new([3.0, 2.0, 1.0, 0.0]));
//! ```
//!
//! # NaN bit patterns
//!
//! Operations on SIMD vectors of floats do not make any guarantees about the
//! specific bit-patterns of output NaN values (meaning the sign bit,
//! quiet/signaling bit, and payload). This is unlike standard library
//! operations on float primitives, which do define rules for what NaN bit
//! patterns are returned.
//!
//! The reason for this is that enforcing guarantees would add substantial
//! overhead to operations, and is generally not worth it.
//!
//! # Wrapping semantics
//!
//! SIMD vectors of integers treat operators as wrapping, as if [`Wrapping<T>`]
//! was used. Thus, SIMD vectors do not implement `wrapping_*` functions,
//! because that is the default behavior. This means there are no overflow
//! checks, even in debug builds.
//!
//! The reason for this is that for most applications where SIMD is appropriate,
//! it is "not a bug" to wrap, and even debug builds are unlikely to tolerate
//! the loss of performance.
//!
//! This "no panicking" approach extends to more than just integer overflows. It
//! is also true for things like [`f32x4::clamp`]. Even though [`f32::clamp`]
//! panics for invalid inputs, the SIMD version does not, because it may be used
//! with per-element branching, where invalid elements get discarded later by
//! [`select`].
//!
//! # Casting
//!
//! The SIMD types implement the [`bytemuck::Pod`] trait, which means that it
//! is possible to do bitwise casts between SIMD types of the same size with
//! the [`bytemuck::cast()`] function and others. `bytemuck` is re-exported by
//! this crate for convenience.
//!
//! This typically does not have much, if any, runtime overhead in optimized
//! builds.
//!
//! # Feature flags
//!
//! * `std`: This causes the feature to link to `std`.
//!   * Currently this just improves the performance of `sqrt` when an explicit
//!     SIMD `sqrt` isn't available.
//!
//! [`select`]: f32x4::select
//! [`shuffle`]: f32x4::shuffle
//! [`shuffle_consts`]: f32x4::shuffle_consts
//! [`Wrapping<T>`]: core::num::Wrapping

// Note(Lokathor): Due to standard library magic, the std-only methods for f32
// and f64 will automatically be available simply by declaring this.
#[cfg(feature = "std")]
extern crate std;

// TODO
// Add/Sub/Mul/Div with constant
// Shuffle left/right/by index

use core::ops::*;

#[allow(unused_imports)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use safe_arch::*;

use bytemuck::*;

use crate::utils::*;

// Re-export so that users don't need to add a bytemuck dependency of their own
pub use bytemuck;

#[expect(deprecated)]
pub use simd::{AlignTo, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe, Select, ShuffleExt};

#[macro_use]
mod simd;
#[macro_use]
mod simd_float;
#[macro_use]
mod simd_int;
#[macro_use]
mod simd_uint;

#[macro_use]
mod utils;

mod f32x16_;
pub use f32x16_::*;

mod f32x8_;
pub use f32x8_::*;

mod f32x4_;
pub use f32x4_::*;

mod f64x8_;
pub use f64x8_::*;

mod f64x4_;
pub use f64x4_::*;

mod f64x2_;
pub use f64x2_::*;

mod i8x16_;
pub use i8x16_::*;

mod i16x16_;
pub use i16x16_::*;

mod i16x32_;
pub use i16x32_::*;

mod i8x32_;
pub use i8x32_::*;

mod i8x64_;
pub use i8x64_::*;

mod i16x8_;
pub use i16x8_::*;

mod i32x4_;
pub use i32x4_::*;

mod i32x8_;
pub use i32x8_::*;

mod i32x16_;
pub use i32x16_::*;

mod i64x2_;
pub use i64x2_::*;

mod i64x4_;
pub use i64x4_::*;

mod i64x8_;
pub use i64x8_::*;

mod u8x16_;
pub use u8x16_::*;

mod u8x32_;
pub use u8x32_::*;

mod u8x64_;
pub use u8x64_::*;

mod u16x8_;
pub use u16x8_::*;

mod u16x16_;
pub use u16x16_::*;

mod u16x32_;
pub use u16x32_::*;

mod u32x4_;
pub use u32x4_::*;

mod u32x8_;
pub use u32x8_::*;

mod u32x16_;
pub use u32x16_::*;

mod u64x2_;
pub use u64x2_::*;

mod u64x4_;
pub use u64x4_::*;

mod u64x8_;
pub use u64x8_::*;
