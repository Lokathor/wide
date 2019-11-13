//! Intrinsics for the [x86](https://en.wikipedia.org/wiki/X86) processor family.

use super::*;

use core::arch::x86::*;

#[cfg(target_feature = "rdrand")]
#[path = "rdrand.rs"]
mod rdrand;
#[cfg(target_feature = "rdrand")]
pub use rdrand::*;

#[cfg(target_feature = "sse")]
#[path = "sse.rs"]
mod sse;
#[cfg(target_feature = "sse")]
pub use sse::*;

#[cfg(target_feature = "sse2")]
#[path = "sse2.rs"]
mod sse2;
#[cfg(target_feature = "sse2")]
pub use sse2::*;

#[cfg(target_feature = "sse3")]
#[path = "sse3.rs"]
mod sse3;
#[cfg(target_feature = "sse3")]
pub use sse3::*;

#[cfg(target_feature = "ssse3")]
#[path = "ssse3.rs"]
mod ssse3;
#[cfg(target_feature = "ssse3")]
pub use ssse3::*;

#[cfg(target_feature = "sse4.1")]
#[path = "sse4_1.rs"]
mod sse4_1;
#[cfg(target_feature = "sse4.1")]
pub use sse4_1::*;

#[cfg(target_feature = "sse4.2")]
#[path = "sse4_2.rs"]
mod sse4_2;
#[cfg(target_feature = "sse4.2")]
pub use sse4_2::*;

#[cfg(target_feature = "fma")]
#[path = "fma.rs"]
mod fma;
#[cfg(target_feature = "fma")]
pub use fma::*;

/// As [`_rdtsc`](https://doc.rust-lang.org/core/arch/x86/fn._rdtsc.html).
#[inline]
pub fn rdtsc() -> u64 {
  unsafe { _rdtsc() }
}
