#![cfg(target_feature = "rdrand")]
#![allow(clippy::module_name_repetitions)]

use super::*;

/// Attempts to produce a `u16` from the hardware RNG.
///
/// Intel suggests that you try to call this function up to 10 times in a row if
/// you get a failure. After that there's probably something _actually_ wrong.
///
/// Note: this is the same speed as [`rdrand32_step`], so you usually might as
/// well use that one.
///
/// See
/// [`_rdrand16_step`](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_rdrand16_step)
pub fn rdrand16_step() -> Option<u16> {
  let mut x = 0_u16;
  if 1 == unsafe { _rdrand16_step(&mut x) } {
    Some(x)
  } else {
    None
  }
}

/// Attempts to produce a `u32` from the hardware RNG.
///
/// Intel suggests that you try to call this function up to 10 times in a row if
/// you get a failure. After that there's probably something _actually_ wrong.
///
/// Note: this is the same speed as [`rdrand64_step`], so you usually might as
/// well use that one if you're on a `x86_64`.
///
/// See
/// [`_rdrand32_step`](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_rdrand32_step)
pub fn rdrand32_step() -> Option<u32> {
  let mut x = 0_u32;
  if 1 == unsafe { _rdrand32_step(&mut x) } {
    Some(x)
  } else {
    None
  }
}

/// Attempts to produce a `u64` from the hardware RNG.
///
/// Intel suggests that you try to call this function up to 10 times in a row if
/// you get a failure. After that there's probably something _actually_ wrong.
///
/// Only available on `x86_64`
///
/// See
/// [`_rdrand64_step`](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_rdrand64_step)
#[cfg(target_arch = "x86_64")]
pub fn rdrand64_step() -> Option<u64> {
  let mut x = 0_u64;
  if 1 == unsafe { _rdrand64_step(&mut x) } {
    Some(x)
  } else {
    None
  }
}
