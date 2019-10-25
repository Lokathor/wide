#![cfg(target_feature = "rdrand")]
#![allow(bad_style)]

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn test_rdrand16_step() {
  let _: Option<u16> = rdrand16_step();
}

#[test]
fn test_rdrand32_step() {
  let _: Option<u32> = rdrand32_step();
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_rdrand64_step() {
  let _: Option<u64> = rdrand64_step();
}
