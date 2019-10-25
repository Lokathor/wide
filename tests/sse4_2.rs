#![cfg(target_feature = "sse4.1")]
#![allow(bad_style)]

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn m128i_cmp_gt_i64() {
  let ai: m128i = cast([0_i64, 5]);
  let bi: m128i = cast([0_i64, 0]);
  let out: [i64; 2] = cast(ai.cmp_gt_i64(bi));
  assert_eq!(out, [0, -1]);
}

// Note(Lokathor): The CRC inputs are just random inputs I typed in, and then
// the outputs are just whatever the output was, which makes for a somewhat poor
// test, but at least we'll detect any changes that pop up later.

#[test]
fn test_crc32_u8() {
  let crc: u32 = -1_i32 as u32;
  let new_crc: u32 = crc32_u8(crc, 7);
  assert_eq!(new_crc, 2_034_812_997);
}

#[test]
fn test_crc32_u16() {
  let crc: u32 = -1_i32 as u32;
  let new_crc: u32 = crc32_u16(crc, 765);
  assert_eq!(new_crc, 1_795_419_319);
}

#[test]
fn test_crc32_u32() {
  let crc: u32 = -1_i32 as u32;
  let new_crc: u32 = crc32_u32(crc, 7_654_322);
  assert_eq!(new_crc, 200_296_263);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_crc32_u64() {
  let crc: u64 = -1_i64 as u64;
  let new_crc: u64 = crc32_u64(crc, 7_654_322_321_098);
  assert_eq!(new_crc, 2_461_165_299);
}
