#![cfg(target_feature = "ssse3")]
#![allow(bad_style)]

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn m128i_abs_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let out: [i8; 16] = cast(ai.abs_i8());
  assert_eq!(
    out,
    [0_i8, 1, 1, 127, -128, 100, 7, 2, 3, 1, 126, 125, 103, 10, 15, 16,]
  );
}

#[test]
fn m128i_abs_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let out: [i16; 8] = cast(ai.abs_i16());
  assert_eq!(out, [0_i16, 1, 1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
}

#[test]
fn m128i_abs_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let out: [i32; 4] = cast(ai.abs_i32());
  assert_eq!(out, [0, 1, core::i32::MAX, core::i32::MIN]);
}

#[test]
fn m128i_horizontal_add_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.horizontal_add_i16(bi));
  assert_eq!(
    out,
    [1, core::i16::MAX - 1, core::i16::MIN + 100, 9, 3, 3, 7, 29]
  );
}

#[test]
fn m128i_horizontal_saturating_add_i16() {
  let ai: m128i =
    cast([0_i16, 1, 1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.horizontal_saturating_add_i16(bi));
  assert_eq!(out, [1, core::i16::MAX, core::i16::MIN + 100, 9, 3, 3, 7, 29]);
}

#[test]
fn m128i_horizontal_add_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 2, -1]);
  let out: [i32; 4] = cast(ai.horizontal_add_i32(bi));
  assert_eq!(out, [-1, -1, 3, 1]);
}

#[test]
fn m128i_horizontal_sub_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.horizontal_sub_i16(bi));
  assert_eq!(out, [-1, core::i16::MIN, core::i16::MAX - 99, 5, -1, 1, -9, -11]);
}

#[test]
fn m128i_horizontal_saturating_sub_i16() {
  let ai: m128i =
    cast([0_i16, 1, 1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.horizontal_saturating_sub_i16(bi));
  assert_eq!(out, [-1, core::i16::MIN + 2, core::i16::MIN, 5, -1, 1, -9, -11]);
}

#[test]
fn m128i_horizontal_sub_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 2, -1]);
  let out: [i32; 4] = cast(ai.horizontal_sub_i32(bi));
  assert_eq!(out, [1, -1, -1, 3]);
}

#[test]
fn m128i_mul_hadd_u8_to_i16() {
  let ai: m128i =
    cast([0_u8, 1, 255, 127, 128, 100, 7, 2, 3, 1, 126, 125, 103, 10, 15, 16]);
  let bi: m128i =
    cast([1_u8, 2, 2, 1, 255, 8, 9, 20, 60, 40, 2, 9, 27, 26, 30, 31]);
  let out: [i16; 8] = cast(ai.mul_hadd_u8_to_i16(bi));
  assert_eq!(out, [2, 637, 672, 103, 220, 1377, 3041, 946]);
}

#[test]
fn m128i_mul_higher_ish_i16() {
  let ai: m128i =
    cast([654_i16, 2342, -1123, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([6464_i16, 3424, 22342, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.mul_higher_ish_i16(bi));
  assert_eq!(out, [129, 245, -766, 1, 1, 0, 0, 0]);
}

#[test]
fn m128i_shuffle_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.shuffle_i8(bi));
  assert_eq!(
    out,
    [1, -1, -1, 1, 0, 3, 1, -128, 103, 3, -1, 1, 0, -126, 15, 16]
  );
}

#[test]
fn m128i_sign_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i =
    cast([1_i8, 2, 2, 1, -1, 8, 9, 20, 60, 40, 2, 9, -27, 26, 30, 31]);
  let out: [i8; 16] = cast(ai.sign_i8(bi));
  assert_eq!(
    out,
    [0, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, -103, 10, 15, 16]
  );
}

#[test]
fn m128i_sign_i16() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let bi: m128i = cast([1_i16, 2, 2, 1, -1, 8, 9, 20]);
  let out: [i16; 8] = cast(ai.sign_i16(bi));
  assert_eq!(out, [0, 1, -1, 32767, -32768, 100, 7, 2]);
}

#[test]
fn m128i_sign_i32() {
  let ai: m128i = cast([0_i32, -1, core::i32::MAX, core::i32::MIN]);
  let bi: m128i = cast([1_i32, 2, 0, -1]);
  let out: [i32; 4] = cast(ai.sign_i32(bi));
  assert_eq!(out, [0, -1, 0, core::i32::MIN]);
}
