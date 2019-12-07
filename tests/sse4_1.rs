#![cfg(target_feature = "sse4.1")]
#![allow(bad_style)]

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn m128i_blend_var_i8() {
  let ai: m128i = cast([5_i8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
  let bi: m128i = cast([7_i8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]);
  let mask: m128i =
    cast([0_i8, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1]);
  let out: [i8; 16] = cast(ai.blend_var_i8(bi, mask));
  assert_eq!(out, [5, 5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 5, 7, 7]);
}

#[test]
fn m128d_blend_var() {
  let a: m128d = cast([5.0_f64, 5.0]);
  let b: m128d = cast([7.0_f64, 7.0]);
  let mask: m128d = cast([-1.0_f64, 0.0]);
  let out: [f64; 2] = cast(a.blend_var(b, mask));
  assert_eq!(out, [7.0_f64, 5.0]);
}

#[test]
fn m128_blend_var() {
  let a: m128 = cast([5.0_f32, 5.0, 5.0, 5.0]);
  let b: m128 = cast([7.0_f32, 7.0, 7.0, 7.0]);
  let mask: m128 = cast([0.0_f32, 0.0, -1.0, -1.0]);
  let out: [f32; 4] = cast(a.blend_var(b, mask));
  assert_eq!(out, [5.0_f32, 5.0, 7.0, 7.0]);
}

#[test]
fn m128_ceil() {
  let a: m128 = cast([5.1_f32, 4.9, 5.0, -1.1]);
  let out: [f32; 4] = cast(a.ceil());
  assert_eq!(out, [6.0_f32, 5.0, 5.0, -1.0]);
}

#[test]
fn m128d_ceil() {
  let a: m128d = cast([5.1_f64, -1.1]);
  let out: [f64; 2] = cast(a.ceil());
  assert_eq!(out, [6.0_f64, -1.0]);
}

#[test]
fn m128_ceil_rhs0() {
  let a: m128 = cast([5.1_f32, 4.9, 5.0, -1.1]);
  let out: [f32; 4] = cast(a.ceil_rhs0(a));
  assert_eq!(out, [6.0_f32, 4.9, 5.0, -1.1]);
}

#[test]
fn m128d_ceil_rhs0() {
  let a: m128d = cast([5.1_f64, -1.1]);
  let out: [f64; 2] = cast(a.ceil_rhs0(a));
  assert_eq!(out, [6.0_f64, -1.1]);
}

#[test]
fn m128_floor() {
  let a: m128 = cast([5.1_f32, 4.9, 5.0, -1.1]);
  let out: [f32; 4] = cast(a.floor());
  assert_eq!(out, [5.0_f32, 4.0, 5.0, -2.0]);
}

#[test]
fn m128d_floor() {
  let a: m128d = cast([5.1_f64, -1.1]);
  let out: [f64; 2] = cast(a.floor());
  assert_eq!(out, [5.0_f64, -2.0]);
}

#[test]
fn m128_floor_rhs0() {
  let a: m128 = cast([5.1_f32, 4.9, 5.0, -1.1]);
  let out: [f32; 4] = cast(a.floor_rhs0(a));
  assert_eq!(out, [5.0_f32, 4.9, 5.0, -1.1]);
}

#[test]
fn m128d_floor_rhs0() {
  let a: m128d = cast([5.1_f64, -1.1]);
  let out: [f64; 2] = cast(a.floor_rhs0(a));
  assert_eq!(out, [5.0_f64, -1.1]);
}

#[test]
fn m128i_cmp_eq_i64() {
  let ai: m128i = cast([core::i64::MAX, core::i64::MIN]);
  let bi: m128i = cast([core::i64::MAX, 3]);
  let out: [i64; 2] = cast(ai.cmp_eq_i64(bi));
  assert_eq!(out, [-1, 0]);
}

#[test]
fn m128i_sign_extend_i16_i32() {
  let ai: m128i =
    cast([0_i16, 1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let out: [i32; 4] = cast(ai.sign_extend_i16_i32());
  assert_eq!(out, [0, 1, -1, i32::from(core::i16::MAX)]);
}

#[test]
fn m128i_sign_extend_i16_i64() {
  let ai: m128i =
    cast([1_i16, -1, -1, core::i16::MAX, core::i16::MIN, 100, 7, 2]);
  let out: [i64; 2] = cast(ai.sign_extend_i16_i64());
  assert_eq!(out, [1, -1]);
}

#[test]
fn m128i_sign_extend_i32_i64() {
  let ai: m128i = cast([1_i32, -1, 0, core::i32::MAX]);
  let out: [i64; 2] = cast(ai.sign_extend_i32_i64());
  assert_eq!(out, [1, -1]);
}

#[test]
fn m128i_sign_extend_i8_i16() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let out: [i16; 8] = cast(ai.sign_extend_i8_i16());
  assert_eq!(out, [0_i16, 1, -1, 127, -128, 100, 7, 2]);
}

#[test]
fn m128i_sign_extend_i8_i32() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let out: [i32; 4] = cast(ai.sign_extend_i8_i32());
  assert_eq!(out, [0_i32, 1, -1, 127]);
}

#[test]
fn m128i_sign_extend_i8_i64() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let out: [i64; 2] = cast(ai.sign_extend_i8_i64());
  assert_eq!(out, [0_i64, 1]);
}

#[test]
fn m128i_zero_extend_u16_i32() {
  let ai: m128i = cast([0_u16, 1, 7, core::u16::MAX, 7, 100, 7, 2]);
  let out: [i32; 4] = cast(ai.zero_extend_u16_i32());
  assert_eq!(out, [0, 1, 7, i32::from(core::u16::MAX)]);
}

#[test]
fn m128i_zero_extend_u16_i64() {
  let ai: m128i = cast([0_u16, 1, 7, core::u16::MAX, 7, 100, 7, 2]);
  let out: [i64; 2] = cast(ai.zero_extend_u16_i64());
  assert_eq!(out, [0, 1]);
}

#[test]
fn m128i_zero_extend_u32_i64() {
  let ai: m128i = cast([0_u32, 1, 7, core::u32::MAX]);
  let out: [i64; 2] = cast(ai.zero_extend_u32_i64());
  assert_eq!(out, [0, 1]);
}

#[test]
fn m128i_zero_extend_u8_i16() {
  let ai: m128i =
    cast([0_u8, 1, 7, 127, 200, 100, 7, 2, 3, 1, 126, 125, 103, 10, 15, 16]);
  let out: [i16; 8] = cast(ai.zero_extend_u8_i16());
  assert_eq!(out, [0_i16, 1, 7, 127, 200, 100, 7, 2]);
}

#[test]
fn m128i_zero_extend_u8_i32() {
  let ai: m128i =
    cast([0_u8, 175, 7, 127, 200, 100, 7, 2, 3, 1, 126, 125, 103, 10, 15, 16]);
  let out: [i32; 4] = cast(ai.zero_extend_u8_i32());
  assert_eq!(out, [0_i32, 175, 7, 127]);
}

#[test]
fn m128i_zero_extend_u8_i64() {
  let ai: m128i =
    cast([0_u8, 175, 7, 127, 200, 100, 7, 2, 3, 1, 126, 125, 103, 10, 15, 16]);
  let out: [i64; 2] = cast(ai.zero_extend_u8_i64());
  assert_eq!(out, [0_i64, 175]);
}

#[test]
fn m128i_max_i32() {
  let ai: m128i = cast([0_i32, 1, -1, 127]);
  let bi: m128i = cast([7_i32, 7, 7, 7]);
  let out: [i32; 4] = cast(ai.max_i32(bi));
  assert_eq!(out, [7, 7, 7, 127]);
}

#[test]
fn m128i_max_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i = cast([7_i8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]);
  let out: [i8; 16] = cast(ai.max_i8(bi));
  assert_eq!(out, [7, 7, 7, 127, 7, 100, 7, 7, 7, 7, 7, 7, 103, 10, 15, 16]);
}

#[test]
fn m128i_max_u16() {
  let ai: m128i = cast([0_u16, 1, 50, 127, 500, 100, 6, 2]);
  let bi: m128i = cast([7_u16, 7, 7, 7, 7, 7, 7, 7]);
  let out: [u16; 8] = cast(ai.max_u16(bi));
  assert_eq!(out, [7, 7, 50, 127, 500, 100, 7, 7]);
}

#[test]
fn m128i_max_u32() {
  let ai: m128i = cast([0_u32, 1, 55, 127]);
  let bi: m128i = cast([7_u32, 7, 7, 7]);
  let out: [u32; 4] = cast(ai.max_u32(bi));
  assert_eq!(out, [7, 7, 55, 127]);
}

#[test]
fn m128i_min_i32() {
  let ai: m128i = cast([0_i32, 1, -1, 127]);
  let bi: m128i = cast([7_i32, 7, 7, 7]);
  let out: [i32; 4] = cast(ai.min_i32(bi));
  assert_eq!(out, [0, 1, -1, 7]);
}

#[test]
fn m128i_min_i8() {
  let ai: m128i = cast([
    0_i8, 1, -1, 127, -128, 100, 7, 2, 3, 1, -126, -125, 103, 10, 15, 16,
  ]);
  let bi: m128i = cast([7_i8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]);
  let out: [i8; 16] = cast(ai.min_i8(bi));
  assert_eq!(out, [0, 1, -1, 7, -128, 7, 7, 2, 3, 1, -126, -125, 7, 7, 7, 7]);
}

#[test]
fn m128i_min_u16() {
  let ai: m128i = cast([0_u16, 1, 50, 127, 500, 100, 6, 2]);
  let bi: m128i = cast([7_u16, 7, 7, 7, 7, 7, 7, 7]);
  let out: [u16; 8] = cast(ai.min_u16(bi));
  assert_eq!(out, [0, 1, 7, 7, 7, 7, 6, 2]);
}

#[test]
fn m128i_min_u32() {
  let ai: m128i = cast([0_u32, 1, 55, 127]);
  let bi: m128i = cast([7_u32, 7, 7, 7]);
  let out: [u32; 4] = cast(ai.min_u32(bi));
  assert_eq!(out, [0, 1, 7, 7]);
}

#[test]
fn m128i_min_and_position_u16() {
  let ai: m128i = cast([7_u16, 5, 50, 127, 500, 100, 6, 12]);
  let out: [u16; 8] = cast(ai.min_and_position_u16());
  assert_eq!(out, [5, 1, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn m128i_widen_mul_i32_i64() {
  let ai: m128i = cast([-1_i32, 50, 12, 127]);
  let bi: m128i = cast([7_i32, 7, 7, 7]);
  let out: [i64; 2] = cast(ai.widen_mul_i32_i64(bi));
  assert_eq!(out, [-7, 84]);
}

#[test]
fn m128i_mul_i32() {
  let ai: m128i = cast([-1_i32, 50, 12, 127]);
  let bi: m128i = cast([7_i32, 7, 7, 7]);
  let out: [i32; 4] = cast(ai.mul_i32(bi));
  assert_eq!(out, [-7, 350, 84, 889]);
}

#[test]
fn m128i_test_all_bits_one() {
  let ai: m128i = cast(core::u128::MAX);
  assert_eq!(ai.test_all_bits_one(), 1);
  let ai: m128i = cast(core::u128::MAX - 1);
  assert_eq!(ai.test_all_bits_one(), 0);
}

#[test]
fn m128i_test_cf() {
  let ai: m128i = cast([0, core::u64::MAX]);
  let bi: m128i = cast([0, core::u64::MAX]);
  assert_eq!(ai.test_cf(bi), 1);
}

#[test]
fn m128i_test_zf() {
  let ai: m128i = cast([0, core::u64::MAX]);
  let bi: m128i = cast([core::u64::MAX, 0]);
  assert_eq!(ai.test_zf(bi), 1);
}

#[test]
fn m128i_test_not_zf_cf() {
  let ai: m128i = cast([0, core::u64::MAX]);
  let bi: m128i = cast([core::u64::MAX, core::u64::MAX]);
  assert_eq!(ai.test_not_zf_cf(bi), 1);
}
