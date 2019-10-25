#![cfg(target_feature = "sse3")]
#![allow(bad_style)]

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn m128d_add_sub() {
  let a: m128d = cast([5.0_f64, 5.0]);
  let b: m128d = cast([2.0_f64, 1.0]);
  let out: [f64; 2] = cast(a.add_sub(b));
  assert_eq!(out, [3.0, 6.0]);
}

#[test]
fn m128_add_sub() {
  let a: m128 = cast([5.0_f32, 5.0, 5.0, 5.0]);
  let b: m128 = cast([2.0_f32, 1.0, 3.0, 7.0]);
  let out: [f32; 4] = cast(a.add_sub(b));
  assert_eq!(out, [3.0_f32, 6.0, 2.0, 12.0]);
}

#[test]
fn m128d_horizontal_add() {
  let a: m128d = cast([3.0_f64, 4.0]);
  let b: m128d = cast([5.0_f64, 6.0]);
  let out: [f64; 2] = cast(a.horizontal_add(b));
  assert_eq!(out, [7.0, 11.0]);
}

#[test]
fn m128_horizontal_add() {
  let a: m128 = cast([3.0_f32, 4.0, 8.0, 2.0]);
  let b: m128 = cast([5.0_f32, 6.0, 10.0, 9.0]);
  let out: [f32; 4] = cast(a.horizontal_add(b));
  assert_eq!(out, [7.0_f32, 10.0, 11.0, 19.0]);
}

#[test]
fn m128d_horizontal_sub() {
  let a: m128d = cast([9.0_f64, 3.0]);
  let b: m128d = cast([5.0_f64, 6.0]);
  let out: [f64; 2] = cast(a.horizontal_sub(b));
  assert_eq!(out, [6.0, -1.0]);
}

#[test]
fn m128_horizontal_sub() {
  let a: m128 = cast([3.0_f32, 12.0, 8.0, 2.0]);
  let b: m128 = cast([5.0_f32, 6.0, 10.0, 7.0]);
  let out: [f32; 4] = cast(a.horizontal_sub(b));
  assert_eq!(out, [-9.0_f32, 6.0, -1.0, 3.0]);
}

#[test]
fn m128i_load_quick_unaligned() {
  let out: m128i = m128i::load_quick_unaligned(&12345);
  let out_bits: i128 = cast(out);
  assert_eq!(out_bits, 12345);
}

#[test]
fn m128d_load_splat() {
  let out: m128d = m128d::load_splat(&5.0);
  let out_f64s: [f64; 2] = cast(out);
  assert_eq!(out_f64s, [5.0_f64, 5.0]);
}

#[test]
fn m128d_duplicate_low() {
  let a: m128d = cast([3.0_f64, 12.0]);
  let out: m128d = a.duplicate_low();
  let out_f64s: [f64; 2] = cast(out);
  assert_eq!(out_f64s, [3.0_f64, 3.0]);
}

#[test]
fn m128_duplicate_odd() {
  let a: m128 = cast([3.0_f32, 12.0, 7.0, 6.0]);
  let out: m128 = a.duplicate_odd();
  let out_f32s: [f32; 4] = cast(out);
  assert_eq!(out_f32s, [12.0_f32, 12.0, 6.0, 6.0]);
}

#[test]
fn m128_duplicate_even() {
  let a: m128 = cast([3.0_f32, 12.0, 7.0, 6.0]);
  let out: m128 = a.duplicate_even();
  let out_f32s: [f32; 4] = cast(out);
  assert_eq!(out_f32s, [3.0_f32, 3.0, 7.0, 7.0]);
}
