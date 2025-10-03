use core::f64;

use wide::*;

use bytemuck::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<f64x8>(), 64);
  assert_eq!(core::mem::align_of::<f64x8>(), 64);
}

#[test]
fn impl_debug_for_f64x8() {
  let expected = "(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)";
  let actual =
    format!("{:?}", f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
  assert_eq!(expected, actual);

  let expected = "(1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000)";
  let actual =
    format!("{:.3?}", f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
  assert_eq!(expected, actual);
}

#[test]
fn impl_add_for_f64x8() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
  let expected = f64x8::from([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
  assert_eq!(a + b, expected);
}

#[test]
fn impl_sub_for_f64x8() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
  let expected = f64x8::from([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]);
  assert_eq!(a - b, expected);
}

#[test]
fn impl_mul_for_f64x8() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected = f64x8::from([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
  assert_eq!(a * b, expected);
}

#[test]
fn impl_div_for_f64x8() {
  let a = f64x8::from([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
  let b = f64x8::from([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let expected = f64x8::from([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  assert_eq!(a / b, expected);
}

#[test]
fn impl_sub_const_for_f64x8() {
  let a = f64x8::from([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
  let expected = f64x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
  assert_eq!(a - 2.0, expected);
}

#[test]
fn impl_mul_const_for_f64x8() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let expected = f64x8::from([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
  assert_eq!(a * 2.0, expected);
}

#[test]
fn impl_div_const_for_f64x8() {
  let a = f64x8::from([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
  let expected = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  assert_eq!(a / 2.0, expected);
}

#[test]
fn impl_bitand_for_f64x8() {
  let a = f64x8::from([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
  let b = f64x8::from([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
  let expected = f64x8::from([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
  assert_eq!(a & b, expected);
}

#[test]
fn impl_bitor_for_f64x8() {
  let a = f64x8::from([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
  let b = f64x8::from([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
  let expected = f64x8::from([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
  assert_eq!(a | b, expected);
}

#[test]
fn impl_bitxor_for_f64x8() {
  let a = f64x8::from([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
  let b = f64x8::from([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
  let expected = f64x8::from([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
  assert_eq!(a ^ b, expected);
}

#[test]
fn impl_f64x8_cmp_eq() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected: [i64; 8] = [0, -1, 0, 0, 0, 0, 0, 0];
  let actual: [i64; 8] = cast(a.simd_eq(b));
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_cmp_ne() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected: [i64; 8] = [-1, 0, -1, -1, -1, -1, -1, -1];
  let actual: [i64; 8] = cast(a.simd_ne(b));
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_cmp_ge() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected: [i64; 8] = [0, -1, -1, -1, -1, -1, -1, -1];
  let actual: [i64; 8] = cast(a.simd_ge(b));
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_cmp_gt() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected: [i64; 8] = [0, 0, -1, -1, -1, -1, -1, -1];
  let actual: [i64; 8] = cast(a.simd_gt(b));
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_cmp_le() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected: [i64; 8] = [-1, -1, 0, 0, 0, 0, 0, 0];
  let actual: [i64; 8] = cast(a.simd_le(b));
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_cmp_lt() {
  let a = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let b = f64x8::from([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
  let expected: [i64; 8] = [-1, 0, 0, 0, 0, 0, 0, 0];
  let actual: [i64; 8] = cast(a.simd_lt(b));
  assert_eq!(actual, expected);
  let same: [i64; 8] = cast(a.simd_lt(a));
  assert_eq!(same, [0; 8]);
}

#[test]
fn impl_f64x8_blend() {
  let use_t: f64 = f64::from_bits(u64::MAX);
  let t = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let f = f64x8::from([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
  let mask = f64x8::from([use_t, 0.0, use_t, 0.0, use_t, 0.0, use_t, 0.0]);
  let expected = f64x8::from([1.0, 7.0, 3.0, 5.0, 5.0, 3.0, 7.0, 1.0]);
  assert_eq!(mask.blend(t, f), expected);
}

#[test]
fn impl_f64x8_abs() {
  let a = f64x8::from([
    -1.0,
    2.0,
    -3.5,
    f64::NEG_INFINITY,
    1.0,
    -2.0,
    f64::INFINITY,
    -0.0,
  ]);
  let expected =
    f64x8::from([1.0, 2.0, 3.5, f64::INFINITY, 1.0, 2.0, f64::INFINITY, 0.0]);
  assert_eq!(a.abs(), expected);
}

#[test]
fn impl_f64x8_floor() {
  let a = f64x8::from([
    -1.1,
    60.9,
    1.1,
    f64::INFINITY,
    2.5,
    -3.7,
    f64::NEG_INFINITY,
    0.0,
  ]);
  let expected = f64x8::from([
    -2.0,
    60.0,
    1.0,
    f64::INFINITY,
    2.0,
    -4.0,
    f64::NEG_INFINITY,
    0.0,
  ]);
  assert_eq!(a.floor(), expected);
}

#[test]
fn impl_f64x8_ceil() {
  let a = f64x8::from([
    -1.1,
    60.9,
    1.1,
    f64::NEG_INFINITY,
    2.5,
    -3.7,
    f64::INFINITY,
    0.0,
  ]);
  let expected = f64x8::from([
    -1.0,
    61.0,
    2.0,
    f64::NEG_INFINITY,
    3.0,
    -3.0,
    f64::INFINITY,
    0.0,
  ]);
  assert_eq!(a.ceil(), expected);
}

#[test]
fn impl_f64x8_fast_max() {
  let a = f64x8::from([
    1.0,
    5.0,
    3.0,
    -0.0,
    2.0,
    -1.0,
    f64::NEG_INFINITY,
    f64::INFINITY,
  ]);
  let b = f64x8::from([
    2.0,
    f64::NEG_INFINITY,
    f64::INFINITY,
    0.0,
    3.0,
    1.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
  ]);
  let expected = f64x8::from([
    2.0,
    5.0,
    f64::INFINITY,
    0.0,
    3.0,
    1.0,
    f64::INFINITY,
    f64::INFINITY,
  ]);
  assert_eq!(a.fast_max(b), expected);
}

#[test]
fn impl_f64x8_max() {
  let a =
    f64x8::from([1.0, 5.0, 3.0, -0.0, 2.0, -1.0, f64::INFINITY, f64::NAN]);
  let b = f64x8::from([
    2.0,
    f64::NEG_INFINITY,
    f64::INFINITY,
    0.0,
    3.0,
    1.0,
    f64::NAN,
    f64::INFINITY,
  ]);
  let expected = f64x8::from([
    2.0,
    5.0,
    f64::INFINITY,
    0.0,
    3.0,
    1.0,
    f64::INFINITY,
    f64::INFINITY,
  ]);
  assert_eq!(a.max(b), expected);
}

#[test]
fn impl_f64x8_fast_min() {
  let a = f64x8::from([
    1.0,
    5.0,
    3.0,
    -0.0,
    2.0,
    -1.0,
    f64::NEG_INFINITY,
    f64::INFINITY,
  ]);
  let b = f64x8::from([
    2.0,
    f64::NEG_INFINITY,
    f64::INFINITY,
    0.0,
    3.0,
    1.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
  ]);
  let expected = f64x8::from([
    1.0,
    f64::NEG_INFINITY,
    3.0,
    -0.0,
    2.0,
    -1.0,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
  ]);
  assert_eq!(a.fast_min(b), expected);
}

#[test]
fn impl_f64x8_min() {
  let a =
    f64x8::from([1.0, 5.0, 3.0, -0.0, 2.0, -1.0, f64::NAN, f64::INFINITY]);
  let b = f64x8::from([
    2.0,
    f64::NEG_INFINITY,
    f64::INFINITY,
    0.0,
    3.0,
    1.0,
    f64::INFINITY,
    f64::NAN,
  ]);
  let expected = f64x8::from([
    1.0,
    f64::NEG_INFINITY,
    3.0,
    0.0,
    2.0,
    -1.0,
    f64::INFINITY,
    f64::INFINITY,
  ]);
  assert_eq!(a.min(b), expected);
}

#[test]
fn impl_f64x8_is_nan() {
  let a =
    f64x8::from([0.0, f64::NAN, f64::NAN, 0.0, 1.0, -1.0, f64::NAN, f64::NAN]);
  let expected: [u64; 8] = [0, u64::MAX, u64::MAX, 0, 0, 0, u64::MAX, u64::MAX];
  let actual: [u64; 8] = cast(a.is_nan());
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_is_finite() {
  let a = f64x8::from([
    f64::NAN,
    1.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    2.0,
    3.0,
    f64::NAN,
    f64::INFINITY,
  ]);
  let expected: [u64; 8] = [0, u64::MAX, 0, 0, u64::MAX, u64::MAX, 0, 0];
  let actual: [u64; 8] = cast(a.is_finite());
  assert_eq!(actual, expected);
}

#[test]
fn impl_f64x8_round() {
  let a = f64x8::from([1.1, 2.5, 3.7, 4.0, -1.1, -2.5, -3.7, -4.0]);
  let expected = f64x8::from([1.0, 2.0, 4.0, 4.0, -1.0, -2.0, -4.0, -4.0]);
  assert_eq!(a.round(), expected);
}

#[test]
fn impl_f64x8_round_int() {
  for (f, i) in [
    (1.0, 1i64),
    (1.1, 1),
    (-2.1, -2),
    (2.5, 2),
    (0.0, 0),
    (-0.0, 0),
    (f64::NAN, 0),
    (f64::INFINITY, i64::MAX),
    (f64::NEG_INFINITY, i64::MIN),
  ] {
    let a = f64x8::from(f);
    let expected = i64x8::from(i);
    let actual = a.round_int();
    assert_eq!(actual, expected);
  }
}

#[test]
fn impl_f64x8_mul_add() {
  let a = f64x8::from([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
  let b = f64x8::from([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
  let c = f64x8::splat(1.0);
  let expected = f64x8::from([9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]);
  assert_eq!(a.mul_add(b, c), expected);
}

#[test]
fn impl_f64x8_mul_neg_add() {
  let a = f64x8::from([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
  let b = f64x8::from([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
  let c = f64x8::splat(1.0);
  let expected =
    f64x8::from([-7.0, -14.0, -23.0, -34.0, -47.0, -62.0, -79.0, -98.0]);
  assert_eq!(a.mul_neg_add(b, c), expected);
}

#[test]
fn impl_f64x8_flip_signs() {
  let a = f64x8::from([1.0, 1.0, -1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
  let b = f64x8::from([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0]);
  let expected = f64x8::from([1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
  assert_eq!(a.flip_signs(b), expected);
}

#[test]
fn impl_f64x8_copysign() {
  let a = f64x8::from([1.0, 1.0, -1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
  let b = f64x8::from([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0]);
  let expected = f64x8::from([1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
  assert_eq!(a.copysign(b), expected);
}

#[cfg(target_feature = "avx512f")]
#[test]
fn impl_f64x8_asin_acos() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f64 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let (actual_asins, actual_acoses) = f64x8::from(origs).asin_acos();
    for i in 0..8 {
      let orig = origs[i];
      let check = |name: &str, vals: f64x8, expected: f64| {
        let actual_arr: [f64; 8] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 6e-7,
          "Wanted {}({}) ≈ {} but got {}",
          name,
          orig,
          expected,
          actual
        );
      };
      check("asin", actual_asins, orig.asin());
      check("acos", actual_acoses, orig.acos());
    }
  }
}

#[cfg(target_feature = "avx512f")]
#[test]
fn impl_f64x8_asin() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f64 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let actual_asins = f64x8::from(origs).asin();
    for i in 0..8 {
      let orig = origs[i];
      let actual_arr: [f64; 8] = cast(actual_asins);
      let actual = actual_arr[i];
      let expected = orig.asin();
      assert!(
        (actual - expected).abs() < 6e-7,
        "asin({}) ≈ {} but got {}",
        orig,
        expected,
        actual
      );
    }
  }
}

#[cfg(target_feature = "avx512f")]
#[test]
fn impl_f64x8_acos() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f64 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let actual_acoses = f64x8::from(origs).acos();
    for i in 0..8 {
      let orig = origs[i];
      let actual_arr: [f64; 8] = cast(actual_acoses);
      let actual = actual_arr[i];
      let expected = orig.acos();
      assert!(
        (actual - expected).abs() < 6e-7,
        "acos({}) ≈ {} but got {}",
        orig,
        expected,
        actual
      );
    }
  }
}

#[cfg(target_feature = "avx512f")]
#[test]
fn impl_f64x8_atan() {
  let inc = 1.0 / 2501.0 / 8.0;
  for x in -2500..=2500 {
    let base = (x * 8) as f64 * inc;
    let origs = [
      base,
      base + inc,
      base + 2.0 * inc,
      base + 3.0 * inc,
      base + 4.0 * inc,
      base + 5.0 * inc,
      base + 6.0 * inc,
      base + 7.0 * inc,
    ];
    let actual_atans = f64x8::from(origs).atan();
    for i in 0..8 {
      let orig = origs[i];
      let actual_arr: [f64; 8] = cast(actual_atans);
      let actual = actual_arr[i];
      let expected = orig.atan();
      assert!(
        (actual - expected).abs() < 1e-15,
        "atan({}) ≈ {} but got {}",
        orig,
        expected,
        actual
      );
    }
  }
}

#[cfg(target_feature = "avx512f")]
#[test]
fn impl_f64x8_atan2() {
  let inc_y = 1.0 / 51.0 / 8.0;
  let inc_x = 1.0 / 2501.0 / 8.0;
  for y in -50..=50 {
    let base_y = (y * 8) as f64 * inc_y;
    let origs_y = [
      base_y,
      base_y + inc_y,
      base_y + 2.0 * inc_y,
      base_y + 3.0 * inc_y,
      base_y + 4.0 * inc_y,
      base_y + 5.0 * inc_y,
      base_y + 6.0 * inc_y,
      base_y + 7.0 * inc_y,
    ];
    let vy = f64x8::from(origs_y);
    for x in -2500..=2500 {
      let base_x = (x * 8) as f64 * inc_x;
      let origs_x = [
        base_x,
        base_x + inc_x,
        base_x + 2.0 * inc_x,
        base_x + 3.0 * inc_x,
        base_x + 4.0 * inc_x,
        base_x + 5.0 * inc_x,
        base_x + 6.0 * inc_x,
        base_x + 7.0 * inc_x,
      ];
      let vx = f64x8::from(origs_x);
      let actual = vy.atan2(vx);
      for i in 0..8 {
        let orig_y = origs_y[i];
        let orig_x = origs_x[i];
        let actual_arr: [f64; 8] = cast(actual);
        let actual_val = actual_arr[i];
        let expected = orig_y.atan2(orig_x);
        assert!(
          (actual_val - expected).abs() < 1e-15,
          "atan2({}, {}) ≈ {} but got {}",
          orig_y,
          orig_x,
          expected,
          actual_val
        );
      }
    }
  }
}

#[cfg(target_feature = "avx512f")]
#[test]
fn impl_f64x8_sin_cos() {
  for x in -2500..=2500 {
    let base = (x * 8) as f64;
    let angles = [
      base,
      base + 1.0,
      base + 2.0,
      base + 3.0,
      base + 4.0,
      base + 5.0,
      base + 6.0,
      base + 7.0,
    ];
    let (actual_sins, actual_coses) = f64x8::from(angles).sin_cos();
    for i in 0..8 {
      let angle = angles[i];
      let actual_arr_s: [f64; 8] = cast(actual_sins);
      let actual_arr_c: [f64; 8] = cast(actual_coses);
      let expected_s = angle.sin();
      let expected_c = angle.cos();
      assert!(
        (actual_arr_s[i] - expected_s).abs() < 6e-8,
        "sin({}) ≈ {} but got {}",
        angle,
        expected_s,
        actual_arr_s[i]
      );
      assert!(
        (actual_arr_c[i] - expected_c).abs() < 6e-8,
        "cos({}) ≈ {} but got {}",
        angle,
        expected_c,
        actual_arr_c[i]
      );
    }
  }
}

#[test]
fn impl_f64x8_to_degrees() {
  let pi = core::f64::consts::PI;
  let a = f64x8::from([
    0.0,
    pi / 2.0,
    pi,
    2.0 * pi,
    3.0 * pi,
    4.0 * pi,
    -pi / 4.0,
    -pi,
  ]);
  let expected =
    f64x8::from([0.0, 90.0, 180.0, 360.0, 540.0, 720.0, -45.0, -180.0]);
  assert_eq!(a.to_degrees(), expected);
}

#[test]
fn impl_f64x8_to_radians() {
  let pi = core::f64::consts::PI;
  let a = f64x8::from([0.0, 90.0, 180.0, 360.0, 450.0, 720.0, -45.0, -180.0]);
  let expected = f64x8::from([
    0.0,
    pi / 2.0,
    pi,
    2.0 * pi,
    2.5 * pi,
    4.0 * pi,
    -pi / 4.0,
    -pi,
  ]);
  assert_eq!(a.to_radians(), expected);
}

#[test]
fn impl_f64x8_sqrt() {
  for &(f, e) in &[
    (f64::INFINITY, f64::INFINITY),
    (0.0, 0.0),
    (-0.0, -0.0),
    (4.0, 2.0),
    (9.0, 3.0),
    (16.0, 4.0),
    (25.0, 5.0),
    (5000.0 * 5000.0, 5000.0),
  ] {
    let expected = f64x8::from(e);
    let actual = f64x8::from(f).sqrt();
    assert_eq!(actual, expected);
  }
  // NaNs should propagate
  let nan_out = f64x8::from(f64::NAN).sqrt();
  assert!(cast::<_, i64x8>(nan_out.is_nan()).all());
}

#[test]
fn impl_f64x8_exp() {
  for f in [(-2.0), (-1.0), (0.0), (1.0), (1.5), (2.0), (10.0)].iter().copied()
  {
    let expected = f64x8::from((f as f64).exp());
    let actual = f64x8::from(f).exp();
    let diff_from_std: [f64; 8] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.000000000000001);
  }
}

#[test]
fn test_f64x8_move_mask() {
  let a = f64x8::from([-1.0, 0.0, -2.0, -3.0, 1.0, -4.0, 0.0, -5.0]);
  // negative lanes -> bit 1
  let expected = 0b10101101;
  assert_eq!(a.move_mask(), expected);
}

#[test]
fn test_f64x8_any_all_none() {
  let m =
    f64x8::from([0.0, -1.0, 2.0, f64::NAN, 0.0, 3.0, f64::NAN, -0.0]).is_nan();
  assert!(m.any());
  assert!(!m.all());
  assert!(!m.none());

  let none = f64x8::splat(1.0).is_nan();
  assert!(!none.any());
  assert!(none.none());
}

#[test]
fn impl_f64x8_ln() {
  for &f in &[0.1, 0.5, 1.0, 2.718282, 10.0, 35.0, 1250.0] {
    let expected = f64x8::splat((f as f64).ln());
    let actual = f64x8::from(f).ln();
    let diff: [f64; 8] = cast((actual - expected).abs());
    assert!(diff.iter().all(|&d| d < 1e-12));
  }
}

#[test]
fn impl_f64x8_powf_single() {
  let base = f64x8::splat(2.0);
  for &e in &[0.1, 0.5, 1.0, 2.718282, 3.0, 4.0, 2.5, -1.0] {
    let expected = f64x8::splat(2.0_f64.powf(e));
    let actual = base.powf(e);
    let diff: [f64; 8] = cast((actual - expected).abs());
    assert!(diff.iter().all(|&d| d < 1e-6));
  }
}

#[test]
fn impl_f64x8_powf_multiple() {
  let p = f64x8::from([29.0, 0.1, 0.5, 1.0, 2.0, -0.2, -1.5, 3.4]);
  let f = f64x8::from([1.2, 2.0, 3.0, 1.5, 2.5, 4.5, 0.5, 9.2]);
  let res = f.pow_f64x8(p);
  let pp: [f64; 8] = cast(p);
  let ff: [f64; 8] = cast(f);
  let rr: [f64; 8] = cast(res);
  for i in 0..8 {
    let exp = ff[i].powf(pp[i]);
    let got = rr[i];
    if exp.is_nan() {
      assert!(got.is_nan());
    } else {
      assert!((exp - got).abs() < 1e-4);
    }
  }
}

#[test]
fn impl_f64x8_reduce_add() {
  let p = f64x8::splat(0.001);
  assert_eq!(p.reduce_add(), 0.008);
}

#[test]
fn impl_f64x8_from_i32x8() {
  let i = i32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let f = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  assert_eq!(f64x8::from(i), f);
  assert_eq!(f64x8::from_i32x8(i), f);
}

#[cfg(feature = "serde")]
#[test]
fn impl_f64x8_ser_de_roundtrip() {
  let serialized = bincode::serialize(&f64x8::ZERO).unwrap();
  let deserialized: f64x8 = bincode::deserialize(&serialized).unwrap();
  assert_eq!(f64x8::ZERO, deserialized);
}
