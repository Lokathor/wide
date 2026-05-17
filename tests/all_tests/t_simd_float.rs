use wide::{f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i32x4, i32x8, i32x16};

use crate::utils::{for_simd_types, random_iter, simd_chunks};

#[test]
fn test_abs() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NEG_INFINITY,
      6.0,
      15.0,
      -19.0,
      -9.0,
      4.5,
      -20.0,
      T::INFINITY,
      5.0,
      -4.0,
      13.0,
      9.5,
      -3.0,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(value.map(T::abs));
      let actual = Simd::new(value).abs();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_signum() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0,
      -0.0,
      1.0,
      -1.0,
      24.01,
      -24.01,
      T::MAX,
      T::MIN,
      T::INFINITY,
      T::NEG_INFINITY,
      T::NAN,
      T::NAN,
      24.01,
      -24.01,
      T::MAX,
      T::MIN,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(value.map(T::signum));
      let actual = Simd::new(value).signum();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_floor() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.1,
      60.9,
      1.1,
      T::INFINITY,
      96.6,
      -53.2,
      0.1,
      9.2,
      6.9,
      -3.4,
      85.3,
      -79.8,
      4.2,
      -6.4,
      7.3,
      -9.1,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(value.map(T::floor));
      let actual = Simd::new(value).floor();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_ceil() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.1,
      60.9,
      1.1,
      T::INFINITY,
      96.6,
      -53.2,
      0.1,
      9.2,
      6.9,
      -3.4,
      85.3,
      -79.8,
      4.2,
      -6.4,
      7.3,
      -9.1,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(value.map(T::ceil));
      let actual = Simd::new(value).ceil();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_fast_max() {
  for_simd_types!(|T: Float, N| {
    for [value, other] in simd_chunks!(
      [1.0, 5.0, 3.0, 0.0, 6.0, -8.0, 12.0, 9.0, 2.0, -3.0, T::INFINITY],
      [2.0, -3.0, T::INFINITY, 10.0, 19.0, -5.0, -1.0, -9.0, 1.0, 5.0, 3.0],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| value[i].max(other[i])));
      let actual = Simd::new(value).fast_max(Simd::new(other));

      assert_eq!(
        actual ^ expected,
        Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}"
      );
    }
  });
}

#[test]
fn test_max() {
  for_simd_types!(|T: Float, N| {
    for [value, other] in simd_chunks!(
      [
        1.0,
        5.0,
        3.0,
        0.0,
        6.0,
        -8.0,
        T::NAN,
        T::NEG_INFINITY,
        2.0,
        -3.0,
        T::INFINITY
      ],
      [
        2.0,
        -3.0,
        T::NAN,
        10.0,
        19.0,
        -5.0,
        T::NAN,
        -9.0,
        T::INFINITY,
        T::NEG_INFINITY,
        3.0
      ],
    )
    .chain(random_iter())
    {
      let expected = Simd::new(std::array::from_fn(|i| value[i].max(other[i])));
      let actual = Simd::new(value).max(Simd::new(other));

      assert_eq!(
        actual ^ expected,
        Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}"
      );
    }
  });
}

#[test]
fn test_fast_min() {
  for_simd_types!(|T: Float, N| {
    for [value, other] in simd_chunks!(
      [1.0, 5.0, 3.0, 0.0, 6.0, -8.0, 12.0, 9.0, 2.0, -3.0, T::INFINITY],
      [2.0, -3.0, T::INFINITY, 10.0, 19.0, -5.0, -1.0, -9.0, 1.0, 5.0, 3.0],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| value[i].min(other[i])));
      let actual = Simd::new(value).fast_min(Simd::new(other));

      assert_eq!(
        actual ^ expected,
        Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}"
      );
    }
  });
}

#[test]
fn test_min() {
  for_simd_types!(|T: Float, N| {
    for [value, other] in simd_chunks!(
      [
        1.0,
        5.0,
        3.0,
        0.0,
        6.0,
        -8.0,
        T::NAN,
        T::NEG_INFINITY,
        2.0,
        -3.0,
        T::INFINITY
      ],
      [
        2.0,
        -3.0,
        T::NAN,
        10.0,
        19.0,
        -5.0,
        T::NAN,
        -9.0,
        T::INFINITY,
        T::NEG_INFINITY,
        3.0
      ],
    )
    .chain(random_iter())
    {
      let expected = Simd::new(std::array::from_fn(|i| value[i].min(other[i])));
      let actual = Simd::new(value).min(Simd::new(other));

      assert_eq!(
        actual ^ expected,
        Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}"
      );
    }
  });
}

#[test]
fn test_clamp() {
  for_simd_types!(|T: Float, N| {
    for [value, min, max] in simd_chunks!(
      [5.0, 10.0, 10.0, T::NAN, T::INFINITY, T::NEG_INFINITY],
      [3.0, 11.0, 5.0, 1.0, -1.0, -1.0],
      [8.0, 14.0, 9.0, 3.0, 1.0, 1.0],
    ) {
      let expected =
        Simd::new(std::array::from_fn(|i| value[i].clamp(min[i], max[i])));
      let actual = Simd::new(value).clamp(Simd::new(min), Simd::new(max));

      assert_eq!(
        actual ^ expected,
        Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}"
      );
    }
  });
}

#[test]
fn test_midpoint() {
  for_simd_types!(|T: Float, N| {
    for [value, other] in simd_chunks!(
      [
        5.2,
        -16349.0,
        3467890356635.1,
        2401.0,
        -21.0,
        -236456708943.0,
        2340894786738.2,
        -4235.0,
        -21.0,
      ],
      [
        -21.0,
        -236456708943.0,
        2340894786738.2,
        -4235.0,
        5.2,
        -16349.0,
        3467890356635.1,
        2401.0,
        5.2,
      ],
    ) {
      let expected =
        Simd::new(std::array::from_fn(|i| value[i].midpoint(other[i])));
      let actual = Simd::new(value).midpoint(Simd::new(other));

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_is_nan() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NAN,
      6.0,
      15.0,
      T::NAN,
      T::NEG_INFINITY,
      T::INFINITY,
      16.0,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(
        value.map(|x| if x.is_nan() { T::from_bits(!0) } else { 0.0 }),
      );
      let actual = Simd::new(value).is_nan();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_is_finite() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NAN,
      6.0,
      15.0,
      T::NAN,
      T::NEG_INFINITY,
      T::INFINITY,
      16.0,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(
        value.map(|x| if x.is_finite() { T::from_bits(!0) } else { 0.0 }),
      );
      let actual = Simd::new(value).is_finite();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_is_inf() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NAN,
      6.0,
      15.0,
      T::NAN,
      T::NEG_INFINITY,
      T::INFINITY,
      16.0,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(
        value.map(|x| if x.is_infinite() { T::from_bits(!0) } else { 0.0 }),
      );
      let actual = Simd::new(value).is_inf();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_round() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0,
      0.1,
      0.5,
      0.7,
      -0.0,
      -0.1,
      -0.5,
      -0.7,
      2.0,
      2.1,
      2.5,
      2.7,
      -2.0,
      -2.1,
      -2.5,
      -2.7,
      5.0,
      5.1,
      5.5,
      5.7,
      -5.0,
      -5.1,
      -5.5,
      -5.7,
      T::MAX,
      T::MIN,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ])
    .chain(random_iter())
    {
      // TODO:  Currently `round` actually behaves like `round_ties_even`.
      // Decide the correct behavior then add documentation.
      let expected = Simd::new(value.map(T::round_ties_even));
      let actual = Simd::new(value).round();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}",
      );
    }
  });
}

#[test]
fn test_fast_round_int() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0, 0.1, 0.5, 0.7, -0.0, -0.1, -0.5, -0.7, 2.0, 2.1, 2.5, 2.7, -2.0,
      -2.1, -2.5, -2.7, 5.0, 5.1, 5.5, 5.7, -5.0, -5.1, -5.5, -5.7,
    ]) {
      // TODO:  Currently `round` actually behaves like `round_ties_even`.
      // Decide the correct behavior then add documentation.
      let expected =
        SimdSigned::new(value.map(|x| x.round_ties_even() as Signed));
      let actual = Simd::new(value).fast_round_int();

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_round_int() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0,
      0.1,
      0.5,
      0.7,
      -0.0,
      -0.1,
      -0.5,
      -0.7,
      2.0,
      2.1,
      2.5,
      2.7,
      -2.0,
      -2.1,
      -2.5,
      -2.7,
      5.0,
      5.1,
      5.5,
      5.7,
      -5.0,
      -5.1,
      -5.5,
      -5.7,
      T::MAX,
      T::MIN,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ])
    .chain(random_iter())
    {
      // TODO:  Currently `round` actually behaves like `round_ties_even`.
      // Decide the correct behavior then add documentation.
      let expected = SimdSigned::new(value.map(|x| {
        x.round_ties_even().clamp(Signed::MIN as T, Signed::MAX as T) as Signed
      }));
      let actual = Simd::new(value).round_int();

      assert!(
        actual == expected,
        "\nexpected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}",
      );
    }
  });
}

#[test]
fn test_trunc() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0,
      0.1,
      0.5,
      0.7,
      -0.0,
      -0.1,
      -0.5,
      -0.7,
      2.0,
      2.1,
      2.5,
      2.7,
      -2.0,
      -2.1,
      -2.5,
      -2.7,
      5.0,
      5.1,
      5.5,
      5.7,
      -5.0,
      -5.1,
      -5.5,
      -5.7,
      2401.63,
      -2401.63,
      4911111.2,
      -4911111.2,
      18388608.0,
      18388608.0,
      9223372036854775807.0,
      -9223372036854775808.0,
      T::MAX,
      T::MIN,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(value.map(T::trunc));
      let actual = Simd::new(value).trunc();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_fast_trunc_int() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0, 0.1, 0.5, 0.7, -0.0, -0.1, -0.5, -0.7, 2.0, 2.1, 2.5, 2.7, -2.0,
      -2.1, -2.5, -2.7, 5.0, 5.1, 5.5, 5.7, -5.0, -5.1, -5.5, -5.7,
    ]) {
      let expected = SimdSigned::new(value.map(|x| x.trunc() as Signed));
      let actual = Simd::new(value).fast_trunc_int();

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_trunc_int() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0,
      0.1,
      0.5,
      0.7,
      -0.0,
      -0.1,
      -0.5,
      -0.7,
      2.0,
      2.1,
      2.5,
      2.7,
      -2.0,
      -2.1,
      -2.5,
      -2.7,
      5.0,
      5.1,
      5.5,
      5.7,
      -5.0,
      -5.1,
      -5.5,
      -5.7,
      2401.63,
      -2401.63,
      4911111.2,
      -4911111.2,
      18388608.0,
      18388608.0,
      9223372036854775807.0,
      -9223372036854775808.0,
      T::MAX,
      T::MIN,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ])
    .chain(random_iter())
    {
      let expected = SimdSigned::new(value.map(|x| {
        x.trunc().clamp(Signed::MIN as T, Signed::MAX as T) as Signed
      }));
      let actual = Simd::new(value).trunc_int();

      assert!(
        actual == expected,
        "\nexpected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}",
      );
    }
  });
}

#[test]
fn test_fract() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.0,
      0.1,
      0.5,
      0.7,
      -0.0,
      -0.1,
      -0.5,
      -0.7,
      2.0,
      2.1,
      2.5,
      2.7,
      -2.0,
      -2.1,
      -2.5,
      -2.7,
      5.0,
      5.1,
      5.5,
      5.7,
      -5.0,
      -5.1,
      -5.5,
      -5.7,
      T::MAX,
      T::MIN,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ])
    .chain(random_iter())
    {
      let expected = Simd::new(value.map(T::fract));
      let actual = Simd::new(value).fract();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_mul_add() {
  for_simd_types!(|T: Float, N| {
    for [value, a, b] in simd_chunks!(
      [
        2.0,
        3.0,
        4.0,
        5.0,
        6.7,
        9.2,
        11.5,
        12.2,
        1.0,
        2.0,
        -34578.0,
        4.0,
        5.0,
        6.0,
        7.0,
        4538093452.0,
      ],
      [
        4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, 5.6, 2.0, 3.0, 23.0, 5.0, 6.0, 7.0,
        8.0, 9.0,
      ],
      [
        1.5, 8.9, 4.2, 5.6, 2.0, 3.5, 4.0, 5.1, 9.0, 4.0, 5.32, 6.03, 7.12,
        8.0, 6.0, 53.0,
      ],
    ) {
      let expected =
        Simd::new(std::array::from_fn(|i| value[i].mul_add(a[i], b[i])));
      let actual = Simd::new(value).mul_add(Simd::new(a), Simd::new(b));

      assert!(
        (actual - expected).abs().simd_le(expected.abs() * 1e-6).all(),
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
      );

      #[cfg(any(
        all(
          target_feature = "fma",
          any(target_arch = "x86", target_arch = "x86_64"),
        ),
        all(target_feature = "neon", target_arch = "aarch64"),
      ))]
      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
      );
    }
  });
}

#[test]
fn test_mul_neg_add() {
  for_simd_types!(|T: Float, N| {
    for [value, a, b] in simd_chunks!(
      [
        2.0,
        3.0,
        4.0,
        5.0,
        6.7,
        9.2,
        11.5,
        12.2,
        1.0,
        2.0,
        -34578.0,
        4.0,
        5.0,
        6.0,
        7.0,
        4538093452.0,
      ],
      [
        4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, 5.6, 2.0, 3.0, 23.0, 5.0, 6.0, 7.0,
        8.0, 9.0,
      ],
      [
        1.5, 8.9, 4.2, 5.6, 2.0, 3.5, 4.0, 5.1, 9.0, 4.0, 5.32, 6.03, 7.12,
        8.0, 6.0, 53.0,
      ],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| b[i] - value[i] * a[i]));
      let actual = Simd::new(value).mul_neg_add(Simd::new(a), Simd::new(b));

      assert!(
        (actual - expected).abs().simd_le(expected.abs() * 1e-6).all(),
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
      );
    }
  });
}

#[test]
fn test_mul_sub() {
  for_simd_types!(|T: Float, N| {
    for [value, a, b] in simd_chunks!(
      [
        2.0,
        3.0,
        4.0,
        5.0,
        6.7,
        9.2,
        11.5,
        12.2,
        1.0,
        2.0,
        -34578.0,
        4.0,
        5.0,
        6.0,
        7.0,
        4538093452.0,
      ],
      [
        4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, 5.6, 2.0, 3.0, 23.0, 5.0, 6.0, 7.0,
        8.0, 9.0,
      ],
      [
        1.5, 8.9, 4.2, 5.6, 2.0, 3.5, 4.0, 5.1, 9.0, 4.0, 5.32, 6.03, 7.12,
        8.0, 6.0, 53.0,
      ],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| value[i] * a[i] - b[i]));
      let actual = Simd::new(value).mul_sub(Simd::new(a), Simd::new(b));

      assert!(
        (actual - expected).abs().simd_le(expected.abs() * 1e-6).all(),
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
      );
    }
  });
}

#[test]
fn test_mul_neg_sub() {
  for_simd_types!(|T: Float, N| {
    for [value, a, b] in simd_chunks!(
      [
        2.0,
        3.0,
        4.0,
        5.0,
        6.7,
        9.2,
        11.5,
        12.2,
        1.0,
        2.0,
        -34578.0,
        4.0,
        5.0,
        6.0,
        7.0,
        4538093452.0,
      ],
      [
        4.0, 5.0, 6.0, 7.0, 1.5, 8.9, 4.2, 5.6, 2.0, 3.0, 23.0, 5.0, 6.0, 7.0,
        8.0, 9.0,
      ],
      [
        1.5, 8.9, 4.2, 5.6, 2.0, 3.5, 4.0, 5.1, 9.0, 4.0, 5.32, 6.03, 7.12,
        8.0, 6.0, 53.0,
      ],
    ) {
      let expected =
        Simd::new(std::array::from_fn(|i| -value[i] * a[i] - b[i]));
      let actual = Simd::new(value).mul_neg_sub(Simd::new(a), Simd::new(b));

      assert!(
        (actual - expected).abs().simd_le(expected.abs() * 1e-6).all(),
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
      );
    }
  });
}

#[test]
fn test_div_euclid() {
  for_simd_types!(|T: Float, N| {
    for [left, right] in simd_chunks!(
      [
        4.0, 9.0, 10.0, 12.0, 5.0, 6.0, 7.0, 8.24, 18.0, 20.0, 15.0, 16.4,
        -21.0, 24.0, -30.0, 32.0,
      ],
      [
        2.0, 2.0, -5.0, -3.0, 2.0, 1.5, 3.0, -2.5, 3.5, 4.0, 5.1, 8.0, 7.68,
        6.0, 10.0, -16.0,
      ],
    ) {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].div_euclid(right[i])));
      let actual = Simd::new(left).div_euclid(Simd::new(right));

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_rem_euclid() {
  for_simd_types!(|T: Float, N| {
    for [left, right] in simd_chunks!(
      [
        4.0, 9.0, 10.0, 12.0, 5.0, 6.0, 7.0, 8.24, 18.0, 20.0, 15.0, 16.4,
        -21.0, 24.0, -30.0, 32.0,
      ],
      [
        2.0, 2.0, -5.0, -3.0, 2.0, 1.5, 3.0, -2.5, 3.5, 4.0, 5.1, 8.0, 7.68,
        6.0, 10.0, -16.0,
      ],
    ) {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].rem_euclid(right[i])));
      let actual = Simd::new(left).rem_euclid(Simd::new(right));

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_flip_signs() {
  for_simd_types!(|T: Float, N| {
    for [value, sign] in simd_chunks!(
      [
        1.0, 1.0, -1.0, -1.0, 5.2, 6.7, -8.2, -12.5, 3.0, -6.4, 7.2, -24.01,
        3.2, 1.6, -0.8, 0.4,
      ],
      [
        2.0, -3.0, 4.0, -5.0, 5.2, 6.7, -8.2, -12.5, 3.3, -4.0, -5.5, 6.6,
        -6.9, 5.4, 3.1, -6.0,
      ],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| {
        if sign[i].is_sign_negative() { -value[i] } else { value[i] }
      }));
      let actual = Simd::new(value).flip_signs(Simd::new(sign));

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_copysign() {
  for_simd_types!(|T: Float, N| {
    for [value, sign] in simd_chunks!(
      [
        1.0, 1.0, -1.0, -1.0, 5.2, 6.7, -8.2, -12.5, 3.0, -6.4, 7.2, -24.01,
        3.2, 1.6, -0.8, 0.4,
      ],
      [
        2.0, -3.0, 4.0, -5.0, 5.2, 6.7, -8.2, -12.5, 3.3, -4.0, -5.5, 6.6,
        -6.9, 5.4, 3.1, -6.0,
      ],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| {
        if sign[i].is_sign_negative() {
          -value[i].abs()
        } else {
          value[i].abs()
        }
      }));
      let actual = Simd::new(value).copysign(Simd::new(sign));

      assert_eq!(actual, expected);
    }
  });
}

#[test]
fn test_asin_acos() {
  for_simd_types!(|T: Float, N| {
    let inc = 1.0 / 2501.0 / 8.0;
    for x in -2500..=2500 {
      let base = (x * 8) as T * inc;
      for value in simd_chunks!([
        base,
        base + inc,
        base + 2.0 * inc,
        base + 3.0 * inc,
        base + 4.0 * inc,
        base + 5.0 * inc,
        base + 6.0 * inc,
        base + 7.0 * inc,
      ]) {
        let expected =
          (Simd::new(value.map(T::asin)), Simd::new(value.map(T::acos)));
        let actual = Simd::new(value).asin_acos();

        assert!(
          (actual.0 - expected.0).abs().simd_le(6e-7).all()
            && (actual.1 - expected.1).abs().simd_le(6e-7).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_asin() {
  for_simd_types!(|T: Float, N| {
    let inc = 1.0 / 2501.0 / 8.0;
    for x in -2500..=2500 {
      let base = (x * 8) as T * inc;
      for value in simd_chunks!([
        base,
        base + inc,
        base + 2.0 * inc,
        base + 3.0 * inc,
        base + 4.0 * inc,
        base + 5.0 * inc,
        base + 6.0 * inc,
        base + 7.0 * inc,
      ]) {
        let expected = Simd::new(value.map(T::asin));
        let actual = Simd::new(value).asin();

        assert!(
          (actual - expected).abs().simd_le(6e-7).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_acos() {
  for_simd_types!(|T: Float, N| {
    let inc = 1.0 / 2501.0 / 8.0;
    for x in -2500..=2500 {
      let base = (x * 8) as T * inc;
      for value in simd_chunks!([
        base,
        base + inc,
        base + 2.0 * inc,
        base + 3.0 * inc,
        base + 4.0 * inc,
        base + 5.0 * inc,
        base + 6.0 * inc,
        base + 7.0 * inc,
      ]) {
        let expected = Simd::new(value.map(T::acos));
        let actual = Simd::new(value).acos();

        assert!(
          (actual - expected).abs().simd_le(6e-7).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_atan() {
  for_simd_types!(|T: Float, N| {
    let inc = 1.0 / 2501.0 / 8.0;
    for x in -2500..=2500 {
      let base = (x * 8) as T * inc;
      for value in simd_chunks!([
        base,
        base + inc,
        base + 2.0 * inc,
        base + 3.0 * inc,
        base + 4.0 * inc,
        base + 5.0 * inc,
        base + 6.0 * inc,
        base + 7.0 * inc,
      ]) {
        let expected = Simd::new(value.map(T::atan));
        let actual = Simd::new(value).atan();

        let tol = if size_of::<T>() == 8 { 1e-15 } else { 6e-7 };
        assert!(
          (actual - expected).abs().simd_le(tol).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_atan2() {
  for_simd_types!(|T: Float, N| {
    let inc_y = 1.0 / 51.0 / 8.0;
    let inc_x = 1.0 / 2501.0 / 8.0;
    for y in -50..=50 {
      let base_y = (y * 8) as T * inc_y;
      for x in (-2500..=-2400).chain(-50..=50).chain(2400..=2500) {
        let base_x = (x * 8) as T * inc_x;
        for [value, other] in simd_chunks!(
          [base_y, base_y + inc_y, base_y + 2.0 * inc_y, base_y + 3.0 * inc_y],
          [base_x, base_x + inc_x, base_x + 2.0 * inc_x, base_x + 3.0 * inc_x],
        ) {
          let expected =
            Simd::new(std::array::from_fn(|i| value[i].atan2(other[i])));
          let actual = Simd::new(value).atan2(Simd::new(other));

          let tol = if size_of::<T>() == 8 { 1e-15 } else { 6e-7 };
          assert!(
            (actual - expected).abs().simd_le(tol).all(),
            "expected: {expected:?}\n  actual: {actual:?}",
          );
        }
      }
    }
  });
}

#[test]
fn test_sin_cos() {
  for_simd_types!(|T: Float, N| {
    for x in -2500..=2500 {
      let base = (x * 4) as T;
      for angles in simd_chunks!([
        base,
        base + 1.0,
        base + 2.0,
        base + 3.0,
        base + 4.0,
        base + 5.0,
        base + 6.0,
        base + 7.0,
      ]) {
        let expected =
          (Simd::new(angles.map(T::sin)), Simd::new(angles.map(T::cos)));
        let actual = Simd::new(angles).sin_cos();

        let tol = if size_of::<T>() == 8 { 6e-8 } else { 2e-7 };
        assert!(
          (actual.0 - expected.0).abs().simd_le(tol).all()
            && (actual.1 - expected.1).abs().simd_le(tol).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_sin() {
  for_simd_types!(|T: Float, N| {
    for x in -2500..=2500 {
      let base = (x * 4) as T;
      for angles in simd_chunks!([
        base,
        base + 1.0,
        base + 2.0,
        base + 3.0,
        base + 4.0,
        base + 5.0,
        base + 6.0,
        base + 7.0,
      ]) {
        let expected = Simd::new(angles.map(T::sin));
        let actual = Simd::new(angles).sin();

        let tol = if size_of::<T>() == 8 { 6e-8 } else { 2e-7 };
        assert!(
          (actual - expected).abs().simd_le(tol).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_cos() {
  for_simd_types!(|T: Float, N| {
    for x in -2500..=2500 {
      let base = (x * 4) as T;
      for angles in simd_chunks!([
        base,
        base + 1.0,
        base + 2.0,
        base + 3.0,
        base + 4.0,
        base + 5.0,
        base + 6.0,
        base + 7.0,
      ]) {
        let expected = Simd::new(angles.map(T::cos));
        let actual = Simd::new(angles).cos();

        let tol = if size_of::<T>() == 8 { 6e-8 } else { 2e-7 };
        assert!(
          (actual - expected).abs().simd_le(tol).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_tan() {
  for_simd_types!(|T: Float, N| {
    for x in -2500..=2500 {
      let base = (x * 4) as T;
      for angles in simd_chunks!([
        base,
        base + 1.0,
        base + 2.0,
        base + 3.0,
        base + 4.0,
        base + 5.0,
        base + 6.0,
        base + 7.0,
      ]) {
        let expected = Simd::new(angles.map(T::tan));
        let actual = Simd::new(angles).tan();

        assert!(
          (actual - expected).abs().simd_le(expected.abs() * 1e-6).all(),
          "expected: {expected:?}\n  actual: {actual:?}",
        );
      }
    }
  });
}

#[test]
fn test_to_degrees() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NEG_INFINITY,
      6.0,
      15.0,
      -19.0,
      -9.0,
      4.5,
      -20.0,
      T::INFINITY,
      5.0,
      -4.0,
      13.0,
      9.5,
      -3.0,
    ]) {
      let expected = Simd::new(value.map(T::to_degrees));
      let actual = Simd::new(value).to_degrees();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-6)
          | actual.simd_eq(expected))
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_to_radians() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NEG_INFINITY,
      6.0,
      15.0,
      -19.0,
      -9.0,
      4.5,
      -20.0,
      T::INFINITY,
      5.0,
      -4.0,
      13.0,
      9.5,
      -3.0,
    ]) {
      let expected = Simd::new(value.map(T::to_radians));
      let actual = Simd::new(value).to_radians();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-6)
          | actual.simd_eq(expected))
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_recip() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NEG_INFINITY,
      T::NAN,
      15.0,
      -19e5,
      -9.0,
      4.5e10,
      -20.0,
      T::INFINITY,
      5.0,
      -4.0,
      13.343433,
      9.5,
      -3.0,
    ]) {
      let expected = Simd::new(value.map(T::recip));
      let actual = Simd::new(value).recip();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-3)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_recip_sqrt() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NEG_INFINITY,
      T::NAN,
      15.0,
      -19e5,
      -9.0,
      4.5e10,
      -20.0,
      T::INFINITY,
      5.0,
      -4.0,
      13.343433,
      9.5,
      -3.0,
    ]) {
      let expected = Simd::new(value.map(|x| x.sqrt().recip()));
      let actual = Simd::new(value).recip_sqrt();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-3)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_sqrt() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -1.0,
      2.0,
      -3.5,
      T::NEG_INFINITY,
      T::NAN,
      15.0,
      -19e5,
      -9.0,
      4.5e10,
      -20.0,
      T::INFINITY,
      5.0,
      -4.0,
      13.343433,
      9.5,
      -3.0,
    ]) {
      let expected = Simd::new(value.map(T::sqrt));
      let actual = Simd::new(value).sqrt();

      assert_eq!(
        actual ^ expected,
        Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}"
      );
    }
  });
}

#[test]
fn test_exp() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -2.0,
      -1.1,
      0.0,
      1.3,
      1.5,
      2.0,
      10.4,
      2000.0,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ]) {
      let expected = Simd::new(value.map(T::exp));
      let actual = Simd::new(value).exp();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-7)
          | actual.simd_eq(expected)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_exp2() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      -2.0,
      -1.1,
      0.0,
      1.3,
      1.5,
      2.0,
      10.4,
      2000.0,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ]) {
      let expected = Simd::new(value.map(T::exp2));
      let actual = Simd::new(value).exp2();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-7)
          | actual.simd_eq(expected)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_ln() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.1,
      0.5,
      1.0,
      2.718282,
      10.0,
      35.0,
      1250.0,
      0.0,
      -1.0,
      -2401.0,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ]) {
      let expected = Simd::new(value.map(T::ln));
      let actual = Simd::new(value).ln();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-7)
          | actual.simd_eq(expected)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_log2() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.1,
      0.5,
      1.0,
      2.718282,
      10.0,
      35.0,
      1250.0,
      0.0,
      -1.0,
      -2401.0,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ]) {
      let expected = Simd::new(value.map(T::log2));
      let actual = Simd::new(value).log2();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-7)
          | actual.simd_eq(expected)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_log10() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([
      0.1,
      0.5,
      1.0,
      2.718282,
      10.0,
      35.0,
      1250.0,
      0.0,
      -1.0,
      -2401.0,
      T::NAN,
      T::INFINITY,
      T::NEG_INFINITY,
    ]) {
      let expected = Simd::new(value.map(T::log10));
      let actual = Simd::new(value).log10();

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-7)
          | actual.simd_eq(expected)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_pow_simd() {
  // TODO: fix `powf` which currently breaks when, possibly among other cases,
  // `self` is negative and `n` is an odd number. These inputs lead to an
  // incorrect result:
  //
  // simd_chunks!(
  //  [
  //    1.2, 2.0, 3.0, 1.5, 9.2, 6.1, 2.5, 5.3, -4.5, -5.1, -5.2, -5.3, -3.0,
  //    -3.1, -3.0, -4.0, 5.1,
  //  ],
  //  [
  //    0.1, 0.5, 1.0, 2.718282, 3.0, 4.0, 2.5, -1.0, 1.4, 2.0, 1.0, 3.0, 0.1,
  //    2.7, 4.0, -3.0, 29.0,
  //  ],
  // )

  for_simd_types!(|T: Float, N| {
    for [value, n] in simd_chunks!(
      [1.2, 2.0, 3.0, 1.5, 9.2, 6.1, 2.5, 5.3, 5.1],
      [0.1, 0.5, 1.0, 2.718282, 3.0, 4.0, 2.5, -1.0, 29.0],
    ) {
      let expected = Simd::new(std::array::from_fn(|i| value[i].powf(n[i])));
      let actual = pow_simd(Simd::new(value), Simd::new(n));

      assert!(
        ((actual - expected).abs().simd_le(expected.abs() * 1e-6)
          | actual.is_nan() & expected.is_nan())
        .all(),
        "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       n: {n:?}",
      );
    }
  });
}

#[test]
fn test_powf() {
  // TODO: fix `powf` which currently breaks when, possibly among other cases,
  // `self` is negative and `n` is an odd number. These inputs lead to an
  // incorrect result:
  //
  // simd_chunks!(
  //  [
  //    1.2, 2.0, 3.0, 1.5, 9.2, 6.1, 2.5, 5.3, -4.5, -5.1, -5.2, -5.3, -3.0,
  //    -3.1, -3.0, -4.0, 5.1,
  //  ],
  //  [
  //    0.1, 0.5, 1.0, 2.718282, 3.0, 4.0, 2.5, -1.0, 1.4, 2.0, 1.0, 3.0, 0.1,
  //    2.7, 4.0, -3.0, 29.0,
  //  ],
  // )

  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([1.2, 2.0, 3.0, 1.5, 9.2, 6.1, 2.5, 5.3]) {
      for n in [
        0.1, 0.5, 1.0, 2.718282, 3.0, 4.0, 2.5, -1.0, 1.4, 2.0, 1.0, 3.0, 2.7,
        4.0, -3.0, 29.0,
      ] {
        let expected = Simd::new(std::array::from_fn(|i| value[i].powf(n)));
        let actual = Simd::new(value).powf(n);

        assert!(
          ((actual - expected).abs().simd_le(expected.abs() * 1e-6)
            | actual.is_nan() & expected.is_nan())
          .all(),
          "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       n: {n:?}",
        );
      }
    }
  });
}

#[test]
fn test_is_sign_positive() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([2401.0, -123.0, T::INFINITY, T::NEG_INFINITY])
      .chain(random_iter())
    {
      let expected = Simd::new(
        value
          .map(|x| if x.is_sign_positive() { T::from_bits(!0) } else { 0.0 }),
      );
      let actual = Simd::new(value).is_sign_positive();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_is_sign_negative() {
  for_simd_types!(|T: Float, N| {
    for value in simd_chunks!([2401.0, -123.0, T::INFINITY, T::NEG_INFINITY])
      .chain(random_iter())
    {
      let expected = Simd::new(
        value
          .map(|x| if x.is_sign_negative() { T::from_bits(!0) } else { 0.0 }),
      );
      let actual = Simd::new(value).is_sign_negative();

      assert!(
        actual ^ expected == Simd::ZERO,
        "expected: {expected:?}\n  actual: {actual:?}",
      );
    }
  });
}

#[test]
fn test_from_signed() {
  // `from_{signed}` is inconsistently missing from types.

  let signed = i32x4::new([1, 2, 3, 4]);
  let expected = f32x4::new([1.0, 2.0, 3.0, 4.0]);
  let actual = f32x4::from_i32x4(signed);
  assert_eq!(actual, expected);

  let signed = i32x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
  let expected = f32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  let actual = f32x8::from_i32x8(signed);
  assert_eq!(actual, expected);

  let signed =
    i32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  let expected = f32x16::new([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]);
  let actual = f32x16::from_i32x16(signed);
  assert_eq!(actual, expected);
}

#[test]
fn test_from_small_signed() {
  // This only exists for `f64` types.

  let value = i32x4::from([1, 2, 3, 4]);
  let expected = f64x4::from([1.0, 2.0, 3.0, 4.0]);
  assert_eq!(f64x4::from(value), expected);
  assert_eq!(f64x4::from_i32x4(value), expected);

  let value = i32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
  let expected = f64x8::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  assert_eq!(f64x8::from(value), expected);
  assert_eq!(f64x8::from_i32x8(value), expected);
}

#[test]
fn test_from_i32x4_lower2() {
  // `from_i32x4_lower2` only exists for `f64x2`.

  let value = i32x4::new([1, 2, 3, 4]);
  let expected = f64x2::new([1.0, 2.0]);
  let actual = f64x2::from_i32x4_lower2(value);
  assert_eq!(actual, expected);
}
