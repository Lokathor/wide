use wide::{f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i32x4, i32x8, i32x16};

use crate::utils::{for_simd_types, fuzz_iter};

#[test]
fn test_reduce_add() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = value.into_iter().sum::<T>();
            let actual = Simd::from_array(value).reduce_add();

            assert!(
                (actual - expected).abs() <= expected.abs() * 1e-5,
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_reduce_mul() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = value.into_iter().product::<T>();
            let actual = Simd::from_array(value).reduce_mul();

            assert!(
                (actual - expected).abs() <= expected.abs() * 1e-5,
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_abs() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::abs));
            let actual = Simd::from_array(value).abs();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::signum));
            let actual = Simd::from_array(value).signum();

            assert!(
                (actual.simd_eq(expected) | actual.is_nan() & expected.is_nan()).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_floor() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::floor));
            let actual = Simd::from_array(value).floor();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::ceil));
            let actual = Simd::from_array(value).ceil();

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
        for [value, other] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].max(other[i])));
            let actual = Simd::from_array(value).fast_max(Simd::from_array(other));

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
        for [value, other] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].max(other[i])));
            let actual = Simd::from_array(value).max(Simd::from_array(other));

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
        for [value, other] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].min(other[i])));
            let actual = Simd::from_array(value).fast_min(Simd::from_array(other));

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
        for [value, other] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].min(other[i])));
            let actual = Simd::from_array(value).min(Simd::from_array(other));

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
        for [value, min, max] in fuzz_iter::<[[T; N]; 3]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| {
                if value[i].is_nan() || min[i].is_nan() || max[i].is_nan() {
                    T::NAN
                } else if min[i] > max[i] {
                    min[i]
                } else {
                    value[i].clamp(min[i], max[i])
                }
            }));
            let actual =
                Simd::from_array(value).clamp(Simd::from_array(min), Simd::from_array(max));

            assert!(
                (actual.simd_eq(expected) | expected.is_nan() & actual.is_nan()).all(),
                "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n     min: {min:?}\n     max: {max:?}"
            );
        }
    });
}

#[test]
fn test_fast_clamp() {
    for_simd_types!(|T: Float, N| {
        for [value, mut min, mut max] in fuzz_iter::<[[T; N]; 3]>() {
            for i in 0..N {
                if min[i].is_nan() {
                    min[i] = 0.0;
                }
                if max[i].is_nan() {
                    max[i] = 0.0;
                }
            }

            let expected = Simd::from_array(std::array::from_fn(|i| {
                if value[i].is_nan() {
                    T::NAN
                } else if min[i] > max[i] {
                    min[i]
                } else {
                    value[i].clamp(min[i], max[i])
                }
            }));
            let actual =
                Simd::from_array(value).fast_clamp(Simd::from_array(min), Simd::from_array(max));

            assert!(
                (actual.simd_eq(expected) | expected.is_nan() & actual.is_nan()).all(),
                "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n     min: {min:?}\n     max: {max:?}"
            );
        }
    });
}

#[test]
fn test_reduce_max() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let actual = Simd::from_array(value).reduce_max();
            let expected = value.into_iter().reduce(T::max).unwrap();

            assert!(
                actual == expected || actual.is_nan() && expected.is_nan(),
                "  actual: {actual:?}\nexpected: {expected:?}\n   value: {value:?}"
            );
        }
    });
}

#[test]
fn test_fast_reduce_max() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let actual = Simd::from_array(value).fast_reduce_max();
            let expected = value.into_iter().reduce(T::max).unwrap();

            assert!(
                actual == expected
                    || value.into_iter().any(T::is_nan)
                        && (value.contains(&actual) || actual.is_nan()),
                "  actual: {actual:?}\nexpected: {expected:?}\n   value: {value:?}"
            );
        }
    });
}

#[test]
fn test_reduce_min() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let actual = Simd::from_array(value).reduce_min();
            let expected = value.into_iter().reduce(T::min).unwrap();

            assert!(
                actual == expected || actual.is_nan() && expected.is_nan(),
                "  actual: {actual:?}\nexpected: {expected:?}\n   value: {value:?}"
            );
        }
    });
}

#[test]
fn test_fast_reduce_min() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let actual = Simd::from_array(value).fast_reduce_min();
            let expected = value.into_iter().reduce(T::min).unwrap();

            assert!(
                actual == expected
                    || value.into_iter().any(T::is_nan)
                        && (value.contains(&actual) || actual.is_nan()),
                "  actual: {actual:?}\nexpected: {expected:?}\n   value: {value:?}"
            );
        }
    });
}

#[test]
fn test_midpoint() {
    for_simd_types!(|T: Float, N| {
        for [value, other] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].midpoint(other[i])));
            let actual = Simd::from_array(value).midpoint(Simd::from_array(other));

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_is_nan() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected =
                Simd::from_array(value.map(|x| if x.is_nan() { T::from_bits(!0) } else { 0.0 }));
            let actual = Simd::from_array(value).is_nan();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected =
                Simd::from_array(value.map(|x| if x.is_finite() { T::from_bits(!0) } else { 0.0 }));
            let actual = Simd::from_array(value).is_finite();

            assert!(
                actual ^ expected == Simd::ZERO,
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_is_infinite() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(|x| {
                if x.is_infinite() {
                    T::from_bits(!0)
                } else {
                    0.0
                }
            }));
            let actual = Simd::from_array(value).is_infinite();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::round));
            let actual = Simd::from_array(value).round();

            assert!(
                actual ^ expected == Simd::ZERO,
                "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}",
            );
        }
    });
}

#[test]
fn test_round_ties_even() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::round_ties_even));
            let actual = Simd::from_array(value).round_ties_even();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value).round_int();
            let actual = Simd::from_array(value).fast_round_int();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_round_int() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            // TODO:  Currently `round` actually behaves like `round_ties_even`.
            // Decide the correct behavior then add documentation.
            let expected_1 = SimdSigned::from_array(
                value.map(|x| x.round().clamp(Signed::MIN as T, Signed::MAX as T) as Signed),
            );
            let expected_2 = SimdSigned::from_array(value.map(|x| {
                x.round_ties_even()
                    .clamp(Signed::MIN as T, Signed::MAX as T) as Signed
            }));
            let actual = Simd::from_array(value).round_int();

            assert!(
                (actual.simd_eq(expected_1) | actual.simd_eq(expected_2)).all(),
                "\nexpected_1: {expected_1:?}\nexpected_2: {expected_2:?}\n  actual: {actual:?}\n   value: {value:?}",
            );
        }
    });
}

#[test]
fn test_trunc() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::trunc));
            let actual = Simd::from_array(value).trunc();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = SimdSigned::from_array(value.map(|x| x.trunc() as Signed));
            let actual = Simd::from_array(value).fast_trunc_int();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_trunc_int() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = SimdSigned::from_array(
                value.map(|x| x.trunc().clamp(Signed::MIN as T, Signed::MAX as T) as Signed),
            );
            let actual = Simd::from_array(value).trunc_int();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::fract));
            let actual = Simd::from_array(value).fract();

            assert!(
                actual ^ expected == Simd::ZERO,
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_fast_mul_add() {
    for_simd_types!(|T: Float, N| {
        for [value, a, b] in fuzz_iter::<[[T; N]; 3]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].mul_add(a[i], b[i])));
            let actual =
                Simd::from_array(value).fast_mul_add(Simd::from_array(a), Simd::from_array(b));

            assert!(
                (actual - expected)
                    .abs()
                    .simd_le(expected.abs() * 1e-6)
                    .all(),
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
fn test_fast_mul_neg_add() {
    for_simd_types!(|T: Float, N| {
        for [value, a, b] in fuzz_iter::<[[T; N]; 3]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| b[i] - value[i] * a[i]));
            let actual =
                Simd::from_array(value).fast_mul_neg_add(Simd::from_array(a), Simd::from_array(b));

            assert!(
                (actual - expected)
                    .abs()
                    .simd_le(expected.abs() * 1e-6)
                    .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
            );
        }
    });
}

#[test]
fn test_fast_mul_sub() {
    for_simd_types!(|T: Float, N| {
        for [value, a, b] in fuzz_iter::<[[T; N]; 3]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i] * a[i] - b[i]));
            let actual =
                Simd::from_array(value).fast_mul_sub(Simd::from_array(a), Simd::from_array(b));

            assert!(
                (actual - expected)
                    .abs()
                    .simd_le(expected.abs() * 1e-6)
                    .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
            );
        }
    });
}

#[test]
fn test_fast_mul_neg_sub() {
    for_simd_types!(|T: Float, N| {
        for [value, a, b] in fuzz_iter::<[[T; N]; 3]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| -value[i] * a[i] - b[i]));
            let actual =
                Simd::from_array(value).fast_mul_neg_sub(Simd::from_array(a), Simd::from_array(b));

            assert!(
                (actual - expected)
                    .abs()
                    .simd_le(expected.abs() * 1e-6)
                    .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n   value: {value:?}\n       a: {a:?}\n       b: {b:?}",
            );
        }
    });
}

#[test]
fn test_div_euclid() {
    for_simd_types!(|T: Float, N| {
        for [left, right] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| left[i].div_euclid(right[i])));
            let actual = Simd::from_array(left).div_euclid(Simd::from_array(right));

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_rem_euclid() {
    for_simd_types!(|T: Float, N| {
        for [left, right] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| left[i].rem_euclid(right[i])));
            let actual = Simd::from_array(left).rem_euclid(Simd::from_array(right));

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_flipsign() {
    for_simd_types!(|T: Float, N| {
        for [value, sign] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| {
                if sign[i].is_sign_negative() {
                    -value[i]
                } else {
                    value[i]
                }
            }));
            let actual = Simd::from_array(value).flipsign(Simd::from_array(sign));

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_copysign() {
    for_simd_types!(|T: Float, N| {
        for [value, sign] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| {
                if sign[i].is_sign_negative() {
                    -value[i].abs()
                } else {
                    value[i].abs()
                }
            }));
            let actual = Simd::from_array(value).copysign(Simd::from_array(sign));

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_asin_acos() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = (
                Simd::from_array(value.map(T::asin)),
                Simd::from_array(value.map(T::acos)),
            );
            let actual = Simd::from_array(value).asin_acos();

            assert!(
                (actual.0 - expected.0).abs().simd_le(6e-7).all()
                    && (actual.1 - expected.1).abs().simd_le(6e-7).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_asin() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::asin));
            let actual = Simd::from_array(value).asin();

            assert!(
                (actual - expected).abs().simd_le(6e-7).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_acos() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::acos));
            let actual = Simd::from_array(value).acos();

            assert!(
                (actual - expected).abs().simd_le(6e-7).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_atan() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::atan));
            let actual = Simd::from_array(value).atan();

            let tol = if size_of::<T>() == 8 { 1e-15 } else { 6e-7 };
            assert!(
                (actual - expected).abs().simd_le(tol).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_atan2() {
    for_simd_types!(|T: Float, N| {
        for [value, other] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| value[i].atan2(other[i])));
            let actual = Simd::from_array(value).atan2(Simd::from_array(other));

            let tol = if size_of::<T>() == 8 { 1e-15 } else { 6e-7 };
            assert!(
                (actual - expected).abs().simd_le(tol).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_sin_cos() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = (
                Simd::from_array(value.map(T::sin)),
                Simd::from_array(value.map(T::cos)),
            );
            let actual = Simd::from_array(value).sin_cos();

            let tol = if size_of::<T>() == 8 { 6e-8 } else { 2e-7 };
            assert!(
                (actual.0 - expected.0).abs().simd_le(tol).all()
                    && (actual.1 - expected.1).abs().simd_le(tol).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_sin() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::sin));
            let actual = Simd::from_array(value).sin();

            let tol = if size_of::<T>() == 8 { 6e-8 } else { 2e-7 };
            assert!(
                (actual - expected).abs().simd_le(tol).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_cos() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::cos));
            let actual = Simd::from_array(value).cos();

            let tol = if size_of::<T>() == 8 { 6e-8 } else { 2e-7 };
            assert!(
                (actual - expected).abs().simd_le(tol).all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_tan() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::tan));
            let actual = Simd::from_array(value).tan();

            assert!(
                (actual - expected)
                    .abs()
                    .simd_le(expected.abs() * 1e-6)
                    .all(),
                "expected: {expected:?}\n  actual: {actual:?}",
            );
        }
    });
}

#[test]
fn test_to_degrees() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::to_degrees));
            let actual = Simd::from_array(value).to_degrees();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::to_radians));
            let actual = Simd::from_array(value).to_radians();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::recip));
            let actual = Simd::from_array(value).recip();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(|x| x.sqrt().recip()));
            let actual = Simd::from_array(value).recip_sqrt();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::sqrt));
            let actual = Simd::from_array(value).sqrt();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::exp));
            let actual = Simd::from_array(value).exp();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::exp2));
            let actual = Simd::from_array(value).exp2();

            let tol = if size_of::<T>() == 8 { 1e-12 } else { 1e-7 };
            let min_tol = if size_of::<T>() == 8 { 5e-324 } else { 1e-45 };
            assert!(
                ((actual - expected)
                    .abs()
                    .simd_le((expected.abs() * tol).max(Simd::splat(min_tol)))
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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::ln));
            let actual = Simd::from_array(value).ln();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::log2));
            let actual = Simd::from_array(value).log2();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::log10));
            let actual = Simd::from_array(value).log10();

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
fn test_powf_simd() {
    for_simd_types!(|T: Float, N| {
        for [x, n] in fuzz_iter::<[[T; N]; 2]>() {
            let expected = Simd::from_array(std::array::from_fn(|i| x[i].powf(n[i])));
            let actual = Simd::from_array(x).powf_simd(Simd::from_array(n));

            assert!(
                ((actual - expected).abs().simd_le(expected.abs() * 1e-6)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n       x: {x:?}\n       n: {n:?}",
            );
        }
    });
}

#[test]
fn test_is_sign_positive() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(|x| {
                if x.is_sign_positive() {
                    T::from_bits(!0)
                } else {
                    0.0
                }
            }));
            let actual = Simd::from_array(value).is_sign_positive();

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
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(|x| {
                if x.is_sign_negative() {
                    T::from_bits(!0)
                } else {
                    0.0
                }
            }));
            let actual = Simd::from_array(value).is_sign_negative();

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

    let signed = i32x4::from_array([1, 2, 3, 4]);
    let expected = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    let actual = f32x4::from_i32x4(signed);
    assert_eq!(actual, expected);

    let signed = i32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
    let expected = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let actual = f32x8::from_i32x8(signed);
    assert_eq!(actual, expected);

    let signed = i32x16::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let expected = f32x16::from_array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
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

    let value = i32x4::from_array([1, 2, 3, 4]);
    let expected = f64x2::from_array([1.0, 2.0]);
    let actual = f64x2::from_i32x4_lower2(value);
    assert_eq!(actual, expected);
}

#[test]
fn test_exp_m1() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::exp_m1));
            let actual = Simd::from_array(value).exp_m1();
            let tol = expected.abs().fast_max(Simd::splat(1 as T)) * Simd::splat(T::EPSILON * 2.0);

            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n value: {value:?}",
            );
        }

        let mut x = 0xdead_beef_u64;
        let tol = if std::mem::size_of::<T>() == 4 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        let max_input = if std::mem::size_of::<T>() == 4 {
            88.37
        } else {
            709.4
        };

        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = if std::mem::size_of::<T>() == 4 {
                f32::from_bits(x as u32) as f64
            } else {
                f64::from_bits(x)
            };

            let v = if u.is_finite() && u.abs() <= max_input {
                u
            } else {
                0.0
            };
            let t_v = v as T;
            let expected = Simd::from(T::exp_m1(t_v));
            let actual = Simd::from(t_v).exp_m1();

            let diff = (actual - expected).abs();
            let max_err = expected.abs().fast_max(Simd::from(1.0 as T)) * Simd::from(tol as T);

            assert!(
                diff.simd_le(max_err).all(),
                "exp_m1({v:e}) actual: {actual:?} expected: {expected:?}"
            );
        }

        // Verify fast-exit for deeply negative values: exp_m1(x) = -1.0 exactly
        let deep: T = if std::mem::size_of::<T>() == 4 {
            (-200.0) as T
        } else {
            (-800.0) as T
        };
        let input = Simd::splat(deep);
        let result = input.exp_m1().to_array();
        for &v in result.iter() {
            assert!(
                (v - (-1.0 as T)).abs() <= T::EPSILON * 1.0,
                "exp_m1({deep:e}) != -1.0, got {v:e}"
            );
        }
    });
}

#[test]
fn test_ln_1p() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::ln_1p));
            let actual = Simd::from_array(value).ln_1p();
            let tol = expected.abs().fast_max(Simd::splat(1 as T)) * Simd::splat(T::EPSILON * 1.0);

            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n value: {value:?}",
            );
        }

        let mut x = 0xcafe_babe_u64;
        let tol = if std::mem::size_of::<T>() == 4 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        let max_input = if std::mem::size_of::<T>() == 4 {
            f32::MAX as f64
        } else {
            f64::MAX
        };

        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = if std::mem::size_of::<T>() == 4 {
                f32::from_bits(x as u32) as f64
            } else {
                f64::from_bits(x)
            };

            let v = if u.is_finite() && u > -0.999 && u < max_input {
                u
            } else {
                0.0
            };
            let t_v = v as T;
            let expected = Simd::from(T::ln_1p(t_v));
            let actual = Simd::from(t_v).ln_1p();

            let diff = (actual - expected).abs();
            let max_err = expected.abs().fast_max(Simd::from(1.0 as T)) * Simd::from(tol as T);

            assert!(
                diff.simd_le(max_err).all(),
                "ln_1p({v:e}) actual: {actual:?} expected: {expected:?}"
            );
        }

        for &v in &[1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 0.01, 0.1, 1.0] {
            let t_v = v as T;
            let expected = Simd::from(T::ln_1p(t_v));
            let actual = Simd::from(t_v).ln_1p();
            let diff = (actual - expected).abs();
            let limit = if std::mem::size_of::<T>() == 4 {
                1e-6 as T
            } else {
                1e-12 as T
            };
            assert!(
                diff.simd_le(Simd::from(limit)).all(),
                "ln_1p({v:e}) off by {diff:?}"
            );
        }

        // Edge cases: ln_1p(-1) → -inf, ln_1p(x < -1) → NaN, ln_1p(-0.0) → -0.0
        let neg_one = Simd::splat(-1.0 as T);
        let result = neg_one.ln_1p().to_array();
        for &v in result.iter() {
            assert!(v == T::NEG_INFINITY, "ln_1p(-1) should be -inf, got {v:e}");
        }

        let neg_two = Simd::splat(-2.0 as T);
        let result = neg_two.ln_1p().to_array();
        for &v in result.iter() {
            assert!(v.is_nan(), "ln_1p(-2) should be NaN, got {v:e}");
        }

        let neg_zero = Simd::splat(-T::from_bits(0));
        let result = neg_zero.ln_1p().to_array();
        for &v in result.iter() {
            assert!(
                v.is_sign_negative() && v == T::from_bits(0),
                "ln_1p(-0.0) should be -0.0, got {v:e}"
            );
        }
    });
}

#[test]
fn test_sinh() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::sinh));
            let actual = Simd::from_array(value).sinh();
            let tol = expected.abs().fast_max(Simd::splat(1 as T)) * Simd::splat(T::EPSILON * 1.0);

            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n value: {value:?}",
            );
        }

        let mut x = 0xfeed_face_u64;
        let tol = if std::mem::size_of::<T>() == 4 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        let max_input = if std::mem::size_of::<T>() == 4 {
            88.37
        } else {
            710.0
        };

        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = if std::mem::size_of::<T>() == 4 {
                f32::from_bits(x as u32) as f64
            } else {
                f64::from_bits(x)
            };

            let v = if u.is_finite() && u.abs() <= max_input {
                u
            } else {
                0.0
            };
            let t_v = v as T;
            let expected = Simd::from(T::sinh(t_v));
            let actual = Simd::from(t_v).sinh();

            let diff = (actual - expected).abs();
            let max_err = expected.abs().fast_max(Simd::from(1.0 as T)) * Simd::from(tol as T);

            assert!(
                diff.simd_le(max_err).all(),
                "sinh({v:e}) actual: {actual:?} expected: {expected:?}"
            );
        }
    });
}

#[test]
fn test_cosh() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::cosh));
            let actual = Simd::from_array(value).cosh();
            let tol = expected.abs().fast_max(Simd::splat(1 as T)) * Simd::splat(T::EPSILON * 1.0);

            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n value: {value:?}",
            );
        }

        let mut x = 0xbad_cafe_u64;
        let tol = if std::mem::size_of::<T>() == 4 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        let max_input = if std::mem::size_of::<T>() == 4 {
            88.37
        } else {
            710.0
        };

        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = if std::mem::size_of::<T>() == 4 {
                f32::from_bits(x as u32) as f64
            } else {
                f64::from_bits(x)
            };

            let v = if u.is_finite() && u.abs() <= max_input {
                u
            } else {
                0.0
            };
            let t_v = v as T;
            let expected = Simd::from(T::cosh(t_v));
            let actual = Simd::from(t_v).cosh();

            let diff = (actual - expected).abs();
            let max_err = expected.abs().fast_max(Simd::from(1.0 as T)) * Simd::from(tol as T);

            assert!(
                diff.simd_le(max_err).all(),
                "cosh({v:e}) actual: {actual:?} expected: {expected:?}"
            );
        }
    });
}

#[test]
fn test_tanh() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::tanh));
            let actual = Simd::from_array(value).tanh();
            let tol = expected.abs().fast_max(Simd::splat(1 as T)) * Simd::splat(T::EPSILON * 1.0);

            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n value: {value:?}",
            );
        }

        let mut x = 0x1337_c0de_u64;
        let tol = if std::mem::size_of::<T>() == 4 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };

        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = if std::mem::size_of::<T>() == 4 {
                f32::from_bits(x as u32) as f64
            } else {
                f64::from_bits(x)
            };

            let v = if u.is_finite() { u } else { 0.0 };
            let t_v = v as T;
            let expected = Simd::from(T::tanh(t_v));
            let actual = Simd::from(t_v).tanh();

            let diff = (actual - expected).abs();
            let max_err = expected.abs().fast_max(Simd::from(1.0 as T)) * Simd::from(tol as T);

            assert!(
                diff.simd_le(max_err).all(),
                "tanh({v:e}) actual: {actual:?} expected: {expected:?}"
            );
        }
    });
}

#[test]
fn test_cbrt() {
    for_simd_types!(|T: Float, N| {
        for value in fuzz_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::cbrt));
            let actual = Simd::from_array(value).cbrt();
            // SLEEF polynomial + 1 Newton step on 1/cbrt + 1 polish on cbrt.
            // Accumulated roundoff in the scaling (r = e - 3k) and refinement
            // steps keeps this at 2 ULP worst-case rather than 1 ULP.
            let tol = expected.abs().fast_max(Simd::splat(1 as T)) * Simd::splat(T::EPSILON * 2.0);

            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "expected: {expected:?}\n  actual: {actual:?}\n value: {value:?}",
            );
        }

        let mut x = 0xcafe_babe_u64;
        let tol = if std::mem::size_of::<T>() == 4 {
            f32::EPSILON as f64 * 2.0
        } else {
            f64::EPSILON * 2.0
        };

        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = if std::mem::size_of::<T>() == 4 {
                f32::from_bits(x as u32) as f64
            } else {
                f64::from_bits(x)
            };

            let v = if u.is_finite() { u } else { 0.0 };
            let t_v = v as T;
            let expected = Simd::from(T::cbrt(t_v));
            let actual = Simd::from(t_v).cbrt();

            let diff = (actual - expected).abs();
            let max_err = expected.abs().fast_max(Simd::from(1.0 as T)) * Simd::from(tol as T);

            assert!(
                diff.simd_le(max_err).all(),
                "cbrt({v:e}) actual: {actual:?} expected: {expected:?}"
            );
        }
    });
}

#[test]
fn test_exp_overflow_boundary() {
    for_simd_types!(|T: Float, N| {
        let max_x: T = if std::mem::size_of::<T>() == 4 {
            88.723 as T
        } else {
            709.783 as T
        };
        let near_max = max_x * (0.999 as T);
        let over_max = max_x * (1.001 as T);

        for v in [max_x, near_max, over_max, -max_x] {
            let input = Simd::splat(v);
            let expected = Simd::splat(T::exp(v));
            let actual = input.exp();
            let tol = expected.abs().fast_max(Simd::splat(T::MIN_POSITIVE))
                * Simd::splat(T::EPSILON * 1.0);
            assert!(
                ((actual - expected).abs().simd_le(tol)
                    | actual.simd_eq(expected)
                    | actual.is_nan() & expected.is_nan())
                .all(),
                "exp({v:e}) expected: {expected:?} actual: {actual:?}"
            );
        }
    });
}

#[test]
fn test_exp_subnormal() {
    for_simd_types!(|T: Float, N| {
        let subnormal_input: T = if std::mem::size_of::<T>() == 4 {
            (-88.0) as T
        } else {
            (-710.0) as T
        };
        let deeper: T = if std::mem::size_of::<T>() == 4 {
            (-95.0) as T
        } else {
            (-730.0) as T
        };
        for v in [subnormal_input, deeper] {
            let input = Simd::splat(v);
            let expected = Simd::splat(T::exp(v));
            let actual = input.exp();
            let tol = expected.abs().fast_max(Simd::splat(T::MIN_POSITIVE))
                * Simd::splat(T::EPSILON * 1.0);
            assert!(
                (actual - expected).abs().simd_le(tol).all(),
                "exp({v:e}) = {actual:?} expected {expected:?}"
            );
        }
    });
}

#[test]
fn test_math_nan_propagation() {
    for_simd_types!(|T: Float, N| {
        let nan = T::NAN;
        let mut arr = [0.0 as T; N];
        arr[0] = nan;
        if N > 1 {
            arr[1] = 1.0 as T;
        }
        let input = Simd::from(arr);

        let result = input.exp().to_array();
        assert!(result[0].is_nan(), "exp: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "exp: lane 1 finite");
        }

        let result = input.exp_m1().to_array();
        assert!(result[0].is_nan(), "exp_m1: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "exp_m1: lane 1 finite");
        }

        let result = input.ln().to_array();
        assert!(result[0].is_nan(), "ln: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "ln: lane 1 finite");
        }

        let result = input.ln_1p().to_array();
        assert!(result[0].is_nan(), "ln_1p: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "ln_1p: lane 1 finite");
        }

        let result = input.sinh().to_array();
        assert!(result[0].is_nan(), "sinh: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "sinh: lane 1 finite");
        }

        let result = input.cosh().to_array();
        assert!(result[0].is_nan(), "cosh: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "cosh: lane 1 finite");
        }

        let result = input.tanh().to_array();
        assert!(result[0].is_nan(), "tanh: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "tanh: lane 1 finite");
        }

        let result = input.cbrt().to_array();
        assert!(result[0].is_nan(), "cbrt: lane 0 NaN");
        if N > 1 {
            assert!(!result[1].is_nan(), "cbrt: lane 1 finite");
        }
    });
}

#[test]
fn test_sign_preservation() {
    for_simd_types!(|T: Float, N| {
        let neg_zero = -T::from_bits(0);
        let input = Simd::splat(neg_zero);

        let result = input.exp_m1().to_array();
        for &v in result.iter() {
            assert!(
                v.is_sign_negative(),
                "exp_m1(-0.0) should be -0.0, got {v:e}"
            );
        }

        let result = input.tanh().to_array();
        for &v in result.iter() {
            assert!(v.is_sign_negative(), "tanh(-0.0) should be -0.0, got {v:e}");
        }

        let result = input.cbrt().to_array();
        for &v in result.iter() {
            assert!(v.is_sign_negative(), "cbrt(-0.0) should be -0.0, got {v:e}");
        }

        let result = input.ln_1p().to_array();
        for &v in result.iter() {
            assert!(
                v.is_sign_negative(),
                "ln_1p(-0.0) should be -0.0, got {v:e}"
            );
        }
    });
}
