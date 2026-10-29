use wide::{i16x8, i16x16, i16x32, i32x4, i32x8, i32x16};

use crate::utils::{for_simd_types, random_iter, simd_chunks};

#[test]
fn test_is_positive() {
    for_simd_types!(|T: Signed, N| {
        for value in simd_chunks!([1, -1, 2, 3, -2, -5, 0, 6, 9, -4, 0, T::MIN, T::MAX]) {
            let expected = Simd::from_array(value.map(|x| if x.is_positive() { !0 } else { 0 }));
            let actual = Simd::from_array(value).is_positive();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_is_negative() {
    for_simd_types!(|T: Signed, N| {
        for value in simd_chunks!([1, -1, 2, 3, -2, -5, 0, 6, 9, -4, 0, T::MIN, T::MAX]) {
            let expected = Simd::from_array(value.map(|x| if x.is_negative() { !0 } else { 0 }));
            let actual = Simd::from_array(value).is_negative();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_abs() {
    for_simd_types!(|T: Signed, N| {
        for value in
            simd_chunks!([1, -15, -2, -5, 0, 9, -4, 0, T::MIN, T::MAX]).chain(random_iter())
        {
            let expected = Simd::from_array(value.map(T::wrapping_abs));
            let actual = Simd::from_array(value).abs();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_unsigned_abs() {
    for_simd_types!(|T: Signed, N| {
        for value in
            simd_chunks!([1, -15, -2, -5, 0, 9, -4, 0, T::MIN, T::MAX]).chain(random_iter())
        {
            let expected = SimdUnsigned::from_array(value.map(T::unsigned_abs));
            let actual = Simd::from_array(value).unsigned_abs();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_signum() {
    for_simd_types!(|T: Signed, N| {
        for value in
            simd_chunks!([1, -15, -2, -5, 0, 9, -4, 0, T::MIN, T::MAX]).chain(random_iter())
        {
            let expected = Simd::from_array(value.map(T::signum));
            let actual = Simd::from_array(value).signum();

            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn test_add_unsigned() {
    for_simd_types!(|T: Signed, N| {
        for (a, b) in random_iter::<([T; N], [Unsigned; N])>() {
            let expected =
                Simd::from_array(std::array::from_fn(|i| a[i].wrapping_add_unsigned(b[i])));
            let actual = Simd::from_array(a).add_unsigned(SimdUnsigned::from_array(b));

            assert_eq!(actual, expected, "a: {a:?}\nb: {b:?}");
        }
    });
}

#[test]
fn test_sub_unsigned() {
    for_simd_types!(|T: Signed, N| {
        for (a, b) in random_iter::<([T; N], [Unsigned; N])>() {
            let expected =
                Simd::from_array(std::array::from_fn(|i| a[i].wrapping_sub_unsigned(b[i])));
            let actual = Simd::from_array(a).sub_unsigned(SimdUnsigned::from_array(b));

            assert_eq!(actual, expected, "a: {a:?}\nb: {b:?}");
        }
    });
}

#[test]
fn test_saturating_neg() {
    for_simd_types!(|T: Signed, N| {
        for value in random_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::saturating_neg));
            let actual = Simd::from_array(value).saturating_neg();

            assert_eq!(actual, expected, "value: {value:?}");
        }
    });
}

#[test]
fn test_saturating_abs() {
    for_simd_types!(|T: Signed, N| {
        for value in random_iter::<[T; N]>() {
            let expected = Simd::from_array(value.map(T::saturating_abs));
            let actual = Simd::from_array(value).saturating_abs();

            assert_eq!(actual, expected, "value: {value:?}");
        }
    });
}

#[test]
fn test_saturating_add_unsigned() {
    for_simd_types!(|T: Signed, N| {
        for (a, b) in random_iter::<([T; N], [Unsigned; N])>() {
            let expected =
                Simd::from_array(std::array::from_fn(|i| a[i].saturating_add_unsigned(b[i])));
            let actual = Simd::from_array(a).saturating_add_unsigned(SimdUnsigned::from_array(b));

            assert_eq!(actual, expected, "a: {a:?}\nb: {b:?}");
        }
    });
}

#[test]
fn test_saturating_sub_unsigned() {
    for_simd_types!(|T: Signed, N| {
        for (a, b) in random_iter::<([T; N], [Unsigned; N])>() {
            let expected =
                Simd::from_array(std::array::from_fn(|i| a[i].saturating_sub_unsigned(b[i])));
            let actual = Simd::from_array(a).saturating_sub_unsigned(SimdUnsigned::from_array(b));

            assert_eq!(actual, expected, "a: {a:?}\nb: {b:?}");
        }
    });
}

#[test]
fn test_overflowing_abs() {
    for_simd_types!(|T: Signed, N| {
        for value in random_iter::<[T; N]>() {
            let expected = (
                Simd::from_array(std::array::from_fn(|i| value[i].overflowing_abs().0)),
                Simd::from_array(std::array::from_fn(|i| {
                    if value[i].overflowing_abs().1 { !0 } else { 0 }
                })),
            );
            let actual = Simd::from_array(value).overflowing_abs();

            assert_eq!(actual, expected, "value: {value:?}");
        }
    });
}

#[test]
fn test_overflowing_add_unsigned() {
    for_simd_types!(|T: Signed, N| {
        for (a, b) in random_iter::<([T; N], [Unsigned; N])>() {
            let expected = (
                Simd::from_array(std::array::from_fn(|i| {
                    a[i].overflowing_add_unsigned(b[i]).0
                })),
                Simd::from_array(std::array::from_fn(|i| {
                    if a[i].overflowing_add_unsigned(b[i]).1 {
                        !0
                    } else {
                        0
                    }
                })),
            );
            let actual = Simd::from_array(a).overflowing_add_unsigned(SimdUnsigned::from_array(b));

            assert_eq!(actual, expected, "\n{a:?}\n{b:?}");
        }
    });
}

#[test]
fn test_overflowing_sub_unsigned() {
    for_simd_types!(|T: Signed, N| {
        for (a, b) in random_iter::<([T; N], [Unsigned; N])>() {
            let expected = (
                Simd::from_array(std::array::from_fn(|i| {
                    a[i].overflowing_sub_unsigned(b[i]).0
                })),
                Simd::from_array(std::array::from_fn(|i| {
                    if a[i].overflowing_sub_unsigned(b[i]).1 {
                        !0
                    } else {
                        0
                    }
                })),
            );
            let actual = Simd::from_array(a).overflowing_sub_unsigned(SimdUnsigned::from_array(b));

            assert_eq!(actual, expected, "\n{a:?}\n{b:?}");
        }
    });
}

#[test]
fn test_mul_scale_round() {
    // `mul_scale_round` is inconsistently missing from types.

    let a = i16x8::from_array([100, 200, 300, 400, 500, -600, 700, -800]);
    let b = i16x8::from_array([900, 1000, 1100, 1200, 1300, -1400, -1500, 1600]);
    let actual = a.mul_scale_round(b);
    let expected = i16x8::from_array([3, 6, 10, 15, 20, 26, -32, -39]);
    assert_eq!(actual, expected);

    let a = i16x16::from([
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    ]);
    let b = i16x16::from([
        0, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    ]);
    let expected = i16x16::from([0, 3, 6, 10, 15, 20, 26, 32, 39, 47, 55, 64, 73, 83, 94, 105]);
    let actual = a.mul_scale_round(b);
    assert_eq!(actual, expected);
}

#[test]
fn test_mul_scale_round_n() {
    // `mul_scale_round_n` is inconsistently missing from types.

    let a = i16x8::from_array([100, 200, 300, 400, 500, -600, 700, -800]);
    let actual = a.mul_scale_round_n(0x4000);
    let expected = i16x8::from_array([50, 100, 150, 200, 250, -300, 350, -400]);
    assert_eq!(actual, expected);

    let a = i16x16::from([
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    ]);
    let expected = i16x16::from([
        0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 551, 601, 651, 701, 751,
    ]);
    // slightly higher than 0.5 to test rounding
    let actual = a.mul_scale_round_n(16400);
    assert_eq!(expected, actual);
}

#[test]
fn test_dot() {
    // `dot` is inconsistently missing from types.

    let value = i16x8::from_array([1, 2, 3, 4, 5, 6, i16::MIN + 1, i16::MIN]);
    let other = i16x8::from_array([17, -18, 190, -20, 21, -22, 3, 2]);
    let expected = i32x4::from_array([-19, 490, -27, -163837]);
    let actual = value.dot(other);
    assert_eq!(actual, expected);

    let value = i16x16::from_array([
        1,
        2,
        3,
        4,
        5,
        6,
        i16::MIN + 1,
        i16::MIN,
        10,
        20,
        30,
        40,
        50,
        60,
        i16::MAX - 1,
        i16::MAX,
    ]);
    let other = i16x16::from_array([
        17, -18, 190, -20, 21, -22, 3, 2, 170, -180, 1900, -200, 210, -220, 30, 20,
    ]);
    let expected = i32x8::from_array([-19, 490, -27, -163837, -1900, 49000, -2700, 1638320]);
    let actual = value.dot(other);
    assert_eq!(actual, expected);

    let value = i16x32::from([
        1,
        2,
        3,
        4,
        5,
        6,
        i16::MIN + 1,
        i16::MIN,
        10,
        20,
        30,
        40,
        50,
        60,
        i16::MAX - 1,
        i16::MAX,
        1,
        2,
        3,
        4,
        5,
        6,
        i16::MIN + 1,
        i16::MIN,
        10,
        20,
        30,
        40,
        50,
        60,
        i16::MAX - 1,
        i16::MAX,
    ]);
    let other = i16x32::from([
        17, -18, 190, -20, 21, -22, 3, 2, 170, -180, 1900, -200, 210, -220, 30, 20, 17, -18, 190,
        -20, 21, -22, 3, 2, 170, -180, 1900, -200, 210, -220, 30, 20,
    ]);
    let expected = i32x16::from([
        -19, 490, -27, -163837, -1900, 49000, -2700, 1638320, -19, 490, -27, -163837, -1900, 49000,
        -2700, 1638320,
    ]);
    let actual = value.dot(other);
    assert_eq!(actual, expected);
}
