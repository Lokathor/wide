use crate::utils::{for_simd_types, random_iter};

#[test]
fn test_add_signed() {
    for_simd_types!(|T: Unsigned, N| {
        for (a, b) in random_iter::<([T; N], [Signed; N])>() {
            let expected =
                Simd::from_array(std::array::from_fn(|i| a[i].wrapping_add_signed(b[i])));
            let actual = Simd::from_array(a).add_signed(SimdSigned::from_array(b));

            assert_eq!(actual, expected, "a: {a:?}\nb: {b:?}");
        }
    });
}

#[test]
fn test_sub_signed() {
    for_simd_types!(|T: Unsigned, N| {
        for (a, b) in random_iter::<([T; N], [Signed; N])>() {
            let expected =
                Simd::from_array(std::array::from_fn(|i| a[i].wrapping_sub_signed(b[i])));
            let actual = Simd::from_array(a).sub_signed(SimdSigned::from_array(b));

            assert_eq!(actual, expected, "a: {a:?}\nb: {b:?}");
        }
    });
}

#[test]
fn test_overflowing_add_signed() {
    for_simd_types!(|T: Unsigned, N| {
        for (a, b) in random_iter::<([T; N], [Signed; N])>() {
            let expected = (
                Simd::from_array(std::array::from_fn(|i| a[i].overflowing_add_signed(b[i]).0)),
                Simd::from_array(std::array::from_fn(|i| {
                    if a[i].overflowing_add_signed(b[i]).1 {
                        !0
                    } else {
                        0
                    }
                })),
            );
            let actual = Simd::from_array(a).overflowing_add_signed(SimdSigned::from_array(b));

            assert_eq!(actual, expected, "\n{a:?}\n{b:?}");
        }
    });
}

#[test]
fn test_overflowing_sub_signed() {
    for_simd_types!(|T: Unsigned, N| {
        for (a, b) in random_iter::<([T; N], [Signed; N])>() {
            let expected = (
                Simd::from_array(std::array::from_fn(|i| a[i].overflowing_sub_signed(b[i]).0)),
                Simd::from_array(std::array::from_fn(|i| {
                    if a[i].overflowing_sub_signed(b[i]).1 {
                        !0
                    } else {
                        0
                    }
                })),
            );
            let actual = Simd::from_array(a).overflowing_sub_signed(SimdSigned::from_array(b));

            assert_eq!(actual, expected, "\n{a:?}\n{b:?}");
        }
    });
}
