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
