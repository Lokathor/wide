use wide::{
  f32x4, f32x8, f32x16, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i32x16,
  u8x16, u8x32, u16x8,
};

use crate::utils::{for_simd_types, random_iter, simd_chunks};

#[test]
fn test_saturating_add() {
  for_simd_types!(|T: Integer, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MAX - 1, T::MAX - 1, 15, 20, 100, T::MAX - 1, T::MAX / 2],
      [17, 18, 1, 2, 20, 5, T::MAX - 5, 50, 100],
    )
    .chain(random_iter())
    {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_add(right[i])));
      let actual = Simd::new(left).saturating_add(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MAX - 1, T::MIN + 1, T::MIN + 2, T::MIN / 2, 9],
      [-17, 18, T::MAX, -2, -20, -100, 10],
    )
    .chain(random_iter())
    {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_add(right[i])));
      let actual = Simd::new(left).saturating_add(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
}

#[test]
fn test_saturating_sub() {
  for_simd_types!(|T: Integer, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MIN + 1, T::MIN + 1, 15, 20, 100, T::MIN + 1, T::MIN / 2],
      [17, 18, 1, 2, 20, 5, T::MAX - 5, 50, 100],
    )
    .chain(random_iter())
    {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_sub(right[i])));
      let actual = Simd::new(left).saturating_sub(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MAX - 1, T::MIN + 1, T::MIN + 2, T::MAX / 2],
      [17, -18, T::MIN, 2, 20, -100],
    )
    .chain(random_iter())
    {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_sub(right[i])));
      let actual = Simd::new(left).saturating_sub(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
}

#[test]
fn test_saturating_mul() {
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        T::MIN + 1,
        T::MIN,
        2,
        3,
        4,
        5,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        T::MAX / 2,
        T::MIN / 2,
      ],
      [17, -18, 1, 1, -1, -2, -6, 3, 3, 2, 1, 3, 3],
    )
    .chain(random_iter())
    {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_mul(right[i])));
      let actual = Simd::new(left).saturating_mul(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Unsigned, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        3,
        4,
        5,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        3,
        4,
        3,
        2,
        2,
        1,
      ],
      [
        17,
        18,
        9,
        1,
        0,
        3,
        4,
        3,
        2,
        2,
        1,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
      ],
    )
    .chain(random_iter())
    {
      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_mul(right[i])));
      let actual = Simd::new(left).saturating_mul(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_saturating_div() {
  for_simd_types!(|T: Integer, N| {
    for [left, mut right] in simd_chunks!(
      [11, 15, 2, 3, T::MAX, 0, T::MAX, T::MAX - 1],
      [2, 5, 5, 8, 10, 5, 2, 10],
    )
    .chain(random_iter())
    {
      for right in &mut right {
        if *right == 0 {
          *right = 3;
        }
      }

      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_div(right[i])));
      let actual = Simd::new(left).saturating_div(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}",
      );
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, mut right] in simd_chunks!(
      [11, 15, -13, -16, T::MIN, 0, T::MIN, T::MIN + 1],
      [-2, -5, 3, -6, -2, -1, -1, -1],
    )
    .chain(random_iter())
    {
      for right in &mut right {
        if *right == 0 {
          *right = 3;
        }
      }

      let expected =
        Simd::new(std::array::from_fn(|i| left[i].saturating_div(right[i])));
      let actual = Simd::new(left).saturating_div(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}",
      );
    }
  });
}

#[test]
fn test_overflowing_add() {
  for_simd_types!(|T: Integer, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MAX - 1, T::MAX - 1, 15, 20, 100, T::MAX - 1, T::MAX / 2],
      [17, 18, 1, 2, 20, 5, T::MAX - 5, 50, 100],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_add(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_add(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_add(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MAX - 1, T::MIN + 1, T::MIN + 2, T::MIN / 2, 9],
      [-17, 18, T::MAX, -2, -20, -100, 10],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_add(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_add(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_add(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
}

#[test]
fn test_overflowing_sub() {
  for_simd_types!(|T: Integer, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MIN + 1, T::MIN + 1, 15, 20, 100, T::MIN + 1, T::MIN / 2],
      [17, 18, 1, 2, 20, 5, T::MAX - 5, 50, 100],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_sub(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_sub(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_sub(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [1, 2, T::MAX - 1, T::MIN + 1, T::MIN + 2, T::MAX / 2],
      [17, -18, T::MIN, 2, 20, -100],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_sub(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_sub(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_sub(Simd::new(right));

      assert_eq!(expected, actual);
    }
  });
}

#[test]
fn test_overflowing_mul() {
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        T::MIN + 1,
        T::MIN,
        2,
        3,
        4,
        5,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        T::MAX / 2,
        T::MIN / 2,
      ],
      [17, -18, 1, 1, -1, -2, -6, 3, 3, 2, 1, 3, 3],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_mul(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_mul(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_mul(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Unsigned, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        3,
        4,
        5,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        3,
        4,
        3,
        2,
        2,
        1,
      ],
      [
        17,
        18,
        9,
        1,
        0,
        3,
        4,
        3,
        2,
        2,
        1,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
      ],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_mul(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_mul(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_mul(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_overflowing_div() {
  for_simd_types!(|T: Integer, N| {
    for [left, mut right] in simd_chunks!(
      [11, 15, 2, 3, T::MAX, 0, T::MAX, T::MAX - 1],
      [2, 5, 5, 8, 10, 5, 2, 10],
    )
    .chain(random_iter())
    {
      for right in &mut right {
        if *right == 0 {
          *right = 3;
        }
      }

      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_div(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_div(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_div(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, mut right] in simd_chunks!(
      [11, 15, -13, -16, T::MIN, 0, T::MIN, T::MIN + 1],
      [-2, -5, 3, -6, -2, -1, -1, -1],
    )
    .chain(random_iter())
    {
      for right in &mut right {
        if *right == 0 {
          *right = 3;
        }
      }

      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_div(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_div(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_div(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_overflowing_rem() {
  for_simd_types!(|T: Integer, N| {
    for [left, mut right] in simd_chunks!(
      [11, 15, 2, 3, T::MAX, 0, T::MAX, T::MAX - 1],
      [2, 5, 5, 8, 10, 5, 2, 10],
    )
    .chain(random_iter())
    {
      for right in &mut right {
        if *right == 0 {
          *right = 3;
        }
      }

      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_rem(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_rem(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_rem(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Signed, N| {
    for [left, mut right] in simd_chunks!(
      [11, 15, -13, -16, T::MIN, 0, T::MIN, T::MIN + 1],
      [-2, -5, 3, -6, -2, -1, -1, -1],
    )
    .chain(random_iter())
    {
      for right in &mut right {
        if *right == 0 {
          *right = 3;
        }
      }

      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].overflowing_rem(right[i]).0)),
        Simd::new(std::array::from_fn(|i| {
          if left[i].overflowing_rem(right[i]).1 { !0 } else { 0 }
        })),
      );
      let actual = Simd::new(left).overflowing_rem(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_widening_mul() {
  for_simd_types!(|T: Signed, N, DoubleSizedSimd| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        T::MIN + 1,
        T::MIN,
        2,
        3,
        4,
        5,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        T::MAX / 2,
        T::MIN / 2,
      ],
      [17, -18, 1, 1, -1, -2, -6, 3, 3, 2, 1, 3, 3],
    )
    .chain(random_iter())
    {
      let expected = DoubleSizedSimd::new(std::array::from_fn(|i| {
        (left[i] as DoubleSizedT) * (right[i] as DoubleSizedT)
      }));
      let actual = Simd::new(left).widening_mul(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Unsigned, N, DoubleSizedSimd| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        3,
        4,
        5,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        3,
        4,
        3,
        2,
        2,
        1,
      ],
      [
        17,
        18,
        9,
        1,
        0,
        3,
        4,
        3,
        2,
        2,
        1,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
      ],
    )
    .chain(random_iter())
    {
      let expected = DoubleSizedSimd::new(std::array::from_fn(|i| {
        (left[i] as DoubleSizedT) * (right[i] as DoubleSizedT)
      }));
      let actual = Simd::new(left).widening_mul(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_mul_keep_low_high() {
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        T::MIN + 1,
        T::MIN,
        2,
        3,
        4,
        5,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        T::MAX / 2,
        T::MIN / 2,
      ],
      [17, -18, 1, 1, -1, -2, -6, 3, 3, 2, 1, 3, 3],
    )
    .chain(random_iter())
    {
      let expected = (
        SimdUnsigned::new(std::array::from_fn(|i| {
          left[i].wrapping_mul(right[i]).cast_unsigned()
        })),
        Simd::new(std::array::from_fn(|i| {
          ((left[i] as DoubleSizedT).wrapping_mul(right[i] as DoubleSizedT)
            >> T::BITS) as T
        })),
      );
      let actual = Simd::new(left).mul_keep_low_high(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Unsigned, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        3,
        4,
        5,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        3,
        4,
        3,
        2,
        2,
        1,
      ],
      [
        17,
        18,
        9,
        1,
        0,
        3,
        4,
        3,
        2,
        2,
        1,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
      ],
    )
    .chain(random_iter())
    {
      let expected = (
        Simd::new(std::array::from_fn(|i| left[i].wrapping_mul(right[i]))),
        Simd::new(std::array::from_fn(|i| {
          ((left[i] as DoubleSizedT).wrapping_mul(right[i] as DoubleSizedT)
            >> T::BITS) as T
        })),
      );
      let actual = Simd::new(left).mul_keep_low_high(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_mul_keep_high() {
  for_simd_types!(|T: Signed, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        T::MIN + 1,
        T::MIN,
        2,
        3,
        4,
        5,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        T::MAX / 2,
        T::MIN / 2,
      ],
      [17, -18, 1, 1, -1, -2, -6, 3, 3, 2, 1, 3, 3],
    )
    .chain(random_iter())
    {
      let expected = Simd::new(std::array::from_fn(|i| {
        ((left[i] as DoubleSizedT).wrapping_mul(right[i] as DoubleSizedT)
          >> T::BITS) as T
      }));
      let actual = Simd::new(left).mul_keep_high(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
  for_simd_types!(|T: Unsigned, N| {
    for [left, right] in simd_chunks!(
      [
        1,
        2,
        3,
        4,
        5,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
        3,
        4,
        3,
        2,
        2,
        1,
      ],
      [
        17,
        18,
        9,
        1,
        0,
        3,
        4,
        3,
        2,
        2,
        1,
        T::MAX / 4,
        T::MAX / 3,
        T::MAX / 2,
        T::MAX - 1,
        T::MAX,
        T::MAX,
      ],
    )
    .chain(random_iter())
    {
      let expected = Simd::new(std::array::from_fn(|i| {
        ((left[i] as DoubleSizedT).wrapping_mul(right[i] as DoubleSizedT)
          >> T::BITS) as T
      }));
      let actual = Simd::new(left).mul_keep_high(Simd::new(right));

      assert!(
        actual == expected,
        "expected: {expected:?}\n  actual: {actual:?}\n    left: {left:?}\n   right: {right:?}"
      );
    }
  });
}

#[test]
fn test_from_big_truncate() {
  // `from_{big}_truncate` is inconsistently missing from types.

  let value = i16x16::new([
    10000, 1001, 2, 3, 4, 5, 6, 32767, 10000, 1001, 2, 128, -129, -128, 127,
    255,
  ]);
  let expected = i8x16::new([
    16, -23, 2, 3, 4, 5, 6, -1, 16, -23, 2, -128, 127, -128, 127, -1,
  ]);
  let actual = i8x16::from_i16x16_truncate(value);
  assert_eq!(actual, expected);

  let value = i32x8::new([10000, 1001, 2, 3, 4, 5, -65536, 65536]);
  let expected = i16x8::new([10000, 1001, 2, 3, 4, 5, 0, 0]);
  let actual = i16x8::from_i32x8_truncate(value);
  assert_eq!(actual, expected);
}

#[test]
fn test_from_big_saturate() {
  // `from_{big}_saturate` is inconsistently missing from types.

  let value = i16x16::new([
    10000, 1001, 2, 3, 4, 5, 6, 32767, 10000, 1001, 2, 128, -129, -128, 127,
    255,
  ]);
  let expected = i8x16::new([
    127, 127, 2, 3, 4, 5, 6, 127, 127, 127, 2, 127, -128, -128, 127, 127,
  ]);
  let actual = i8x16::from_i16x16_saturate(value);
  assert_eq!(actual, expected);

  let value = i32x8::new([10000, 1001, 2, 3, 4, 5, -65535, 65536]);
  let expected = i16x8::new([10000, 1001, 2, 3, 4, 5, -32768, 32767]);
  let actual = i16x8::from_i32x8_saturate(value);
  assert_eq!(actual, expected);
}

#[test]
fn test_round_float() {
  // `round_float` only exists for select types.

  let value = i32x4::new([-1, 30, i32::MIN, i32::MAX]);
  let expected = f32x4::new([-1.0, 30.0, i32::MIN as f32, i32::MAX as f32]);
  let actual = value.round_float();
  assert_eq!(actual, expected);

  let value =
    i32x16::new([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]);
  let expected = f32x16::new([
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
    -2.0, -1.0,
  ]);
  let actual = value.round_float();
  assert_eq!(actual, expected);

  let value = i32x8::new([-1, 30, i32::MIN, i32::MAX, 29, 35, -8, 0]);
  let expected = f32x8::new([
    -1.0,
    30.0,
    i32::MIN as f32,
    i32::MAX as f32,
    29.0,
    35.0,
    -8.0,
    0.0,
  ]);
  let actual = value.round_float();
  assert_eq!(actual, expected);
}

#[test]
fn test_swizzle() {
  // `swizzle` is inconsistently missing from types.

  for (value, indices, expected) in [
    (
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ),
    (
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [15, 17, -13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0],
      [16, 0, 0, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0, 1],
    ),
  ] {
    let value = i8x16::new(value);
    let indices = i8x16::new(indices);
    let expected = i8x16::new(expected);
    let actual = value.swizzle(indices);
    assert_eq!(actual, expected);
  }
}

#[test]
fn test_swizzle_relaxed() {
  // `swizzle_relaxed` is inconsistently missing from types.

  for (value, indices, expected) in [
    (
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ),
    (
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      [15, -17, -13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0],
      [16, 0, 0, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0, 1],
    ),
  ] {
    let value = i8x16::new(value);
    let indices = i8x16::new(indices);
    let expected = i8x16::new(expected);
    let actual = value.swizzle_relaxed(indices);
    assert_eq!(actual, expected);
  }
}

#[test]
fn test_swizzle_half() {
  // `swizzle_half` is inconsistently missing from types.

  let value = i8x32::new([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
  ]);
  let indices = i8x32::new([
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
  ]);
  let expected = i8x32::new([
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 32, 31, 30, 29, 28,
    27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
  ]);
  let actual = value.swizzle_half(indices);
  assert_eq!(actual, expected);
}

/// Scalar reference: unsigned index in [0,31] selects table[idx], else 0.
fn ref_swizzle32(table: [i8; 32], idx: [i8; 32]) -> [i8; 32] {
  let mut out = [0i8; 32];
  for i in 0..32 {
    let ix = idx[i] as u8 as usize; // unsigned interpretation
    out[i] = if ix < 32 { table[ix] } else { 0 };
  }
  out
}

#[test]
fn test_i8x32_swizzle() {
  let table_arr: [i8; 32] = core::array::from_fn(|i| (i as i8) + 1); // 1..=32
  let table = i8x32::new(table_arr);
  let cases: [[i8; 32]; 4] = [
    core::array::from_fn(|i| i as i8), // identity
    core::array::from_fn(|i| 31 - i as i8), // reverse
    core::array::from_fn(|i| ((i + 16) % 32) as i8), // cross-half rotate by 16
    {
      let mut a = [3i8; 32];
      a[0] = 32; // out of range -> 0
      a[1] = 100; // out of range -> 0
      a[2] = -1; // 255 unsigned -> 0
      a
    },
  ];
  for idx_arr in cases {
    let expected = i8x32::new(ref_swizzle32(table_arr, idx_arr));
    let actual = table.swizzle(i8x32::new(idx_arr));
    assert_eq!(actual, expected, "idx={:?}", idx_arr);
  }
}

#[test]
fn test_i8x32_swizzle_relaxed() {
  let table_arr: [i8; 32] = core::array::from_fn(|i| (i as i8) + 1);
  let table = i8x32::new(table_arr);
  let cases: [[i8; 32]; 3] = [
    core::array::from_fn(|i| i as i8),               // identity
    core::array::from_fn(|i| 31 - i as i8),          // reverse
    core::array::from_fn(|i| ((i + 16) % 32) as i8), // cross-half
  ];
  for idx_arr in cases {
    let expected = i8x32::new(ref_swizzle32(table_arr, idx_arr)); // all in-range here
    let actual = table.swizzle_relaxed(i8x32::new(idx_arr));
    assert_eq!(actual, expected, "idx={:?}", idx_arr);
  }
}

#[test]
fn test_u8x32_swizzle() {
  let table_arr: [u8; 32] = core::array::from_fn(|i| (i as u8) + 1); // 1..=32
  let table = u8x32::new(table_arr);
  // strict: unsigned indices, out-of-range (incl. 128 and 255) -> 0
  let mut idx_arr = [4u8; 32];
  idx_arr[0] = 0;
  idx_arr[1] = 31;
  idx_arr[2] = 32; // OOR -> 0
  idx_arr[3] = 128; // OOR -> 0 (would be negative i8 after cast)
  idx_arr[4] = 255; // OOR -> 0
  let mut expected = [0u8; 32];
  for i in 0..32 {
    let ix = idx_arr[i] as usize;
    expected[i] = if ix < 32 { table_arr[ix] } else { 0 };
  }
  let actual = table.swizzle(u8x32::new(idx_arr));
  assert_eq!(actual, u8x32::new(expected), "idx={:?}", idx_arr);

  // relaxed: in-range only, must match table lookup
  let rev: [u8; 32] = core::array::from_fn(|i| 31 - i as u8);
  let rev_expected: [u8; 32] = core::array::from_fn(|i| table_arr[(31 - i) as usize]);
  assert_eq!(table.swizzle_relaxed(u8x32::new(rev)), u8x32::new(rev_expected));
}

#[test]
fn test_from_u8x16_low() {
  // This function only exists for select types.

  let value =
    u8x16::new([1, 2, 3, 4, 5, 6, 7, u8::MAX, 9, 10, 11, 12, 13, 14, 15, 16]);
  let expected = i16x8::new([1, 2, 3, 4, 5, 6, 7, u8::MAX as i16]);
  let actual = i16x8::from_u8x16_low(value);
  assert_eq!(actual, expected);

  let value =
    u8x16::from([255, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255, 128]);
  let expected = u16x8::from([255, 2, 3, 4, 5, 6, 7, 8]);
  let actual = u16x8::from_u8x16_low(value);
  assert_eq!(actual, expected);
}

#[test]
fn test_from_u8x16_high() {
  // This function only exists for select types.

  let value =
    u8x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255, 128]);
  let expected = i16x8::new([9, 10, 11, 12, 13, 14, 255, 128]);
  let actual = i16x8::from_u8x16_high(value);
  assert_eq!(actual, expected);

  let value =
    u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255, 128]);
  let expected = u16x8::from([9, 10, 11, 12, 13, 14, 255, 128]);
  let actual = u16x8::from_u8x16_high(value);
  assert_eq!(actual, expected);
}

#[test]
fn test_narrow_i16x8() {
  // This function only exists for select types.

  let a = i16x8::new([-1, 2, -3, 4, -5, 6, -7, 8]);
  let b = i16x8::new([9, 10, 11, 12, 13, -14, 15, -16]);
  let expected =
    u8x16::new([0, 2, 0, 4, 0, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 0]);
  let actual = u8x16::narrow_i16x8(a, b);
  assert_eq!(actual, expected);
}
