use wide::*;

use bytemuck::*;

#[test]
fn size_align() {
  assert_eq!(core::mem::size_of::<f32x4>(), 16);
  assert_eq!(core::mem::align_of::<f32x4>(), 16);
}

#[test]
fn impl_debug_for_f32x4() {
  let expected = "(1.0, 2.0, 3.0, 4.0)";
  let actual = format!("{:?}", f32x4::from([1.0, 2.0, 3.0, 4.0]));
  assert_eq!(expected, actual);

  let expected = "(1.000, 2.000, 3.000, 4.000)";
  let actual = format!("{:.3?}", f32x4::from([1.0, 2.0, 3.0, 4.0]));
  assert_eq!(expected, actual);
}

#[test]
fn impl_add_for_f32x4() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([5.0, 6.0, 7.0, 8.0]);
  let expected = f32x4::from([6.0, 8.0, 10.0, 12.0]);
  let actual = a + b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_sub_for_f32x4() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([5.0, 7.0, 17.0, 1.0]);
  let expected = f32x4::from([-4.0, -5.0, -14.0, 3.0]);
  let actual = a - b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_mul_for_f32x4() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([5.0, 7.0, 17.0, 1.0]);
  let expected = f32x4::from([5.0, 14.0, 51.0, 4.0]);
  let actual = a * b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_div_for_f32x4() {
  let a = f32x4::from([4.0, 9.0, 10.0, 12.0]);
  let b = f32x4::from([2.0, 2.0, 5.0, -3.0]);
  let expected = f32x4::from([2.0, 4.5, 2.0, -4.0]);
  let actual = a / b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitand_for_f32x4() {
  let a = f32x4::from([0.0, 0.0, 1.0, 1.0]);
  let b = f32x4::from([0.0, 1.0, 0.0, 1.0]);
  let expected = f32x4::from([0.0, 0.0, 0.0, 1.0]);
  let actual = a & b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitor_for_f32x4() {
  let a = f32x4::from([0.0, 0.0, 1.0, 1.0]);
  let b = f32x4::from([0.0, 1.0, 0.0, 1.0]);
  let expected = f32x4::from([0.0, 1.0, 1.0, 1.0]);
  let actual = a | b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_bitxor_for_f32x4() {
  let a = f32x4::from([0.0, 0.0, 1.0, 1.0]);
  let b = f32x4::from([0.0, 1.0, 0.0, 1.0]);
  let expected = f32x4::from([0.0, 1.0, 1.0, 0.0]);
  let actual = a ^ b;
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_cmp_eq() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([2.0, 2.0, 2.0, 2.0]);
  let expected: [i32; 4] = [0, -1, 0, 0];
  let actual: [i32; 4] = cast(a.cmp_eq(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_cmp_ne() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([2.0, 2.0, 2.0, 2.0]);
  let expected: [i32; 4] = [-1, 0, -1, -1];
  let actual: [i32; 4] = cast(a.cmp_ne(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_cmp_ge() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([2.0, 2.0, 2.0, 2.0]);
  let expected: [i32; 4] = [0, -1, -1, -1];
  let actual: [i32; 4] = cast(a.cmp_ge(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_cmp_gt() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([2.0, 2.0, 2.0, 2.0]);
  let expected: [i32; 4] = [0, 0, -1, -1];
  let actual: [i32; 4] = cast(a.cmp_gt(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_cmp_le() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([2.0, 2.0, 2.0, 2.0]);
  let expected: [i32; 4] = [-1, -1, 0, 0];
  let actual: [i32; 4] = cast(a.cmp_le(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_cmp_lt() {
  let a = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let b = f32x4::from([2.0, 2.0, 2.0, 2.0]);
  let expected: [i32; 4] = [-1, 0, 0, 0];
  let actual: [i32; 4] = cast(a.cmp_lt(b));
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_blend() {
  let use_t: f32 = f32::from_bits(u32::MAX);
  let t = f32x4::from([1.0, 2.0, 3.0, 4.0]);
  let f = f32x4::from([5.0, 6.0, 7.0, 8.0]);
  let mask = f32x4::from([use_t, 0.0, use_t, 0.0]);
  let expected = f32x4::from([1.0, 6.0, 3.0, 8.0]);
  let actual = mask.blend(t, f);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_abs() {
  let a = f32x4::from([-1.0, 2.0, -3.5, f32::NEG_INFINITY]);
  let expected = f32x4::from([1.0, 2.0, 3.5, f32::INFINITY]);
  let actual = a.abs();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_max() {
  let a = f32x4::from([1.0, 5.0, 3.0, f32::NAN]);
  let b = f32x4::from([2.0, f32::NEG_INFINITY, f32::INFINITY, 10.0]);
  let expected = f32x4::from([2.0, 5.0, f32::INFINITY, 10.0]);
  let actual = a.max(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_min() {
  let a = f32x4::from([1.0, 5.0, 3.0, f32::NAN]);
  let b = f32x4::from([2.0, f32::NEG_INFINITY, f32::INFINITY, 10.0]);
  let expected = f32x4::from([1.0, f32::NEG_INFINITY, 3.0, 10.0]);
  let actual = a.min(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_is_nan() {
  let a = f32x4::from([0.0, f32::NAN, f32::NAN, 0.0]);
  let expected = [0, u32::MAX, u32::MAX, 0];
  let actual: [u32; 4] = cast(a.is_nan());
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_is_finite() {
  let a = f32x4::from([f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY]);
  let expected = [0, u32::MAX, 0, 0];
  let actual: [u32; 4] = cast(a.is_finite());
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_round() {
  let a = f32x4::from([1.1, 2.5, 3.7, 4.0]);
  let expected = f32x4::from([1.0, 2.0, 4.0, 4.0]);
  let actual = a.round();
  assert_eq!(expected, actual);
  //
  let a = f32x4::from([-1.1, -2.5, -3.7, -4.0]);
  let expected = f32x4::from([-1.0, -2.0, -4.0, -4.0]);
  let actual = a.round();
  assert_eq!(expected, actual);
  //
  let a = f32x4::from([f32::INFINITY, f32::NEG_INFINITY, 5.5, 5.0]);
  let expected = f32x4::from([f32::INFINITY, f32::NEG_INFINITY, 6.0, 5.0]);
  let actual = a.round();
  assert_eq!(expected, actual);
  //
  let a = f32x4::from(f32::NAN);
  let expected: [u32; 4] = [u32::MAX; 4];
  let actual: [u32; 4] = cast(a.round().is_nan());
  assert_eq!(expected, actual);
  //
  let a = f32x4::from(-0.0);
  let expected = a;
  let actual = a.round();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_round_int() {
  for (f, i) in [
    (1.0, 1),
    (1.1, 1),
    (-2.1, -2),
    (2.5, 2),
    (0.0, 0),
    (-0.0, 0),
    (f32::NAN, i32::MIN),
    (f32::INFINITY, i32::MIN),
    (f32::NEG_INFINITY, i32::MIN),
  ]
  .iter()
  .copied()
  {
    let a = f32x4::from(f);
    let expected = i32x4::from(i);
    let actual = a.round_int();
    assert_eq!(expected, actual);
  }
}

#[test]
fn impl_f32x4_mul_add() {
  let a = f32x4::from([2.0, 3.0, 4.0, 5.0]);
  let b = f32x4::from([4.0, 5.0, 6.0, 7.0]);
  let c = f32x4::from([1.0, 1.0, 1.0, 1.0]);
  let expected = f32x4::from([9.0, 16.0, 25.0, 36.0]);
  let actual = a.mul_add(b, c);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_mul_neg_add() {
  let a = f32x4::from([2.0, 3.0, 4.0, 5.0]);
  let b = f32x4::from([4.0, 5.0, 6.0, 7.0]);
  let c = f32x4::from([1.0, 1.0, 1.0, 1.0]);
  let expected = f32x4::from([-7.0, -14.0, -23.0, -34.0]);
  let actual = a.mul_neg_add(b, c);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_flip_signs() {
  let a = f32x4::from([1.0, 1.0, -1.0, -1.0]);
  let b = f32x4::from([2.0, -3.0, 4.0, -5.0]);
  let expected = f32x4::from([1.0, -1.0, -1.0, 1.0]);
  let actual = a.flip_signs(b);
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_sin_cos() {
  for x in -2500..=2500 {
    let base = (x * 4) as f32;
    let angles = [base, base + 1.0, base + 2.0, base + 3.0];
    let (actual_sins, actual_coses) = f32x4::from(angles).sin_cos();
    for i in 0..4 {
      let angle = angles[i];
      let check = |name: &str, vals: f32x4, expected: f32| {
        let actual_arr: [f32; 4] = cast(vals);
        let actual = actual_arr[i];
        assert!(
          (actual - expected).abs() < 0.00000006,
          "Wanted {name}({angle}) to be {expected} but got {actual}",
          name = name,
          angle = angle,
          expected = expected,
          actual = actual
        );
      };
      check("sin", actual_sins, angle.sin());
      check("cos", actual_coses, angle.cos());
    }
  }
}

#[test]
fn impl_f32x4_to_degrees() {
  let pi = core::f32::consts::PI;
  let a = f32x4::from([0.0, pi / 2.0, pi, 2.0 * pi]);
  let expected = f32x4::from([0.0, 90.0, 180.0, 360.0]);
  let actual = a.to_degrees();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_to_radians() {
  let pi = core::f32::consts::PI;
  let a = f32x4::from([0.0, 90.0, 180.0, 360.0]);
  let expected = f32x4::from([0.0, pi / 2.0, pi, 2.0 * pi]);
  let actual = a.to_radians();
  assert_eq!(expected, actual);
}

#[test]
fn impl_f32x4_sqrt() {
  for (f, e) in [
    (f32::INFINITY, f32::INFINITY),
    (0.0, 0.0),
    (-0.0, -0.0),
    (4.0, 2.0),
    (9.0, 3.0),
    (16.0, 4.0),
    (25.0, 5.0),
    (5000.0 * 5000.0, 5000.0),
  ]
  .iter()
  .copied()
  {
    let expected = f32x4::from(e);
    let actual = f32x4::from(f).sqrt();
    assert_eq!(expected, actual);
  }
  assert_eq!(
    cast::<_, i32x4>(f32x4::from(f32::NAN).sqrt().is_nan()),
    i32x4::from(-1)
  );
  assert_eq!(
    cast::<_, i32x4>(f32x4::from(f32::NEG_INFINITY).sqrt().is_nan()),
    i32x4::from(-1)
  );
  assert_eq!(
    cast::<_, i32x4>(f32x4::from(-1.0).sqrt().is_nan()),
    i32x4::from(-1)
  );
}

#[test]
fn impl_f32x4_exp() {
  for f in [(-2.0), (-1.0), (0.0), (1.0), (1.5), (2.0), (10.0)].iter().copied()
  {
    let expected = f32x4::from((f as f32).exp());
    let actual = f32x4::from(f).exp();
    let diff_from_std: [f32; 4] = cast((actual - expected).abs());
    assert!(diff_from_std[0] < 0.000000000000001);
  }
}

#[test]
fn test_f32x4_move_mask() {
  let a = f32x4::from([-1.0, 0.0, -2.0, -3.0]);
  let expected = 0b1101;
  let actual = a.move_mask();
  assert_eq!(expected, actual);
  //
  let a = f32x4::from([1.0, 0.0, 2.0, -3.0]);
  let expected = 0b1000;
  let actual = a.move_mask();
  assert_eq!(expected, actual);
}

#[test]
fn test_f32x4_any() {
  let a = f32x4::from([-1.0, 0.0, -2.0, -3.0]);
  assert!(a.any());
  //
  let a = f32x4::from([1.0, 0.0, 2.0, 3.0]);
  assert!(!a.any());
}

#[test]
fn test_f32x4_all() {
  let a = f32x4::from([-1.0, -0.0, -2.0, -3.0]);
  assert!(a.all());
  //
  let a = f32x4::from([1.0, -0.0, 2.0, 3.0]);
  assert!(!a.all());
}

#[test]
fn test_f32x4_none() {
  let a = f32x4::from([1.0, 0.0, 2.0, 3.0]);
  assert!(a.none());
  //
  let a = f32x4::from([1.0, -0.0, 2.0, 3.0]);
  assert!(!a.none());
}
