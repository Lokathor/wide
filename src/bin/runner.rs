/*
use wide::*;

use bytemuck::*;

/// idct from jpeg-decode
#[inline(never)]
fn dequantize_and_idct_block_8x8(
  coefficients: &[i16; 64],
  quantization_table: &[i16; 64],
  output_linestride: usize,
  output: &mut [u8],
) {
  fn idct8(data: [i16x8; 8]) -> [i16x8; 8] {
    let p2 = data[2];
    let p3 = data[6];
    let p1 = p2.saturating_add(p3).mul_scale_round(i16x8::splat(17734)); // 0.5411961
    let t2 = p1
      .saturating_sub(p3)
      .saturating_sub(p3.mul_scale_round(i16x8::splat(27779))); // 0.847759065
    let t3 = p1.saturating_add(p2.mul_scale_round(i16x8::splat(25079))); // 0.765366865

    let p2 = data[0];
    let p3 = data[4];
    let t0 = p2.saturating_add(p3);
    let t1 = p2.saturating_sub(p3);

    let x0 = t0.saturating_add(t3);
    let x3 = t0.saturating_sub(t3);
    let x1 = t1.saturating_add(t2);
    let x2 = t1.saturating_sub(t2);

    let t0 = data[7];
    let t1 = data[5];
    let t2 = data[3];
    let t3 = data[1];

    let p3 = t0.saturating_add(t2);
    let p4 = t1.saturating_add(t3);
    let p1 = t0.saturating_add(t3);
    let p2 = t1.saturating_add(t2);
    let p5 = p3.saturating_add(p4);
    let p5 = p5.saturating_add(p5.mul_scale_round(5763)); // 0.175875602

    let t0a = t0.mul_scale_round(9786); // 0.298631336
    let t1a = t1.saturating_add(t1).saturating_add(t1.mul_scale_round(1741)); // 0.053119869

    let t2a = t2
      .saturating_add(t2)
      .saturating_add(t2)
      .saturating_add(t2.mul_scale_round(i16x8::splat(2383))); // 0.072711026
    let t3a = t3.saturating_add(t3.mul_scale_round(16427)); // 0.501321110

    let p1a = p5.saturating_sub(p1.mul_scale_round(29490)); // 0.899976223
    let p2a = p5
      .saturating_sub(p2)
      .saturating_sub(p2)
      .saturating_sub(p2.mul_scale_round(i16x8::splat(18446))); // 0.562915447

    let p3a = p3.mul_scale_round(-31509).saturating_sub(p3); // -0.961570560
    let p4a = p4.mul_scale_round(-12785); // -0.390180644

    let t3b = p1a.saturating_add(p4a).saturating_add(t3a);
    let t2b = p2a.saturating_add(p3a).saturating_add(t2a);
    let t1b = p2a.saturating_add(p4a).saturating_add(t1a);
    let t0b = p1a.saturating_add(p3a).saturating_add(t0a);

    [
      x0.saturating_add(t3b),
      x1.saturating_add(t2b),
      x2.saturating_add(t1b),
      x3.saturating_add(t0b),
      x3.saturating_sub(t0b),
      x2.saturating_sub(t1b),
      x1.saturating_sub(t2b),
      x0.saturating_sub(t3b),
    ]
  }

  fn transpose8(data: [i16x8; 8]) -> [i16x8; 8] {
    fn transpose_column(data: &[i16x8; 8], index: usize) -> i16x8 {
      i16x8::new([
        data[0].as_array_ref()[index],
        data[1].as_array_ref()[index],
        data[2].as_array_ref()[index],
        data[3].as_array_ref()[index],
        data[4].as_array_ref()[index],
        data[5].as_array_ref()[index],
        data[6].as_array_ref()[index],
        data[7].as_array_ref()[index],
      ])
    }

    [
      transpose_column(&data, 0),
      transpose_column(&data, 1),
      transpose_column(&data, 2),
      transpose_column(&data, 3),
      transpose_column(&data, 4),
      transpose_column(&data, 5),
      transpose_column(&data, 6),
      transpose_column(&data, 7),
    ]
  }

  fn from_slice(c: &[i16; 64], row: usize) -> i16x8 {
    let r = &c[row * 8..row * 8 + 8];
    i16x8::new([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
  }

  fn to_u8(result: i16x8, output: &mut [u8]) {
    output[0] = result.as_array_ref()[0] as u8;
    output[1] = result.as_array_ref()[1] as u8;
    output[2] = result.as_array_ref()[2] as u8;
    output[3] = result.as_array_ref()[3] as u8;
    output[4] = result.as_array_ref()[4] as u8;
    output[5] = result.as_array_ref()[5] as u8;
    output[6] = result.as_array_ref()[6] as u8;
    output[7] = result.as_array_ref()[7] as u8;
  }

  const SHIFT: i16 = 3;

  let data = [
    from_slice(coefficients, 0) * from_slice(quantization_table, 0) << SHIFT,
    from_slice(coefficients, 1) * from_slice(quantization_table, 1) << SHIFT,
    from_slice(coefficients, 2) * from_slice(quantization_table, 2) << SHIFT,
    from_slice(coefficients, 3) * from_slice(quantization_table, 3) << SHIFT,
    from_slice(coefficients, 4) * from_slice(quantization_table, 4) << SHIFT,
    from_slice(coefficients, 5) * from_slice(quantization_table, 5) << SHIFT,
    from_slice(coefficients, 6) * from_slice(quantization_table, 6) << SHIFT,
    from_slice(coefficients, 7) * from_slice(quantization_table, 7) << SHIFT,
  ];

  let pass1 = idct8(data);
  let transpose1 = transpose8(pass1);
  let pass2 = idct8(transpose1);
  let result = transpose8(pass2);

  const OFFSET: i16 = 128 << (SHIFT + 3);
  const ROUNDING_BIAS: i16 = (1 << (SHIFT + 3)) >> 1;

  let add = i16x8::splat(OFFSET + ROUNDING_BIAS);

  for i in 0..8 {
    to_u8(
      result[i].saturating_add(add) >> (SHIFT + 3),
      &mut output[i * output_linestride..i * output_linestride + 8],
    );
  }
}

fn test_dequantize_and_idct_block_8x8(i: i16) -> u8 {
  #[cfg_attr(rustfmt, rustfmt_skip)]
    let coefficients: [i16; 8 * 8] = [
        i, -39, 58, -2, 3, 3, 0, 1,
        11, 27, 4, -3, 3, 0, 1, 0,
        -6, -13, -9, -1, -2, -1, 0, 0,
        -4, 0, -1, -2, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        -3, -2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ];

  #[cfg_attr(rustfmt, rustfmt_skip)]
    let quantization_table: [i16; 8 * 8] = [
        8, 6, 5, 8, 12, 20, 26, 31,
        6, 6, 7, 10, 13, 29, 30, 28,
        7, 7, 8, 12, 20, 29, 35, 28,
        7, 9, 11, 15, 26, 44, 40, 31,
        9, 11, 19, 28, 34, 55, 52, 39,
        12, 18, 28, 32, 41, 52, 57, 46,
        25, 32, 39, 44, 52, 61, 60, 51,
        36, 46, 48, 49, 56, 50, 52, 50
    ];
  let output_linestride: usize = 8;
  let mut output = [0u8; 8 * 8];
  dequantize_and_idct_block_8x8(
    &coefficients,
    &quantization_table,
    output_linestride,
    &mut output,
  );
  #[cfg_attr(rustfmt, rustfmt_skip)]
    let expected_output = [
        118, 92, 110, 83, 77, 93, 144, 198,
        172, 116, 114, 87, 78, 93, 146, 191,
        194, 107, 91, 76, 71, 93, 160, 198,
        196, 100, 80, 74, 67, 92, 174, 209,
        182, 104, 88, 81, 68, 89, 178, 206,
        105, 64, 59, 59, 63, 94, 183, 201,
        35, 27, 28, 37, 72, 121, 203, 204,
        38, 45, 41, 47, 99, 154, 223, 208
    ];

  output.iter().sum()
  //assert_eq!(output, expected_output);
}
*/
fn main() {
  /*
  let mut s = 0;
  for _i in 0..10000000 {
    s += test_dequantize_and_idct_block_8x8(_i as i16) as i32;
  }*/
  println!("done");
}
