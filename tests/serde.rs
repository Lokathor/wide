#![cfg(all(feature = "serde", any(target_arch = "x86", target_arch = "x86_64")))]

use lokacore::*;

#[path = "intel-simd-help.rs"]
mod intel_simd_help;
use intel_simd_help::*;

#[test]
fn m128_serde_roundtrip() {
  use bincode::{deserialize, serialize};

  let m: m128 = cast([5.0_f32, 6.0, 7.0, 8.0]);

  // Roundtrip into new RNG.
  let buf = serialize(&m).unwrap();
  let m_return: m128 = deserialize(&buf[..]).unwrap();
  let m_return_f32s: [f32; 4] = cast(m_return);

  assert_eq!(m_return_f32s, [5.0_f32, 6.0, 7.0, 8.0]);
}

#[test]
fn m128i_serde_roundtrip() {
  use bincode::{deserialize, serialize};

  let m: m128i = cast(50_607_080_i128);

  // Roundtrip into new RNG.
  let buf = serialize(&m).unwrap();
  let m_return: m128i = deserialize(&buf[..]).unwrap();
  let m_return_128: i128 = cast(m_return);

  assert_eq!(m_return_128, 50_607_080_i128);
}

#[test]
fn m128d_serde_roundtrip() {
  use bincode::{deserialize, serialize};

  let m: m128d = cast([5.0_f64, 6.0]);

  // Roundtrip into new RNG.
  let buf = serialize(&m).unwrap();
  let m_return: m128d = deserialize(&buf[..]).unwrap();
  let m_return_f64s: [f64; 2] = cast(m_return);

  assert_eq!(m_return_f64s, [5.0_f64, 6.0]);
}
