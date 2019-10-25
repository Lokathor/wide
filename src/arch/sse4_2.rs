#![cfg(target_feature="sse")]
#![cfg(target_feature="sse2")]
#![cfg(target_feature="sse3")]
#![cfg(target_feature="ssse3")]
#![cfg(target_feature="sse4.1")]
#![cfg(target_feature="sse4.2")]

use super::*;

/// # SSE4.2 Operations
impl m128i {
  /// Lanewise `i64` greater than, bool-ish output.
  #[inline(always)]
  pub fn cmp_gt_i64(self, rhs: Self) -> Self {
    Self(unsafe { _mm_cmpgt_epi64(self.0, rhs.0) })
  }
}

/// Accumulates the `u8` given into the CRC32 value, returning the new CRC32.
pub fn crc32_u8(crc: u32, byte: u8) -> u32 {
  unsafe { _mm_crc32_u8(crc, byte) }
}

/// Accumulates the `u16` given into the CRC32 value, returning the new CRC32.
pub fn crc32_u16(crc: u32, half_word: u16) -> u32 {
  unsafe { _mm_crc32_u16(crc, half_word) }
}

/// Accumulates the `u32` given into the CRC32 value, returning the new CRC32.
pub fn crc32_u32(crc: u32, word: u32) -> u32 {
  unsafe { _mm_crc32_u32(crc, word) }
}

/// Accumulates the `u64` given into the CRC32 value, returning the new CRC32.
#[cfg(target_arch = "x86_64")]
pub fn crc32_u64(crc: u64, double_word: u64) -> u64 {
  unsafe { _mm_crc32_u64(crc, double_word) }
}
