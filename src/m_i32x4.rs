use super::*;

cfg_if! {
  if #[cfg(target_feature="sse")] {
    #[repr(C, align(16))]
    pub struct i32x4 {
      sse: m128i
    }
  } else {
    #[repr(C, align(16))]
    pub struct i32x4 {
      arr: [i32; 4]
    }
  }
}
#[test]
fn declaration_tests_i32x4() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<i32x4>(), 16);
  assert_eq!(align_of::<i32x4>(), 16);
}
impl Clone for i32x4 {
  #[inline(always)]
  fn clone(&self) -> Self {
    *self
  }
}
impl Copy for i32x4 {}
impl Default for i32x4 {
  #[inline(always)]
  fn default() -> Self {
    Self::zeroed()
  }
}
unsafe impl Zeroable for i32x4 {}
unsafe impl Pod for i32x4 {}

#[allow(non_camel_case_types)]
pub union ConstUnionHack_i32x4 {
  narrow_arr: [i32; 4],
  wide_thing: i32x4,
}
#[test]
#[allow(non_snake_case)]
fn declaration_tests_ConstUnionHack_i32x4() {
  use core::mem::{align_of, size_of};
  assert_eq!(size_of::<ConstUnionHack_i32x4>(), size_of::<i32x4>());
  assert_eq!(align_of::<ConstUnionHack_i32x4>(), align_of::<i32x4>());
}

/// Declares an `i32x4` const identifier.
///
/// ## Broadcast A Single Value
///
/// * **Usage:** `const_i32_as_i32x4!(#[meta]* vis ident, val);`
///
/// The value should be a single `i32` expression, which is then duplicated into
/// all lanes of the constant declaration.
///
/// ```rust
/// const_i32_as_i32x4!(
///   /// The maximum `i32` value
///   pub MAX, core::i32::MAX
/// );
/// ```
///
/// ## Select Each Lane
///
/// * **Usage:** `const_i32_as_i32x4!(#[meta]* vis ident, a, b, c, d);`
///
/// Each of `a`, `b`, `c`, and `d` are an `i32` expression when are then placed
/// into the constant declaration (low lane to high lane).
///
/// ```rust
/// const_i32_as_i32x4!(
///   /// 1, 2, 3, 4
///   pub ONE_TWO_THREE_FOUR, 1, 2, 3, 4
/// );
/// ```
#[macro_export]
macro_rules! const_i32_as_i32x4 {
  // broadcast a single value
  ($(#[$attrs:meta])* $v:vis $i:ident, $val:expr) => {
    $(#[$attrs])*
    $v const $i: i32x4 = {
      let cuh = ConstUnionHack_i32x4 {
        narrow_arr: [$val, $val, $val, $val],
      };
      unsafe { cuh.wide_thing }
    };
  };
  // select each lane's value
  ($(#[$attrs:meta])* $v:vis $i:ident, $a:expr, $b:expr, $c:expr, $d:expr) => {
    $(#[$attrs])*
    $v const $i: i32x4 = {
      let cuh = ConstUnionHack_i32x4 {
        narrow_arr: [$a, $b, $c, $d],
      };
      unsafe { cuh.wide_thing }
    };
  };
}

impl i32x4 {
  //
  // core::i32
  //

  const_i32_as_i32x4!(
    /// Maximum `i32` value.
    pub MAX, core::i32::MAX
  );

  const_i32_as_i32x4!(
    /// Minimum `i32` value.
    pub MIN, core::i32::MIN
  );

  //
  // others
  //

  const_i32_as_i32x4!(
    /// 0
    pub ZERO, 0_i32
  );

  const_i32_as_i32x4!(
    /// 1
    pub ONE, 1_i32
  );
}
