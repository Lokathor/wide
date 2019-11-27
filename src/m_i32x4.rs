use super::*;

magic! {
  if #[cfg(target_feature="sse2")] {
    #[repr(C, align(16))]
    pub struct i32x4 {
      pub(crate) sse: m128i
    }
  } else {
    #[repr(C, align(16))]
    pub struct i32x4 {
      pub(crate) arr: [i32; 4]
    }
  }
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
#[repr(C, align(16))]
pub union ConstUnionHack_i32x4 {
  pub narrow_arr: [i32; 4],
  pub wide_thing: i32x4,
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
/// use wide::*;
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
/// use wide::*;
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

impl BitAnd for i32x4 {
  type Output = Self;
  fn bitand(self, rhs: Self) -> Self {
    magic! { if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.bitand(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0].bitand(rhs.arr[0]),
        self.arr[1].bitand(rhs.arr[1]),
        self.arr[2].bitand(rhs.arr[2]),
        self.arr[3].bitand(rhs.arr[3]),
      ] }
    }}
  }
}

impl BitAndAssign for i32x4 {
  fn bitand_assign(&mut self, rhs: Self) {
    magic! { if #[cfg(target_feature="sse2")] {
      self.sse.bitand_assign(rhs.sse)
    } else {
      self.arr[0].bitand_assign(rhs.arr[0]);
      self.arr[1].bitand_assign(rhs.arr[1]);
      self.arr[2].bitand_assign(rhs.arr[2]);
      self.arr[3].bitand_assign(rhs.arr[3]);
    }}
  }
}

impl i32x4 {
  #[inline]
  pub fn cmp_eq(self, rhs: Self) -> Self {
    magic! { if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.cmp_eq_i32(rhs.sse) }
    } else {
      let op = |a:i32, b:i32|{
        if a == b {
          -1
        } else {
          0
        }
      };
      Self { arr: [
        op(self.arr[0], rhs.arr[0]),
        op(self.arr[1], rhs.arr[1]),
        op(self.arr[2], rhs.arr[2]),
        op(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }

  #[inline]
  pub fn cmp_gt(self, rhs: Self) -> Self {
    magic! { if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.cmp_gt_i32(rhs.sse) }
    } else {
      let op = |a:i32, b:i32|{
        if a > b {
          -1
        } else {
          0
        }
      };
      Self { arr: [
        op(self.arr[0], rhs.arr[0]),
        op(self.arr[1], rhs.arr[1]),
        op(self.arr[2], rhs.arr[2]),
        op(self.arr[3], rhs.arr[3]),
      ] }
    }}
  }
}

impl Not for i32x4 {
  type Output = Self;

  fn not(self) -> Self {
    magic! { if #[cfg(target_feature="sse2")] {
      Self { sse: !self.sse }
    } else {
      Self { arr: [
        !self.arr[0],
        !self.arr[1],
        !self.arr[2],
        !self.arr[3],
      ] }
    }}
  }
}

impl i32x4 {
  #[inline(always)]
  pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self {
    magic! {if #[cfg(target_feature="sse2")] {
      Self { sse: m128i::set_reverse_i32(a,b,c,d) }
    } else {
      Self { arr: [a,b,c,d] }
    }}
  }
}

impl From<i32> for i32x4 {
  fn from(i: i32) -> Self {
    Self::new(i, i, i, i)
  }
}

impl Shl<i32> for i32x4 {
  type Output = Self;
  fn shl(self, rhs: i32) -> Self {
    magic! {if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.shift_left_i32(i32x4::from(rhs).sse) }
    } else {
      Self { arr: [
        self.arr[0] << rhs,
        self.arr[1] << rhs,
        self.arr[2] << rhs,
        self.arr[3] << rhs,
      ] }
    }}
  }
}

impl BitXor for i32x4 {
  type Output = Self;
  fn bitxor(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.bitxor(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0].bitxor(rhs.arr[0]),
        self.arr[1].bitxor(rhs.arr[1]),
        self.arr[2].bitxor(rhs.arr[2]),
        self.arr[3].bitxor(rhs.arr[3]),
      ] }
    }}
  }
}

impl Add for i32x4 {
  type Output = Self;
  fn add(self, rhs: Self) -> Self {
    magic! {if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.add_i32(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0].add(rhs.arr[0]),
        self.arr[1].add(rhs.arr[1]),
        self.arr[2].add(rhs.arr[2]),
        self.arr[3].add(rhs.arr[3]),
      ] }
    }}
  }
}

impl BitOr for i32x4 {
  type Output = Self;
  fn bitor(self, rhs: Self) -> Self {
    magic! { if #[cfg(target_feature="sse2")] {
      Self { sse: self.sse.bitor(rhs.sse) }
    } else {
      Self { arr: [
        self.arr[0].bitor(rhs.arr[0]),
        self.arr[1].bitor(rhs.arr[1]),
        self.arr[2].bitor(rhs.arr[2]),
        self.arr[3].bitor(rhs.arr[3]),
      ] }
    }}
  }
}

impl BitOrAssign for i32x4 {
  fn bitor_assign(&mut self, rhs: Self) {
    magic! { if #[cfg(target_feature="sse2")] {
      self.sse.bitor_assign(rhs.sse)
    } else {
      self.arr[0].bitor_assign(rhs.arr[0]);
      self.arr[1].bitor_assign(rhs.arr[1]);
      self.arr[2].bitor_assign(rhs.arr[2]);
      self.arr[3].bitor_assign(rhs.arr[3]);
    }}
  }
}
