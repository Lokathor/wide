/// Emits functionality shared by all SIMD signed-integer types.
///
/// Functions that need a separate implementation for each type (for
/// performance) use `$fn_{name}:item` syntax, and functions that have one
/// shared implementation for all ints are written out normally inside this
/// macro.
///
/// This macro also invokes `impl_simd`.
macro_rules! impl_simd_signed {
  (
    // SAFETY: The contents of this macro assume that:
    //
    // - `T` implements `Pod`
    // - `Pod` can be implemented for `Simd`
    // - `size_of::<Simd>()` is `size_of::<T>() * N`
    // - `align_of::<Simd>()` is `size_of::<Simd>()`
    // - `Pod` can be implemented for the optional native SIMD types
    unsafe {
      T = $T:ident,
      N = $N:literal,
      Simd = $Simd:ident,
      UnsignedSimd = $UnsignedSimd:ident,
      T_BITS = $T_BITS:literal,
      T_BITS_MUL_2 = $T_BITS_MUL_2:literal,
      BitmaskType = $BitmaskType:ty,
      [$($index:literal),* $(,)?],
      optional_type_x86_inner { $(X86Inner = $X86Inner:ident)? },
      optional_type_arm_inner { $(ArmInner = $ArmInner:ident)? },
      optional_type_wasm_inner { $(WasmInner = $WasmInner:ident)? },
    }

    // General SIMD functions
    $fn_simd_lt:item
    $fn_simd_gt:item
    $fn_simd_le:item
    $fn_simd_ge:item

    // Integer functions
    $fn_shr_unsigned_simd:item
    $fn_shr_u32:item
    $fn_max:item
    $fn_min:item
    $fn_reduce_max:item
    $fn_reduce_min:item
    $fn_unbounded_shr:item
    $fn_unbounded_shr_scalar:item
    $fn_saturating_add:item
    $fn_saturating_sub:item
    $fn_overflowing_mul:item
    optional_fn_widening_mul { $($fn_widening_mul:item)? }
    $fn_mul_keep_low_high:item
    $fn_mul_keep_high:item

    // Signed-integer functions
    $fn_abs:item
    $fn_is_positive:item
    $fn_is_negative:item
  ) => {
    impl_simd!(
      unsafe {
        T = $T,
        N = $N,
        Simd = $Simd,
        UnsignedSimd = $UnsignedSimd,
        optional_type_x86_inner { $(X86Inner = $X86Inner)? },
        optional_type_arm_inner { $(ArmInner = $ArmInner)? },
        optional_type_wasm_inner { $(WasmInner = $WasmInner)? },
      }

      #[inline]
      fn simd_eq(self, other: Self) -> Self {
        self.cast_unsigned().simd_eq(other.cast_unsigned()).cast_signed()
      }

      #[inline]
      fn simd_ne(self, other: Self) -> Self {
        self.cast_unsigned().simd_ne(other.cast_unsigned()).cast_signed()
      }

      $fn_simd_lt

      $fn_simd_gt

      $fn_simd_le

      $fn_simd_ge

      #[inline]
      pub fn replace_const<const INDEX: usize>(self, value: $T) -> Self {
        self.cast_unsigned().replace_const::<INDEX>(value.cast_unsigned()).cast_signed()
      }

      #[inline]
      pub fn extract_const<const INDEX: usize>(self) -> $T {
        self.cast_unsigned().extract_const::<INDEX>().cast_signed()
      }

      #[inline]
      pub fn reduce_add(self) -> $T {
        // Wrapping addition is the same for signed and unsigned integers.
        cast::<$Simd, $UnsignedSimd>(self).reduce_add().cast_signed()
      }

      #[inline]
      pub fn reduce_mul(self) -> $T {
        // Wrapping multiplication is the same for signed and unsigned integers.
        cast::<$Simd, $UnsignedSimd>(self).reduce_mul().cast_signed()
      }

      #[inline]
      pub fn bitselect(self, if_one: Self, if_zero: Self) -> Self {
        self.cast_unsigned()
          .bitselect(if_one.cast_unsigned(), if_zero.cast_unsigned())
          .cast_signed()
      }

      #[inline]
      fn select(self, if_true: Self, if_false: Self) -> Self {
        self.cast_unsigned()
          .select(if_true.cast_unsigned(), if_false.cast_unsigned())
          .cast_signed()
      }

      #[inline]
      pub fn to_bitmask(self) -> $BitmaskType {
        self.cast_unsigned().to_bitmask()
      }

      #[inline]
      pub fn any(self) -> bool {
        self.cast_unsigned().any()
      }

      #[inline]
      pub fn all(self) -> bool {
        self.cast_unsigned().all()
      }

      #[inline]
      pub fn unpack_lo(self, other: Self) -> Self {
        self.cast_unsigned().unpack_lo(other.cast_unsigned()).cast_signed()
      }

      #[inline]
      pub fn unpack_hi(self, other: Self) -> Self {
        self.cast_unsigned().unpack_hi(other.cast_unsigned()).cast_signed()
      }

      #[inline]
      pub fn shuffle(self, indices: $UnsignedSimd) -> Self {
        self.cast_unsigned().shuffle(indices).cast_signed()
      }

      #[inline]
      pub fn shuffle_zeroing(self, indices: $UnsignedSimd) -> Self {
        self.cast_unsigned().shuffle_zeroing(indices).cast_signed()
      }

      #[inline]
      pub fn shuffle_wrapping(self, indices: $UnsignedSimd) -> Self {
        self.cast_unsigned().shuffle_wrapping(indices).cast_signed()
      }

      #[inline]
      fn shuffle(self: [$Simd; 2], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 2], [$UnsignedSimd; 2]>(self).shuffle(indices))
      }

      #[inline]
      fn shuffle_zeroing(self: [$Simd; 2], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 2], [$UnsignedSimd; 2]>(self).shuffle_zeroing(indices))
      }

      #[inline]
      fn shuffle_wrapping(self: [$Simd; 2], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 2], [$UnsignedSimd; 2]>(self).shuffle_wrapping(indices))
      }

      #[inline]
      fn shuffle(self: [$Simd; 3], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 3], [$UnsignedSimd; 3]>(self).shuffle(indices))
      }

      #[inline]
      fn shuffle_zeroing(self: [$Simd; 3], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 3], [$UnsignedSimd; 3]>(self).shuffle_zeroing(indices))
      }

      #[inline]
      fn shuffle_wrapping(self: [$Simd; 3], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 3], [$UnsignedSimd; 3]>(self).shuffle_wrapping(indices))
      }

      #[inline]
      fn shuffle(self: [$Simd; 4], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 4], [$UnsignedSimd; 4]>(self).shuffle(indices))
      }

      #[inline]
      fn shuffle_zeroing(self: [$Simd; 4], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 4], [$UnsignedSimd; 4]>(self).shuffle_zeroing(indices))
      }

      #[inline]
      fn shuffle_wrapping(self: [$Simd; 4], indices: $UnsignedSimd) -> $Simd {
        cast(cast::<[$Simd; 4], [$UnsignedSimd; 4]>(self).shuffle_wrapping(indices))
      }

      #[inline]
      pub fn transpose(data: [Self; $N]) -> [Self; $N] {
        cast($UnsignedSimd::transpose(cast::<[$Simd; $N], [$UnsignedSimd; $N]>(data)))
      }
    );

    impl_simd_integer!(
      unsafe {
        T = $T,
        N = $N,
        Simd = $Simd,
        UnsignedSimd = $UnsignedSimd,
        SignedSimd = $Simd,
        T_BITS = $T_BITS,
        T_BITS_MUL_2 = $T_BITS_MUL_2,
        [$($index),*],
      }

      #[inline]
      fn not(self) -> Self::Output {
        cast::<$UnsignedSimd, $Simd>(!cast::<$Simd, $UnsignedSimd>(self))
      }

      #[inline]
      fn add(self, rhs: Self) -> Self::Output {
        // Wrapping addition is the same for signed and unsigned integers.
        cast::<$UnsignedSimd, $Simd>(
          cast::<$Simd, $UnsignedSimd>(self) + cast::<$Simd, $UnsignedSimd>(rhs),
        )
      }

      #[inline]
      fn sub(self, rhs: Self) -> Self::Output {
        // Wrapping subtraction is the same for signed and unsigned integers.
        cast::<$UnsignedSimd, $Simd>(
          cast::<$Simd, $UnsignedSimd>(self) - cast::<$Simd, $UnsignedSimd>(rhs),
        )
      }

      #[inline]
      fn mul(self, rhs: Self) -> Self::Output {
        // Wrapping multiplication is the same for signed and unsigned integers.
        cast::<$UnsignedSimd, $Simd>(
          cast::<$Simd, $UnsignedSimd>(self) * cast::<$Simd, $UnsignedSimd>(rhs),
        )
      }

      #[inline]
      fn bitand(self, rhs: Self) -> Self::Output {
        cast::<$UnsignedSimd, $Simd>(
          cast::<$Simd, $UnsignedSimd>(self) & cast::<$Simd, $UnsignedSimd>(rhs),
        )
      }

      #[inline]
      fn bitor(self, rhs: Self) -> Self::Output {
        cast::<$UnsignedSimd, $Simd>(
          cast::<$Simd, $UnsignedSimd>(self) | cast::<$Simd, $UnsignedSimd>(rhs),
        )
      }

      #[inline]
      fn bitxor(self, rhs: Self) -> Self::Output {
        cast::<$UnsignedSimd, $Simd>(
          cast::<$Simd, $UnsignedSimd>(self) ^ cast::<$Simd, $UnsignedSimd>(rhs),
        )
      }

      #[inline]
      fn shl(self, rhs: $UnsignedSimd) -> Self {
        cast(cast::<$Simd, $UnsignedSimd>(self) << rhs)
      }

      #[inline]
      fn shl(self, rhs: u32) -> Self {
        cast(cast::<$Simd, $UnsignedSimd>(self) << rhs)
      }

      $fn_shr_unsigned_simd

      $fn_shr_u32

      $fn_max

      $fn_min

      $fn_reduce_max

      $fn_reduce_min

      #[inline]
      pub fn unbounded_shl(self, rhs: $UnsignedSimd) -> Self {
        // Shift left is the same for unsigned and signed integers.
        cast(cast::<$Simd, $UnsignedSimd>(self).unbounded_shl(rhs))
      }

      #[inline]
      pub fn unbounded_shl_scalar(self, rhs: u32) -> Self {
        // Shift left is the same for unsigned and signed integers.
        cast(cast::<$Simd, $UnsignedSimd>(self).unbounded_shl_scalar(rhs))
      }

      $fn_unbounded_shr

      $fn_unbounded_shr_scalar

      $fn_saturating_add

      $fn_saturating_sub

      #[inline]
      pub fn saturating_mul(self, rhs: Self) -> Self {
        let (result, overflow) = self.overflowing_mul(rhs);
        let limit = Self::MAX ^ (self ^ rhs).is_negative();
        overflow.select(limit, result)
      }

      #[inline]
      pub fn overflowing_add(self, rhs: Self) -> (Self, Self) {
        let result = self + rhs;
        let overflow = (!(self ^ rhs) & (self ^ result)).is_negative();

        (result, overflow)
      }

      #[inline]
      pub fn overflowing_sub(self, rhs: Self) -> (Self, Self) {
        let result = self - rhs;
        let overflow = ((self ^ rhs) & (self ^ result)).is_negative();

        (result, overflow)
      }

      $fn_overflowing_mul

      #[inline]
      pub fn overflowing_div(self, rhs: Self) -> (Self, Self) {
        // The second field is equivalent to
        // `self.simd_eq(Self::MIN) & rhs.simd_eq(-1)` but may be cheaper.
        (self / rhs, ((self ^ Self::MAX) & rhs).simd_eq(!Self::ZERO))
      }

      #[inline]
      pub fn overflowing_rem(self, rhs: Self) -> (Self, Self) {
        // The second field is equivalent to
        // `self.simd_eq(Self::MIN) & rhs.simd_eq(-1)` but may be cheaper.
        (self % rhs, ((self ^ Self::MAX) & rhs).simd_eq(!Self::ZERO))
      }

      optional_fn_widening_mul { $($fn_widening_mul)? }

      $fn_mul_keep_low_high

      $fn_mul_keep_high
    );

    impl Select<$Simd> for $UnsignedSimd {
      #[inline]
      fn select(self, if_true: $Simd, if_false: $Simd) -> $Simd {
        self.cast_signed().select(if_true, if_false)
      }
    }

    impl Select<$UnsignedSimd> for $Simd {
      #[inline]
      fn select(self, if_true: $UnsignedSimd, if_false: $UnsignedSimd) -> $UnsignedSimd {
        self.cast_unsigned().select(if_true, if_false)
      }
    }

    /// The following functionality exists for all SIMD vectors of signed
    /// integers.
    impl $Simd {
      /// Returns the bit patterns of `self` reinterpreted as unsigned integers
      /// of the same size.
      #[inline]
      #[must_use]
      pub const fn cast_unsigned(self) -> $UnsignedSimd {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$Simd, $UnsignedSimd>(self) }
      }

      /// Computes the absolute value of each input element, returned as an
      /// unsigned integer in order to avoid wrapping.
      #[inline]
      #[must_use]
      pub fn unsigned_abs(self) -> $UnsignedSimd {
        cast::<$Simd, $UnsignedSimd>(self.abs())
      }

      /// Returns the absolute value of each input element.
      #[must_use]
      $fn_abs

      /// Returns numbers representing the sign of each element.
      ///
      /// - `0` if the element is zero
      /// - `1` if the element is positive
      /// - `-1` if the element is negative
      #[inline]
      #[must_use]
      pub fn signum(self) -> Self {
        // Flip signs because the result for true in `is_positive/negative` is
        // `-1` (all bits set).
        self.is_negative() - self.is_positive()
      }

      /// Returns a [mask] that is true for each positive element, and false if
      /// it is zero or negative.
      ///
      /// [mask]: crate#masks
      #[must_use]
      $fn_is_positive

      /// Returns a [mask] that is true for each negative element, and false if
      /// it is zero or positive.
      ///
      /// [mask]: crate#masks
      #[must_use]
      $fn_is_negative
    }
  };
}
