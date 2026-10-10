/// Emits functionality shared by all SIMD unsigned-integer types.
///
/// Functions that need a separate implementation for each type (for
/// performance) use `$fn_{name}:item` syntax, and functions that have one
/// shared implementation for all uints are written out normally inside this
/// macro.
///
/// This macro also invokes `impl_simd`.
macro_rules! impl_simd_unsigned {
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
      IntSimd = $IntSimd:ident,
      T_BITS = $T_BITS:literal,
      T_BITS_MUL_2 = $T_BITS_MUL_2:literal,
      [$($index:literal),* $(,)?],
      optional_type_x86_inner { $(X86Inner = $X86Inner:ident)? },
      optional_type_arm_inner { $(ArmInner = $ArmInner:ident)? },
      optional_type_wasm_inner { $(WasmInner = $WasmInner:ident)? },
    }

    // General SIMD functions
    $fn_not:item
    $fn_add:item
    $fn_sub:item
    $fn_mul:item
    $fn_bitand:item
    $fn_bitor:item
    $fn_bitxor:item
    $fn_simd_eq:item
    $fn_simd_ne:item
    $fn_simd_lt:item
    $fn_simd_gt:item
    $fn_simd_le:item
    $fn_simd_ge:item
    $fn_replace_const:item
    $fn_extract_const:item
    $fn_reduce_add:item
    $fn_reduce_mul:item
    $fn_bitselect:item
    $fn_select:item
    $fn_to_bitmask:item
    $fn_any:item
    $fn_all:item
    $fn_unpack_lo:item
    $fn_unpack_hi:item
    $fn_shuffle:item
    $fn_shuffle_zeroing:item
    $fn_shuffle_wrapping:item
    $fn_shuffle_2:item
    $fn_shuffle_zeroing_2:item
    $fn_shuffle_wrapping_2:item
    $fn_shuffle_3:item
    $fn_shuffle_zeroing_3:item
    $fn_shuffle_wrapping_3:item
    $fn_shuffle_4:item
    $fn_shuffle_zeroing_4:item
    $fn_shuffle_wrapping_4:item
    $fn_transpose:item

    // Uint-specific functions
    $fn_shl_unsigned_simd:item
    $fn_shl_u32:item
    $fn_shr_unsigned_simd:item
    $fn_shr_u32:item
    $fn_max:item
    $fn_min:item
    $fn_reduce_max:item
    $fn_reduce_min:item
    $fn_unbounded_shl:item
    $fn_unbounded_shl_scalar:item
    $fn_unbounded_shr:item
    $fn_unbounded_shr_scalar:item
    $fn_saturating_add:item
    $fn_saturating_sub:item
    $fn_overflowing_mul:item
    optional_fn_widening_mul { $($fn_widening_mul:item)? }
    $fn_mul_keep_low_high:item
    $fn_mul_keep_high:item
  ) => {
    impl_simd!(
      unsafe {
        T = $T,
        N = $N,
        Simd = $Simd,
        UintSimd = $Simd,
        optional_type_x86_inner { $(X86Inner = $X86Inner)? },
        optional_type_arm_inner { $(ArmInner = $ArmInner)? },
        optional_type_wasm_inner { $(WasmInner = $WasmInner)? },
      }

      $fn_simd_eq

      $fn_simd_ne

      $fn_simd_lt

      $fn_simd_gt

      $fn_simd_le

      $fn_simd_ge

      $fn_replace_const

      $fn_extract_const

      $fn_reduce_add

      $fn_reduce_mul

      $fn_bitselect

      $fn_select

      $fn_to_bitmask

      $fn_any

      $fn_all

      $fn_unpack_lo

      $fn_unpack_hi

      $fn_shuffle

      $fn_shuffle_zeroing

      $fn_shuffle_wrapping

      $fn_shuffle_2

      $fn_shuffle_zeroing_2

      $fn_shuffle_wrapping_2

      $fn_shuffle_3

      $fn_shuffle_zeroing_3

      $fn_shuffle_wrapping_3

      $fn_shuffle_4

      $fn_shuffle_zeroing_4

      $fn_shuffle_wrapping_4

      $fn_transpose
    );

    impl_simd_integer!(
      unsafe {
        T = $T,
        N = $N,
        Simd = $Simd,
        UnsignedSimd = $Simd,
        SignedSimd = $IntSimd,
        T_BITS = $T_BITS,
        T_BITS_MUL_2 = $T_BITS_MUL_2,
        [$($index),*],
      }

      $fn_not

      $fn_add

      $fn_sub

      $fn_mul

      $fn_bitand

      $fn_bitor

      $fn_bitxor

      $fn_shl_unsigned_simd

      $fn_shl_u32

      $fn_shr_unsigned_simd

      $fn_shr_u32

      $fn_max

      $fn_min

      $fn_reduce_max

      $fn_reduce_min

      $fn_unbounded_shl

      $fn_unbounded_shl_scalar

      $fn_unbounded_shr

      $fn_unbounded_shr_scalar

      $fn_saturating_add

      $fn_saturating_sub

      #[inline]
      pub fn saturating_mul(self, rhs: Self) -> Self {
        let (low, high) = self.mul_keep_low_high(rhs);
        low | high.simd_ne(Self::ZERO)
      }

      #[inline]
      pub fn overflowing_add(self, rhs: Self) -> (Self, Self) {
        let result = self + rhs;
        let overflow = result.simd_lt(self);

        (result, overflow)
      }

      #[inline]
      pub fn overflowing_sub(self, rhs: Self) -> (Self, Self) {
        let result = self - rhs;
        let overflow = result.simd_gt(self);

        (result, overflow)
      }

      $fn_overflowing_mul

      #[inline]
      pub fn overflowing_div(self, rhs: Self) -> (Self, Self) {
        (self / rhs, Self::ZERO)
      }

      #[inline]
      pub fn overflowing_rem(self, rhs: Self) -> (Self, Self) {
        (self % rhs, Self::ZERO)
      }

      optional_fn_widening_mul { $($fn_widening_mul)? }

      $fn_mul_keep_low_high

      $fn_mul_keep_high
    );

    /// The following functionality exists for all SIMD vectors of unsigned
    /// integers.
    impl $Simd {
      /// Returns the bit patterns of `self` reinterpreted as signed integers of
      /// the same size.
      #[inline]
      #[must_use]
      pub const fn cast_signed(self) -> $IntSimd {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$Simd, $IntSimd>(self) }
      }
    }
  };
}
