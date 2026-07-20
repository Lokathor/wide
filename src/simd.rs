macro_rules! impl_simd {
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
      optional_type_x86_inner { $(X86Inner = $X86Inner:ident)? },
      optional_type_arm_inner { $(ArmInner = $ArmInner:ident)? },
      optional_type_wasm_inner { $(WasmInner = $WasmInner:ident)? },
    }

    $fn_simd_eq:item
    $fn_simd_ne:item
    $fn_simd_lt:item
    $fn_simd_gt:item
    $fn_simd_le:item
    $fn_simd_ge:item
    $fn_bitselect:item
    $fn_select:item
    $fn_to_bitmask:item
    $fn_any:item
    $fn_all:item
    $fn_transpose:item
  ) => {
    impl From<[$T; $N]> for $Simd {
      /// Converts an array to a SIMD vector.
      #[inline]
      fn from(arr: [$T; $N]) -> Self {
        Self::new(arr)
      }
    }

    impl From<$Simd> for [$T; $N] {
      /// Converts a SIMD vector to an array.
      #[inline]
      fn from(simd: $Simd) -> Self {
        simd.to_array()
      }
    }

    impl From<$T> for $Simd {
      /// Converts a single value to a SIMD vector by setting all elements to
      /// that value.
      #[inline]
      fn from(elem: $T) -> Self {
        Self::splat(elem)
      }
    }

    impl From<&[$T]> for $Simd {
      /// Converts a slice to a SIMD vector, filling in zeros if there are not
      /// enough elements, and panicking if there are too many elements.
      ///
      /// Note that in the future, handling of too many elements may change.
      #[inline]
      fn from(value: &[$T]) -> Self {
        assert!(
          value.len() <= $N,
          concat!(
            "slice has more elements than `",
            stringify!($Simd),
            "` can store",
          ),
        );

        // SAFETY: `$Simd` accepts all bit-patterns, including all zeros.
        let mut result = unsafe { core::mem::zeroed::<$Simd>() };

        // SAFETY: `value` is valid for its own length, and `result` is valid
        // because its length is checked to be less than or equal to
        // `value.len()`. Both pointers are properly aligned because they
        // originate from a slice of `$T`. The regions of memory do not overlap
        // because they originate from a shared reference and a mutable
        // reference.
        unsafe {
          core::ptr::copy_nonoverlapping::<$T>(
            value.as_ptr(),
            result.as_mut_array().as_mut_ptr(),
            value.len()
          );
        }

        result
      }
    }

    $(
      #[cfg(target_arch = "x86")]
      impl From<core::arch::x86::$X86Inner> for $Simd {
        /// Converts a native intrinsics SIMD type to a high-level SIMD type.
        #[inline]
        fn from(value: core::arch::x86::$X86Inner) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<core::arch::x86::$X86Inner, $Simd>(value) }
        }
      }

      #[cfg(target_arch = "x86")]
      impl From<$Simd> for core::arch::x86::$X86Inner {
        /// Converts a high-level SIMD type to a native intrinsics SIMD type.
        #[inline]
        fn from(value: $Simd) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<$Simd, core::arch::x86::$X86Inner>(value) }
        }
      }

      #[cfg(target_arch = "x86_64")]
      impl From<core::arch::x86_64::$X86Inner> for $Simd {
        /// Converts a native intrinsics SIMD type to a high-level SIMD type.
        #[inline]
        fn from(value: core::arch::x86_64::$X86Inner) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<core::arch::x86_64::$X86Inner, $Simd>(value) }
        }
      }

      #[cfg(target_arch = "x86_64")]
      impl From<$Simd> for core::arch::x86_64::$X86Inner {
        /// Converts a high-level SIMD type to a native intrinsics SIMD type.
        #[inline]
        fn from(value: $Simd) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<$Simd, core::arch::x86_64::$X86Inner>(value) }
        }
      }
    )?
    $(
      #[cfg(target_arch = "aarch64")]
      impl From<core::arch::aarch64::$ArmInner> for $Simd {
        /// Converts a native intrinsics SIMD type to a high-level SIMD type.
        #[inline]
        fn from(value: core::arch::aarch64::$ArmInner) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<core::arch::aarch64::$ArmInner, $Simd>(value) }
        }
      }

      #[cfg(target_arch = "aarch64")]
      impl From<$Simd> for core::arch::aarch64::$ArmInner {
        /// Converts a high-level SIMD type to a native intrinsics SIMD type.
        #[inline]
        fn from(value: $Simd) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<$Simd, core::arch::aarch64::$ArmInner>(value) }
        }
      }
    )?
    $(
      #[cfg(target_arch = "wasm32")]
      impl From<core::arch::wasm32::$WasmInner> for $Simd {
        /// Converts a native intrinsics SIMD type to a high-level SIMD type.
        #[inline]
        fn from(value: core::arch::wasm32::$WasmInner) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<core::arch::wasm32::$WasmInner, $Simd>(value) }
        }
      }

      #[cfg(target_arch = "wasm32")]
      impl From<$Simd> for core::arch::wasm32::$WasmInner {
        /// Converts a high-level SIMD type to a native intrinsics SIMD type.
        #[inline]
        fn from(value: $Simd) -> Self {
          // SAFETY: Both types are expected to accept all bit-patterns and to
          // only contain initialized memory.
          unsafe { core::mem::transmute::<$Simd, core::arch::wasm32::$WasmInner>(value) }
        }
      }
    )?

    macro_rules! impl_formatting_trait {
      ($Trait:path) => {
        impl $Trait for $Simd {
          #[allow(clippy::missing_inline_in_public_items)]
          fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            write!(f, "(")?;
            for (i, x) in self.to_array().iter().enumerate() {
              if i > 0 {
                write!(f, ", ")?;
              }
              <$T as $Trait>::fmt(x, f)?;
            }
            write!(f, ")")
          }
        }
      }
    }
    impl_formatting_trait!(core::fmt::Debug);
    impl_formatting_trait!(core::fmt::Display);
    impl_formatting_trait!(core::fmt::LowerExp);
    impl_formatting_trait!(core::fmt::UpperExp);

    #[expect(deprecated)]
    impl CmpEq for $Simd {
      type Output = Self;

      $fn_simd_eq
    }

    #[expect(deprecated)]
    impl CmpEq<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_eq(self, rhs: $T) -> Self::Output {
        self.simd_eq(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpNe for $Simd {
      type Output = Self;

      $fn_simd_ne
    }

    #[expect(deprecated)]
    impl CmpNe<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_ne(self, rhs: $T) -> Self::Output {
        self.simd_ne(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpLt for $Simd {
      type Output = Self;

      $fn_simd_lt
    }

    #[expect(deprecated)]
    impl CmpLt<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_lt(self, rhs: $T) -> Self::Output {
        self.simd_lt(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpGt for $Simd {
      type Output = Self;

      $fn_simd_gt
    }

    #[expect(deprecated)]
    impl CmpGt<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_gt(self, rhs: $T) -> Self::Output {
        self.simd_gt(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpLe for $Simd {
      type Output = Self;

      $fn_simd_le
    }

    #[expect(deprecated)]
    impl CmpLe<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_le(self, rhs: $T) -> Self::Output {
        self.simd_le(Self::splat(rhs))
      }
    }

    #[expect(deprecated)]
    impl CmpGe for $Simd {
      type Output = Self;

      $fn_simd_ge
    }

    #[expect(deprecated)]
    impl CmpGe<$T> for $Simd {
      type Output = Self;

      #[inline]
      fn simd_ge(self, rhs: $T) -> Self::Output {
        self.simd_ge(Self::splat(rhs))
      }
    }

    impl AlignTo for $Simd {
      type Elem = $T;
    }

    /// The following functionality exists for all SIMD vectors.
    impl $Simd {
      /// Converts an array to a SIMD vector.
      #[inline]
      #[must_use]
      pub const fn new(array: [$T; $N]) -> Self {
        unsafe { core::mem::transmute::<[$T; $N], $Simd>(array) }
      }

      /// Constructs a new SIMD vector with all elements set to the given value.
      #[inline]
      #[must_use]
      pub const fn splat(elem: $T) -> Self {
        unsafe { core::mem::transmute::<[$T; $N], $Simd>([elem; $N]) }
      }

      /// Converts a SIMD vector to an array.
      #[inline]
      #[must_use]
      pub const fn to_array(self) -> [$T; $N] {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<$Simd, [$T; $N]>(self) }
      }

      /// Returns an array reference containing the entire SIMD vector.
      #[inline]
      #[must_use]
      pub const fn as_array(&self) -> &[$T; $N] {
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { core::mem::transmute::<&$Simd, &[$T; $N]>(self) }
      }

      /// Returns a mutable array reference containing the entire SIMD vector.
      #[inline]
      #[must_use]
      pub const fn as_mut_array(&mut self) -> &mut [$T; $N] {
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { core::mem::transmute::<&mut $Simd, &mut [$T; $N]>(self) }
      }

      /// Returns a [mask] that checks if each element of `self` is equal to the
      /// corresponding element of `other`.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[expect(deprecated)]
      pub fn simd_eq<Rhs>(self, other: Rhs) -> <Self as CmpEq<Rhs>>::Output
      where
        Self: CmpEq<Rhs>,
      {
        CmpEq::simd_eq(self, other)
      }

      /// Returns a [mask] that checks if each element of `self` is not equal to
      /// the corresponding element of `other`.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[expect(deprecated)]
      pub fn simd_ne<Rhs>(self, other: Rhs) -> <Self as CmpNe<Rhs>>::Output
      where
        Self: CmpNe<Rhs>,
      {
        CmpNe::simd_ne(self, other)
      }

      /// Returns a [mask] that checks if each element of `self` is less than
      /// the corresponding element of `other`.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[expect(deprecated)]
      pub fn simd_lt<Rhs>(self, other: Rhs) -> <Self as CmpLt<Rhs>>::Output
      where
        Self: CmpLt<Rhs>,
      {
        CmpLt::simd_lt(self, other)
      }

      /// Returns a [mask] that checks if each element of `self` is greater than
      /// the corresponding element of `other`.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[expect(deprecated)]
      pub fn simd_gt<Rhs>(self, other: Rhs) -> <Self as CmpGt<Rhs>>::Output
      where
        Self: CmpGt<Rhs>,
      {
        CmpGt::simd_gt(self, other)
      }

      /// Returns a [mask] that checks if each element of `self` is less than or
      /// equal to the corresponding element of `other`.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[expect(deprecated)]
      pub fn simd_le<Rhs>(self, other: Rhs) -> <Self as CmpLe<Rhs>>::Output
      where
        Self: CmpLe<Rhs>,
      {
        CmpLe::simd_le(self, other)
      }

      /// Returns a [mask] that checks if each element of `self` is greater than
      /// or equal to the corresponding element of `other`.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[expect(deprecated)]
      pub fn simd_ge<Rhs>(self, other: Rhs) -> <Self as CmpGe<Rhs>>::Output
      where
        Self: CmpGe<Rhs>,
      {
        CmpGe::simd_ge(self, other)
      }

      /// Bitwise selection.
      ///
      /// For each bit of `self`:
      ///
      /// - If the bit is one, return the corresponding bit of `if_one`
      /// - If the bit is zero, return the corresponding bit of `if_zero`
      ///
      /// If you know `self` is a [mask], meaning each element is either all
      /// zeros or all ones, consider using [`select`] which is faster.
      ///
      /// [mask]: crate#masks
      /// [`select`]: Self::select
      #[must_use]
      $fn_bitselect

      /// Elementwise selection.
      ///
      /// For each element of `self`:
      ///
      /// - If all bits are one, return the corresponding element of `if_true`
      /// - If all bits are zero, return the corresponding element of `if_false`
      ///
      /// This function assumes `self` is a [mask], meaning each element is
      /// either all zeros or all ones. For bitwise selection use [`bitselect`].
      ///
      /// [mask]: crate#masks
      /// [`bitselect`]: Self::bitselect
      #[must_use]
      $fn_select

      /// Converts to a bitmask, where each bit is `1` if the element of `self`
      /// is true or `0` if the element of `self` is false.
      ///
      /// Each bit of the output corresponds to an element of `self`. The least
      /// significant bit corresponds to the lowest element. Remaining bits are
      /// `0`.
      ///
      /// This function assumes `self` is a [mask], meaning each element is
      /// either all zeros or all ones. If the input is not a mask, the result
      /// is unspecified.
      ///
      /// [mask]: crate#masks
      #[must_use]
      #[doc(alias("movemask", "move_mask"))]
      $fn_to_bitmask

      /// Returns `true` if any element of `self` is true.
      ///
      /// This function assumes `self` is a [mask], meaning each element is
      /// either all zeros or all ones. If the input is not a mask, the result
      /// is unspecified.
      ///
      /// [mask]: crate#masks
      #[must_use]
      $fn_any

      /// Returns `true` if all elements of `self` are true.
      ///
      /// This function assumes `self` is a [mask], meaning each element is
      /// either all zeros or all ones. If the input is not a mask, the result
      /// is unspecified.
      ///
      /// [mask]: crate#masks
      #[must_use]
      $fn_all

      /// Returns `true` if none of the elements of `self` are true.
      ///
      /// This function assumes `self` is a [mask], meaning each element is
      /// either all zeros or all ones. If the input is not a mask, the result
      /// is unspecified.
      ///
      /// [mask]: crate#masks
      #[inline]
      #[must_use]
      pub fn none(self) -> bool {
        !self.any()
      }

      /// Transposes an array of SIMD vectors interpreted as a square matrix.
      #[must_use]
      $fn_transpose

      /// Elementwise selection.
      ///
      /// For each element of `self`:
      ///
      /// - If all bits are one, return the corresponding element of `if_true`
      /// - If all bits are zero, return the corresponding element of `if_false`
      ///
      /// Originally this function did not specify whether it supported per-bit
      /// selection, or if it assumed `self` is a [mask], meaning each element
      /// is either all zeros or all ones (for better performance). Because of
      /// this, [`blend`] has been split into two new functions: [`select`] and
      /// [`bitselect`].
      ///
      /// [mask]: crate#masks
      /// [`blend`]: Self::blend
      /// [`select`]: Self::select
      /// [`bitselect`]: Self::bitselect
      #[deprecated(
        since = "1.6.0",
        note = "split into `select` and `bitselect` functions"
      )]
      #[inline]
      #[must_use]
      pub fn blend(self, if_true: Self, if_false: Self) -> Self {
        self.select(if_true, if_false)
      }
    }

    mod bytemuck {
      use bytemuck::{Pod, Zeroable};

      use crate::$Simd;

      // SAFETY: All SIMD types in this library contain fully initialized memory
      // and accept all bits patterns.
      unsafe impl Zeroable for $Simd {}

      // SAFETY: All SIMD types in this library contain fully initialized memory
      // and accept all bits patterns.
      unsafe impl Pod for $Simd {}
    }

    #[cfg(feature = "serde")]
    mod serde {
      use serde_core::{Deserialize, Serialize, ser::SerializeTuple};

      use crate::$Simd;

      impl Serialize for $Simd {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
          S: serde_core::Serializer,
        {
          let array = self.as_array();
          let mut seq = serializer.serialize_tuple($N)?;
          for e in array {
            seq.serialize_element(e)?;
          }
          seq.end()
        }
      }

      impl<'de> Deserialize<'de> for $Simd {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
          D: serde_core::Deserializer<'de>,
        {
          Ok(<[$T; $N]>::deserialize(deserializer)?.into())
        }
      }
    }
  };
}

macro_rules! impl_unary_operator {
  ($Simd:ident, $Op:ident, $op:ident, $impl:item $(, $(#[$doc:meta])*)?) => {
    impl $Op for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      $impl
    }

    impl $Op for &$Simd {
      type Output = $Simd;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self) -> Self::Output {
        (*self).$op()
      }
    }
  }
}

macro_rules! impl_binary_operator {
  (
    $T:ident,
    $Simd:ident,
    $Op:ident,
    $op:ident,
    $OpAssign:ident,
    $op_assign:ident,
    $impl:item
    $(,
      $(#[$doc:meta])*,
      $(#[$doc_scalar:meta])*,
      $(#[$scalar_doc:meta])*
    )?
  ) => {
    impl $Op for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      $impl
    }

    impl $Op<$T> for $Simd {
      type Output = Self;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: $T) -> Self::Output {
        self.$op(Self::splat(rhs))
      }
    }

    impl $Op<$Simd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        $Simd::splat(self).$op(rhs)
      }
    }

    impl $OpAssign for $Simd {
      $($(#[$doc])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: Self) {
        *self = (*self).$op(rhs);
      }
    }

    impl $OpAssign<$T> for $Simd {
      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: $T) {
        *self = (*self).$op(Self::splat(rhs));
      }
    }

    impl $Op<&Self> for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: &Self) -> Self::Output {
        self.$op(*rhs)
      }
    }

    impl $Op<&$T> for $Simd {
      type Output = Self;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: &$T) -> Self::Output {
        self.$op(Self::splat(*rhs))
      }
    }

    impl $Op<&$Simd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        $Simd::splat(self).$op(*rhs)
      }
    }

    impl $OpAssign<&Self> for $Simd {
      $($(#[$doc])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: &Self) {
        *self = (*self).$op(*rhs);
      }
    }

    impl $OpAssign<&$T> for $Simd {
      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op_assign(&mut self, rhs: &$T) {
        *self = (*self).$op(Self::splat(*rhs));
      }
    }

    impl $Op<$Simd> for &$Simd {
      type Output = $Simd;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        (*self).$op(rhs)
      }
    }

    impl $Op<$T> for &$Simd {
      type Output = $Simd;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: $T) -> Self::Output {
        (*self).$op($Simd::splat(rhs))
      }
    }

    impl $Op<$Simd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        $Simd::splat(*self).$op(rhs)
      }
    }

    impl $Op<&$Simd> for &$Simd {
      type Output = $Simd;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        (*self).$op(*rhs)
      }
    }

    impl $Op<&$T> for &$Simd {
      type Output = $Simd;

      $($(#[$doc_scalar])*)?
      #[inline]
      fn $op(self, rhs: &$T) -> Self::Output {
        (*self).$op($Simd::splat(*rhs))
      }
    }

    impl $Op<&$Simd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        $Simd::splat(*self).$op(*rhs)
      }
    }
  }
}

macro_rules! impl_shift_operator {
  (
    $T:ident,
    $Simd:ident,
    $UnsignedSimd:ident,
    $SignedSimd:ident,
    $Op:ident,
    $op:ident,
    $OpAssign:ident,
    $op_assign:ident,
    $impl_unsigned_simd:item,
    $impl_u32:item
    $(,
      $(#[$doc:meta])*,
      $(#[$doc_scalar:meta])*,
      $(#[$scalar_doc:meta])*
    )?
  ) => {
    impl $Op<$UnsignedSimd> for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      $impl_unsigned_simd
    }

    impl $Op<$SignedSimd> for $Simd {
      type Output = Self;

      $($(#[$doc])*)?
      #[inline]
      fn $op(self, rhs: $SignedSimd) -> Self::Output {
        self.$op(cast::<$SignedSimd, $UnsignedSimd>(rhs))
      }
    }

    impl $Op<$UnsignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $UnsignedSimd) -> Self::Output {
        $Simd::splat(self).$op(rhs)
      }
    }

    impl $Op<$SignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $SignedSimd) -> Self::Output {
        $Simd::splat(self).$op(rhs)
      }
    }

    impl<Rhs> $OpAssign<Rhs> for $Simd
    where
      Self: $Op<Rhs, Output = Self>,
    {
      #[inline]
      fn $op_assign(&mut self, rhs: Rhs) {
        *self = (*self).$op(rhs);
      }
    }

    impl<Rhs> $Op<&Rhs> for $Simd
    where
      Self: $Op<Rhs, Output = Self>,
      Rhs: Copy,
    {
      type Output = Self;

      #[inline]
      fn $op(self, rhs: &Rhs) -> Self::Output {
        self.$op(*rhs)
      }
    }

    impl $Op<&$UnsignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$UnsignedSimd) -> Self::Output {
        $Simd::splat(self).$op(*rhs)
      }
    }

    impl $Op<&$SignedSimd> for $T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$SignedSimd) -> Self::Output {
        $Simd::splat(self).$op(*rhs)
      }
    }

    impl<Rhs> $Op<Rhs> for &$Simd
    where
      $Simd: $Op<Rhs, Output = $Simd>,
    {
      type Output = $Simd;

      #[inline]
      fn $op(self, rhs: Rhs) -> Self::Output {
        (*self).$op(rhs)
      }
    }

    impl $Op<$UnsignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $UnsignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(rhs)
      }
    }

    impl $Op<$SignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: $SignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(rhs)
      }
    }

    impl $Op<&$UnsignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$UnsignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(*rhs)
      }
    }

    impl $Op<&$SignedSimd> for &$T {
      type Output = $Simd;

      $($(#[$scalar_doc])*)?
      #[inline]
      fn $op(self, rhs: &$SignedSimd) -> Self::Output {
        $Simd::splat(*self).$op(*rhs)
      }
    }

    impl $Op<u32> for $Simd {
      type Output = Self;

      $($(#[$doc_scalar])*)?
      $impl_u32
    }

    macro_rules! impl_scalar_with_cast {
      ($T2:ident) => {
        impl $Op<$T2> for $Simd {
          type Output = Self;

          $($(#[$doc_scalar])*)?
          #[inline]
          fn $op(self, rhs: $T2) -> Self::Output {
            self.$op(rhs as u32)
          }
        }
      }
    }
    impl_scalar_with_cast!(i8);
    impl_scalar_with_cast!(i16);
    impl_scalar_with_cast!(i32);
    impl_scalar_with_cast!(i64);
    impl_scalar_with_cast!(i128);
    impl_scalar_with_cast!(isize);
    impl_scalar_with_cast!(u8);
    impl_scalar_with_cast!(u16);
    impl_scalar_with_cast!(u64);
    impl_scalar_with_cast!(u128);
    impl_scalar_with_cast!(usize);
  }
}
