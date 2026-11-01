use core::ops::{BitAnd, BitXor};

use bytemuck::cast;

/// Modifies an `impl` block such that calling functions with target features
/// that are guaranteed at compile time does not require an `unsafe` block.
///
/// This macro is only needed due to a language limitation. If language support
/// for this pattern gets stabilized, this macro can be removed.
///
/// This macro disables formatting, since rustfmt does not understand its
/// syntax. If declarative attribute macros get stabilized, this macro can use
/// attribute syntax which will reenable formatting.
///
/// Each target feature must be explicitly listed inside this macro's
/// implementation in order to be supported. If a target feature does not work,
/// it may be missing from this macro's list.
macro_rules! safe_target_feature {
    // In order to support both inherent and trait implementations this must use
    // redirection
    (impl $Self:ident $body:tt) => {
        impl $Self {
            safe_target_feature! { Self = $Self, $body }
        }
    };
    (impl $Trait:ident for $Self:ident $body:tt) => {
        impl $Trait for $Self {
            safe_target_feature! { Self = $Self, $body }
        }
    };

    (
        Self = $Self:ty,
        {
            // `impl` blocks that use `safe_target_feature` may contain macro
            // calls at the start, so accept them and re-emit them
            $($macro_name:ident!$macro_args:tt)*

            $(
                // Match a function item
                $(#[$meta:meta])+
                $vis:vis
                $($keyword_or_name:ident)+ // This trick enables support for keywords before `fn`
                $(<$(const $CONST_PARAM:ident: $ConstParamTy:ty),* $(,)?>)?
                ($($param:ident $(: $ParamTy:ty)?),* $(,)?) // The type is optional so that `self` works
                $(-> $OutputTy:ty)?
                $body:block
            )*
        }
    ) => {
        $($macro_name!$macro_args)*

        $(
            // Paste the original function signature
            $(#[$meta])+
            $vis $($keyword_or_name)+
            $(<$(const $CONST_PARAM: $ConstParamTy),*>)?
            ($($param $(: $ParamTy)?),*)
            $(-> $OutputTy)?
            // Paste a modified body
            {
                // Internally paste the original function. This trait is used so
                // that the function is not exposed outside of this block.
                trait OriginalFn: Sized {
                    #[track_caller]
                    unsafe fn original_fn
                    $(<$(const $CONST_PARAM: $ConstParamTy),*>)?
                    ($($param $(: $ParamTy)?),*)
                    $(-> $OutputTy)?;
                }

                impl OriginalFn for $Self {
                    // Enable target features that are guaranteed at compile time
                    #[cfg_attr(target_feature = "avx", target_feature(enable = "avx"))]
                    #[cfg_attr(target_feature = "avx2", target_feature(enable = "avx2"))]
                    #[cfg_attr(target_feature = "avx512bw", target_feature(enable = "avx512bw"))]
                    #[cfg_attr(target_feature = "avx512dq", target_feature(enable = "avx512dq"))]
                    #[cfg_attr(target_feature = "avx512f", target_feature(enable = "avx512f"))]
                    #[cfg_attr(target_feature = "avx512ifma", target_feature(enable = "avx512ifma"))]
                    #[cfg_attr(target_feature = "avx512vbmi", target_feature(enable = "avx512vbmi"))]
                    #[cfg_attr(target_feature = "avx512vl", target_feature(enable = "avx512vl"))]
                    #[cfg_attr(target_feature = "fma", target_feature(enable = "fma"))]
                    #[cfg_attr(target_feature = "neon", target_feature(enable = "neon"))]
                    #[cfg_attr(target_feature = "sse", target_feature(enable = "sse"))]
                    #[cfg_attr(target_feature = "sse2", target_feature(enable = "sse2"))]
                    #[cfg_attr(target_feature = "sse3", target_feature(enable = "sse3"))]
                    #[cfg_attr(target_feature = "sse4.1", target_feature(enable = "sse4.1"))]
                    #[cfg_attr(target_feature = "sse4.2", target_feature(enable = "sse4.2"))]
                    #[cfg_attr(target_feature = "ssse3", target_feature(enable = "ssse3"))]
                    #[inline]
                    unsafe fn original_fn
                    $(<$(const $CONST_PARAM: $ConstParamTy),*>)?
                    ($($param $(: $ParamTy)?),*)
                    $(-> $OutputTy)?
                    $body
                }

                // SAFETY: `original_fn` only enables target features that are
                // guaranteed at compile time via cfg.
                unsafe { <Self as OriginalFn>::original_fn$(::<$($CONST_PARAM),*>)?($($param),*) }
            }
        )*
    };
}

macro_rules! pick {
  ($(if #[cfg($($test:meta),*)] {
      $($if_tokens:tt)*
    })else+ else {
      $($else_tokens:tt)*
    }) => {
    pick!{
      @__forests [ ] ;
      $( [ {$($test),*} {$($if_tokens)*} ], )*
      [ { } {$($else_tokens)*} ],
    }
  };
  (if #[cfg($($if_meta:meta),*)] {
      $($if_tokens:tt)*
    } $(else if #[cfg($($else_meta:meta),*)] {
      $($else_tokens:tt)*
    })*) => {
    pick!{
      @__forests [ ] ;
      [ {$($if_meta),*} {$($if_tokens)*} ],
      $( [ {$($else_meta),*} {$($else_tokens)*} ], )*
    }
  };
  (@__forests [$($not:meta,)*];) => {
    /* halt expansion */
  };
  (@__forests [$($not:meta,)*]; [{$($m:meta),*} {$($tokens:tt)*}], $($rest:tt)*) => {
    #[cfg(all( $($m,)* not(any($($not),*)) ))]
    pick!{ @__identity $($tokens)* }
    pick!{ @__forests [ $($not,)* $($m,)* ] ; $($rest)* }
  };
  (@__identity $($tokens:tt)*) => {
    $($tokens)*
  };
}

macro_rules! impl_unary_operator {
  ($Simd:ident, $Op:ident, $op:ident, $op_backend:ident, $(#[$doc:meta])*) => {
    impl $Op for $Simd {
      type Output = Self;

      $(#[$doc])*
      #[inline]
      fn $op(self) -> Self::Output {
        self.$op_backend()
      }
    }

    impl $Op for &$Simd {
      type Output = $Simd;

      $(#[$doc])*
      #[inline]
      fn $op(self) -> Self::Output {
        self.$op_backend()
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
    $op_backend:ident,
    $(#[$doc:meta])*,
    $(#[$doc_scalar:meta])*,
    $(#[$scalar_doc:meta])*
  ) => {
    impl $Op for $Simd {
      type Output = Self;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: Self) -> Self::Output {
        self.$op_backend(rhs)
      }
    }

    impl $Op<$T> for $Simd {
      type Output = Self;

      $(#[$doc_scalar])*
      #[inline]
      fn $op(self, rhs: $T) -> Self::Output {
        self.$op_backend(Self::splat(rhs))
      }
    }

    impl $Op<$Simd> for $T {
      type Output = $Simd;

      $(#[$scalar_doc])*
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        $Simd::splat(self).$op_backend(rhs)
      }
    }

    impl $OpAssign for $Simd {
      $(#[$doc])*
      #[inline]
      fn $op_assign(&mut self, rhs: Self) {
        *self = self.$op_backend(rhs);
      }
    }

    impl $OpAssign<$T> for $Simd {
      $(#[$doc_scalar])*
      #[inline]
      fn $op_assign(&mut self, rhs: $T) {
        *self = self.$op_backend(Self::splat(rhs));
      }
    }

    impl $Op<&Self> for $Simd {
      type Output = Self;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: &Self) -> Self::Output {
        self.$op_backend(*rhs)
      }
    }

    impl $Op<&$T> for $Simd {
      type Output = Self;

      $(#[$doc_scalar])*
      #[inline]
      fn $op(self, rhs: &$T) -> Self::Output {
        self.$op_backend(Self::splat(*rhs))
      }
    }

    impl $Op<&$Simd> for $T {
      type Output = $Simd;

      $(#[$scalar_doc])*
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        $Simd::splat(self).$op_backend(*rhs)
      }
    }

    impl $OpAssign<&Self> for $Simd {
      $(#[$doc])*
      #[inline]
      fn $op_assign(&mut self, rhs: &Self) {
        *self = (*self).$op_backend(*rhs);
      }
    }

    impl $OpAssign<&$T> for $Simd {
      $(#[$doc_scalar])*
      #[inline]
      fn $op_assign(&mut self, rhs: &$T) {
        *self = self.$op_backend(Self::splat(*rhs));
      }
    }

    impl $Op<$Simd> for &$Simd {
      type Output = $Simd;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        (*self).$op_backend(rhs)
      }
    }

    impl $Op<$T> for &$Simd {
      type Output = $Simd;

      $(#[$doc_scalar])*
      #[inline]
      fn $op(self, rhs: $T) -> Self::Output {
        self.$op_backend($Simd::splat(rhs))
      }
    }

    impl $Op<$Simd> for &$T {
      type Output = $Simd;

      $(#[$scalar_doc])*
      #[inline]
      fn $op(self, rhs: $Simd) -> Self::Output {
        $Simd::splat(*self).$op_backend(rhs)
      }
    }

    impl $Op<&$Simd> for &$Simd {
      type Output = $Simd;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        self.$op_backend(*rhs)
      }
    }

    impl $Op<&$T> for &$Simd {
      type Output = $Simd;

      $(#[$doc_scalar])*
      #[inline]
      fn $op(self, rhs: &$T) -> Self::Output {
        self.$op_backend($Simd::splat(*rhs))
      }
    }

    impl $Op<&$Simd> for &$T {
      type Output = $Simd;

      $(#[$scalar_doc])*
      #[inline]
      fn $op(self, rhs: &$Simd) -> Self::Output {
        $Simd::splat(*self).$op_backend(*rhs)
      }
    }
  }
}

macro_rules! impl_shift_operator {
  (
    T: $ElementTrait:ident,
    $Op:ident,
    $op:ident,
    $OpAssign:ident,
    $op_assign:ident,
    $op_backend:ident,
    $op_scalar_backend:ident,
    $(#[$doc:meta])*,
    $(#[$doc_scalar:meta])*,
    $(#[$scalar_doc:meta])*
  ) => {
    impl<T, const N: usize, Rhs> $Op<Simd<Rhs, N>> for Simd<T, N>
    where
        T: $ElementTrait,
        Simd<T, N>: SupportedSimd,
        Simd<T::Unsigned, N>: SupportedSimd,
        Rhs: PrimitiveInteger<Unsigned = T::Unsigned>,
        Simd<Rhs, N>: SupportedSimd,
    {
      type Output = Self;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: Simd<Rhs, N>) -> Self::Output {
        // SAFETY: `Rhs` and `Rhs::Unsigned` are equallity-sized integers. Thus
        // it is sound to transmute between the two.
        let rhs = unsafe {
            core::mem::transmute_copy::<Simd<Rhs, N>, Simd<Rhs::Unsigned, N>>(&rhs)
        };

        self.$op_backend(rhs)
      }
    }

    impl<T, const N: usize, Rhs> $OpAssign<Simd<Rhs, N>> for Simd<T, N>
    where
        T: $ElementTrait,
        Simd<T, N>: SupportedSimd,
        Simd<T::Unsigned, N>: SupportedSimd,
        Rhs: PrimitiveInteger<Unsigned = T::Unsigned>,
        Simd<Rhs, N>: SupportedSimd,
    {
      $(#[$doc])*
      #[inline]
      fn $op_assign(&mut self, rhs: Simd<Rhs, N>) {
        *self = self.$op(rhs);
      }
    }

    impl<T, const N: usize, Rhs> $Op<&Simd<Rhs, N>> for Simd<T, N>
    where
        T: $ElementTrait,
        Simd<T, N>: SupportedSimd,
        Simd<T::Unsigned, N>: SupportedSimd,
        Rhs: PrimitiveInteger<Unsigned = T::Unsigned>,
        Simd<Rhs, N>: SupportedSimd,
    {
      type Output = Self;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: &Simd<Rhs, N>) -> Self::Output {
        self.$op(*rhs)
      }
    }

    impl<T, const N: usize, Rhs> $OpAssign<&Simd<Rhs, N>> for Simd<T, N>
    where
        T: $ElementTrait,
        Simd<T, N>: SupportedSimd,
        Simd<T::Unsigned, N>: SupportedSimd,
        Rhs: PrimitiveInteger<Unsigned = T::Unsigned>,
        Simd<Rhs, N>: SupportedSimd,
    {
      $(#[$doc])*
      #[inline]
      fn $op_assign(&mut self, rhs: &Simd<Rhs, N>) {
        *self = self.$op(*rhs);
      }
    }

    impl<T, const N: usize, Rhs> $Op<Simd<Rhs, N>> for &Simd<T, N>
    where
        T: $ElementTrait,
        Simd<T, N>: SupportedSimd,
        Simd<T::Unsigned, N>: SupportedSimd,
        Rhs: PrimitiveInteger<Unsigned = T::Unsigned>,
        Simd<Rhs, N>: SupportedSimd,
    {
      type Output = Simd<T, N>;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: Simd<Rhs, N>) -> Self::Output {
        (*self).$op(rhs)
      }
    }

    impl<T, const N: usize, Rhs> $Op<&Simd<Rhs, N>> for &Simd<T, N>
    where
        T: $ElementTrait,
        Simd<T, N>: SupportedSimd,
        Simd<T::Unsigned, N>: SupportedSimd,
        Rhs: PrimitiveInteger<Unsigned = T::Unsigned>,
        Simd<Rhs, N>: SupportedSimd,
    {
      type Output = Simd<T, N>;

      $(#[$doc])*
      #[inline]
      fn $op(self, rhs: &Simd<Rhs, N>) -> Self::Output {
        (*self).$op(*rhs)
      }
    }

    macro_rules! impl_scalar {
      ($T:ident) => {
        impl<T, const N: usize> $Op<$T> for Simd<T, N>
        where
          T: SimdElement,
          Simd<T, N>: SupportedSimd,
        {
          type Output = Self;

          $(#[$doc_scalar])*
          #[inline]
          fn $op(self, rhs: $T) -> Self::Output {
            self.$op_scalar_backend(rhs as u32)
          }
        }

        impl<const N: usize, Rhs> $Op<Simd<Rhs, N>> for $T
        where
            Simd<$T, N>: SupportedSimd,
            Simd<<$T as SimdElement>::Unsigned, N>: SupportedSimd,
            Rhs: PrimitiveInteger<Unsigned = <$T as SimdElement>::Unsigned>,
            Simd<Rhs, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_doc])*
          #[inline]
          fn $op(self, rhs: Simd<Rhs, N>) -> Self::Output {
            Simd::<$T, N>::splat(self).$op(rhs)
          }
        }

        impl<T, const N: usize> $OpAssign<$T> for Simd<T, N>
        where
          T: SimdElement,
          Simd<T, N>: SupportedSimd,
        {
          $(#[$doc_scalar])*
          #[inline]
          fn $op_assign(&mut self, rhs: $T) {
            *self = self.$op_scalar_backend(rhs as u32);
          }
        }

        impl<T, const N: usize> $Op<&$T> for Simd<T, N>
        where
          T: SimdElement,
          Simd<T, N>: SupportedSimd,
        {
          type Output = Self;

          $(#[$doc_scalar])*
          #[inline]
          fn $op(self, rhs: &$T) -> Self::Output {
            self.$op_scalar_backend(*rhs as u32)
          }
        }

        impl<const N: usize, Rhs> $Op<&Simd<Rhs, N>> for $T
        where
            Simd<$T, N>: SupportedSimd,
            Simd<<$T as SimdElement>::Unsigned, N>: SupportedSimd,
            Rhs: PrimitiveInteger<Unsigned = <$T as SimdElement>::Unsigned>,
            Simd<Rhs, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_doc])*
          #[inline]
          fn $op(self, rhs: &Simd<Rhs, N>) -> Self::Output {
            Simd::<$T, N>::splat(self).$op(*rhs)
          }
        }

        impl<T, const N: usize> $OpAssign<&$T> for Simd<T, N>
        where
          T: SimdElement,
          Simd<T, N>: SupportedSimd,
        {
          $(#[$doc_scalar])*
          #[inline]
          fn $op_assign(&mut self, rhs: &$T) {
            *self = self.$op_scalar_backend(*rhs as u32);
          }
        }

        impl<T, const N: usize> $Op<$T> for &Simd<T, N>
        where
          T: SimdElement,
          Simd<T, N>: SupportedSimd,
        {
          type Output = Simd<T, N>;

          $(#[$doc_scalar])*
          #[inline]
          fn $op(self, rhs: $T) -> Self::Output {
            self.$op_scalar_backend(rhs as u32)
          }
        }

        impl<const N: usize, Rhs> $Op<Simd<Rhs, N>> for &$T
        where
            Simd<$T, N>: SupportedSimd,
            Simd<<$T as SimdElement>::Unsigned, N>: SupportedSimd,
            Rhs: PrimitiveInteger<Unsigned = <$T as SimdElement>::Unsigned>,
            Simd<Rhs, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_doc])*
          #[inline]
          fn $op(self, rhs: Simd<Rhs, N>) -> Self::Output {
            Simd::<$T, N>::splat(*self).$op(rhs)
          }
        }

        impl<T, const N: usize> $Op<&$T> for &Simd<T, N>
        where
          T: SimdElement,
          Simd<T, N>: SupportedSimd,
        {
          type Output = Simd<T, N>;

          $(#[$doc_scalar])*
          #[inline]
          fn $op(self, rhs: &$T) -> Self::Output {
            self.$op_scalar_backend(*rhs as u32)
          }
        }

        impl<const N: usize, Rhs> $Op<&Simd<Rhs, N>> for &$T
        where
            Simd<$T, N>: SupportedSimd,
            Simd<<$T as SimdElement>::Unsigned, N>: SupportedSimd,
            Rhs: PrimitiveInteger<Unsigned = <$T as SimdElement>::Unsigned>,
            Simd<Rhs, N>: SupportedSimd,
        {
          type Output = Simd<$T, N>;

          $(#[$scalar_doc])*
          #[inline]
          fn $op(self, rhs: &Simd<Rhs, N>) -> Self::Output {
            Simd::<$T, N>::splat(*self).$op(*rhs)
          }
        }
      }
    }
    impl_scalar!(i8);
    impl_scalar!(i16);
    impl_scalar!(i32);
    impl_scalar!(i64);
    impl_scalar!(i128);
    impl_scalar!(isize);
    impl_scalar!(u8);
    impl_scalar!(u16);
    impl_scalar!(u32);
    impl_scalar!(u64);
    impl_scalar!(u128);
    impl_scalar!(usize);
  }
}

macro_rules! const_f32_as_f32x4 {
    ($i:ident, $f:expr) => {
        #[allow(non_upper_case_globals)]
        pub const $i: f32x4 = f32x4::from_array([$f; 4]);
    };
}

macro_rules! const_f32_as_f32x8 {
    ($i:ident, $f:expr) => {
        #[allow(non_upper_case_globals)]
        pub const $i: f32x8 = f32x8::from_array([$f; 8]);
    };
}

macro_rules! const_f32_as_f32x16 {
    ($i:ident, $f:expr) => {
        #[allow(non_upper_case_globals)]
        pub const $i: f32x16 = f32x16::from_array([$f; 16]);
    };
}

macro_rules! const_f64_as_f64x2 {
    ($i:ident, $f:expr) => {
        #[allow(non_upper_case_globals)]
        pub const $i: f64x2 = f64x2::from_array([$f; 2]);
    };
}

macro_rules! const_f64_as_f64x4 {
    ($i:ident, $f:expr) => {
        #[allow(non_upper_case_globals)]
        pub const $i: f64x4 = f64x4::from_array([$f; 4]);
    };
}

macro_rules! const_f64_as_f64x8 {
    ($i:ident, $f:expr) => {
        #[allow(non_upper_case_globals)]
        pub const $i: f64x8 = f64x8::from_array([$f; 8]);
    };
}

// TODO: make these generic over `fast_mul_add`? Worth it?

macro_rules! polynomial_2 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        x2.fast_mul_add($c2, x.fast_mul_add($c1, $c0))
    }};
}

macro_rules! polynomial_3 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        $c3.fast_mul_add(x, $c2)
            .fast_mul_add(x2, $c1.fast_mul_add(x, $c0))
    }};
}

macro_rules! polynomial_4 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr ,$c3:expr, $c4:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        $c3.fast_mul_add(x, $c2)
            .fast_mul_add(x2, $c1.fast_mul_add(x, $c0))
            + $c4 * x4
    }};
}

macro_rules! polynomial_5 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        $c3.fast_mul_add(x, $c2).fast_mul_add(
            x2,
            $c5.fast_mul_add(x, $c4)
                .fast_mul_add(x4, $c1.fast_mul_add(x, $c0)),
        )
    }};
}

macro_rules! polynomial_5n {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        x2.fast_mul_add(
            x.fast_mul_add($c3, $c2),
            (x4.fast_mul_add($c4 + x, x.fast_mul_add($c1, $c0))),
        )
    }};
}

macro_rules! polynomial_6 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr ,$c6:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        x4.fast_mul_add(
            x2.fast_mul_add($c6, x.fast_mul_add($c5, $c4)),
            x2.fast_mul_add(x.fast_mul_add($c3, $c2), x.fast_mul_add($c1, $c0)),
        )
    }};
}

macro_rules! polynomial_6n {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        x4.fast_mul_add(
            x.fast_mul_add($c5, x2 + $c4),
            x2.fast_mul_add(x.fast_mul_add($c3, $c2), x.fast_mul_add($c1, $c0)),
        )
    }};
}

macro_rules! polynomial_7 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr, $c6:expr, $c7:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        x4.fast_mul_add(
            x2.fast_mul_add(x.fast_mul_add($c7, $c6), x.fast_mul_add($c5, $c4)),
            x2.fast_mul_add(x.fast_mul_add($c3, $c2), x.fast_mul_add($c1, $c0)),
        )
    }};
}

macro_rules! polynomial_8 {
    ($x:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr,  $c6:expr, $c7:expr, $c8:expr $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        x4.fast_mul_add(
            x2.fast_mul_add($c7.fast_mul_add(x, $c6), x.fast_mul_add($c5, $c4)),
            x8.fast_mul_add(
                $c8,
                x2.fast_mul_add(x.fast_mul_add($c3, $c2), x.fast_mul_add($c1, $c0)),
            ),
        )
    }};
}

macro_rules! polynomial_13 {
    // calculates polynomial c13*x^13 + c12*x^12 + ... + c1*x + c0
    ($x:expr,  $c2:expr, $c3:expr, $c4:expr, $c5:expr,$c6:expr, $c7:expr, $c8:expr,$c9:expr, $c10:expr, $c11:expr, $c12:expr, $c13:expr  $(,)?) => {{
        let x = $x;
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        x8.fast_mul_add(
            x4.fast_mul_add(
                x.fast_mul_add($c13, $c12),
                x2.fast_mul_add(x.fast_mul_add($c11, $c10), x.fast_mul_add($c9, $c8)),
            ),
            x4.fast_mul_add(
                x2.fast_mul_add(x.fast_mul_add($c7, $c6), x.fast_mul_add($c5, $c4)),
                x2.fast_mul_add(x.fast_mul_add($c3, $c2), x),
            ),
        )
    }};
}

/// A mask for the low `W` bits, the only part of a 64-bit `add_mul_lo` or
/// `add_mul_hi` operand that is read.
#[inline]
pub const fn add_mul_operand_mask_u64<const W: u32>() -> u64 {
    const { assert!(W >= 1 && W <= 64, "`add_mul` width must be in `1..=64`") };
    u64::MAX >> (64 - W)
}

/// One `u64` lane of `add_mul_lo`: `acc + ((a * b) mod 2^W)`.
#[inline]
pub fn add_mul_lo_lane_u64<const W: u32>(acc: u64, a: u64, b: u64) -> u64 {
    let mask = add_mul_operand_mask_u64::<W>();
    let product = (a & mask) as u128 * (b & mask) as u128;
    acc.wrapping_add(product as u64 & mask)
}

/// One `u64` lane of `add_mul_hi`: `acc + ((a * b) >> W)`.
#[inline]
pub fn add_mul_hi_lane_u64<const W: u32>(acc: u64, a: u64, b: u64) -> u64 {
    let mask = add_mul_operand_mask_u64::<W>();
    let product = (a & mask) as u128 * (b & mask) as u128;
    acc.wrapping_add((product >> W) as u64)
}

/// A mask for the low `W` bits, the only part of a 32-bit `add_mul_lo` or
/// `add_mul_hi` operand that is read.
#[inline]
pub const fn add_mul_operand_mask_u32<const W: u32>() -> u32 {
    const { assert!(W >= 1 && W <= 32, "`add_mul` width must be in `1..=32`") };
    u32::MAX >> (32 - W)
}

/// One `u32` lane of `add_mul_lo`: `acc + ((a * b) mod 2^W)`.
#[inline]
pub fn add_mul_lo_lane_u32<const W: u32>(acc: u32, a: u32, b: u32) -> u32 {
    let mask = add_mul_operand_mask_u32::<W>();
    let product = (a & mask) as u64 * (b & mask) as u64;
    acc.wrapping_add(product as u32 & mask)
}

/// One `u32` lane of `add_mul_hi`: `acc + ((a * b) >> W)`.
#[inline]
pub fn add_mul_hi_lane_u32<const W: u32>(acc: u32, a: u32, b: u32) -> u32 {
    let mask = add_mul_operand_mask_u32::<W>();
    let product = (a & mask) as u64 * (b & mask) as u64;
    acc.wrapping_add((product >> W) as u32)
}

#[allow(dead_code)]
pub fn generic_bit_blend<T>(mask: T, y: T, n: T) -> T
where
    T: Copy + BitXor<Output = T> + BitAnd<Output = T>,
{
    n ^ ((n ^ y) & mask)
}

#[allow(unused)]
pub fn software_sqrt(x: f64) -> f64 {
    use core::num::Wrapping;
    type wu32 = Wrapping<u32>;
    const fn w(u: u32) -> wu32 {
        Wrapping(u)
    }
    let mut z: f64;
    let sign: wu32 = w(0x80000000);
    let mut ix0: i32;
    let mut s0: i32;
    let mut q: i32;
    let mut m: i32;
    let mut t: i32;
    let mut i: i32;
    let mut r: wu32;
    let mut t1: wu32;
    let mut s1: wu32;
    let mut ix1: wu32;
    let mut q1: wu32;
    // extract data

    pick! {
      if #[cfg(target_endian = "little")]
      {
        let [low, high]: [u32; 2] = cast(x);
        ix0 = high as i32;
        ix1 = w(low);
      }
      else
      {
        let [high, low]: [u32; 2] = cast(x);
        ix0 = high as i32;
        ix1 = w(low);
      }
    }

    // inf and nan
    {
        if x.is_nan() {
            return f64::NAN;
        }
        if ix0 & 0x7ff00000 == 0x7ff00000 {
            return x * x + x;
        }
    }
    // handle zero
    {
        if ix0 <= 0 {
            if ((ix0 & (!sign).0 as i32) | (ix1.0 as i32)) == 0 {
                return x;
            } else if ix0 < 0 {
                return (x - x) / (x - x);
            }
        }
    }
    // normalize
    {
        m = ix0 >> 20;
        if m == 0 {
            // subnormal
            while ix0 == 0 {
                m -= 21;
                ix0 |= (ix1 >> 11).0 as i32;
                ix1 <<= 21;
            }
            i = 0;
            while ix0 & 0x00100000 == 0 {
                ix0 <<= 1;
                i += 1;
            }
            m -= i - 1;
            ix0 |= (ix1.0 >> (31 - i)) as i32;
            ix1 <<= i as usize;
        }
        // un-bias exponent
        m -= 1023;
        ix0 = (ix0 & 0x000fffff) | 0x00100000;
        if (m & 1) != 0 {
            // odd m, double the input to make it even
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
        }
        m >>= 1;
    }
    // generate sqrt bit by bit
    {
        ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
        ix1 += ix1;
        // q and q1 store the sqrt(x);
        q = 0;
        q1 = w(0);
        s0 = 0;
        s1 = w(0);
        // our bit that moves from right to left
        r = w(0x00200000);
        while r != w(0) {
            t = s0 + (r.0 as i32);
            if t <= ix0 {
                s0 = t + (r.0 as i32);
                ix0 -= t;
                q += (r.0 as i32);
            }
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
            r >>= 1;
        }
        r = sign;
        while r != w(0) {
            t1 = s1 + r;
            t = s0;
            if (t < ix0) || ((t == ix0) && (t1 <= ix1)) {
                s1 = t1 + r;
                if t1 & sign == sign && (s1 & sign) == w(0) {
                    s0 += 1;
                }
                ix0 -= t;
                if ix1 < t1 {
                    ix0 -= 1;
                }
                ix1 -= t1;
                q1 += r;
            }
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
            r >>= 1;
        }
    }
    // use floating add to find out rounding direction
    {
        if ix0 | (ix1.0 as i32) != 0 {
            z = 1.0 - 1.0e-300;
            if z >= 1.0 {
                z = 1.0 + 1.0e-300;
                if q1 == w(0xffffffff) {
                    q1 = w(0);
                    q += 1;
                } else if z > 1.0 {
                    if q1 == w(0xfffffffe) {
                        q += 1;
                    }
                    q1 += w(2);
                } else {
                    q1 += q1 & w(1);
                }
            }
        }
    }
    // finish up
    ix0 = (q >> 1) + 0x3fe00000;
    ix1 = q1 >> 1;
    if q & 1 == 1 {
        ix1 |= sign;
    }
    ix0 += m << 20;

    pick! {
      if #[cfg(target_endian = "little")]
      {
        cast::<[u32; 2], f64>([ix1.0, ix0 as u32])
      }
      else
      {
        cast::<[u32; 2], f64>([ix0 as u32, ix1.0])
      }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::software_sqrt;

    #[test]
    fn test_software_sqrt() {
        assert!(software_sqrt(f64::NAN).is_nan());
        assert_eq!(software_sqrt(f64::INFINITY), f64::INFINITY);
        assert_eq!(software_sqrt(0.0), 0.0);
        assert_eq!(software_sqrt(-0.0), -0.0);
        assert!(software_sqrt(-1.0).is_nan());
        assert!(software_sqrt(f64::NEG_INFINITY).is_nan());
        assert_eq!(software_sqrt(4.0), 2.0);
        assert_eq!(software_sqrt(9.0), 3.0);
        assert_eq!(software_sqrt(16.0), 4.0);
        assert_eq!(software_sqrt(25.0), 5.0);
        assert_eq!(software_sqrt(5000.0 * 5000.0), 5000.0);
    }
}
