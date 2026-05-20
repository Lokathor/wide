macro_rules! integer_impl_div_rem {
  ($T:ident, $Simd:ident, [$($index:literal),* $(,)?] $(,)?) => {
    impl Div for $Simd {
      type Output = Self;

      /// Lanewise divide.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      fn div(self, rhs: Self) -> Self::Output {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([$(self_array[$index].wrapping_div(rhs_array[$index])),*])
      }
    }

    impl Div<$T> for $Simd {
      type Output = Self;

      /// Lanewise divide.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      fn div(self, rhs: $T) -> Self::Output {
        let self_array = self.to_array();

        Self::new([$(self_array[$index].wrapping_div(rhs)),*])
      }
    }

    impl Div<$Simd> for $T {
      type Output = $Simd;

      /// Lanewise divide.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      fn div(self, rhs: $Simd) -> Self::Output {
        let rhs_array = rhs.to_array();

        $Simd::new([$(self.wrapping_div(rhs_array[$index])),*])
      }
    }

    impl Rem for $Simd {
      type Output = Self;

      /// Lanewise remainder.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      fn rem(self, rhs: Self) -> Self::Output {
        let self_array = self.to_array();
        let rhs_array = rhs.to_array();

        Self::new([$(self_array[$index].wrapping_rem(rhs_array[$index])),*])
      }
    }

    impl Rem<$T> for $Simd {
      type Output = Self;

      /// Lanewise remainder.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      fn rem(self, rhs: $T) -> Self::Output {
        let self_array = self.to_array();

        Self::new([$(self_array[$index].wrapping_rem(rhs)),*])
      }
    }

    impl Rem<$Simd> for $T {
      type Output = $Simd;

      /// Lanewise remainder.
      ///
      /// Note that because division has no hardware support, this operation is
      /// very slow and should be avoided if possible.
      #[inline]
      fn rem(self, rhs: $Simd) -> Self::Output {
        let rhs_array = rhs.to_array();

        $Simd::new([$(self.wrapping_rem(rhs_array[$index])),*])
      }
    }
  };
}

macro_rules! int_uint_consts {
  ($type:ty, $lanes:expr, $simd:ty, $bits:expr) => {
    // ensure the size of the SIMD type is the same as the size of the array and
    // number of bits is OK
    const _: () = assert!(
      core::mem::size_of::<$simd>() == core::mem::size_of::<[$type; $lanes]>()
        && core::mem::size_of::<$simd>() * 8 == $bits as usize
    );

    impl $simd {
      pub const ONE: $simd = <$simd>::new([1; $lanes]);
      pub const ZERO: $simd = <$simd>::new([0; $lanes]);
      pub const MAX: $simd = <$simd>::new([<$type>::MAX; $lanes]);
      pub const MIN: $simd = <$simd>::new([<$type>::MIN; $lanes]);

      /// The number of lanes in this SIMD vector.
      pub const LANES: u16 = $lanes;

      /// The size of this SIMD vector in bits.
      pub const BITS: u16 = $bits;
    }
  };
}

macro_rules! integer_fn_clamp {
  () => {
    /// Restrict each element to a certain interval.
    ///
    /// If `min > max`, the result is unspeficied. Consider manually checking
    /// for that case.
    #[inline]
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
      self.max(min).min(max)
    }
  };
}

macro_rules! signed_fn_signum {
  () => {
    /// Returns numbers representing the sign of each element.
    ///
    /// - `0` if the number is zero
    /// - `1` if the number is positive
    /// - `-1` if the number is negative
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
      // Flip signs because the result for true in `is_positive/negative` is
      // `-1` (all bits set).
      self.is_negative() - self.is_positive()
    }
  };
}
