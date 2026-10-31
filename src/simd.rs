//! Defines the library's types and APIs applicable to arbitrary SIMD vectors.

use core::mem::transmute_copy;

use bytemuck::{Pod, Zeroable, pod_align_to, pod_align_to_mut};
#[cfg(feature = "serde")]
use serde_core::{Deserialize, Serialize, ser::SerializeTuple};

use crate::{PrimitiveSigned, PrimitiveUnsigned};

/// TODO(generics): Add documentation and explain how to use SIMD vectors.
///
/// # Layout
///
/// TODO(generics): Define this type's precise layout guarantees (size,
/// alignment, what about non-power-of-two lane counts).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Simd<T, const N: usize>(pub(crate) <Self as SimdBackend>::Inner)
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd;

/// A trait for types that can be used as SIMD vector elements.
///
/// Note that because [`Simd<T, N>`] only supports certain combinations of `T`
/// and `N`, there may be `T` types that implement [`SimdElement`] but are not
/// actually supported in any real SIMD type. For example, [`usize`] implements
/// [`SimdElement`], but there is currently no [`Simd<usize, N>`] type that
/// implements [`SupportedSimd`].
#[expect(private_bounds)]
pub trait SimdElement: Sealed + Copy {
    /// An unsigned integer with the same size as `self`.
    ///
    /// For floats and integers, the unsigned type is guaranteed to have their
    /// exact size. However, it is possible that SIMD element types added in the
    /// future will not have an exact unsigned integer equivalent, and will use
    /// a type with a different size.
    type Unsigned: PrimitiveUnsigned<Signed = Self::Signed>;

    /// A signed integer with the same size as `self`.
    ///
    /// This is the signed variant of [`Self::Unsigned`].
    type Signed: PrimitiveSigned<Unsigned = Self::Unsigned>;
}

/// A trait for [`Simd<T, N>`] types that are actually supported.
///
/// Due to Rust language limitations, [`Simd<T, N>`] only supports certain
/// combinations of `T` and `N`. The trait bound `Simd<T, N>: SupportedSimd`
/// must be used in generic contexts to make sure the given SIMD type is
/// supported.
///
/// In the future, if Rust stabilizes the appropriate language features,
/// [`Simd<T, N>`] would support all combinations of `T` and `N`, the trait
/// bound would be removed and this trait would be deprecated.
#[expect(private_bounds)]
pub trait SupportedSimd: Sealed + SimdBackend {
    /// The unsigned integer used in [`Simd::to_bitmask`].
    ///
    /// For most SIMD types, this is [`u32`]. For [`u8x64`] and [`i8x64`], this
    /// is [`u64`].
    type Bitmask: PrimitiveUnsigned;
}

/// Defines the `Inner` type (the type contained inside `Simd<T, N>`) and the
/// internal implementations of functions applicable to arbitrary SIMD vectors.
pub(crate) trait SimdBackend: Sized {
    type Inner: Copy;

    fn simd_eq_backend(self, other: Self) -> Self;

    fn simd_ne_backend(self, other: Self) -> Self;

    fn simd_lt_backend(self, other: Self) -> Self;

    fn simd_gt_backend(self, other: Self) -> Self;

    fn simd_le_backend(self, other: Self) -> Self;

    fn simd_ge_backend(self, other: Self) -> Self;

    fn select_backend(self, if_true: Self, if_false: Self) -> Self;

    fn to_bitmask_backend(self) -> Self::Bitmask
    where
        Self: SupportedSimd;

    fn any_backend(self) -> bool;

    fn all_backend(self) -> bool;

    fn interleave_low_backend(self, other: Self) -> Self;

    fn interleave_high_backend(self, other: Self) -> Self;

    fn shuffle_backend(self, indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_zeroing_backend(self, indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_wrapping_backend(self, indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_2_backend(inputs: [Self; 2], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_zeroing_2_backend(inputs: [Self; 2], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_wrapping_2_backend(inputs: [Self; 2], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_3_backend(inputs: [Self; 3], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_zeroing_3_backend(inputs: [Self; 3], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_wrapping_3_backend(inputs: [Self; 3], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_4_backend(inputs: [Self; 4], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_zeroing_4_backend(inputs: [Self; 4], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn shuffle_wrapping_4_backend(inputs: [Self; 4], indices: Self::UnsignedSimd) -> Self
    where
        Self: SimdAssociatedTypes + SupportSimdOf<Self::UnsignedT>;

    fn transpose_backend(data: Self::ArrayOf<Self>) -> Self::ArrayOf<Self>
    where
        Self: SimdAssociatedTypes;

    fn replace_const_backend<const INDEX: usize>(self, value: Self::T) -> Self
    where
        Self: SimdAssociatedTypes;

    fn extract_const_backend<const INDEX: usize>(self) -> Self::T
    where
        Self: SimdAssociatedTypes;
}

/// Provides SIMD backend traits access to SIMD associated types.
///
/// SIMD backend traits are not generic over `T` nor `N` (they cannot due to
/// type-system limitations). As a result, they cannot easily refer to types
/// that depends on `T` and `N`.
///
/// Such types include `[Self; N]` (used in [`Simd::transpose`]) and
/// `Simd<T::Unsigned, N>` (used in [`Simd::shuffle`]).
///
/// To solve this problem, SIMD backend functions that need access to such types
/// add `where Self: SimdAssociatedTypes` which gives them access to appropriate
/// types.
///
/// [`SimdAssociatedTypes`] has one shared generic implementation for all
/// supported [`Simd<T, N>`] types. This means that the compiler, in
/// contexts generic over `T` and `N`, understands what these types actually are
/// (for example, for `Self = Simd<T, N>`, that [`Self::T`]  is `T` and that
/// [`Self::ArrayOf<U>`] is `[U; N]`).
pub(crate) trait SimdAssociatedTypes: Sized {
    /// For `Self = Simd<T, N>`, this type is `T`.
    type T;

    /// For `Self = Simd<T, N>`, this type is `[U; N]`.
    type ArrayOf<U>;

    /// For `Self = Simd<T, N>`, this type is `Simd<U, N>`.
    type SimdOf<U>
    where
        Self: SupportSimdOf<U>;

    /// For `Self = Simd<T, N>`, this type is `T::Unsigned`.
    type UnsignedT;

    /// For `Self = Simd<T, N>`, this type is `Simd<T::Unsigned, N>`.
    type UnsignedSimd
    where
        Self: SupportSimdOf<Self::UnsignedT>;

    /// For `Self = Simd<T, N>`, this type is `T::Signed`.
    type SignedT;

    /// For `Self = Simd<T, N>`, this type is `Simd<T::Signed, N>`.
    type SignedSimd
    where
        Self: SupportSimdOf<Self::SignedT>;
}

/// An extension of [`SimdAssociatedTypes`] for when `Self::SimdOf<U>` must
/// implement [`SupportedSimd`].
///
/// When SIMD backend functions refer to SIMD vectors of generic `T` types
/// other than `T` (using [`SimdAssociatedTypes::SimdOf<U>`]), they must have a
/// trait bound that guarantees the type `Self::SimdOf<U>` is actually defined
/// (which requires it to implement [`SupportedSimd`]).
///
/// The `Self::SimdOf<U>: SupportedSimd` bound cannot be used directly. That
/// results in a cycle. Instead, that bound is represented with
/// `Self: SupportSimdOf<U>` (for `Self::SimdOf<U>: SupportedSimd`.
pub(crate) trait SupportSimdOf<U> {
    type SimdOfU;
}

/// A sealed trait for SIMD element types that support mask operations.
///
/// All current SIMD element types implement this. However, it is possible that
/// in the future, there will be SIMD element types that do not satisfy
/// [`Pod`]-like requirements, and thus cannot support the existing mask
/// operations.
pub trait MaskOps: Sealed {}

/// A trait for lanewise SIMD selection.
///
/// The method [`select`] is used to select lanes from two SIMD values based on
/// the elements of a SIMD [mask].
///
/// The type of values to select from could be a SIMD vector (e.g.,
/// [`f32x4`]), or it could be a composite type over SIMD vectors (e.g., a
/// `Vec3<f32x4>` structure).
///
/// This trait is meant to be implemented for such downstream types. If you
/// simply want to call [`select`], you do not need to import this trait as SIMD
/// vectors have an inherent method for this.
///
/// # Example
///
/// ```
/// # use wide::{u32x4, Select};
/// #
/// #[derive(Debug, PartialEq)]
/// struct Point {
///     x: u32x4,
///     y: u32x4,
/// }
///
/// impl Select<Point> for u32x4 {
///     fn select(self, if_true: Point, if_false: Point) -> Point {
///         Point {
///             x: self.select(if_true.x, if_false.x),
///             y: self.select(if_true.y, if_false.y),
///         }
///     }
/// }
///
/// let mask = u32x4::from_array([0, !0, !0, 0]);
/// let if_true = Point { x: u32x4::splat(2), y: u32x4::ZERO };
/// let if_false = Point { x: u32x4::ZERO, y: u32x4::splat(5) };
///
/// assert_eq!(
///     mask.select(if_true, if_false),
///     Point {
///         x: u32x4::from_array([0, 2, 2, 0]),
///         y: u32x4::from_array([5, 0, 0, 5]),
///     },
/// );
/// ```
///
/// [`select`]: crate::f32x4::select
/// [`f32x4`]: crate::f32x4
/// [mask]: crate#masks
pub trait Select<T> {
    /// Lanewise SIMD selection.
    ///
    /// If `self` is a SIMD vector, it is treated as a [mask], where
    /// each lane is considered true if all bits are one, or false if all bits are
    /// zero.
    ///
    /// For each lane:
    ///
    /// - If `self` is true, return the corresponding lane of `if_true`
    /// - If `self` is false, return the corresponding lane of `if_false`
    ///
    /// [mask]: crate#masks
    #[must_use]
    fn select(self, if_true: T, if_false: T) -> T;
}

/// An extension trait implemented for arrays of SIMD vectors, which provides
/// shuffling from multiple input vectors.
///
/// # Example
///
/// ```
/// use wide::{f32x4, ShuffleExt, u32x4};
///
/// let simd_a = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
/// let simd_b = f32x4::from_array([100.0, 101.0, 102.0, 103.0]);
///
/// // Here, indices `0..4` map to `simd_a`, and indices `4..8` map to `simd_b`
/// let from_two_vectors = [simd_a, simd_b].shuffle(u32x4::from_array([2, 3, 4, 5]));
///
/// assert_eq!(from_two_vectors, f32x4::from_array([2.0, 3.0, 100.0, 101.0]));
/// ```
#[expect(private_bounds)]
pub trait ShuffleExt: Sealed {
    /// The type representing indices.
    ///
    /// This is always a SIMD vector of the unsigned integer with the same size as
    /// `T` and the same number of elements.
    type Indices;

    /// The type returned by shuffle functions.
    ///
    /// This is always the type of SIMD vector contained in the array.
    type Output;

    /// Returns a SIMD vector whose elements are selected from multiple input
    /// vectors using the corresponding runtime `indices`.
    ///
    /// If `N` is the number of elements in each vector, indices in the range
    /// `0..N` select values from `self[0]`, indices in the range `N..N * 2`
    /// select values from `self[1]`, etc.
    ///
    /// If an index is out of bounds (greater than or equal to `N * INPUTS`), the
    /// corresponding result element is unspecified.
    ///
    /// # Example
    ///
    /// ```
    /// use wide::{f32x4, ShuffleExt, u32x4};
    ///
    /// let simd_a = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
    /// let simd_b = f32x4::from_array([100.0, 101.0, 102.0, 103.0]);
    ///
    /// // Here, indices `0..4` map to `simd_a`, and indices `4..8` map to `simd_b`
    /// let from_two_vectors = [simd_a, simd_b].shuffle(u32x4::from_array([2, 3, 4, 5]));
    ///
    /// assert_eq!(from_two_vectors, f32x4::from_array([2.0, 3.0, 100.0, 101.0]));
    /// ```
    ///
    /// # Type-specific guarantees
    ///
    /// For all 8-bit types, it is guaranteed that for out of bounds indices,
    /// either zero is returned or the index wraps around, non-deterministically
    /// (unlike other types, which can return arbitrary values).
    #[must_use]
    fn shuffle(self, indices: Self::Indices) -> Self::Output;

    /// Returns a SIMD vector whose elements are selected from multiple input
    /// vectors using the corresponding runtime `indices`.
    ///
    /// If `N` is the number of elements in each vector, indices in the range
    /// `0..N` select values from `self[0]`, indices in the range `N..N * 2`
    /// select values from `self[1]`, etc.
    ///
    /// If an index is out of bounds (greater than or equal to `N * INPUTS`), the
    /// corresponding result element is the number zero.
    ///
    /// # Example
    ///
    /// ```
    /// use wide::{f32x4, ShuffleExt, u32x4};
    ///
    /// let simd_a = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
    /// let simd_b = f32x4::from_array([100.0, 101.0, 102.0, 103.0]);
    ///
    /// // Here, indices `0..4` map to `simd_a`, and indices `4..8` map to `simd_b`
    /// let from_two_vectors = [simd_a, simd_b].shuffle_zeroing(u32x4::from_array([2, 3, 100, 5]));
    ///
    /// assert_eq!(from_two_vectors, f32x4::from_array([2.0, 3.0, 0.0, 101.0]));
    /// ```
    #[must_use]
    fn shuffle_zeroing(self, indices: Self::Indices) -> Self::Output;

    /// Returns a SIMD vector whose elements are selected from multiple input
    /// vectors using the corresponding runtime `indices`.
    ///
    /// If `N` is the number of elements in each vector, indices in the range
    /// `0..N` select values from `self[0]`, indices in the range `N..N * 2`
    /// select values from `self[1]`, etc.
    ///
    /// Indices are wrapped by `N * INPUTS`.
    ///
    /// # Example
    ///
    /// ```
    /// use wide::{f32x4, ShuffleExt, u32x4};
    ///
    /// let simd_a = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
    /// let simd_b = f32x4::from_array([100.0, 101.0, 102.0, 103.0]);
    ///
    /// // Here, indices `0..4` map to `simd_a`, and indices `4..8` map to `simd_b`
    /// let from_two_vectors = [simd_a, simd_b].shuffle_wrapping(u32x4::from_array([2, 3, 9, 5]));
    ///
    /// assert_eq!(from_two_vectors, f32x4::from_array([2.0, 3.0, 1.0, 101.0]));
    /// ```
    #[must_use]
    fn shuffle_wrapping(self, indices: Self::Indices) -> Self::Output;
}

/// A deprecated trait for the [`simd_eq`] function.
///
/// [`simd_eq`]: crate::f32x4::simd_eq
#[deprecated(since = "1.5.0", note = "use the inherent function `simd_eq` instead")]
pub trait CmpEq<Rhs = Self> {
    /// The type returned by [`simd_eq`].
    ///
    /// [`simd_eq`]: Self::simd_eq
    type Output;

    /// Returns a [mask] that checks if each element of `self` is equal to the
    /// corresponding element of `rhs`.
    ///
    /// This is a method of the deprecated [`CmpEq`] trait. Use the inherent
    /// function instead.
    ///
    /// [mask]: crate#masks
    fn simd_eq(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_gt`] function.
///
/// [`simd_gt`]: crate::f32x4::simd_gt
#[deprecated(since = "1.5.0", note = "use the inherent function `simd_gt` instead")]
pub trait CmpGt<Rhs = Self> {
    /// The type returned by [`simd_gt`].
    ///
    /// [`simd_gt`]: Self::simd_gt
    type Output;

    /// Returns a [mask] that checks if each element of `self` is greater than the
    /// corresponding element of `rhs`.
    ///
    /// This is a method of the deprecated [`CmpGt`] trait. Use the inherent
    /// function instead.
    ///
    /// [mask]: crate#masks
    fn simd_gt(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_ge`] function.
///
/// [`simd_ge`]: crate::f32x4::simd_ge
#[deprecated(since = "1.5.0", note = "use the inherent function `simd_ge` instead")]
pub trait CmpGe<Rhs = Self> {
    /// The type returned by [`simd_ge`].
    ///
    /// [`simd_ge`]: Self::simd_ge
    type Output;

    /// Returns a [mask] that checks if each element of `self` is greater than or
    /// equal to the corresponding element of `rhs`.
    ///
    /// This is a method of the deprecated [`CmpGe`] trait. Use the inherent
    /// function instead.
    ///
    /// [mask]: crate#masks
    fn simd_ge(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_ne`] function.
///
/// [`simd_ne`]: crate::f32x4::simd_ne
#[deprecated(since = "1.5.0", note = "use the inherent function `simd_ne` instead")]
pub trait CmpNe<Rhs = Self> {
    /// The type returned by [`simd_ne`].
    ///
    /// [`simd_ne`]: Self::simd_ne
    type Output;

    /// Returns a [mask] that checks if each element of `self` is not equal to the
    /// corresponding element of `rhs`.
    ///
    /// This is a method of the deprecated [`CmpNe`] trait. Use the inherent
    /// function instead.
    ///
    /// [mask]: crate#masks
    fn simd_ne(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_lt`] function.
///
/// [`simd_lt`]: crate::f32x4::simd_lt
#[deprecated(since = "1.5.0", note = "use the inherent function `simd_lt` instead")]
pub trait CmpLt<Rhs = Self> {
    /// The type returned by [`simd_lt`].
    ///
    /// [`simd_lt`]: Self::simd_lt
    type Output;

    /// Returns a [mask] that checks if each element of `self` is less than the
    /// corresponding element of `rhs`.
    ///
    /// This is a method of the deprecated [`CmpLt`] trait. Use the inherent
    /// function instead.
    ///
    /// [mask]: crate#masks
    fn simd_lt(self, rhs: Rhs) -> Self::Output;
}

/// A deprecated trait for the [`simd_le`] function.
///
/// [`simd_le`]: crate::f32x4::simd_le
#[deprecated(since = "1.5.0", note = "use the inherent function `simd_le` instead")]
pub trait CmpLe<Rhs = Self> {
    /// The type returned by [`simd_le`].
    ///
    /// [`simd_le`]: Self::simd_le
    type Output;

    /// Returns a [mask] that checks if each element of `self` is less than or
    /// equal to the corresponding element of `rhs`.
    ///
    /// This is a method of the deprecated [`CmpLe`] trait. Use the inherent
    /// function instead.
    ///
    /// [mask]: crate#masks
    fn simd_le(self, rhs: Rhs) -> Self::Output;
}

/// A trait for SIMD variants of [`align_to`] functions.
///
/// [`align_to`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to
pub trait AlignTo
where
    Self: Pod + Default + PartialEq + From<Self::Elem>,
    Self::Elem: Pod + Default + PartialEq,
{
    /// The element type of this SIMD vector.
    type Elem;

    /// A SIMD variant of [`align_to`].
    ///
    /// [`align_to`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to
    #[inline]
    fn simd_align_to(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
        pod_align_to(slice)
    }

    /// A SIMD variant of [`align_to_mut`].
    ///
    /// [`align_to_mut`]: https://doc.rust-lang.org/std/primitive.slice.html#method.align_to_mut
    #[inline]
    fn simd_align_to_mut(
        slice: &mut [Self::Elem],
    ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
        pod_align_to_mut(slice)
    }
}

trait Sealed {}

impl SimdElement for f32 {
    type Unsigned = u32;
    type Signed = i32;
}
impl SimdElement for f64 {
    type Unsigned = u64;
    type Signed = i64;
}
impl SimdElement for i8 {
    type Unsigned = u8;
    type Signed = i8;
}
impl SimdElement for i16 {
    type Unsigned = u16;
    type Signed = i16;
}
impl SimdElement for i32 {
    type Unsigned = u32;
    type Signed = i32;
}
impl SimdElement for i64 {
    type Unsigned = u64;
    type Signed = i64;
}
impl SimdElement for i128 {
    type Unsigned = u128;
    type Signed = i128;
}
impl SimdElement for isize {
    type Unsigned = usize;
    type Signed = isize;
}
impl SimdElement for u8 {
    type Unsigned = u8;
    type Signed = i8;
}
impl SimdElement for u16 {
    type Unsigned = u16;
    type Signed = i16;
}
impl SimdElement for u32 {
    type Unsigned = u32;
    type Signed = i32;
}
impl SimdElement for u64 {
    type Unsigned = u64;
    type Signed = i64;
}
impl SimdElement for u128 {
    type Unsigned = u128;
    type Signed = i128;
}
impl SimdElement for usize {
    type Unsigned = usize;
    type Signed = isize;
}

impl MaskOps for f32 {}
impl MaskOps for f64 {}
impl MaskOps for i8 {}
impl MaskOps for i16 {}
impl MaskOps for i32 {}
impl MaskOps for i64 {}
impl MaskOps for i128 {}
impl MaskOps for isize {}
impl MaskOps for u8 {}
impl MaskOps for u16 {}
impl MaskOps for u32 {}
impl MaskOps for u64 {}
impl MaskOps for u128 {}
impl MaskOps for usize {}

macro_rules! define_simd {
  (
      T = $T:ident,
      N = $N:literal,
      N_STR = $N_STR:literal,
      Simd = $Simd:ident,
      UnsignedSimd = $UnsignedSimd:ident,
      Bitmask = $Bitmask:ident,
      optional_type_x86_inner { $(X86Inner = $X86Inner:ident)? },
      optional_type_arm_inner { $(ArmInner = $ArmInner:ident)? },
      optional_type_wasm_inner { $(WasmInner = $WasmInner:ident)? },
  ) => {
    #[doc = concat!("A SIMD vector with ", $N_STR, " elements of type [`", stringify!($T), "`].")]
    pub type $Simd = Simd<$T, $N>;

    /// Constructs a specific SIMD type from its `Inner` type using its
    /// type-alias name.
    ///
    /// This function is used because type-aliases do not support tuple-struct
    /// construction syntax.
    ///
    /// TODO(generics): Decide if this function should be removed in favor of
    /// constructing non-`Self` SIMD types using `Simd(inner)` instead of
    /// `{simd-type}(Inner)`. Wait until the generics API is ready to be merged
    /// onto `main`, only then remove this function if decided so (wait to avoid
    /// merge conflicts with `main`).
    pub(crate) fn $Simd(inner: <$Simd as SimdBackend>::Inner) -> $Simd {
      Simd(inner)
    }

    impl SupportedSimd for Simd<$T, $N> {
      type Bitmask = $Bitmask;
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
  };
}

impl<T, const N: usize> PartialEq for Simd<T, N>
where
    T: SimdElement + PartialEq,
    Simd<T, N>: SupportedSimd,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Use `all_backend` and not `all` in order to ignore the `T: MaskOps`
        // bound
        self.simd_eq(*other).all_backend()
    }
}

impl<T, const N: usize> Eq for Simd<T, N>
where
    T: SimdElement + Eq,
    Simd<T, N>: SupportedSimd,
{
}

impl<T, const N: usize> Default for Simd<T, N>
where
    T: SimdElement + Default,
    Simd<T, N>: SupportedSimd,
{
    #[inline]
    fn default() -> Self {
        // Currently, all supported `T` types have a default value where all
        // bits equal zero.

        // SAFETY: All current SIMD types accept the zero bit-pattern.
        unsafe { core::mem::zeroed::<Simd<T, N>>() }
    }
}

impl<T, const N: usize> From<[T; N]> for Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
{
    /// Converts an array to a SIMD vector.
    #[inline]
    fn from(arr: [T; N]) -> Self {
        Self::from_array(arr)
    }
}

impl<T, const N: usize> From<Simd<T, N>> for [T; N]
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
{
    /// Converts a SIMD vector to an array.
    #[inline]
    fn from(simd: Simd<T, N>) -> Self {
        simd.to_array()
    }
}

impl<T, const N: usize> From<T> for Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
{
    /// Converts a single value to a SIMD vector by setting all elements to
    /// that value.
    #[inline]
    fn from(elem: T) -> Self {
        Self::splat(elem)
    }
}

impl<T, const N: usize> From<&[T]> for Simd<T, N>
where
    T: SimdElement + Default,
    Simd<T, N>: SupportedSimd,
{
    /// Converts a slice to a SIMD vector, filling in [`Default::default()`] if
    /// there are not enough elements, and panicking if there are too many
    /// elements.
    ///
    /// Note that in the future, handling of too many elements may change.
    #[inline]
    fn from(value: &[T]) -> Self {
        assert!(
            value.len() <= N,
            "slice has more elements than the SIMD vector can store",
        );

        let mut result = Self::default();

        // SAFETY: `value` is valid for its own length, and `result` is valid
        // because its length is checked to be less than or equal to
        // `value.len()`. Both pointers are properly aligned because they
        // originate from a slice of `T`. The regions of memory do not overlap
        // because they originate from a shared reference and a mutable
        // reference.
        unsafe {
            core::ptr::copy_nonoverlapping::<T>(
                value.as_ptr(),
                result.as_mut_array().as_mut_ptr(),
                value.len(),
            );
        }

        result
    }
}

macro_rules! impl_formatting_trait {
    ($Trait:path) => {
        impl<T, const N: usize> $Trait for Simd<T, N>
        where
            T: SimdElement + $Trait,
            Simd<T, N>: SupportedSimd,
        {
            #[allow(clippy::missing_inline_in_public_items)]
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "(")?;
                for (i, x) in self.to_array().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    <T as $Trait>::fmt(x, f)?;
                }
                write!(f, ")")
            }
        }
    };
}
impl_formatting_trait!(core::fmt::Debug);
impl_formatting_trait!(core::fmt::Display);
impl_formatting_trait!(core::fmt::LowerExp);
impl_formatting_trait!(core::fmt::UpperExp);

impl<T, const N: usize> Select<Simd<T, N>> for Simd<T, N>
where
    T: SimdElement + MaskOps,
    Simd<T, N>: SupportedSimd,
{
    #[inline]
    fn select(self, if_true: Self, if_false: Self) -> Self {
        self.select_backend(if_true, if_false)
    }
}

impl<T, const N: usize> ShuffleExt for [Simd<T, N>; 2]
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
    Simd<T::Unsigned, N>: SupportedSimd,
{
    type Indices = Simd<T::Unsigned, N>;
    type Output = Simd<T, N>;

    #[inline]
    fn shuffle(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_2_backend(self, indices)
    }

    #[inline]
    fn shuffle_zeroing(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_zeroing_2_backend(self, indices)
    }

    #[inline]
    fn shuffle_wrapping(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_wrapping_2_backend(self, indices)
    }
}

impl<T, const N: usize> ShuffleExt for [Simd<T, N>; 3]
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
    Simd<T::Unsigned, N>: SupportedSimd,
{
    type Indices = Simd<T::Unsigned, N>;
    type Output = Simd<T, N>;

    #[inline]
    fn shuffle(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_3_backend(self, indices)
    }

    #[inline]
    fn shuffle_zeroing(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_zeroing_3_backend(self, indices)
    }

    #[inline]
    fn shuffle_wrapping(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_wrapping_3_backend(self, indices)
    }
}

impl<T, const N: usize> ShuffleExt for [Simd<T, N>; 4]
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
    Simd<T::Unsigned, N>: SupportedSimd,
{
    type Indices = Simd<T::Unsigned, N>;
    type Output = Simd<T, N>;

    #[inline]
    fn shuffle(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_4_backend(self, indices)
    }

    #[inline]
    fn shuffle_zeroing(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_zeroing_4_backend(self, indices)
    }

    #[inline]
    fn shuffle_wrapping(self, indices: Self::Indices) -> Self::Output {
        Simd::<T, N>::shuffle_wrapping_4_backend(self, indices)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpEq for Simd<T, N>
where
    T: SimdElement + PartialEq,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_eq(self, rhs: Self) -> Self {
        self.simd_eq_backend(rhs)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpEq<T> for Simd<T, N>
where
    T: SimdElement + PartialEq,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_eq(self, rhs: T) -> Self::Output {
        self.simd_eq(Self::splat(rhs))
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpNe for Simd<T, N>
where
    T: SimdElement + PartialEq,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_ne(self, rhs: Self) -> Self {
        self.simd_ne_backend(rhs)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpNe<T> for Simd<T, N>
where
    T: SimdElement + PartialEq,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_ne(self, rhs: T) -> Self::Output {
        self.simd_ne(Self::splat(rhs))
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpLt for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_lt(self, rhs: Self) -> Self {
        self.simd_lt_backend(rhs)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpLt<T> for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_lt(self, rhs: T) -> Self::Output {
        self.simd_lt(Self::splat(rhs))
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpGt for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_gt(self, rhs: Self) -> Self {
        self.simd_gt_backend(rhs)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpGt<T> for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_gt(self, rhs: T) -> Self::Output {
        self.simd_gt(Self::splat(rhs))
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpLe for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_le(self, rhs: Self) -> Self {
        self.simd_le_backend(rhs)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpLe<T> for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_le(self, rhs: T) -> Self::Output {
        self.simd_le(Self::splat(rhs))
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpGe for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_ge(self, rhs: Self) -> Self {
        self.simd_ge_backend(rhs)
    }
}

#[expect(deprecated)]
impl<T, const N: usize> CmpGe<T> for Simd<T, N>
where
    T: SimdElement + PartialOrd,
    Simd<T, N>: SupportedSimd,
{
    type Output = Self;

    #[inline]
    fn simd_ge(self, rhs: T) -> Self::Output {
        self.simd_ge(Self::splat(rhs))
    }
}

impl<T, const N: usize> AlignTo for Simd<T, N>
where
    T: SimdElement + Pod + Default + PartialEq,
    Simd<T, N>: SupportedSimd,
{
    type Elem = T;
}

/// The following functionality exists for all SIMD vectors.
impl<T, const N: usize> Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
{
    /// Converts an array to a SIMD vector.
    #[inline]
    #[must_use]
    pub const fn from_array(array: [T; N]) -> Self {
        unsafe { core::mem::transmute::<[T; N], Simd<T, N>>(array) }
    }

    /// Constructs a new SIMD vector with all elements set to the given value.
    #[inline]
    #[must_use]
    pub const fn splat(elem: T) -> Self {
        unsafe { core::mem::transmute::<[T; N], Simd<T, N>>([elem; N]) }
    }

    /// Converts a SIMD vector to an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [T; N] {
        // SAFETY: Both types accept all bit-patterns and only contain
        // initialized memory.
        unsafe { core::mem::transmute::<Simd<T, N>, [T; N]>(self) }
    }

    /// Returns an array reference containing the entire SIMD vector.
    #[inline]
    #[must_use]
    pub const fn as_array(&self) -> &[T; N] {
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { core::mem::transmute::<&Simd<T, N>, &[T; N]>(self) }
    }

    /// Returns a mutable array reference containing the entire SIMD vector.
    #[inline]
    #[must_use]
    pub const fn as_mut_array(&mut self) -> &mut [T; N] {
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { core::mem::transmute::<&mut Simd<T, N>, &mut [T; N]>(self) }
    }

    /// Returns a [mask] that checks if each element of `self` is equal to the
    /// corresponding element of `other`.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[expect(deprecated)]
    pub fn simd_eq<Rhs>(self, other: Rhs) -> <Self as CmpEq<Rhs>>::Output
    where
        T: PartialEq,
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
        T: PartialEq,
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
        T: PartialOrd,
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
        T: PartialOrd,
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
        T: PartialOrd,
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
        T: PartialOrd,
        Self: CmpGe<Rhs>,
    {
        CmpGe::simd_ge(self, other)
    }

    /// Returns `self` with the element at position `INDEX` replaced by
    /// `value`.
    ///
    /// If `INDEX` is out of bounds, compilation fails.
    #[doc(alias = "set", alias = "with")]
    #[inline]
    #[must_use]
    pub fn replace_const<const INDEX: usize>(self, value: T) -> Self {
        const {
            assert!(
                INDEX < N,
                "attempt to call `Simd::replace_const` with an out of bounds index"
            );
        }

        self.replace_const_backend::<INDEX>(value)
    }

    /// Returns the element of `self` at position `INDEX`.
    ///
    /// If `INDEX` is out of bounds, compilation fails.
    #[doc(alias = "get")]
    #[inline]
    #[must_use]
    pub fn extract_const<const INDEX: usize>(self) -> T {
        const {
            assert!(
                INDEX < N,
                "attempt to call `Simd::extract_const` with an out of bounds index"
            );
        }

        self.extract_const_backend::<INDEX>()
    }

    /// Lanewise SIMD selection.
    ///
    /// For each lane:
    ///
    /// - If all bits of `self` are one, return the corresponding lane of
    ///   `if_true`
    /// - If all bits of `self` are zero, return the corresponding lane of
    ///   `if_false`
    ///
    /// This function assumes `self` is a [mask], meaning each element is
    /// either all zeros or all ones. For bitwise selection use [`bitselect`].
    ///
    /// In addition to selecting SIMD vectors, this method can be used to
    /// select composite types (e.g., a downstream `Vec3<f32x4>` structure)
    /// if you implement `Select<Composite> for Simd`.
    ///
    /// [mask]: crate#masks
    /// [`bitselect`]: f32x4::bitselect
    #[inline]
    #[must_use]
    pub fn select<Output>(self, if_true: Output, if_false: Output) -> Output
    where
        T: MaskOps,
        Self: Select<Output>,
    {
        Select::select(self, if_true, if_false)
    }

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
    #[inline]
    #[must_use]
    #[doc(alias("movemask", "move_mask"))]
    pub fn to_bitmask(self) -> <Self as SupportedSimd>::Bitmask
    where
        T: MaskOps,
    {
        self.to_bitmask_backend()
    }

    /// Returns `true` if any element of `self` is true.
    ///
    /// This function assumes `self` is a [mask], meaning each element is
    /// either all zeros or all ones. If the input is not a mask, the result
    /// is unspecified.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn any(self) -> bool
    where
        T: MaskOps,
    {
        self.any_backend()
    }

    /// Returns `true` if all elements of `self` are true.
    ///
    /// This function assumes `self` is a [mask], meaning each element is
    /// either all zeros or all ones. If the input is not a mask, the result
    /// is unspecified.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn all(self) -> bool
    where
        T: MaskOps,
    {
        self.all_backend()
    }

    /// Returns `true` if none of the elements of `self` are true.
    ///
    /// This function assumes `self` is a [mask], meaning each element is
    /// either all zeros or all ones. If the input is not a mask, the result
    /// is unspecified.
    ///
    /// [mask]: crate#masks
    #[inline]
    #[must_use]
    pub fn none(self) -> bool
    where
        T: MaskOps,
    {
        !self.any()
    }

    /// Interleaves the lower halfs of two SIMD vectors, discarding the higher
    /// halfs.
    ///
    /// Equivalent to `[self[0], other[0], self[1], other[1], ...]`.
    #[inline]
    #[must_use]
    pub fn interleave_low(self, other: Self) -> Self {
        self.interleave_low_backend(other)
    }

    /// Interleaves the higher halfs of two SIMD vectors, discarding the lower
    /// halfs.
    ///
    /// Equivalent to
    /// `[self[N / 2], other[N / 2], self[N / 2 + 1], other[N / 2 + 1], ...]`.
    #[inline]
    #[must_use]
    pub fn interleave_high(self, other: Self) -> Self {
        self.interleave_high_backend(other)
    }

    /// Returns a SIMD vector whose elements are selected from `self` using
    /// the corresponding runtime `indices`.
    ///
    /// If an index is out of bounds, the corresponding result element is
    /// unspecified.
    ///
    /// Equivalent to
    /// `[self[indices[0]], self[indices[1]], ..., self[[indices[N - 1]]]]`.
    ///
    /// # Type-specific guarantees
    ///
    /// For [`u8x16`] and [`i8x16`], it is guaranteed that for out of bounds
    /// indices, if the high bit of an index is set, zero is returned.
    /// Otherwise, either zero is returned or the index wraps around,
    /// non-deterministically.
    ///
    /// For all other 8-bit types, it is guaranteed that for out of bounds
    /// indices, either zero is returned or the index wraps around,
    /// non-deterministically (unlike other types, which can return arbitrary
    /// values).
    #[inline]
    #[must_use]
    pub fn shuffle(self, indices: Simd<T::Unsigned, N>) -> Self
    where
        Simd<T::Unsigned, N>: SupportedSimd,
    {
        self.shuffle_backend(indices)
    }

    /// Returns a SIMD vector whose elements are selected from `self` using
    /// the corresponding runtime `indices`.
    ///
    /// If an index is out of bounds, the corresponding result element is
    /// the number zero.
    ///
    /// Equivalent to
    /// `[self[indices[0]], self[indices[1]], ..., self[[indices[N - 1]]]]`
    /// with a zero fallback.
    #[inline]
    #[must_use]
    pub fn shuffle_zeroing(self, indices: Simd<T::Unsigned, N>) -> Self
    where
        Simd<T::Unsigned, N>: SupportedSimd,
    {
        self.shuffle_zeroing_backend(indices)
    }

    /// Returns a SIMD vector whose elements are selected from `self` using
    /// the corresponding runtime `indices`.
    ///
    /// Indices are wrapped by the number of elements in `self`.
    ///
    /// Equivalent to
    /// `[self[indices[0] % N], self[indices[1] % N], ..., self[[indices[N - 1] % N]]]`.
    #[inline]
    #[must_use]
    pub fn shuffle_wrapping(self, indices: Simd<T::Unsigned, N>) -> Self
    where
        Simd<T::Unsigned, N>: SupportedSimd,
    {
        self.shuffle_wrapping_backend(indices)
    }

    /// Transposes an array of SIMD vectors interpreted as a square matrix.
    #[inline]
    #[must_use]
    pub fn transpose(data: [Self; N]) -> [Self; N] {
        Self::transpose_backend(data)
    }

    /// Converts an array to a SIMD vector.
    ///
    /// This function has been deprecated and renamed to [`from_array`].
    ///
    /// [`from_array`]: Self::from_array
    #[inline]
    #[must_use]
    #[deprecated(since = "1.8.0", note = "renamed to `from_array`")]
    pub const fn new(array: [T; N]) -> Self {
        Self::from_array(array)
    }

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
    /// [`bitselect`]: f32x4::bitselect
    #[deprecated(
        since = "1.6.0",
        note = "split into `select` and `bitselect` functions"
    )]
    #[inline]
    #[must_use]
    pub fn blend(self, if_true: Self, if_false: Self) -> Self
    where
        T: MaskOps,
    {
        self.select(if_true, if_false)
    }

    /// Interleaves the lower halfs of two SIMD vectors, discarding the higher
    /// halfs.
    ///
    /// Equivalent to `[self[0], other[0], self[1], other[1], ...]`.
    ///
    /// This function has been deprecated and renamed to [`interleave_low`].
    ///
    /// [`interleave_low`]: Self::interleave_low
    #[inline]
    #[must_use]
    #[deprecated(since = "1.8.0", note = "renamed to `interleave_low`")]
    pub fn unpack_lo(self, other: Self) -> Self {
        self.interleave_low(other)
    }

    /// Interleaves the higher halfs of two SIMD vectors, discarding the lower
    /// halfs.
    ///
    /// Equivalent to
    /// `[self[N / 2], other[N / 2], self[N / 2 + 1], other[N / 2 + 1], ...]`.
    ///
    /// This function has been deprecated and renamed to [`interleave_high`].
    ///
    /// [`interleave_high`]: Self::interleave_high
    #[inline]
    #[must_use]
    #[deprecated(since = "1.8.0", note = "renamed to `interleave_high`")]
    pub fn unpack_hi(self, other: Self) -> Self {
        self.interleave_high(other)
    }
}

// SAFETY: Currently, all SIMD types in this library contain fully initialized
// memory and accept all bits patterns.
unsafe impl<T, const N: usize> Zeroable for Simd<T, N>
where
    T: SimdElement + Zeroable,
    Simd<T, N>: SupportedSimd,
{
}

// SAFETY: Currently, all SIMD types in this library contain fully initialized
// memory and accept all bits patterns.
unsafe impl<T, const N: usize> Pod for Simd<T, N>
where
    T: SimdElement + Pod,
    Simd<T, N>: SupportedSimd,
{
}

#[cfg(feature = "serde")]
impl<T, const N: usize> Serialize for Simd<T, N>
where
    T: SimdElement + Serialize,
    Simd<T, N>: SupportedSimd,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde_core::Serializer,
    {
        let array = self.as_array();
        let mut seq = serializer.serialize_tuple(N)?;
        for e in array {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Deserialize<'de> for Simd<T, N>
where
    T: SimdElement + Deserialize<'de>,
    Simd<T, N>: SupportedSimd,
{
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_core::Deserializer<'de>,
    {
        use core::mem::MaybeUninit;

        use serde_core::de::Error;

        struct ArrayVisitor<T, const N: usize>(core::marker::PhantomData<T>);

        impl<'de, T, const N: usize> serde_core::de::Visitor<'de> for ArrayVisitor<T, N>
        where
            T: SimdElement + Deserialize<'de>,
        {
            type Value = [T; N];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("an array of length {N}")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<[T; N], A::Error>
            where
                A: serde_core::de::SeqAccess<'de>,
            {
                let mut result = [MaybeUninit::uninit(); N];

                for (index, element) in result.iter_mut().enumerate() {
                    *element.write(
                        seq.next_element()?
                            .ok_or_else(|| A::Error::invalid_length(index, &self))?,
                    );
                }

                // `<[MaybeUninit<T>; N]>::transpose` is unstable
                // SAFETY: These types represent the same thing
                let result =
                    unsafe { transmute_copy::<[MaybeUninit<T>; N], MaybeUninit<[T; N]>>(&result) };

                // SAFETY: If the function has not early-returned already, all
                // `result` slots have been written to and initialized.
                let result = unsafe { result.assume_init() };

                if seq.next_element::<T>()?.is_some() {
                    return Err(A::Error::invalid_length(N + 1, &self));
                }

                Ok(result)
            }
        }

        Ok(Self::from_array(deserializer.deserialize_tuple(
            N,
            ArrayVisitor::<T, N>(core::marker::PhantomData),
        )?))
    }
}

/// Automatically fills SIMD backend functions that have the same implementation
/// for all SIMD float types.
macro_rules! simd_backend_float_autofill {
    ($T:ident, $N:literal) => {
        #[inline]
        fn replace_const_backend<const INDEX: usize>(self, value: $T) -> Self {
            Self::from_bits(self.to_bits().replace_const::<INDEX>(value.to_bits()))
        }

        #[inline]
        fn extract_const_backend<const INDEX: usize>(self) -> $T {
            $T::from_bits(self.to_bits().extract_const::<INDEX>())
        }

        #[inline]
        fn shuffle_backend(self, indices: Simd<<$T as SimdElement>::Unsigned, $N>) -> Self {
            Self::from_bits(self.to_bits().shuffle(indices))
        }

        #[inline]
        fn shuffle_zeroing_backend(self, indices: Simd<<$T as SimdElement>::Unsigned, $N>) -> Self {
            Self::from_bits(self.to_bits().shuffle_zeroing(indices))
        }

        #[inline]
        fn shuffle_wrapping_backend(
            self,
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            Self::from_bits(self.to_bits().shuffle_wrapping(indices))
        }

        #[inline]
        fn shuffle_2_backend(
            inputs: [Self; 2],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 2], [Simd<<$T as SimdElement>::Unsigned, $N>; 2]>(inputs)
                    .shuffle(indices),
            )
        }

        #[inline]
        fn shuffle_zeroing_2_backend(
            inputs: [Self; 2],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 2], [Simd<<$T as SimdElement>::Unsigned, $N>; 2]>(inputs)
                    .shuffle_zeroing(indices),
            )
        }

        #[inline]
        fn shuffle_wrapping_2_backend(
            inputs: [Self; 2],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 2], [Simd<<$T as SimdElement>::Unsigned, $N>; 2]>(inputs)
                    .shuffle_wrapping(indices),
            )
        }

        #[inline]
        fn shuffle_3_backend(
            inputs: [Self; 3],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 3], [Simd<<$T as SimdElement>::Unsigned, $N>; 3]>(inputs)
                    .shuffle(indices),
            )
        }

        #[inline]
        fn shuffle_zeroing_3_backend(
            inputs: [Self; 3],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 3], [Simd<<$T as SimdElement>::Unsigned, $N>; 3]>(inputs)
                    .shuffle_zeroing(indices),
            )
        }

        #[inline]
        fn shuffle_wrapping_3_backend(
            inputs: [Self; 3],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 3], [Simd<<$T as SimdElement>::Unsigned, $N>; 3]>(inputs)
                    .shuffle_wrapping(indices),
            )
        }

        #[inline]
        fn shuffle_4_backend(
            inputs: [Self; 4],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 4], [Simd<<$T as SimdElement>::Unsigned, $N>; 4]>(inputs)
                    .shuffle(indices),
            )
        }

        #[inline]
        fn shuffle_zeroing_4_backend(
            inputs: [Self; 4],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 4], [Simd<<$T as SimdElement>::Unsigned, $N>; 4]>(inputs)
                    .shuffle_zeroing(indices),
            )
        }

        #[inline]
        fn shuffle_wrapping_4_backend(
            inputs: [Self; 4],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 4], [Simd<<$T as SimdElement>::Unsigned, $N>; 4]>(inputs)
                    .shuffle_wrapping(indices),
            )
        }
    };
}

/// Automatically fills SIMD backend functions that have the same implementation
/// for all SIMD signed integer types.
macro_rules! simd_backend_signed_autofill {
    ($T:ident, $N:literal) => {
        #[inline]
        fn simd_eq_backend(self, other: Self) -> Self {
            self.cast_unsigned()
                .simd_eq(other.cast_unsigned())
                .cast_signed()
        }

        #[inline]
        fn simd_ne_backend(self, other: Self) -> Self {
            self.cast_unsigned()
                .simd_ne(other.cast_unsigned())
                .cast_signed()
        }

        #[inline]
        fn replace_const_backend<const INDEX: usize>(self, value: $T) -> Self {
            self.cast_unsigned()
                .replace_const::<INDEX>(value.cast_unsigned())
                .cast_signed()
        }

        #[inline]
        fn extract_const_backend<const INDEX: usize>(self) -> $T {
            self.cast_unsigned().extract_const::<INDEX>().cast_signed()
        }

        #[inline]
        fn select_backend(self, if_true: Self, if_false: Self) -> Self {
            self.cast_unsigned()
                .select(if_true.cast_unsigned(), if_false.cast_unsigned())
                .cast_signed()
        }

        #[inline]
        fn to_bitmask_backend(self) -> <Self as SupportedSimd>::Bitmask {
            self.cast_unsigned().to_bitmask()
        }

        #[inline]
        fn any_backend(self) -> bool {
            self.cast_unsigned().any()
        }

        #[inline]
        fn all_backend(self) -> bool {
            self.cast_unsigned().all()
        }

        #[inline]
        fn interleave_low_backend(self, other: Self) -> Self {
            self.cast_unsigned()
                .interleave_low(other.cast_unsigned())
                .cast_signed()
        }

        #[inline]
        fn interleave_high_backend(self, other: Self) -> Self {
            self.cast_unsigned()
                .interleave_high(other.cast_unsigned())
                .cast_signed()
        }

        #[inline]
        fn shuffle_backend(self, indices: Simd<<$T as SimdElement>::Unsigned, $N>) -> Self {
            self.cast_unsigned().shuffle(indices).cast_signed()
        }

        #[inline]
        fn shuffle_zeroing_backend(self, indices: Simd<<$T as SimdElement>::Unsigned, $N>) -> Self {
            self.cast_unsigned().shuffle_zeroing(indices).cast_signed()
        }

        #[inline]
        fn shuffle_wrapping_backend(
            self,
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            self.cast_unsigned().shuffle_wrapping(indices).cast_signed()
        }

        #[inline]
        fn shuffle_2_backend(
            inputs: [Self; 2],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 2], [Simd<<$T as SimdElement>::Unsigned, $N>; 2]>(inputs)
                    .shuffle(indices),
            )
        }

        #[inline]
        fn shuffle_zeroing_2_backend(
            inputs: [Self; 2],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 2], [Simd<<$T as SimdElement>::Unsigned, $N>; 2]>(inputs)
                    .shuffle_zeroing(indices),
            )
        }

        #[inline]
        fn shuffle_wrapping_2_backend(
            inputs: [Self; 2],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 2], [Simd<<$T as SimdElement>::Unsigned, $N>; 2]>(inputs)
                    .shuffle_wrapping(indices),
            )
        }

        #[inline]
        fn shuffle_3_backend(
            inputs: [Self; 3],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 3], [Simd<<$T as SimdElement>::Unsigned, $N>; 3]>(inputs)
                    .shuffle(indices),
            )
        }

        #[inline]
        fn shuffle_zeroing_3_backend(
            inputs: [Self; 3],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 3], [Simd<<$T as SimdElement>::Unsigned, $N>; 3]>(inputs)
                    .shuffle_zeroing(indices),
            )
        }

        #[inline]
        fn shuffle_wrapping_3_backend(
            inputs: [Self; 3],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 3], [Simd<<$T as SimdElement>::Unsigned, $N>; 3]>(inputs)
                    .shuffle_wrapping(indices),
            )
        }

        #[inline]
        fn shuffle_4_backend(
            inputs: [Self; 4],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 4], [Simd<<$T as SimdElement>::Unsigned, $N>; 4]>(inputs)
                    .shuffle(indices),
            )
        }

        #[inline]
        fn shuffle_zeroing_4_backend(
            inputs: [Self; 4],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 4], [Simd<<$T as SimdElement>::Unsigned, $N>; 4]>(inputs)
                    .shuffle_zeroing(indices),
            )
        }

        #[inline]
        fn shuffle_wrapping_4_backend(
            inputs: [Self; 4],
            indices: Simd<<$T as SimdElement>::Unsigned, $N>,
        ) -> Self {
            cast(
                cast::<[Simd<$T, $N>; 4], [Simd<<$T as SimdElement>::Unsigned, $N>; 4]>(inputs)
                    .shuffle_wrapping(indices),
            )
        }

        #[inline]
        fn transpose_backend(data: [Self; $N]) -> [Self; $N] {
            cast(Simd::<<$T as SimdElement>::Unsigned, $N>::transpose(
                cast::<[Simd<$T, $N>; $N], [Simd<<$T as SimdElement>::Unsigned, $N>; $N]>(data),
            ))
        }
    };
}

impl<T, const N: usize> SimdAssociatedTypes for Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
{
    type T = T;
    type ArrayOf<U> = [U; N];
    type SimdOf<U>
        = <Self as SupportSimdOf<U>>::SimdOfU
    where
        Self: SupportSimdOf<U>;

    type UnsignedT = T::Unsigned;
    type UnsignedSimd
        = <Self as SupportSimdOf<T::Unsigned>>::SimdOfU
    where
        Self: SupportSimdOf<T::Unsigned>;

    type SignedT = T::Signed;
    type SignedSimd
        = <Self as SupportSimdOf<T::Signed>>::SimdOfU
    where
        Self: SupportSimdOf<T::Signed>;
}

impl<T, const N: usize, U> SupportSimdOf<U> for Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
    U: SimdElement,
    Simd<U, N>: SupportedSimd,
{
    type SimdOfU = Simd<U, N>;
}

impl Sealed for f32 {}
impl Sealed for f64 {}
impl Sealed for i8 {}
impl Sealed for i16 {}
impl Sealed for i32 {}
impl Sealed for i64 {}
impl Sealed for i128 {}
impl Sealed for isize {}
impl Sealed for u8 {}
impl Sealed for u16 {}
impl Sealed for u32 {}
impl Sealed for u64 {}
impl Sealed for u128 {}
impl Sealed for usize {}

impl<T, const N: usize> Sealed for [T; N] {}

impl<T, const N: usize> Sealed for Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: SupportedSimd,
{
}

define_simd!(
    T = f32,
    N = 4,
    N_STR = "four",
    Simd = f32x4,
    UnsignedSimd = u32x4,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128 },
    optional_type_arm_inner { ArmInner = float32x4_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = f32,
    N = 8,
    N_STR = "eight",
    Simd = f32x8,
    UnsignedSimd = u32x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256 },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = f32,
    N = 16,
    N_STR = "16",
    Simd = f32x16,
    UnsignedSimd = u32x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512 },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = f64,
    N = 2,
    N_STR = "two",
    Simd = f64x2,
    UnsignedSimd = u64x2,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128d },
    optional_type_arm_inner { ArmInner = float64x2_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = f64,
    N = 4,
    N_STR = "four",
    Simd = f64x4,
    UnsignedSimd = u64x4,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256d },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = f64,
    N = 8,
    N_STR = "eight",
    Simd = f64x8,
    UnsignedSimd = u64x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512d },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i8,
    N = 16,
    N_STR = "16",
    Simd = i8x16,
    UnsignedSimd = u8x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = int8x16_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = i8,
    N = 32,
    N_STR = "32",
    Simd = i8x32,
    UnsignedSimd = u8x32,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i8,
    N = 64,
    N_STR = "64",
    Simd = i8x64,
    UnsignedSimd = u8x64,
    Bitmask = u64,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i16,
    N = 8,
    N_STR = "eight",
    Simd = i16x8,
    UnsignedSimd = u16x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = int16x8_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = i16,
    N = 16,
    N_STR = "16",
    Simd = i16x16,
    UnsignedSimd = u16x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i16,
    N = 32,
    N_STR = "32",
    Simd = i16x32,
    UnsignedSimd = u16x32,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i32,
    N = 4,
    N_STR = "four",
    Simd = i32x4,
    UnsignedSimd = u32x4,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = int32x4_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = i32,
    N = 8,
    N_STR = "eight",
    Simd = i32x8,
    UnsignedSimd = u32x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i32,
    N = 16,
    N_STR = "16",
    Simd = i32x16,
    UnsignedSimd = u32x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i64,
    N = 2,
    N_STR = "two",
    Simd = i64x2,
    UnsignedSimd = u64x2,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = int64x2_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = i64,
    N = 4,
    N_STR = "four",
    Simd = i64x4,
    UnsignedSimd = u64x4,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = i64,
    N = 8,
    N_STR = "eight",
    Simd = i64x8,
    UnsignedSimd = u64x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u8,
    N = 16,
    N_STR = "16",
    Simd = u8x16,
    UnsignedSimd = u8x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = uint8x16_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = u8,
    N = 32,
    N_STR = "32",
    Simd = u8x32,
    UnsignedSimd = u8x32,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u8,
    N = 64,
    N_STR = "64",
    Simd = u8x64,
    UnsignedSimd = u8x64,
    Bitmask = u64,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u16,
    N = 8,
    N_STR = "eight",
    Simd = u16x8,
    UnsignedSimd = u16x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = uint16x8_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = u16,
    N = 16,
    N_STR = "16",
    Simd = u16x16,
    UnsignedSimd = u16x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u16,
    N = 32,
    N_STR = "32",
    Simd = u16x32,
    UnsignedSimd = u16x32,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u32,
    N = 4,
    N_STR = "four",
    Simd = u32x4,
    UnsignedSimd = u32x4,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = uint32x4_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = u32,
    N = 8,
    N_STR = "eight",
    Simd = u32x8,
    UnsignedSimd = u32x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u32,
    N = 16,
    N_STR = "16",
    Simd = u32x16,
    UnsignedSimd = u32x16,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u64,
    N = 2,
    N_STR = "two",
    Simd = u64x2,
    UnsignedSimd = u64x2,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m128i },
    optional_type_arm_inner { ArmInner = uint64x2_t },
    optional_type_wasm_inner { WasmInner = v128 },
);
define_simd!(
    T = u64,
    N = 4,
    N_STR = "four",
    Simd = u64x4,
    UnsignedSimd = u64x4,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m256i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
define_simd!(
    T = u64,
    N = 8,
    N_STR = "eight",
    Simd = u64x8,
    UnsignedSimd = u64x8,
    Bitmask = u32,
    optional_type_x86_inner { X86Inner = __m512i },
    optional_type_arm_inner {},
    optional_type_wasm_inner {},
);
