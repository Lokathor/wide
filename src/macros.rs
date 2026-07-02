macro_rules! integer_fn_saturating_div {
  ([$($index:literal),* $(,)?] $(,)?) => {
    /// Lanewise saturating divide.
    ///
    /// Note that because division has no hardware support, this operation is
    /// very slow and should be avoided if possible.
    #[inline]
    #[must_use]
    pub fn saturating_div(self, rhs: Self) -> Self {
      let self_array = self.to_array();
      let rhs_array = rhs.to_array();

      Self::new([$(self_array[$index].saturating_div(rhs_array[$index])),*])
    }
  };
}
