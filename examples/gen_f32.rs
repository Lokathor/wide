#![allow(dead_code)]

fn main() {
  let mut buffer = String::new();
  //
  //write_consts(&mut buffer);
  //write_math_op_impls(&mut buffer);
  //write_formatting(&mut buffer);
  //write_from_impls(&mut buffer);
  write_methods(&mut buffer);
  //
  use std::{fs::File, io::prelude::*};
  let mut file =
    File::create("target/gen_f32_out.rs").expect("couldn't make file");
  file.write_all(buffer.as_bytes()).expect("couldn't write all the data.");
}

fn write_consts(buf: &mut String) {
  // f32 in std::f32
  for const_name in
    ["EPSILON", "INFINITY", "MAX", "MIN", "MIN_POSITIVE", "NAN", "NEG_INFINITY"]
      .iter()
  {
    buf.push_str(&format!(
      "pub const {}: [f32;4] = [std::f32::{},std::f32::{},std::f32::{},std::f32::{}];\n",
      const_name, const_name, const_name, const_name, const_name
    ));
  }

  // u32 in std::f32
  for const_name in ["DIGITS", "MANTISSA_DIGITS", "RADIX"].iter() {
    buf.push_str(&format!(
      "pub const {}: [u32;4] = [std::f32::{},std::f32::{},std::f32::{},std::f32::{}];\n",
      const_name, const_name, const_name, const_name, const_name
    ));
  }

  // i32 in std::f32
  for const_name in ["MAX_10_EXP", "MAX_EXP", "MIN_10_EXP", "MIN_EXP"].iter() {
    buf.push_str(&format!(
      "pub const {}: [i32;4] = [std::f32::{},std::f32::{},std::f32::{},std::f32::{}];\n",
      const_name, const_name, const_name, const_name, const_name
    ));
  }

  // std::f32::consts, because of idiot design
  for const_name in [
    "E",
    "FRAC_1_PI",
    "FRAC_2_PI",
    "FRAC_2_SQRT_PI",
    "FRAC_1_SQRT_2",
    "FRAC_PI_2",
    "FRAC_PI_3",
    "FRAC_PI_4",
    "FRAC_PI_6",
    "FRAC_PI_8",
    "LN_2",
    "LN_10",
    "LOG2_E",
    "LOG10_E",
    "PI",
    "SQRT_2",
  ]
  .iter()
  {
    buf.push_str(&format!("pub const {}: [f32;4] = [std::f32::consts::{},std::f32::consts::{},std::f32::consts::{},std::f32::consts::{}];\n", const_name, const_name, const_name, const_name, const_name));
  }
}

fn write_math_op_impls(buf: &mut String) {
  // self, rhs
  for (trait_name, method_name, symbol) in [
    ("Add", "add", "+"),
    ("Div", "div", "/"),
    ("Mul", "mul", "*"),
    ("Rem", "rem", "%"), // TODO: rem isn't sse compatible
    ("Sub", "sub", "-"),
  ]
  .iter()
  {
    // op to same-owned
    buf.push_str(&format!(
      r#"
    impl {} for f32x4 {{
      type Output = Self;
      fn {}(self, rhs: Self) -> Self {{
        cfg_block! {{if #[cfg(target_feature="sse")] {{
          Self {{ sse: self.sse.{}(rhs.sse) }}
        }} else {{
          Self {{ arr: [
            self.arr[0] {} rhs.arr[0],
            self.arr[1] {} rhs.arr[1],
            self.arr[2] {} rhs.arr[2],
            self.arr[3] {} rhs.arr[3],
          ] }}
        }}}}
      }}
    }}
    "#,
      trait_name, method_name, method_name, symbol, symbol, symbol, symbol
    ));
    // op to same-ref
    buf.push_str(&format!(
      r#"
    impl {}<&'_ f32x4> for f32x4 {{
      type Output = Self;
      fn {}(self, rhs: &Self) -> Self {{
        cfg_block! {{if #[cfg(target_feature="sse")] {{
          Self {{ sse: self.sse.{}(rhs.sse) }}
        }} else {{
          Self {{ arr: [
            self.arr[0] {} rhs.arr[0],
            self.arr[1] {} rhs.arr[1],
            self.arr[2] {} rhs.arr[2],
            self.arr[3] {} rhs.arr[3],
          ] }}
        }}}}
      }}
    }}
    "#,
      trait_name, method_name, method_name, symbol, symbol, symbol, symbol
    ));
  }

  // &mut self, rhs
  for (trait_name, method_name, symbol) in [
    ("AddAssign", "add_assign", "+"),
    ("DivAssign", "div_assign", "/"),
    ("MulAssign", "mul_assign", "*"),
    ("RemAssign", "rem_assign", "%"),
    ("SubAssign", "sub_assign", "-"),
  ]
  .iter()
  {
    // op to same-owned
    buf.push_str(&format!(
      r#"
    impl {} for f32x4 {{
      fn {}(&mut self, rhs: Self) {{
        *self = *self {} rhs
      }}
    }}
    "#,
      trait_name, method_name, symbol
    ));
    // op to same-ref
    buf.push_str(&format!(
      r#"
    impl {}<&'_ f32x4> for f32x4 {{
      fn {}(&mut self, rhs: &Self) {{
        *self = *self {} rhs
      }}
    }}
    "#,
      trait_name, method_name, symbol
    ));
  }
}

fn write_formatting(buf: &mut String) {
  for trait_name in ["Debug", "Display", "LowerExp", "UpperExp"].iter() {
    buf.push_str(&format!(
      r#"
    impl {} for f32x4 {{
      fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {{
        write!(f, "f32x4(")?;
        {}::fmt(&self[0], f)?;
        {}::fmt(&self[1], f)?;
        {}::fmt(&self[2], f)?;
        {}::fmt(&self[3], f)?;
        write!(f, ")")
      }}
    }}
    "#,
      trait_name, trait_name, trait_name, trait_name, trait_name
    ));
  }
}

fn write_from_impls(buf: &mut String) {
  for src_type in ["i8", "u8", "i16", "u16"].iter() {
    buf.push_str(&format!(
      r#"
    impl From<[{}; 4]> for f32x4 {{
      fn from([a,b,c,d]: [{}; 4]) -> Self {{
        Self::new(
          f32::from(a),
          f32::from(b),
          f32::from(c),
          f32::from(d),
        )
      }}
    }}
    "#,
      src_type, src_type
    ));
  }
}

fn write_methods(buf: &mut String) {
  for unary_func in [
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cbrt",
    "ceil",
    "classify",
    "cos",
    "cosh",
    "exp",
    "exp2",
    "exp_m1",
    "floor",
    "fract",
    "ln",
    "ln_1p",
    "log10",
    "log2",
    "recip",
    "round",
    "signum",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "to_degrees",
    "to_radians",
    "trunc",
  ]
  .iter()
  {
    buf.push_str(&format!(
      r#"
    #[inline]
    pub fn {name}(self) -> Self {{
      let a: [f32; 4] = cast(self);
      cast([a[0].{name}(),a[1].{name}(),a[2].{name}(),a[3].{name}()])
    }}
    "#,
      name = unary_func
    ));
  }

  for binary_func in
    ["atan2", "copysign", "hypot", "log", "max", "min", "powf", "powi"].iter()
  {
    buf.push_str(&format!(
      r#"
    #[inline]
    pub fn {name}(self, b: Self) -> Self {{
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      cast([a[0].{name}(b[0]),a[1].{name}(b[1]),a[2].{name}(b[2]),a[3].{name}(b[3])])
    }}
    "#,
      name = binary_func
    ));
  }

  for trinary_func in ["mul_add"].iter() {
    buf.push_str(&format!(r#"
    #[inline]
    pub fn {name}(self, b: Self, c: Self) -> Self {{
      let a: [f32; 4] = cast(self);
      let b: [f32; 4] = cast(b);
      let c: [f32; 4] = cast(c);
      cast([a[0].{name}(b[0], c[0]),a[1].{name}(b[1], c[0]),a[2].{name}(b[2], c[0]),a[3].{name}(b[3], c[0])])
    }}
    "#,name = trinary_func));
  }

  for tuple_output in ["sin_cos"].iter() {
    buf.push_str(&format!(
      r#"
    #[inline]
    pub fn {name}(self) -> (Self, Self) {{
      let a: [f32; 4] = cast(self);
      let (zero_sin, zero_cos) = a[0];
      let (one_sin, one_cos) = a[1];
      let (two_sin, two_cos) = a[2];
      let (three_sin, three_cos) = a[3];
      (cast([zero_sin, one_sin, two_sin, three_sin]),cast([zero_cos, one_cos, two_cos, three_cos]))
    }}
    "#,
      name = tuple_output
    ));
  }
}
