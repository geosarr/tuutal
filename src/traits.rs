/// To initialize some variables.
pub trait DefaultValue {
    fn tol(n: i32) -> Self;
    fn one_half() -> Self;
    fn one() -> Self;
}

macro_rules! impl_default_value(
    ( $( $t:ident ),* )=> {
        $(
            impl DefaultValue for $t {
              fn tol(n: i32) -> $t {
                (10 as $t).powi(n)
              }
              fn one_half() -> $t{
                0.5 as $t
              }
              fn one() -> $t {
                1 as $t
              }
            }
        )*
    }
);

impl_default_value!(f32, f64);
