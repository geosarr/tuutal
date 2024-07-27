use crate::Array1;

pub(crate) fn l2_diff<T>(a: &Array1<T>, b: &Array1<T>) -> T
where
    for<'a> &'a T: std::ops::Sub<Output = T>,
    T: num_traits::Float + std::ops::Mul<Output = T> + std::iter::Sum,
{
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<T>()
        .sqrt()
}

pub(crate) fn is_between<T>(arr: &Array1<T>, lower: &Array1<T>, upper: &Array1<T>) -> bool
where
    for<'a> &'a T: PartialOrd,
{
    arr.iter()
        .zip(lower)
        .zip(upper)
        .all(|((x, l), u)| (l <= x) && (x <= u))
}
