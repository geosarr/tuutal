use crate::VecType;

pub(crate) fn l2_diff<T>(a: &VecType<T>, b: &VecType<T>) -> T
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

pub(crate) fn is_between<T>(arr: &VecType<T>, lower: &VecType<T>, upper: &VecType<T>) -> bool
where
    for<'a> &'a T: PartialOrd,
{
    arr.iter()
        .zip(lower)
        .zip(upper)
        .all(|((x, l), u)| (l <= x) && (x <= u))
}
