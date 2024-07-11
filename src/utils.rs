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
