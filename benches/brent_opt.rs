#![feature(test)]
extern crate test;
extern crate tuutal;

use test::Bencher;
use tuutal::{brent_bounded, brent_unbounded};

#[bench]
fn no_brent_unbounded_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = brent_unbounded(|x: f32| x, Some(&[-1., 1.]), 10000, 1e-5);
    });
}

#[bench]
fn easy_brent_unbounded_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = brent_unbounded(|x: f32| x.powi(2), Some(&[-1., 1.]), 10000, 1e-5);
    });
}

#[bench]
fn brent_bounded_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = brent_bounded(|x: f32| x.powi(2), (-1., 1.), 1e-5, 10000);
    });
}
