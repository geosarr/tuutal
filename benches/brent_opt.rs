#![feature(test)]
extern crate test;
extern crate tuutal;

use test::Bencher;
use tuutal::brent_opt;

#[bench]
fn no_brent_opt_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = brent_opt(|x: f32| x, -1., 1., 10000, 1e-5);
    });
}

#[bench]
fn easy_brent_opt_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = brent_opt(|x: f32| x.powi(2), -1., 1., 10000, 1e-5);
    });
}
