#![feature(test)]
extern crate test;
extern crate tuutal;

use test::Bencher;
use tuutal::bracket;

#[bench]
fn no_bracket_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = bracket(|x: f32| x, -1., 1., 110., 10000);
    });
}

#[bench]
fn easy_bracket_bench(bench: &mut Bencher) {
    bench.iter(|| {
        let _solution = bracket(|x: f32| x.powi(2), -1., 1., 110., 10000);
    });
}
