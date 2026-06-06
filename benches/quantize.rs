//! Criterion benches for the iqdb-quantize public surface.
//!
//! Four samples:
//!
//! - `sq8::quantize` — encode a single `f32` vector to SQ8.
//! - `sq8::distance_cosine` — asymmetric SQ8 distance against an f32 query.
//! - `bq::quantize` — encode a single `f32` vector to BQ.
//! - `bq::hamming` — packed-XOR Hamming on a stored BQ code.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use iqdb_quantize::{BinaryQuantizer, Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;

const DIM: usize = 768;

fn corpus_pair() -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..DIM).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..DIM).map(|i| (i as f32).cos()).collect();
    (a, b)
}

fn bench_sq8(c: &mut Criterion) {
    let (a, b) = corpus_pair();
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&a[..], &b[..]]).expect("training inputs valid");
    let code = sq.quantize(&a).expect("dim matches training");
    let query = b.clone();

    let mut group = c.benchmark_group("sq8");
    group.bench_function("quantize", |bench| {
        bench.iter(|| {
            let out = sq.quantize(black_box(&a)).expect("dim matches");
            black_box(out)
        });
    });
    group.bench_function("distance_cosine", |bench| {
        bench.iter(|| {
            let d = sq
                .distance(black_box(&query), black_box(&code), DistanceMetric::Cosine)
                .expect("dim matches");
            black_box(d)
        });
    });
    group.finish();
}

fn bench_bq(c: &mut Criterion) {
    let (a, b) = corpus_pair();
    let mut bq = BinaryQuantizer::new();
    bq.train(&[&a[..], &b[..]]).expect("training inputs valid");
    let code = bq.quantize(&a).expect("dim matches training");
    let query = b.clone();

    let mut group = c.benchmark_group("bq");
    group.bench_function("quantize", |bench| {
        bench.iter(|| {
            let out = bq.quantize(black_box(&a)).expect("dim matches");
            black_box(out)
        });
    });
    group.bench_function("hamming", |bench| {
        bench.iter(|| {
            let d = bq
                .distance(black_box(&query), black_box(&code), DistanceMetric::Hamming)
                .expect("dim matches");
            black_box(d)
        });
    });
    group.finish();
}

criterion_group!(quantize, bench_sq8, bench_bq);
criterion_main!(quantize);
