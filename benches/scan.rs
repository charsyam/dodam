use std::fs::File;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dodam::engine::DodamEngine;
use dodam::execution::{FilterExpr, Projection};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn bench_scan(c: &mut Criterion) {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let files = [
        ("uncompressed", Compression::UNCOMPRESSED),
        ("snappy", Compression::SNAPPY),
        ("zstd", Compression::ZSTD(Default::default())),
    ]
    .into_iter()
    .map(|(name, compression)| {
        let path = tempdir.path().join(format!("bench-{name}.parquet"));
        write_bench_parquet(&path, 1_048_576, 65_536, compression);
        (name, path)
    })
    .collect::<Vec<_>>();

    let runtime = Runtime::new().expect("tokio runtime");
    let engine = DodamEngine::default();

    let mut group = c.benchmark_group("scan");
    group.sample_size(30);
    for (compression, path) in files {
        group.bench_with_input(BenchmarkId::new("full", compression), &path, |b, path| {
            b.iter(|| {
                runtime
                    .block_on(engine.scan_parquet(path.clone(), 8192, None, Projection::All, None))
                    .expect("full scan")
            })
        });
        group.bench_with_input(
            BenchmarkId::new("projected", compression),
            &path,
            |b, path| {
                b.iter(|| {
                    runtime
                        .block_on(engine.scan_parquet(
                            path.clone(),
                            8192,
                            None,
                            Projection::Columns(vec!["id".to_string()]),
                            None,
                        ))
                        .expect("projected scan")
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("filtered", compression),
            &path,
            |b, path| {
                b.iter(|| {
                    runtime
                        .block_on(engine.scan_parquet(
                            path.clone(),
                            8192,
                            None,
                            Projection::Columns(vec!["payload".to_string()]),
                            Some(FilterExpr::parse("id>=200000 AND id<200010").expect("filter")),
                        ))
                        .expect("filtered scan")
                })
            },
        );
    }
    group.finish();

    keep_tempdir_alive(tempdir);
}

fn write_bench_parquet(
    path: &std::path::Path,
    rows: usize,
    row_group_size: usize,
    compression: Compression,
) {
    let mut fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ];
    fields.extend((0..8).map(|index| Field::new(format!("i32_{index}"), DataType::Int32, false)));
    fields.extend((0..3).map(|index| Field::new(format!("i64_{index}"), DataType::Int64, false)));
    fields.extend((0..4).map(|index| Field::new(format!("f64_{index}"), DataType::Float64, false)));
    let schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("row-{value}")),
        )),
    ];
    columns.extend((0..8).map(|column| {
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(move |value| (value as i32).wrapping_add(column * 17)),
        )) as ArrayRef
    }));
    columns.extend((0..3).map(|column| {
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(move |value| value as i64 * 31 + column as i64),
        )) as ArrayRef
    }));
    columns.extend((0..4).map(|column| {
        Arc::new(Float64Array::from_iter_values(
            (0..rows).map(move |value| value as f64 * 0.125 + column as f64),
        )) as ArrayRef
    }));
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("record batch");

    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_size))
        .set_compression(compression)
        .build();
    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn keep_tempdir_alive(_tempdir: TempDir) {}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
