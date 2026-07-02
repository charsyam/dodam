use std::fs::File;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dodam::engine::{DodamEngine, JoinAlgorithm, JoinParquetRequest};
use dodam::execution::{AggregateExpr, JoinType, Projection};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn bench_olap(c: &mut Criterion) {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let aggregate_path = tempdir.path().join("aggregate.parquet");
    let left_path = tempdir.path().join("join-left.parquet");
    let right_path = tempdir.path().join("join-right.parquet");
    write_aggregate_parquet(&aggregate_path, 1_048_576, 65_536);
    write_join_left_parquet(&left_path, 32_768, 8_192);
    write_join_right_parquet(&right_path, 16_384, 8_192);

    let runtime = Runtime::new().expect("tokio runtime");
    let engine = DodamEngine::default();

    let mut aggregate = c.benchmark_group("aggregate");
    aggregate.sample_size(10);
    aggregate.warm_up_time(Duration::from_secs(1));
    aggregate.measurement_time(Duration::from_secs(3));
    aggregate.bench_function("global/count_sum_avg_min_max", |b| {
        b.iter(|| {
            runtime
                .block_on(engine.aggregate_parquet(
                    aggregate_path.clone(),
                    8192,
                    vec![
                        AggregateExpr::CountStar,
                        AggregateExpr::Sum("value".to_string()),
                        AggregateExpr::Avg("value".to_string()),
                        AggregateExpr::Min("payload".to_string()),
                        AggregateExpr::Max("payload".to_string()),
                    ],
                    None,
                ))
                .expect("global aggregate")
        })
    });
    aggregate.bench_function("grouped/1024_groups", |b| {
        b.iter(|| {
            runtime
                .block_on(engine.aggregate_parquet_grouped(
                    aggregate_path.clone(),
                    8192,
                    vec![
                        AggregateExpr::CountStar,
                        AggregateExpr::Sum("value".to_string()),
                    ],
                    vec!["bucket".to_string()],
                    None,
                ))
                .expect("grouped aggregate")
        })
    });
    aggregate.finish();

    let mut join = c.benchmark_group("join");
    join.sample_size(10);
    join.warm_up_time(Duration::from_secs(1));
    join.measurement_time(Duration::from_secs(3));
    for (name, memory_limit, join_type) in [
        ("hash_inner", u64::MAX, JoinType::Inner),
        ("partitioned_inner", 256 * 1024, JoinType::Inner),
        ("hash_full", u64::MAX, JoinType::Full),
        ("hash_semi", u64::MAX, JoinType::Semi),
        ("partitioned_semi", 256 * 1024, JoinType::Semi),
    ] {
        join.bench_with_input(
            BenchmarkId::new(name, "32k_x_16k"),
            &(memory_limit, join_type),
            |b, (memory_limit, join_type)| {
                b.iter(|| {
                    let stream = runtime
                        .block_on(engine.join_parquet_batches(JoinParquetRequest {
                            left_path: left_path.clone(),
                            right_path: right_path.clone(),
                            batch_size: 8192,
                            left_keys: vec!["key".to_string()],
                            right_keys: vec!["key".to_string()],
                            left_prefix: "l".to_string(),
                            right_prefix: "r".to_string(),
                            left_projection: Projection::Columns(vec![
                                "id".to_string(),
                                "key".to_string(),
                            ]),
                            right_projection: Projection::Columns(vec![
                                "key".to_string(),
                                "payload".to_string(),
                            ]),
                            left_filter: None,
                            right_filter: None,
                            output_projection: Projection::All,
                            join_memory_limit_bytes: *memory_limit,
                            join_algorithm: JoinAlgorithm::Auto,
                            join_type: *join_type,
                        }))
                        .expect("join batches");
                    stream.collect::<dodam::error::Result<Vec<_>>>()
                })
            },
        );
    }
    join.finish();

    keep_tempdir_alive(tempdir);
}

fn write_aggregate_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("bucket", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("g{:04}", value % 1024)),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|value| value as i64 * 7),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("payload-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_join_left_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_join_right_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("right-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_parquet(
    path: &std::path::Path,
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    row_group_size: usize,
) {
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("record batch");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_size))
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn keep_tempdir_alive(_tempdir: TempDir) {}

criterion_group!(benches, bench_olap);
criterion_main!(benches);
