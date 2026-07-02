use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dodam::engine::DodamEngine;
use dodam::sql::{QueryOutput, execute_sql};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::runtime::Runtime;

const DODAM_BENCH_BATCH_SIZE: usize = 16 * 1024;

fn bench_duckdb_e2e(c: &mut Criterion) {
    if Command::new("duckdb").arg("--version").output().is_err() {
        eprintln!("duckdb binary not found; skipping duckdb_e2e benchmark");
        return;
    }
    let dodam_bin = ensure_dodam_release_binary();
    let output_format = bench_copy_format();

    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let dim_path = tempdir.path().join("dim.parquet");
    let sparse_facts_path = tempdir.path().join("sparse-facts.parquet");
    let sparse_dim_path = tempdir.path().join("sparse-dim.parquet");
    let duplicate_dim_path = tempdir.path().join("duplicate-dim.parquet");
    let narrow_facts_path = tempdir.path().join("narrow-facts.parquet");
    let wide_pruned_facts_path = tempdir.path().join("wide-pruned-facts.parquet");
    let small_rg_facts_path = tempdir.path().join("small-rg-facts.parquet");
    let multi_facts_path = tempdir.path().join("multi-facts.parquet");
    let multi_dim_path = tempdir.path().join("multi-dim.parquet");
    let string_facts_path = tempdir.path().join("string-facts.parquet");
    let string_dim_path = tempdir.path().join("string-dim.parquet");
    let unmatched_facts_path = tempdir.path().join("unmatched-facts.parquet");
    let unmatched_dim_path = tempdir.path().join("unmatched-dim.parquet");
    let wide_facts_path = tempdir.path().join("wide-facts.parquet");
    let wide_dim_path = tempdir.path().join("wide-dim.parquet");
    let output_dir = tempdir.path().join("outputs");
    std::fs::create_dir(&output_dir).expect("create benchmark output dir");
    write_facts_parquet(&facts_path, 262_144, 16_384);
    write_dim_parquet(&dim_path, 65_536, 16_384);
    write_sparse_facts_parquet(&sparse_facts_path, 262_144, 16_384);
    write_sparse_dim_parquet(&sparse_dim_path, 65_536, 16_384);
    write_duplicate_dim_parquet(&duplicate_dim_path, 131_072, 16_384);
    write_narrow_facts_parquet(&narrow_facts_path, 262_144, 16_384);
    write_wide_pruned_facts_parquet(&wide_pruned_facts_path, 262_144, 16_384);
    write_facts_parquet(&small_rg_facts_path, 262_144, 4_096);
    write_multi_facts_parquet(&multi_facts_path, 262_144, 16_384);
    write_multi_dim_parquet(&multi_dim_path, 65_536, 16_384);
    write_string_facts_parquet(&string_facts_path, 131_072, 16_384);
    write_string_dim_parquet(&string_dim_path, 32_768, 16_384);
    write_unmatched_facts_parquet(&unmatched_facts_path, 131_072, 16_384);
    write_unmatched_dim_parquet(&unmatched_dim_path, 32_768, 16_384);
    write_wide_facts_parquet(&wide_facts_path, 131_072, 16_384);
    write_wide_dim_parquet(&wide_dim_path, 32_768, 16_384);

    let facts = facts_path.display().to_string();
    let dim = dim_path.display().to_string();
    let sparse_facts = sparse_facts_path.display().to_string();
    let sparse_dim = sparse_dim_path.display().to_string();
    let duplicate_dim = duplicate_dim_path.display().to_string();
    let narrow_facts = narrow_facts_path.display().to_string();
    let wide_pruned_facts = wide_pruned_facts_path.display().to_string();
    let small_rg_facts = small_rg_facts_path.display().to_string();
    let multi_facts = multi_facts_path.display().to_string();
    let multi_dim = multi_dim_path.display().to_string();
    let string_facts = string_facts_path.display().to_string();
    let string_dim = string_dim_path.display().to_string();
    let unmatched_facts = unmatched_facts_path.display().to_string();
    let unmatched_dim = unmatched_dim_path.display().to_string();
    let wide_facts = wide_facts_path.display().to_string();
    let wide_dim = wide_dim_path.display().to_string();

    let cases = vec![
        copy_case(
            "filter_limit",
            format!(
                "SELECT payload FROM '{}' WHERE id >= 100000 AND id < 100010 ORDER BY id LIMIT 10",
                facts
            ),
            format!(
                "SELECT payload FROM read_parquet('{}') WHERE id >= 100000 AND id < 100010 ORDER BY id LIMIT 10",
                facts
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "global_aggregate",
            format!(
                "SELECT count(*), sum(value), avg(value), min(payload), max(payload) FROM '{}'",
                facts
            ),
            format!(
                "SELECT count(*), sum(value), avg(value), min(payload), max(payload) FROM read_parquet('{}')",
                facts
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "grouped_aggregate",
            format!(
                "SELECT bucket, count(*), sum(value) FROM '{}' GROUP BY bucket",
                facts
            ),
            format!(
                "SELECT bucket, count(*), sum(value) FROM read_parquet('{}') GROUP BY bucket",
                facts
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "inner_join_materialize",
            format!(
                "SELECT f.id, d.payload FROM '{}' f JOIN '{}' d ON f.key = d.key",
                facts, dim
            ),
            format!(
                "SELECT f.id, d.payload FROM read_parquet('{}') f JOIN read_parquet('{}') d ON f.key = d.key",
                facts, dim
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "join_grouped_aggregate_ordered",
            format!(
                "SELECT f.bucket, count(*), sum(f.value) FROM '{}' f JOIN '{}' d ON f.key = d.key GROUP BY f.bucket ORDER BY count(*) DESC, f.bucket LIMIT 8",
                facts, dim
            ),
            format!(
                "SELECT f.bucket, count(*), sum(f.value) FROM read_parquet('{}') f JOIN read_parquet('{}') d ON f.key = d.key GROUP BY f.bucket ORDER BY count(*) DESC, f.bucket LIMIT 8",
                facts, dim
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "left_join_materialize",
            format!(
                "SELECT f.id, d.payload FROM '{}' f LEFT JOIN '{}' d ON f.key = d.key",
                facts, dim
            ),
            format!(
                "SELECT f.id, d.payload FROM read_parquet('{}') f LEFT JOIN read_parquet('{}') d ON f.key = d.key",
                facts, dim
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "right_join_materialize",
            format!(
                "SELECT f.id, d.payload FROM '{}' f RIGHT JOIN '{}' d ON f.key = d.key",
                facts, dim
            ),
            format!(
                "SELECT f.id, d.payload FROM read_parquet('{}') f RIGHT JOIN read_parquet('{}') d ON f.key = d.key",
                facts, dim
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "full_join_materialize",
            format!(
                "SELECT f.id, d.payload FROM '{}' f FULL OUTER JOIN '{}' d ON f.key = d.key",
                facts, dim
            ),
            format!(
                "SELECT f.id, d.payload FROM read_parquet('{}') f FULL OUTER JOIN read_parquet('{}') d ON f.key = d.key",
                facts, dim
            ),
            &output_dir,
            output_format,
        ),
        copy_case(
            "semi_join_materialize",
            format!(
                "SELECT f.id FROM '{}' f LEFT SEMI JOIN '{}' d ON f.key = d.key",
                facts, dim
            ),
            format!(
                "SELECT f.id FROM read_parquet('{}') f SEMI JOIN read_parquet('{}') d ON f.key = d.key",
                facts, dim
            ),
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_non_dense_i32",
            &sparse_facts,
            &sparse_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_duplicate_build",
            &facts,
            &duplicate_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_duplicate_build_narrow_fact",
            &narrow_facts,
            &duplicate_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_duplicate_build_wide_pruned_fact",
            &wide_pruned_facts,
            &duplicate_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_duplicate_build_small_row_groups",
            &small_rg_facts,
            &duplicate_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "semi_join_duplicate_build",
            &facts,
            &duplicate_dim,
            "f.key = d.key",
            "LEFT SEMI JOIN",
            "SEMI JOIN",
            "f.id",
            &output_dir,
            output_format,
        ),
        join_case(
            "semi_join_duplicate_build_narrow_fact",
            &narrow_facts,
            &duplicate_dim,
            "f.key = d.key",
            "LEFT SEMI JOIN",
            "SEMI JOIN",
            "f.id",
            &output_dir,
            output_format,
        ),
        join_case(
            "semi_join_duplicate_build_wide_pruned_fact",
            &wide_pruned_facts,
            &duplicate_dim,
            "f.key = d.key",
            "LEFT SEMI JOIN",
            "SEMI JOIN",
            "f.id",
            &output_dir,
            output_format,
        ),
        join_case(
            "semi_join_duplicate_build_small_row_groups",
            &small_rg_facts,
            &duplicate_dim,
            "f.key = d.key",
            "LEFT SEMI JOIN",
            "SEMI JOIN",
            "f.id",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_multi_key",
            &multi_facts,
            &multi_dim,
            "f.k1 = d.k1 AND f.k2 = d.k2",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_string_key",
            &string_facts,
            &string_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "full_join_unmatched_heavy",
            &unmatched_facts,
            &unmatched_dim,
            "f.key = d.key",
            "FULL OUTER JOIN",
            "FULL OUTER JOIN",
            "f.id, d.payload",
            &output_dir,
            output_format,
        ),
        join_case(
            "inner_join_wide_output",
            &wide_facts,
            &wide_dim,
            "f.key = d.key",
            "JOIN",
            "JOIN",
            "f.id, f.payload_a, f.payload_b, f.payload_c, d.payload_x, d.payload_y",
            &output_dir,
            output_format,
        ),
    ];

    let runtime = Runtime::new().expect("tokio runtime");
    let engine = DodamEngine::default();
    let mut group = c.benchmark_group(format!("duckdb_e2e_{}", output_format.extension()));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for case in &cases {
        group.bench_with_input(
            BenchmarkId::new("dodam_engine", case.name),
            &case.dodam_sql,
            |b, sql| {
                b.iter(|| {
                    runtime
                        .block_on(execute_sql(&engine, sql, DODAM_BENCH_BATCH_SIZE))
                        .map(output_rows)
                        .expect("dodam query")
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dodam_cli", case.name),
            &case.dodam_cli_sql,
            |b, sql| {
                b.iter(|| run_dodam_cli(&dodam_bin, sql, &case.dodam_output_path));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("duckdb_cli", case.name),
            &case.duckdb_sql,
            |b, sql| {
                b.iter(|| run_duckdb(sql, &case.duckdb_output_path));
            },
        );
    }
    group.finish();

    keep_tempdir_alive(tempdir);
}

fn join_case(
    name: &'static str,
    left: &str,
    right: &str,
    condition: &str,
    dodam_join: &str,
    duckdb_join: &str,
    projection: &str,
    output_dir: &Path,
    output_format: BenchCopyFormat,
) -> E2eCase {
    copy_case(
        name,
        format!("SELECT {projection} FROM '{left}' f {dodam_join} '{right}' d ON {condition}"),
        format!(
            "SELECT {projection} FROM read_parquet('{left}') f {duckdb_join} read_parquet('{right}') d ON {condition}"
        ),
        output_dir,
        output_format,
    )
}

fn copy_case(
    name: &'static str,
    dodam_select: String,
    duckdb_select: String,
    output_dir: &Path,
    output_format: BenchCopyFormat,
) -> E2eCase {
    copy_case_with_format(
        name,
        dodam_select,
        duckdb_select,
        output_dir,
        output_format.format_name(),
        output_format.extension(),
    )
}

fn copy_case_with_format(
    name: &'static str,
    dodam_select: String,
    duckdb_select: String,
    output_dir: &Path,
    format: &str,
    extension: &str,
) -> E2eCase {
    let dodam_output_path = output_dir.join(format!("dodam-{name}.{extension}"));
    let duckdb_output_path = output_dir.join(format!("duckdb-{name}.{extension}"));
    E2eCase {
        name,
        dodam_cli_sql: format!(
            "COPY ({dodam_select}) TO '{}' (FORMAT {format})",
            dodam_output_path.display()
        ),
        duckdb_sql: format!(
            "COPY ({duckdb_select}) TO '{}' (FORMAT {format})",
            duckdb_output_path.display()
        ),
        dodam_sql: dodam_select,
        dodam_output_path,
        duckdb_output_path,
    }
}

#[derive(Debug, Clone, Copy)]
enum BenchCopyFormat {
    Csv,
    Parquet,
}

impl BenchCopyFormat {
    fn format_name(self) -> &'static str {
        match self {
            Self::Csv => "CSV",
            Self::Parquet => "PARQUET",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Parquet => "parquet",
        }
    }
}

fn bench_copy_format() -> BenchCopyFormat {
    match std::env::var("DODAM_E2E_COPY_FORMAT") {
        Ok(value) if value.eq_ignore_ascii_case("csv") => BenchCopyFormat::Csv,
        Ok(value) if value.eq_ignore_ascii_case("parquet") => BenchCopyFormat::Parquet,
        Ok(value) => panic!("DODAM_E2E_COPY_FORMAT must be CSV or PARQUET, got {value}"),
        Err(_) => BenchCopyFormat::Parquet,
    }
}

struct E2eCase {
    name: &'static str,
    dodam_sql: String,
    dodam_cli_sql: String,
    duckdb_sql: String,
    dodam_output_path: PathBuf,
    duckdb_output_path: PathBuf,
}

fn output_rows(output: QueryOutput) -> usize {
    match output {
        QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => {
            batches.iter().map(|batch| batch.num_rows()).sum()
        }
        QueryOutput::Explain { plan } => plan.len(),
    }
}

fn run_duckdb(sql: &str, output_path: &Path) {
    remove_output_file(output_path);
    let output = Command::new("duckdb")
        .args(["-csv", "-noheader", "-c", sql])
        .output()
        .expect("run duckdb");
    assert!(
        output.status.success(),
        "duckdb failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_output_file(output_path);
}

fn run_dodam_cli(binary: &std::path::Path, sql: &str, output_path: &Path) {
    remove_output_file(output_path);
    let batch_size = DODAM_BENCH_BATCH_SIZE.to_string();
    let output = Command::new(binary)
        .args(["query", sql, "--batch-size", &batch_size])
        .stdout(Stdio::null())
        .output()
        .expect("run dodam cli");
    assert!(
        output.status.success(),
        "dodam failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_output_file(output_path);
}

fn remove_output_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("remove benchmark output {}: {error}", path.display()),
    }
}

fn assert_output_file(path: &Path) {
    let metadata = std::fs::metadata(path)
        .unwrap_or_else(|error| panic!("benchmark output {} missing: {error}", path.display()));
    assert!(
        metadata.len() > 0,
        "benchmark output {} is empty",
        path.display()
    );
}

fn ensure_dodam_release_binary() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let binary = manifest_dir.join("target/release/dodam");
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "dodam"])
        .current_dir(&manifest_dir)
        .status()
        .expect("build dodam release binary");
    assert!(status.success(), "failed to build dodam release binary");
    binary
}

fn write_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("bucket", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
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

fn write_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_sparse_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| ((value % 65_536) * 17 + 3) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("sparse-fact-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_sparse_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value * 17 + 3) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("sparse-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_duplicate_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("dup-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_narrow_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_wide_pruned_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("unused_i64_a", DataType::Int64, false),
        Field::new("unused_i64_b", DataType::Int64, false),
        Field::new("unused_text_a", DataType::Utf8, false),
        Field::new("unused_text_b", DataType::Utf8, false),
        Field::new("unused_text_c", DataType::Utf8, false),
        Field::new("unused_text_d", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|value| value as i64 * 11),
        )),
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|value| value as i64 * 13),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-a-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-b-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-c-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unused-d-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_multi_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 8192) as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| ((value / 8192) % 8) as i32),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_multi_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 8192) as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| ((value / 8192) % 8) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("multi-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_string_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("k{:05}", value % 32_768)),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_string_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("k{value:05}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("string-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_unmatched_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 65_536) as i32),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_unmatched_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value + 32_768) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("unmatched-dim-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_wide_facts_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("payload_a", DataType::Utf8, false),
        Field::new("payload_b", DataType::Utf8, false),
        Field::new("payload_c", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| (value % 32_768) as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-a-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-b-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-c-{value}")),
        )),
    ];
    write_parquet(path, schema, columns, row_group_size);
}

fn write_wide_dim_parquet(path: &std::path::Path, rows: usize, row_group_size: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload_x", DataType::Utf8, false),
        Field::new("payload_y", DataType::Utf8, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|value| value as i32),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-x-{value}")),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..rows).map(|value| format!("wide-y-{value}")),
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

criterion_group!(benches, bench_duckdb_e2e);
criterion_main!(benches);
