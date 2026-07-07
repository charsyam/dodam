use std::fs::File;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, ListArray, StringArray, StructArray, TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_cast::display::array_value_to_string;
use dodam::copy::{CopyFileQuerySink, parse_copy_to_select};
use dodam::engine::DodamEngine;
use dodam::error::DodamError;
use dodam::execution::RecordBatchSink;
use dodam::sql::{QueryOutput, SqlSinkExecutionOptions, execute_sql, execute_sql_to_result_sink};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

const BATCH_SIZE: usize = 4;

#[tokio::test]
async fn duckdb_differential_scan_filter_and_nulls() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, payload FROM '{}' WHERE key >= 2 OR payload IS NULL ORDER BY id",
            facts_path.display()
        ),
        &format!(
            "SELECT id, key, payload FROM read_parquet('{}') WHERE key >= 2 OR payload IS NULL ORDER BY id",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, payload FROM '{}' WHERE payload IS NOT NULL AND NOT (key = 2) ORDER BY id",
            facts_path.display()
        ),
        &format!(
            "SELECT id, key, payload FROM read_parquet('{}') WHERE payload IS NOT NULL AND NOT (key = 2) ORDER BY id",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, payload FROM '{}' WHERE key IN (1, 3) ORDER BY id",
            facts_path.display()
        ),
        &format!(
            "SELECT id, key, payload FROM read_parquet('{}') WHERE key IN (1, 3) ORDER BY id",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, payload FROM '{}' WHERE key IN (2, NULL) OR id = 1 ORDER BY id",
            facts_path.display()
        ),
        &format!(
            "SELECT id, key, payload FROM read_parquet('{}') WHERE key IN (2, NULL) OR id = 1 ORDER BY id",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, payload FROM '{}' WHERE key NOT IN (1, NULL) OR id = 1 ORDER BY id",
            facts_path.display()
        ),
        &format!(
            "SELECT id, key, payload FROM read_parquet('{}') WHERE key NOT IN (1, NULL) OR id = 1 ORDER BY id",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_null_boolean_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    let predicates = [
        "key = NULL",
        "key <> NULL",
        "key IS NULL",
        "key IS NOT NULL",
        "(key = 2 OR key IS NULL) AND payload IS NOT NULL",
        "NOT (key = 2 OR payload IS NULL)",
        "key IN (SELECT key FROM facts_sub WHERE key IS NULL) OR id = 1",
        "key NOT IN (SELECT key FROM facts_sub WHERE key IS NULL) OR id = 1",
    ];
    for predicate in predicates {
        let dodam_predicate =
            predicate.replace("facts_sub", &format!("'{}'", facts_path.display()));
        let duckdb_predicate = predicate.replace(
            "facts_sub",
            &format!("read_parquet('{}')", facts_path.display()),
        );
        assert_same_as_duckdb(
            &format!(
                "SELECT id, key, payload FROM '{}' WHERE {dodam_predicate} ORDER BY id",
                facts_path.display()
            ),
            &format!(
                "SELECT id, key, payload FROM read_parquet('{}') WHERE {duckdb_predicate} ORDER BY id",
                facts_path.display()
            ),
            tempdir.path(),
        )
        .await;
    }
}

#[tokio::test]
async fn duckdb_differential_aggregate_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT count(*), count(value), sum(value), avg(value), min(value), max(value), min(payload), max(payload) FROM '{}'",
            facts_path.display()
        ),
        &format!(
            "SELECT count(*), count(value), sum(value), avg(value), min(value), max(value), min(payload), max(payload) FROM read_parquet('{}')",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT key, count(*), count(value), sum(value), avg(value), min(payload), max(payload) FROM '{}' GROUP BY key ORDER BY key",
            facts_path.display()
        ),
        &format!(
            "SELECT key, count(*), count(value), sum(value), avg(value), min(payload), max(payload) FROM read_parquet('{}') GROUP BY key ORDER BY key",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_join_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let dim_path = tempdir.path().join("dim.parquet");
    write_facts_parquet(&facts_path);
    write_dim_parquet(&dim_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key, d.name FROM '{}' f JOIN '{}' d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key, d.name FROM read_parquet('{}') f JOIN read_parquet('{}') d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key, d.name FROM '{}' f LEFT JOIN '{}' d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key, d.name FROM read_parquet('{}') f LEFT JOIN read_parquet('{}') d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key, d.name FROM '{}' f RIGHT JOIN '{}' d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key, d.name FROM read_parquet('{}') f RIGHT JOIN read_parquet('{}') d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key, d.name FROM '{}' f FULL OUTER JOIN '{}' d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key, d.name FROM read_parquet('{}') f FULL OUTER JOIN read_parquet('{}') d ON f.key = d.key ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb_unordered_case(
        "join scalar projection expressions",
        &format!(
            "SELECT f.id, COALESCE(d.name, 'missing') AS dim_name, f.value * 2 AS doubled_value, CASE WHEN d.name IS NULL THEN 'unmatched' ELSE 'matched' END AS match_state FROM '{}' f LEFT JOIN '{}' d ON f.key = d.key",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, COALESCE(d.name, 'missing') AS dim_name, f.value * 2 AS doubled_value, CASE WHEN d.name IS NULL THEN 'unmatched' ELSE 'matched' END AS match_state FROM read_parquet('{}') f LEFT JOIN read_parquet('{}') d ON f.key = d.key",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, COALESCE(d.name, 'missing') AS dim_name FROM '{}' f LEFT JOIN '{}' d ON f.key = d.key WHERE f.id IN (1, 3) ORDER BY dim_name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, COALESCE(d.name, 'missing') AS dim_name FROM read_parquet('{}') f LEFT JOIN read_parquet('{}') d ON f.key = d.key WHERE f.id IN (1, 3) ORDER BY dim_name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, COALESCE(d.name, 'missing') AS dim_name FROM '{}' f LEFT JOIN '{}' d ON f.key = d.key WHERE f.id IN (1, 3) ORDER BY COALESCE(d.name, 'missing')",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, COALESCE(d.name, 'missing') AS dim_name FROM read_parquet('{}') f LEFT JOIN read_parquet('{}') d ON f.key = d.key WHERE f.id IN (1, 3) ORDER BY COALESCE(d.name, 'missing')",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key FROM '{}' f LEFT SEMI JOIN '{}' d ON f.key = d.key ORDER BY f.id",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key FROM read_parquet('{}') f SEMI JOIN read_parquet('{}') d ON f.key = d.key ORDER BY f.id",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key, d.name FROM '{}' f LEFT JOIN '{}' d ON f.key = d.key WHERE f.key IS NULL ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key, d.name FROM read_parquet('{}') f LEFT JOIN read_parquet('{}') d ON f.key = d.key WHERE f.key IS NULL ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.key, d.name FROM '{}' f FULL OUTER JOIN '{}' d ON f.key = d.key WHERE f.key IS NULL OR d.key IS NULL ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        &format!(
            "SELECT f.id, f.key, d.name FROM read_parquet('{}') f FULL OUTER JOIN read_parquet('{}') d ON f.key = d.key WHERE f.key IS NULL OR d.key IS NULL ORDER BY f.id, d.name",
            facts_path.display(),
            dim_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_multi_key_and_string_join() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let left_path = tempdir.path().join("multi-left.parquet");
    let right_path = tempdir.path().join("multi-right.parquet");
    write_multi_left_parquet(&left_path);
    write_multi_right_parquet(&right_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT l.id, r.label FROM '{}' l JOIN '{}' r ON l.k1 = r.k1 AND l.k2 = r.k2 ORDER BY l.id, r.label",
            left_path.display(),
            right_path.display()
        ),
        &format!(
            "SELECT l.id, r.label FROM read_parquet('{}') l JOIN read_parquet('{}') r ON l.k1 = r.k1 AND l.k2 = r.k2 ORDER BY l.id, r.label",
            left_path.display(),
            right_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_order_limit() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, value FROM '{}' ORDER BY key DESC, id ASC LIMIT 4",
            facts_path.display()
        ),
        &format!(
            "SELECT id, key, value FROM read_parquet('{}') ORDER BY key DESC, id ASC LIMIT 4",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_derived_tables() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM (SELECT id, key, payload FROM '{}' WHERE key >= 2) AS t WHERE t.payload IS NOT NULL ORDER BY t.id",
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM (SELECT id, key, payload FROM read_parquet('{}') WHERE key >= 2) AS t WHERE t.payload IS NOT NULL ORDER BY t.id",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT key, count(*), min(payload), max(payload) FROM (SELECT key, payload FROM '{}' WHERE value IS NOT NULL) AS t GROUP BY key ORDER BY key",
            facts_path.display()
        ),
        &format!(
            "SELECT key, count(*), min(payload), max(payload) FROM (SELECT key, payload FROM read_parquet('{}') WHERE value IS NOT NULL) AS t GROUP BY key ORDER BY key",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT DISTINCT payload FROM (SELECT id, payload FROM '{}' WHERE id >= 1) AS t ORDER BY payload",
            facts_path.display()
        ),
        &format!(
            "SELECT DISTINCT payload FROM (SELECT id, payload FROM read_parquet('{}') WHERE id >= 1) AS t ORDER BY payload",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT l.id, r.payload FROM (SELECT id, payload FROM '{}' WHERE id <= 4) AS l JOIN (SELECT id, payload FROM '{}' WHERE key >= 2) AS r ON l.id = r.id ORDER BY l.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT l.id, r.payload FROM (SELECT id, payload FROM read_parquet('{}') WHERE id <= 4) AS l JOIN (SELECT id, payload FROM read_parquet('{}') WHERE key >= 2) AS r ON l.id = r.id ORDER BY l.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT l.payload, count(*), min(r.id), max(r.id) FROM (SELECT id, payload FROM '{}' WHERE id <= 5 AND payload IS NOT NULL) AS l JOIN (SELECT id, payload FROM '{}' WHERE key >= 2 AND payload IS NOT NULL) AS r ON l.payload = r.payload GROUP BY l.payload ORDER BY l.payload",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT l.payload, count(*), min(r.id), max(r.id) FROM (SELECT id, payload FROM read_parquet('{}') WHERE id <= 5 AND payload IS NOT NULL) AS l JOIN (SELECT id, payload FROM read_parquet('{}') WHERE key >= 2 AND payload IS NOT NULL) AS r ON l.payload = r.payload GROUP BY l.payload ORDER BY l.payload",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT DISTINCT l.payload FROM (SELECT id, payload FROM '{}' WHERE id <= 5 AND payload IS NOT NULL) AS l JOIN (SELECT id, payload FROM '{}' WHERE key >= 2 AND payload IS NOT NULL) AS r ON l.payload = r.payload ORDER BY l.payload",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT DISTINCT l.payload FROM (SELECT id, payload FROM read_parquet('{}') WHERE id <= 5 AND payload IS NOT NULL) AS l JOIN (SELECT id, payload FROM read_parquet('{}') WHERE key >= 2 AND payload IS NOT NULL) AS r ON l.payload = r.payload ORDER BY l.payload",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_scalar_type_filters() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    let cases = [
        "flag = true",
        "score >= 1.5",
        "amount = '123.45'",
        "amount = '123.4500'",
        "created_at >= '2024-01-02 00:00:00'",
        "created_at_utc >= '2024-01-02 09:00:00+09:00'",
        "event_date >= '2024-01-02'",
    ];
    for predicate in cases {
        assert_same_as_duckdb(
            &format!(
                "SELECT id FROM '{}' WHERE {predicate} ORDER BY id",
                types_path.display()
            ),
            &format!(
                "SELECT id FROM read_parquet('{}') WHERE {predicate} ORDER BY id",
                types_path.display()
            ),
            tempdir.path(),
        )
        .await;
    }
}

#[tokio::test]
async fn duckdb_differential_decimal_timestamp_boundary_filters() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    let cases = [
        "amount < '0.00' OR amount = '-7.000'",
        "amount >= '-7.0000' AND amount <= '123.4500'",
        "created_at < '1970-01-01 00:00:01'",
        "created_at_utc = '1970-01-01 09:00:00+09:00'",
        "event_date < '2024-01-02' OR event_date IS NULL",
    ];
    for predicate in cases {
        assert_same_as_duckdb(
            &format!(
                "SELECT id FROM '{}' WHERE {predicate} ORDER BY id",
                types_path.display()
            ),
            &format!(
                "SELECT id FROM read_parquet('{}') WHERE {predicate} ORDER BY id",
                types_path.display()
            ),
            tempdir.path(),
        )
        .await;
    }
}

#[tokio::test]
async fn duckdb_differential_all_null_aggregate_groups() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("aggregate-nulls.parquet");
    write_aggregate_nulls_parquet(&path);

    assert_same_as_duckdb(
        &format!(
            "SELECT grp, count(*), count(value), sum(value), avg(value), min(value), max(value), min(label), max(label) FROM '{}' GROUP BY grp ORDER BY grp",
            path.display()
        ),
        &format!(
            "SELECT grp, count(*), count(value), sum(value), avg(value), min(value), max(value), min(label), max(label) FROM read_parquet('{}') GROUP BY grp ORDER BY grp",
            path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_type_projection_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, flag, score, note, amount, event_date FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, flag, score, note, amount, event_date FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE amount = '123.45' OR event_date = '1970-01-01' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE amount = '123.45' OR event_date = '1970-01-01' ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_cast_display_semantics() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(amount AS VARCHAR), CAST(event_date AS VARCHAR), CAST(created_at AS VARCHAR), COALESCE(CAST(score AS VARCHAR), 'missing') FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(amount AS VARCHAR), CAST(event_date AS VARCHAR), CAST(created_at AS VARCHAR), COALESCE(CAST(score AS VARCHAR), 'missing') FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_like_filters() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("like_values.parquet");
    write_like_values_parquet(&path);

    let cases = [
        "text LIKE 'alpha'",
        "text LIKE 'alpha%'",
        "text LIKE '%beta'",
        "text LIKE '%special%requests%'",
        "text NOT LIKE '%requests%'",
        "text LIKE 'a_ph_'",
        "text LIKE '100!%%' ESCAPE '!'",
    ];
    for predicate in cases {
        assert_same_as_duckdb(
            &format!(
                "SELECT id FROM '{}' WHERE {predicate} ORDER BY id",
                path.display()
            ),
            &format!(
                "SELECT id FROM read_parquet('{}') WHERE {predicate} ORDER BY id",
                path.display()
            ),
            tempdir.path(),
        )
        .await;
    }
}

#[tokio::test]
async fn duckdb_differential_scalar_projection_expressions() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id + 10 AS plus_ten, id * 2 AS doubled, CAST(id AS VARCHAR) AS id_text, COALESCE(note, 'fallback') AS note_text FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id + 10 AS plus_ten, id * 2 AS doubled, CAST(id AS VARCHAR) AS id_text, COALESCE(note, 'fallback') AS note_text FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_scalar_where_expressions() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, note FROM '{}' WHERE id + 1 >= 3 AND COALESCE(note, 'fallback') <> 'fallback' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, note FROM read_parquet('{}') WHERE id + 1 >= 3 AND COALESCE(note, 'fallback') <> 'fallback' ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE CAST(id AS VARCHAR) = '2' OR COALESCE(note, NULL) IS NULL ORDER BY id LIMIT 2",
            types_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE CAST(id AS VARCHAR) = '2' OR COALESCE(note, NULL) IS NULL ORDER BY id LIMIT 2",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_string_functions_and_case() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT lower(note) AS lower_note, upper(note) AS upper_note, length(note) AS note_len, CASE WHEN id = 1 THEN 'one' WHEN note IS NULL THEN 'missing' ELSE 'other' END AS bucket FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT lower(note) AS lower_note, upper(note) AS upper_note, length(note) AS note_len, CASE WHEN id = 1 THEN 'one' WHEN note IS NULL THEN 'missing' ELSE 'other' END AS bucket FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE upper(COALESCE(note, 'fallback')) = 'GAMMA' OR length(note) = 0 ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE upper(COALESCE(note, 'fallback')) = 'GAMMA' OR length(note) = 0 ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_substring_expression() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, substring(note FROM 1 FOR 2) AS prefix FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, substring(note FROM 1 FOR 2) AS prefix FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE substring(note FROM 1 FOR 1) IN ('a', 'g') ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE substring(note FROM 1 FOR 1) IN ('a', 'g') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_tpch_lite_queries() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let lineitem_path = tempdir.path().join("lineitem.parquet");
    let orders_path = tempdir.path().join("orders.parquet");
    let customer_path = tempdir.path().join("customer.parquet");
    write_tpch_lineitem_parquet(&lineitem_path);
    write_tpch_orders_parquet(&orders_path);
    write_tpch_customer_parquet(&customer_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT l_returnflag, l_linestatus, count(*), sum(l_quantity), sum(l_extendedprice), sum(l_discount), min(l_shipdate), max(l_shipdate) FROM '{}' WHERE l_shipdate <= '1998-09-02' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
            lineitem_path.display()
        ),
        &format!(
            "SELECT l_returnflag, l_linestatus, count(*), sum(l_quantity), sum(l_extendedprice), sum(l_discount), min(l_shipdate), max(l_shipdate) FROM read_parquet('{}') WHERE l_shipdate <= '1998-09-02' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
            lineitem_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT sum(l_extendedprice * l_discount) FROM '{}' WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR AND l_discount BETWEEN 5 AND 7 AND l_quantity < 24",
            lineitem_path.display()
        ),
        &format!(
            "SELECT sum(l_extendedprice * l_discount) FROM read_parquet('{}') WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR AND l_discount BETWEEN 5 AND 7 AND l_quantity < 24",
            lineitem_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT c.c_mktsegment, count(*), sum(o.o_totalprice) FROM '{}' c JOIN '{}' o ON c.c_custkey = o.o_custkey WHERE c.c_mktsegment = 'BUILDING' GROUP BY c.c_mktsegment ORDER BY c.c_mktsegment",
            customer_path.display(),
            orders_path.display()
        ),
        &format!(
            "SELECT c.c_mktsegment, count(*), sum(o.o_totalprice) FROM read_parquet('{}') c JOIN read_parquet('{}') o ON c.c_custkey = o.o_custkey WHERE c.c_mktsegment = 'BUILDING' GROUP BY c.c_mktsegment ORDER BY c.c_mktsegment",
            customer_path.display(),
            orders_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_tpch_q6_canonical_shape() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let lineitem_path = tempdir.path().join("lineitem.parquet");
    write_tpch_q6_lineitem_parquet(&lineitem_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT sum(l_extendedprice * l_discount) AS revenue FROM '{}' WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01 AND l_quantity < 24",
            lineitem_path.display()
        ),
        &format!(
            "SELECT sum(l_extendedprice * l_discount) AS revenue FROM read_parquet('{}') WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01 AND l_quantity < 24",
            lineitem_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_in_subquery() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE payload IN (SELECT payload FROM '{}' WHERE key >= 2 AND payload IS NOT NULL) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE payload IN (SELECT payload FROM read_parquet('{}') WHERE key >= 2 AND payload IS NOT NULL) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE payload IS NOT NULL AND payload NOT IN (SELECT payload FROM '{}' WHERE key >= 2 AND payload IS NOT NULL) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE payload IS NOT NULL AND payload NOT IN (SELECT payload FROM read_parquet('{}') WHERE key >= 2 AND payload IS NOT NULL) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE payload IN (SELECT payload FROM '{}' WHERE key IS NULL OR key = 3) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE payload IN (SELECT payload FROM read_parquet('{}') WHERE key IS NULL OR key = 3) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE payload NOT IN (SELECT payload FROM '{}' WHERE key IS NULL OR key = 3) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE payload NOT IN (SELECT payload FROM read_parquet('{}') WHERE key IS NULL OR key = 3) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE payload = 'f')",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE id = (SELECT id FROM read_parquet('{}') WHERE payload = 'f')",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE payload = 'missing') ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE id = (SELECT id FROM read_parquet('{}') WHERE payload = 'missing') ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE payload = (SELECT payload FROM '{}' WHERE id = 3) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE payload = (SELECT payload FROM read_parquet('{}') WHERE id = 3) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_subquery_null_semantics() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE key IN (SELECT key FROM '{}' WHERE key IS NULL OR key = 2) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE key IN (SELECT key FROM read_parquet('{}') WHERE key IS NULL OR key = 2) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE key NOT IN (SELECT key FROM '{}' WHERE key IS NULL OR key = 2) OR id = 1 ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE key NOT IN (SELECT key FROM read_parquet('{}') WHERE key IS NULL OR key = 2) OR id = 1 ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE id = (SELECT key FROM '{}' WHERE key IS NULL) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE id = (SELECT key FROM read_parquet('{}') WHERE key IS NULL) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_exists_subquery() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE EXISTS (SELECT id FROM '{}' WHERE payload = 'f') ORDER BY id LIMIT 3",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE EXISTS (SELECT id FROM read_parquet('{}') WHERE payload = 'f') ORDER BY id LIMIT 3",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, payload FROM '{}' WHERE NOT EXISTS (SELECT id FROM '{}' WHERE payload = 'missing') ORDER BY id LIMIT 3",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id, payload FROM read_parquet('{}') WHERE NOT EXISTS (SELECT id FROM read_parquet('{}') WHERE payload = 'missing') ORDER BY id LIMIT 3",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.payload FROM '{}' f WHERE EXISTS (SELECT id FROM '{}' g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.id, f.payload FROM read_parquet('{}') f WHERE EXISTS (SELECT id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.payload FROM '{}' f WHERE NOT EXISTS (SELECT id FROM '{}' g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.id, f.payload FROM read_parquet('{}') f WHERE NOT EXISTS (SELECT id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.key, count(*) FROM '{}' f WHERE NOT EXISTS (SELECT id FROM '{}' g WHERE g.key = f.key AND g.id > 4) GROUP BY f.key ORDER BY f.key",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.key, count(*) FROM read_parquet('{}') f WHERE NOT EXISTS (SELECT id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) GROUP BY f.key ORDER BY f.key",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.payload FROM '{}' f WHERE f.id = 1 OR EXISTS (SELECT id FROM '{}' g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.id, f.payload FROM read_parquet('{}') f WHERE f.id = 1 OR EXISTS (SELECT id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_correlated_subquery_boolean_combinations() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    let cases = [
        (
            format!(
                "SELECT f.id FROM '{}' f WHERE f.payload IS NOT NULL AND EXISTS (SELECT id FROM '{}' g WHERE g.key = f.key AND g.id > 4) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT f.id FROM read_parquet('{}') f WHERE f.payload IS NOT NULL AND EXISTS (SELECT id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT f.id FROM '{}' f WHERE f.key IS NULL OR NOT EXISTS (SELECT id FROM '{}' g WHERE g.key = f.key AND g.id > 4) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT f.id FROM read_parquet('{}') f WHERE f.key IS NULL OR NOT EXISTS (SELECT id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT f.id FROM '{}' f WHERE f.key IN (SELECT g.key FROM '{}' g WHERE g.key = f.key OR g.key IS NULL) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT f.id FROM read_parquet('{}') f WHERE f.key IN (SELECT g.key FROM read_parquet('{}') g WHERE g.key = f.key OR g.key IS NULL) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
    ];
    for (dodam_sql, duckdb_sql) in cases {
        assert_same_as_duckdb(&dodam_sql, &duckdb_sql, tempdir.path()).await;
    }
}

#[tokio::test]
async fn duckdb_differential_correlated_in_and_scalar_subquery() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.payload FROM '{}' f WHERE f.id IN (SELECT g.id FROM '{}' g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.id, f.payload FROM read_parquet('{}') f WHERE f.id IN (SELECT g.id FROM read_parquet('{}') g WHERE g.key = f.key AND g.id > 4) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.payload FROM '{}' f WHERE f.payload IS NOT NULL AND f.id IN (SELECT g.id FROM '{}' g WHERE g.key = f.key AND (g.id > 4 OR g.payload IS NULL)) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.id, f.payload FROM read_parquet('{}') f WHERE f.payload IS NOT NULL AND f.id IN (SELECT g.id FROM read_parquet('{}') g WHERE g.key = f.key AND (g.id > 4 OR g.payload IS NULL)) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT f.id, f.payload FROM '{}' f WHERE f.id = (SELECT g.id FROM '{}' g WHERE g.key = f.key ORDER BY g.id LIMIT 1) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT f.id, f.payload FROM read_parquet('{}') f WHERE f.id = (SELECT g.id FROM read_parquet('{}') g WHERE g.key = f.key ORDER BY g.id LIMIT 1) ORDER BY f.id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_scalar_subquery_errors() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_both_error(
        &format!(
            "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE key = 2) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE id = (SELECT id FROM read_parquet('{}') WHERE key = 2) ORDER BY id",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_scalar_subquery_edge_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    let cases = [
        (
            format!(
                "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE payload = 'missing') ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT id FROM read_parquet('{}') WHERE id = (SELECT id FROM read_parquet('{}') WHERE payload = 'missing') ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT id FROM '{}' WHERE id <> (SELECT key FROM '{}' WHERE key IS NULL) OR id = 1 ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT id FROM read_parquet('{}') WHERE id <> (SELECT key FROM read_parquet('{}') WHERE key IS NULL) OR id = 1 ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT id FROM '{}' WHERE id = (SELECT key FROM '{}' WHERE key = 3 ORDER BY id LIMIT 1) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT id FROM read_parquet('{}') WHERE id = (SELECT key FROM read_parquet('{}') WHERE key = 3 ORDER BY id LIMIT 1) ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT id FROM '{}' WHERE key NOT IN (SELECT key FROM '{}' WHERE key IS NULL OR key = 3) OR id = 1 ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT id FROM read_parquet('{}') WHERE key NOT IN (SELECT key FROM read_parquet('{}') WHERE key IS NULL OR key = 3) OR id = 1 ORDER BY id",
                facts_path.display(),
                facts_path.display()
            ),
        ),
    ];
    for (dodam_sql, duckdb_sql) in cases {
        assert_same_as_duckdb(&dodam_sql, &duckdb_sql, tempdir.path()).await;
    }
}

#[tokio::test]
async fn duckdb_differential_seeded_randomized_smoke() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let types_path = tempdir.path().join("types.parquet");
    write_facts_parquet(&facts_path);
    write_types_parquet(&types_path);

    const SEED: u64 = 0xD0DA_2026_0001;
    let mut rng = TestRng::new(SEED);
    let cases = (0..96)
        .map(|case_id| {
            if rng.chance(3, 5) {
                random_facts_query(&mut rng, &facts_path, case_id)
            } else {
                random_types_query(&mut rng, &types_path, case_id)
            }
        })
        .collect::<Vec<_>>();

    for case in cases {
        assert_same_as_duckdb_case(
            &format!("seed={SEED:#x} case={}", case.case_id),
            &case.dodam_sql,
            &case.duckdb_sql,
            tempdir.path(),
        )
        .await;
    }
}

#[tokio::test]
async fn duckdb_differential_seeded_randomized_join_smoke() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let dim_path = tempdir.path().join("dim.parquet");
    let multi_left_path = tempdir.path().join("multi-left.parquet");
    let multi_right_path = tempdir.path().join("multi-right.parquet");
    write_facts_parquet(&facts_path);
    write_dim_parquet(&dim_path);
    write_multi_left_parquet(&multi_left_path);
    write_multi_right_parquet(&multi_right_path);

    const SEED: u64 = 0xD0DA_2026_0002;
    let mut rng = TestRng::new(SEED);
    let cases = (0..72)
        .map(|case_id| {
            if rng.chance(2, 3) {
                random_single_key_join_query(&mut rng, &facts_path, &dim_path, case_id)
            } else {
                random_multi_key_join_query(&mut rng, &multi_left_path, &multi_right_path, case_id)
            }
        })
        .collect::<Vec<_>>();

    for case in cases {
        assert_same_as_duckdb_unordered_case(
            &format!("seed={SEED:#x} case={}", case.case_id),
            &case.dodam_sql,
            &case.duckdb_sql,
            tempdir.path(),
        )
        .await;
    }
}

#[tokio::test]
async fn duckdb_differential_error_semantics_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let dim_path = tempdir.path().join("dim.parquet");
    write_facts_parquet(&facts_path);
    write_dim_parquet(&dim_path);

    let cases = [
        (
            format!("SELECT missing FROM '{}'", facts_path.display()),
            format!(
                "SELECT missing FROM read_parquet('{}')",
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE key = 2)",
                facts_path.display(),
                facts_path.display()
            ),
            format!(
                "SELECT id FROM read_parquet('{}') WHERE id = (SELECT id FROM read_parquet('{}') WHERE key = 2)",
                facts_path.display(),
                facts_path.display()
            ),
        ),
        (
            format!(
                "SELECT key FROM '{}' f JOIN '{}' d ON f.key = d.key",
                facts_path.display(),
                dim_path.display()
            ),
            format!(
                "SELECT key FROM read_parquet('{}') f JOIN read_parquet('{}') d ON f.key = d.key",
                facts_path.display(),
                dim_path.display()
            ),
        ),
    ];
    for (dodam_sql, duckdb_sql) in cases {
        assert_both_error(&dodam_sql, &duckdb_sql, tempdir.path()).await;
    }
}

#[tokio::test]
async fn dodam_sql_error_contract_matrix() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let dim_path = tempdir.path().join("dim.parquet");
    let types_path = tempdir.path().join("types.parquet");
    write_facts_parquet(&facts_path);
    write_dim_parquet(&dim_path);
    write_types_parquet(&types_path);

    assert_dodam_unknown_column(
        &format!("SELECT missing FROM '{}'", facts_path.display()),
        "missing",
    )
    .await;
    assert_dodam_unknown_column(
        &format!(
            "SELECT id FROM '{}' WHERE missing = 1",
            facts_path.display()
        ),
        "missing",
    )
    .await;
    assert_dodam_unknown_column(
        &format!("SELECT id FROM '{}' ORDER BY missing", facts_path.display()),
        "missing",
    )
    .await;
    assert_dodam_error_contains(
        &format!(
            "SELECT key FROM '{}' f JOIN '{}' d ON f.key = d.key",
            facts_path.display(),
            dim_path.display()
        ),
        "ambiguous column key",
    )
    .await;
    assert_dodam_error_contains(
        &format!("SELECT z.id FROM '{}' f", facts_path.display()),
        "unknown table qualifier: z",
    )
    .await;
    assert_dodam_error_contains(
        &format!(
            "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE key = 2)",
            facts_path.display(),
            facts_path.display()
        ),
        "scalar subquery must return at most one row",
    )
    .await;
    assert_dodam_error_contains(
        &format!(
            "SELECT key, count(*) FROM '{}' GROUP BY ALL",
            facts_path.display()
        ),
        "GROUP BY ALL",
    )
    .await;
    assert_dodam_error_contains(
        &format!("SELECT id + payload FROM '{}'", facts_path.display()),
        "cannot use Utf8 in integer arithmetic",
    )
    .await;
    assert_dodam_error_contains(
        &format!("SELECT flag + 1 FROM '{}'", types_path.display()),
        "cannot use Boolean in integer arithmetic",
    )
    .await;
    assert_dodam_error_contains(
        &format!(
            "SELECT CAST('not-a-date' AS DATE) FROM '{}'",
            facts_path.display()
        ),
        "invalid DATE literal",
    )
    .await;
    assert_dodam_error_contains(
        &format!(
            "SELECT CAST('2024-99-99 00:00:00' AS TIMESTAMP) FROM '{}'",
            facts_path.display()
        ),
        "invalid DATE literal",
    )
    .await;
}

#[tokio::test]
async fn dodam_copy_error_contract_matrix() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let output_path = tempdir.path().join("bad.parquet");
    write_facts_parquet(&facts_path);

    assert_dodam_copy_error_contains(
        &format!(
            "COPY (SELECT id FROM '{}') TO '{}' (FORMAT parquet, COMPRESSION brotli)",
            facts_path.display(),
            output_path.display()
        ),
        "COPY PARQUET COMPRESSION BROTLI is not supported",
    )
    .await;
    assert_dodam_copy_error_contains(
        &format!(
            "COPY (SELECT id FROM '{}') TO '{}' (FORMAT parquet, ROW_GROUP_SIZE 0)",
            facts_path.display(),
            output_path.display()
        ),
        "ROW_GROUP_SIZE expects a positive integer",
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_extended_type_matrix() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let types_path = tempdir.path().join("types.parquet");
    write_types_parquet(&types_path);

    let cases = [
        (
            "SELECT id, amount FROM types_table WHERE amount = '123.4500'",
            "SELECT id, amount FROM types_table WHERE amount = '123.4500'",
        ),
        (
            "SELECT id, amount FROM types_table WHERE amount <> '0.0000' OR amount IS NULL",
            "SELECT id, amount FROM types_table WHERE amount <> '0.0000' OR amount IS NULL",
        ),
        (
            "SELECT id FROM types_table WHERE created_at < '2024-01-02 12:00:00' OR created_at IS NULL",
            "SELECT id FROM types_table WHERE created_at < '2024-01-02 12:00:00' OR created_at IS NULL",
        ),
        (
            "SELECT event_date, count(*) FROM types_table GROUP BY event_date",
            "SELECT event_date, count(*) FROM types_table GROUP BY event_date",
        ),
        (
            "SELECT id, CASE WHEN flag = true THEN 'yes' WHEN flag = false THEN 'no' ELSE 'unknown' END FROM types_table",
            "SELECT id, CASE WHEN flag = true THEN 'yes' WHEN flag = false THEN 'no' ELSE 'unknown' END FROM types_table",
        ),
        (
            "SELECT id, CAST(id AS DOUBLE), CAST(flag AS INTEGER), CAST(flag AS VARCHAR) FROM types_table",
            "SELECT id, CAST(id AS DOUBLE), CAST(flag AS INTEGER), CAST(flag AS VARCHAR) FROM types_table",
        ),
    ];
    for (case_id, (dodam_template, duckdb_template)) in cases.into_iter().enumerate() {
        let dodam_sql =
            dodam_template.replace("types_table", &format!("'{}'", types_path.display()));
        let duckdb_sql = duckdb_template.replace(
            "types_table",
            &format!("read_parquet('{}')", types_path.display()),
        );
        assert_same_as_duckdb_unordered_case(
            &format!("type_matrix case={case_id}"),
            &dodam_sql,
            &duckdb_sql,
            tempdir.path(),
        )
        .await;
    }

    assert_both_error(
        &format!(
            "SELECT CAST(note AS INTEGER) FROM '{}' WHERE note IS NOT NULL",
            types_path.display()
        ),
        &format!(
            "SELECT CAST(note AS INTEGER) FROM read_parquet('{}') WHERE note IS NOT NULL",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(event_date AS VARCHAR), CAST(created_at AS VARCHAR) FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(event_date AS VARCHAR), CAST(created_at AS VARCHAR) FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(amount AS VARCHAR), COALESCE(CAST(amount AS VARCHAR), 'missing') FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(amount AS VARCHAR), COALESCE(CAST(amount AS VARCHAR), 'missing') FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(amount + amount AS VARCHAR), CAST(amount - amount AS VARCHAR) FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(amount + amount AS VARCHAR), CAST(amount - amount AS VARCHAR) FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(amount + amount3 AS VARCHAR), CAST(amount3 - amount AS VARCHAR) FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(amount + amount3 AS VARCHAR), CAST(amount3 - amount AS VARCHAR) FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CASE WHEN amount = amount3 THEN 'eq' WHEN amount3 > amount THEN 'gt' ELSE 'no' END FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CASE WHEN amount = amount3 THEN 'eq' WHEN amount3 > amount THEN 'gt' ELSE 'no' END FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id FROM '{}' WHERE amount + amount > '0.00' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id FROM read_parquet('{}') WHERE amount + amount > '0.00' ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(CAST('2024-01-02' AS DATE) AS VARCHAR), CAST(CAST('2024-01-02 03:04:05' AS TIMESTAMP) AS VARCHAR), CAST(CAST(created_at AS DATE) AS VARCHAR), CAST(CAST(event_date AS TIMESTAMP) AS VARCHAR) FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(CAST('2024-01-02' AS DATE) AS VARCHAR), CAST(CAST('2024-01-02 03:04:05' AS TIMESTAMP) AS VARCHAR), CAST(CAST(created_at AS DATE) AS VARCHAR), CAST(CAST(event_date AS TIMESTAMP) AS VARCHAR) FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(DATE '2024-01-03' AS VARCHAR), CAST(TIMESTAMP '2024-01-03 04:05:06.789' AS VARCHAR) FROM '{}' ORDER BY id",
            types_path.display()
        ),
        &format!(
            "SELECT id, CAST(DATE '2024-01-03' AS VARCHAR), CAST(TIMESTAMP '2024-01-03 04:05:06.789' AS VARCHAR) FROM read_parquet('{}') ORDER BY id",
            types_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_aggregate_with_subquery_predicate() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT key, count(*), sum(value) FROM '{}' WHERE id IN (SELECT id FROM '{}' WHERE key = 3) GROUP BY key ORDER BY key",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT key, count(*), sum(value) FROM read_parquet('{}') WHERE id IN (SELECT id FROM read_parquet('{}') WHERE key = 3) GROUP BY key ORDER BY key",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT key, count(*), sum(value) FROM '{}' WHERE id <= (SELECT max(id) FROM '{}' WHERE key = 2) GROUP BY key ORDER BY key",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT key, count(*), sum(value) FROM read_parquet('{}') WHERE id <= (SELECT max(id) FROM read_parquet('{}') WHERE key = 2) GROUP BY key ORDER BY key",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT key, count(*), sum(value) FROM '{}' WHERE key NOT IN (SELECT key FROM '{}' WHERE key IS NULL OR key = 3) OR key IS NULL GROUP BY key ORDER BY key",
            facts_path.display(),
            facts_path.display()
        ),
        &format!(
            "SELECT key, count(*), sum(value) FROM read_parquet('{}') WHERE key NOT IN (SELECT key FROM read_parquet('{}') WHERE key IS NULL OR key = 3) OR key IS NULL GROUP BY key ORDER BY key",
            facts_path.display(),
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_copy_parquet_readback() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let output_path = tempdir.path().join("copy-output.parquet");
    write_facts_parquet(&facts_path);

    let copy_sql = format!(
        "COPY (SELECT key, count(*), sum(value), min(payload), max(payload) FROM '{}' GROUP BY key ORDER BY key) TO '{}' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE 2, DICTIONARY true)",
        facts_path.display(),
        output_path.display()
    );
    run_dodam_copy(&copy_sql).await;

    assert_same_as_duckdb(
        &format!("SELECT * FROM '{}' ORDER BY key", output_path.display()),
        &format!(
            "SELECT * FROM read_parquet('{}') ORDER BY key",
            output_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_copy_parquet_interop_options() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    let options = [
        (
            "snappy_dict",
            "COMPRESSION snappy, ROW_GROUP_SIZE 2, DICTIONARY true",
        ),
        (
            "uncompressed_plain",
            "COMPRESSION uncompressed, ROW_GROUP_SIZE 4, DICTIONARY false",
        ),
        (
            "zstd_dict",
            "COMPRESSION zstd, ROW_GROUP_SIZE 2, DICTIONARY true, WRITE_BATCH_SIZE 2, DATA_PAGE_ROW_COUNT_LIMIT 2",
        ),
    ];
    for (name, options) in options {
        let dodam_output = tempdir.path().join(format!("dodam-{name}.parquet"));
        let copy_sql = format!(
            "COPY (SELECT id, key, value, payload FROM '{}' ORDER BY id) TO '{}' (FORMAT parquet, {options})",
            facts_path.display(),
            dodam_output.display()
        );
        run_dodam_copy(&copy_sql).await;
        assert_same_as_duckdb(
            &format!(
                "SELECT id, key, value, payload FROM '{}' ORDER BY id",
                dodam_output.display()
            ),
            &format!(
                "SELECT id, key, value, payload FROM read_parquet('{}') ORDER BY id",
                dodam_output.display()
            ),
            tempdir.path(),
        )
        .await;
    }

    let duckdb_output = tempdir.path().join("duckdb-zstd.parquet");
    run_duckdb_command(&format!(
        "COPY (SELECT id, key, value, payload FROM read_parquet('{}') ORDER BY id) TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
        facts_path.display(),
        duckdb_output.display()
    ));
    assert_same_as_duckdb(
        &format!(
            "SELECT id, key, value, payload FROM '{}' ORDER BY id",
            duckdb_output.display()
        ),
        &format!(
            "SELECT id, key, value, payload FROM read_parquet('{}') ORDER BY id",
            duckdb_output.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_copy_parquet_type_schema_fidelity() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let input_path = tempdir.path().join("types-input.parquet");
    let output_path = tempdir.path().join("types-output.parquet");
    write_types_parquet(&input_path);

    let copy_sql = format!(
        "COPY (SELECT id, amount, event_date, created_at, flag FROM '{}' ORDER BY id) TO '{}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 2)",
        input_path.display(),
        output_path.display()
    );
    run_dodam_copy(&copy_sql).await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, amount, event_date, flag FROM '{}' ORDER BY id",
            output_path.display()
        ),
        &format!(
            "SELECT id, amount, event_date, flag FROM read_parquet('{}') ORDER BY id",
            output_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, CAST(created_at AS VARCHAR) FROM '{}' ORDER BY id",
            output_path.display()
        ),
        &format!(
            "SELECT id, CAST(created_at AS VARCHAR) FROM read_parquet('{}') ORDER BY id",
            output_path.display()
        ),
        tempdir.path(),
    )
    .await;

    let QueryOutput::Scan { batches } = execute_sql(
        &DodamEngine::default(),
        &format!(
            "SELECT id, amount, event_date, created_at, flag FROM '{}' ORDER BY id",
            output_path.display()
        ),
        BATCH_SIZE,
    )
    .await
    .expect("read typed parquet copy") else {
        panic!("expected scan output");
    };
    let schema = batches.first().expect("typed output batch").schema();
    assert!(matches!(
        schema.field(1).data_type(),
        DataType::Decimal128(10, 2)
    ));
    assert!(matches!(schema.field(2).data_type(), DataType::Date32));
    assert!(matches!(
        schema.field(3).data_type(),
        DataType::Timestamp(TimeUnit::Millisecond, None)
    ));
    assert!(matches!(schema.field(4).data_type(), DataType::Boolean));
}

#[tokio::test]
async fn duckdb_differential_unordered_result_policy() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    write_facts_parquet(&facts_path);

    assert_same_as_duckdb_unordered_case(
        "unordered aggregate result uses sorted multiset comparison",
        &format!(
            "SELECT key, count(*), sum(value) FROM '{}' GROUP BY key",
            facts_path.display()
        ),
        &format!(
            "SELECT key, count(*), sum(value) FROM read_parquet('{}') GROUP BY key",
            facts_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn dodam_copy_parquet_nested_roundtrip() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let input_path = tempdir.path().join("nested-input.parquet");
    let output_path = tempdir.path().join("nested-output.parquet");
    write_nested_values_parquet(&input_path);

    let copy_sql = format!(
        "COPY (SELECT id, tags, attrs FROM '{}' ORDER BY id) TO '{}' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE 2)",
        input_path.display(),
        output_path.display()
    );
    run_dodam_copy(&copy_sql).await;

    let output = run_dodam(&format!(
        "SELECT id, tags, attrs FROM '{}' ORDER BY id",
        output_path.display()
    ))
    .await;
    let input = run_dodam(&format!(
        "SELECT id, tags, attrs FROM '{}' ORDER BY id",
        input_path.display()
    ))
    .await;
    assert_eq!(output, input);

    let QueryOutput::Scan { batches } = execute_sql(
        &DodamEngine::default(),
        &format!(
            "SELECT id, tags, attrs FROM '{}' ORDER BY id",
            output_path.display()
        ),
        BATCH_SIZE,
    )
    .await
    .expect("read nested parquet copy") else {
        panic!("expected scan output");
    };
    let schema = batches.first().expect("nested output batch").schema();
    assert!(matches!(schema.field(1).data_type(), DataType::List(_)));
    assert!(matches!(schema.field(2).data_type(), DataType::Struct(_)));
}

#[tokio::test]
async fn duckdb_differential_nested_struct_field_projection() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    let tempdir = tempfile::tempdir().expect("tempdir");
    let input_path = tempdir.path().join("nested-input.parquet");
    write_nested_values_parquet(&input_path);

    assert_same_as_duckdb(
        &format!(
            "SELECT id, attrs.rank AS rank, attrs.label AS label FROM '{}' ORDER BY id",
            input_path.display()
        ),
        &format!(
            "SELECT id, attrs.rank AS rank, attrs.label AS label FROM read_parquet('{}') ORDER BY id",
            input_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, attrs.detail.score AS score, attrs.detail.code AS code FROM '{}' WHERE attrs.detail.score >= 20 OR attrs.detail.code = 'a' ORDER BY id",
            input_path.display()
        ),
        &format!(
            "SELECT id, attrs.detail.score AS score, attrs.detail.code AS code FROM read_parquet('{}') WHERE attrs.detail.score >= 20 OR attrs.detail.code = 'a' ORDER BY id",
            input_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, array_length(tags) AS tag_count, tags[1] AS first_tag FROM '{}' WHERE attrs.rank > 1 OR tags[1] = 10 ORDER BY id",
            input_path.display()
        ),
        &format!(
            "SELECT id, array_length(tags) AS tag_count, tags[1] AS first_tag FROM read_parquet('{}') WHERE attrs.rank > 1 OR tags[1] = 10 ORDER BY id",
            input_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, tags[array_length(tags)] AS last_tag FROM '{}' WHERE tags[array_length(tags)] = 2 OR attrs.rank > 30 ORDER BY id",
            input_path.display()
        ),
        &format!(
            "SELECT id, tags[array_length(tags)] AS last_tag FROM read_parquet('{}') WHERE tags[array_length(tags)] = 2 OR attrs.rank > 30 ORDER BY id",
            input_path.display()
        ),
        tempdir.path(),
    )
    .await;

    assert_same_as_duckdb(
        &format!(
            "SELECT id, array_length(attrs.more_tags) AS nested_tag_count FROM '{}' WHERE array_length(attrs.more_tags) > 1 OR attrs.detail.score >= 40 ORDER BY id",
            input_path.display()
        ),
        &format!(
            "SELECT id, array_length(attrs.more_tags) AS nested_tag_count FROM read_parquet('{}') WHERE array_length(attrs.more_tags) > 1 OR attrs.detail.score >= 40 ORDER BY id",
            input_path.display()
        ),
        tempdir.path(),
    )
    .await;
}

#[tokio::test]
async fn duckdb_differential_long_run_seeded_randomized() {
    let Some(_duckdb) = DuckDbGuard::new() else {
        return;
    };
    if std::env::var("DODAM_LONG_DIFF").ok().as_deref() != Some("1") {
        eprintln!("set DODAM_LONG_DIFF=1 to run the long randomized differential suite");
        return;
    }
    let tempdir = tempfile::tempdir().expect("tempdir");
    let facts_path = tempdir.path().join("facts.parquet");
    let dim_path = tempdir.path().join("dim.parquet");
    let types_path = tempdir.path().join("types.parquet");
    let multi_left_path = tempdir.path().join("multi-left.parquet");
    let multi_right_path = tempdir.path().join("multi-right.parquet");
    write_facts_parquet(&facts_path);
    write_dim_parquet(&dim_path);
    write_types_parquet(&types_path);
    write_multi_left_parquet(&multi_left_path);
    write_multi_right_parquet(&multi_right_path);

    let seeds = long_diff_seeds();
    let case_count = long_diff_case_count();
    let only_case = long_diff_only_case();
    for seed in seeds {
        let mut rng = TestRng::new(seed);
        for case_id in 0..case_count {
            let case = match rng.index(3) {
                0 => random_facts_query(&mut rng, &facts_path, case_id),
                1 => random_types_query(&mut rng, &types_path, case_id),
                _ => {
                    if rng.chance(2, 3) {
                        random_single_key_join_query(&mut rng, &facts_path, &dim_path, case_id)
                    } else {
                        random_multi_key_join_query(
                            &mut rng,
                            &multi_left_path,
                            &multi_right_path,
                            case_id,
                        )
                    }
                }
            };
            if only_case.is_some_and(|only_case| only_case != case_id) {
                continue;
            }
            assert_same_as_duckdb_unordered_case(
                &format!("long_diff seed={seed:#x} case={case_id}"),
                &case.dodam_sql,
                &case.duckdb_sql,
                tempdir.path(),
            )
            .await;
        }
    }
}

fn long_diff_seeds() -> Vec<u64> {
    if let Ok(seed) = std::env::var("DODAM_LONG_DIFF_SEED") {
        return vec![parse_long_diff_seed(&seed)];
    }
    if let Ok(seeds) = std::env::var("DODAM_LONG_DIFF_SEEDS") {
        let parsed = seeds
            .split(',')
            .filter(|seed| !seed.trim().is_empty())
            .map(parse_long_diff_seed)
            .collect::<Vec<_>>();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    vec![
        0xD0DA_2026_1001,
        0xD0DA_2026_1002,
        0xD0DA_2026_1003,
        0xD0DA_2026_1004,
    ]
}

fn parse_long_diff_seed(raw: &str) -> u64 {
    let raw = raw.trim();
    if let Some(hex) = raw.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).expect("DODAM_LONG_DIFF_SEED must be a u64")
    } else {
        raw.parse::<u64>()
            .expect("DODAM_LONG_DIFF_SEED must be a u64")
    }
}

fn long_diff_case_count() -> usize {
    std::env::var("DODAM_LONG_DIFF_CASES")
        .ok()
        .map(|raw| {
            raw.parse::<usize>()
                .expect("DODAM_LONG_DIFF_CASES must be a positive integer")
        })
        .filter(|cases| *cases > 0)
        .unwrap_or(192)
}

fn long_diff_only_case() -> Option<usize> {
    std::env::var("DODAM_LONG_DIFF_CASE").ok().map(|raw| {
        raw.parse::<usize>()
            .expect("DODAM_LONG_DIFF_CASE must be an integer")
    })
}

#[tokio::test]
async fn dodam_copy_csv_header_and_escaping() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let input_path = tempdir.path().join("csv-values.parquet");
    let output_path = tempdir.path().join("copy-output.csv");
    write_csv_values_parquet(&input_path);

    let copy_sql = format!(
        "COPY (SELECT id, text FROM '{}' ORDER BY id) TO '{}' (FORMAT csv, HEADER true)",
        input_path.display(),
        output_path.display()
    );
    run_dodam_copy(&copy_sql).await;

    let contents = std::fs::read_to_string(output_path).expect("read csv output");
    assert_eq!(
        contents,
        "id,text\n1,plain\n2,\"comma,value\"\n3,\"quote \"\"value\"\"\"\n4,\"line\nbreak\"\n5,\n"
    );
}

async fn assert_same_as_duckdb(dodam_sql: &str, duckdb_sql: &str, tempdir: &Path) {
    assert_same_as_duckdb_case("", dodam_sql, duckdb_sql, tempdir).await;
}

async fn assert_same_as_duckdb_case(
    case_name: &str,
    dodam_sql: &str,
    duckdb_sql: &str,
    tempdir: &Path,
) {
    let dodam_rows = run_dodam(dodam_sql).await;
    let duckdb_rows = run_duckdb(duckdb_sql, tempdir);
    assert_eq!(
        dodam_rows, duckdb_rows,
        "\nCase:\n{case_name}\n\nDodam SQL:\n{dodam_sql}\n\nDuckDB SQL:\n{duckdb_sql}"
    );
}

async fn assert_same_as_duckdb_unordered_case(
    case_name: &str,
    dodam_sql: &str,
    duckdb_sql: &str,
    tempdir: &Path,
) {
    let mut dodam_rows = run_dodam(dodam_sql).await;
    let mut duckdb_rows = run_duckdb(duckdb_sql, tempdir);
    dodam_rows.sort();
    duckdb_rows.sort();
    assert_eq!(
        dodam_rows, duckdb_rows,
        "\nCase:\n{case_name}\n\nDodam SQL:\n{dodam_sql}\n\nDuckDB SQL:\n{duckdb_sql}"
    );
}

async fn assert_both_error(dodam_sql: &str, duckdb_sql: &str, tempdir: &Path) {
    let dodam_error = run_dodam_result(dodam_sql)
        .await
        .expect_err("Dodam query should fail");
    let duckdb_error =
        run_duckdb_result(duckdb_sql, tempdir).expect_err("DuckDB query should fail");
    assert!(
        !dodam_error.is_empty() && !duckdb_error.is_empty(),
        "\nDodam SQL:\n{dodam_sql}\n\nDuckDB SQL:\n{duckdb_sql}"
    );
}

async fn assert_dodam_unknown_column(sql: &str, expected_column: &str) {
    let error = run_dodam_error(sql).await;
    match error {
        DodamError::UnknownColumn(column) => assert_eq!(column, expected_column, "\nSQL:\n{sql}"),
        other => panic!("expected UnknownColumn({expected_column}), got {other}\n\nSQL:\n{sql}"),
    }
}

async fn assert_dodam_error_contains(sql: &str, expected: &str) {
    let error = run_dodam_error(sql).await.to_string();
    assert!(
        error.contains(expected),
        "expected Dodam error to contain {expected:?}, got {error:?}\n\nSQL:\n{sql}"
    );
}

async fn assert_dodam_copy_error_contains(sql: &str, expected: &str) {
    let error = run_dodam_copy_result(sql)
        .await
        .expect_err("Dodam COPY should fail");
    assert!(
        error.contains(expected),
        "expected Dodam COPY error to contain {expected:?}, got {error:?}\n\nSQL:\n{sql}"
    );
}

async fn run_dodam(sql: &str) -> Vec<String> {
    run_dodam_result(sql)
        .await
        .unwrap_or_else(|error| panic!("execute dodam sql failed:\n{sql}\n\n{error}"))
}

async fn run_dodam_error(sql: &str) -> DodamError {
    match execute_sql(&DodamEngine::default(), sql, BATCH_SIZE).await {
        Ok(_) => panic!("Dodam query should fail:\n{sql}"),
        Err(error) => error,
    }
}

async fn run_dodam_result(sql: &str) -> std::result::Result<Vec<String>, String> {
    let output = execute_sql(&DodamEngine::default(), sql, BATCH_SIZE)
        .await
        .map_err(|error| error.to_string())?;
    match output {
        QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => {
            Ok(canonical_rows(&batches))
        }
        QueryOutput::Explain { .. } => panic!("differential tests do not compare EXPLAIN output"),
    }
}

fn run_duckdb(sql: &str, tempdir: &Path) -> Vec<String> {
    run_duckdb_result(sql, tempdir).expect("run duckdb query")
}

fn run_duckdb_result(sql: &str, tempdir: &Path) -> std::result::Result<Vec<String>, String> {
    let output_path = tempdir.join("duckdb-output.tsv");
    let copy_sql = format!(
        "COPY ({sql}) TO '{}' (FORMAT CSV, HEADER false, DELIMITER '|', NULL '__DODAM_NULL__')",
        output_path.display()
    );
    let output = Command::new("duckdb")
        .args(["-csv", "-noheader", "-c", &copy_sql])
        .output()
        .expect("run duckdb");
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let contents = std::fs::read_to_string(output_path).map_err(|error| error.to_string())?;
    Ok(contents
        .lines()
        .map(|line| line.split('|').collect::<Vec<_>>().join("|"))
        .collect())
}

fn run_duckdb_command(sql: &str) {
    let output = Command::new("duckdb")
        .args(["-c", sql])
        .output()
        .expect("run duckdb command");
    assert!(
        output.status.success(),
        "duckdb command failed:\n{sql}\n\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

async fn run_dodam_copy(sql: &str) {
    let copy = parse_copy_to_select(sql)
        .expect("parse COPY")
        .expect("COPY statement");
    let mut sink = CopyFileQuerySink::new(
        &copy.path,
        copy.format,
        copy.header,
        copy.parquet_options,
        None,
        false,
    )
    .expect("create COPY sink");
    execute_sql_to_result_sink(
        &DodamEngine::default(),
        &copy.sql,
        BATCH_SIZE,
        &mut sink,
        SqlSinkExecutionOptions::default(),
    )
    .await
    .expect("execute COPY query");
    sink.finish().expect("finish COPY sink");
}

async fn run_dodam_copy_result(sql: &str) -> std::result::Result<(), String> {
    let Some(copy) = parse_copy_to_select(sql).map_err(|error| error.to_string())? else {
        return Err("expected COPY statement".to_string());
    };
    let mut sink = CopyFileQuerySink::new(
        &copy.path,
        copy.format,
        copy.header,
        copy.parquet_options,
        None,
        false,
    )
    .map_err(|error| error.to_string())?;
    execute_sql_to_result_sink(
        &DodamEngine::default(),
        &copy.sql,
        BATCH_SIZE,
        &mut sink,
        SqlSinkExecutionOptions::default(),
    )
    .await
    .map_err(|error| error.to_string())?;
    sink.finish().map_err(|error| error.to_string())
}

fn canonical_rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let values = batch
                .columns()
                .iter()
                .map(|column| {
                    if column.is_null(row) {
                        "__DODAM_NULL__".to_string()
                    } else {
                        array_value_to_string(column.as_ref(), row).expect("format arrow value")
                    }
                })
                .collect::<Vec<_>>();
            rows.push(values.join("|"));
        }
    }
    rows
}

struct GeneratedSql {
    case_id: usize,
    dodam_sql: String,
    duckdb_sql: String,
}

struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next_u64() as usize) % len
    }

    fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next_u64() % denominator < numerator
    }

    fn choose<'a>(&mut self, values: &'a [&'a str]) -> &'a str {
        values[self.index(values.len())]
    }
}

fn random_facts_query(rng: &mut TestRng, facts_path: &Path, case_id: usize) -> GeneratedSql {
    let dodam_table = format!("'{}'", facts_path.display());
    let duckdb_table = format!("read_parquet('{}')", facts_path.display());

    let aggregate = rng.chance(1, 4);
    let (dodam_predicate, duckdb_predicate) = random_facts_predicate(rng, facts_path, "f", true);
    let (dodam_sql, duckdb_sql) = if aggregate {
        (
            format!(
                "SELECT key, count(*), count(value), sum(value), avg(value), min(payload), max(payload) FROM {dodam_table} f WHERE {dodam_predicate} GROUP BY key ORDER BY key"
            ),
            format!(
                "SELECT key, count(*), count(value), sum(value), avg(value), min(payload), max(payload) FROM {duckdb_table} f WHERE {duckdb_predicate} GROUP BY key ORDER BY key"
            ),
        )
    } else {
        let projection = rng.choose(&[
            "id, key, value, payload",
            "id, key, COALESCE(payload, 'missing') AS payload_text",
            "id, id + 1 AS id_plus, value",
            "id, value * 2 AS doubled_value, payload",
            "id, CASE WHEN key IS NULL THEN 'null-key' WHEN key = 2 THEN 'two' ELSE 'other' END AS key_class",
        ]);
        (
            format!("SELECT {projection} FROM {dodam_table} f WHERE {dodam_predicate} ORDER BY id"),
            format!(
                "SELECT {projection} FROM {duckdb_table} f WHERE {duckdb_predicate} ORDER BY id"
            ),
        )
    };

    GeneratedSql {
        case_id,
        dodam_sql,
        duckdb_sql,
    }
}

fn random_facts_predicate(
    rng: &mut TestRng,
    facts_path: &Path,
    outer_alias: &str,
    allow_subquery: bool,
) -> (String, String) {
    let dodam_table = format!("'{}'", facts_path.display());
    let duckdb_table = format!("read_parquet('{}')", facts_path.display());
    let base_predicates = [
        "true",
        "key IS NULL",
        "key IS NOT NULL",
        "value IS NULL OR key = 2",
        "payload IS NOT NULL AND NOT (key = 2)",
        "key IN (1, 3)",
        "key IN (2, NULL) OR id = 1",
        "key NOT IN (1, NULL) OR id = 1",
        "id >= 2 AND id <= 5",
        "COALESCE(payload, 'missing') <> 'missing'",
    ];
    let subquery_predicates = [
        "id IN (SELECT id FROM facts_sub WHERE key = 3)",
        "id = (SELECT key FROM facts_sub WHERE key = 3 ORDER BY id LIMIT 1)",
        "EXISTS (SELECT 1 FROM facts_sub f2 WHERE f2.key = outer_key AND f2.value IS NOT NULL)",
    ];
    let predicate = if allow_subquery && rng.chance(1, 3) {
        rng.choose(&subquery_predicates)
    } else {
        rng.choose(&base_predicates)
    };
    (
        predicate
            .replace("facts_sub", &dodam_table)
            .replace("outer_key", &format!("{outer_alias}.key")),
        predicate
            .replace("facts_sub", &duckdb_table)
            .replace("outer_key", &format!("{outer_alias}.key")),
    )
}

fn random_single_key_join_query(
    rng: &mut TestRng,
    facts_path: &Path,
    dim_path: &Path,
    case_id: usize,
) -> GeneratedSql {
    let dodam_left = format!("'{}'", facts_path.display());
    let dodam_right = format!("'{}'", dim_path.display());
    let duckdb_left = format!("read_parquet('{}')", facts_path.display());
    let duckdb_right = format!("read_parquet('{}')", dim_path.display());
    let join = rng.choose(&["JOIN", "INNER JOIN", "LEFT JOIN", "FULL OUTER JOIN"]);
    let predicate = rng.choose(&[
        "true",
        "f.payload IS NOT NULL",
        "d.name IS NULL OR d.name <> 'four'",
        "f.value IS NULL OR f.value >= 0",
        "f.key IS NULL OR d.key IS NOT NULL",
    ]);
    let aggregate = rng.chance(1, 3);
    let (dodam_sql, duckdb_sql) = if aggregate {
        (
            format!(
                "SELECT d.name, count(*), count(f.value), sum(f.value) FROM {dodam_left} f {join} {dodam_right} d ON f.key = d.key WHERE {predicate} GROUP BY d.name"
            ),
            format!(
                "SELECT d.name, count(*), count(f.value), sum(f.value) FROM {duckdb_left} f {join} {duckdb_right} d ON f.key = d.key WHERE {predicate} GROUP BY d.name"
            ),
        )
    } else {
        let projection = rng.choose(&[
            "f.id, f.key, d.name",
            "f.id, f.payload, d.name",
            "f.id, f.value, d.name",
            "f.id, COALESCE(d.name, 'missing') AS dim_name, f.payload",
            "f.id, f.value * 2 AS doubled_value, d.name",
            "f.id, CASE WHEN d.name IS NULL THEN 'unmatched' ELSE 'matched' END AS match_state",
        ]);
        (
            format!(
                "SELECT {projection} FROM {dodam_left} f {join} {dodam_right} d ON f.key = d.key WHERE {predicate}"
            ),
            format!(
                "SELECT {projection} FROM {duckdb_left} f {join} {duckdb_right} d ON f.key = d.key WHERE {predicate}"
            ),
        )
    };
    GeneratedSql {
        case_id,
        dodam_sql,
        duckdb_sql,
    }
}

fn random_multi_key_join_query(
    rng: &mut TestRng,
    left_path: &Path,
    right_path: &Path,
    case_id: usize,
) -> GeneratedSql {
    let dodam_left = format!("'{}'", left_path.display());
    let dodam_right = format!("'{}'", right_path.display());
    let duckdb_left = format!("read_parquet('{}')", left_path.display());
    let duckdb_right = format!("read_parquet('{}')", right_path.display());
    let join = rng.choose(&["JOIN", "INNER JOIN", "LEFT JOIN"]);
    let predicate = rng.choose(&[
        "true",
        "l.k2 IS NOT NULL",
        "r.label IS NULL OR r.label <> 'null-k2'",
        "l.k1 IS NULL OR r.k1 IS NOT NULL",
    ]);
    let projection = rng.choose(&[
        "l.id, l.k1, l.k2, r.label",
        "l.id, r.label",
        "l.id, COALESCE(r.label, 'missing') AS label_text",
        "l.id, CASE WHEN r.label IS NULL THEN 'unmatched' ELSE 'matched' END AS match_state",
    ]);
    GeneratedSql {
        case_id,
        dodam_sql: format!(
            "SELECT {projection} FROM {dodam_left} l {join} {dodam_right} r ON l.k1 = r.k1 AND l.k2 = r.k2 WHERE {predicate}"
        ),
        duckdb_sql: format!(
            "SELECT {projection} FROM {duckdb_left} l {join} {duckdb_right} r ON l.k1 = r.k1 AND l.k2 = r.k2 WHERE {predicate}"
        ),
    }
}

fn random_types_query(rng: &mut TestRng, types_path: &Path, case_id: usize) -> GeneratedSql {
    let dodam_table = format!("'{}'", types_path.display());
    let duckdb_table = format!("read_parquet('{}')", types_path.display());
    let predicate = rng.choose(&[
        "true",
        "flag = true OR flag IS NULL",
        "score >= -1.0",
        "amount >= '-7.0000' AND amount <= '123.4500'",
        "created_at >= '2024-01-02 00:00:00' OR created_at IS NULL",
        "created_at_utc = '1970-01-01 09:00:00+09:00' OR id = 1",
        "event_date < '2024-01-02' OR event_date IS NULL",
        "substring(note FROM 1 FOR 1) IN ('a', 'g') OR id = 4",
        "COALESCE(note, 'fallback') <> 'fallback'",
    ]);
    let projection = rng.choose(&[
        "id, flag, score, note, amount, event_date",
        "id, COALESCE(note, 'fallback') AS note_text",
        "id, id + 10 AS plus_ten, CAST(id AS VARCHAR) AS id_text",
        "id, lower(note) AS lower_note, upper(note) AS upper_note, length(note) AS note_len",
        "id, CASE WHEN amount IS NULL THEN 'missing' WHEN amount < 0 THEN 'negative' ELSE 'nonnegative' END AS amount_class",
    ]);
    GeneratedSql {
        case_id,
        dodam_sql: format!("SELECT {projection} FROM {dodam_table} WHERE {predicate} ORDER BY id"),
        duckdb_sql: format!(
            "SELECT {projection} FROM {duckdb_table} WHERE {predicate} ORDER BY id"
        ),
    }
}

fn write_facts_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, true),
        Field::new("value", DataType::Int64, true),
        Field::new("payload", DataType::Utf8, true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4, 5, 6]);
    let keys = Int32Array::from(vec![Some(1), Some(2), None, Some(2), Some(3), Some(3)]);
    let values = Int64Array::from(vec![Some(10), Some(20), Some(30), None, Some(-5), Some(7)]);
    let payloads = StringArray::from(vec![
        Some("a"),
        Some("b"),
        None,
        Some("d"),
        Some("e"),
        Some("f"),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(ids),
            Arc::new(keys),
            Arc::new(values),
            Arc::new(payloads),
        ],
    );
}

fn write_dim_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let keys = Int32Array::from_iter_values([1, 2, 2, 4]);
    let names = StringArray::from_iter_values(["one", "two-a", "two-b", "four"]);
    write_parquet(path, schema, vec![Arc::new(keys), Arc::new(names)]);
}

fn write_like_values_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4, 5, 6]);
    let text = StringArray::from(vec![
        Some("alpha"),
        Some("alphabet"),
        Some("special requests"),
        Some("100% match"),
        Some("aleph"),
        None,
    ]);
    write_parquet(path, schema, vec![Arc::new(ids), Arc::new(text)]);
}

fn write_multi_left_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Utf8, true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4, 5]);
    let k1 = Int32Array::from(vec![Some(1), Some(1), Some(2), Some(2), None]);
    let k2 = StringArray::from(vec![Some("a"), Some("b"), Some("a"), None, Some("x")]);
    write_parquet(
        path,
        schema,
        vec![Arc::new(ids), Arc::new(k1), Arc::new(k2)],
    );
}

fn write_multi_right_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Utf8, true),
        Field::new("label", DataType::Utf8, false),
    ]));
    let k1 = Int32Array::from(vec![Some(1), Some(1), Some(2), Some(2), None]);
    let k2 = StringArray::from(vec![Some("a"), Some("a"), Some("a"), None, Some("x")]);
    let labels = StringArray::from_iter_values(["ra", "ra2", "rb", "null-k2", "null-k1"]);
    write_parquet(
        path,
        schema,
        vec![Arc::new(k1), Arc::new(k2), Arc::new(labels)],
    );
}

fn write_types_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("flag", DataType::Boolean, true),
        Field::new("score", DataType::Float64, true),
        Field::new("note", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(10, 2), true),
        Field::new("amount3", DataType::Decimal128(10, 3), true),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new(
            "created_at_utc",
            DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
            true,
        ),
        Field::new("event_date", DataType::Date32, true),
        Field::new("event_date64", DataType::Date64, true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4]);
    let flags = BooleanArray::from(vec![Some(true), Some(false), None, Some(true)]);
    let scores = Float64Array::from(vec![Some(1.5), Some(-1.0), None, Some(3.25)]);
    let notes = StringArray::from(vec![Some("alpha"), None, Some("gamma"), Some("")]);
    let amounts = Decimal128Array::from(vec![Some(12345), Some(-700), None, Some(0)])
        .with_precision_and_scale(10, 2)
        .expect("decimal precision");
    let amount3 = Decimal128Array::from(vec![Some(123450), Some(-7000), None, Some(5)])
        .with_precision_and_scale(10, 3)
        .expect("decimal precision");
    let created_at = TimestampMillisecondArray::from(vec![
        Some(1_704_067_200_000),
        Some(1_704_153_600_000),
        None,
        Some(0),
    ]);
    let created_at_utc = TimestampMillisecondArray::from(vec![
        Some(1_704_067_200_000),
        Some(1_704_153_600_000),
        None,
        Some(0),
    ])
    .with_timezone("+00:00");
    let event_date = Date32Array::from(vec![Some(19_723), Some(19_724), None, Some(0)]);
    let event_date64 = Date64Array::from(vec![
        Some(1_704_067_200_000),
        Some(1_704_153_600_000),
        None,
        Some(0),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(ids),
            Arc::new(flags),
            Arc::new(scores),
            Arc::new(notes),
            Arc::new(amounts),
            Arc::new(amount3),
            Arc::new(created_at),
            Arc::new(created_at_utc),
            Arc::new(event_date),
            Arc::new(event_date64),
        ],
    );
}

fn write_aggregate_nulls_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    let groups = Int32Array::from_iter_values([1, 1, 2, 2, 3]);
    let values = Int64Array::from(vec![None, None, Some(10), None, Some(-5)]);
    let labels = StringArray::from(vec![None, None, Some("b"), Some("a"), None]);
    write_parquet(
        path,
        schema,
        vec![Arc::new(groups), Arc::new(values), Arc::new(labels)],
    );
}

fn write_csv_values_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4, 5]);
    let text = StringArray::from(vec![
        Some("plain"),
        Some("comma,value"),
        Some("quote \"value\""),
        Some("line\nbreak"),
        None,
    ]);
    write_parquet(path, schema, vec![Arc::new(ids), Arc::new(text)]);
}

fn write_nested_values_parquet(path: &Path) {
    let list_field = Arc::new(Field::new("item", DataType::Int32, true));
    let attrs_fields = vec![
        Field::new("rank", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
        Field::new(
            "detail",
            DataType::Struct(
                vec![
                    Field::new("score", DataType::Int32, true),
                    Field::new("code", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        ),
        Field::new(
            "more_tags",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
    ]
    .into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", DataType::List(list_field), true),
        Field::new("attrs", DataType::Struct(attrs_fields), true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4]);
    let tags = ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(1), Some(2)]),
        None,
        Some(vec![None, Some(4)]),
        Some(vec![]),
    ]);
    let more_tags = ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(9), Some(8), Some(7)]),
        Some(vec![Some(5)]),
        None,
        Some(vec![]),
    ]);
    let attrs = StructArray::from(vec![
        (
            Arc::new(Field::new("rank", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40)]))
                as Arc<dyn arrow::array::Array>,
        ),
        (
            Arc::new(Field::new("label", DataType::Utf8, true)),
            Arc::new(StringArray::from(vec![
                Some("hot"),
                Some("cold"),
                None,
                Some("flat"),
            ])) as Arc<dyn arrow::array::Array>,
        ),
        (
            Arc::new(Field::new(
                "detail",
                DataType::Struct(
                    vec![
                        Field::new("score", DataType::Int32, true),
                        Field::new("code", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            )),
            Arc::new(StructArray::from(vec![
                (
                    Arc::new(Field::new("score", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![Some(7), Some(20), None, Some(40)]))
                        as Arc<dyn arrow::array::Array>,
                ),
                (
                    Arc::new(Field::new("code", DataType::Utf8, true)),
                    Arc::new(StringArray::from(vec![
                        Some("a"),
                        Some("b"),
                        None,
                        Some("z"),
                    ])) as Arc<dyn arrow::array::Array>,
                ),
            ])) as Arc<dyn arrow::array::Array>,
        ),
        (
            Arc::new(Field::new(
                "more_tags",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            )),
            Arc::new(more_tags) as Arc<dyn arrow::array::Array>,
        ),
    ]);
    write_parquet(
        path,
        schema,
        vec![Arc::new(ids), Arc::new(tags), Arc::new(attrs)],
    );
}

fn write_tpch_lineitem_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int32, false),
        Field::new("l_returnflag", DataType::Utf8, false),
        Field::new("l_linestatus", DataType::Utf8, false),
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Int64, false),
        Field::new("l_discount", DataType::Int64, false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));
    let orderkeys = Int32Array::from_iter_values([1, 1, 2, 3, 4, 5, 6, 7]);
    let returnflags = StringArray::from_iter_values(["N", "N", "R", "A", "N", "R", "A", "N"]);
    let linestatuses = StringArray::from_iter_values(["O", "O", "F", "F", "O", "F", "F", "O"]);
    let quantities = Int64Array::from_iter_values([10, 20, 15, 30, 5, 22, 40, 12]);
    let extendedprices =
        Int64Array::from_iter_values([1000, 2000, 1500, 3000, 500, 2200, 4000, 1200]);
    let discounts = Int64Array::from_iter_values([5, 7, 6, 3, 8, 6, 4, 6]);
    let shipdates = Date32Array::from_iter_values([
        date_days(1994, 1, 15),
        date_days(1994, 6, 30),
        date_days(1995, 3, 10),
        date_days(1998, 9, 2),
        date_days(1998, 9, 3),
        date_days(1994, 12, 31),
        date_days(1993, 12, 31),
        date_days(1994, 2, 1),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(orderkeys),
            Arc::new(returnflags),
            Arc::new(linestatuses),
            Arc::new(quantities),
            Arc::new(extendedprices),
            Arc::new(discounts),
            Arc::new(shipdates),
        ],
    );
}

fn write_tpch_q6_lineitem_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Float64, false),
        Field::new("l_discount", DataType::Float64, false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));
    let quantities = Int64Array::from_iter_values([10, 20, 15, 30, 5, 22, 40, 12]);
    let extendedprices = Float64Array::from_iter_values([
        1000.0, 2000.0, 1500.0, 3000.0, 500.0, 2200.0, 4000.0, 1200.0,
    ]);
    let discounts =
        Float64Array::from_iter_values([0.05, 0.07, 0.06, 0.03, 0.08, 0.06, 0.04, 0.06]);
    let shipdates = Date32Array::from_iter_values([
        date_days(1994, 1, 15),
        date_days(1994, 6, 30),
        date_days(1995, 3, 10),
        date_days(1998, 9, 2),
        date_days(1998, 9, 3),
        date_days(1994, 12, 31),
        date_days(1993, 12, 31),
        date_days(1994, 2, 1),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(quantities),
            Arc::new(extendedprices),
            Arc::new(discounts),
            Arc::new(shipdates),
        ],
    );
}

fn write_tpch_orders_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_totalprice", DataType::Int64, false),
        Field::new("o_orderdate", DataType::Date32, false),
    ]));
    let orderkeys = Int32Array::from_iter_values([1, 2, 3, 4, 5, 6]);
    let custkeys = Int32Array::from_iter_values([10, 20, 10, 30, 40, 10]);
    let totalprices = Int64Array::from_iter_values([3000, 1500, 3000, 500, 2200, 4000]);
    let orderdates = Date32Array::from_iter_values([
        date_days(1995, 3, 15),
        date_days(1995, 3, 20),
        date_days(1994, 1, 1),
        date_days(1995, 4, 1),
        date_days(1994, 12, 31),
        date_days(1995, 3, 10),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(orderkeys),
            Arc::new(custkeys),
            Arc::new(totalprices),
            Arc::new(orderdates),
        ],
    );
}

fn write_tpch_customer_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_mktsegment", DataType::Utf8, false),
    ]));
    let custkeys = Int32Array::from_iter_values([10, 20, 30, 40]);
    let segments =
        StringArray::from_iter_values(["BUILDING", "AUTOMOBILE", "BUILDING", "FURNITURE"]);
    write_parquet(path, schema, vec![Arc::new(custkeys), Arc::new(segments)]);
}

fn date_days(year: i32, month: u32, day: u32) -> i32 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = i64::from(month);
    let day = i64::from(day);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i32::try_from(era * 146_097 + doe - 719_468).expect("date32 days")
}

fn write_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<ArrayRef>) {
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("record batch");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(3))
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

struct DuckDbGuard;

impl DuckDbGuard {
    fn new() -> Option<Self> {
        if Command::new("duckdb").arg("--version").output().is_ok() {
            Some(Self)
        } else {
            eprintln!("duckdb binary not found; skipping differential test");
            None
        }
    }
}
