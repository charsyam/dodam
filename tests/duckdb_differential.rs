use std::fs::File;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, StringArray, TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_cast::display::array_value_to_string;
use dodam::engine::DodamEngine;
use dodam::sql::{QueryOutput, execute_sql};
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

async fn assert_same_as_duckdb(dodam_sql: &str, duckdb_sql: &str, tempdir: &Path) {
    let dodam_rows = run_dodam(dodam_sql).await;
    let duckdb_rows = run_duckdb(duckdb_sql, tempdir);
    assert_eq!(
        dodam_rows, duckdb_rows,
        "\nDodam SQL:\n{dodam_sql}\n\nDuckDB SQL:\n{duckdb_sql}"
    );
}

async fn run_dodam(sql: &str) -> Vec<String> {
    let output = execute_sql(&DodamEngine::default(), sql, BATCH_SIZE)
        .await
        .expect("execute dodam sql");
    match output {
        QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => {
            canonical_rows(&batches)
        }
        QueryOutput::Explain { .. } => panic!("differential tests do not compare EXPLAIN output"),
    }
}

fn run_duckdb(sql: &str, tempdir: &Path) -> Vec<String> {
    let output_path = tempdir.join("duckdb-output.tsv");
    let copy_sql = format!(
        "COPY ({sql}) TO '{}' (FORMAT CSV, HEADER false, DELIMITER '|', NULL '__DODAM_NULL__')",
        output_path.display()
    );
    let output = Command::new("duckdb")
        .args(["-csv", "-noheader", "-c", &copy_sql])
        .output()
        .expect("run duckdb");
    assert!(
        output.status.success(),
        "duckdb failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let contents = std::fs::read_to_string(output_path).expect("read duckdb output");
    contents
        .lines()
        .map(|line| line.split('|').collect::<Vec<_>>().join("|"))
        .collect()
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
            Arc::new(created_at),
            Arc::new(created_at_utc),
            Arc::new(event_date),
            Arc::new(event_date64),
        ],
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
