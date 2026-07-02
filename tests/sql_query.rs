use std::fs::File;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, ListArray, StringArray, StructArray, TimestampMillisecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use dodam::catalog::PersistentCatalog;
use dodam::engine::{
    DodamEngine, JoinAlgorithm, JoinExecutionStrategy, JoinParquetRequest, JoinTableRequest,
};
use dodam::execution::{FilterExpr, JoinType, Projection, SortExpr, SortKey};
use dodam::plan::{
    LogicalPlan, PhysicalJoinStrategy, PhysicalOperator, PhysicalPlanner, PhysicalPlanningOptions,
    StagePlanner,
};
use dodam::sql::{QueryOutput, execute_sql};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

#[tokio::test]
async fn executes_projection_filter_order_limit_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE id >= 2 AND id < 5 ORDER BY id DESC LIMIT 2",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![4, 3]);
}

#[tokio::test]
async fn executes_derived_table_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM (SELECT id, payload FROM '{}' WHERE id >= 2) AS t WHERE t.payload = 'a' ORDER BY t.id DESC LIMIT 2",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute derived table sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![4, 2]);
}

#[tokio::test]
async fn executes_aggregate_over_derived_table_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT payload, count(*), min(id), max(id) FROM (SELECT id, payload FROM '{}' WHERE id >= 1) AS t GROUP BY payload ORDER BY payload",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute aggregate derived table sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b", "c"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![2, 2, 1]);
    assert_eq!(i64s_from_column(&batches, 2), vec![2, 1, 5]);
    assert_eq!(i64s_from_column(&batches, 3), vec![4, 3, 5]);
}

#[tokio::test]
async fn executes_distinct_over_derived_table_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT DISTINCT payload FROM (SELECT id, payload FROM '{}' WHERE id >= 1) AS t ORDER BY payload",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute distinct derived table sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b", "c"]);
}

#[tokio::test]
async fn executes_derived_table_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT l.id, r.payload FROM (SELECT id, payload FROM '{}' WHERE id < 4) AS l JOIN (SELECT id, payload FROM '{}' WHERE id >= 2) AS r ON l.id = r.id ORDER BY l.id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute derived table join sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![2, 3]);
    assert_eq!(strings_from_column(&batches, 1), vec!["a", "b"]);
}

#[tokio::test]
async fn executes_aggregate_over_derived_table_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT l.payload, count(*), min(r.id), max(r.id) FROM (SELECT id, payload FROM '{}' WHERE id < 5) AS l JOIN (SELECT id, payload FROM '{}' WHERE id >= 1) AS r ON l.payload = r.payload GROUP BY l.payload ORDER BY l.payload",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute aggregate derived table join sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![6, 4]);
    assert_eq!(i64s_from_column(&batches, 2), vec![2, 1]);
    assert_eq!(i64s_from_column(&batches, 3), vec![4, 3]);
}

#[tokio::test]
async fn executes_distinct_over_derived_table_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT DISTINCT l.payload FROM (SELECT id, payload FROM '{}' WHERE id < 5) AS l JOIN (SELECT id, payload FROM '{}' WHERE id >= 1) AS r ON l.payload = r.payload ORDER BY l.payload",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute distinct derived table join sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b"]);
}

#[tokio::test]
async fn executes_in_subquery_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE payload IN (SELECT payload FROM '{}' WHERE id >= 4) ORDER BY id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute IN subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 2, 4, 5]);
}

#[tokio::test]
async fn executes_in_subquery_with_sql_null_semantics() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_nullable_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE payload IN (SELECT payload FROM '{}' WHERE id >= 3) ORDER BY id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute IN subquery with NULL sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![4]);

    let sql = format!(
        "SELECT id FROM '{}' WHERE payload NOT IN (SELECT payload FROM '{}' WHERE id >= 3) ORDER BY id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute NOT IN subquery with NULL sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert!(ids_from_batches(&batches).is_empty());
}

#[tokio::test]
async fn executes_scalar_subquery_filter_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE payload = 'c')",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute scalar subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![5]);
}

#[tokio::test]
async fn executes_scalar_subquery_edge_semantics() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_nullable_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE payload = 'missing')",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute empty scalar subquery sql");
    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert!(ids_from_batches(&batches).is_empty());

    let sql = format!(
        "SELECT id FROM '{}' WHERE payload = (SELECT payload FROM '{}' WHERE id = 3)",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute NULL scalar subquery sql");
    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert!(ids_from_batches(&batches).is_empty());

    let sql = format!(
        "SELECT id FROM '{}' WHERE id = (SELECT id FROM '{}' WHERE id < 2)",
        path.display(),
        path.display()
    );
    let error = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect_err("multiple-row scalar subquery should fail");
    assert!(
        error
            .to_string()
            .contains("scalar subquery must return at most one row"),
        "{error}"
    );
}

#[tokio::test]
async fn executes_exists_subquery_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE EXISTS (SELECT id FROM '{}' WHERE payload = 'c') ORDER BY id LIMIT 2",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute EXISTS subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 1]);

    let sql = format!(
        "SELECT id FROM '{}' WHERE NOT EXISTS (SELECT id FROM '{}' WHERE payload = 'missing') ORDER BY id LIMIT 2",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute NOT EXISTS subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 1]);
}

#[tokio::test]
async fn executes_correlated_exists_subquery_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT l.id FROM '{}' l WHERE EXISTS (SELECT id FROM '{}' r WHERE r.payload = l.payload AND r.id >= 4) ORDER BY l.id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute correlated EXISTS subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 2, 4, 5]);

    let sql = format!(
        "SELECT l.id FROM '{}' l WHERE NOT EXISTS (SELECT id FROM '{}' r WHERE r.payload = l.payload AND r.id >= 4) ORDER BY l.id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute correlated NOT EXISTS subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![1, 3]);
}

#[tokio::test]
async fn executes_correlated_in_and_scalar_subquery_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT l.id FROM '{}' l WHERE l.id IN (SELECT r.id FROM '{}' r WHERE r.payload = l.payload AND r.id >= 4) ORDER BY l.id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute correlated IN subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![4, 5]);

    let sql = format!(
        "SELECT l.id FROM '{}' l WHERE l.id = (SELECT r.id FROM '{}' r WHERE r.payload = l.payload ORDER BY r.id LIMIT 1) ORDER BY l.id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute correlated scalar subquery sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 1, 5]);
}

#[tokio::test]
async fn executes_exists_subquery_inside_boolean_expression_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE id = 0 OR EXISTS (SELECT id FROM '{}' WHERE payload = 'missing') ORDER BY id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute uncorrelated EXISTS in OR sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0]);

    let sql = format!(
        "SELECT l.id FROM '{}' l WHERE l.id = 1 OR EXISTS (SELECT id FROM '{}' r WHERE r.payload = l.payload AND r.id >= 4) ORDER BY l.id",
        path.display(),
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute correlated EXISTS in OR sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 1, 2, 4, 5]);
}

#[tokio::test]
async fn executes_sql_against_registered_catalog_table() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let table_dir = tempdir.path().join("orders");
    std::fs::create_dir(&table_dir).expect("table dir");
    write_test_parquet(&table_dir.join("part-000.parquet"));
    PersistentCatalog::new(tempdir.path())
        .register_local_parquet("orders", &table_dir)
        .expect("register table");

    let engine = DodamEngine::default().with_catalog_root(tempdir.path());
    let output = execute_sql(
        &engine,
        "SELECT id FROM orders WHERE id >= 3 ORDER BY id LIMIT 2",
        2,
    )
    .await
    .expect("execute catalog sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![3, 4]);
}

#[tokio::test]
async fn prunes_registered_table_by_hive_partition_directory() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let table_dir = tempdir.path().join("events");
    let first_partition = table_dir.join("dt=2026-07-01");
    let second_partition = table_dir.join("dt=2026-07-02");
    std::fs::create_dir_all(&first_partition).expect("first partition");
    std::fs::create_dir_all(&second_partition).expect("second partition");
    write_test_parquet(&first_partition.join("part-000.parquet"));
    write_test_parquet(&second_partition.join("part-000.parquet"));
    PersistentCatalog::new(tempdir.path())
        .register_local_parquet("events", &table_dir)
        .expect("register table");

    let engine = DodamEngine::default().with_catalog_root(tempdir.path());
    let output = execute_sql(
        &engine,
        "SELECT id FROM events WHERE dt = '2026-07-01' ORDER BY id",
        2,
    )
    .await
    .expect("execute partition-pruned sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 1, 2, 3, 4, 5]);

    PersistentCatalog::new(tempdir.path())
        .refresh_table("events")
        .expect("refresh table");
    let output = execute_sql(
        &engine,
        "SELECT dt, count(*) AS n FROM events GROUP BY dt ORDER BY dt",
        2,
    )
    .await
    .expect("execute refreshed snapshot sql");
    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(
        strings_from_column(&batches, 0),
        vec!["2026-07-01".to_string(), "2026-07-02".to_string()]
    );
    assert_eq!(u64s_from_column(&batches, 1), vec![6, 6]);
}

#[tokio::test]
async fn materializes_hive_partition_columns_in_scan_and_group_by() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let table_dir = tempdir.path().join("events");
    let first_partition = table_dir.join("dt=2026-07-01");
    let second_partition = table_dir.join("dt=2026-07-02");
    std::fs::create_dir_all(&first_partition).expect("first partition");
    std::fs::create_dir_all(&second_partition).expect("second partition");
    write_test_parquet(&first_partition.join("part-000.parquet"));
    write_test_parquet(&second_partition.join("part-000.parquet"));
    PersistentCatalog::new(tempdir.path())
        .register_local_parquet("events", &table_dir)
        .expect("register table");

    let engine = DodamEngine::default().with_catalog_root(tempdir.path());
    let output = execute_sql(&engine, "SELECT dt FROM events WHERE id = 0 ORDER BY dt", 2)
        .await
        .expect("select partition column");
    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(
        strings_from_column(&batches, 0),
        vec!["2026-07-01".to_string(), "2026-07-02".to_string()]
    );

    let output = execute_sql(
        &engine,
        "SELECT dt, count(*) AS n FROM events GROUP BY dt ORDER BY dt",
        2,
    )
    .await
    .expect("group by partition column");
    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(
        strings_from_column(&batches, 0),
        vec!["2026-07-01".to_string(), "2026-07-02".to_string()]
    );
    assert_eq!(u64s_from_column(&batches, 1), vec![6, 6]);
}

#[tokio::test]
async fn registered_catalog_table_uses_fragment_snapshot() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let table_dir = tempdir.path().join("events");
    let first_partition = table_dir.join("dt=2026-07-01");
    let second_partition = table_dir.join("dt=2026-07-02");
    std::fs::create_dir_all(&first_partition).expect("first partition");
    write_test_parquet(&first_partition.join("part-000.parquet"));
    PersistentCatalog::new(tempdir.path())
        .register_local_parquet("events", &table_dir)
        .expect("register table");
    std::fs::create_dir_all(&second_partition).expect("second partition");
    write_test_parquet(&second_partition.join("part-000.parquet"));

    let engine = DodamEngine::default().with_catalog_root(tempdir.path());
    let output = execute_sql(&engine, "SELECT id FROM events ORDER BY id", 2)
        .await
        .expect("execute snapshot sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![0, 1, 2, 3, 4, 5]);
}

#[tokio::test]
async fn executes_alias_or_filter_and_multi_key_order_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT id AS selected_id FROM '{}' WHERE payload = 'a' OR id = 1 ORDER BY payload ASC, id DESC",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![4, 2, 0, 1]);
}

#[tokio::test]
async fn supports_table_alias_and_qualified_columns_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT t.id AS selected_id FROM '{}' AS t WHERE t.payload = 'a' ORDER BY t.id DESC LIMIT 2",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "selected_id");
    assert_eq!(ids_from_batches(&batches), vec![4, 2]);
}

#[tokio::test]
async fn explains_scan_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "EXPLAIN SELECT id FROM '{}' WHERE id >= 2 ORDER BY id DESC LIMIT 2",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("explain sql");

    let QueryOutput::Explain { plan } = output else {
        panic!("expected explain output");
    };
    assert!(plan.contains("SortExec"));
    assert!(plan.contains("ProjectionExec projection=[id]"));
    assert!(plan.contains("FilterExec predicate=residual"));
    assert!(plan.contains("ScanExec format=Parquet"));
    assert!(plan.contains("pushdown_predicates=1"));
}

#[tokio::test]
async fn scan_plan_exposes_logical_and_declarative_physical_nodes() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let plan = DodamEngine::default()
        .plan_parquet_scan(
            path,
            2,
            Some(2),
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("id >= 2").expect("filter")),
            Some(SortKey::from(SortExpr::parse("id desc").expect("sort"))),
        )
        .await
        .expect("plan scan");

    let logical = plan.to_logical_plan();
    let planned_from_logical = logical.to_physical_plan();
    assert_eq!(planned_from_logical.operator(), &PhysicalOperator::Limit);
    assert!(
        planned_from_logical
            .render_text()
            .contains("ScanExec format=Parquet")
    );
    let LogicalPlan::TableScan(scan) = logical else {
        panic!("expected logical table scan");
    };
    assert_eq!(scan.batch_size, 2);
    assert_eq!(scan.limit, Some(2));
    assert!(scan.filter.is_some());
    assert!(scan.order_by.is_some());
    assert_eq!(scan.source.fragments.len(), 1);

    let physical = plan.to_plan_node();
    assert_eq!(physical.operator(), &PhysicalOperator::Limit);
    assert!(physical.render_text().contains("ScanExec format=Parquet"));

    let lowered_physical = plan.to_logical_plan().to_physical_plan();
    let batches = DodamEngine::default()
        .execute_physical_plan_node(lowered_physical)
        .expect("execute lowered physical plan")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect lowered scan");
    assert_eq!(ids_from_batches(&batches), vec![5, 4]);

    let graph = StagePlanner::plan_execution_graph(plan.to_logical_plan().to_physical_plan());
    let streams = DodamEngine::default()
        .execute_execution_graph_locally(graph)
        .expect("execute local scan task graph");
    let task_rows = streams
        .into_iter()
        .flat_map(|stream| {
            stream
                .collect::<Result<Vec<_>, _>>()
                .expect("collect task stream")
        })
        .collect::<Vec<_>>();
    assert_eq!(ids_from_batches(&task_rows), vec![5, 4]);

    let gather_physical = PhysicalPlanner::new(PhysicalPlanningOptions {
        insert_exchanges: true,
        ..PhysicalPlanningOptions::default()
    })
    .plan(&plan.to_logical_plan());
    let gather_graph = StagePlanner::plan_execution_graph(gather_physical);
    let gather_streams = DodamEngine::default()
        .execute_execution_graph_locally(gather_graph)
        .expect("execute gather task graph");
    let gather_rows = gather_streams
        .into_iter()
        .flat_map(|stream| {
            stream
                .collect::<Result<Vec<_>, _>>()
                .expect("collect gather stream")
        })
        .collect::<Vec<_>>();
    assert_eq!(ids_from_batches(&gather_rows), vec![5, 4]);
}

#[tokio::test]
async fn executes_grouped_aggregate_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT payload, count(*), sum(id) FROM '{}' GROUP BY payload",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 3)
        .await
        .expect("execute sql");

    let QueryOutput::Aggregate { metrics, batches } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(metrics.rows, 6);
    assert_eq!(metrics.groups.len(), 3);
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b", "c"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![3, 2, 1]);
    assert_eq!(i64s_from_column(&batches, 2), vec![6, 4, 5]);
    assert_eq!(batches[0].schema().field(1).name(), "count(*)");
    assert_eq!(batches[0].schema().field(2).name(), "sum(id)");
}

#[tokio::test]
async fn supports_table_alias_in_grouped_aggregate_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT t.payload AS p, sum(t.id) AS total FROM '{}' t GROUP BY t.payload ORDER BY total DESC",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 3)
        .await
        .expect("execute sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "p");
    assert_eq!(batches[0].schema().field(1).name(), "total");
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "c", "b"]);
    assert_eq!(i64s_from_column(&batches, 1), vec![6, 5, 4]);
}

#[tokio::test]
async fn accepts_aggregate_alias_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!("SELECT count(*) AS n FROM '{}'", path.display());
    let output = execute_sql(&DodamEngine::default(), &sql, 3)
        .await
        .expect("execute sql");

    let QueryOutput::Aggregate { metrics, batches } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(metrics.rows, 6);
    assert_eq!(u64s_from_column(&batches, 0), vec![6]);
    assert_eq!(batches[0].schema().field(0).name(), "n");
}

#[tokio::test]
async fn orders_and_limits_grouped_aggregate_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT payload, count(*) AS n FROM '{}' GROUP BY payload ORDER BY n DESC LIMIT 2",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 3)
        .await
        .expect("execute sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![3, 2]);
    assert_eq!(batches[0].schema().field(1).name(), "n");
}

#[tokio::test]
async fn orders_grouped_aggregate_by_expression_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT payload, count(*) FROM '{}' GROUP BY payload ORDER BY count(*) ASC, payload DESC",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 3)
        .await
        .expect("execute sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(strings_from_column(&batches, 0), vec!["c", "b", "a"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![1, 2, 3]);
}

#[tokio::test]
async fn filters_grouped_aggregate_with_having_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT payload AS p, count(*) AS n FROM '{}' GROUP BY payload HAVING n > 1 ORDER BY p",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 3)
        .await
        .expect("execute sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "p");
    assert_eq!(batches[0].schema().field(1).name(), "n");
    assert_eq!(strings_from_column(&batches, 0), vec!["a", "b"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![3, 2]);
}

#[tokio::test]
async fn executes_distinct_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let sql = format!(
        "SELECT DISTINCT payload FROM '{}' ORDER BY payload",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(strings_from_batches(&batches), vec!["a", "b", "c"]);
}

#[tokio::test]
async fn supports_extended_filter_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_nullable_test_parquet(&path);

    let sql = format!(
        "SELECT id AS selected_id FROM '{}' WHERE (payload IN ('a', 'c') AND NOT id = 2) OR payload IS NULL ORDER BY selected_id",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "selected_id");
    assert_eq!(ids_from_batches(&batches), vec![0, 3, 4, 5]);
}

#[tokio::test]
async fn filters_richer_parquet_types_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE amount = '123.45' OR created_at >= '2024-01-02 00:00:00' OR created_at_utc >= '2024-01-02 00:00:00' OR event_date >= '2024-01-02' OR event_date64 >= '2024-01-02' OR flag = true ORDER BY id",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute rich filter sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![1, 3, 4]);
}

#[tokio::test]
async fn aggregates_temporal_min_max_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let sql = format!(
        "SELECT min(event_date), max(event_date), min(event_date64), max(event_date64), min(created_at), max(created_at), min(created_at_utc), max(created_at_utc) FROM '{}'",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute temporal aggregate sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    let batch = &batches[0];
    let min_date = batch
        .column(0)
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap();
    let max_date = batch
        .column(1)
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap();
    let min_date64 = batch
        .column(2)
        .as_any()
        .downcast_ref::<Date64Array>()
        .unwrap();
    let max_date64 = batch
        .column(3)
        .as_any()
        .downcast_ref::<Date64Array>()
        .unwrap();
    let min_ts = batch
        .column(4)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    let max_ts = batch
        .column(5)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    let min_ts_utc = batch
        .column(6)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    let max_ts_utc = batch
        .column(7)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();

    assert_eq!(min_date.value(0), 0);
    assert_eq!(max_date.value(0), 19_724);
    assert_eq!(min_date64.value(0), 0);
    assert_eq!(max_date64.value(0), 1_704_153_600_000);
    assert_eq!(min_ts.value(0), 0);
    assert_eq!(max_ts.value(0), 1_704_153_600_000);
    assert_eq!(min_ts_utc.value(0), 0);
    assert_eq!(max_ts_utc.value(0), 1_704_153_600_000);
    assert_eq!(
        batch.schema().field(6).data_type(),
        &DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into()))
    );
}

#[tokio::test]
async fn projects_nested_and_richer_parquet_types_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let sql = format!(
        "SELECT id, tags, attrs FROM '{}' WHERE id <= 3 ORDER BY id",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute rich projection sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![1, 2, 3]);
    let schema = batches[0].schema();
    assert!(matches!(schema.field(1).data_type(), DataType::List(_)));
    assert!(matches!(schema.field(2).data_type(), DataType::Struct(_)));
}

#[tokio::test]
async fn executes_scalar_projection_expressions_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let sql = format!(
        "SELECT id + 10 AS plus_ten, id * 2 AS doubled, CAST(id AS VARCHAR) AS id_text, COALESCE(note, 'fallback') AS note_text FROM '{}' ORDER BY id LIMIT 3",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute scalar projection expression sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(numeric_i64s_from_column(&batches, 0), vec![11, 12, 13]);
    assert_eq!(numeric_i64s_from_column(&batches, 1), vec![2, 4, 6]);
    assert_eq!(strings_from_column(&batches, 2), vec!["1", "2", "3"]);
    assert_eq!(
        strings_from_column(&batches, 3),
        vec!["alpha", "fallback", "gamma"]
    );
    assert_eq!(batches[0].schema().field(0).name(), "plus_ten");
    assert_eq!(batches[0].schema().field(3).name(), "note_text");
}

#[tokio::test]
async fn filters_with_scalar_expressions_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let sql = format!(
        "SELECT id, note FROM '{}' WHERE id + 1 >= 3 AND COALESCE(note, 'fallback') <> 'fallback' ORDER BY id",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute scalar WHERE expression sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![3, 4]);
    assert_eq!(strings_from_column(&batches, 1), vec!["gamma", ""]);

    let sql = format!(
        "SELECT id FROM '{}' WHERE CAST(id AS VARCHAR) = '2' OR COALESCE(note, NULL) IS NULL ORDER BY id LIMIT 2",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute scalar WHERE null expression sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![2]);
}

#[tokio::test]
async fn executes_string_functions_and_case_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let sql = format!(
        "SELECT lower(note) AS lower_note, upper(note) AS upper_note, length(note) AS note_len, CASE WHEN id = 1 THEN 'one' WHEN note IS NULL THEN 'missing' ELSE 'other' END AS bucket FROM '{}' ORDER BY id",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute string functions and case sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(
        strings_from_column_with_nulls(&batches, 0),
        vec![
            Some("alpha".to_string()),
            None,
            Some("gamma".to_string()),
            Some("".to_string())
        ]
    );
    assert_eq!(
        strings_from_column_with_nulls(&batches, 1),
        vec![
            Some("ALPHA".to_string()),
            None,
            Some("GAMMA".to_string()),
            Some("".to_string())
        ]
    );
    assert_eq!(
        numeric_i64s_from_column_with_nulls(&batches, 2),
        vec![Some(5), None, Some(5), Some(0)]
    );
    assert_eq!(
        strings_from_column(&batches, 3),
        vec!["one", "missing", "other", "other"]
    );

    let sql = format!(
        "SELECT id FROM '{}' WHERE upper(COALESCE(note, 'fallback')) = 'GAMMA' OR length(note) = 0 ORDER BY id",
        path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute string function WHERE sql");
    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![3, 4]);
}

#[tokio::test]
async fn executes_inner_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id AS order_id, c.name AS customer_name FROM '{}' AS o INNER JOIN '{}' AS c ON o.customer_id = c.id WHERE c.name = 'alice' ORDER BY order_id",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "order_id");
    assert_eq!(batches[0].schema().field(1).name(), "customer_name");
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 12]);
    assert_eq!(
        strings_from_column(&batches, 1),
        vec!["alice".to_string(), "alice".to_string()]
    );
}

#[tokio::test]
async fn executes_join_sql_without_explicit_aliases() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT orders.id, customers.name FROM '{}' JOIN '{}' ON customer_id = id WHERE customers.name = 'alice' ORDER BY orders.id",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 12]);
    assert_eq!(
        strings_from_column(&batches, 1),
        vec!["alice".to_string(), "alice".to_string()]
    );
}

#[tokio::test]
async fn executes_two_table_comma_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id AS order_id, c.name AS customer_name FROM '{}' AS o, '{}' AS c WHERE o.customer_id = c.id AND c.name = 'alice' ORDER BY order_id",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute comma join sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 12]);
    assert_eq!(
        strings_from_column(&batches, 1),
        vec!["alice".to_string(), "alice".to_string()]
    );
}

#[tokio::test]
async fn filters_with_column_to_column_comparison_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT id FROM '{}' WHERE customer_id < id AND id < 12 ORDER BY id",
        orders_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute column comparison sql");
    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 11]);

    let sql = format!(
        "SELECT o.id FROM '{}' AS o, '{}' AS c WHERE o.customer_id = c.id AND o.customer_id < o.id ORDER BY o.id",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute join residual column comparison sql");
    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 11, 12]);
}

#[tokio::test]
async fn executes_comma_join_with_join_key_inside_or_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id FROM '{}' AS o, '{}' AS c WHERE (o.customer_id = c.id AND c.name = 'alice') OR (o.customer_id = c.id AND c.name = 'bob') ORDER BY o.id",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute comma join with OR join key sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 11, 12]);
}

#[tokio::test]
async fn executes_comma_join_aggregate_expression_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let lineitem_path = tempdir.path().join("lineitem.parquet");
    write_tpch_like_orders_parquet(&orders_path);
    write_tpch_like_lineitem_parquet(&lineitem_path);

    let sql = format!(
        "SELECT l_shipmode, sum(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, sum(l_extendedprice * (1 - l_discount)) AS revenue FROM '{}' AS orders, '{}' AS lineitem WHERE o_orderkey = l_orderkey GROUP BY l_shipmode ORDER BY l_shipmode",
        orders_path.display(),
        lineitem_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute comma join aggregate expression sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(
        strings_from_column(&batches, 0),
        vec!["MAIL".to_string(), "SHIP".to_string()]
    );
    assert_eq!(i64s_from_column(&batches, 1), vec![1, 1]);
    assert_eq!(f64s_from_column(&batches, 2), vec![1400.0, 380.0]);
}

#[tokio::test]
async fn executes_join_aggregate_output_expression_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let lineitem_path = tempdir.path().join("lineitem.parquet");
    let part_path = tempdir.path().join("part.parquet");
    write_q14_lineitem_parquet(&lineitem_path);
    write_q14_part_parquet(&part_path);

    let sql = format!(
        "SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM '{}' AS lineitem, '{}' AS part WHERE l_partkey = p_partkey",
        lineitem_path.display(),
        part_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute join aggregate output expression sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "promo_revenue");
    assert_eq!(f64s_from_column(&batches, 0), vec![70.0]);
}

#[tokio::test]
async fn executes_three_table_comma_join_aggregate_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let customer_path = tempdir.path().join("customer.parquet");
    let orders_path = tempdir.path().join("orders.parquet");
    let lineitem_path = tempdir.path().join("lineitem.parquet");
    write_q13_customer_parquet(&customer_path);
    write_q13_orders_parquet(&orders_path);
    write_q13_lineitem_parquet(&lineitem_path);

    let sql = format!(
        "SELECT c_custkey, sum(l_quantity) AS total_quantity FROM '{}' AS customer, '{}' AS orders, '{}' AS lineitem WHERE c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_custkey ORDER BY c_custkey",
        customer_path.display(),
        orders_path.display(),
        lineitem_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute three table comma join aggregate sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(i64s_from_column(&batches, 0), vec![1, 2, 3]);
    assert_eq!(i64s_from_column(&batches, 1), vec![12, 3, 11]);
}

#[tokio::test]
async fn executes_tpch_q13_style_join_aggregate_without_explicit_aliases() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let customer_path = tempdir.path().join("customer.parquet");
    let orders_path = tempdir.path().join("orders.parquet");
    write_q13_customer_parquet(&customer_path);
    write_q13_orders_parquet(&orders_path);

    let sql = format!(
        "SELECT c_count, count(*) AS custdist FROM (SELECT c_custkey, count(o_orderkey) AS c_count FROM '{}' LEFT OUTER JOIN '{}' ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' GROUP BY c_custkey) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC",
        customer_path.display(),
        orders_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute TPC-H Q13-style sql");

    let QueryOutput::Aggregate { batches, .. } = output else {
        panic!("expected aggregate output");
    };
    assert_eq!(u64s_from_column(&batches, 0), vec![1, 0]);
    assert_eq!(u64s_from_column(&batches, 1), vec![2, 1]);
}

#[tokio::test]
async fn executes_multi_column_inner_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let left_path = tempdir.path().join("left.parquet");
    let right_path = tempdir.path().join("right.parquet");
    write_composite_left_parquet(&left_path);
    write_composite_right_parquet(&right_path);

    let sql = format!(
        "SELECT l.id FROM '{}' AS l JOIN '{}' AS r ON l.k1 = r.k1 AND l.k2 = r.k2 ORDER BY l.id",
        left_path.display(),
        right_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(ids_from_batches(&batches), vec![1, 3]);
}

#[tokio::test]
async fn explains_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "EXPLAIN SELECT o.id, c.name FROM '{}' o JOIN '{}' c ON o.customer_id = c.id WHERE c.name = 'alice'",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("explain sql");

    let QueryOutput::Explain { plan } = output else {
        panic!("expected explain output");
    };
    assert!(plan.contains("JoinExec type=Inner"));
    assert!(plan.contains("strategy=hash"));
    assert!(plan.contains("left_keys=[customer_id]"));
    assert!(plan.contains("side=left"));
    assert!(plan.contains("side=right"));
}

#[tokio::test]
async fn executes_partitioned_multi_column_join_batches() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let left_path = tempdir.path().join("left.parquet");
    let right_path = tempdir.path().join("right.parquet");
    write_composite_left_parquet(&left_path);
    write_composite_right_parquet(&right_path);

    let mut stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path,
            right_path,
            batch_size: 1,
            left_keys: vec!["k1".to_string(), "k2".to_string()],
            right_keys: vec!["k1".to_string(), "k2".to_string()],
            left_prefix: "l".to_string(),
            right_prefix: "r".to_string(),
            left_projection: Projection::Columns(vec![
                "id".to_string(),
                "k1".to_string(),
                "k2".to_string(),
            ]),
            right_projection: Projection::Columns(vec!["k1".to_string(), "k2".to_string()]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await
        .expect("join batches");
    let batches = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect join");
    let metrics = stream.scan_plan_metrics();

    let mut ids = i32s_from_column(&batches, 0);
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 3]);
    assert!(metrics.join_spill_files > 0, "{metrics:?}");
}

#[tokio::test]
async fn plans_partitioned_join_and_explains_estimates() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let left_path = tempdir.path().join("left.parquet");
    let right_path = tempdir.path().join("right.parquet");
    write_composite_left_parquet(&left_path);
    write_composite_right_parquet(&right_path);

    let request = JoinParquetRequest {
        left_path,
        right_path,
        batch_size: 1,
        left_keys: vec!["k1".to_string(), "k2".to_string()],
        right_keys: vec!["k1".to_string(), "k2".to_string()],
        left_prefix: "l".to_string(),
        right_prefix: "r".to_string(),
        left_projection: Projection::All,
        right_projection: Projection::All,
        left_filter: None,
        right_filter: None,
        output_projection: Projection::All,
        join_memory_limit_bytes: 1,
        join_algorithm: JoinAlgorithm::Auto,
        join_type: JoinType::Inner,
    };

    let plan = DodamEngine::default()
        .plan_parquet_join(request)
        .await
        .expect("plan join");

    assert!(matches!(
        plan.strategy,
        JoinExecutionStrategy::PartitionedHash { .. }
    ));
    assert!(plan.left_scan.estimated_bytes > 0);
    assert!(plan.right_scan.estimated_bytes > 0);
    let explain = plan.explain();
    assert!(explain.contains("strategy=partitioned_hash"));
    assert!(explain.contains("side=left"));
    assert!(explain.contains("estimated_left_bytes="));
}

#[tokio::test]
async fn plans_hash_join_build_side_from_estimated_scan_bytes() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let request = JoinParquetRequest {
        left_path: orders_path,
        right_path: customers_path,
        batch_size: 2,
        left_keys: vec!["customer_id".to_string()],
        right_keys: vec!["id".to_string()],
        left_prefix: "o".to_string(),
        right_prefix: "c".to_string(),
        left_projection: Projection::All,
        right_projection: Projection::All,
        left_filter: None,
        right_filter: None,
        output_projection: Projection::All,
        join_memory_limit_bytes: u64::MAX,
        join_algorithm: JoinAlgorithm::Auto,
        join_type: JoinType::Inner,
    };

    let plan = DodamEngine::default()
        .plan_parquet_join(request)
        .await
        .expect("plan join");

    let JoinExecutionStrategy::Hash { build_side } = plan.strategy else {
        panic!("expected hash join strategy");
    };
    if plan.left_scan.estimated_bytes <= plan.right_scan.estimated_bytes {
        assert_eq!(format!("{build_side:?}"), "Left");
    } else {
        assert_eq!(format!("{build_side:?}"), "Right");
    }
    assert!(plan.explain().contains("strategy=hash"));
}

#[tokio::test]
async fn join_plan_exposes_logical_and_physical_strategy_descriptors() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let plan = DodamEngine::default()
        .plan_parquet_join(JoinParquetRequest {
            left_path: orders_path,
            right_path: customers_path,
            batch_size: 2,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::All,
            right_projection: Projection::All,
            left_filter: None,
            right_filter: None,
            output_projection: Projection::Columns(vec!["o.id".to_string(), "c.name".to_string()]),
            join_memory_limit_bytes: u64::MAX,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await
        .expect("plan join");

    let logical = plan.to_logical_plan();
    let planned_from_logical = PhysicalPlanner::new(PhysicalPlanningOptions {
        default_join_strategy: plan.physical_strategy(),
        ..PhysicalPlanningOptions::default()
    })
    .plan(&logical);
    assert_eq!(planned_from_logical.operator(), &PhysicalOperator::HashJoin);
    assert!(
        planned_from_logical
            .render_text()
            .contains("output_projection=[o.id,c.name]")
    );
    let batches = DodamEngine::default()
        .execute_physical_plan_node(planned_from_logical.clone())
        .expect("execute lowered join")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect lowered join");
    let mut rows = i32s_from_column(&batches, 0)
        .into_iter()
        .zip(strings_from_column(&batches, 1))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (10, "alice".to_string()),
            (11, "bob".to_string()),
            (12, "alice".to_string())
        ]
    );
    let LogicalPlan::Join {
        left,
        right,
        join_type,
        left_keys,
        right_keys,
        left_prefix,
        right_prefix,
        output_projection,
    } = logical
    else {
        panic!("expected logical join");
    };
    assert!(matches!(*left, LogicalPlan::TableScan(_)));
    assert!(matches!(*right, LogicalPlan::TableScan(_)));
    assert_eq!(join_type, JoinType::Inner);
    assert_eq!(left_keys, vec!["customer_id".to_string()]);
    assert_eq!(right_keys, vec!["id".to_string()]);
    assert_eq!(left_prefix, "o");
    assert_eq!(right_prefix, "c");
    assert_eq!(
        output_projection,
        Projection::Columns(vec!["o.id".to_string(), "c.name".to_string()])
    );

    assert!(matches!(
        plan.physical_strategy(),
        PhysicalJoinStrategy::Hash { .. }
    ));
    assert_eq!(plan.to_plan_node().operator(), &PhysicalOperator::HashJoin);
}

#[tokio::test]
async fn local_execution_graph_executes_hash_repartition_join() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let plan = DodamEngine::default()
        .plan_parquet_join(JoinParquetRequest {
            left_path: orders_path,
            right_path: customers_path,
            batch_size: 2,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::All,
            right_projection: Projection::All,
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await
        .expect("plan join");
    let physical = PhysicalPlanner::new(PhysicalPlanningOptions {
        default_join_strategy: plan.physical_strategy(),
        insert_exchanges: true,
        default_shuffle_partitions: 4,
    })
    .plan(&plan.to_logical_plan());
    let graph = StagePlanner::plan_execution_graph(physical);
    let expected_stages = graph.stages.len();
    let expected_tasks = graph.tasks.len();

    let output = DodamEngine::default()
        .execute_execution_graph_locally_with_metrics(graph)
        .expect("execute hash repartition graph");
    let metrics = output.metrics;
    let stage_metrics = output.stage_metrics.clone();
    let batches = output
        .streams
        .into_iter()
        .flat_map(|stream| {
            stream
                .collect::<Result<Vec<_>, _>>()
                .expect("collect hash repartition stream")
        })
        .collect::<Vec<_>>();
    assert_eq!(stage_metrics.len(), expected_stages);
    assert_eq!(
        stage_metrics
            .iter()
            .map(|stage| stage.tasks_executed)
            .sum::<usize>(),
        metrics.tasks_executed
    );
    assert_eq!(
        stage_metrics
            .iter()
            .map(|stage| stage.shuffle_read_rows)
            .sum::<usize>(),
        metrics.shuffle_read_rows
    );
    assert!(
        stage_metrics
            .iter()
            .any(|stage| stage.shuffle_write_files > 0),
        "{stage_metrics:?}"
    );
    assert!(
        stage_metrics
            .iter()
            .any(|stage| stage.shuffle_read_files > 0),
        "{stage_metrics:?}"
    );
    assert_eq!(metrics.stages_executed, expected_stages);
    assert_eq!(metrics.tasks_executed, expected_tasks);
    assert!(metrics.task_output_batches > 0);
    assert!(metrics.task_output_rows >= 6);
    assert!(metrics.shuffle_write_files > 0, "{metrics:?}");
    assert!(metrics.shuffle_write_batches > 0, "{metrics:?}");
    assert!(metrics.shuffle_write_rows > 0, "{metrics:?}");
    assert!(metrics.shuffle_write_bytes > 0, "{metrics:?}");
    assert!(metrics.shuffle_read_files > 0, "{metrics:?}");
    assert!(metrics.shuffle_read_batches > 0, "{metrics:?}");
    assert!(metrics.shuffle_read_rows > 0, "{metrics:?}");
    assert!(metrics.shuffle_read_bytes > 0, "{metrics:?}");
    let mut rows = i32s_from_column(&batches, 0)
        .into_iter()
        .zip(i32s_from_column(&batches, 1))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(10, 1), (11, 2), (12, 1)]);
}

#[tokio::test]
async fn plans_table_source_join() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let engine = DodamEngine::default();
    let orders = engine
        .plan_table_source(orders_path)
        .await
        .expect("plan orders");
    let customers = engine
        .plan_table_source(customers_path)
        .await
        .expect("plan customers");

    let plan = engine
        .plan_table_join(JoinTableRequest {
            left: orders,
            right: customers,
            batch_size: 2,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::All,
            right_projection: Projection::All,
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: u64::MAX,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .expect("plan table join");

    assert!(plan.explain().contains("JoinExec"));
    assert!(plan.explain().contains("side=left"));
    assert!(plan.explain().contains("side=right"));
}

#[tokio::test]
async fn executes_left_and_right_outer_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let left_sql = format!(
        "SELECT o.id, c.name FROM '{}' AS o LEFT JOIN '{}' AS c ON o.customer_id = c.id ORDER BY o.id",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Scan { batches } = execute_sql(&DodamEngine::default(), &left_sql, 2)
        .await
        .expect("execute left join")
    else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 11, 12, 13]);
    assert_eq!(
        optional_strings_from_column(&batches, 1),
        vec![
            Some("alice".to_string()),
            Some("bob".to_string()),
            Some("alice".to_string()),
            None,
        ]
    );

    let left_residual_sql = format!(
        "SELECT o.id, c.name FROM '{}' AS o LEFT JOIN '{}' AS c ON o.customer_id = c.id AND c.name NOT LIKE 'bob%' ORDER BY o.id",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Scan { batches } = execute_sql(&DodamEngine::default(), &left_residual_sql, 2)
        .await
        .expect("execute left join with right-side ON residual")
    else {
        panic!("expected scan output");
    };
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 11, 12, 13]);
    assert_eq!(
        optional_strings_from_column(&batches, 1),
        vec![
            Some("alice".to_string()),
            None,
            Some("alice".to_string()),
            None,
        ]
    );

    let right_sql = format!(
        "SELECT o.id, c.name FROM '{}' AS o RIGHT JOIN '{}' AS c ON o.customer_id = c.id ORDER BY c.id, o.id",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Scan { batches } = execute_sql(&DodamEngine::default(), &right_sql, 2)
        .await
        .expect("execute right join")
    else {
        panic!("expected scan output");
    };
    assert_eq!(
        optional_i32s_from_column(&batches, 0),
        vec![Some(10), Some(12), Some(11), None,]
    );
    assert_eq!(
        strings_from_column(&batches, 1),
        vec![
            "alice".to_string(),
            "alice".to_string(),
            "bob".to_string(),
            "carol".to_string(),
        ]
    );
}

#[tokio::test]
async fn executes_full_outer_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id, c.name FROM '{}' AS o FULL OUTER JOIN '{}' AS c ON o.customer_id = c.id",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Scan { batches } = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute full join")
    else {
        panic!("expected scan output");
    };

    let mut rows = optional_i32s_from_column(&batches, 0)
        .into_iter()
        .zip(optional_strings_from_column(&batches, 1))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| format!("{left:?}").cmp(&format!("{right:?}")));

    assert_eq!(
        rows,
        vec![
            (None, Some("carol".to_string())),
            (Some(10), Some("alice".to_string())),
            (Some(11), Some("bob".to_string())),
            (Some(12), Some("alice".to_string())),
            (Some(13), None),
        ]
    );
}

#[tokio::test]
async fn executes_left_semi_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("duplicate-customers.parquet");
    write_orders_parquet(&orders_path);
    write_duplicate_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id FROM '{}' AS o LEFT SEMI JOIN '{}' AS c ON o.customer_id = c.id ORDER BY o.id",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Scan { batches } = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute semi join")
    else {
        panic!("expected scan output");
    };

    assert_eq!(batches[0].schema().field(0).name(), "o.id");
    assert_eq!(i32s_from_column(&batches, 0), vec![10, 11, 12]);
}

#[tokio::test]
async fn executes_global_aggregate_over_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT count(*), sum(o.id) FROM '{}' AS o JOIN '{}' AS c ON o.customer_id = c.id",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Aggregate { metrics, batches } = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute join aggregate")
    else {
        panic!("expected aggregate output");
    };

    assert_eq!(metrics.rows, 3);
    assert_eq!(u64s_from_column(&batches, 0), vec![3]);
    assert_eq!(i64s_from_column(&batches, 1), vec![33]);
}

#[tokio::test]
async fn filters_streaming_join_aggregate_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT count(*), sum(o.id) FROM '{}' AS o JOIN '{}' AS c ON o.customer_id = c.id WHERE c.name = 'alice'",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Aggregate { metrics, batches } = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute filtered join aggregate")
    else {
        panic!("expected aggregate output");
    };

    assert_eq!(metrics.rows, 2);
    assert_eq!(u64s_from_column(&batches, 0), vec![2]);
    assert_eq!(i64s_from_column(&batches, 1), vec![22]);
}

#[tokio::test]
async fn executes_grouped_aggregate_over_join_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT c.name, count(*), sum(o.id) FROM '{}' AS o JOIN '{}' AS c ON o.customer_id = c.id GROUP BY c.name ORDER BY c.name",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Aggregate { metrics, batches } = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute grouped join aggregate")
    else {
        panic!("expected aggregate output");
    };

    assert_eq!(metrics.rows, 3);
    assert_eq!(metrics.groups.len(), 2);
    assert_eq!(strings_from_column(&batches, 0), vec!["alice", "bob"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![2, 1]);
    assert_eq!(i64s_from_column(&batches, 2), vec![22, 11]);
    assert_eq!(batches[0].schema().field(0).name(), "c.name");
    assert_eq!(batches[0].schema().field(2).name(), "sum(o.id)");
}

#[tokio::test]
async fn filters_grouped_join_aggregate_with_having_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT c.name AS customer, count(*) AS n FROM '{}' AS o JOIN '{}' AS c ON o.customer_id = c.id GROUP BY c.name HAVING n > 1 ORDER BY customer",
        orders_path.display(),
        customers_path.display()
    );
    let QueryOutput::Aggregate { batches, .. } = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute grouped join aggregate with having")
    else {
        panic!("expected aggregate output");
    };

    assert_eq!(batches[0].schema().field(0).name(), "customer");
    assert_eq!(batches[0].schema().field(1).name(), "n");
    assert_eq!(strings_from_column(&batches, 0), vec!["alice"]);
    assert_eq!(u64s_from_column(&batches, 1), vec![2]);
}

#[tokio::test]
async fn executes_partitioned_full_outer_join_batches() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let left_path = tempdir.path().join("left.parquet");
    let right_path = tempdir.path().join("right.parquet");
    write_partitioned_full_left_parquet(&left_path);
    write_partitioned_full_right_parquet(&right_path);

    let engine = DodamEngine::default();
    let base_request = JoinParquetRequest {
        left_path: left_path.clone(),
        right_path: right_path.clone(),
        batch_size: 128,
        left_keys: vec!["k".to_string()],
        right_keys: vec!["id".to_string()],
        left_prefix: "l".to_string(),
        right_prefix: "r".to_string(),
        left_projection: Projection::All,
        right_projection: Projection::All,
        left_filter: None,
        right_filter: None,
        output_projection: Projection::All,
        join_memory_limit_bytes: u64::MAX,
        join_algorithm: JoinAlgorithm::Auto,
        join_type: JoinType::Full,
    };
    let base_plan = engine
        .plan_parquet_join(base_request.clone())
        .await
        .expect("plan full join");
    let memory_limit = base_plan
        .left_scan
        .estimated_bytes
        .min(base_plan.right_scan.estimated_bytes)
        .saturating_sub(1)
        .max(1);

    let request = JoinParquetRequest {
        join_memory_limit_bytes: memory_limit,
        ..base_request
    };
    let plan = engine
        .plan_parquet_join(request.clone())
        .await
        .expect("plan partitioned full join");
    assert!(matches!(
        plan.strategy,
        JoinExecutionStrategy::PartitionedHash { .. }
    ));

    let mut stream = engine
        .join_parquet_batches(request)
        .await
        .expect("execute partitioned full join");
    let batches = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect join");
    assert!(stream.scan_plan_metrics().join_spill_files > 0);

    let rows = optional_i32s_from_column(&batches, 0)
        .into_iter()
        .zip(optional_strings_from_column(&batches, 3))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1500);
    assert!(rows.contains(&(Some(0), None)));
    assert!(rows.contains(&(Some(500), Some("n500".to_string()))));
    assert!(rows.contains(&(Some(999), Some("n999".to_string()))));
    assert!(rows.contains(&(None, Some("n1000".to_string()))));
    assert!(rows.contains(&(None, Some("n1499".to_string()))));
}

#[tokio::test]
async fn executes_partitioned_semi_join_batches() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("duplicate-customers.parquet");
    write_orders_parquet(&orders_path);
    write_duplicate_customers_parquet(&customers_path);

    let mut stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path: orders_path,
            right_path: customers_path,
            batch_size: 1,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::All,
            right_projection: Projection::Columns(vec!["id".to_string()]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Semi,
        })
        .await
        .expect("join batches");
    let batches = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect join");
    let metrics = stream.scan_plan_metrics();

    assert!(metrics.join_spill_files > 0, "{metrics:?}");
    assert_eq!(batches[0].schema().field(0).name(), "o.id");
    assert_eq!(batches[0].schema().field(1).name(), "o.customer_id");
    let mut ids = i32s_from_column(&batches, 0);
    ids.sort_unstable();
    assert_eq!(ids, vec![10, 11, 12]);
    assert_eq!(metrics.join_output_rows, 3);
}

#[tokio::test]
async fn executes_join_wildcard_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT * FROM '{}' o JOIN '{}' c ON o.customer_id = c.id ORDER BY o.id LIMIT 1",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "o.id");
    assert_eq!(batches[0].schema().field(1).name(), "o.customer_id");
    assert_eq!(batches[0].schema().field(2).name(), "c.id");
    assert_eq!(batches[0].schema().field(3).name(), "c.name");
    assert_eq!(i32s_from_column(&batches, 0), vec![10]);
    assert_eq!(strings_from_column(&batches, 3), vec!["alice".to_string()]);
}

#[tokio::test]
async fn streams_join_probe_batches_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id AS order_id, c.name AS customer_name FROM '{}' o JOIN '{}' c ON o.customer_id = c.id",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 1)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert!(
        batches.len() > 1,
        "expected one output batch per matching probe batch"
    );
    let mut ids = i32s_from_column(&batches, 0);
    ids.sort_unstable();
    let mut names = strings_from_column(&batches, 1);
    names.sort();
    assert_eq!(ids, vec![10, 11, 12]);
    assert_eq!(
        names,
        vec!["alice".to_string(), "alice".to_string(), "bob".to_string()]
    );
}

#[tokio::test]
async fn joins_when_left_side_is_selected_as_build_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT c.name AS customer_name, o.id AS order_id FROM '{}' c JOIN '{}' o ON c.id = o.customer_id ORDER BY order_id",
        customers_path.display(),
        orders_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 1)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "customer_name");
    assert_eq!(batches[0].schema().field(1).name(), "order_id");
    assert_eq!(
        strings_from_column(&batches, 0),
        vec!["alice".to_string(), "bob".to_string(), "alice".to_string()]
    );
    assert_eq!(i32s_from_column(&batches, 1), vec![10, 11, 12]);
}

#[tokio::test]
async fn executes_partitioned_spill_join_batches() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path: orders_path,
            right_path: customers_path,
            batch_size: 1,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::Columns(vec!["id".to_string(), "customer_id".to_string()]),
            right_projection: Projection::Columns(vec!["id".to_string(), "name".to_string()]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await
        .expect("join batches");
    let batches = stream.collect::<Result<Vec<_>, _>>().expect("collect join");

    assert!(!batches.is_empty());
    assert_eq!(batches[0].schema().field(0).name(), "o.id");
    assert_eq!(batches[0].schema().field(3).name(), "c.name");
    let mut ids = i32s_from_column(&batches, 0);
    ids.sort_unstable();
    let mut names = strings_from_column(&batches, 3);
    names.sort();
    assert_eq!(ids, vec![10, 11, 12]);
    assert_eq!(
        names,
        vec!["alice".to_string(), "alice".to_string(), "bob".to_string()]
    );
}

#[tokio::test]
async fn exposes_partitioned_join_metrics() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let mut stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path: orders_path,
            right_path: customers_path,
            batch_size: 1,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::Columns(vec!["id".to_string(), "customer_id".to_string()]),
            right_projection: Projection::Columns(vec!["id".to_string(), "name".to_string()]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await
        .expect("join batches");
    let batches = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect join");
    let metrics = stream.scan_plan_metrics();

    assert!(!batches.is_empty());
    assert!(metrics.join_spill_files > 0);
    assert!(metrics.join_spill_bytes > 0);
    assert!(metrics.join_build_rows > 0);
    assert!(metrics.join_peak_build_bytes > 0);
    assert!(metrics.join_probe_rows > 0);
    assert_eq!(metrics.join_output_rows, 3);
}

#[tokio::test]
async fn separates_heavy_hitter_join_keys() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let customers_path = tempdir.path().join("heavy-customers.parquet");
    let orders_path = tempdir.path().join("wide-orders.parquet");
    write_heavy_customers_parquet(&customers_path);
    write_heavy_orders_parquet(&orders_path);

    let mut stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path: customers_path,
            right_path: orders_path,
            batch_size: 256,
            left_keys: vec!["id".to_string()],
            right_keys: vec!["customer_id".to_string()],
            left_prefix: "c".to_string(),
            right_prefix: "o".to_string(),
            left_projection: Projection::Columns(vec!["id".to_string(), "name".to_string()]),
            right_projection: Projection::Columns(vec![
                "id".to_string(),
                "customer_id".to_string(),
            ]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await
        .expect("join batches");
    let batches = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect join");
    let metrics = stream.scan_plan_metrics();

    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1_210_020
    );
    assert!(
        batches.len() > 1,
        "heavy hitter output should be emitted in bounded chunks"
    );
    assert!(metrics.join_spill_files > 0, "{metrics:?}");
    assert!(metrics.join_heavy_hitters > 0, "{metrics:?}");
    assert_eq!(metrics.join_output_rows, 1_210_020);
}

#[tokio::test]
async fn executes_skewed_partitioned_full_outer_join_without_nested_loop_fallback() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let customers_path = tempdir.path().join("heavy-customers.parquet");
    let orders_path = tempdir.path().join("wide-orders.parquet");
    write_heavy_customers_parquet(&customers_path);
    write_heavy_orders_parquet(&orders_path);

    let mut stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path: customers_path,
            right_path: orders_path,
            batch_size: 256,
            left_keys: vec!["id".to_string()],
            right_keys: vec!["customer_id".to_string()],
            left_prefix: "c".to_string(),
            right_prefix: "o".to_string(),
            left_projection: Projection::Columns(vec!["id".to_string(), "name".to_string()]),
            right_projection: Projection::Columns(vec![
                "id".to_string(),
                "customer_id".to_string(),
            ]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: 1,
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Full,
        })
        .await
        .expect("join batches");
    let batches = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect join");
    let metrics = stream.scan_plan_metrics();

    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1_210_020
    );
    assert!(
        batches.len() > 1,
        "skewed full outer join output should be emitted in bounded chunks"
    );
    assert!(metrics.join_spill_files > 0, "{metrics:?}");
    assert!(metrics.join_heavy_hitters > 0, "{metrics:?}");
    assert_eq!(metrics.join_nested_loop_fallbacks, 0, "{metrics:?}");
    assert_eq!(metrics.join_output_rows, 1_210_020);
}

#[tokio::test]
async fn executes_sort_merge_join_batches() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let stream = DodamEngine::default()
        .join_parquet_batches(JoinParquetRequest {
            left_path: orders_path,
            right_path: customers_path,
            batch_size: 2,
            left_keys: vec!["customer_id".to_string()],
            right_keys: vec!["id".to_string()],
            left_prefix: "o".to_string(),
            right_prefix: "c".to_string(),
            left_projection: Projection::Columns(vec!["id".to_string(), "customer_id".to_string()]),
            right_projection: Projection::Columns(vec!["id".to_string(), "name".to_string()]),
            left_filter: None,
            right_filter: None,
            output_projection: Projection::All,
            join_memory_limit_bytes: u64::MAX,
            join_algorithm: JoinAlgorithm::SortMerge,
            join_type: JoinType::Inner,
        })
        .await
        .expect("join batches");
    let batches = stream.collect::<Result<Vec<_>, _>>().expect("collect join");

    assert!(!batches.is_empty());
    let mut ids = i32s_from_column(&batches, 0);
    ids.sort_unstable();
    let mut names = strings_from_column(&batches, 3);
    names.sort();
    assert_eq!(ids, vec![10, 11, 12]);
    assert_eq!(
        names,
        vec!["alice".to_string(), "alice".to_string(), "bob".to_string()]
    );
}

#[tokio::test]
async fn keeps_join_filter_and_order_columns_available_for_pushdown_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let orders_path = tempdir.path().join("orders.parquet");
    let customers_path = tempdir.path().join("customers.parquet");
    write_orders_parquet(&orders_path);
    write_customers_parquet(&customers_path);

    let sql = format!(
        "SELECT o.id AS order_id FROM '{}' o JOIN '{}' c ON o.customer_id = c.id WHERE c.name = 'alice' ORDER BY o.customer_id, order_id DESC",
        orders_path.display(),
        customers_path.display()
    );
    let output = execute_sql(&DodamEngine::default(), &sql, 2)
        .await
        .expect("execute sql");

    let QueryOutput::Scan { batches } = output else {
        panic!("expected scan output");
    };
    assert_eq!(batches[0].schema().field(0).name(), "order_id");
    assert_eq!(i32s_from_column(&batches, 0), vec![12, 10]);
}

#[tokio::test]
async fn rejects_unsupported_sql() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path);

    let error = execute_sql(
        &DodamEngine::default(),
        &format!(
            "SELECT id FROM '{}' WHERE regexp_replace(payload, 'a', 'b') = 'b'",
            path.display()
        ),
        2,
    )
    .await
    .expect_err("unsupported WHERE function should be rejected");

    assert!(
        error
            .to_string()
            .to_ascii_lowercase()
            .contains("unsupported"),
        "{error}"
    );
}

fn write_nullable_test_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, true),
    ]));
    let ids = Int32Array::from_iter_values(0..6);
    let payloads = StringArray::from(vec![Some("a"), Some("b"), Some("a"), None, Some("c"), None]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(payloads)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_rich_type_parquet(path: &std::path::Path) {
    let list_field = Arc::new(Field::new("item", DataType::Int32, true));
    let attrs_fields = vec![
        Field::new("rank", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
    ]
    .into();
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
        Field::new("tags", DataType::List(list_field), true),
        Field::new("attrs", DataType::Struct(attrs_fields), true),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3, 4]);
    let flags = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
    let scores = Float64Array::from(vec![Some(1.5), None, Some(-2.0), Some(0.0)]);
    let notes = StringArray::from(vec![Some("alpha"), None, Some("gamma"), Some("")]);
    let amounts = Decimal128Array::from(vec![Some(12345), None, Some(-700), Some(0)])
        .with_precision_and_scale(10, 2)
        .expect("decimal precision");
    let created_at = TimestampMillisecondArray::from(vec![
        Some(1_704_067_200_000),
        None,
        Some(1_704_153_600_000),
        Some(0),
    ]);
    let created_at_utc = TimestampMillisecondArray::from(vec![
        Some(1_704_067_200_000),
        None,
        Some(1_704_153_600_000),
        Some(0),
    ])
    .with_timezone("+00:00");
    let event_date = Date32Array::from(vec![Some(19_723), None, Some(19_724), Some(0)]);
    let event_date64 = Date64Array::from(vec![
        Some(1_704_067_200_000),
        None,
        Some(1_704_153_600_000),
        Some(0),
    ]);
    let tags = ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(1), Some(2)]),
        None,
        Some(vec![None, Some(4)]),
        Some(vec![]),
    ]);
    let attrs = StructArray::from(vec![
        (
            Arc::new(Field::new("rank", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40)])) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new("label", DataType::Utf8, true)),
            Arc::new(StringArray::from(vec![
                Some("hot"),
                Some("cold"),
                None,
                Some("flat"),
            ])) as Arc<dyn Array>,
        ),
    ]);
    let batch = RecordBatch::try_new(
        schema.clone(),
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
            Arc::new(tags),
            Arc::new(attrs),
        ],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_orders_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("customer_id", DataType::Int32, false),
    ]));
    let ids = Int32Array::from_iter_values([10, 11, 12, 13]);
    let customer_ids = Int32Array::from_iter_values([1, 2, 1, 4]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(customer_ids)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_customers_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3]);
    let names = StringArray::from_iter_values(["alice", "bob", "carol"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(names)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_q13_customer_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "c_custkey",
        DataType::Int32,
        false,
    )]));
    let keys = Int32Array::from_iter_values([1, 2, 3]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(keys)]).expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_q13_orders_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_comment", DataType::Utf8, false),
    ]));
    let orderkeys = Int32Array::from_iter_values([10, 11, 12, 13]);
    let custkeys = Int32Array::from_iter_values([1, 1, 2, 3]);
    let comments = StringArray::from_iter_values([
        "ordinary order",
        "special pending requests",
        "regular request",
        "special requests",
    ]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(orderkeys), Arc::new(custkeys), Arc::new(comments)],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_q13_lineitem_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int32, false),
        Field::new("l_quantity", DataType::Int64, false),
    ]));
    let orderkeys = Int32Array::from_iter_values([10, 11, 12, 13]);
    let quantities = Int64Array::from_iter_values([5, 7, 3, 11]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(orderkeys), Arc::new(quantities)],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_tpch_like_orders_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_orderpriority", DataType::Utf8, false),
    ]));
    let orderkeys = Int32Array::from_iter_values([1, 2, 3]);
    let priorities = StringArray::from_iter_values(["1-URGENT", "3-MEDIUM", "2-HIGH"]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(orderkeys), Arc::new(priorities)],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_tpch_like_lineitem_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int32, false),
        Field::new("l_shipmode", DataType::Utf8, false),
        Field::new("l_extendedprice", DataType::Int64, false),
        Field::new("l_discount", DataType::Float64, false),
    ]));
    let orderkeys = Int32Array::from_iter_values([1, 2, 3]);
    let shipmodes = StringArray::from_iter_values(["MAIL", "MAIL", "SHIP"]);
    let prices = Int64Array::from_iter_values([1000, 500, 400]);
    let discounts = Float64Array::from_iter_values([0.05, 0.10, 0.05]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(orderkeys),
            Arc::new(shipmodes),
            Arc::new(prices),
            Arc::new(discounts),
        ],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_q14_lineitem_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int32, false),
        Field::new("l_extendedprice", DataType::Int64, false),
        Field::new("l_discount", DataType::Float64, false),
    ]));
    let partkeys = Int32Array::from_iter_values([1, 2, 1]);
    let prices = Int64Array::from_iter_values([1000, 750, 500]);
    let discounts = Float64Array::from_iter_values([0.10, 0.20, 0.00]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(partkeys), Arc::new(prices), Arc::new(discounts)],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_q14_part_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_type", DataType::Utf8, false),
    ]));
    let partkeys = Int32Array::from_iter_values([1, 2]);
    let types = StringArray::from_iter_values(["PROMO ANODIZED STEEL", "STANDARD BRASS"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(partkeys), Arc::new(types)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_duplicate_customers_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values([1, 1, 2, 3]);
    let names = StringArray::from_iter_values(["alice", "alice-2", "bob", "carol"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(names)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_partitioned_full_left_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k", DataType::Int32, false),
    ]));
    let ids = Int32Array::from_iter_values(0..1000);
    let keys = Int32Array::from_iter_values(0..1000);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(keys)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(128))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_partitioned_full_right_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values(500..1500);
    let names = StringArray::from_iter_values((500..1500).map(|id| format!("n{id}")));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(names)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(128))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_composite_left_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values([1, 2, 3]);
    let k1 = Int32Array::from_iter_values([10, 10, 20]);
    let k2 = StringArray::from_iter_values(["a", "b", "a"]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(k1), Arc::new(k2)],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_composite_right_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, false),
        Field::new("k2", DataType::Utf8, false),
    ]));
    let k1 = Int32Array::from_iter_values([10, 20]);
    let k2 = StringArray::from_iter_values(["a", "a"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(k1), Arc::new(k2)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_heavy_customers_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values(std::iter::repeat_n(1, 1100).chain(2..22));
    let names = StringArray::from_iter_values(std::iter::repeat_n("alice", 1120));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(names)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(256))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_heavy_orders_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("customer_id", DataType::Int32, false),
    ]));
    let ids = Int32Array::from_iter_values(0..1120);
    let customer_ids = Int32Array::from_iter_values(std::iter::repeat_n(1, 1100).chain(2..22));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(customer_ids)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(256))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_test_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values(0..6);
    let payloads = StringArray::from_iter_values(["a", "b", "a", "b", "a", "c"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(payloads)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn ids_from_batches(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .iter()
                .map(|value| value.expect("non-null id"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn strings_from_batches(batches: &[RecordBatch]) -> Vec<String> {
    strings_from_column(batches, 0)
}

fn i32s_from_column(batches: &[RecordBatch], column: usize) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("i32 column")
                .iter()
                .map(|value| value.expect("non-null i32"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn optional_i32s_from_column(batches: &[RecordBatch], column: usize) -> Vec<Option<i32>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("i32 column")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn strings_from_column(batches: &[RecordBatch], column: usize) -> Vec<String> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column")
                .iter()
                .map(|value| value.expect("non-null string").to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn strings_from_column_with_nulls(batches: &[RecordBatch], column: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column")
                .iter()
                .map(|value| value.map(str::to_string))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn optional_strings_from_column(batches: &[RecordBatch], column: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column")
                .iter()
                .map(|value| value.map(ToString::to_string))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn u64s_from_column(batches: &[RecordBatch], column: usize) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("u64 column")
                .iter()
                .map(|value| value.expect("non-null u64"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn i64s_from_column(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("i64 column")
                .iter()
                .map(|value| value.expect("non-null i64"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn f64s_from_column(batches: &[RecordBatch], column: usize) -> Vec<f64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("f64 column")
                .iter()
                .map(|value| value.expect("non-null f64"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn numeric_i64s_from_column(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| {
            if let Some(values) = batch.column(column).as_any().downcast_ref::<Int32Array>() {
                values
                    .iter()
                    .map(|value| i64::from(value.expect("non-null i32")))
                    .collect::<Vec<_>>()
            } else {
                batch
                    .column(column)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("numeric i64-compatible column")
                    .iter()
                    .map(|value| value.expect("non-null i64"))
                    .collect::<Vec<_>>()
            }
        })
        .collect()
}

fn numeric_i64s_from_column_with_nulls(batches: &[RecordBatch], column: usize) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|batch| {
            if let Some(values) = batch.column(column).as_any().downcast_ref::<Int32Array>() {
                values
                    .iter()
                    .map(|value| value.map(i64::from))
                    .collect::<Vec<_>>()
            } else {
                batch
                    .column(column)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("numeric i64-compatible column")
                    .iter()
                    .collect::<Vec<_>>()
            }
        })
        .collect()
}
