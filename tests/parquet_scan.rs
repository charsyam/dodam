use std::fs::File;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, ListArray, StringArray, StructArray, TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use dodam::catalog::{
    FileFragment, StorageFormat, StorageLocation, TableScanSource, TableStatistics,
};
use dodam::engine::DodamEngine;
use dodam::execution::{AggregateExpr, AggregateValue, FilterExpr, Projection, SortExpr};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

#[tokio::test]
async fn rejects_non_parquet_table_scan_source_for_parquet_scan() {
    let source = TableScanSource {
        fragments: vec![FileFragment::new(
            StorageLocation::LocalPath("data.csv".into()),
            StorageFormat::Csv,
        )],
        schema: None,
        format: StorageFormat::Csv,
        statistics: TableStatistics::default(),
    };

    let error = match DodamEngine::default().scan_table_source_batches(
        source,
        4,
        None,
        Projection::All,
        None,
        None,
    ) {
        Ok(_) => panic!("csv source should be rejected"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("unsupported storage format"));
}

#[tokio::test]
async fn plans_table_scan_source_with_parquet_schema_and_statistics() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    write_test_parquet_with_row_group_size(&tempdir.path().join("part-001.parquet"), 6, 2);
    write_test_parquet_with_row_group_size(&tempdir.path().join("part-002.parquet"), 4, 2);

    let source = DodamEngine::default()
        .plan_table_source(tempdir.path().to_path_buf())
        .await
        .expect("plan table source");

    let schema = source.schema.expect("schema");
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "payload");
    assert_eq!(source.fragments.len(), 2);
    assert_eq!(source.statistics.fragments, 2);
    assert_eq!(source.statistics.rows, 10);
    assert_eq!(source.statistics.row_groups, 5);
    assert!(source.statistics.compressed_bytes > 0);
    assert!(source.fragments.iter().all(|fragment| {
        fragment
            .statistics
            .is_some_and(|statistics| statistics.compressed_bytes > 0)
    }));
}

#[tokio::test]
async fn scans_and_aggregates_table_scan_source() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    write_test_parquet(&tempdir.path().join("part-000.parquet"), 10);

    let engine = DodamEngine::default();
    let source = engine
        .plan_table_source(tempdir.path().to_path_buf())
        .await
        .expect("plan table source");

    let metrics = engine
        .scan_table(
            source.clone(),
            4,
            Some(3),
            Projection::Columns(vec!["id".to_string()]),
            None,
            None,
        )
        .await
        .expect("scan table");
    assert_eq!(metrics.rows, 3);
    assert_eq!(metrics.columns, 1);

    let metrics = engine
        .aggregate_table(
            source,
            4,
            vec![AggregateExpr::parse("count(*)").expect("aggregate")],
            Vec::new(),
            None,
        )
        .expect("aggregate table");
    assert_eq!(metrics.values[0].value, AggregateValue::Count(10));
}

#[tokio::test]
async fn rejects_table_scan_source_with_mismatched_parquet_schemas() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    write_test_parquet(&tempdir.path().join("part-001.parquet"), 3);
    write_id_only_test_parquet(&tempdir.path().join("part-002.parquet"), 3);

    let error = DodamEngine::default()
        .plan_table_source(tempdir.path().to_path_buf())
        .await
        .expect_err("mismatched schemas should fail");

    assert!(
        error
            .to_string()
            .contains("table fragments must have identical schemas")
    );
}

#[tokio::test]
async fn scans_parquet_batches_with_limit() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(path, 4, Some(7), Projection::All, None)
        .await
        .expect("scan parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 2);
    assert_eq!(metrics.rows, 7);
    assert_eq!(metrics.columns, 2);
    assert_eq!(metrics.row_groups_total, 1);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 0);
    assert!(metrics.compressed_bytes_total > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total
    );
    assert_eq!(metrics.compressed_bytes_pruned, 0);
    assert!(metrics.metadata_nanos > 0);
    assert!(metrics.planning_nanos > 0);
    assert!(metrics.decode_nanos > 0);
}

#[tokio::test]
async fn limit_caps_rows_across_multiple_fragments() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    write_test_parquet(&tempdir.path().join("part-002.parquet"), 5);
    write_test_parquet(&tempdir.path().join("part-001.parquet"), 5);

    let metrics = DodamEngine::default()
        .scan_parquet(
            tempdir.path().to_path_buf(),
            16,
            Some(7),
            Projection::All,
            None,
        )
        .await
        .expect("scan parquet directory");

    assert_eq!(metrics.fragments, 2);
    assert_eq!(metrics.rows, 7);
    assert_eq!(metrics.columns, 2);
}

#[tokio::test]
async fn scans_parquet_directory_in_stable_order() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    write_test_parquet(&tempdir.path().join("part-002.parquet"), 3);
    write_test_parquet(&tempdir.path().join("part-001.parquet"), 4);

    let metrics = DodamEngine::default()
        .scan_parquet(
            tempdir.path().to_path_buf(),
            16,
            None,
            Projection::All,
            None,
        )
        .await
        .expect("scan parquet directory");

    assert_eq!(metrics.fragments, 2);
    assert_eq!(metrics.batches, 2);
    assert_eq!(metrics.rows, 7);
    assert_eq!(metrics.columns, 2);
    assert_eq!(metrics.row_groups_total, 2);
    assert_eq!(metrics.row_groups_scanned, 2);
    assert_eq!(metrics.row_groups_pruned, 0);
    assert!(metrics.compressed_bytes_total > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total
    );
    assert_eq!(metrics.compressed_bytes_pruned, 0);
}

#[tokio::test]
async fn pushes_projection_into_parquet_scan() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::Columns(vec!["payload".to_string()]),
            None,
        )
        .await
        .expect("scan projected parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 3);
    assert_eq!(metrics.rows, 10);
    assert_eq!(metrics.columns, 1);
    assert_eq!(metrics.row_groups_total, 1);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 0);
    assert!(metrics.compressed_bytes_total > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total
    );
    assert_eq!(metrics.compressed_bytes_pruned, 0);
}

#[tokio::test]
async fn rejects_unknown_projection_column() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let error = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::Columns(vec!["missing".to_string()]),
            None,
        )
        .await
        .expect_err("unknown projection column should fail");

    assert!(error.to_string().contains("missing"));
}

#[tokio::test]
async fn filters_batches_with_vectorized_equality() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::All,
            Some(FilterExpr::parse("payload=row-7").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.rows, 1);
    assert_eq!(metrics.columns, 2);
    assert_eq!(metrics.row_groups_total, 1);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 0);
    assert!(metrics.compressed_bytes_total > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total
    );
    assert_eq!(metrics.compressed_bytes_pruned, 0);
}

#[tokio::test]
async fn parses_filters_with_spaced_operators_and_case_insensitive_and() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::All,
            Some(FilterExpr::parse("id >= 3 aNd id < 6").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.batches, 2);
    assert_eq!(metrics.rows, 3);
    assert_eq!(metrics.columns, 2);
}

#[tokio::test]
async fn parses_filters_with_quoted_string_literals() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::All,
            Some(FilterExpr::parse("payload = 'row-7'").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.rows, 1);
    assert_eq!(metrics.columns, 2);
}

#[tokio::test]
async fn applies_limit_after_filtering() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            Some(1),
            Projection::All,
            Some(FilterExpr::parse("id=3").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.rows, 1);
    assert_eq!(metrics.columns, 2);
    assert_eq!(metrics.row_groups_total, 1);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 0);
    assert!(metrics.compressed_bytes_total > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total
    );
    assert_eq!(metrics.compressed_bytes_pruned, 0);
}

#[tokio::test]
async fn reads_filter_column_even_when_it_is_not_projected() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::Columns(vec!["payload".to_string()]),
            Some(FilterExpr::parse("id=3").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.rows, 1);
    assert_eq!(metrics.columns, 1);
    assert_eq!(metrics.row_groups_total, 1);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 0);
    assert!(metrics.compressed_bytes_total > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned,
        metrics.compressed_bytes_total
    );
    assert_eq!(metrics.compressed_bytes_pruned, 0);
}

#[tokio::test]
async fn scan_api_returns_vectorized_record_batches() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let stream = DodamEngine::default()
        .scan_parquet_batches(
            path,
            4,
            Some(3),
            Projection::Columns(vec!["payload".to_string()]),
            Some(FilterExpr::parse("id=3").expect("filter")),
        )
        .await
        .expect("scan batch stream");
    let batches = stream
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("batches");

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "payload");
}

#[tokio::test]
async fn scans_nullable_and_richer_parquet_types() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let source = DodamEngine::default()
        .plan_table_source(path.clone())
        .await
        .expect("plan rich type parquet");
    let schema = source.schema.as_ref().expect("schema");
    assert_eq!(
        schema.field_with_name("amount").unwrap().data_type(),
        &DataType::Decimal128(10, 2)
    );
    assert_eq!(
        schema.field_with_name("created_at").unwrap().data_type(),
        &DataType::Timestamp(TimeUnit::Millisecond, None)
    );
    assert_eq!(
        schema
            .field_with_name("created_at_utc")
            .unwrap()
            .data_type(),
        &DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into()))
    );
    assert_eq!(
        schema.field_with_name("event_date").unwrap().data_type(),
        &DataType::Date32
    );
    assert_eq!(
        schema.field_with_name("event_date64").unwrap().data_type(),
        &DataType::Date64
    );
    assert!(matches!(
        schema.field_with_name("tags").unwrap().data_type(),
        DataType::List(_)
    ));
    assert!(matches!(
        schema.field_with_name("attrs").unwrap().data_type(),
        DataType::Struct(_)
    ));

    let stream = DodamEngine::default()
        .scan_table_source_batches(
            source,
            2,
            None,
            Projection::Columns(vec![
                "id".to_string(),
                "flag".to_string(),
                "score".to_string(),
                "note".to_string(),
                "amount".to_string(),
                "created_at".to_string(),
                "created_at_utc".to_string(),
                "event_date".to_string(),
                "event_date64".to_string(),
                "tags".to_string(),
                "attrs".to_string(),
            ]),
            None,
            Some(SortExpr::parse("id").expect("sort").into()),
        )
        .expect("scan rich type parquet");
    let batches = stream
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("batches");
    let batch = arrow_select::concat::concat_batches(&batches[0].schema(), batches.iter())
        .expect("concat batches");

    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 11);
    let flags = batch
        .column(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("flag column");
    assert_eq!(
        flags.iter().collect::<Vec<_>>(),
        vec![Some(true), None, Some(false), Some(true)]
    );
    let scores = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("score column");
    assert_eq!(
        scores.iter().collect::<Vec<_>>(),
        vec![Some(1.5), None, Some(-2.0), Some(0.0)]
    );
    let notes = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("note column");
    assert_eq!(
        notes.iter().collect::<Vec<_>>(),
        vec![Some("alpha"), None, Some("gamma"), Some("")]
    );
    let amounts = batch
        .column(4)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("amount column");
    assert_eq!(
        amounts.iter().collect::<Vec<_>>(),
        vec![Some(12345), None, Some(-700), Some(0)]
    );
    assert_eq!(amounts.precision(), 10);
    assert_eq!(amounts.scale(), 2);
    let timestamps = batch
        .column(5)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .expect("created_at column");
    assert_eq!(
        timestamps.iter().collect::<Vec<_>>(),
        vec![
            Some(1_704_067_200_000),
            None,
            Some(1_704_153_600_000),
            Some(0)
        ]
    );
    let timestamps_utc = batch
        .column(6)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .expect("created_at_utc column");
    assert_eq!(
        timestamps_utc.iter().collect::<Vec<_>>(),
        vec![
            Some(1_704_067_200_000),
            None,
            Some(1_704_153_600_000),
            Some(0)
        ]
    );
    assert_eq!(
        timestamps_utc.data_type(),
        &DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into()))
    );
    let dates = batch
        .column(7)
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("event_date column");
    assert_eq!(
        dates.iter().collect::<Vec<_>>(),
        vec![Some(19_723), None, Some(19_724), Some(0)]
    );
    let dates64 = batch
        .column(8)
        .as_any()
        .downcast_ref::<Date64Array>()
        .expect("event_date64 column");
    assert_eq!(
        dates64.iter().collect::<Vec<_>>(),
        vec![
            Some(1_704_067_200_000),
            None,
            Some(1_704_153_600_000),
            Some(0)
        ]
    );
    let tags = batch
        .column(9)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("tags column");
    assert_eq!(tags.len(), 4);
    assert_eq!(tags.value_length(0), 2);
    assert!(tags.is_null(1));
    assert_eq!(tags.value_length(2), 2);
    assert_eq!(tags.value_length(3), 0);
    let attrs = batch
        .column(10)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("attrs column");
    assert_eq!(attrs.len(), 4);
    assert_eq!(attrs.column_by_name("rank").expect("rank").len(), 4);
}

#[tokio::test]
async fn filters_richer_parquet_types() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let decimal_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("amount = '123.45'").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("decimal filter batches");
    assert_eq!(ids_from_batches(&decimal_batches), vec![1]);

    let decimal_trailing_zero_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("amount = '123.4500'").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("decimal trailing zero filter batches");
    assert_eq!(ids_from_batches(&decimal_trailing_zero_batches), vec![1]);

    let timestamp_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("created_at >= '2024-01-02 00:00:00'").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("timestamp filter batches");
    assert_eq!(ids_from_batches(&timestamp_batches), vec![3]);

    let timestamp_tz_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("created_at_utc >= '2024-01-02 00:00:00'").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("timestamp timezone filter batches");
    assert_eq!(ids_from_batches(&timestamp_tz_batches), vec![3]);

    let timestamp_tz_offset_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(
                FilterExpr::parse("created_at_utc >= '2024-01-02 09:00:00+09:00'").expect("filter"),
            ),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("timestamp timezone offset filter batches");
    assert_eq!(ids_from_batches(&timestamp_tz_offset_batches), vec![3]);

    let date_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("event_date >= '2024-01-02'").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("date filter batches");
    assert_eq!(ids_from_batches(&date_batches), vec![3]);

    let date64_batches = DodamEngine::default()
        .scan_parquet_batches(
            path.clone(),
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("event_date64 >= '2024-01-02'").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("date64 filter batches");
    assert_eq!(ids_from_batches(&date64_batches), vec![3]);

    let boolean_batches = DodamEngine::default()
        .scan_parquet_batches(
            path,
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("flag = true").expect("filter")),
        )
        .await
        .expect("plan scan")
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("boolean filter batches");
    assert_eq!(ids_from_batches(&boolean_batches), vec![1, 4]);
}

#[tokio::test]
async fn prunes_row_groups_with_richer_type_statistics() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("rich.parquet");
    write_rich_type_parquet(&path);

    let decimal_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("amount = '123.45'").expect("filter")),
        )
        .await
        .expect("scan decimal filtered parquet");
    assert_eq!(decimal_metrics.rows, 1);
    assert_eq!(decimal_metrics.row_groups_total, 2);
    assert_eq!(decimal_metrics.row_groups_scanned, 1);
    assert_eq!(decimal_metrics.row_groups_pruned, 1);

    let decimal_trailing_zero_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("amount = '123.4500'").expect("filter")),
        )
        .await
        .expect("scan decimal trailing zero filtered parquet");
    assert_eq!(decimal_trailing_zero_metrics.rows, 1);
    assert_eq!(decimal_trailing_zero_metrics.row_groups_total, 2);
    assert_eq!(decimal_trailing_zero_metrics.row_groups_scanned, 1);
    assert_eq!(decimal_trailing_zero_metrics.row_groups_pruned, 1);

    let timestamp_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("created_at >= '2024-01-02 00:00:00'").expect("filter")),
        )
        .await
        .expect("scan timestamp filtered parquet");
    assert_eq!(timestamp_metrics.rows, 1);
    assert_eq!(timestamp_metrics.row_groups_total, 2);
    assert_eq!(timestamp_metrics.row_groups_scanned, 1);
    assert_eq!(timestamp_metrics.row_groups_pruned, 1);

    let timestamp_tz_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("created_at_utc >= '2024-01-02 00:00:00'").expect("filter")),
        )
        .await
        .expect("scan timestamp timezone filtered parquet");
    assert_eq!(timestamp_tz_metrics.rows, 1);
    assert_eq!(timestamp_tz_metrics.row_groups_total, 2);
    assert_eq!(timestamp_tz_metrics.row_groups_scanned, 1);
    assert_eq!(timestamp_tz_metrics.row_groups_pruned, 1);

    let timestamp_tz_offset_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(
                FilterExpr::parse("created_at_utc >= '2024-01-02 09:00:00+09:00'").expect("filter"),
            ),
        )
        .await
        .expect("scan timestamp timezone offset filtered parquet");
    assert_eq!(timestamp_tz_offset_metrics.rows, 1);
    assert_eq!(timestamp_tz_offset_metrics.row_groups_total, 2);
    assert_eq!(timestamp_tz_offset_metrics.row_groups_scanned, 1);
    assert_eq!(timestamp_tz_offset_metrics.row_groups_pruned, 1);

    let date_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("event_date >= '2024-01-02'").expect("filter")),
        )
        .await
        .expect("scan date filtered parquet");
    assert_eq!(date_metrics.rows, 1);
    assert_eq!(date_metrics.row_groups_total, 2);
    assert_eq!(date_metrics.row_groups_scanned, 1);
    assert_eq!(date_metrics.row_groups_pruned, 1);

    let date64_metrics = DodamEngine::default()
        .scan_parquet(
            path.clone(),
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("event_date64 >= '2024-01-02'").expect("filter")),
        )
        .await
        .expect("scan date64 filtered parquet");
    assert_eq!(date64_metrics.rows, 1);
    assert_eq!(date64_metrics.row_groups_total, 2);
    assert_eq!(date64_metrics.row_groups_scanned, 1);
    assert_eq!(date64_metrics.row_groups_pruned, 1);

    let boolean_metrics = DodamEngine::default()
        .scan_parquet(
            path,
            16,
            None,
            Projection::Columns(vec!["id".to_string()]),
            Some(FilterExpr::parse("flag = false").expect("filter")),
        )
        .await
        .expect("scan boolean filtered parquet");
    assert_eq!(boolean_metrics.rows, 1);
    assert_eq!(boolean_metrics.row_groups_total, 2);
    assert_eq!(boolean_metrics.row_groups_scanned, 1);
    assert_eq!(boolean_metrics.row_groups_pruned, 1);
}

#[tokio::test]
async fn order_by_makes_limit_deterministic() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let stream = DodamEngine::default()
        .scan_parquet_ordered_batches(
            path,
            4,
            Some(4),
            Projection::Columns(vec!["id".to_string()]),
            None,
            SortExpr::parse("id desc").expect("sort"),
        )
        .await
        .expect("scan ordered batch stream");
    let batches = stream
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("batches");

    assert_eq!(ids_from_batches(&batches), vec![9, 8, 7, 6]);
}

#[tokio::test]
async fn order_by_without_limit_returns_all_rows() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 5);

    let stream = DodamEngine::default()
        .scan_parquet_ordered_batches(
            path,
            2,
            None,
            Projection::Columns(vec!["id".to_string()]),
            None,
            SortExpr::parse("id desc").expect("sort"),
        )
        .await
        .expect("scan ordered batch stream");
    let batches = stream
        .collect::<dodam::error::Result<Vec<_>>>()
        .expect("batches");

    assert_eq!(ids_from_batches(&batches), vec![4, 3, 2, 1, 0]);
}

#[tokio::test]
async fn prunes_row_groups_with_filter_statistics() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet_with_row_group_size(&path, 12, 4);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            16,
            None,
            Projection::All,
            Some(FilterExpr::parse("id=10").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.rows, 1);
    assert_eq!(metrics.columns, 2);
    assert_eq!(metrics.row_groups_total, 3);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 2);
    assert!(metrics.compressed_bytes_total > 0);
    assert!(metrics.compressed_bytes_scanned > 0);
    assert!(metrics.compressed_bytes_pruned > 0);
    assert_eq!(
        metrics.compressed_bytes_scanned + metrics.compressed_bytes_pruned,
        metrics.compressed_bytes_total
    );
}

#[tokio::test]
async fn filters_with_and_expression() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            4,
            None,
            Projection::All,
            Some(FilterExpr::parse("id>=3 AND id<6").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 2);
    assert_eq!(metrics.rows, 3);
    assert_eq!(metrics.columns, 2);
}

#[tokio::test]
async fn prunes_row_groups_with_range_expression() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet_with_row_group_size(&path, 12, 4);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            16,
            None,
            Projection::Columns(vec!["payload".to_string()]),
            Some(FilterExpr::parse("id>=8 AND id<12").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.rows, 4);
    assert_eq!(metrics.columns, 1);
    assert_eq!(metrics.row_groups_total, 3);
    assert_eq!(metrics.row_groups_scanned, 1);
    assert_eq!(metrics.row_groups_pruned, 2);
}

#[tokio::test]
async fn non_prunable_predicate_stays_as_residual_filter() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet_with_row_group_size(&path, 12, 4);

    let metrics = DodamEngine::default()
        .scan_parquet(
            path,
            16,
            None,
            Projection::All,
            Some(FilterExpr::parse("id!=10").expect("filter")),
        )
        .await
        .expect("scan filtered parquet");

    assert_eq!(metrics.fragments, 1);
    assert!(metrics.batches >= 1);
    assert_eq!(metrics.rows, 11);
    assert_eq!(metrics.columns, 2);
    assert_eq!(metrics.row_groups_total, 3);
    assert_eq!(metrics.row_groups_scanned, 3);
    assert_eq!(metrics.row_groups_pruned, 0);
}

#[tokio::test]
async fn reuses_parquet_metadata_cache_across_scans() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet_with_row_group_size(&path, 12, 4);
    let engine = DodamEngine::default();

    engine
        .scan_parquet(path.clone(), 16, None, Projection::All, None)
        .await
        .expect("first scan");
    engine
        .scan_parquet(
            path,
            16,
            None,
            Projection::Columns(vec!["payload".to_string()]),
            Some(FilterExpr::parse("id>=8 AND id<12").expect("filter")),
        )
        .await
        .expect("second scan");

    assert_eq!(engine.metadata_cache_len(), 1);
}

#[tokio::test]
async fn computes_global_numeric_aggregates() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .aggregate_parquet(
            path,
            4,
            vec![
                AggregateExpr::parse("count(*)").expect("aggregate"),
                AggregateExpr::parse("count(id)").expect("aggregate"),
                AggregateExpr::parse("sum(id)").expect("aggregate"),
                AggregateExpr::parse("avg(id)").expect("aggregate"),
                AggregateExpr::parse("min(id)").expect("aggregate"),
                AggregateExpr::parse("max(id)").expect("aggregate"),
            ],
            None,
        )
        .await
        .expect("aggregate parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.batches, 3);
    assert_eq!(metrics.rows, 10);
    assert_eq!(metrics.values[0].value, AggregateValue::Count(10));
    assert_eq!(metrics.values[1].value, AggregateValue::Count(10));
    assert_eq!(metrics.values[2].value, AggregateValue::Int64(Some(45)));
    assert_eq!(metrics.values[3].value, AggregateValue::Float64(Some(4.5)));
    assert_eq!(metrics.values[4].value, AggregateValue::Int64(Some(0)));
    assert_eq!(metrics.values[5].value, AggregateValue::Int64(Some(9)));
}

#[tokio::test]
async fn computes_aggregates_after_filtering() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .aggregate_parquet(
            path,
            4,
            vec![
                AggregateExpr::parse("count(*)").expect("aggregate"),
                AggregateExpr::parse("sum(id)").expect("aggregate"),
            ],
            Some(FilterExpr::parse("id>=3 AND id<6").expect("filter")),
        )
        .await
        .expect("aggregate parquet");

    assert_eq!(metrics.rows, 3);
    assert_eq!(metrics.values[0].value, AggregateValue::Count(3));
    assert_eq!(metrics.values[1].value, AggregateValue::Int64(Some(12)));
}

#[tokio::test]
async fn computes_string_min_max_aggregates() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet(&path, 10);

    let metrics = DodamEngine::default()
        .aggregate_parquet(
            path,
            4,
            vec![
                AggregateExpr::parse("min(payload)").expect("aggregate"),
                AggregateExpr::parse("max(payload)").expect("aggregate"),
            ],
            None,
        )
        .await
        .expect("aggregate parquet");

    assert_eq!(
        metrics.values[0].value,
        AggregateValue::Utf8(Some("row-0".to_string()))
    );
    assert_eq!(
        metrics.values[1].value,
        AggregateValue::Utf8(Some("row-9".to_string()))
    );
}

#[tokio::test]
async fn computes_grouped_aggregates() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_test_parquet_with_payloads(&path, &[0, 1, 2, 3, 4, 5], &["a", "b", "a", "b", "a", "c"]);

    let metrics = DodamEngine::default()
        .aggregate_parquet_grouped(
            path,
            4,
            vec![
                AggregateExpr::parse("count(*)").expect("aggregate"),
                AggregateExpr::parse("sum(id)").expect("aggregate"),
            ],
            vec!["payload".to_string()],
            None,
        )
        .await
        .expect("aggregate parquet");

    assert_eq!(metrics.fragments, 1);
    assert_eq!(metrics.rows, 6);
    assert_eq!(metrics.groups.len(), 3);
    assert_eq!(metrics.groups[0].keys[0].to_string(), "a");
    assert_eq!(metrics.groups[0].values[0].value, AggregateValue::Count(3));
    assert_eq!(
        metrics.groups[0].values[1].value,
        AggregateValue::Int64(Some(6))
    );
    assert_eq!(metrics.groups[1].keys[0].to_string(), "b");
    assert_eq!(metrics.groups[1].values[0].value, AggregateValue::Count(2));
    assert_eq!(
        metrics.groups[1].values[1].value,
        AggregateValue::Int64(Some(4))
    );
    assert_eq!(metrics.groups[2].keys[0].to_string(), "c");
    assert_eq!(metrics.groups[2].values[0].value, AggregateValue::Count(1));
    assert_eq!(
        metrics.groups[2].values[1].value,
        AggregateValue::Int64(Some(5))
    );
}

#[tokio::test]
async fn computes_single_int_key_grouped_aggregates_with_fast_functions() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("part-000.parquet");
    write_group_fast_path_parquet(&path);

    let metrics = DodamEngine::default()
        .aggregate_parquet_grouped(
            path,
            3,
            vec![
                AggregateExpr::parse("count(value)").expect("aggregate"),
                AggregateExpr::parse("sum(value)").expect("aggregate"),
                AggregateExpr::parse("avg(value)").expect("aggregate"),
                AggregateExpr::parse("min(payload)").expect("aggregate"),
                AggregateExpr::parse("max(payload)").expect("aggregate"),
            ],
            vec!["bucket".to_string()],
            None,
        )
        .await
        .expect("aggregate parquet");

    assert_eq!(metrics.groups.len(), 2);
    assert_eq!(metrics.groups[0].keys[0].to_string(), "1");
    assert_eq!(metrics.groups[0].values[0].value, AggregateValue::Count(2));
    assert_eq!(
        metrics.groups[0].values[1].value,
        AggregateValue::Int64(Some(30))
    );
    assert_eq!(
        metrics.groups[0].values[2].value,
        AggregateValue::Float64(Some(15.0))
    );
    assert_eq!(
        metrics.groups[0].values[3].value,
        AggregateValue::Utf8(Some("a".to_string()))
    );
    assert_eq!(
        metrics.groups[0].values[4].value,
        AggregateValue::Utf8(Some("b".to_string()))
    );
    assert_eq!(metrics.groups[1].keys[0].to_string(), "2");
    assert_eq!(metrics.groups[1].values[0].value, AggregateValue::Count(3));
    assert_eq!(
        metrics.groups[1].values[1].value,
        AggregateValue::Int64(Some(45))
    );
    assert_eq!(
        metrics.groups[1].values[2].value,
        AggregateValue::Float64(Some(15.0))
    );
    assert_eq!(
        metrics.groups[1].values[3].value,
        AggregateValue::Utf8(Some("c".to_string()))
    );
    assert_eq!(
        metrics.groups[1].values[4].value,
        AggregateValue::Utf8(Some("e".to_string()))
    );
}

fn write_test_parquet(path: &std::path::Path, rows: usize) {
    write_test_parquet_with_row_group_size(path, rows, rows);
}

fn write_test_parquet_with_row_group_size(
    path: &std::path::Path,
    rows: usize,
    row_group_size: usize,
) {
    let ids = (0..rows).map(|value| value as i32).collect::<Vec<_>>();
    let payloads = (0..rows)
        .map(|value| format!("row-{value}"))
        .collect::<Vec<_>>();
    let payloads = payloads.iter().map(String::as_str).collect::<Vec<_>>();
    write_test_parquet_with_ids_payloads_and_row_group_size(path, &ids, &payloads, row_group_size);
}

fn write_test_parquet_with_payloads(path: &std::path::Path, ids: &[i32], payloads: &[&str]) {
    write_test_parquet_with_ids_payloads_and_row_group_size(path, ids, payloads, ids.len());
}

fn write_id_only_test_parquet(path: &std::path::Path, rows: usize) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values((0..rows).map(|value| value as i32));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
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

fn write_group_fast_path_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let buckets = Int32Array::from_iter_values([1, 1, 2, 2, 2]);
    let values = Int64Array::from_iter_values([10, 20, 5, 15, 25]);
    let payloads = StringArray::from_iter_values(["b", "a", "d", "c", "e"]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(buckets), Arc::new(values), Arc::new(payloads)],
    )
    .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(3))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn write_test_parquet_with_ids_payloads_and_row_group_size(
    path: &std::path::Path,
    ids: &[i32],
    payloads: &[&str],
    row_group_size: usize,
) {
    assert_eq!(ids.len(), payloads.len());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let ids = Int32Array::from_iter_values(ids.iter().copied());
    let payloads = StringArray::from_iter_values(payloads.iter().copied());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(payloads)])
        .expect("record batch");

    let file = File::create(path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_size))
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
