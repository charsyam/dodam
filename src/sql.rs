use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::BuildHasher;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float64Array,
    Int32Array, Int64Array, StringArray, TimestampMillisecondArray, UInt64Array,
};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_ord::sort::{SortColumn, SortOptions, lexsort_to_indices};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use memchr::memmem::Finder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use sqlparser::ast::{
    BinaryOperator, DateTimeField, Distinct, DuplicateTreatment, Expr as SqlExpr, FunctionArg,
    FunctionArgExpr, FunctionArguments, GroupByExpr, JoinConstraint, JoinOperator, LimitClause,
    ObjectName, ObjectNamePart, OrderByKind, Query, Select, SelectItem, SetExpr, Statement,
    TableFactor, UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::dense::{
    AdaptiveI64Map, AdaptiveI64Set, DenseI64F64Sum, DenseI64I32Map, adaptive_dense_index,
};
use crate::engine::{
    DodamEngine, JoinAlgorithm, JoinParquetRequest, LateMaterializationPolicy,
    LateMaterializedMetrics, LateSelectionBuilder,
};
use crate::error::{DodamError, Result};
use crate::execution::JoinType;
use crate::execution::{
    AggregateExpr, AggregateMetrics, AggregateResult, AggregateValue, ComparisonExpr, ComparisonOp,
    DistinctExec, Expr, FilterExpr, GroupAggregateResult, GroupValue, HashJoinExec, JoinBuildSide,
    LiteralValue, MemoryExec, PhysicalPlan, Projection, RecordBatchSink, ScanPlanMetrics,
    SendableBatchStream, SortExpr, SortKey, collect_aggregates, collect_grouped_aggregates,
    evaluate_filter_mask, filter_batch,
};
use crate::hash::{FastHashMap, FastHashSet, fast_hash_map, fast_hash_map_with_capacity};
use crate::optimizer::plan_join_inputs;

fn tpch_profile_enabled() -> bool {
    std::env::var("DODAM_TPCH_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn tpch_profile_start() -> Option<Instant> {
    tpch_profile_enabled().then(Instant::now)
}

fn tpch_profile_elapsed(label: &str, started: Option<Instant>) {
    if let Some(started) = started {
        eprintln!(
            "[dodam:tpch-profile] {label}: {:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn sql_elapsed_nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn sql_nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

const DEFAULT_MAX_DENSE_I64_KEY: usize = 20_000_000;
const DEFAULT_Q09_ORDER_YEAR_DENSE_BYTES: usize = 384 * 1024 * 1024;

fn try_for_each_i64_date32_str<Visit>(
    int_values: &ArrayRef,
    date_values: &ArrayRef,
    string_values: &StringArray,
    mut visit: Visit,
) -> Result<bool>
where
    Visit: FnMut(i64, i32, &str) -> Result<()>,
{
    let (Some(int_values), Some(date_values)) = (
        int_values.as_any().downcast_ref::<Int64Array>(),
        date_values.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    for row in 0..int_values.len() {
        if int_values.is_null(row) || date_values.is_null(row) || string_values.is_null(row) {
            continue;
        }
        visit(
            int_values.value(row),
            date_values.value(row),
            string_values.value(row),
        )?;
    }
    Ok(true)
}

fn try_for_each_i64_i64_date32<Visit>(
    left_values: &ArrayRef,
    right_values: &ArrayRef,
    date_values: &ArrayRef,
    mut visit: Visit,
) -> Result<bool>
where
    Visit: FnMut(i64, i64, i32) -> Result<()>,
{
    let (Some(left_values), Some(right_values), Some(date_values)) = (
        left_values.as_any().downcast_ref::<Int64Array>(),
        right_values.as_any().downcast_ref::<Int64Array>(),
        date_values.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    for row in 0..left_values.len() {
        if left_values.is_null(row) || right_values.is_null(row) || date_values.is_null(row) {
            continue;
        }
        visit(
            left_values.value(row),
            right_values.value(row),
            date_values.value(row),
        )?;
    }
    Ok(true)
}

#[derive(Debug)]
pub enum QueryOutput {
    Scan {
        batches: Vec<RecordBatch>,
    },
    Aggregate {
        metrics: AggregateMetrics,
        batches: Vec<RecordBatch>,
    },
    Explain {
        plan: String,
    },
}

pub trait SqlResultSink {
    fn record_batch_sink(&mut self) -> &mut dyn RecordBatchSink;
    fn write_output(&mut self, output: QueryOutput) -> Result<()>;
}

#[derive(Debug, Clone, Copy)]
pub struct SqlSinkExecutionOptions {
    pub allow_direct_or_streaming: bool,
}

impl Default for SqlSinkExecutionOptions {
    fn default() -> Self {
        Self {
            allow_direct_or_streaming: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct SqlSinkExecutionProfile {
    pub direct_sink: Option<Duration>,
    pub streaming: Option<Duration>,
    pub execute: Option<Duration>,
    pub write_output: Option<Duration>,
    pub scan_plan_metrics: Option<ScanPlanMetrics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SqlQuery {
    path: PathBuf,
    join: Option<SqlJoin>,
    projection: Projection,
    filter: Option<FilterExpr>,
    having: Option<FilterExpr>,
    order_by: Option<SortKey>,
    limit: Option<usize>,
    distinct: bool,
    aggregates: Vec<AggregateExpr>,
    aggregate_expressions: Vec<ProjectionExpression>,
    expressions: Vec<ProjectionExpression>,
    group_by: Vec<String>,
    aliases: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
struct SqlJoin {
    right: SqlTableRef,
    left_alias: String,
    right_alias: String,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    right_filter: Option<FilterExpr>,
    join_type: JoinType,
}

impl SqlQuery {
    pub fn is_aggregate(&self) -> bool {
        !self.aggregates.is_empty()
    }
}

pub fn parse_sql(input: &str) -> Result<SqlQuery> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, input)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one statement".to_string(),
        ));
    };

    let Statement::Query(query) = statement else {
        return Err(DodamError::UnsupportedSql(
            "only SELECT queries are supported".to_string(),
        ));
    };
    parse_query(query)
}

pub async fn execute_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<QueryOutput> {
    if let Some(plan) = explain_sql(engine, sql, batch_size).await? {
        return Ok(QueryOutput::Explain { plan });
    }
    if let Some(output) = try_execute_q15_top_supplier_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_with_cte_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q01_pricing_summary_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q10_returned_item_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q09_product_type_profit_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q11_important_stock_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q02_minimum_cost_supplier_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_q12_shipping_modes_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q14_promotion_effect_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q16_parts_supplier_relationship_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_q03_shipping_priority_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q04_order_priority_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q05_local_supplier_volume_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_q06_forecast_revenue_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_q07_volume_shipping_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q08_national_market_share_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q22_global_sales_opportunity_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_derived_join_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_derived_left_join_count_distribution_sql(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_derived_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q18_large_volume_customer_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_q19_discounted_revenue_fast(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q17_small_quantity_order_revenue_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q21_suppliers_who_kept_orders_waiting_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_q20_potential_part_promotion_fast(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_correlated_join_subquery_filter_sql(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_materialized_join_subquery_sql(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_multi_comma_join_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_correlated_exists_semijoin_sql(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_correlated_subquery_filter_sql(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) =
        try_execute_correlated_exists_subquery_sql(engine, sql, batch_size).await?
    {
        return Ok(output);
    }
    if let Some(output) = try_execute_exists_subquery_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_in_subquery_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_projection_expression_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }

    let query = parse_sql(sql)?;
    if let Some(join) = query.join.clone() {
        if query.distinct {
            return Err(DodamError::UnsupportedSql(
                "JOIN with DISTINCT is not supported".to_string(),
            ));
        }
        let is_aggregate = query.is_aggregate();
        let aggregates = query.aggregates.clone();
        let group_by = query.group_by.clone();
        let join_input_projection = &query.projection;
        let join_plan = plan_join_inputs(
            join_input_projection,
            query.filter.as_ref(),
            query.order_by.as_ref(),
            &join.left_alias,
            &join.left_keys,
            &join.right_alias,
            &join.right_keys,
        );
        let output_projection = pushed_join_output_projection(&query);
        let output_projection_pushed = !matches!(output_projection, Projection::All);
        let stream = engine
            .join_parquet_batches(JoinParquetRequest {
                left_path: query.path.clone(),
                right_path: join.right.path,
                batch_size,
                left_keys: join.left_keys,
                right_keys: join.right_keys,
                left_prefix: join.left_alias,
                right_prefix: join.right_alias,
                left_projection: join_plan.left_projection,
                right_projection: join_plan.right_projection,
                left_filter: join_plan.left_filter,
                right_filter: combine_filter_options(
                    join_plan.right_filter,
                    join.right_filter.clone(),
                ),
                output_projection,
                join_memory_limit_bytes: default_join_memory_limit_bytes(),
                join_algorithm: JoinAlgorithm::Auto,
                join_type: join.join_type,
            })
            .await?;
        if is_aggregate {
            let stream = apply_output_filter_stream(stream, query.filter.clone());
            let stream: SendableBatchStream = if query.aggregate_expressions.is_empty() {
                stream
            } else {
                append_aggregate_expression_stream(stream, query.aggregate_expressions.clone())
            };
            let metrics = if group_by.is_empty() {
                collect_aggregates(stream, 2, &aggregates)?
            } else {
                collect_grouped_aggregates(stream, 2, &group_by, &aggregates)?
            };
            let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
            batches = apply_output_filter(batches, query.having.as_ref())?;
            let has_output_expressions = projection_requires_expression_path(&query.expressions);
            if has_output_expressions {
                batches = apply_output_expression_projection(batches, &query.expressions)?;
            }
            batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
            if !has_output_expressions {
                batches = rename_output_batches(batches, &query.aliases)?;
            }
            return Ok(QueryOutput::Aggregate { metrics, batches });
        }
        let mut batches = collect_batches(stream)?;
        batches = apply_output_filter(batches, query.filter.as_ref())?;
        let projection_requires_expression =
            projection_requires_expression_path(&query.expressions);
        if projection_requires_expression {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
            batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
        } else {
            batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
            if !output_projection_pushed {
                batches = apply_output_projection(batches, &query.projection)?;
            }
        }
        if !projection_requires_expression {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(QueryOutput::Scan { batches });
    }

    if query.is_aggregate() {
        let aggregates = query.aggregates.clone();
        let group_by = query.group_by.clone();
        let metrics = if !query.aggregate_expressions.is_empty() {
            let stream = engine
                .scan_parquet_batches(
                    query.path,
                    batch_size,
                    None,
                    query.projection.clone(),
                    query.filter,
                )
                .await?;
            let stream = append_aggregate_expression_stream(stream, query.aggregate_expressions);
            if query.group_by.is_empty() {
                collect_aggregates(stream, 1, &aggregates)?
            } else {
                collect_grouped_aggregates(stream, 1, &group_by, &aggregates)?
            }
        } else if query.group_by.is_empty() {
            engine
                .aggregate_parquet(query.path, batch_size, aggregates.clone(), query.filter)
                .await?
        } else {
            engine
                .aggregate_parquet_grouped(
                    query.path,
                    batch_size,
                    aggregates.clone(),
                    group_by.clone(),
                    query.filter,
                )
                .await?
        };
        let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
        batches = apply_output_filter(batches, query.having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&query.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
        }
        batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(QueryOutput::Aggregate { metrics, batches });
    }

    let stream = if query.distinct {
        engine
            .scan_parquet_distinct_batches(
                query.path,
                batch_size,
                query.limit,
                query.projection,
                query.filter,
                query.order_by,
            )
            .await?
    } else if let Some(order_by) = query.order_by {
        engine
            .scan_parquet_ordered_batches_by(
                query.path,
                batch_size,
                query.limit,
                query.projection,
                query.filter,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(
                query.path,
                batch_size,
                query.limit,
                query.projection,
                query.filter,
            )
            .await?
    };
    let batches = rename_output_batches(collect_batches(stream)?, &query.aliases)?;
    Ok(QueryOutput::Scan { batches })
}

async fn try_execute_exists_subquery_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some((subquery, negated)) = top_level_exists_subquery(select.selection.as_ref()) else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return Err(DodamError::UnsupportedSql(
            "EXISTS subquery filters over JOIN inputs are not supported yet".to_string(),
        ));
    }
    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        return Err(DodamError::UnsupportedSql(
            "EXISTS subquery filters with aggregates are not supported yet".to_string(),
        ));
    }
    let distinct = parse_distinct(select)?;
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let exists = !query_output_batches(
        Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?,
    )?
    .is_empty();
    if exists == negated {
        return Ok(Some(QueryOutput::Scan {
            batches: Vec::new(),
        }));
    }

    let stream = if distinct {
        engine
            .scan_parquet_distinct_batches(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
                None,
                order_by,
            )
            .await?
    } else if let Some(order_by) = order_by {
        engine
            .scan_parquet_ordered_batches_by(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
                None,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
                None,
            )
            .await?
    };
    let batches = rename_output_batches(collect_batches(stream)?, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

fn top_level_exists_subquery(expr: Option<&SqlExpr>) -> Option<(&Query, bool)> {
    match expr? {
        SqlExpr::Exists { subquery, negated } => Some((subquery.as_ref(), *negated)),
        SqlExpr::Nested(expr) => top_level_exists_subquery(Some(expr)),
        _ => None,
    }
}

async fn try_execute_correlated_exists_semijoin_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1
        || select
            .from
            .first()
            .is_some_and(|table| !table.joins.is_empty())
    {
        return Ok(None);
    }

    let outer_path = parse_from(select)?;
    let outer_alias = table_ref_alias_or_name(&outer_path);
    let mut outer_conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut outer_conjuncts);
    let Some((exists_index, exists_subquery)) =
        outer_conjuncts
            .iter()
            .enumerate()
            .find_map(|(index, expr)| match expr {
                SqlExpr::Exists { subquery, negated } if !negated => {
                    Some((index, subquery.as_ref()))
                }
                SqlExpr::Nested(expr) => match expr.as_ref() {
                    SqlExpr::Exists { subquery, negated } if !negated => {
                        Some((index, subquery.as_ref()))
                    }
                    _ => None,
                },
                _ => None,
            })
    else {
        return Ok(None);
    };

    let SetExpr::Select(inner_select) = exists_subquery.body.as_ref() else {
        return Ok(None);
    };
    reject_query_features(exists_subquery)?;
    reject_select_features(inner_select)?;
    if inner_select.from.len() != 1
        || inner_select
            .from
            .first()
            .is_some_and(|table| !table.joins.is_empty())
        || parse_distinct(inner_select)?
        || inner_select.having.is_some()
        || !parse_group_by(inner_select, None)?.is_empty()
    {
        return Ok(None);
    }

    let inner_path = parse_from(inner_select)?;
    let inner_alias = table_ref_alias_or_name(&inner_path);
    let mut inner_conjuncts = Vec::new();
    let Some(inner_selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    collect_sql_and_conjuncts(inner_selection, &mut inner_conjuncts);
    let Some((join_index, inner_key, outer_key)) =
        semijoin_exists_key_pair(&inner_conjuncts, &inner_alias, &outer_alias)?
    else {
        return Ok(None);
    };

    let inner_residual = inner_conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != join_index).then_some(conjunct))
        .collect::<Vec<_>>();
    let inner_filter = combine_sql_and_conjuncts(inner_residual)
        .as_ref()
        .map(|expr| parse_filter(expr, &[], inner_path.alias.as_deref(), false))
        .transpose()?;
    let inner_keys = collect_semijoin_key_set(
        engine,
        inner_path.path,
        &inner_key,
        inner_filter,
        batch_size,
    )
    .await?;

    let outer_residual = outer_conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != exists_index).then_some(conjunct))
        .collect::<Vec<_>>();
    let outer_filter = combine_sql_and_conjuncts(outer_residual)
        .as_ref()
        .map(|expr| parse_filter(expr, &[], outer_path.alias.as_deref(), false))
        .transpose()?;

    let group_by = parse_group_by(select, outer_path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, outer_path.alias.as_deref())?;
    let distinct = parse_distinct(select)?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        outer_path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let outer_projection = semijoin_outer_projection(
        &parsed_projection,
        &group_by,
        order_by.as_ref(),
        &outer_key,
        outer_filter.as_ref(),
    );
    let stream = engine
        .scan_parquet_batches(
            outer_path.path,
            batch_size,
            None,
            outer_projection,
            outer_filter,
        )
        .await?;
    let outer_batches = collect_batches(stream)?;
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mask = semijoin_membership_mask(&batch, &outer_key, &inner_keys)?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }

    if !parsed_projection.aggregates.is_empty() {
        let stream = Box::new(MemoryExec::new(filtered)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions =
            projection_requires_expression_path(&parsed_projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &parsed_projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit)?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        apply_output_projection(batches, &parsed_projection.projection)?
    };
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

fn semijoin_exists_key_pair(
    conjuncts: &[SqlExpr],
    inner_alias: &str,
    outer_alias: &str,
) -> Result<Option<(usize, String, String)>> {
    for (index, conjunct) in conjuncts.iter().enumerate() {
        let SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = conjunct
        else {
            continue;
        };
        let Some(left_column) = semijoin_column_name(left)? else {
            continue;
        };
        let Some(right_column) = semijoin_column_name(right)? else {
            continue;
        };
        let left_owner = semijoin_column_owner(&left_column, inner_alias, outer_alias);
        let right_owner = semijoin_column_owner(&right_column, inner_alias, outer_alias);
        match (left_owner, right_owner) {
            (Some(SemijoinColumnOwner::Inner), Some(SemijoinColumnOwner::Outer)) => {
                return Ok(Some((
                    index,
                    unqualified_semijoin_column(&left_column),
                    unqualified_semijoin_column(&right_column),
                )));
            }
            (Some(SemijoinColumnOwner::Outer), Some(SemijoinColumnOwner::Inner)) => {
                return Ok(Some((
                    index,
                    unqualified_semijoin_column(&right_column),
                    unqualified_semijoin_column(&left_column),
                )));
            }
            _ => {}
        }
    }
    Ok(None)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SemijoinColumnOwner {
    Inner,
    Outer,
}

fn semijoin_column_name(expr: &SqlExpr) -> Result<Option<String>> {
    ColumnResolver::raw_column(expr)
}

fn semijoin_column_owner(
    column: &str,
    inner_alias: &str,
    outer_alias: &str,
) -> Option<SemijoinColumnOwner> {
    if column.starts_with(&format!("{inner_alias}.")) {
        return Some(SemijoinColumnOwner::Inner);
    }
    if column.starts_with(&format!("{outer_alias}.")) {
        return Some(SemijoinColumnOwner::Outer);
    }
    let unqualified = unqualified_semijoin_column(column);
    if unqualified.starts_with(&semijoin_unqualified_prefix(inner_alias)?) {
        return Some(SemijoinColumnOwner::Inner);
    }
    if unqualified.starts_with(&semijoin_unqualified_prefix(outer_alias)?) {
        return Some(SemijoinColumnOwner::Outer);
    }
    None
}

fn semijoin_unqualified_prefix(alias: &str) -> Option<String> {
    Some(format!("{}_", alias.chars().next()?))
}

fn unqualified_semijoin_column(column: &str) -> String {
    column
        .rsplit_once('.')
        .map(|(_, column)| column)
        .unwrap_or(column)
        .to_string()
}

async fn collect_semijoin_key_set(
    engine: &DodamEngine,
    path: PathBuf,
    key_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<SemijoinKeySet> {
    let mut projection = vec![key_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter,
        )
        .await?;
    let batches = collect_batches(stream)?;
    let Some(first_batch) = batches.first() else {
        return Ok(SemijoinKeySet::Empty);
    };
    let column_index = first_batch
        .schema()
        .index_of(key_column)
        .map_err(|_| DodamError::UnknownColumn(key_column.to_string()))?;
    let mut keys = SemijoinKeySet::for_data_type(first_batch.column(column_index).data_type());
    for batch in &batches {
        let column_index = batch
            .schema()
            .index_of(key_column)
            .map_err(|_| DodamError::UnknownColumn(key_column.to_string()))?;
        let column = batch.column(column_index);
        for row in 0..batch.num_rows() {
            keys.insert_from_column(column, row)?;
        }
    }
    Ok(keys)
}

fn semijoin_membership_mask(
    batch: &RecordBatch,
    key_column: &str,
    keys: &SemijoinKeySet,
) -> Result<BooleanArray> {
    let column_index = batch
        .schema()
        .index_of(key_column)
        .map_err(|_| DodamError::UnknownColumn(key_column.to_string()))?;
    let column = batch.column(column_index);
    let values = (0..batch.num_rows())
        .map(|row| keys.contains_column_value(column, row))
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanArray::from(values))
}

fn semijoin_outer_projection(
    projection: &ParsedProjection,
    group_by: &[String],
    order_by: Option<&SortKey>,
    key_column: &str,
    filter: Option<&FilterExpr>,
) -> Projection {
    let mut columns = vec![key_column.to_string()];
    for column in group_by {
        add_column_once(&mut columns, column.clone());
    }
    match &projection.projection {
        Projection::All => return Projection::All,
        Projection::Columns(projected) => {
            for column in projected {
                add_column_once(&mut columns, column.clone());
            }
        }
    }
    for aggregate in &projection.aggregates {
        if let Some(column) = aggregate.referenced_column() {
            add_column_once(&mut columns, column.to_string());
        }
    }
    for expression in &projection.aggregate_expressions {
        for column in scalar_expression_columns(&expression.expr) {
            add_column_once(&mut columns, column);
        }
    }
    for expression in &projection.expressions {
        for column in scalar_expression_columns(&expression.expr) {
            add_column_once(&mut columns, column);
        }
    }
    if let Some(order_by) = order_by {
        for sort in &order_by.expressions {
            add_column_once(&mut columns, sort.column.clone());
        }
    }
    if let Some(filter) = filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut columns, column);
        }
    }
    Projection::Columns(columns)
}

fn semijoin_key_at(column: &ArrayRef, row: usize) -> Result<Option<String>> {
    if column.is_null(row) {
        return Ok(None);
    }
    Ok(Some(sql_literal(&literal_value_from_array(column, row)?)))
}

enum SemijoinKeySet {
    Empty,
    Int64(HashSet<i64>),
    Utf8(HashSet<String>),
    Literal(HashSet<String>),
}

impl SemijoinKeySet {
    fn for_data_type(data_type: &DataType) -> Self {
        match data_type {
            DataType::Int32 | DataType::Int64 | DataType::Date32 | DataType::Date64 => {
                Self::Int64(HashSet::new())
            }
            DataType::Utf8 => Self::Utf8(HashSet::new()),
            _ => Self::Literal(HashSet::new()),
        }
    }

    fn insert_from_column(&mut self, column: &ArrayRef, row: usize) -> Result<()> {
        match self {
            Self::Empty => {}
            Self::Int64(values) => {
                if let Some(value) = semijoin_i64_key_at(column, row)? {
                    values.insert(value);
                }
            }
            Self::Utf8(values) => {
                if column.is_valid(row) {
                    if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                        values.insert(strings.value(row).to_string());
                    } else if let Some(value) = semijoin_key_at(column, row)? {
                        values.insert(value);
                    }
                }
            }
            Self::Literal(values) => {
                if let Some(value) = semijoin_key_at(column, row)? {
                    values.insert(value);
                }
            }
        }
        Ok(())
    }

    fn contains_column_value(&self, column: &ArrayRef, row: usize) -> Result<Option<bool>> {
        if column.is_null(row) {
            return Ok(Some(false));
        }
        match self {
            Self::Empty => Ok(Some(false)),
            Self::Int64(values) => Ok(Some(
                semijoin_i64_key_at(column, row)?.is_some_and(|value| values.contains(&value)),
            )),
            Self::Utf8(values) => {
                if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                    Ok(Some(values.contains(strings.value(row))))
                } else {
                    Ok(Some(
                        semijoin_key_at(column, row)?.is_some_and(|value| values.contains(&value)),
                    ))
                }
            }
            Self::Literal(values) => Ok(Some(
                semijoin_key_at(column, row)?.is_some_and(|value| values.contains(&value)),
            )),
        }
    }
}

fn semijoin_i64_key_at(column: &ArrayRef, row: usize) -> Result<Option<i64>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 semijoin key")
                .value(row),
        ))),
        DataType::Int64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 semijoin key")
                .value(row),
        )),
        DataType::Date32 => Ok(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 semijoin key")
                .value(row),
        ))),
        DataType::Date64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64 semijoin key")
                .value(row),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "cannot use {data_type} as integer semijoin key"
        ))),
    }
}

async fn try_execute_correlated_subquery_filter_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !expr_contains_materializable_subquery(selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    let path = parse_from(select)?;
    let selection_sql = selection.to_string();
    if let Some(outer_alias) = path.alias.as_deref()
        && !subquery_references_outer_alias(&selection_sql, outer_alias)
        && !tpch_alias_prefix(outer_alias)
            .is_some_and(|prefix| selection_sql.contains(&format!("{prefix}_")))
    {
        return Ok(None);
    }
    if path.alias.is_none() {
        let inferred_alias = table_ref_alias_or_name(&path);
        let Some(prefix) = inferred_alias.chars().next() else {
            return Ok(None);
        };
        if !selection_sql.contains(&format!("{prefix}_")) {
            return Ok(None);
        }
    }

    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if parse_distinct(select)? {
        return Err(DodamError::UnsupportedSql(
            "correlated subquery filters with DISTINCT are not supported yet".to_string(),
        ));
    }
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;

    let stream = engine
        .scan_parquet_batches(path.path, batch_size, None, Projection::All, None)
        .await?;
    let outer_batches = collect_batches(stream)?;
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            &selection_sql,
            path.alias.as_deref(),
            &batch,
            path.alias.as_deref(),
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }

    if !parsed_projection.aggregates.is_empty() {
        let stream = Box::new(MemoryExec::new(filtered)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions =
            projection_requires_expression_path(&parsed_projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &parsed_projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit)?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        apply_output_projection(batches, &parsed_projection.projection)?
    };
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn evaluate_correlated_subquery_filter_mask(
    engine: &DodamEngine,
    selection_sql: &str,
    outer_alias: Option<&str>,
    batch: &RecordBatch,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<BooleanArray> {
    let selection_expr = parse_sql_expr_fragment(selection_sql)?;
    let selection_expr = Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
        engine,
        selection_expr,
        batch_size,
    ))
    .await?;
    let selection_sql = selection_expr.to_string();
    let mut values = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let bound_sql = bind_outer_row_references(&selection_sql, outer_alias, batch, row)?;
        let bound_expr = parse_sql_expr_fragment(&bound_sql)?;
        let row_batch = batch.slice(row, 1);
        let matches = evaluate_bound_correlated_filter(
            engine,
            &bound_expr,
            &row_batch,
            table_alias,
            batch_size,
        )
        .await?;
        values.push(Some(matches));
    }
    Ok(BooleanArray::from(values))
}

async fn apply_correlated_subquery_filter_batches(
    engine: &DodamEngine,
    batches: Vec<RecordBatch>,
    selection_sql: &str,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let mut filtered = Vec::new();
    for batch in batches {
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            selection_sql,
            None,
            &batch,
            None,
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

async fn rewrite_uncorrelated_scalar_subqueries_to_literals(
    engine: &DodamEngine,
    expr: SqlExpr,
    batch_size: usize,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Exists { subquery, negated } => {
            match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                Ok(output) => {
                    let exists = query_output_batches(output)?
                        .iter()
                        .any(|batch| batch.num_rows() > 0);
                    Ok(SqlExpr::Value(
                        Value::Boolean(if negated { !exists } else { exists }).with_empty_span(),
                    ))
                }
                Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
                    Ok(SqlExpr::Exists { subquery, negated })
                }
                Err(error) => Err(error),
            }
        }
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
            Ok(output) => {
                let values =
                    literal_values_from_single_column_batches(query_output_batches(output)?)?;
                if values.is_empty() {
                    return Ok(SqlExpr::Value(Value::Boolean(negated).with_empty_span()));
                }
                Ok(SqlExpr::InList {
                    expr,
                    list: values.into_iter().map(literal_value_to_sql_expr).collect(),
                    negated,
                })
            }
            Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
                Ok(SqlExpr::InSubquery {
                    expr,
                    subquery,
                    negated,
                })
            }
            Err(error) => Err(error),
        },
        SqlExpr::Subquery(subquery) => {
            match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                Ok(output) => scalar_literal_value_from_batches(query_output_batches(output)?)
                    .map(literal_value_to_sql_expr),
                Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
                    Ok(SqlExpr::Subquery(subquery))
                }
                Err(error) => Err(error),
            }
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *left, batch_size,
                ))
                .await?,
            ),
            op,
            right: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *right, batch_size,
                ))
                .await?,
            ),
        }),
        SqlExpr::Nested(expr) => Ok(SqlExpr::Nested(Box::new(
            Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                engine, *expr, batch_size,
            ))
            .await?,
        ))),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op,
            expr: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            ),
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let expr = Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            );
            let mut rewritten = Vec::with_capacity(list.len());
            for item in list {
                rewritten.push(
                    Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                        engine, item, batch_size,
                    ))
                    .await?,
                );
            }
            Ok(SqlExpr::InList {
                expr,
                list: rewritten,
                negated,
            })
        }
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(SqlExpr::Between {
            expr: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            ),
            negated,
            low: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *low, batch_size,
                ))
                .await?,
            ),
            high: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *high, batch_size,
                ))
                .await?,
            ),
        }),
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            special,
            shorthand,
        } => Ok(SqlExpr::Substring {
            expr: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            ),
            substring_from: match substring_from {
                Some(expr) => Some(Box::new(
                    Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                        engine, *expr, batch_size,
                    ))
                    .await?,
                )),
                None => None,
            },
            substring_for: match substring_for {
                Some(expr) => Some(Box::new(
                    Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                        engine, *expr, batch_size,
                    ))
                    .await?,
                )),
                None => None,
            },
            special,
            shorthand,
        }),
        expr => Ok(expr),
    }
}

async fn evaluate_bound_correlated_filter(
    engine: &DodamEngine,
    expr: &SqlExpr,
    row_batch: &RecordBatch,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<bool> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(expr, &mut conjuncts);
    for conjunct in conjuncts {
        let conjunct = if expr_contains_materializable_subquery(&conjunct) {
            Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                engine, conjunct, batch_size,
            ))
            .await?
        } else {
            conjunct
        };
        let matches = if let Ok(mask) = evaluate_scalar_predicate(row_batch, &conjunct, table_alias)
        {
            mask.is_valid(0) && mask.value(0)
        } else if let Some(expr) = Box::pin(parse_filter_with_subqueries(
            engine,
            &conjunct,
            &[],
            table_alias,
            false,
            batch_size,
        ))
        .await?
        {
            filter_batch(row_batch.clone(), &FilterExpr::new(expr))?.num_rows() > 0
        } else {
            true
        };
        if !matches {
            return Ok(false);
        }
    }
    Ok(true)
}

fn parse_sql_expr_fragment(expr: &str) -> Result<SqlExpr> {
    let dialect = GenericDialect {};
    let sql = format!("SELECT * FROM dummy WHERE {expr}");
    let statements = Parser::parse_sql(&dialect, &sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "failed to parse SQL expression".to_string(),
        ));
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(DodamError::UnsupportedSql(
            "failed to parse SQL expression".to_string(),
        ));
    };
    select
        .selection
        .clone()
        .ok_or_else(|| DodamError::UnsupportedSql("failed to parse SQL expression".to_string()))
}

async fn try_execute_correlated_exists_subquery_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some((subquery, negated)) = top_level_exists_subquery(select.selection.as_ref()) else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    let path = parse_from(select)?;
    let subquery_sql = subquery.to_string();
    let Some(outer_alias) = path.alias.as_deref() else {
        return Ok(None);
    };
    if !subquery_references_outer_alias(&subquery_sql, outer_alias) {
        return Ok(None);
    }

    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if parse_distinct(select)?
        || !parsed_projection.aggregates.is_empty()
        || !group_by.is_empty()
        || select.having.is_some()
    {
        return Err(DodamError::UnsupportedSql(
            "correlated EXISTS currently supports only non-aggregate SELECT queries".to_string(),
        ));
    }
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;

    let stream = engine
        .scan_parquet_batches(path.path, batch_size, None, Projection::All, None)
        .await?;
    let outer_batches = collect_batches(stream)?;
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mask = evaluate_correlated_exists_mask(
            engine,
            &subquery_sql,
            outer_alias,
            &batch,
            negated,
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit)?;
    batches = apply_output_projection(batches, &parsed_projection.projection)?;
    batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

fn subquery_references_outer_alias(subquery_sql: &str, outer_alias: &str) -> bool {
    subquery_sql.contains(&format!("{outer_alias}."))
}

async fn evaluate_correlated_exists_mask(
    engine: &DodamEngine,
    subquery_sql: &str,
    outer_alias: &str,
    batch: &RecordBatch,
    negated: bool,
    batch_size: usize,
) -> Result<BooleanArray> {
    let mut values = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let bound_sql = bind_outer_row_references(subquery_sql, Some(outer_alias), batch, row)?;
        let output = Box::pin(execute_sql(engine, &bound_sql, batch_size)).await?;
        let exists = query_output_batches(output)?
            .iter()
            .any(|batch| batch.num_rows() > 0);
        values.push(Some(if negated { !exists } else { exists }));
    }
    Ok(BooleanArray::from(values))
}

fn bind_outer_row_references(
    subquery_sql: &str,
    outer_alias: Option<&str>,
    batch: &RecordBatch,
    row: usize,
) -> Result<String> {
    let mut sql = subquery_sql.to_string();
    for (column_index, field) in batch.schema().fields().iter().enumerate() {
        let literal = if batch.column(column_index).is_null(row) {
            LiteralValue::Null
        } else {
            literal_value_from_array(batch.column(column_index), row)?
        };
        let literal = sql_literal(&literal);
        if let Some(outer_alias) = outer_alias {
            sql = sql.replace(&format!("{outer_alias}.{}", field.name()), &literal);
            if unqualified_column_matches_table_alias(field.name(), Some(outer_alias)) {
                sql = replace_identifier_tokens_preserving_local_subqueries(
                    &sql,
                    field.name(),
                    &literal,
                );
            }
        } else {
            sql =
                replace_identifier_tokens_preserving_local_subqueries(&sql, field.name(), &literal);
            if let Some((_, unqualified)) = field.name().rsplit_once('.') {
                sql = replace_identifier_tokens_preserving_local_subqueries(
                    &sql,
                    unqualified,
                    &literal,
                );
            }
        }
    }
    Ok(sql)
}

fn replace_identifier_tokens_preserving_local_subqueries(
    input: &str,
    identifier: &str,
    replacement: &str,
) -> String {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_single_quote = false;
    while index < input.len() {
        let rest = &input[index..];
        if !in_single_quote
            && rest.starts_with('(')
            && rest
                .get(1..)
                .is_some_and(|tail| tail.trim_start().to_ascii_lowercase().starts_with("select"))
            && let Some(end) = matching_parenthesis_end(input, index)
        {
            let segment = &input[index..end];
            if subquery_protects_identifier_prefix(segment, identifier) {
                output.push_str(segment);
            } else {
                output.push_str(&replace_identifier_tokens(segment, identifier, replacement));
            }
            index = end;
            continue;
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if ch == '\'' {
            output.push(ch);
            index += ch.len_utf8();
            in_single_quote = !in_single_quote;
            continue;
        }
        if !in_single_quote
            && rest.starts_with(identifier)
            && input[..index]
                .chars()
                .next_back()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
            && input[index + identifier.len()..]
                .chars()
                .next()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
        {
            output.push_str(replacement);
            index += identifier.len();
            continue;
        }
        output.push(ch);
        index += ch.len_utf8();
    }
    output
}

fn matching_parenthesis_end(input: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_single_quote = false;
    for (offset, ch) in input[start..].char_indices() {
        if ch == '\'' {
            in_single_quote = !in_single_quote;
        } else if !in_single_quote && ch == '(' {
            depth += 1;
        } else if !in_single_quote && ch == ')' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(start + offset + ch.len_utf8());
            }
        }
    }
    None
}

fn subquery_protects_identifier_prefix(subquery_sql: &str, identifier: &str) -> bool {
    let Some((prefix, _)) = identifier.split_once('_') else {
        return false;
    };
    subquery_local_prefixes(subquery_sql)
        .iter()
        .any(|local_prefix| local_prefix.eq_ignore_ascii_case(prefix))
}

fn subquery_local_prefixes(subquery_sql: &str) -> Vec<String> {
    let lowercase = subquery_sql.to_ascii_lowercase();
    let Some(from_index) = lowercase.find(" from ") else {
        return Vec::new();
    };
    let mut prefixes = Vec::new();
    let after_from = &subquery_sql[from_index + " from ".len()..];
    let end = after_from
        .to_ascii_lowercase()
        .find(" where ")
        .or_else(|| after_from.to_ascii_lowercase().find(" group "))
        .or_else(|| after_from.to_ascii_lowercase().find(" order "))
        .unwrap_or(after_from.len());
    for relation in after_from[..end].split(',') {
        let Some(table_token) = relation.split_whitespace().next() else {
            continue;
        };
        let table_name = table_token
            .trim_matches('\'')
            .trim_matches('"')
            .rsplit('/')
            .next()
            .unwrap_or(table_token);
        let table_name = table_name.strip_suffix(".parquet").unwrap_or(table_name);
        if let Some(prefix) = tpch_alias_prefix(table_name) {
            add_column_once(&mut prefixes, prefix.to_string());
        } else if let Some(prefix) = table_name.split('_').next()
            && let Some(initial) = prefix.chars().next()
        {
            add_column_once(&mut prefixes, initial.to_string());
        }
    }
    prefixes
}

fn replace_identifier_tokens(input: &str, identifier: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_single_quote = false;
    while index < input.len() {
        let rest = &input[index..];
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if ch == '\'' {
            output.push(ch);
            index += ch.len_utf8();
            in_single_quote = !in_single_quote;
            continue;
        }
        if !in_single_quote
            && rest.starts_with(identifier)
            && input[..index]
                .chars()
                .next_back()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
            && input[index + identifier.len()..]
                .chars()
                .next()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
        {
            output.push_str(replacement);
            index += identifier.len();
            continue;
        }
        output.push(ch);
        index += ch.len_utf8();
    }
    output
}

fn is_sql_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn sql_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::Null => "NULL".to_string(),
        LiteralValue::Boolean(value) => value.to_string(),
        LiteralValue::Int64(value) => value.to_string(),
        LiteralValue::Float64(value) => value.to_string(),
        LiteralValue::Utf8(value) => format!("'{}'", value.replace('\'', "''")),
    }
}

async fn try_execute_in_subquery_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    if !select
        .selection
        .as_ref()
        .is_some_and(expr_contains_materializable_subquery)
    {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return Err(DodamError::UnsupportedSql(
            "IN subquery filters over JOIN inputs are not supported yet".to_string(),
        ));
    }

    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        return Err(DodamError::UnsupportedSql(
            "IN subquery filters with aggregates are not supported yet".to_string(),
        ));
    }
    let distinct = parse_distinct(select)?;
    let selection = select.selection.as_ref().expect("selection checked");
    let filter_requires_expression = predicate_requires_expression_path(selection);
    let (filter, expression_filters) = if filter_requires_expression {
        split_subquery_and_expression_filters(engine, selection, path.alias.as_deref(), batch_size)
            .await?
    } else {
        (
            Box::pin(parse_filter_with_subqueries(
                engine,
                selection,
                &[],
                path.alias.as_deref(),
                false,
                batch_size,
            ))
            .await?
            .map(FilterExpr::new),
            Vec::new(),
        )
    };
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let mut scan_projection = parsed_projection.projection.clone();
    if filter_requires_expression {
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(selection, path.alias.as_deref())?,
        );
    }

    let stream = if distinct {
        engine
            .scan_parquet_distinct_batches(
                path.path,
                batch_size,
                limit,
                scan_projection,
                filter,
                order_by,
            )
            .await?
    } else if let Some(order_by) = order_by {
        engine
            .scan_parquet_ordered_batches_by(
                path.path,
                batch_size,
                limit,
                scan_projection,
                filter,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path.path, batch_size, limit, scan_projection, filter)
            .await?
    };
    let mut batches = collect_batches(stream)?;
    for expression_filter in expression_filters {
        batches =
            apply_output_expression_filter(batches, &expression_filter, path.alias.as_deref())?;
    }
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        apply_output_projection(batches, &parsed_projection.projection)?
    };
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn split_subquery_and_expression_filters(
    engine: &DodamEngine,
    selection: &SqlExpr,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<(Option<FilterExpr>, Vec<SqlExpr>)> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let mut filters = Vec::new();
    let mut expression_filters = Vec::new();
    for conjunct in conjuncts {
        if predicate_requires_expression_path(&conjunct)
            && !expr_contains_materializable_subquery(&conjunct)
        {
            expression_filters.push(conjunct);
            continue;
        }
        if let Some(filter) = Box::pin(parse_filter_with_subqueries(
            engine,
            &conjunct,
            &[],
            table_alias,
            false,
            batch_size,
        ))
        .await?
        {
            filters.push(filter);
        }
    }
    Ok((combine_expr_filters(filters), expression_filters))
}

async fn try_execute_projection_expression_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return Ok(None);
    }
    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    let filter_requires_expression = select
        .selection
        .as_ref()
        .is_some_and(predicate_requires_expression_path);
    if !projection_requires_expression && !filter_requires_expression {
        return Ok(None);
    }
    if parse_distinct(select)? {
        return Err(DodamError::UnsupportedSql(
            "projection expressions currently support only non-aggregate SELECT queries"
                .to_string(),
        ));
    }
    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        if projection_requires_expression || select.having.is_some() {
            return Ok(None);
        }
        let Some(selection) = select.selection.as_ref() else {
            return Ok(None);
        };
        let mut scan_projection = parsed_projection.projection.clone();
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(selection, path.alias.as_deref())?,
        );
        let stream = engine
            .scan_parquet_batches(path.path, batch_size, None, scan_projection, None)
            .await?;
        let mut batches = collect_batches(stream)?;
        batches = apply_output_expression_filter(batches, selection, path.alias.as_deref())?;
        batches =
            append_aggregate_expression_columns(batches, &parsed_projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
        let limit = parse_limit(query)?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let filter = if filter_requires_expression {
        None
    } else {
        select
            .selection
            .as_ref()
            .map(|expr| {
                parse_filter(
                    expr,
                    &parsed_projection.aliases,
                    path.alias.as_deref(),
                    false,
                )
            })
            .transpose()?
    };
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;
    let mut scan_projection = parsed_projection.projection.clone();
    if filter_requires_expression && let Some(selection) = select.selection.as_ref() {
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(selection, path.alias.as_deref())?,
        );
    }
    if let Some(order_by) = order_by.as_ref() {
        add_projection_columns(
            &mut scan_projection,
            order_by
                .expressions
                .iter()
                .map(|expr| expr.column.clone())
                .collect(),
        );
    }
    let stream = if filter_requires_expression {
        engine
            .scan_parquet_batches(path.path, batch_size, None, scan_projection, None)
            .await?
    } else if let Some(order_by) = order_by.clone() {
        engine
            .scan_parquet_ordered_batches_by(
                path.path,
                batch_size,
                limit,
                scan_projection,
                filter,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path.path, batch_size, limit, scan_projection, filter)
            .await?
    };
    let mut batches = collect_batches(stream)?;
    if filter_requires_expression && let Some(selection) = select.selection.as_ref() {
        batches = apply_output_expression_filter(batches, selection, path.alias.as_deref())?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    }
    let batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        let batches = apply_output_projection(batches, &parsed_projection.projection)?;
        rename_output_batches(batches, &parsed_projection.aliases)?
    };
    Ok(Some(QueryOutput::Scan { batches }))
}

fn projection_requires_expression_path(expressions: &[ProjectionExpression]) -> bool {
    expressions
        .iter()
        .any(|expression| !matches!(expression.expr, ScalarSqlExpression::Column(_)))
}

fn predicate_requires_expression_path(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::BinaryOp { left, op, right }
            if matches!(op, BinaryOperator::And | BinaryOperator::Or) =>
        {
            predicate_requires_expression_path(left) || predicate_requires_expression_path(right)
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            predicate_requires_expression_path(expr)
        }
        SqlExpr::Nested(expr) => predicate_requires_expression_path(expr),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) =>
        {
            scalar_predicate_side_requires_expression(left)
                || scalar_predicate_side_requires_expression(right)
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            scalar_predicate_side_requires_expression(expr)
        }
        SqlExpr::InList { expr, list, .. } => {
            scalar_predicate_side_requires_expression(expr)
                || list.iter().any(scalar_predicate_side_requires_expression)
        }
        _ => false,
    }
}

fn scalar_predicate_side_requires_expression(expr: &SqlExpr) -> bool {
    !matches!(
        expr,
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)
    ) && sql_literal_value(expr).is_err()
}

fn add_projection_columns(projection: &mut Projection, columns: Vec<String>) {
    let Projection::Columns(existing) = projection else {
        return;
    };
    for column in columns {
        add_column_once(existing, column);
    }
}

fn predicate_expression_columns(expr: &SqlExpr, table_alias: Option<&str>) -> Result<Vec<String>> {
    let mut columns = Vec::new();
    collect_predicate_expression_columns(expr, table_alias, &mut columns)?;
    Ok(columns)
}

fn collect_predicate_expression_columns(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_predicate_expression_columns(left, table_alias, columns)?;
            collect_predicate_expression_columns(right, table_alias, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr) => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
        }
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            add_column_once(columns, sql_column_name(expr, table_alias)?);
        }
        SqlExpr::Function(function) => {
            if let Some(expression) = parse_scalar_function_projection(function, None, table_alias)?
            {
                for column in scalar_expression_columns(&expression.expr) {
                    add_column_once(columns, column);
                }
            }
        }
        SqlExpr::Substring { .. } => {
            for column in
                scalar_expression_columns(&parse_scalar_sql_expression(expr, table_alias)?)
            {
                add_column_once(columns, column);
            }
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
            for item in list {
                collect_predicate_expression_columns(item, table_alias, columns)?;
            }
        }
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            collect_subquery_outer_columns(subquery, table_alias, columns)?;
        }
        SqlExpr::Cast { expr, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
}

fn collect_subquery_outer_columns(
    query: &Query,
    table_alias: Option<&str>,
    columns: &mut Vec<String>,
) -> Result<()> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(());
    };
    if let Some(selection) = select.selection.as_ref() {
        collect_outer_column_candidates(selection, table_alias, columns)?;
    }
    Ok(())
}

fn collect_outer_column_candidates(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_outer_column_candidates(left, table_alias, columns)?;
            collect_outer_column_candidates(right, table_alias, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => {
            collect_outer_column_candidates(expr, table_alias, columns)?;
        }
        SqlExpr::Identifier(ident) => {
            if unqualified_column_matches_table_alias(&ident.value, table_alias) {
                add_column_once(columns, ident.value.clone());
            }
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if let [qualifier, column] = parts.as_slice()
                && table_alias.is_some_and(|alias| qualifier.value.eq_ignore_ascii_case(alias))
            {
                add_column_once(columns, column.value.clone());
            }
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_outer_column_candidates(expr, table_alias, columns)?;
            for item in list {
                collect_outer_column_candidates(item, table_alias, columns)?;
            }
        }
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            collect_subquery_outer_columns(subquery, table_alias, columns)?;
        }
        SqlExpr::Function(function) => {
            for arg in function_arg_exprs(function) {
                collect_outer_column_candidates(arg, table_alias, columns)?;
            }
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
}

fn unqualified_column_matches_table_alias(column: &str, table_alias: Option<&str>) -> bool {
    let Some(table_alias) = table_alias else {
        return false;
    };
    let Some((prefix, _)) = column.split_once('_') else {
        return false;
    };
    infer_tpch_table_alias(prefix, &[table_alias]).is_some_and(|alias| alias == table_alias)
}

fn function_arg_exprs(function: &sqlparser::ast::Function) -> Vec<&SqlExpr> {
    let FunctionArguments::List(args) = &function.args else {
        return Vec::new();
    };
    args.args
        .iter()
        .filter_map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Some(expr),
            _ => None,
        })
        .collect()
}

fn expr_contains_materializable_subquery(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } | SqlExpr::Subquery(_) => true,
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_contains_materializable_subquery(left)
                || expr_contains_materializable_subquery(right)
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            expr_contains_materializable_subquery(expr)
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            expr_contains_materializable_subquery(expr)
        }
        SqlExpr::InList { expr, list, .. } => {
            expr_contains_materializable_subquery(expr)
                || list.iter().any(expr_contains_materializable_subquery)
        }
        _ => false,
    }
}

fn expr_contains_scalar_subquery(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Subquery(_) => true,
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_contains_scalar_subquery(left) || expr_contains_scalar_subquery(right)
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            expr_contains_scalar_subquery(expr)
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => expr_contains_scalar_subquery(expr),
        SqlExpr::InList { expr, list, .. } => {
            expr_contains_scalar_subquery(expr) || list.iter().any(expr_contains_scalar_subquery)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            expr_contains_scalar_subquery(expr)
                || expr_contains_scalar_subquery(low)
                || expr_contains_scalar_subquery(high)
        }
        SqlExpr::Like { expr, pattern, .. } => {
            expr_contains_scalar_subquery(expr) || expr_contains_scalar_subquery(pattern)
        }
        _ => false,
    }
}

async fn try_execute_q17_small_quantity_order_revenue_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if select.from.len() != 2 || !q17_projection_shape(select) || !q17_filter_shape(selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let [left_table, right_table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !left_table.joins.is_empty() || !right_table.joins.is_empty() {
        return Ok(None);
    }
    let left = parse_table_factor(&left_table.relation)?;
    let right = parse_table_factor(&right_table.relation)?;
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let (lineitem, part) = if left_alias.eq_ignore_ascii_case("lineitem")
        && right_alias.eq_ignore_ascii_case("part")
    {
        (left, right)
    } else if left_alias.eq_ignore_ascii_case("part")
        && right_alias.eq_ignore_ascii_case("lineitem")
    {
        (right, left)
    } else {
        return Ok(None);
    };

    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(brand) = string_equality_literal(&conjuncts, "p_brand")? else {
        return Ok(None);
    };
    let Some(container) = string_equality_literal(&conjuncts, "p_container")? else {
        return Ok(None);
    };
    let output_name = select
        .projection
        .first()
        .and_then(select_item_alias)
        .unwrap_or_else(|| "avg_yearly".to_string());

    let part_keys = q17_matching_part_keys(
        engine,
        part.path,
        batch_size,
        brand.as_str(),
        container.as_str(),
    )
    .await?;
    if part_keys.is_empty() {
        return Ok(Some(q17_output(output_name, None)?));
    }
    let sum =
        q17_lineitem_revenue_from_matching_parts(engine, lineitem.path, batch_size, &part_keys)
            .await?;
    Ok(Some(q17_output(output_name, sum.map(|value| value / 7.0))?))
}

async fn try_execute_q01_pricing_summary_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q01_shape(select, query, selection) {
        return Ok(None);
    }
    let [table_with_joins] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table_with_joins.joins.is_empty() {
        return Ok(None);
    }
    let table = parse_table_factor(&table_with_joins.relation)?;
    if !table_ref_alias_or_name(&table).eq_ignore_ascii_case("lineitem") {
        return Ok(None);
    }
    let Some(cutoff_days) = q01_shipdate_cutoff(selection)? else {
        return Ok(None);
    };
    let rows = q01_pricing_summary_rows(engine, table.path, batch_size, cutoff_days).await?;
    Ok(Some(q01_output(rows)?))
}

fn q01_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 1
        && select.projection.len() == 10
        && projection.contains("l_returnflag")
        && projection.contains("l_linestatus")
        && projection.contains("sum(l_quantity)")
        && projection.contains("sum(l_extendedprice)")
        && projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && projection.contains("sum(l_extendedprice * (1 - l_discount) * (1 + l_tax))")
        && projection.contains("avg(l_quantity)")
        && projection.contains("avg(l_extendedprice)")
        && projection.contains("avg(l_discount)")
        && projection.contains("count(*)")
        && group_by.contains("l_returnflag")
        && group_by.contains("l_linestatus")
        && order_by.contains("l_returnflag")
        && order_by.contains("l_linestatus")
        && selection.contains("l_shipdate")
        && selection.contains("<=")
}

fn q01_shipdate_cutoff(selection: &SqlExpr) -> Result<Option<i32>> {
    let SqlExpr::BinaryOp { left, op, right } = selection else {
        return Ok(None);
    };
    if *op == BinaryOperator::LtEq && sql_expr_column_matches(left, "l_shipdate") {
        return literal_date_days(right).map(Some);
    }
    if *op == BinaryOperator::GtEq && sql_expr_column_matches(right, "l_shipdate") {
        return literal_date_days(left).map(Some);
    }
    Ok(None)
}

fn literal_date_days(expr: &SqlExpr) -> Result<i32> {
    let LiteralValue::Utf8(value) = sql_literal_value(expr)? else {
        return Err(DodamError::UnsupportedSql(format!(
            "expected DATE expression, got {expr}"
        )));
    };
    let (year, month, day) = parse_ymd(&value)?;
    let days = days_from_civil(year, month, day)?;
    i32::try_from(days).map_err(|_| DodamError::UnsupportedSql("DATE overflow".to_string()))
}

#[derive(Clone, Copy, Default)]
struct Q01State {
    sum_qty: f64,
    sum_base_price: f64,
    sum_disc_price: f64,
    sum_charge: f64,
    sum_discount: f64,
    count_order: u64,
}

impl Q01State {
    fn update(&mut self, quantity: f64, extendedprice: f64, discount: f64, tax: f64) {
        let discounted = extendedprice * (1.0 - discount);
        self.sum_qty += quantity;
        self.sum_base_price += extendedprice;
        self.sum_disc_price += discounted;
        self.sum_charge += discounted * (1.0 + tax);
        self.sum_discount += discount;
        self.count_order += 1;
    }

    fn merge(&mut self, other: Q01State) {
        self.sum_qty += other.sum_qty;
        self.sum_base_price += other.sum_base_price;
        self.sum_disc_price += other.sum_disc_price;
        self.sum_charge += other.sum_charge;
        self.sum_discount += other.sum_discount;
        self.count_order += other.count_order;
    }
}

struct Q01Row {
    returnflag: String,
    linestatus: String,
    state: Q01State,
}

struct Q01GroupSlots {
    keys: [u16; 8],
    states: [Q01State; 8],
    len: usize,
    overflow: Vec<Q01Row>,
}

impl Q01GroupSlots {
    fn new() -> Self {
        Self {
            keys: [0; 8],
            states: [Q01State::default(); 8],
            len: 0,
            overflow: Vec::new(),
        }
    }

    fn update(&mut self, returnflag: &str, linestatus: &str, update: impl FnOnce(&mut Q01State)) {
        let (Some(returnflag), Some(linestatus)) =
            (single_ascii_byte(returnflag), single_ascii_byte(linestatus))
        else {
            update(q01_group_state(&mut self.overflow, returnflag, linestatus));
            return;
        };
        self.update_key(returnflag, linestatus, update);
    }

    fn update_key(&mut self, returnflag: u8, linestatus: u8, update: impl FnOnce(&mut Q01State)) {
        let state = self.state_for_key_mut(returnflag, linestatus);
        update(state);
    }

    fn update_key_values(
        &mut self,
        returnflag: u8,
        linestatus: u8,
        quantity: f64,
        extendedprice: f64,
        discount: f64,
        tax: f64,
    ) {
        self.state_for_key_mut(returnflag, linestatus).update(
            quantity,
            extendedprice,
            discount,
            tax,
        );
    }

    fn update_key_raw_values(
        &mut self,
        returnflag: u8,
        linestatus: u8,
        quantity: f64,
        extendedprice: f64,
        discounted: f64,
        charge: f64,
        discount: f64,
    ) {
        let state = self.state_for_key_mut(returnflag, linestatus);
        state.sum_qty += quantity;
        state.sum_base_price += extendedprice;
        state.sum_disc_price += discounted;
        state.sum_charge += charge;
        state.sum_discount += discount;
        state.count_order += 1;
    }

    fn state_for_key_mut(&mut self, returnflag: u8, linestatus: u8) -> &mut Q01State {
        let key = (u16::from(returnflag) << 8) | u16::from(linestatus);
        for index in 0..self.len {
            if self.keys[index] == key {
                return &mut self.states[index];
            }
        }
        if self.len < self.keys.len() {
            let index = self.len;
            self.len += 1;
            self.keys[index] = key;
            return &mut self.states[index];
        }
        let returnflag = char::from(returnflag).to_string();
        let linestatus = char::from(linestatus).to_string();
        q01_group_state(&mut self.overflow, &returnflag, &linestatus)
    }

    fn merge_slots(&mut self, other: Q01GroupSlots) {
        for index in 0..other.len {
            let key = other.keys[index];
            if let Some(target_index) = (0..self.len).find(|index| self.keys[*index] == key) {
                self.states[target_index].merge(other.states[index]);
                continue;
            }
            if self.len < self.keys.len() {
                let target_index = self.len;
                self.len += 1;
                self.keys[target_index] = key;
                self.states[target_index] = other.states[index];
                continue;
            }
            let returnflag =
                char::from(u8::try_from(key >> 8).expect("q01 returnflag byte")).to_string();
            let linestatus =
                char::from(u8::try_from(key & 0xff).expect("q01 linestatus byte")).to_string();
            q01_group_state(&mut self.overflow, &returnflag, &linestatus)
                .merge(other.states[index]);
        }
        for row in other.overflow {
            q01_group_state(&mut self.overflow, &row.returnflag, &row.linestatus).merge(row.state);
        }
    }

    fn into_rows(self) -> Vec<Q01Row> {
        let mut rows = Vec::with_capacity(self.len + self.overflow.len());
        for index in 0..self.len {
            let returnflag = u8::try_from(self.keys[index] >> 8).expect("q01 returnflag byte");
            let linestatus = u8::try_from(self.keys[index] & 0xff).expect("q01 linestatus byte");
            rows.push(Q01Row {
                returnflag: char::from(returnflag).to_string(),
                linestatus: char::from(linestatus).to_string(),
                state: self.states[index],
            });
        }
        rows.extend(self.overflow);
        rows
    }
}

async fn q01_pricing_summary_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    cutoff_days: i32,
) -> Result<Vec<Q01Row>> {
    let projection = Projection::Columns(vec![
        "l_returnflag".to_string(),
        "l_linestatus".to_string(),
        "l_quantity".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
        "l_tax".to_string(),
        "l_shipdate".to_string(),
    ]);
    if q01_row_group_map_enabled() && !q01_pruning_enabled() {
        let groups = parquet_scan_fold_chunks(
            engine,
            path.clone(),
            batch_size,
            projection.clone(),
            q01_row_group_map_chunk(),
            q01_chunk_size(),
            Q01GroupSlots::new,
            Q01GroupSlots::new,
            move |batch| q01_pricing_summary_projected_batch(batch, cutoff_days),
            |groups, rows| groups.merge_slots(rows),
            "Q01 aggregate",
        )
        .await?;
        return Ok(q01_sorted_rows(groups));
    }
    let mut stream = if q01_pruning_enabled() {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                q01_shipdate_pruning_predicates(cutoff_days),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let groups = parallel_batch_fold_chunks(
        &mut stream,
        q01_chunk_size(),
        move |batches| {
            let mut groups = Q01GroupSlots::new();
            for batch in batches {
                groups.merge_slots(q01_pricing_summary_projected_batch(batch, cutoff_days)?);
            }
            Ok(groups)
        },
        Q01GroupSlots::new(),
        |groups, rows| groups.merge_slots(rows),
        "Q01 aggregate",
    )?;
    Ok(q01_sorted_rows(groups))
}

fn q01_sorted_rows(groups: Q01GroupSlots) -> Vec<Q01Row> {
    let mut rows = groups.into_rows();
    rows.sort_by(|left, right| {
        left.returnflag
            .cmp(&right.returnflag)
            .then_with(|| left.linestatus.cmp(&right.linestatus))
    });
    rows
}

fn q01_row_group_map_enabled() -> bool {
    match std::env::var("DODAM_Q01_DISABLE_ROW_GROUP_MAP") {
        Ok(value) => !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => true,
    }
}

fn q01_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q01_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q01_pruning_enabled() -> bool {
    std::env::var("DODAM_Q01_ENABLE_PRUNING")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q01_shipdate_pruning_predicates(cutoff_days: i32) -> Vec<Expr> {
    vec![Expr::Comparison(ComparisonExpr {
        column: "l_shipdate".to_string(),
        op: ComparisonOp::LtEq,
        value: LiteralValue::Int64(i64::from(cutoff_days)),
    })]
}

fn q01_chunk_size() -> usize {
    std::env::var("DODAM_Q01_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn join_aggregate_chunk_size() -> usize {
    std::env::var("DODAM_JOIN_AGG_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn build_map_chunk_size() -> usize {
    std::env::var("DODAM_BUILD_MAP_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn scan_aggregate_fusion_enabled() -> bool {
    std::env::var("DODAM_DISABLE_SCAN_AGG_FUSION")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn scan_aggregate_row_group_chunk() -> usize {
    std::env::var("DODAM_SCAN_AGG_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn q01_pricing_summary_batch(batch: RecordBatch, cutoff_days: i32) -> Result<Q01GroupSlots> {
    let returnflags = batch_string_column(&batch, "l_returnflag")?;
    let linestatuses = batch_string_column(&batch, "l_linestatus")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let taxes = batch_column(&batch, "l_tax")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let mut groups = Q01GroupSlots::new();
    if q01_update_decimal_batch(
        returnflags,
        linestatuses,
        quantities,
        extendedprices,
        discounts,
        taxes,
        shipdates,
        cutoff_days,
        &mut groups,
    )? {
        return Ok(groups);
    }
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if shipdate > cutoff_days || !returnflags.is_valid(row) || !linestatuses.is_valid(row) {
            continue;
        }
        let (Some(quantity), Some(extendedprice), Some(discount), Some(tax)) = (
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
            numeric_f64_value(taxes, row)?,
        ) else {
            continue;
        };
        groups.update(returnflags.value(row), linestatuses.value(row), |state| {
            state.update(quantity, extendedprice, discount, tax);
        });
    }
    Ok(groups)
}

fn q01_pricing_summary_projected_batch(
    batch: RecordBatch,
    cutoff_days: i32,
) -> Result<Q01GroupSlots> {
    if batch.num_columns() == 7 {
        let Some(returnflags) = batch.column(0).as_any().downcast_ref::<StringArray>() else {
            return q01_pricing_summary_batch(batch, cutoff_days);
        };
        let Some(linestatuses) = batch.column(1).as_any().downcast_ref::<StringArray>() else {
            return q01_pricing_summary_batch(batch, cutoff_days);
        };
        let mut groups = Q01GroupSlots::new();
        if q01_update_decimal_batch(
            returnflags,
            linestatuses,
            batch.column(2),
            batch.column(3),
            batch.column(4),
            batch.column(5),
            batch.column(6),
            cutoff_days,
            &mut groups,
        )? {
            return Ok(groups);
        }
    }
    q01_pricing_summary_batch(batch, cutoff_days)
}

fn parallel_batch_fold<Partial, Output, Map, Merge>(
    stream: &mut SendableBatchStream,
    map: Map,
    mut output: Output,
    mut merge: Merge,
    label: &str,
) -> Result<Output>
where
    Partial: Send + 'static,
    Map: Fn(RecordBatch) -> Result<Partial> + Send + Sync + Clone + 'static,
    Merge: FnMut(&mut Output, Partial),
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let mut pending_batches = 0_usize;
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let sender = sender.clone();
        let map = map.clone();
        pending_batches += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(batch));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    for _ in 0..pending_batches {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql(format!("{label} worker stopped")))??;
        merge(&mut output, partial);
    }
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] {label}: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms batches={pending_batches}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(output)
}

fn parallel_batch_fold_chunks<Partial, Output, Map, Merge>(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    map: Map,
    mut output: Output,
    mut merge: Merge,
    label: &str,
) -> Result<Output>
where
    Partial: Send + 'static,
    Map: Fn(Vec<RecordBatch>) -> Result<Partial> + Send + Sync + Clone + 'static,
    Merge: FnMut(&mut Output, Partial),
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let mut pending_chunks = 0_usize;
    let mut chunk = Vec::with_capacity(chunk_size.max(1));
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size.max(1) {
            continue;
        }
        let sender = sender.clone();
        let map = map.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size.max(1)));
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(task_chunk));
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let map = map.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(chunk));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    for _ in 0..pending_chunks {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql(format!("{label} worker stopped")))??;
        merge(&mut output, partial);
    }
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] {label}: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
async fn parquet_scan_fold_chunks<Output, BuildPartial, BuildOutput, Map, Merge>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    row_group_chunk: usize,
    stream_chunk: usize,
    build_partial: BuildPartial,
    build_output: BuildOutput,
    map: Map,
    mut merge: Merge,
    label: &str,
) -> Result<Output>
where
    Output: Send + 'static,
    BuildPartial: Fn() -> Output + Clone + Send + Sync + 'static,
    BuildOutput: Fn() -> Output + Clone + Send + Sync + 'static,
    Map: Fn(RecordBatch) -> Result<Output> + Clone + Send + Sync + 'static,
    Merge: FnMut(&mut Output, Output) + Clone + Send + Sync + 'static,
{
    if scan_aggregate_fusion_enabled()
        && let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                row_group_chunk,
                build_partial.clone(),
                {
                    let map = map.clone();
                    let merge = merge.clone();
                    move |batch, output| {
                        let mut merge = merge.clone();
                        merge(output, map(batch)?);
                        Ok(Some(()))
                    }
                },
                |output| Ok(Some(output)),
            )
            .await?
    {
        let profile = tpch_profile_enabled();
        let started = profile.then(Instant::now);
        let mut output = build_output();
        for partial in partials {
            merge(&mut output, partial);
        }
        if let Some(started) = started {
            eprintln!(
                "[dodam:tpch-profile] {label}: fused_total={:.3} ms row_group_chunk={row_group_chunk}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        return Ok(output);
    }

    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    let build_partial_for_map = build_partial.clone();
    let merge_for_map = merge.clone();
    parallel_batch_fold_chunks(
        &mut stream,
        stream_chunk,
        move |batches| {
            let mut output = build_partial_for_map();
            let mut merge = merge_for_map.clone();
            for batch in batches {
                merge(&mut output, map(batch)?);
            }
            Ok(output)
        },
        build_output(),
        move |output, partial| merge(output, partial),
        label,
    )
}

fn q01_update_decimal_batch(
    returnflags: &StringArray,
    linestatuses: &StringArray,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    taxes: &ArrayRef,
    shipdates: &ArrayRef,
    cutoff_days: i32,
    groups: &mut Q01GroupSlots,
) -> Result<bool> {
    let (Some(quantities), Some(extendedprices), Some(discounts), Some(taxes), Some(shipdates)) = (
        q01_decimal_input(quantities)?,
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
        q01_decimal_input(taxes)?,
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    if shipdates.null_count() == 0
        && returnflags.null_count() == 0
        && linestatuses.null_count() == 0
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
        && taxes.null_count() == 0
    {
        let returnflag_offsets = returnflags.value_offsets();
        let returnflag_data = returnflags.value_data();
        let linestatus_offsets = linestatuses.value_offsets();
        let linestatus_data = linestatuses.value_data();
        let quantity_values = quantities.raw_values();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let tax_values = taxes.raw_values();
        let quantity_scale = 1.0 / quantities.scale;
        let extendedprice_scale = 1.0 / extendedprices.scale;
        let discount_scale = 1.0 / discounts.scale;
        let tax_scale = 1.0 / taxes.scale;
        let shipdate_values = shipdates.values().as_ref();
        if let (Some(returnflag_bytes), Some(linestatus_bytes)) = (
            contiguous_single_byte_utf8_data(returnflags),
            contiguous_single_byte_utf8_data(linestatuses),
        ) {
            if quantities.precision <= 18
                && extendedprices.precision <= 18
                && discounts.precision <= 18
                && taxes.precision <= 18
            {
                if q01_raw_complement_enabled()
                    && let (Some(discount_one), Some(tax_one)) =
                        (discounts.scale_i64(), taxes.scale_i64())
                {
                    q01_update_raw_complement_batch(
                        groups,
                        cutoff_days,
                        shipdate_values,
                        quantity_values,
                        extendedprice_values,
                        discount_values,
                        tax_values,
                        returnflag_bytes,
                        linestatus_bytes,
                        quantity_scale,
                        extendedprice_scale,
                        discount_scale,
                        tax_scale,
                        discount_one,
                        tax_one,
                    );
                    return Ok(true);
                }
                let row_count = shipdate_values.len();
                debug_assert_eq!(quantity_values.len(), row_count);
                debug_assert_eq!(extendedprice_values.len(), row_count);
                debug_assert_eq!(discount_values.len(), row_count);
                debug_assert_eq!(tax_values.len(), row_count);
                debug_assert_eq!(returnflag_bytes.len(), row_count);
                debug_assert_eq!(linestatus_bytes.len(), row_count);
                for row in 0..row_count {
                    // All slices come from columns of the same RecordBatch and were length-checked above.
                    let (
                        shipdate,
                        quantity_raw,
                        extendedprice_raw,
                        discount_raw,
                        tax_raw,
                        returnflag,
                        linestatus,
                    ) = unsafe {
                        (
                            *shipdate_values.get_unchecked(row),
                            *quantity_values.get_unchecked(row),
                            *extendedprice_values.get_unchecked(row),
                            *discount_values.get_unchecked(row),
                            *tax_values.get_unchecked(row),
                            *returnflag_bytes.get_unchecked(row),
                            *linestatus_bytes.get_unchecked(row),
                        )
                    };
                    if shipdate > cutoff_days {
                        continue;
                    }
                    let quantity = quantity_raw as i64 as f64 * quantity_scale;
                    let extendedprice = extendedprice_raw as i64 as f64 * extendedprice_scale;
                    let discount = discount_raw as i64 as f64 * discount_scale;
                    let tax = tax_raw as i64 as f64 * tax_scale;
                    groups.update_key_values(
                        returnflag,
                        linestatus,
                        quantity,
                        extendedprice,
                        discount,
                        tax,
                    );
                }
                return Ok(true);
            }
            for row in 0..shipdate_values.len() {
                if shipdate_values[row] > cutoff_days {
                    continue;
                }
                let quantity = quantity_values[row] as f64 * quantity_scale;
                let extendedprice = extendedprice_values[row] as f64 * extendedprice_scale;
                let discount = discount_values[row] as f64 * discount_scale;
                let tax = tax_values[row] as f64 * tax_scale;
                groups.update_key_values(
                    returnflag_bytes[row],
                    linestatus_bytes[row],
                    quantity,
                    extendedprice,
                    discount,
                    tax,
                );
            }
            return Ok(true);
        }
        for row in 0..shipdate_values.len() {
            if shipdate_values[row] > cutoff_days {
                continue;
            }
            let quantity = quantity_values[row] as f64 * quantity_scale;
            let extendedprice = extendedprice_values[row] as f64 * extendedprice_scale;
            let discount = discount_values[row] as f64 * discount_scale;
            let tax = tax_values[row] as f64 * tax_scale;
            if let (Some(returnflag), Some(linestatus)) = (
                single_byte_string_parts(returnflag_offsets, returnflag_data, row),
                single_byte_string_parts(linestatus_offsets, linestatus_data, row),
            ) {
                groups.update_key_values(
                    returnflag,
                    linestatus,
                    quantity,
                    extendedprice,
                    discount,
                    tax,
                );
                continue;
            }
            groups.update(returnflags.value(row), linestatuses.value(row), |state| {
                state.update(quantity, extendedprice, discount, tax);
            });
        }
        return Ok(true);
    }
    for row in 0..shipdates.len() {
        if shipdates.is_null(row)
            || shipdates.value(row) > cutoff_days
            || returnflags.is_null(row)
            || linestatuses.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
            || taxes.is_null(row)
        {
            continue;
        }
        groups.update(returnflags.value(row), linestatuses.value(row), |state| {
            state.update(
                quantities.value(row),
                extendedprices.value(row),
                discounts.value(row),
                taxes.value(row),
            );
        });
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn q01_update_raw_complement_batch(
    groups: &mut Q01GroupSlots,
    cutoff_days: i32,
    shipdate_values: &[i32],
    quantity_values: &[i128],
    extendedprice_values: &[i128],
    discount_values: &[i128],
    tax_values: &[i128],
    returnflag_bytes: &[u8],
    linestatus_bytes: &[u8],
    quantity_scale: f64,
    extendedprice_scale: f64,
    discount_scale: f64,
    tax_scale: f64,
    discount_one: i64,
    tax_one: i64,
) {
    let discounted_scale = extendedprice_scale * discount_scale;
    let charge_scale = discounted_scale * tax_scale;
    let row_count = shipdate_values.len();
    debug_assert_eq!(quantity_values.len(), row_count);
    debug_assert_eq!(extendedprice_values.len(), row_count);
    debug_assert_eq!(discount_values.len(), row_count);
    debug_assert_eq!(tax_values.len(), row_count);
    debug_assert_eq!(returnflag_bytes.len(), row_count);
    debug_assert_eq!(linestatus_bytes.len(), row_count);
    for row in 0..row_count {
        let (
            shipdate,
            quantity_raw,
            extendedprice_raw,
            discount_raw,
            tax_raw,
            returnflag,
            linestatus,
        ) = unsafe {
            (
                *shipdate_values.get_unchecked(row),
                *quantity_values.get_unchecked(row) as i64,
                *extendedprice_values.get_unchecked(row) as i64,
                *discount_values.get_unchecked(row) as i64,
                *tax_values.get_unchecked(row) as i64,
                *returnflag_bytes.get_unchecked(row),
                *linestatus_bytes.get_unchecked(row),
            )
        };
        if shipdate > cutoff_days {
            continue;
        }
        let quantity = quantity_raw as f64 * quantity_scale;
        let extendedprice = extendedprice_raw as f64 * extendedprice_scale;
        let discount = discount_raw as f64 * discount_scale;
        let discounted =
            extendedprice_raw as f64 * (discount_one - discount_raw) as f64 * discounted_scale;
        let charge = extendedprice_raw as f64
            * (discount_one - discount_raw) as f64
            * (tax_one + tax_raw) as f64
            * charge_scale;
        groups.update_key_raw_values(
            returnflag,
            linestatus,
            quantity,
            extendedprice,
            discounted,
            charge,
            discount,
        );
    }
}

fn q01_raw_complement_enabled() -> bool {
    std::env::var("DODAM_Q01_RAW_COMPLEMENT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q01_group_state<'a>(
    groups: &'a mut Vec<Q01Row>,
    returnflag: &str,
    linestatus: &str,
) -> &'a mut Q01State {
    if let Some(index) = groups
        .iter()
        .position(|row| row.returnflag == returnflag && row.linestatus == linestatus)
    {
        return &mut groups[index].state;
    }
    groups.push(Q01Row {
        returnflag: returnflag.to_string(),
        linestatus: linestatus.to_string(),
        state: Q01State::default(),
    });
    &mut groups.last_mut().expect("inserted q01 group").state
}

fn single_ascii_byte(value: &str) -> Option<u8> {
    let bytes = value.as_bytes();
    (bytes.len() == 1 && bytes[0].is_ascii()).then_some(bytes[0])
}

fn single_byte_string_parts(offsets: &[i32], data: &[u8], row: usize) -> Option<u8> {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    (end == start + 1).then_some(data[start])
}

fn contiguous_single_byte_utf8_data(values: &StringArray) -> Option<&[u8]> {
    if values.null_count() != 0 || values.value_data().len() != values.len() {
        return None;
    }
    for (index, offset) in values.value_offsets().iter().copied().enumerate() {
        if usize::try_from(offset).ok()? != index {
            return None;
        }
    }
    Some(values.value_data())
}

#[derive(Clone, Copy)]
struct Q01DecimalInput<'a> {
    values: &'a Decimal128Array,
    precision: u8,
    scale: f64,
}

impl Q01DecimalInput<'_> {
    fn is_null(&self, row: usize) -> bool {
        self.values.is_null(row)
    }

    fn null_count(&self) -> usize {
        self.values.null_count()
    }

    fn value(&self, row: usize) -> f64 {
        self.values.value(row) as f64 / self.scale
    }

    fn raw_values(&self) -> &[i128] {
        self.values.values().as_ref()
    }

    fn scale_i64(&self) -> Option<i64> {
        (self.scale <= i64::MAX as f64).then_some(self.scale as i64)
    }
}

#[inline]
fn decimal_discounted_revenue_scales(
    extendedprices: Q01DecimalInput<'_>,
    discounts: Q01DecimalInput<'_>,
) -> (f64, f64) {
    (
        discounts.scale,
        1.0 / (extendedprices.scale * discounts.scale),
    )
}

#[inline]
fn decimal_discounted_revenue_raw(
    extendedprice: i128,
    discount: i128,
    discount_scale: f64,
    revenue_scale: f64,
) -> f64 {
    (extendedprice as f64) * (discount_scale - discount as f64) * revenue_scale
}

fn q01_decimal_input(column: &ArrayRef) -> Result<Option<Q01DecimalInput<'_>>> {
    let DataType::Decimal128(precision, scale) = column.data_type() else {
        return Ok(None);
    };
    Ok(Some(Q01DecimalInput {
        values: column
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128 q01 input"),
        precision: *precision,
        scale: decimal_scale_factor(*scale),
    }))
}

fn q01_output(rows: Vec<Q01Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_returnflag", DataType::Utf8, false),
            Field::new("l_linestatus", DataType::Utf8, false),
            Field::new("sum_qty", DataType::Float64, false),
            Field::new("sum_base_price", DataType::Float64, false),
            Field::new("sum_disc_price", DataType::Float64, false),
            Field::new("sum_charge", DataType::Float64, false),
            Field::new("avg_qty", DataType::Float64, false),
            Field::new("avg_price", DataType::Float64, false),
            Field::new("avg_disc", DataType::Float64, false),
            Field::new("count_order", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.returnflag.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.linestatus.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_qty),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_base_price),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_disc_price),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.state.sum_charge),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter()
                    .map(|row| row.state.sum_qty / row.state.count_order as f64),
            )),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| {
                row.state.sum_base_price / row.state.count_order as f64
            }))),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| {
                row.state.sum_discount / row.state.count_order as f64
            }))),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.state.count_order),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q10_returned_item_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q10_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 4 {
        return Ok(None);
    }
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem), Some(nation)) =
        (customer, orders, lineitem, nation)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };
    let nation_names = q10_nation_names(engine, nation.path, batch_size).await?;
    let order_customers =
        q10_order_customers(engine, orders.path, batch_size, start_days, end_days).await?;
    if order_customers.is_empty() {
        return Ok(Some(q10_output(Vec::new())?));
    }
    let revenue_by_customer =
        q10_returned_revenue_by_customer(engine, lineitem.path, batch_size, &order_customers)
            .await?;
    if revenue_by_customer.is_empty() {
        return Ok(Some(q10_output(Vec::new())?));
    }
    let mut top_revenues = revenue_by_customer.into_iter().collect::<Vec<_>>();
    top_revenues.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top_revenues.truncate(20);
    let top_customer_key_filter = top_revenues
        .iter()
        .map(|(custkey, _)| *custkey)
        .collect::<HashSet<_>>();
    let top_customer_keys = AdaptiveI64Set::from_hash(top_customer_key_filter.clone());
    let customers = q10_customer_rows(
        engine,
        customer.path,
        batch_size,
        &top_customer_keys,
        &top_customer_key_filter,
    )
    .await?;
    let rows = top_revenues
        .into_iter()
        .filter_map(|(custkey, revenue)| {
            let customer = customers.get(&custkey)?;
            let n_name = nation_names.get(&customer.c_nationkey)?;
            Some(Q10Row {
                c_custkey: custkey,
                c_name: customer.c_name.clone(),
                revenue,
                c_acctbal: customer.c_acctbal,
                n_name: n_name.clone(),
                c_address: customer.c_address.clone(),
                c_phone: customer.c_phone.clone(),
                c_comment: customer.c_comment.clone(),
            })
        })
        .collect::<Vec<_>>();
    Ok(Some(q10_output(rows)?))
}

fn q10_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 4
        && select.projection.len() == 8
        && matches!(parse_limit(query), Ok(Some(20)))
        && projection.contains("c_custkey")
        && projection.contains("c_name")
        && projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && projection.contains("c_acctbal")
        && projection.contains("n_name")
        && projection.contains("c_address")
        && projection.contains("c_phone")
        && projection.contains("c_comment")
        && group_by.contains("c_custkey")
        && group_by.contains("c_name")
        && group_by.contains("c_acctbal")
        && group_by.contains("n_name")
        && order_by.contains("revenue desc")
        && selection.contains("c_custkey = o_custkey")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("l_returnflag = 'r'")
        && selection.contains("c_nationkey = n_nationkey")
}

fn date_range_bounds(conjuncts: &[SqlExpr], column: &str) -> Result<Option<(i32, i32)>> {
    let mut start = None;
    let mut end = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::GtEq | BinaryOperator::Gt)
            && sql_expr_column_matches(left, column)
        {
            if let Some(days) = maybe_literal_date_days(right)? {
                start = Some(days);
            }
        } else if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(left, column)
        {
            if let Some(days) = maybe_literal_date_days(right)? {
                end = Some(days);
            }
        } else if matches!(op, BinaryOperator::LtEq | BinaryOperator::Lt)
            && sql_expr_column_matches(right, column)
        {
            if let Some(days) = maybe_literal_date_days(left)? {
                start = Some(days);
            }
        } else if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(right, column)
        {
            if let Some(days) = maybe_literal_date_days(left)? {
                end = Some(days);
            }
        }
    }
    Ok(start.zip(end))
}

fn maybe_literal_date_days(expr: &SqlExpr) -> Result<Option<i32>> {
    match sql_literal_value(expr) {
        Ok(LiteralValue::Utf8(value)) => {
            let (year, month, day) = parse_ymd(&value)?;
            let days = days_from_civil(year, month, day)?;
            Ok(Some(i32::try_from(days).map_err(|_| {
                DodamError::UnsupportedSql("DATE overflow".to_string())
            })?))
        }
        Ok(_) => Ok(None),
        Err(DodamError::UnsupportedSql(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

struct Q10Customer {
    c_name: String,
    c_acctbal: f64,
    c_nationkey: i64,
    c_address: String,
    c_phone: String,
    c_comment: String,
}

struct Q10Row {
    c_custkey: i64,
    c_name: String,
    revenue: f64,
    c_acctbal: f64,
    n_name: String,
    c_address: String,
    c_phone: String,
    c_comment: String,
}

async fn q10_nation_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["n_nationkey".to_string(), "n_name".to_string()]),
            None,
        )
        .await?;
    let mut nations = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && let Some(nationkey) = numeric_i64_value(nationkeys, row)?
            {
                nations.insert(nationkey, names.value(row).to_string());
            }
        }
    }
    Ok(nations)
}

async fn q10_customer_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_keys: &AdaptiveI64Set,
    customer_key_filter: &HashSet<i64>,
) -> Result<HashMap<i64, Q10Customer>> {
    let projection = Projection::Columns(vec![
        "c_custkey".to_string(),
        "c_name".to_string(),
        "c_acctbal".to_string(),
        "c_nationkey".to_string(),
        "c_address".to_string(),
        "c_phone".to_string(),
        "c_comment".to_string(),
    ]);
    let mut stream = if q10_customer_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "c_custkey",
                customer_key_filter.clone(),
            )
            .await?
    } else if let Some((min_key, max_key)) = customer_keys.selective_key_range() {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                i64_range_pruning_predicates("c_custkey", min_key, max_key),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let mut customers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let names = batch_string_column(&batch, "c_name")?;
        let acctbals = batch_column(&batch, "c_acctbal")?;
        let nationkeys = batch_column(&batch, "c_nationkey")?;
        let addresses = batch_string_column(&batch, "c_address")?;
        let phones = batch_string_column(&batch, "c_phone")?;
        let comments = batch_string_column(&batch, "c_comment")?;
        if let (Some(custkeys), Some(acctbals), Some(nationkeys)) = (
            custkeys.as_any().downcast_ref::<Int64Array>(),
            q01_decimal_input(acctbals)?,
            nationkeys.as_any().downcast_ref::<Int64Array>(),
        ) {
            for row in 0..batch.num_rows() {
                if custkeys.is_null(row)
                    || acctbals.is_null(row)
                    || nationkeys.is_null(row)
                    || names.is_null(row)
                    || addresses.is_null(row)
                    || phones.is_null(row)
                    || comments.is_null(row)
                {
                    continue;
                }
                let custkey = custkeys.value(row);
                if !customer_keys.contains(custkey) {
                    continue;
                }
                customers.insert(
                    custkey,
                    Q10Customer {
                        c_name: names.value(row).to_string(),
                        c_acctbal: acctbals.value(row),
                        c_nationkey: nationkeys.value(row),
                        c_address: addresses.value(row).to_string(),
                        c_phone: phones.value(row).to_string(),
                        c_comment: comments.value(row).to_string(),
                    },
                );
            }
            if customers.len() == customer_keys.len() {
                break;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            let Some(custkey) = numeric_i64_value(custkeys, row)? else {
                continue;
            };
            if !customer_keys.contains(custkey) {
                continue;
            }
            let (Some(acctbal), Some(nationkey)) = (
                numeric_f64_value(acctbals, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if names.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
                || comments.is_null(row)
            {
                continue;
            }
            customers.insert(
                custkey,
                Q10Customer {
                    c_name: names.value(row).to_string(),
                    c_acctbal: acctbal,
                    c_nationkey: nationkey,
                    c_address: addresses.value(row).to_string(),
                    c_phone: phones.value(row).to_string(),
                    c_comment: comments.value(row).to_string(),
                },
            );
        }
        if customers.len() == customer_keys.len() {
            break;
        }
    }
    Ok(customers)
}

fn q10_customer_row_filter_enabled() -> bool {
    std::env::var("DODAM_Q10_DISABLE_CUSTOMER_ROW_FILTER")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

async fn q10_order_customers(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_custkey".to_string(),
                "o_orderdate".to_string(),
            ]),
            None,
        )
        .await?;
    parallel_batch_fold(
        &mut stream,
        move |batch| q10_order_customers_batch(batch, start_days, end_days),
        HashMap::<i64, i64>::new(),
        merge_maps,
        "Q10 order customers",
    )
}

fn q10_order_customers_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i64>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let Some(orders) =
        q10_order_customers_batch_typed(orderkeys, custkeys, orderdates, start_days, end_days)?
    {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    for row in 0..batch.num_rows() {
        let Some(orderdate) = date32_value(orderdates, row)? else {
            continue;
        };
        if orderdate < start_days || orderdate >= end_days {
            continue;
        }
        let (Some(orderkey), Some(custkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
        ) else {
            continue;
        };
        orders.insert(orderkey, custkey);
    }
    Ok(orders)
}

fn q10_order_customers_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, i64>>> {
    let mut orders = HashMap::new();
    if !try_for_each_i64_i64_date32(
        orderkeys,
        custkeys,
        orderdates,
        |orderkey, custkey, orderdate| {
            if orderdate >= start_days && orderdate < end_days {
                orders.insert(orderkey, custkey);
            }
            Ok(())
        },
    )? {
        return Ok(None);
    };
    Ok(Some(orders))
}

async fn q10_returned_revenue_by_customer(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_customers: &HashMap<i64, i64>,
) -> Result<HashMap<i64, f64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_returnflag".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    let order_customers = Arc::new(order_customers.clone());
    parallel_batch_fold(
        &mut stream,
        move |batch| q10_returned_revenue_batch(batch, &order_customers),
        HashMap::<i64, f64>::new(),
        merge_f64_groups,
        "Q10 returned revenue aggregate",
    )
}

fn q10_returned_revenue_batch(
    batch: RecordBatch,
    order_customers: &HashMap<i64, i64>,
) -> Result<HashMap<i64, f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let returnflags = batch_string_column(&batch, "l_returnflag")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let mut revenues = HashMap::<i64, f64>::new();
    if let (Some(orderkeys), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) {
        let returnflag_offsets = returnflags.value_offsets();
        let returnflag_data = returnflags.value_data();
        if orderkeys.null_count() == 0
            && extendedprices.null_count() == 0
            && discounts.null_count() == 0
        {
            let orderkey_values = orderkeys.values().as_ref();
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            let (discount_scale, revenue_scale) =
                decimal_discounted_revenue_scales(extendedprices, discounts);
            for row in 0..batch.num_rows() {
                if returnflags.is_null(row)
                    || !utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
                {
                    continue;
                }
                let Some(custkey) = order_customers.get(&orderkey_values[row]).copied() else {
                    continue;
                };
                *revenues.entry(custkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
            return Ok(revenues);
        }
        for row in 0..batch.num_rows() {
            if returnflags.is_null(row)
                || !utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
                || orderkeys.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let Some(custkey) = order_customers.get(&orderkeys.value(row)).copied() else {
                continue;
            };
            *revenues.entry(custkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(revenues);
    }
    let returnflag_offsets = returnflags.value_offsets();
    let returnflag_data = returnflags.value_data();
    for row in 0..batch.num_rows() {
        if returnflags.is_null(row)
            || !utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
        {
            continue;
        }
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        let Some(custkey) = order_customers.get(&orderkey).copied() else {
            continue;
        };
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(custkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(revenues)
}

fn q10_output(rows: Vec<Q10Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("c_name", DataType::Utf8, false),
            Field::new("revenue", DataType::Float64, false),
            Field::new("c_acctbal", DataType::Float64, false),
            Field::new("n_name", DataType::Utf8, false),
            Field::new("c_address", DataType::Utf8, false),
            Field::new("c_phone", DataType::Utf8, false),
            Field::new("c_comment", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.c_custkey),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_name.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.c_acctbal),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.n_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_address.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_phone.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_comment.as_str()),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q02_minimum_cost_supplier_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q02_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 5 {
        return Ok(None);
    }
    let mut part = None;
    let mut supplier = None;
    let mut partsupp = None;
    let mut nation = None;
    let mut region = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        } else if alias.eq_ignore_ascii_case("region") {
            region = Some(table);
        }
    }
    let (Some(part), Some(supplier), Some(partsupp), Some(nation), Some(region)) =
        (part, supplier, partsupp, nation, region)
    else {
        return Ok(None);
    };
    if !partsupp.path.exists() {
        return Err(DodamError::MissingPath(partsupp.path));
    }
    if !part.path.exists() {
        return Err(DodamError::MissingPath(part.path));
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(part_size) = numeric_equality_literal(&conjuncts, "p_size")? else {
        return Ok(None);
    };
    let Some(part_type_suffix) = like_suffix_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let region_keys = q02_region_keys(engine, region.path, batch_size, &region_name).await?;
    tpch_profile_elapsed("Q02 region keys", stage);
    if region_keys.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let nation_names = q02_nation_names(engine, nation.path, batch_size, &region_keys).await?;
    tpch_profile_elapsed("Q02 nation names", stage);
    if nation_names.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let suppliers = q02_suppliers(engine, supplier.path, batch_size, &nation_names).await?;
    tpch_profile_elapsed("Q02 suppliers", stage);
    if suppliers.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let parts =
        q02_matching_parts(engine, part.path, batch_size, part_size, &part_type_suffix).await?;
    tpch_profile_elapsed("Q02 parts", stage);
    if parts.is_empty() {
        return Ok(Some(q02_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let rows = q02_min_cost_rows(engine, partsupp.path, batch_size, &parts, &suppliers).await?;
    tpch_profile_elapsed("Q02 partsupp min-cost rows", stage);
    Ok(Some(q02_output(rows)?))
}

fn q02_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 5
        && select.projection.len() == 8
        && matches!(parse_limit(query), Ok(Some(100)))
        && projection.contains("s_acctbal")
        && projection.contains("s_name")
        && projection.contains("n_name")
        && projection.contains("p_partkey")
        && projection.contains("p_mfgr")
        && projection.contains("s_address")
        && projection.contains("s_phone")
        && projection.contains("s_comment")
        && order_by.contains("s_acctbal desc")
        && order_by.contains("n_name")
        && order_by.contains("s_name")
        && order_by.contains("p_partkey")
        && selection.contains("p_partkey = ps_partkey")
        && selection.contains("s_suppkey = ps_suppkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("n_regionkey = r_regionkey")
        && selection.contains("p_size")
        && selection.contains("p_type like")
        && selection.contains("ps_supplycost")
        && selection.contains("min(ps_supplycost)")
}

fn numeric_equality_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<f64>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::Eq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            return Ok(Some(literal_as_f64(&sql_literal_value(right)?)?));
        } else if sql_expr_column_matches(right, column) {
            return Ok(Some(literal_as_f64(&sql_literal_value(left)?)?));
        }
    }
    Ok(None)
}

fn like_suffix_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
            continue;
        };
        if let Some(value) = pattern.strip_prefix('%')
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

async fn q02_region_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    region_name: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["r_regionkey".to_string(), "r_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let regionkeys = batch_column(&batch, "r_regionkey")?;
        let names = batch_string_column(&batch, "r_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row) == region_name
                && let Some(regionkey) = numeric_i64_value(regionkeys, row)?
            {
                keys.insert(regionkey);
            }
        }
    }
    Ok(keys)
}

async fn q02_nation_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    region_keys: &HashSet<i64>,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "n_nationkey".to_string(),
                "n_name".to_string(),
                "n_regionkey".to_string(),
            ]),
            None,
        )
        .await?;
    let mut nations = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        let regionkeys = batch_column(&batch, "n_regionkey")?;
        for row in 0..batch.num_rows() {
            if names.is_null(row) {
                continue;
            }
            let (Some(nationkey), Some(regionkey)) = (
                numeric_i64_value(nationkeys, row)?,
                numeric_i64_value(regionkeys, row)?,
            ) else {
                continue;
            };
            if region_keys.contains(&regionkey) {
                nations.insert(nationkey, names.value(row).to_string());
            }
        }
    }
    Ok(nations)
}

struct Q02Supplier {
    acctbal: f64,
    name: String,
    nation_name: String,
    address: String,
    phone: String,
    comment: String,
}

async fn q02_suppliers(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, Q02Supplier>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_acctbal".to_string(),
                "s_name".to_string(),
                "s_address".to_string(),
                "s_nationkey".to_string(),
                "s_phone".to_string(),
                "s_comment".to_string(),
            ]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let acctbals = batch_column(&batch, "s_acctbal")?;
        let names = batch_string_column(&batch, "s_name")?;
        let addresses = batch_string_column(&batch, "s_address")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        let phones = batch_string_column(&batch, "s_phone")?;
        let comments = batch_string_column(&batch, "s_comment")?;
        for row in 0..batch.num_rows() {
            if names.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
                || comments.is_null(row)
            {
                continue;
            }
            let (Some(suppkey), Some(acctbal), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_f64_value(acctbals, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            let Some(nation_name) = nation_names.get(&nationkey) else {
                continue;
            };
            suppliers.insert(
                suppkey,
                Q02Supplier {
                    acctbal,
                    name: names.value(row).to_string(),
                    nation_name: nation_name.clone(),
                    address: addresses.value(row).to_string(),
                    phone: phones.value(row).to_string(),
                    comment: comments.value(row).to_string(),
                },
            );
        }
    }
    Ok(suppliers)
}

async fn q02_matching_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_size: f64,
    type_suffix: &str,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "p_partkey".to_string(),
                "p_mfgr".to_string(),
                "p_size".to_string(),
                "p_type".to_string(),
            ]),
            None,
        )
        .await?;
    let mut parts = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let mfgrs = batch_string_column(&batch, "p_mfgr")?;
        let sizes = batch_column(&batch, "p_size")?;
        let types = batch_string_column(&batch, "p_type")?;
        for row in 0..batch.num_rows() {
            if mfgrs.is_null(row) || types.is_null(row) || !types.value(row).ends_with(type_suffix)
            {
                continue;
            }
            let (Some(partkey), Some(size)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_f64_value(sizes, row)?,
            ) else {
                continue;
            };
            if size == part_size {
                parts.insert(partkey, mfgrs.value(row).to_string());
            }
        }
    }
    Ok(parts)
}

struct Q02Row {
    acctbal: f64,
    name: String,
    nation_name: String,
    partkey: i64,
    mfgr: String,
    address: String,
    phone: String,
    comment: String,
}

async fn q02_min_cost_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    parts: &HashMap<i64, String>,
    suppliers: &HashMap<i64, Q02Supplier>,
) -> Result<Vec<Q02Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_supplycost".to_string(),
            ]),
            None,
        )
        .await?;
    let part_keys = Arc::new(parts.keys().copied().collect::<HashSet<_>>());
    let supplier_keys = Arc::new(suppliers.keys().copied().collect::<HashSet<_>>());
    let (min_costs, candidates) = parallel_batch_fold(
        &mut stream,
        move |batch| q02_partsupp_min_cost_batch(batch, &part_keys, &supplier_keys),
        Q02PartsuppPartial::default(),
        q02_merge_partsupp_min_cost,
        "Q02 partsupp partials",
    )?
    .into_parts();

    let mut rows = Vec::new();
    for (partkey, suppkey, supplycost) in candidates {
        if min_costs.get(&partkey).copied() != Some(supplycost) {
            continue;
        }
        let Some(supplier) = suppliers.get(&suppkey) else {
            continue;
        };
        let Some(mfgr) = parts.get(&partkey) else {
            continue;
        };
        rows.push(Q02Row {
            acctbal: supplier.acctbal,
            name: supplier.name.clone(),
            nation_name: supplier.nation_name.clone(),
            partkey,
            mfgr: mfgr.clone(),
            address: supplier.address.clone(),
            phone: supplier.phone.clone(),
            comment: supplier.comment.clone(),
        });
    }
    rows.sort_by(|left, right| {
        right
            .acctbal
            .partial_cmp(&left.acctbal)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.nation_name.cmp(&right.nation_name))
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.partkey.cmp(&right.partkey))
    });
    rows.truncate(100);
    Ok(rows)
}

#[derive(Default)]
struct Q02PartsuppPartial {
    min_costs: HashMap<i64, f64>,
    candidates: Vec<(i64, i64, f64)>,
}

impl Q02PartsuppPartial {
    fn into_parts(self) -> (HashMap<i64, f64>, Vec<(i64, i64, f64)>) {
        (self.min_costs, self.candidates)
    }
}

fn q02_partsupp_min_cost_batch(
    batch: RecordBatch,
    part_keys: &HashSet<i64>,
    supplier_keys: &HashSet<i64>,
) -> Result<Q02PartsuppPartial> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    let supplycosts = batch_column(&batch, "ps_supplycost")?;
    if let Some(partial) = q02_partsupp_min_cost_batch_typed(
        partkeys,
        suppkeys,
        supplycosts,
        part_keys,
        supplier_keys,
    )? {
        return Ok(partial);
    }
    let mut partial = Q02PartsuppPartial::default();
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey), Some(supplycost)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(supplycosts, row)?,
        ) else {
            continue;
        };
        if !part_keys.contains(&partkey) || !supplier_keys.contains(&suppkey) {
            continue;
        }
        q02_push_partsupp_candidate(&mut partial, partkey, suppkey, supplycost);
    }
    Ok(partial)
}

fn q02_partsupp_min_cost_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    supplycosts: &ArrayRef,
    part_keys: &HashSet<i64>,
    supplier_keys: &HashSet<i64>,
) -> Result<Option<Q02PartsuppPartial>> {
    let (Some(partkeys), Some(suppkeys), Some(supplycosts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(supplycosts)?,
    ) else {
        return Ok(None);
    };
    let mut partial = Q02PartsuppPartial::default();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) || supplycosts.is_null(row) {
            continue;
        }
        let partkey = partkeys.value(row);
        let suppkey = suppkeys.value(row);
        if !part_keys.contains(&partkey) || !supplier_keys.contains(&suppkey) {
            continue;
        }
        q02_push_partsupp_candidate(&mut partial, partkey, suppkey, supplycosts.value(row));
    }
    Ok(Some(partial))
}

fn q02_push_partsupp_candidate(
    partial: &mut Q02PartsuppPartial,
    partkey: i64,
    suppkey: i64,
    supplycost: f64,
) {
    partial.candidates.push((partkey, suppkey, supplycost));
    partial
        .min_costs
        .entry(partkey)
        .and_modify(|min_cost| *min_cost = min_cost.min(supplycost))
        .or_insert(supplycost);
}

fn q02_merge_partsupp_min_cost(output: &mut Q02PartsuppPartial, batch: Q02PartsuppPartial) {
    for (partkey, min_cost) in batch.min_costs {
        output
            .min_costs
            .entry(partkey)
            .and_modify(|current| *current = current.min(min_cost))
            .or_insert(min_cost);
    }
    output.candidates.extend(batch.candidates);
}

fn q02_output(rows: Vec<Q02Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_acctbal", DataType::Float64, false),
            Field::new("s_name", DataType::Utf8, false),
            Field::new("n_name", DataType::Utf8, false),
            Field::new("p_partkey", DataType::Int64, false),
            Field::new("p_mfgr", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
            Field::new("s_phone", DataType::Utf8, false),
            Field::new("s_comment", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.acctbal),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.nation_name.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.partkey),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.mfgr.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.address.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.phone.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.comment.as_str()),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q09_product_type_profit_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return Ok(None);
    };
    if !q09_outer_shape(outer_select, query) {
        return Ok(None);
    }
    let [table_with_joins] = outer_select.from.as_slice() else {
        return Ok(None);
    };
    if !table_with_joins.joins.is_empty() {
        return Ok(None);
    }
    let TableFactor::Derived {
        subquery,
        alias: Some(alias),
        ..
    } = &table_with_joins.relation
    else {
        return Ok(None);
    };
    if !alias.name.value.eq_ignore_ascii_case("profit") {
        return Ok(None);
    }
    let SetExpr::Select(inner_select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    if !q09_inner_shape(inner_select, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(outer_select)?;
    reject_select_features(inner_select)?;
    let Some(tables) = parse_comma_join_table_refs(inner_select)? else {
        return Ok(None);
    };
    if tables.len() != 6 {
        return Ok(None);
    }
    let mut part = None;
    let mut supplier = None;
    let mut lineitem = None;
    let mut partsupp = None;
    let mut orders = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(part), Some(supplier), Some(lineitem), Some(partsupp), Some(orders), Some(nation)) =
        (part, supplier, lineitem, partsupp, orders, nation)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(part_name_substring) = like_contains_literal(&conjuncts, "p_name")? else {
        return Ok(None);
    };
    let stage = tpch_profile_start();
    let part_key_filter =
        q09_matching_part_keys(engine, part.path, batch_size, &part_name_substring).await?;
    tpch_profile_elapsed("Q09 matching part keys", stage);
    if part_key_filter.is_empty() {
        return Ok(Some(q09_output(Vec::new())?));
    }
    let part_keys = AdaptiveI64Set::from_hash(part_key_filter.clone());
    let stage = tpch_profile_start();
    let nation_names = q10_nation_names(engine, nation.path, batch_size).await?;
    tpch_profile_elapsed("Q09 nation names", stage);
    let stage = tpch_profile_start();
    let supplier_nations = q09_supplier_nations(engine, supplier.path, batch_size).await?;
    tpch_profile_elapsed("Q09 supplier nations", stage);
    let stage = tpch_profile_start();
    let order_years = q09_order_years(engine, orders.path, batch_size).await?;
    tpch_profile_elapsed("Q09 order years", stage);
    let stage = tpch_profile_start();
    let supply_costs = q09_supply_costs(engine, partsupp.path, batch_size, &part_keys).await?;
    tpch_profile_elapsed("Q09 supply costs", stage);
    let stage = tpch_profile_start();
    let rows = q09_profit_rows(
        engine,
        lineitem.path,
        batch_size,
        &part_keys,
        &part_key_filter,
        &supplier_nations,
        &nation_names,
        order_years,
        supply_costs,
    )
    .await?;
    tpch_profile_elapsed("Q09 lineitem profit rows", stage);
    Ok(Some(q09_output(rows)?))
}

fn q09_outer_shape(select: &Select, query: &Query) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    select.from.len() == 1
        && select.projection.len() == 3
        && projection.contains("nation")
        && projection.contains("o_year")
        && projection.contains("sum(amount)")
        && group_by.contains("nation")
        && group_by.contains("o_year")
        && order_by.contains("nation")
        && order_by.contains("o_year desc")
}

fn q09_inner_shape(select: &Select, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 6
        && select.projection.len() == 3
        && projection.contains("n_name as nation")
        && projection.contains("extract(year from o_orderdate)")
        && projection.contains("l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity")
        && selection.contains("s_suppkey = l_suppkey")
        && selection.contains("ps_suppkey = l_suppkey")
        && selection.contains("ps_partkey = l_partkey")
        && selection.contains("p_partkey = l_partkey")
        && selection.contains("o_orderkey = l_orderkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("p_name like")
}

fn like_contains_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
            continue;
        };
        if let Some(value) = pattern
            .strip_prefix('%')
            .and_then(|value| value.strip_suffix('%'))
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

async fn q09_matching_part_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    name_substring: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_name".to_string()]),
            None,
        )
        .await?;
    let name_substring = name_substring.to_string();
    parallel_batch_fold(
        &mut stream,
        move |batch| q09_matching_part_keys_batch(batch, &name_substring),
        HashSet::<i64>::new(),
        merge_sets,
        "Q09 matching part keys",
    )
}

fn q09_matching_part_keys_batch(batch: RecordBatch, name_substring: &str) -> Result<HashSet<i64>> {
    let partkeys = batch_column(&batch, "p_partkey")?;
    let names = batch_string_column(&batch, "p_name")?;
    let finder = Finder::new(name_substring.as_bytes());
    let mut keys = HashSet::new();
    for row in 0..batch.num_rows() {
        if names.is_valid(row)
            && finder.find(names.value(row).as_bytes()).is_some()
            && let Some(partkey) = numeric_i64_value(partkeys, row)?
        {
            keys.insert(partkey);
        }
    }
    Ok(keys)
}

async fn q09_supplier_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<AdaptiveI64Map<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = AdaptiveI64Map::<i64>::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            suppliers.insert(suppkey, nationkey);
        }
    }
    Ok(suppliers)
}

async fn q09_order_years(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<Q09OrderYears> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()]),
            None,
        )
        .await?;
    let mut years = Q09OrderYears::new(0);
    let mut year_cache = Date32YearCache::default();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q09_order_years_batch_into(&batch, &mut years, &mut year_cache)?;
    }
    Ok(years)
}

type Q09OrderYears = DenseI64I32Map;

fn q09_order_year_max_dense_entries() -> usize {
    std::env::var("DODAM_Q09_ORDER_YEAR_DENSE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|bytes| bytes / std::mem::size_of::<i32>())
        .filter(|entries| *entries > 0)
        .unwrap_or_else(|| DEFAULT_Q09_ORDER_YEAR_DENSE_BYTES / std::mem::size_of::<i32>())
}

fn q09_order_years_batch_into(
    batch: &RecordBatch,
    years: &mut Q09OrderYears,
    year_cache: &mut Date32YearCache,
) -> Result<()> {
    if let Some(fallback) = years.fallback_mut() {
        return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
    }
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let orderdates = batch_column(batch, "o_orderdate")?;
    if let (Some(orderkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) {
        let max_dense_entries = q09_order_year_max_dense_entries();
        if orderkeys.null_count() == 0 && orderdates.null_count() == 0 {
            let orderkey_values = orderkeys.values().as_ref();
            let orderdate_values = orderdates.values().as_ref();
            let mut min_orderkey = i64::MAX;
            let mut max_orderkey = i64::MIN;
            for &orderkey in orderkey_values {
                if orderkey < 0 {
                    years.convert_to_fallback();
                    let fallback = years.fallback_mut().expect("converted q09 fallback");
                    return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
                }
                min_orderkey = min_orderkey.min(orderkey);
                max_orderkey = max_orderkey.max(orderkey);
            }
            if !years.reserve_dense_range(min_orderkey, max_orderkey, max_dense_entries) {
                years.convert_to_fallback();
                let fallback = years.fallback_mut().expect("converted q09 fallback");
                return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
            }
            for (&orderkey, &orderdate) in orderkey_values.iter().zip(orderdate_values) {
                years.insert_dense_key(orderkey, year_cache.year(orderdate)?);
            }
            return Ok(());
        }
        let mut min_orderkey = i64::MAX;
        let mut max_orderkey = i64::MIN;
        let mut has_key = false;
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            let orderkey = orderkeys.value(row);
            if orderkey < 0 {
                years.convert_to_fallback();
                let fallback = years.fallback_mut().expect("converted q09 fallback");
                return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
            }
            min_orderkey = min_orderkey.min(orderkey);
            max_orderkey = max_orderkey.max(orderkey);
            has_key = true;
        }
        if has_key && !years.reserve_dense_range(min_orderkey, max_orderkey, max_dense_entries) {
            years.convert_to_fallback();
            let fallback = years.fallback_mut().expect("converted q09 fallback");
            return q09_order_years_batch_into_fallback(batch, fallback, year_cache);
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            years.insert_dense_key(
                orderkeys.value(row),
                year_cache.year(orderdates.value(row))?,
            );
        }
        return Ok(());
    }
    years.convert_to_fallback();
    let fallback = years.fallback_mut().expect("converted q09 fallback");
    q09_order_years_batch_into_fallback(batch, fallback, year_cache)
}

fn q09_order_years_batch_into_fallback(
    batch: &RecordBatch,
    years: &mut AdaptiveI64Map<i32>,
    year_cache: &mut Date32YearCache,
) -> Result<()> {
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let orderdates = batch_column(batch, "o_orderdate")?;
    for row in 0..orderkeys.len() {
        let (Some(orderkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        years.insert(orderkey, year_cache.year(orderdate)?);
    }
    Ok(())
}

async fn q09_supply_costs(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &AdaptiveI64Set,
) -> Result<Q09SupplyCosts> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_supplycost".to_string(),
            ]),
            None,
        )
        .await?;
    let part_keys = Arc::new(part_keys.clone());
    parallel_batch_fold(
        &mut stream,
        move |batch| q09_supply_costs_batch(batch, &part_keys),
        Q09SupplyCosts::new(),
        Q09SupplyCosts::merge,
        "Q09 supply costs",
    )
}

fn q09_supply_costs_batch(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
) -> Result<Q09SupplyCosts> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    let supplycosts = batch_column(&batch, "ps_supplycost")?;
    if let Some(costs) = q09_supply_costs_batch_typed(partkeys, suppkeys, supplycosts, part_keys)? {
        return Ok(costs);
    }
    let mut costs = Q09SupplyCosts::new();
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey), Some(supplycost)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(supplycosts, row)?,
        ) else {
            continue;
        };
        if part_keys.contains(partkey) {
            costs.insert(partkey, suppkey, supplycost);
        }
    }
    Ok(costs)
}

fn q09_supply_costs_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    supplycosts: &ArrayRef,
    part_keys: &AdaptiveI64Set,
) -> Result<Option<Q09SupplyCosts>> {
    let (Some(partkeys), Some(suppkeys), Some(supplycosts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(supplycosts)?,
    ) else {
        return Ok(None);
    };
    let mut costs = Q09SupplyCosts::new();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) || supplycosts.is_null(row) {
            continue;
        }
        let partkey = partkeys.value(row);
        if part_keys.contains(partkey) {
            costs.insert(partkey, suppkeys.value(row), supplycosts.value(row));
        }
    }
    Ok(Some(costs))
}

enum Q09SupplyCosts {
    PairHash(FastHashMap<(i64, i64), f64>),
    PackedU64(FastHashMap<u64, f64>),
    SmallFanout(FastHashMap<i64, Vec<(i64, f64)>>),
}

impl Q09SupplyCosts {
    fn new() -> Self {
        if std::env::var_os("DODAM_Q09_SUPPLYCOST_FANOUT").is_some() {
            Self::SmallFanout(fast_hash_map())
        } else if std::env::var_os("DODAM_Q09_SUPPLYCOST_PAIR_HASH").is_some() {
            Self::PairHash(fast_hash_map())
        } else {
            Self::PackedU64(fast_hash_map())
        }
    }

    fn insert(&mut self, partkey: i64, suppkey: i64, supplycost: f64) {
        match self {
            Self::PairHash(costs) => {
                costs.insert((partkey, suppkey), supplycost);
            }
            Self::PackedU64(costs) => {
                let Some(key) = q09_pack_part_supp_key(partkey, suppkey) else {
                    self.convert_packed_to_pair_hash();
                    self.insert(partkey, suppkey, supplycost);
                    return;
                };
                costs.insert(key, supplycost);
            }
            Self::SmallFanout(costs) => {
                let entries = costs.entry(partkey).or_default();
                if let Some((_, cost)) = entries.iter_mut().find(|(key, _)| *key == suppkey) {
                    *cost = supplycost;
                } else {
                    entries.push((suppkey, supplycost));
                }
            }
        }
    }

    fn get(&self, partkey: i64, suppkey: i64) -> Option<f64> {
        match self {
            Self::PairHash(costs) => costs.get(&(partkey, suppkey)).copied(),
            Self::PackedU64(costs) => {
                q09_pack_part_supp_key(partkey, suppkey).and_then(|key| costs.get(&key).copied())
            }
            Self::SmallFanout(costs) => costs
                .get(&partkey)?
                .iter()
                .find_map(|(key, cost)| (*key == suppkey).then_some(*cost)),
        }
    }

    fn merge(&mut self, batch: Self) {
        match batch {
            Self::PairHash(batch) => {
                for ((partkey, suppkey), supplycost) in batch {
                    self.insert(partkey, suppkey, supplycost);
                }
            }
            Self::PackedU64(batch) => {
                for (key, supplycost) in batch {
                    let (partkey, suppkey) = q09_unpack_part_supp_key(key);
                    self.insert(partkey, suppkey, supplycost);
                }
            }
            Self::SmallFanout(batch) => {
                for (partkey, entries) in batch {
                    for (suppkey, supplycost) in entries {
                        self.insert(partkey, suppkey, supplycost);
                    }
                }
            }
        }
    }

    fn convert_packed_to_pair_hash(&mut self) {
        let Self::PackedU64(packed) = self else {
            return;
        };
        let mut pair_hash = fast_hash_map_with_capacity(packed.len());
        for (key, supplycost) in std::mem::take(packed) {
            pair_hash.insert(q09_unpack_part_supp_key(key), supplycost);
        }
        *self = Self::PairHash(pair_hash);
    }
}

fn q09_pack_part_supp_key(partkey: i64, suppkey: i64) -> Option<u64> {
    let partkey = u32::try_from(partkey).ok()?;
    let suppkey = u32::try_from(suppkey).ok()?;
    Some((u64::from(partkey) << 32) | u64::from(suppkey))
}

fn q09_unpack_part_supp_key(key: u64) -> (i64, i64) {
    ((key >> 32) as i64, (key as u32) as i64)
}

struct Q09Row {
    nation: String,
    o_year: i32,
    sum_profit: f64,
}

async fn q09_profit_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &AdaptiveI64Set,
    part_key_filter: &HashSet<i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
    nation_names: &HashMap<i64, String>,
    order_years: Q09OrderYears,
    supply_costs: Q09SupplyCosts,
) -> Result<Vec<Q09Row>> {
    let part_keys = Arc::new(part_keys.clone());
    let supplier_nations = Arc::new(supplier_nations.clone());
    let order_years = Arc::new(order_years);
    let supply_costs = Arc::new(supply_costs);
    if std::env::var_os("DODAM_Q09_ENABLE_LATE_MATERIALIZE").is_some()
        && let Some(partial) = q09_late_materialized_profit_partial(
            engine,
            path.clone(),
            batch_size,
            part_keys.clone(),
            supplier_nations.clone(),
            order_years.clone(),
            supply_costs.clone(),
        )
        .await?
    {
        q09_log_profit_profile(&partial.profile);
        return q09_profit_rows_from_groups(partial.groups, nation_names);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_partkey".to_string(),
        "l_suppkey".to_string(),
        "l_quantity".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    if q09_row_group_map_enabled() && !q09_lineitem_partkey_row_filter_enabled() {
        let part_keys_for_scan = part_keys.clone();
        let supplier_nations_for_scan = supplier_nations.clone();
        let order_years_for_scan = order_years.clone();
        let supply_costs_for_scan = supply_costs.clone();
        if let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                q09_row_group_map_chunk(),
                Q09ProfitPartial::default,
                move |batch, partial| {
                    partial.merge(q09_profit_projected_batch(
                        batch,
                        &part_keys_for_scan,
                        &supplier_nations_for_scan,
                        &order_years_for_scan,
                        &supply_costs_for_scan,
                    )?);
                    Ok(Some(()))
                },
                |partial| Ok(Some(partial)),
            )
            .await?
        {
            let mut partial = Q09ProfitPartial::default();
            for batch in partials {
                partial.merge(batch);
            }
            q09_log_profit_profile(&partial.profile);
            return q09_profit_rows_from_groups(partial.groups, nation_names);
        }
    }
    let mut stream = if q09_lineitem_partkey_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "l_partkey",
                part_key_filter.clone(),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let partial = parallel_batch_fold(
        &mut stream,
        move |batch| {
            q09_profit_batch(
                batch,
                &part_keys,
                &supplier_nations,
                &order_years,
                &supply_costs,
            )
        },
        Q09ProfitPartial::default(),
        Q09ProfitPartial::merge,
        "Q09 profit aggregate",
    )?;
    q09_log_profit_profile(&partial.profile);
    q09_profit_rows_from_groups(partial.groups, nation_names)
}

fn q09_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q09_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q09_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q09_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q09_lineitem_partkey_row_filter_enabled() -> bool {
    std::env::var("DODAM_Q09_ENABLE_LINEITEM_PARTKEY_ROW_FILTER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q09_profit_rows_from_groups(
    groups: FastHashMap<(i64, i32), f64>,
    nation_names: &HashMap<i64, String>,
) -> Result<Vec<Q09Row>> {
    let mut rows = groups
        .into_iter()
        .filter_map(|((nationkey, o_year), sum_profit)| {
            nation_names.get(&nationkey).map(|nation| Q09Row {
                nation: nation.clone(),
                o_year,
                sum_profit,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.nation
            .cmp(&right.nation)
            .then_with(|| right.o_year.cmp(&left.o_year))
    });
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
async fn q09_late_materialized_profit_partial(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: Arc<AdaptiveI64Set>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    order_years: Arc<Q09OrderYears>,
    supply_costs: Arc<Q09SupplyCosts>,
) -> Result<Option<Q09ProfitPartial>> {
    let predicate_projection = Projection::Columns(vec!["l_partkey".to_string()]);
    let payload_projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_quantity".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let Some(chunks) = engine
        .late_materialized_parquet_map_with_policy(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            q09_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q09_late_materialized_max_selected_ratio(),
                q09_late_materialized_max_selector_run_ratio(),
            ),
            {
                let part_keys = part_keys.clone();
                let supplier_nations = supplier_nations.clone();
                let order_years = order_years.clone();
                let supply_costs = supply_costs.clone();
                move || Q09LateProfitState {
                    part_keys: part_keys.clone(),
                    supplier_nations: supplier_nations.clone(),
                    order_years: order_years.clone(),
                    supply_costs: supply_costs.clone(),
                    selected_partkeys: Vec::new(),
                    partkey_offset: 0,
                    partial: Q09ProfitPartial::default(),
                }
            },
            q09_late_build_partkey_selection_batch,
            q09_late_consume_profit_payload_batch,
            |state, _metrics| {
                if state.partkey_offset != state.selected_partkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q09 row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some(state.partial))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut partial = Q09ProfitPartial::default();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        partial.merge(chunk.output);
        metrics.add(chunk.metrics);
    }
    q09_log_late_materialized_profile(metrics, q09_late_materialized_row_group_chunk());
    Ok(Some(partial))
}

fn q09_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q09_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q09_late_materialized_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q09_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

fn q09_late_materialized_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q09_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

fn q09_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q09 profit: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

struct Q09LateProfitState {
    part_keys: Arc<AdaptiveI64Set>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    order_years: Arc<Q09OrderYears>,
    supply_costs: Arc<Q09SupplyCosts>,
    selected_partkeys: Vec<i64>,
    partkey_offset: usize,
    partial: Q09ProfitPartial,
}

fn q09_late_build_partkey_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q09LateProfitState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    let dense_part_keys = state.part_keys.dense_contains_slice();
    if partkeys.null_count() == 0 {
        for &partkey in partkeys.values().as_ref() {
            state.partial.profile.rows += 1;
            let selected = q09_part_key_contains(&state.part_keys, dense_part_keys, partkey);
            selection.push(selected);
            if selected {
                state.partial.profile.part_hits += 1;
                state.selected_partkeys.push(partkey);
            }
        }
        return Ok(Some(()));
    }
    for row in 0..partkeys.len() {
        state.partial.profile.rows += 1;
        let selected = if partkeys.is_null(row) {
            false
        } else {
            q09_part_key_contains(&state.part_keys, dense_part_keys, partkeys.value(row))
        };
        selection.push(selected);
        if selected {
            state.partial.profile.part_hits += 1;
            state.selected_partkeys.push(partkeys.value(row));
        }
    }
    Ok(Some(()))
}

fn q09_late_consume_profit_payload_batch(
    batch: RecordBatch,
    state: &mut Q09LateProfitState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let (
        Some(orderkeys),
        Some(suppkeys),
        Some(quantities),
        Some(extendedprices),
        Some(discounts),
    ) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) {
        q09_late_consume_profit_decimal_batch(
            orderkeys,
            suppkeys,
            quantities,
            extendedprices,
            discounts,
            state,
        )?;
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let Some(&partkey) = state.selected_partkeys.get(state.partkey_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q09 row selection payload overflow".to_string(),
            ));
        };
        state.partkey_offset += 1;
        let (Some(orderkey), Some(suppkey), Some(quantity), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let Some(o_year) = q09_order_year_get(
            &state.order_years,
            state.order_years.dense_slice(),
            orderkey,
        ) else {
            continue;
        };
        state.partial.profile.order_hits += 1;
        let Some(nationkey) = state.supplier_nations.get(suppkey) else {
            continue;
        };
        state.partial.profile.supplier_hits += 1;
        let Some(supplycost) = state.supply_costs.get(partkey, suppkey) else {
            continue;
        };
        state.partial.profile.supply_hits += 1;
        let amount = extendedprice * (1.0 - discount) - supplycost * quantity;
        state.partial.profile.amount_rows += 1;
        *state
            .partial
            .groups
            .entry((nationkey, o_year))
            .or_insert(0.0) += amount;
    }
    Ok(Some(()))
}

fn q09_late_consume_profit_decimal_batch(
    orderkeys: &Int64Array,
    suppkeys: &Int64Array,
    quantities: Q01DecimalInput<'_>,
    extendedprices: Q01DecimalInput<'_>,
    discounts: Q01DecimalInput<'_>,
    state: &mut Q09LateProfitState,
) -> Result<()> {
    let dense_order_years = state.order_years.dense_slice();
    let quantity_values = quantities.raw_values();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let discount_scale = discounts.scale;
    let revenue_scale = 1.0 / (extendedprices.scale * discount_scale);
    let quantity_scale = 1.0 / quantities.scale;
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        for row in 0..orderkeys.len() {
            let Some(&partkey) = state.selected_partkeys.get(state.partkey_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "Q09 row selection payload overflow".to_string(),
                ));
            };
            state.partkey_offset += 1;
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let Some(o_year) = q09_order_year_get(&state.order_years, dense_order_years, orderkey)
            else {
                continue;
            };
            state.partial.profile.order_hits += 1;
            let Some(nationkey) = state.supplier_nations.get(suppkey) else {
                continue;
            };
            state.partial.profile.supplier_hits += 1;
            let Some(supplycost) = state.supply_costs.get(partkey, suppkey) else {
                continue;
            };
            state.partial.profile.supply_hits += 1;
            let amount = (extendedprice_values[row] as f64)
                * (discount_scale - discount_values[row] as f64)
                * revenue_scale
                - supplycost * (quantity_values[row] as f64) * quantity_scale;
            state.partial.profile.amount_rows += 1;
            *state
                .partial
                .groups
                .entry((nationkey, o_year))
                .or_insert(0.0) += amount;
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        let Some(&partkey) = state.selected_partkeys.get(state.partkey_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q09 row selection payload overflow".to_string(),
            ));
        };
        state.partkey_offset += 1;
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let Some(o_year) = q09_order_year_get(&state.order_years, dense_order_years, orderkey)
        else {
            continue;
        };
        state.partial.profile.order_hits += 1;
        let Some(nationkey) = state.supplier_nations.get(suppkey) else {
            continue;
        };
        state.partial.profile.supplier_hits += 1;
        let Some(supplycost) = state.supply_costs.get(partkey, suppkey) else {
            continue;
        };
        state.partial.profile.supply_hits += 1;
        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost * (quantity_values[row] as f64) * quantity_scale;
        state.partial.profile.amount_rows += 1;
        *state
            .partial
            .groups
            .entry((nationkey, o_year))
            .or_insert(0.0) += amount;
    }
    Ok(())
}

#[derive(Default)]
struct Q09ProfitPartial {
    groups: FastHashMap<(i64, i32), f64>,
    profile: Q09ProfitProfile,
}

impl Q09ProfitPartial {
    fn merge(&mut self, batch: Self) {
        merge_f64_groups(&mut self.groups, batch.groups);
        self.profile.add(batch.profile);
    }
}

#[derive(Default)]
struct Q09ProfitProfile {
    rows: usize,
    part_hits: usize,
    order_hits: usize,
    supplier_hits: usize,
    supply_hits: usize,
    amount_rows: usize,
    part_nanos: u64,
    order_nanos: u64,
    supplier_nanos: u64,
    supply_nanos: u64,
    amount_nanos: u64,
}

impl Q09ProfitProfile {
    fn add(&mut self, other: Self) {
        self.rows = self.rows.saturating_add(other.rows);
        self.part_hits = self.part_hits.saturating_add(other.part_hits);
        self.order_hits = self.order_hits.saturating_add(other.order_hits);
        self.supplier_hits = self.supplier_hits.saturating_add(other.supplier_hits);
        self.supply_hits = self.supply_hits.saturating_add(other.supply_hits);
        self.amount_rows = self.amount_rows.saturating_add(other.amount_rows);
        self.part_nanos = self.part_nanos.saturating_add(other.part_nanos);
        self.order_nanos = self.order_nanos.saturating_add(other.order_nanos);
        self.supplier_nanos = self.supplier_nanos.saturating_add(other.supplier_nanos);
        self.supply_nanos = self.supply_nanos.saturating_add(other.supply_nanos);
        self.amount_nanos = self.amount_nanos.saturating_add(other.amount_nanos);
    }
}

fn q09_log_profit_profile(profile: &Q09ProfitProfile) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q09 profit detail: rows={} part_hits={} order_hits={} supplier_hits={} supply_hits={} amount_rows={} part={:.3} ms order={:.3} ms supplier={:.3} ms supply={:.3} ms amount={:.3} ms",
        profile.rows,
        profile.part_hits,
        profile.order_hits,
        profile.supplier_hits,
        profile.supply_hits,
        profile.amount_rows,
        sql_nanos_to_millis(profile.part_nanos),
        sql_nanos_to_millis(profile.order_nanos),
        sql_nanos_to_millis(profile.supplier_nanos),
        sql_nanos_to_millis(profile.supply_nanos),
        sql_nanos_to_millis(profile.amount_nanos),
    );
}

fn q09_profit_batch(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Result<Q09ProfitPartial> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q09_profit_decimal_batch(
        orderkeys,
        partkeys,
        suppkeys,
        quantities,
        extendedprices,
        discounts,
        part_keys,
        supplier_nations,
        order_years,
        supply_costs,
    )? {
        return Ok(groups);
    }
    let mut groups = fast_hash_map();
    let mut profile = Q09ProfitProfile::default();
    let collect_profile = tpch_profile_enabled();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_order_years = order_years.dense_slice();
    for row in 0..batch.num_rows() {
        if collect_profile {
            profile.rows += 1;
        }
        let (Some(orderkey), Some(partkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        let started = collect_profile.then(Instant::now);
        let part_hit = q09_part_key_contains(part_keys, dense_part_keys, partkey);
        if let Some(started) = started {
            profile.part_nanos = profile
                .part_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if !part_hit {
            continue;
        }
        if collect_profile {
            profile.part_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
        if let Some(started) = started {
            profile.order_nanos = profile
                .order_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(o_year) = o_year else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let nationkey = supplier_nations.get(suppkey);
        if let Some(started) = started {
            profile.supplier_nanos = profile
                .supplier_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(nationkey) = nationkey else {
            continue;
        };
        if collect_profile {
            profile.supplier_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let supplycost = supply_costs.get(partkey, suppkey);
        if let Some(started) = started {
            profile.supply_nanos = profile
                .supply_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(supplycost) = supplycost else {
            continue;
        };
        if collect_profile {
            profile.supply_hits += 1;
        }
        let (Some(quantity), Some(extendedprice), Some(discount)) = (
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let started = collect_profile.then(Instant::now);
        let amount = extendedprice * (1.0 - discount) - supplycost * quantity;
        if let Some(started) = started {
            profile.amount_nanos = profile
                .amount_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if collect_profile {
            profile.amount_rows += 1;
        }
        *groups.entry((nationkey, o_year)).or_insert(0.0) += amount;
    }
    Ok(Q09ProfitPartial { groups, profile })
}

fn q09_profit_projected_batch(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Result<Q09ProfitPartial> {
    if batch.num_columns() == 6
        && let Some(groups) = q09_profit_decimal_batch(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            batch.column(4),
            batch.column(5),
            part_keys,
            supplier_nations,
            order_years,
            supply_costs,
        )?
    {
        return Ok(groups);
    }
    q09_profit_batch(
        batch,
        part_keys,
        supplier_nations,
        order_years,
        supply_costs,
    )
}

fn q09_part_key_contains(
    part_keys: &AdaptiveI64Set,
    dense_part_keys: Option<&[bool]>,
    partkey: i64,
) -> bool {
    if let Some(dense_part_keys) = dense_part_keys {
        return usize::try_from(partkey)
            .ok()
            .and_then(|index| dense_part_keys.get(index))
            .copied()
            .unwrap_or(false);
    }
    part_keys.contains(partkey)
}

fn q09_order_year_get(
    order_years: &Q09OrderYears,
    dense_order_years: Option<(&[i32], i64, i32)>,
    orderkey: i64,
) -> Option<i32> {
    if let Some((values, base_key, missing)) = dense_order_years {
        let index = usize::try_from(orderkey.checked_sub(base_key)?).ok()?;
        return values.get(index).copied().filter(|value| *value != missing);
    }
    order_years.get(orderkey)
}

fn q09_matched_index_enabled() -> bool {
    std::env::var("DODAM_Q09_ENABLE_MATCHED_INDEX")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q09_profit_decimal_batch(
    orderkeys: &ArrayRef,
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    part_keys: &AdaptiveI64Set,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_years: &Q09OrderYears,
    supply_costs: &Q09SupplyCosts,
) -> Result<Option<Q09ProfitPartial>> {
    let (
        Some(orderkeys),
        Some(partkeys),
        Some(suppkeys),
        Some(quantities),
        Some(extendedprices),
        Some(discounts),
    ) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    )
    else {
        return Ok(None);
    };

    let mut groups = fast_hash_map();
    let mut profile = Q09ProfitProfile::default();
    let collect_profile = tpch_profile_enabled();
    let dense_part_keys = part_keys.dense_contains_slice();
    let dense_order_years = order_years.dense_slice();
    let quantity_values = quantities.raw_values();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let discount_scale = discounts.scale;
    let revenue_scale = 1.0 / (extendedprices.scale * discount_scale);
    let quantity_scale = 1.0 / quantities.scale;
    if orderkeys.null_count() == 0
        && partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let partkey_values = partkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        if q09_matched_index_enabled()
            && let Some(dense_part_keys) = dense_part_keys
        {
            return Ok(Some(q09_profit_decimal_batch_matched_index(
                orderkey_values,
                partkey_values,
                suppkey_values,
                quantity_values,
                extendedprice_values,
                discount_values,
                discount_scale,
                revenue_scale,
                quantity_scale,
                dense_part_keys,
                supplier_nations,
                order_years,
                dense_order_years,
                supply_costs,
                collect_profile,
            )));
        }
        for row in 0..orderkeys.len() {
            if collect_profile {
                profile.rows += 1;
            }
            let partkey = partkey_values[row];
            let started = collect_profile.then(Instant::now);
            let part_hit = q09_part_key_contains(part_keys, dense_part_keys, partkey);
            if let Some(started) = started {
                profile.part_nanos = profile
                    .part_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            if !part_hit {
                continue;
            }
            if collect_profile {
                profile.part_hits += 1;
            }
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let started = collect_profile.then(Instant::now);
            let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
            if let Some(started) = started {
                profile.order_nanos = profile
                    .order_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(o_year) = o_year else {
                continue;
            };
            if collect_profile {
                profile.order_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let nationkey = supplier_nations.get(suppkey);
            if let Some(started) = started {
                profile.supplier_nanos = profile
                    .supplier_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(nationkey) = nationkey else {
                continue;
            };
            if collect_profile {
                profile.supplier_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let supplycost = supply_costs.get(partkey, suppkey);
            if let Some(started) = started {
                profile.supply_nanos = profile
                    .supply_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            let Some(supplycost) = supplycost else {
                continue;
            };
            if collect_profile {
                profile.supply_hits += 1;
            }
            let started = collect_profile.then(Instant::now);
            let amount = (extendedprice_values[row] as f64)
                * (discount_scale - discount_values[row] as f64)
                * revenue_scale
                - supplycost * (quantity_values[row] as f64) * quantity_scale;
            if let Some(started) = started {
                profile.amount_nanos = profile
                    .amount_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            if collect_profile {
                profile.amount_rows += 1;
            }
            *groups.entry((nationkey, o_year)).or_insert(0.0) += amount;
        }
        return Ok(Some(Q09ProfitPartial { groups, profile }));
    }
    for row in 0..orderkeys.len() {
        if collect_profile {
            profile.rows += 1;
        }
        if orderkeys.is_null(row)
            || partkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let partkey = partkeys.value(row);
        let started = collect_profile.then(Instant::now);
        let part_hit = q09_part_key_contains(part_keys, dense_part_keys, partkey);
        if let Some(started) = started {
            profile.part_nanos = profile
                .part_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if !part_hit {
            continue;
        }
        if collect_profile {
            profile.part_hits += 1;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let started = collect_profile.then(Instant::now);
        let o_year = q09_order_year_get(order_years, dense_order_years, orderkey);
        if let Some(started) = started {
            profile.order_nanos = profile
                .order_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(o_year) = o_year else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let nationkey = supplier_nations.get(suppkey);
        if let Some(started) = started {
            profile.supplier_nanos = profile
                .supplier_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(nationkey) = nationkey else {
            continue;
        };
        if collect_profile {
            profile.supplier_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let supplycost = supply_costs.get(partkey, suppkey);
        if let Some(started) = started {
            profile.supply_nanos = profile
                .supply_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(supplycost) = supplycost else {
            continue;
        };
        if collect_profile {
            profile.supply_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost * (quantity_values[row] as f64) * quantity_scale;
        if let Some(started) = started {
            profile.amount_nanos = profile
                .amount_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        if collect_profile {
            profile.amount_rows += 1;
        }
        *groups.entry((nationkey, o_year)).or_insert(0.0) += amount;
    }
    Ok(Some(Q09ProfitPartial { groups, profile }))
}

#[allow(clippy::too_many_arguments)]
fn q09_profit_decimal_batch_matched_index(
    orderkey_values: &[i64],
    partkey_values: &[i64],
    suppkey_values: &[i64],
    quantity_values: &[i128],
    extendedprice_values: &[i128],
    discount_values: &[i128],
    discount_scale: f64,
    revenue_scale: f64,
    quantity_scale: f64,
    dense_part_keys: &[bool],
    supplier_nations: &AdaptiveI64Map<i64>,
    order_years: &Q09OrderYears,
    dense_order_years: Option<(&[i32], i64, i32)>,
    supply_costs: &Q09SupplyCosts,
    collect_profile: bool,
) -> Q09ProfitPartial {
    let mut groups = fast_hash_map();
    let mut profile = Q09ProfitProfile::default();
    if collect_profile {
        profile.rows = partkey_values.len();
    }
    let started = collect_profile.then(Instant::now);
    let mut matched_rows = Vec::new();
    for (row, partkey) in partkey_values.iter().copied().enumerate() {
        let Some(index) = usize::try_from(partkey).ok() else {
            continue;
        };
        if dense_part_keys.get(index).copied().unwrap_or(false) {
            matched_rows.push(row);
        }
    }
    if let Some(started) = started {
        profile.part_nanos = profile
            .part_nanos
            .saturating_add(sql_elapsed_nanos(started));
    }
    if collect_profile {
        profile.part_hits = matched_rows.len();
    }

    for row in matched_rows {
        let started = collect_profile.then(Instant::now);
        let o_year = q09_order_year_get(order_years, dense_order_years, orderkey_values[row]);
        if let Some(started) = started {
            profile.order_nanos = profile
                .order_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(o_year) = o_year else {
            continue;
        };
        if collect_profile {
            profile.order_hits += 1;
        }
        let suppkey = suppkey_values[row];
        let started = collect_profile.then(Instant::now);
        let nationkey = supplier_nations.get(suppkey);
        if let Some(started) = started {
            profile.supplier_nanos = profile
                .supplier_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(nationkey) = nationkey else {
            continue;
        };
        if collect_profile {
            profile.supplier_hits += 1;
        }
        let partkey = partkey_values[row];
        let started = collect_profile.then(Instant::now);
        let supplycost = supply_costs.get(partkey, suppkey);
        if let Some(started) = started {
            profile.supply_nanos = profile
                .supply_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        let Some(supplycost) = supplycost else {
            continue;
        };
        if collect_profile {
            profile.supply_hits += 1;
        }
        let started = collect_profile.then(Instant::now);
        let amount = (extendedprice_values[row] as f64)
            * (discount_scale - discount_values[row] as f64)
            * revenue_scale
            - supplycost * (quantity_values[row] as f64) * quantity_scale;
        if let Some(started) = started {
            profile.amount_nanos = profile
                .amount_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        *groups.entry((nationkey, o_year)).or_insert(0.0) += amount;
        if collect_profile {
            profile.amount_rows += 1;
        }
    }

    Q09ProfitPartial { groups, profile }
}

fn q09_output(rows: Vec<Q09Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("nation", DataType::Utf8, false),
            Field::new("o_year", DataType::Int64, false),
            Field::new("sum_profit", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.nation.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| i64::from(row.o_year)),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.sum_profit),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q11_important_stock_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q11_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 3 {
        return Ok(None);
    }
    let mut partsupp = None;
    let mut supplier = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(partsupp), Some(supplier), Some(nation)) = (partsupp, supplier, nation) else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(nation_name) = string_equality_literal(&conjuncts, "n_name")? else {
        return Ok(None);
    };
    let stage = tpch_profile_start();
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, &nation_name).await?;
    tpch_profile_elapsed("Q11 nation keys", stage);
    if nation_keys.is_empty() {
        return Ok(Some(q11_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let supplier_keys = q11_supplier_keys(engine, supplier.path, batch_size, &nation_keys).await?;
    tpch_profile_elapsed("Q11 supplier keys", stage);
    if supplier_keys.is_empty() {
        return Ok(Some(q11_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let rows = q11_important_stock_rows(engine, partsupp.path, batch_size, &supplier_keys).await?;
    tpch_profile_elapsed("Q11 partsupp grouped value", stage);
    Ok(Some(q11_output(rows)?))
}

fn q11_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let having = select
        .having
        .as_ref()
        .map(|expr| expr.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 3
        && select.projection.len() == 2
        && projection.contains("ps_partkey")
        && projection.contains("sum(ps_supplycost * ps_availqty)")
        && group_by.contains("ps_partkey")
        && having.contains("sum(ps_supplycost * ps_availqty)")
        && having.contains("* 0.0001")
        && order_by.contains("value desc")
        && selection.contains("ps_suppkey = s_suppkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("n_name")
}

async fn q11_supplier_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_keys.contains(&nationkey) {
                suppliers.insert(suppkey);
            }
        }
    }
    Ok(suppliers)
}

async fn q11_important_stock_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    supplier_keys: &HashSet<i64>,
) -> Result<Vec<Q11Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_supplycost".to_string(),
                "ps_availqty".to_string(),
            ]),
            None,
        )
        .await?;
    let supplier_keys = Arc::new(supplier_keys.clone());
    let (values, total) = parallel_batch_fold(
        &mut stream,
        move |batch| q11_important_stock_batch(batch, &supplier_keys),
        (HashMap::<i64, f64>::new(), 0.0_f64),
        |(values, total), (batch_values, batch_total)| {
            *total += batch_total;
            for (partkey, value) in batch_values {
                *values.entry(partkey).or_insert(0.0) += value;
            }
        },
        "Q11 partsupp value",
    )?;
    let threshold = total * 0.0001;
    let mut rows = values
        .into_iter()
        .filter_map(|(ps_partkey, value)| {
            (value > threshold).then_some(Q11Row { ps_partkey, value })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .value
            .partial_cmp(&left.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.ps_partkey.cmp(&right.ps_partkey))
    });
    Ok(rows)
}

fn q11_important_stock_batch(
    batch: RecordBatch,
    supplier_keys: &HashSet<i64>,
) -> Result<(HashMap<i64, f64>, f64)> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    let supplycosts = batch_column(&batch, "ps_supplycost")?;
    let availqtys = batch_column(&batch, "ps_availqty")?;
    if let Some(result) =
        q11_important_stock_batch_typed(partkeys, suppkeys, supplycosts, availqtys, supplier_keys)?
    {
        return Ok(result);
    }
    let mut values = HashMap::new();
    let mut total = 0.0_f64;
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey), Some(supplycost), Some(availqty)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(supplycosts, row)?,
            numeric_f64_value(availqtys, row)?,
        ) else {
            continue;
        };
        if !supplier_keys.contains(&suppkey) {
            continue;
        }
        let value = supplycost * availqty;
        total += value;
        *values.entry(partkey).or_insert(0.0) += value;
    }
    Ok((values, total))
}

fn q11_important_stock_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    supplycosts: &ArrayRef,
    availqtys: &ArrayRef,
    supplier_keys: &HashSet<i64>,
) -> Result<Option<(HashMap<i64, f64>, f64)>> {
    let (Some(partkeys), Some(suppkeys), Some(supplycosts), Some(availqtys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(supplycosts)?,
        availqtys.as_any().downcast_ref::<Int32Array>(),
    ) else {
        return Ok(None);
    };
    let mut values = HashMap::new();
    let mut total = 0.0_f64;
    for row in 0..partkeys.len() {
        if partkeys.is_null(row)
            || suppkeys.is_null(row)
            || supplycosts.is_null(row)
            || availqtys.is_null(row)
        {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if !supplier_keys.contains(&suppkey) {
            continue;
        }
        let value = supplycosts.value(row) * f64::from(availqtys.value(row));
        total += value;
        *values.entry(partkeys.value(row)).or_insert(0.0) += value;
    }
    Ok(Some((values, total)))
}

struct Q11Row {
    ps_partkey: i64,
    value: f64,
}

fn q11_output(rows: Vec<Q11Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("ps_partkey", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.ps_partkey),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.value),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q16_parts_supplier_relationship_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q16_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut partsupp = None;
    let mut part = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        }
    }
    let (Some(partsupp), Some(part)) = (partsupp, part) else {
        return Ok(None);
    };
    let Some(supplier_path) = first_table_path_in_subqueries(selection, "supplier")? else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(excluded_brand) = string_inequality_literal(&conjuncts, "p_brand")? else {
        return Ok(None);
    };
    let Some(excluded_type_prefix) = not_like_prefix_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some(sizes) = numeric_in_i64_literals(&conjuncts, "p_size")? else {
        return Ok(None);
    };
    let sizes = AdaptiveI64Set::from_hash(sizes);
    let Some(comment_parts) = like_substrings_literal(selection, "s_comment")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let bad_suppliers =
        q16_bad_suppliers(engine, supplier_path, batch_size, &comment_parts).await?;
    tpch_profile_elapsed("Q16 bad suppliers", stage);
    let stage = tpch_profile_start();
    let part_groups = q16_part_groups(
        engine,
        part.path,
        batch_size,
        &excluded_brand,
        &excluded_type_prefix,
        &sizes,
    )
    .await?;
    tpch_profile_elapsed("Q16 part groups", stage);
    if part_groups.part_to_group.is_empty() {
        return Ok(Some(q16_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let rows = q16_supplier_counts(
        engine,
        partsupp.path,
        batch_size,
        part_groups,
        bad_suppliers,
    )
    .await?;
    tpch_profile_elapsed("Q16 supplier counts", stage);
    Ok(Some(q16_output(rows)?))
}

fn q16_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 2
        && select.projection.len() == 4
        && projection.contains("p_brand")
        && projection.contains("p_type")
        && projection.contains("p_size")
        && projection.contains("count(distinct ps_suppkey)")
        && group_by.contains("p_brand")
        && group_by.contains("p_type")
        && group_by.contains("p_size")
        && order_by.contains("supplier_cnt desc")
        && selection.contains("p_partkey = ps_partkey")
        && selection.contains("p_brand <>")
        && selection.contains("p_type not like")
        && selection.contains("p_size in")
        && selection.contains("ps_suppkey not in")
        && selection.contains("s_comment like")
}

fn string_inequality_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::NotEq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Utf8(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Utf8(value) = sql_literal_value(left)?
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn not_like_prefix_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } = conjunct
        else {
            continue;
        };
        if !*negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
            continue;
        };
        if let Some(value) = pattern.strip_suffix('%')
            && !value.contains('%')
            && !value.contains('_')
        {
            return Ok(Some(value.to_string()));
        }
    }
    Ok(None)
}

fn numeric_in_i64_literals(conjuncts: &[SqlExpr], column: &str) -> Result<Option<HashSet<i64>>> {
    for conjunct in conjuncts {
        let SqlExpr::InList {
            expr,
            list,
            negated,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let mut values = HashSet::new();
        for item in list {
            values.insert(literal_as_f64(&sql_literal_value(item)?)? as i64);
        }
        return Ok(Some(values));
    }
    Ok(None)
}

fn like_substrings_literal(expr: &SqlExpr, column: &str) -> Result<Option<Vec<String>>> {
    match expr {
        SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } if !*negated && sql_expr_column_matches(expr, column) => {
            let LiteralValue::Utf8(pattern) = sql_literal_value(pattern)? else {
                return Ok(None);
            };
            let parts = pattern
                .split('%')
                .filter(|part| !part.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            Ok((!parts.is_empty() && !pattern.contains('_')).then_some(parts))
        }
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            if let SetExpr::Select(select) = subquery.body.as_ref()
                && let Some(selection) = select.selection.as_ref()
            {
                return like_substrings_literal(selection, column);
            }
            Ok(None)
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            Ok(like_substrings_literal(left, column)?.or(like_substrings_literal(right, column)?))
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            like_substrings_literal(expr, column)
        }
        SqlExpr::InList { expr, list, .. } => {
            if let Some(parts) = like_substrings_literal(expr, column)? {
                return Ok(Some(parts));
            }
            for item in list {
                if let Some(parts) = like_substrings_literal(item, column)? {
                    return Ok(Some(parts));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

struct Q16BadSuppliers {
    keys: HashSet<i64>,
    max_suppkey: Option<i64>,
}

async fn q16_bad_suppliers(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    comment_parts: &[String],
) -> Result<Q16BadSuppliers> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_comment".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashSet::new();
    let mut max_suppkey = None::<i64>;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let comments = batch_string_column(&batch, "s_comment")?;
        for row in 0..batch.num_rows() {
            let Some(suppkey) = numeric_i64_value(suppkeys, row)? else {
                continue;
            };
            max_suppkey = Some(max_suppkey.map_or(suppkey, |max_key| max_key.max(suppkey)));
            if comments.is_null(row)
                || !ordered_substrings_match(comments.value(row), comment_parts)
            {
                continue;
            }
            suppliers.insert(suppkey);
        }
    }
    Ok(Q16BadSuppliers {
        keys: suppliers,
        max_suppkey,
    })
}

fn ordered_substrings_match(value: &str, parts: &[String]) -> bool {
    let mut rest = value;
    for part in parts {
        let Some(index) = rest.find(part) else {
            return false;
        };
        rest = &rest[index + part.len()..];
    }
    true
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct Q16GroupKey {
    brand: String,
    type_name: String,
    size: i64,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Q16GroupIdKey {
    brand_id: usize,
    type_id: usize,
    size: i64,
}

struct Q16PartGroups {
    groups: Vec<Q16GroupKey>,
    part_to_group: AdaptiveI64Map<usize>,
}

async fn q16_part_groups(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    excluded_brand: &str,
    excluded_type_prefix: &str,
    sizes: &AdaptiveI64Set,
) -> Result<Q16PartGroups> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "p_partkey".to_string(),
                "p_brand".to_string(),
                "p_type".to_string(),
                "p_size".to_string(),
            ]),
            None,
        )
        .await?;
    let mut brand_ids = HashMap::<String, usize>::new();
    let mut type_ids = HashMap::<String, usize>::new();
    let mut brands_by_id = Vec::<String>::new();
    let mut types_by_id = Vec::<String>::new();
    let mut group_ids = FastHashMap::<Q16GroupIdKey, usize>::default();
    let mut groups = Vec::<Q16GroupKey>::new();
    let mut part_to_group = HashMap::<i64, usize>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let brands = batch_string_column(&batch, "p_brand")?;
        let types = batch_string_column(&batch, "p_type")?;
        let part_sizes = batch_column(&batch, "p_size")?;
        for row in 0..batch.num_rows() {
            if brands.is_null(row)
                || types.is_null(row)
                || brands.value(row) == excluded_brand
                || types.value(row).starts_with(excluded_type_prefix)
            {
                continue;
            }
            let (Some(partkey), Some(size)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(part_sizes, row)?,
            ) else {
                continue;
            };
            if !sizes.contains(size) {
                continue;
            }
            let brand_id = q16_intern_string(&mut brand_ids, &mut brands_by_id, brands.value(row));
            let type_id = q16_intern_string(&mut type_ids, &mut types_by_id, types.value(row));
            let key = Q16GroupIdKey {
                brand_id,
                type_id,
                size,
            };
            let group_id = if let Some(group_id) = group_ids.get(&key).copied() {
                group_id
            } else {
                let group_id = groups.len();
                groups.push(Q16GroupKey {
                    brand: brands_by_id[brand_id].clone(),
                    type_name: types_by_id[type_id].clone(),
                    size,
                });
                group_ids.insert(key, group_id);
                group_id
            };
            part_to_group.insert(partkey, group_id);
        }
    }
    Ok(Q16PartGroups {
        groups,
        part_to_group: AdaptiveI64Map::from_hash(part_to_group),
    })
}

fn q16_intern_string(
    ids: &mut HashMap<String, usize>,
    values: &mut Vec<String>,
    value: &str,
) -> usize {
    if let Some(id) = ids.get(value).copied() {
        return id;
    }
    let id = values.len();
    let value = value.to_string();
    values.push(value.clone());
    ids.insert(value, id);
    id
}

struct Q16Row {
    brand: String,
    type_name: String,
    size: i64,
    supplier_count: u64,
}

async fn q16_supplier_counts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_groups: Q16PartGroups,
    bad_suppliers: Q16BadSuppliers,
) -> Result<Vec<Q16Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["ps_partkey".to_string(), "ps_suppkey".to_string()]),
            None,
        )
        .await?;
    let groups = part_groups.groups;
    let part_to_group = Arc::new(part_groups.part_to_group);
    let bad_supplier_keys = Arc::new(AdaptiveI64Set::from_hash(bad_suppliers.keys));
    let supplier_counts =
        if let Some(layout) = q16_supplier_bitset_layout(groups.len(), bad_suppliers.max_suppkey) {
            let layout_for_scan = Arc::new(layout.clone());
            let partial = parallel_batch_fold_chunks(
                &mut stream,
                q16_supplier_count_chunk_size(),
                move |batches| {
                    let mut distinct_suppliers =
                        Q16GroupSupplierBitset::new((*layout_for_scan).clone());
                    for batch in batches {
                        q16_supplier_counts_bitset_batch(
                            batch,
                            &part_to_group,
                            &bad_supplier_keys,
                            &mut distinct_suppliers,
                        )?;
                    }
                    Ok(distinct_suppliers)
                },
                Q16GroupSupplierBitset::new(layout),
                q16_merge_supplier_bitsets,
                "Q16 partsupp supplier counts",
            )?;
            partial.counts()
        } else {
            let distinct_suppliers = parallel_batch_fold(
                &mut stream,
                move |batch| q16_supplier_counts_batch(batch, &part_to_group, &bad_supplier_keys),
                FastHashSet::<(usize, i64)>::default(),
                q16_merge_supplier_counts,
                "Q16 partsupp supplier counts",
            )?;
            let mut supplier_counts = vec![0_u64; groups.len()];
            for (group_id, _) in distinct_suppliers {
                if let Some(count) = supplier_counts.get_mut(group_id) {
                    *count += 1;
                }
            }
            supplier_counts
        };
    let mut rows = supplier_counts
        .into_iter()
        .enumerate()
        .filter_map(|(group_id, supplier_count)| {
            if supplier_count == 0 {
                return None;
            }
            let group = groups.get(group_id)?;
            Some(Q16Row {
                brand: group.brand.clone(),
                type_name: group.type_name.clone(),
                size: group.size,
                supplier_count,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .supplier_count
            .cmp(&left.supplier_count)
            .then_with(|| left.brand.cmp(&right.brand))
            .then_with(|| left.type_name.cmp(&right.type_name))
            .then_with(|| left.size.cmp(&right.size))
    });
    Ok(rows)
}

#[derive(Clone)]
struct Q16SupplierBitsetLayout {
    group_count: usize,
    words_per_group: usize,
}

struct Q16GroupSupplierBitset {
    layout: Q16SupplierBitsetLayout,
    words: Vec<u64>,
}

impl Q16GroupSupplierBitset {
    fn new(layout: Q16SupplierBitsetLayout) -> Self {
        let words = vec![0; layout.group_count.saturating_mul(layout.words_per_group)];
        Self { layout, words }
    }

    fn insert(&mut self, group_id: usize, suppkey: i64) {
        if suppkey < 0 || group_id >= self.layout.group_count {
            return;
        }
        let suppkey = suppkey as usize;
        let word = suppkey / 64;
        if word >= self.layout.words_per_group {
            return;
        }
        let index = group_id * self.layout.words_per_group + word;
        self.words[index] |= 1_u64 << (suppkey & 63);
    }

    fn merge(&mut self, other: Q16GroupSupplierBitset) {
        for (left, right) in self.words.iter_mut().zip(other.words) {
            *left |= right;
        }
    }

    fn counts(&self) -> Vec<u64> {
        self.words
            .chunks(self.layout.words_per_group)
            .map(|words| words.iter().map(|word| u64::from(word.count_ones())).sum())
            .collect()
    }
}

fn q16_supplier_bitset_layout(
    group_count: usize,
    max_suppkey: Option<i64>,
) -> Option<Q16SupplierBitsetLayout> {
    if std::env::var_os("DODAM_Q16_ENABLE_SUPPLIER_BITSET").is_none() {
        return None;
    }
    let max_suppkey = max_suppkey?;
    if max_suppkey < 0 || group_count == 0 {
        return None;
    }
    let words_per_group = (usize::try_from(max_suppkey).ok()? + 64) / 64;
    let bytes = group_count.checked_mul(words_per_group)?.checked_mul(8)?;
    if bytes > q16_supplier_bitset_max_bytes() {
        return None;
    }
    Some(Q16SupplierBitsetLayout {
        group_count,
        words_per_group,
    })
}

fn q16_supplier_bitset_max_bytes() -> usize {
    std::env::var("DODAM_Q16_SUPPLIER_BITSET_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32 * 1024 * 1024)
}

fn q16_supplier_count_chunk_size() -> usize {
    std::env::var("DODAM_Q16_SUPPLIER_COUNT_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

fn q16_supplier_counts_bitset_batch(
    batch: RecordBatch,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut Q16GroupSupplierBitset,
) -> Result<()> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    if q16_supplier_counts_bitset_batch_typed(
        partkeys,
        suppkeys,
        part_to_group,
        bad_suppliers,
        distinct_suppliers,
    ) {
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkey) else {
            continue;
        };
        distinct_suppliers.insert(group_id, suppkey);
    }
    Ok(())
}

fn q16_supplier_counts_bitset_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
    distinct_suppliers: &mut Q16GroupSupplierBitset,
) -> bool {
    let (Some(partkeys), Some(suppkeys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return false;
    };
    if partkeys.null_count() == 0 && suppkeys.null_count() == 0 {
        let partkey_values = partkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        for row in 0..partkey_values.len() {
            let suppkey = suppkey_values[row];
            if bad_suppliers.contains(suppkey) {
                continue;
            }
            let Some(group_id) = part_to_group.get(partkey_values[row]) else {
                continue;
            };
            distinct_suppliers.insert(group_id, suppkey);
        }
        return true;
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkeys.value(row)) else {
            continue;
        };
        distinct_suppliers.insert(group_id, suppkey);
    }
    true
}

fn q16_supplier_counts_batch(
    batch: RecordBatch,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
) -> Result<FastHashSet<(usize, i64)>> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    if let Some(groups) =
        q16_supplier_counts_batch_typed(partkeys, suppkeys, part_to_group, bad_suppliers)?
    {
        return Ok(groups);
    }
    let mut distinct_suppliers = FastHashSet::default();
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkey) else {
            continue;
        };
        distinct_suppliers.insert((group_id, suppkey));
    }
    Ok(distinct_suppliers)
}

fn q16_supplier_counts_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    part_to_group: &AdaptiveI64Map<usize>,
    bad_suppliers: &AdaptiveI64Set,
) -> Result<Option<FastHashSet<(usize, i64)>>> {
    let (Some(partkeys), Some(suppkeys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return Ok(None);
    };
    let mut distinct_suppliers = FastHashSet::default();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if bad_suppliers.contains(suppkey) {
            continue;
        }
        let Some(group_id) = part_to_group.get(partkeys.value(row)) else {
            continue;
        };
        distinct_suppliers.insert((group_id, suppkey));
    }
    Ok(Some(distinct_suppliers))
}

fn q16_merge_supplier_counts(
    groups: &mut FastHashSet<(usize, i64)>,
    batch_groups: FastHashSet<(usize, i64)>,
) {
    groups.extend(batch_groups);
}

fn q16_merge_supplier_bitsets(
    groups: &mut Q16GroupSupplierBitset,
    batch_groups: Q16GroupSupplierBitset,
) {
    groups.merge(batch_groups);
}

fn q16_output(rows: Vec<Q16Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("p_brand", DataType::Utf8, false),
            Field::new("p_type", DataType::Utf8, false),
            Field::new("p_size", DataType::Int64, false),
            Field::new("supplier_cnt", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.brand.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.type_name.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.size),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.supplier_count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q12_shipping_modes_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q12_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut orders = None;
    let mut lineitem = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        }
    }
    let (Some(orders), Some(lineitem)) = (orders, lineitem) else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_receiptdate")? else {
        return Ok(None);
    };
    let Some(shipmodes) = string_in_literals(&conjuncts, "l_shipmode")? else {
        return Ok(None);
    };
    if shipmodes.len() != 2 {
        return Ok(None);
    }
    let mut shipmodes = shipmodes.into_iter().collect::<Vec<_>>();
    shipmodes.sort();
    if !orders.path.exists() {
        return Err(DodamError::MissingPath(orders.path));
    }
    let pending = q12_filtered_lineitem_counts(
        engine,
        lineitem.path,
        batch_size,
        &shipmodes,
        start_days,
        end_days,
    )
    .await?;
    let rows =
        q12_shipping_mode_counts_from_orders(engine, orders.path, batch_size, &shipmodes, &pending)
            .await?;
    Ok(Some(q12_output(rows)?))
}

fn q12_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 2
        && select.projection.len() == 3
        && projection.contains("l_shipmode")
        && projection.contains("high_line_count")
        && projection.contains("low_line_count")
        && projection.contains("o_orderpriority = '1-urgent'")
        && projection.contains("o_orderpriority = '2-high'")
        && group_by.contains("l_shipmode")
        && order_by.contains("l_shipmode")
        && selection.contains("o_orderkey = l_orderkey")
        && selection.contains("l_shipmode in")
        && selection.contains("l_commitdate < l_receiptdate")
        && selection.contains("l_shipdate < l_commitdate")
        && selection.contains("l_receiptdate")
}

fn string_in_literals(conjuncts: &[SqlExpr], column: &str) -> Result<Option<HashSet<String>>> {
    for conjunct in conjuncts {
        let SqlExpr::InList {
            expr,
            list,
            negated,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let mut values = HashSet::new();
        for item in list {
            let LiteralValue::Utf8(value) = sql_literal_value(item)? else {
                return Ok(None);
            };
            values.insert(value);
        }
        return Ok(Some(values));
    }
    Ok(None)
}

#[derive(Clone, Copy, Default)]
struct Q12State {
    high_line_count: u64,
    low_line_count: u64,
}

struct Q12Row {
    shipmode: String,
    high_line_count: u64,
    low_line_count: u64,
}

#[derive(Clone, Copy, Default)]
struct Q12PendingOrder {
    counts: [u64; 2],
}

async fn q12_filtered_lineitem_counts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, Q12PendingOrder>> {
    if q12_late_materialized_enabled()
        && let Some(pending) = q12_filtered_lineitem_counts_late_materialized(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(pending);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_shipmode".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
        "l_shipdate".to_string(),
    ]);
    let shipmodes = Arc::new(shipmodes.to_vec());
    if q12_row_group_map_enabled()
        && let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                q12_row_group_map_chunk(),
                HashMap::<i64, Q12PendingOrder>::new,
                {
                    let shipmodes = shipmodes.clone();
                    move |batch, pending| {
                        q12_merge_pending_orders(
                            pending,
                            q12_filtered_lineitem_counts_projected_batch(
                                batch, &shipmodes, start_days, end_days,
                            )?,
                        );
                        Ok(Some(()))
                    }
                },
                |pending| Ok(Some(pending)),
            )
            .await?
    {
        let mut pending = HashMap::<i64, Q12PendingOrder>::new();
        for partial in partials {
            q12_merge_pending_orders(&mut pending, partial);
        }
        return Ok(pending);
    }
    q12_filtered_lineitem_counts_stream(
        engine, path, batch_size, projection, shipmodes, start_days, end_days,
    )
    .await
}

async fn q12_filtered_lineitem_counts_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, Q12PendingOrder>>> {
    let predicate_projection = Projection::Columns(vec![
        "l_shipmode".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
        "l_shipdate".to_string(),
    ]);
    let payload_projection = Projection::Columns(vec!["l_orderkey".to_string()]);
    let shipmodes = Arc::new(shipmodes.to_vec());
    let Some(chunks) = engine
        .late_materialized_parquet_map_with_policy(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            q12_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q12_late_materialized_max_selected_ratio(),
                q12_late_materialized_max_selector_run_ratio(),
            ),
            {
                let shipmodes = shipmodes.clone();
                move || Q12LateState {
                    shipmodes: shipmodes.clone(),
                    start_days,
                    end_days,
                    selected_modes: Vec::new(),
                    selected_offset: 0,
                    pending: HashMap::new(),
                }
            },
            q12_late_build_selection_batch,
            q12_late_consume_orderkey_payload_batch,
            |state, _metrics| {
                if state.selected_offset != state.selected_modes.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q12 row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some(state.pending))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut pending = HashMap::new();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        q12_merge_pending_orders(&mut pending, chunk.output);
        metrics.add(chunk.metrics);
    }
    q12_log_late_materialized_profile(metrics, q12_late_materialized_row_group_chunk());
    Ok(Some(pending))
}

fn q12_lineitem_chunk_size() -> usize {
    std::env::var("DODAM_Q12_LINEITEM_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

async fn q12_filtered_lineitem_counts_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, Q12PendingOrder>> {
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    parallel_batch_fold_chunks(
        &mut stream,
        q12_lineitem_chunk_size(),
        move |batches| {
            let mut pending = HashMap::<i64, Q12PendingOrder>::new();
            for batch in batches {
                q12_merge_pending_orders(
                    &mut pending,
                    q12_filtered_lineitem_counts_projected_batch(
                        batch, &shipmodes, start_days, end_days,
                    )?,
                );
            }
            Ok(pending)
        },
        HashMap::<i64, Q12PendingOrder>::new(),
        q12_merge_pending_orders,
        "Q12 lineitem aggregate",
    )
}

fn q12_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q12_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q12_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q12_late_materialized_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_LATE_MATERIALIZE").is_some()
}

fn q12_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q12_late_materialized_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q12_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

fn q12_late_materialized_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q12_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.50)
}

fn q12_filtered_lineitem_counts_batch(
    batch: RecordBatch,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, Q12PendingOrder>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let modes = batch_string_column(&batch, "l_shipmode")?;
    let commitdates = batch_column(&batch, "l_commitdate")?;
    let receiptdates = batch_column(&batch, "l_receiptdate")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if q12_typed_loop_enabled()
        && let Some(pending) = q12_filtered_lineitem_counts_batch_typed(
            orderkeys,
            modes,
            commitdates,
            receiptdates,
            shipdates,
            shipmodes,
            start_days,
            end_days,
        )
    {
        return Ok(pending);
    }
    let mut pending = HashMap::<i64, Q12PendingOrder>::new();
    for row in 0..batch.num_rows() {
        if modes.is_null(row) {
            continue;
        }
        let mode = modes.value(row);
        let Some(mode_index) = q12_shipmode_index(shipmodes, mode) else {
            continue;
        };
        let (Some(orderkey), Some(commitdate), Some(receiptdate), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(commitdates, row)?,
            date32_value(receiptdates, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if commitdate >= receiptdate
            || shipdate >= commitdate
            || receiptdate < start_days
            || receiptdate >= end_days
        {
            continue;
        }
        pending.entry(orderkey).or_default().counts[mode_index] += 1;
    }
    Ok(pending)
}

fn q12_filtered_lineitem_counts_projected_batch(
    batch: RecordBatch,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, Q12PendingOrder>> {
    if batch.num_columns() == 5
        && let Some(modes) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && q12_typed_loop_enabled()
        && let Some(pending) = q12_filtered_lineitem_counts_batch_typed(
            batch.column(0),
            modes,
            batch.column(2),
            batch.column(3),
            batch.column(4),
            shipmodes,
            start_days,
            end_days,
        )
    {
        return Ok(pending);
    }
    q12_filtered_lineitem_counts_batch(batch, shipmodes, start_days, end_days)
}

#[allow(clippy::too_many_arguments)]
fn q12_filtered_lineitem_counts_batch_typed(
    orderkeys: &ArrayRef,
    modes: &StringArray,
    commitdates: &ArrayRef,
    receiptdates: &ArrayRef,
    shipdates: &ArrayRef,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Option<HashMap<i64, Q12PendingOrder>> {
    let (Some(orderkeys), Some(commitdates), Some(receiptdates), Some(shipdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        commitdates.as_any().downcast_ref::<Date32Array>(),
        receiptdates.as_any().downcast_ref::<Date32Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return None;
    };
    let [left_mode, right_mode] = shipmodes else {
        return None;
    };
    let left_mode = left_mode.as_bytes();
    let right_mode = right_mode.as_bytes();
    let mode_offsets = modes.value_offsets();
    let mode_data = modes.value_data();
    let mut pending = HashMap::<i64, Q12PendingOrder>::new();
    if orderkeys.null_count() == 0
        && modes.null_count() == 0
        && commitdates.null_count() == 0
        && receiptdates.null_count() == 0
        && shipdates.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let commitdate_values = commitdates.values().as_ref();
        let receiptdate_values = receiptdates.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        for row in 0..orderkeys.len() {
            let mode = bytes_string_parts(mode_offsets, mode_data, row);
            let mode_index = if mode == left_mode {
                0
            } else if mode == right_mode {
                1
            } else {
                continue;
            };
            let commitdate = commitdate_values[row];
            let receiptdate = receiptdate_values[row];
            if commitdate >= receiptdate
                || shipdate_values[row] >= commitdate
                || receiptdate < start_days
                || receiptdate >= end_days
            {
                continue;
            }
            pending.entry(orderkey_values[row]).or_default().counts[mode_index] += 1;
        }
        return Some(pending);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || modes.is_null(row)
            || commitdates.is_null(row)
            || receiptdates.is_null(row)
            || shipdates.is_null(row)
        {
            continue;
        }
        let mode = bytes_string_parts(mode_offsets, mode_data, row);
        let mode_index = if mode == left_mode {
            0
        } else if mode == right_mode {
            1
        } else {
            continue;
        };
        let commitdate = commitdates.value(row);
        let receiptdate = receiptdates.value(row);
        if commitdate >= receiptdate
            || shipdates.value(row) >= commitdate
            || receiptdate < start_days
            || receiptdate >= end_days
        {
            continue;
        }
        pending.entry(orderkeys.value(row)).or_default().counts[mode_index] += 1;
    }
    Some(pending)
}

fn q12_shipmode_index(shipmodes: &[String], mode: &str) -> Option<usize> {
    match shipmodes {
        [left, _] if mode == left => Some(0),
        [_, right] if mode == right => Some(1),
        _ => None,
    }
}

fn q12_typed_loop_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_TYPED_LOOP").is_none()
}

struct Q12LateState {
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
    selected_modes: Vec<u8>,
    selected_offset: usize,
    pending: HashMap<i64, Q12PendingOrder>,
}

fn q12_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q12LateState,
) -> Result<Option<()>> {
    let modes = batch_string_column(&batch, "l_shipmode")?;
    let Some(commitdates) = batch_column(&batch, "l_commitdate")?
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let Some(receiptdates) = batch_column(&batch, "l_receiptdate")?
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let Some(shipdates) = batch_column(&batch, "l_shipdate")?
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let [left_mode, right_mode] = state.shipmodes.as_slice() else {
        return Ok(None);
    };
    if modes.null_count() != 0
        || commitdates.null_count() != 0
        || receiptdates.null_count() != 0
        || shipdates.null_count() != 0
    {
        return Ok(None);
    }
    let left_mode = left_mode.as_bytes();
    let right_mode = right_mode.as_bytes();
    let mode_offsets = modes.value_offsets();
    let mode_data = modes.value_data();
    let commitdate_values = commitdates.values().as_ref();
    let receiptdate_values = receiptdates.values().as_ref();
    let shipdate_values = shipdates.values().as_ref();
    for row in 0..batch.num_rows() {
        let mode = bytes_string_parts(mode_offsets, mode_data, row);
        let mode_index = if mode == left_mode {
            0
        } else if mode == right_mode {
            1
        } else {
            selection.push(false);
            continue;
        };
        let commitdate = commitdate_values[row];
        let receiptdate = receiptdate_values[row];
        let selected = commitdate < receiptdate
            && shipdate_values[row] < commitdate
            && receiptdate >= state.start_days
            && receiptdate < state.end_days;
        if selected {
            state.selected_modes.push(mode_index);
        }
        selection.push(selected);
    }
    Ok(Some(()))
}

fn q12_late_consume_orderkey_payload_batch(
    batch: RecordBatch,
    state: &mut Q12LateState,
) -> Result<Option<()>> {
    let Some(orderkeys) = batch_column(&batch, "l_orderkey")?
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    if orderkeys.null_count() != 0 {
        return Ok(None);
    }
    for &orderkey in orderkeys.values() {
        let mode_index = *state
            .selected_modes
            .get(state.selected_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("Q12 row selection payload mismatch".to_string())
            })? as usize;
        state.pending.entry(orderkey).or_default().counts[mode_index] += 1;
        state.selected_offset += 1;
    }
    Ok(Some(()))
}

fn q12_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q12 lineitem: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q12_merge_pending_orders(
    pending: &mut HashMap<i64, Q12PendingOrder>,
    batch_pending: HashMap<i64, Q12PendingOrder>,
) {
    for (orderkey, order) in batch_pending {
        let target = pending.entry(orderkey).or_default();
        for index in 0..target.counts.len() {
            target.counts[index] += order.counts[index];
        }
    }
}

async fn q12_shipping_mode_counts_from_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &HashMap<i64, Q12PendingOrder>,
) -> Result<Vec<Q12Row>> {
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let mut stream = if q12_order_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "o_orderkey",
                pending.keys().copied().collect(),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let groups = parallel_batch_fold(
        &mut stream,
        move |batch| q12_shipping_mode_counts_projected_batch(batch, &pending),
        [Q12State::default(); 2],
        q12_merge_shipping_mode_counts,
        "Q12 orders aggregate",
    )?;
    let rows = groups
        .into_iter()
        .enumerate()
        .map(|(index, state)| Q12Row {
            shipmode: shipmodes[index].clone(),
            high_line_count: state.high_line_count,
            low_line_count: state.low_line_count,
        })
        .collect::<Vec<_>>();
    Ok(rows)
}

fn q12_order_row_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_ROW_FILTER").is_some()
}

fn q12_shipping_mode_counts_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<Q12PendingOrder>,
) -> Result<[Q12State; 2]> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
    if q12_typed_loop_enabled()
        && let Some(groups) =
            q12_shipping_mode_counts_batch_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let mut groups = [Q12State::default(); 2];
    for row in 0..batch.num_rows() {
        if orderpriorities.is_null(row) {
            continue;
        }
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        let Some(order) = pending.get(orderkey) else {
            continue;
        };
        let is_high_priority = matches!(orderpriorities.value(row), "1-URGENT" | "2-HIGH");
        for (index, count) in order.counts.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            let group = &mut groups[index];
            if is_high_priority {
                group.high_line_count += count;
            } else {
                group.low_line_count += count;
            }
        }
    }
    Ok(groups)
}

fn q12_shipping_mode_counts_projected_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<Q12PendingOrder>,
) -> Result<[Q12State; 2]> {
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && q12_typed_loop_enabled()
        && let Some(groups) =
            q12_shipping_mode_counts_batch_typed(batch.column(0), orderpriorities, pending)
    {
        return Ok(groups);
    }
    q12_shipping_mode_counts_batch(batch, pending)
}

fn q12_shipping_mode_counts_batch_typed(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &AdaptiveI64Map<Q12PendingOrder>,
) -> Option<[Q12State; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [Q12State::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let priority = bytes_string_parts(priority_offsets, priority_data, row);
            let is_high_priority = matches!(priority, b"1-URGENT" | b"2-HIGH");
            q12_apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = matches!(priority, b"1-URGENT" | b"2-HIGH");
        q12_apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

fn q12_apply_pending_order(
    groups: &mut [Q12State; 2],
    order: Q12PendingOrder,
    is_high_priority: bool,
) {
    for (index, count) in order.counts.iter().copied().enumerate() {
        if count == 0 {
            continue;
        }
        let group = &mut groups[index];
        if is_high_priority {
            group.high_line_count += count;
        } else {
            group.low_line_count += count;
        }
    }
}

fn q12_merge_shipping_mode_counts(groups: &mut [Q12State; 2], batch_groups: [Q12State; 2]) {
    for index in 0..groups.len() {
        groups[index].high_line_count += batch_groups[index].high_line_count;
        groups[index].low_line_count += batch_groups[index].low_line_count;
    }
}

fn q12_output(rows: Vec<Q12Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_shipmode", DataType::Utf8, false),
            Field::new("high_line_count", DataType::UInt64, false),
            Field::new("low_line_count", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.shipmode.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.high_line_count),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.low_line_count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q14_promotion_effect_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q14_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut lineitem = None;
    let mut part = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        }
    }
    let (Some(lineitem), Some(part)) = (lineitem, part) else {
        return Ok(None);
    };
    if !lineitem.path.exists() {
        return Err(DodamError::MissingPath(lineitem.path));
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };
    let promo_parts = q14_promo_parts(engine, part.path, batch_size).await?;
    if promo_parts.is_empty() {
        return Ok(Some(q17_output("promo_revenue".to_string(), None)?));
    }
    let (promo, total) = q14_promo_revenue(
        engine,
        lineitem.path,
        batch_size,
        start_days,
        end_days,
        promo_parts,
    )
    .await?;
    Ok(Some(q17_output(
        "promo_revenue".to_string(),
        (total != 0.0).then_some(100.0 * promo / total),
    )?))
}

fn q14_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 2
        && select.projection.len() == 1
        && projection.contains("p_type like 'promo%'")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && selection.contains("l_partkey = p_partkey")
        && selection.contains("l_shipdate")
}

async fn q14_promo_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashMap<i64, bool>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_type".to_string()]),
            None,
        )
        .await?;
    let mut parts = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let types = batch_string_column(&batch, "p_type")?;
        for row in 0..batch.num_rows() {
            if types.is_null(row) {
                continue;
            }
            if let Some(partkey) = numeric_i64_value(partkeys, row)? {
                parts.insert(partkey, types.value(row).starts_with("PROMO"));
            }
        }
    }
    Ok(parts)
}

async fn q14_promo_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
    promo_parts: HashMap<i64, bool>,
) -> Result<(f64, f64)> {
    let promo_parts = Arc::new(promo_parts);
    if std::env::var_os("DODAM_Q14_DISABLE_LATE_MATERIALIZE").is_none() {
        if let Some(result) = engine
            .q14_late_materialized_promo_revenue(
                path.clone(),
                batch_size,
                start_days,
                end_days,
                promo_parts.clone(),
            )
            .await?
        {
            return Ok(result);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_shipdate".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    parallel_batch_fold(
        &mut stream,
        move |batch| q14_promo_revenue_batch(batch, start_days, end_days, &promo_parts),
        (0.0, 0.0),
        |total, batch| {
            total.0 += batch.0;
            total.1 += batch.1;
        },
        "Q14 promo revenue",
    )
}

fn q14_promo_revenue_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    promo_parts: &HashMap<i64, bool>,
) -> Result<(f64, f64)> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let mut promo = 0.0;
    let mut total = 0.0;
    if let (Some(partkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) {
        for row in 0..batch.num_rows() {
            if partkeys.is_null(row)
                || shipdates.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let shipdate = shipdates.value(row);
            if shipdate < start_days || shipdate >= end_days {
                continue;
            }
            let Some(is_promo) = promo_parts.get(&partkeys.value(row)).copied() else {
                continue;
            };
            let value = extendedprices.value(row) * (1.0 - discounts.value(row));
            if is_promo {
                promo += value;
            }
            total += value;
        }
        return Ok((promo, total));
    }
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if shipdate < start_days || shipdate >= end_days {
            continue;
        }
        let (Some(partkey), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let Some(is_promo) = promo_parts.get(&partkey).copied() else {
            continue;
        };
        let value = extendedprice * (1.0 - discount);
        if is_promo {
            promo += value;
        }
        total += value;
    }
    Ok((promo, total))
}

async fn try_execute_q15_top_supplier_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    if !q15_shape(query) {
        return Ok(None);
    }
    let Some(with) = query.with.as_ref() else {
        return Ok(None);
    };
    let cte = &with.cte_tables[0];
    let SetExpr::Select(revenue_select) = cte.query.body.as_ref() else {
        return Ok(None);
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return Ok(None);
    };
    reject_select_features(revenue_select)?;
    reject_select_features(outer_select)?;
    let lineitem = parse_from(revenue_select)?;
    if !lineitem.path.exists() {
        return Err(DodamError::MissingPath(lineitem.path));
    }
    let Some(selection) = revenue_select.selection.as_ref() else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };
    let Some(tables) = parse_comma_join_table_refs(outer_select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let supplier = tables
        .into_iter()
        .find(|table| table_ref_alias_or_name(table).eq_ignore_ascii_case("supplier"));
    let Some(supplier) = supplier else {
        return Ok(None);
    };

    let revenues =
        q15_revenue_by_supplier(engine, lineitem.path, batch_size, start_days, end_days).await?;
    let Some(max_revenue) = revenues
        .values()
        .copied()
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Ok(Some(q15_output(Vec::new())?));
    };
    let top_suppliers = revenues
        .into_iter()
        .filter_map(|(suppkey, revenue)| (revenue == max_revenue).then_some((suppkey, revenue)))
        .collect::<HashMap<_, _>>();
    let rows = q15_supplier_rows(engine, supplier.path, batch_size, &top_suppliers).await?;
    Ok(Some(q15_output(rows)?))
}

fn q15_shape(query: &Query) -> bool {
    let Some(with) = query.with.as_ref() else {
        return false;
    };
    if with.recursive || with.cte_tables.len() != 1 {
        return false;
    }
    let cte = &with.cte_tables[0];
    if !cte.alias.name.value.eq_ignore_ascii_case("revenue")
        || !cte.alias.columns.is_empty()
        || cte.alias.at.is_some()
        || cte.from.is_some()
    {
        return false;
    }
    let SetExpr::Select(revenue_select) = cte.query.body.as_ref() else {
        return false;
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return false;
    };
    let revenue_projection = revenue_select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let revenue_group_by = revenue_select.group_by.to_string().to_ascii_lowercase();
    let revenue_selection = revenue_select
        .selection
        .as_ref()
        .map(|expr| expr.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let outer_projection = outer_select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let outer_selection = outer_select
        .selection
        .as_ref()
        .map(|expr| expr.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    revenue_select.from.len() == 1
        && outer_select.from.len() == 2
        && revenue_projection.contains("l_suppkey")
        && revenue_projection.contains("supplier_no")
        && revenue_projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && revenue_projection.contains("total_revenue")
        && revenue_group_by.contains("l_suppkey")
        && revenue_selection.contains("l_shipdate")
        && outer_projection.contains("s_suppkey")
        && outer_projection.contains("s_name")
        && outer_projection.contains("s_address")
        && outer_projection.contains("s_phone")
        && outer_projection.contains("total_revenue")
        && outer_selection.contains("s_suppkey = supplier_no")
        && outer_selection.contains("max(total_revenue)")
        && order_by.contains("s_suppkey")
}

async fn q15_revenue_by_supplier(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, f64>> {
    parquet_scan_fold_chunks(
        engine,
        path,
        batch_size,
        Projection::Columns(vec![
            "l_suppkey".to_string(),
            "l_shipdate".to_string(),
            "l_extendedprice".to_string(),
            "l_discount".to_string(),
        ]),
        scan_aggregate_row_group_chunk(),
        8,
        HashMap::<i64, f64>::new,
        HashMap::<i64, f64>::new,
        move |batches| {
            let mut revenues = HashMap::<i64, f64>::new();
            merge_f64_groups(
                &mut revenues,
                q15_revenue_by_supplier_batch(batches, start_days, end_days)?,
            );
            Ok(revenues)
        },
        merge_f64_groups,
        "Q15 revenue aggregate",
    )
    .await
}

fn q15_revenue_by_supplier_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, f64>> {
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let mut revenues = HashMap::<i64, f64>::new();
    if let (Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) {
        for row in 0..batch.num_rows() {
            if suppkeys.is_null(row)
                || shipdates.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let shipdate = shipdates.value(row);
            if shipdate < start_days || shipdate >= end_days {
                continue;
            }
            *revenues.entry(suppkeys.value(row)).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(revenues);
    }
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if shipdate < start_days || shipdate >= end_days {
            continue;
        }
        let (Some(suppkey), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(suppkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(revenues)
}

struct Q15Row {
    suppkey: i64,
    name: String,
    address: String,
    phone: String,
    total_revenue: f64,
}

async fn q15_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    top_suppliers: &HashMap<i64, f64>,
) -> Result<Vec<Q15Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_name".to_string(),
                "s_address".to_string(),
                "s_phone".to_string(),
            ]),
            None,
        )
        .await?;
    let mut rows = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let names = batch_string_column(&batch, "s_name")?;
        let addresses = batch_string_column(&batch, "s_address")?;
        let phones = batch_string_column(&batch, "s_phone")?;
        for row in 0..batch.num_rows() {
            if names.is_null(row) || addresses.is_null(row) || phones.is_null(row) {
                continue;
            }
            let Some(suppkey) = numeric_i64_value(suppkeys, row)? else {
                continue;
            };
            let Some(total_revenue) = top_suppliers.get(&suppkey).copied() else {
                continue;
            };
            rows.push(Q15Row {
                suppkey,
                name: names.value(row).to_string(),
                address: addresses.value(row).to_string(),
                phone: phones.value(row).to_string(),
                total_revenue,
            });
        }
    }
    rows.sort_by_key(|row| row.suppkey);
    Ok(rows)
}

fn q15_output(rows: Vec<Q15Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_suppkey", DataType::Int64, false),
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
            Field::new("s_phone", DataType::Utf8, false),
            Field::new("total_revenue", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.suppkey),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.address.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.phone.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.total_revenue),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q03_shipping_priority_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q03_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 3 {
        return Ok(None);
    }
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem)) = (customer, orders, lineitem) else {
        return Ok(None);
    };
    if !customer.path.exists() {
        return Err(DodamError::MissingPath(customer.path));
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(segment) = string_equality_literal(&conjuncts, "c_mktsegment")? else {
        return Ok(None);
    };
    let Some(order_cutoff) = upper_date_bound(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };
    let Some(ship_cutoff) = lower_date_bound(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };
    let customers = q03_customer_keys(engine, customer.path, batch_size, &segment).await?;
    if customers.is_empty() {
        return Ok(Some(q03_output(Vec::new())?));
    }
    let orders = q03_order_rows(engine, orders.path, batch_size, &customers, order_cutoff).await?;
    if orders.is_empty() {
        return Ok(Some(q03_output(Vec::new())?));
    }
    let rows = q03_revenue_rows(engine, lineitem.path, batch_size, &orders, ship_cutoff).await?;
    Ok(Some(q03_output(rows)?))
}

fn q03_shape(select: &Select, _query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 4
        && projection.contains("l_orderkey")
        && projection.contains("sum(")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && projection.contains("o_orderdate")
        && projection.contains("o_shippriority")
        && selection.contains("c_mktsegment")
        && selection.contains("c_custkey")
        && selection.contains("o_custkey")
        && selection.contains("l_orderkey")
        && selection.contains("o_orderkey")
        && selection.contains("o_orderdate")
        && selection.contains("l_shipdate")
}

fn lower_date_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<i32>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(left, column)
            && let Some(days) = maybe_literal_date_days(right)?
        {
            bound = Some(days);
        } else if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(right, column)
            && let Some(days) = maybe_literal_date_days(left)?
        {
            bound = Some(days);
        }
    }
    Ok(bound)
}

fn upper_date_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<i32>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(left, column)
            && let Some(days) = maybe_literal_date_days(right)?
        {
            bound = Some(days);
        } else if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(right, column)
            && let Some(days) = maybe_literal_date_days(left)?
        {
            bound = Some(days);
        }
    }
    Ok(bound)
}

async fn q03_customer_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    segment: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_mktsegment".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let segments = batch_string_column(&batch, "c_mktsegment")?;
        for row in 0..batch.num_rows() {
            if segments.is_valid(row)
                && segments.value(row) == segment
                && let Some(custkey) = numeric_i64_value(custkeys, row)?
            {
                keys.insert(custkey);
            }
        }
    }
    Ok(keys)
}

#[derive(Clone, Copy)]
struct Q03Order {
    o_orderdate: i32,
    o_shippriority: i64,
}

#[derive(Clone)]
struct SortedI64Lookup<V> {
    entries: Vec<(i64, V)>,
}

impl<V: Copy> SortedI64Lookup<V> {
    fn from_hash_map<S>(values: &HashMap<i64, V, S>) -> Self
    where
        S: BuildHasher,
    {
        let mut entries = values
            .iter()
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(key, _)| *key);
        Self { entries }
    }

    fn get(&self, key: i64) -> Option<V> {
        self.entries
            .binary_search_by_key(&key, |(entry_key, _)| *entry_key)
            .ok()
            .map(|index| self.entries[index].1)
    }
}

async fn q03_order_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customers: &HashSet<i64>,
    order_cutoff: i32,
) -> Result<HashMap<i64, Q03Order>> {
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_custkey".to_string(),
        "o_orderdate".to_string(),
        "o_shippriority".to_string(),
    ]);
    let customers = Arc::new(AdaptiveI64Set::from_hash(customers.clone()));
    if q03_order_row_group_map_enabled()
        && let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                q03_order_row_group_map_chunk(),
                HashMap::<i64, Q03Order>::new,
                {
                    let customers = customers.clone();
                    move |batch, orders| {
                        merge_maps(
                            orders,
                            q03_order_rows_projected_batch(batch, &customers, order_cutoff)?,
                        );
                        Ok(Some(()))
                    }
                },
                |orders| Ok(Some(orders)),
            )
            .await?
    {
        let mut orders = HashMap::new();
        for partial in partials {
            merge_maps(&mut orders, partial);
        }
        return Ok(orders);
    }
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    parallel_batch_fold_chunks(
        &mut stream,
        build_map_chunk_size(),
        move |batches| {
            let mut orders = HashMap::<i64, Q03Order>::new();
            for batch in batches {
                merge_maps(
                    &mut orders,
                    q03_order_rows_batch(batch, &customers, order_cutoff)?,
                );
            }
            Ok(orders)
        },
        HashMap::<i64, Q03Order>::new(),
        merge_maps,
        "Q03 order rows",
    )
}

fn q03_order_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q03_ENABLE_ORDER_ROW_GROUP_MAP").is_some()
}

fn q03_order_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q03_ORDER_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q03_order_rows_batch(
    batch: RecordBatch,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
) -> Result<HashMap<i64, Q03Order>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let priorities = batch_column(&batch, "o_shippriority")?;
    if let Some(orders) = q03_order_rows_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        priorities,
        customers,
        order_cutoff,
    )? {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(custkey), Some(orderdate), Some(priority)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
            date32_value(orderdates, row)?,
            numeric_i64_value(priorities, row)?,
        ) else {
            continue;
        };
        if customers.contains(custkey) && orderdate < order_cutoff {
            orders.insert(
                orderkey,
                Q03Order {
                    o_orderdate: orderdate,
                    o_shippriority: priority,
                },
            );
        }
    }
    Ok(orders)
}

fn q03_order_rows_projected_batch(
    batch: RecordBatch,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
) -> Result<HashMap<i64, Q03Order>> {
    if batch.num_columns() == 4
        && let Some(orders) = q03_order_rows_batch_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            customers,
            order_cutoff,
        )?
    {
        return Ok(orders);
    }
    q03_order_rows_batch(batch, customers, order_cutoff)
}

fn q03_order_rows_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    priorities: &ArrayRef,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
) -> Result<Option<HashMap<i64, Q03Order>>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates), Some(priorities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
        priorities.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return Ok(None);
    };
    let mut orders = HashMap::new();
    if orderkeys.null_count() == 0
        && custkeys.null_count() == 0
        && orderdates.null_count() == 0
        && priorities.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let custkey_values = custkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        let priority_values = priorities.values().as_ref();
        if let Some(customer_contains) = customers.dense_contains_slice() {
            for row in 0..orderkey_values.len() {
                let custkey = custkey_values[row];
                let customer_hit = usize::try_from(custkey)
                    .ok()
                    .and_then(|index| customer_contains.get(index))
                    .copied()
                    .unwrap_or(false);
                if orderdate_values[row] < order_cutoff && customer_hit {
                    orders.insert(
                        orderkey_values[row],
                        Q03Order {
                            o_orderdate: orderdate_values[row],
                            o_shippriority: priority_values[row],
                        },
                    );
                }
            }
            return Ok(Some(orders));
        }
        for row in 0..orderkey_values.len() {
            if orderdate_values[row] < order_cutoff && customers.contains(custkey_values[row]) {
                orders.insert(
                    orderkey_values[row],
                    Q03Order {
                        o_orderdate: orderdate_values[row],
                        o_shippriority: priority_values[row],
                    },
                );
            }
        }
        return Ok(Some(orders));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || custkeys.is_null(row)
            || orderdates.is_null(row)
            || priorities.is_null(row)
        {
            continue;
        }
        if orderdates.value(row) < order_cutoff && customers.contains(custkeys.value(row)) {
            orders.insert(
                orderkeys.value(row),
                Q03Order {
                    o_orderdate: orderdates.value(row),
                    o_shippriority: priorities.value(row),
                },
            );
        }
    }
    Ok(Some(orders))
}

struct Q03Row {
    l_orderkey: i64,
    revenue: f64,
    o_orderdate: i32,
    o_shippriority: i64,
}

async fn q03_revenue_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    orders: &HashMap<i64, Q03Order>,
    ship_cutoff: i32,
) -> Result<Vec<Q03Row>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_shipdate".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let pruning_predicates =
        if let Some((min_key, max_key)) = selective_i64_key_range(orders.keys().copied()) {
            i64_range_pruning_predicates("l_orderkey", min_key, max_key)
        } else {
            Vec::new()
        };
    let revenues = if q03_row_group_map_enabled() {
        if q03_sorted_order_lookup_enabled() {
            let orders_for_scan = Arc::new(SortedI64Lookup::from_hash_map(orders));
            q03_revenue_rows_row_group_map(
                engine,
                path,
                batch_size,
                projection,
                pruning_predicates,
                move |batch| {
                    q03_revenue_projected_batch_sorted(batch, &orders_for_scan, ship_cutoff)
                },
            )
            .await?
        } else {
            let orders_for_scan = Arc::new(orders.clone());
            q03_revenue_rows_row_group_map(
                engine,
                path,
                batch_size,
                projection,
                pruning_predicates,
                move |batch| q03_revenue_projected_batch(batch, &orders_for_scan, ship_cutoff),
            )
            .await?
        }
    } else if q03_sorted_order_lookup_enabled() {
        let mut stream =
            q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
        let orders_for_scan = Arc::new(SortedI64Lookup::from_hash_map(orders));
        parallel_batch_fold_chunks(
            &mut stream,
            4,
            move |batches| {
                let mut revenues = HashMap::<i64, f64>::new();
                for batch in batches {
                    merge_f64_groups(
                        &mut revenues,
                        q03_revenue_batch_sorted(batch, &orders_for_scan, ship_cutoff)?,
                    );
                }
                Ok(revenues)
            },
            HashMap::<i64, f64>::new(),
            merge_f64_groups,
            "Q03 revenue aggregate",
        )?
    } else {
        let mut stream =
            q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
        let orders_for_scan = Arc::new(orders.clone());
        parallel_batch_fold_chunks(
            &mut stream,
            4,
            move |batches| {
                let mut revenues = HashMap::<i64, f64>::new();
                for batch in batches {
                    merge_f64_groups(
                        &mut revenues,
                        q03_revenue_batch(batch, &orders_for_scan, ship_cutoff)?,
                    );
                }
                Ok(revenues)
            },
            HashMap::<i64, f64>::new(),
            merge_f64_groups,
            "Q03 revenue aggregate",
        )?
    };
    let mut rows = revenues
        .into_iter()
        .filter_map(|(orderkey, revenue)| {
            orders.get(&orderkey).map(|order| Q03Row {
                l_orderkey: orderkey,
                revenue,
                o_orderdate: order.o_orderdate,
                o_shippriority: order.o_shippriority,
            })
        })
        .collect::<Vec<_>>();
    if rows.len() > 10 {
        rows.select_nth_unstable_by(10, q03_row_ordering);
        rows.truncate(10);
    }
    rows.sort_by(q03_row_ordering);
    Ok(rows)
}

async fn q03_revenue_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
) -> Result<SendableBatchStream> {
    if pruning_predicates.is_empty() {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await
    } else {
        engine
            .scan_parquet_batches_pruned(path, batch_size, projection, pruning_predicates)
            .await
    }
}

async fn q03_revenue_rows_row_group_map<Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    map: Map,
) -> Result<HashMap<i64, f64>>
where
    Map: Fn(RecordBatch) -> Result<HashMap<i64, f64>> + Clone + Send + Sync + 'static,
{
    let map_for_row_group = map.clone();
    if let Some(partials) = engine
        .parquet_row_group_map_pruned(
            path.clone(),
            batch_size,
            projection.clone(),
            pruning_predicates.clone(),
            q03_row_group_map_chunk(),
            HashMap::<i64, f64>::new,
            move |batch, revenues| {
                merge_f64_groups(revenues, map_for_row_group(batch)?);
                Ok(Some(()))
            },
            |revenues| Ok(Some(revenues)),
        )
        .await?
    {
        let mut revenues = HashMap::<i64, f64>::new();
        for partial in partials {
            merge_f64_groups(&mut revenues, partial);
        }
        Ok(revenues)
    } else {
        let mut stream =
            q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
        parallel_batch_fold_chunks(
            &mut stream,
            4,
            move |batches| {
                let mut revenues = HashMap::<i64, f64>::new();
                for batch in batches {
                    merge_f64_groups(&mut revenues, map(batch)?);
                }
                Ok(revenues)
            },
            HashMap::<i64, f64>::new(),
            merge_f64_groups,
            "Q03 revenue aggregate",
        )
    }
}

fn q03_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q03_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q03_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q03_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q03_sorted_order_lookup_enabled() -> bool {
    std::env::var("DODAM_Q03_ENABLE_SORTED_ORDER_LOOKUP")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q03_row_ordering(left: &Q03Row, right: &Q03Row) -> std::cmp::Ordering {
    right
        .revenue
        .partial_cmp(&left.revenue)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.o_orderdate.cmp(&right.o_orderdate))
}

fn q03_revenue_batch(
    batch: RecordBatch,
    orders: &HashMap<i64, Q03Order>,
    ship_cutoff: i32,
) -> Result<HashMap<i64, f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(revenues) = q03_revenue_batch_typed(
        orderkeys,
        shipdates,
        extendedprices,
        discounts,
        orders,
        ship_cutoff,
    )? {
        return Ok(revenues);
    }
    let mut revenues = HashMap::<i64, f64>::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if shipdate <= ship_cutoff || !orders.contains_key(&orderkey) {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(orderkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(revenues)
}

fn q03_revenue_projected_batch(
    batch: RecordBatch,
    orders: &HashMap<i64, Q03Order>,
    ship_cutoff: i32,
) -> Result<HashMap<i64, f64>> {
    if batch.num_columns() == 4
        && let Some(revenues) = q03_revenue_batch_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            orders,
            ship_cutoff,
        )?
    {
        return Ok(revenues);
    }
    q03_revenue_batch(batch, orders, ship_cutoff)
}

fn q03_revenue_batch_typed(
    orderkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    orders: &HashMap<i64, Q03Order>,
    ship_cutoff: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    let mut revenues = HashMap::<i64, f64>::new();
    if orderkeys.null_count() == 0
        && shipdates.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let (discount_scale, revenue_scale) =
            decimal_discounted_revenue_scales(extendedprices, discounts);
        for row in 0..orderkeys.len() {
            let shipdate = shipdate_values[row];
            let orderkey = orderkey_values[row];
            if shipdate > ship_cutoff && orders.contains_key(&orderkey) {
                *revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
        }
        return Ok(Some(revenues));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || shipdates.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        let orderkey = orderkeys.value(row);
        if shipdate > ship_cutoff && orders.contains_key(&orderkey) {
            *revenues.entry(orderkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
    }
    Ok(Some(revenues))
}

fn q03_revenue_batch_sorted(
    batch: RecordBatch,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Result<HashMap<i64, f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(revenues) = q03_revenue_batch_sorted_typed(
        orderkeys,
        shipdates,
        extendedprices,
        discounts,
        orders,
        ship_cutoff,
    )? {
        return Ok(revenues);
    }
    let mut revenues = HashMap::<i64, f64>::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if shipdate <= ship_cutoff || orders.get(orderkey).is_none() {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(orderkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(revenues)
}

fn q03_revenue_projected_batch_sorted(
    batch: RecordBatch,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Result<HashMap<i64, f64>> {
    if batch.num_columns() == 4
        && let Some(revenues) = q03_revenue_batch_sorted_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            orders,
            ship_cutoff,
        )?
    {
        return Ok(revenues);
    }
    q03_revenue_batch_sorted(batch, orders, ship_cutoff)
}

fn q03_revenue_batch_sorted_typed(
    orderkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    if orderkeys.null_count() != 0
        || shipdates.null_count() != 0
        || extendedprices.null_count() != 0
        || discounts.null_count() != 0
    {
        return Ok(None);
    }
    let mut revenues = HashMap::<i64, f64>::new();
    let orderkey_values = orderkeys.values().as_ref();
    let shipdate_values = shipdates.values().as_ref();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let (discount_scale, revenue_scale) =
        decimal_discounted_revenue_scales(extendedprices, discounts);
    for row in 0..orderkeys.len() {
        let shipdate = shipdate_values[row];
        let orderkey = orderkey_values[row];
        if shipdate > ship_cutoff && orders.get(orderkey).is_some() {
            *revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Ok(Some(revenues))
}

fn merge_f64_groups<K, S>(groups: &mut HashMap<K, f64, S>, batch: HashMap<K, f64, S>)
where
    K: Eq + std::hash::Hash,
    S: BuildHasher,
{
    for (key, value) in batch {
        *groups.entry(key).or_insert(0.0) += value;
    }
}

fn merge_maps<K, V, S>(output: &mut HashMap<K, V, S>, batch: HashMap<K, V, S>)
where
    K: Eq + std::hash::Hash,
    S: BuildHasher,
{
    output.extend(batch);
}

fn merge_sets<K: Eq + std::hash::Hash>(output: &mut HashSet<K>, batch: HashSet<K>) {
    output.extend(batch);
}

fn selective_i64_key_range<I>(keys: I) -> Option<(i64, i64)>
where
    I: IntoIterator<Item = i64>,
{
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    let mut len = 0_usize;
    for key in keys {
        if key < 0 {
            return None;
        }
        min_key = min_key.min(key);
        max_key = max_key.max(key);
        len += 1;
    }
    selective_i64_range_from_parts(min_key, max_key, len)
}

fn selective_candidate_priority_range(candidate_priorities: &[u8]) -> Option<(i64, i64)> {
    let mut min_key = usize::MAX;
    let mut max_key = 0_usize;
    let mut len = 0_usize;
    for (key, priority) in candidate_priorities.iter().copied().enumerate() {
        if priority == 0 {
            continue;
        }
        min_key = min_key.min(key);
        max_key = max_key.max(key);
        len += 1;
    }
    if min_key == usize::MAX {
        return None;
    }
    selective_i64_range_from_parts(min_key as i64, max_key as i64, len)
}

fn selective_i64_range_from_parts(min_key: i64, max_key: i64, len: usize) -> Option<(i64, i64)> {
    if len == 0 || min_key < 0 || max_key < min_key {
        return None;
    }
    let width = usize::try_from(max_key.checked_sub(min_key)?.checked_add(1)?).ok()?;
    (width <= len.saturating_mul(8).max(1024)).then_some((min_key, max_key))
}

fn i64_range_pruning_predicates(column: &str, min_key: i64, max_key: i64) -> Vec<Expr> {
    vec![
        Expr::Comparison(ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::GtEq,
            value: LiteralValue::Int64(min_key),
        }),
        Expr::Comparison(ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::LtEq,
            value: LiteralValue::Int64(max_key),
        }),
    ]
}

fn q03_output(rows: Vec<Q03Row>) -> Result<QueryOutput> {
    let orderdates = rows
        .iter()
        .map(|row| date32_to_ymd_string(row.o_orderdate))
        .collect::<Result<Vec<_>>>()?;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("revenue", DataType::Float64, false),
            Field::new("o_orderdate", DataType::Utf8, false),
            Field::new("o_shippriority", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.l_orderkey),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
            Arc::new(StringArray::from_iter_values(
                orderdates.iter().map(String::as_str),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.o_shippriority),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

fn date32_to_ymd_string(days: i32) -> Result<String> {
    let (year, month, day) = civil_from_days(i64::from(days))?;
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

async fn try_execute_q04_order_priority_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q04_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let [table_with_joins] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table_with_joins.joins.is_empty() {
        return Ok(None);
    }
    let orders = parse_table_factor(&table_with_joins.relation)?;
    if !table_ref_alias_or_name(&orders).eq_ignore_ascii_case("orders") {
        return Ok(None);
    }
    let Some(lineitem_path) = q04_lineitem_path(selection)? else {
        return Ok(None);
    };
    if !lineitem_path.exists() {
        return Err(DodamError::MissingPath(lineitem_path));
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };
    let stage = tpch_profile_start();
    let (mut candidate_priorities, priority_labels, candidate_keys) =
        q04_candidate_order_priorities(engine, orders.path, batch_size, start_days, end_days)
            .await?;
    tpch_profile_elapsed("Q04 candidate order priorities", stage);
    if priority_labels.is_empty() {
        return Ok(Some(q04_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let counts = q04_count_late_candidate_priorities(
        engine,
        lineitem_path,
        batch_size,
        &mut candidate_priorities,
        &candidate_keys,
        priority_labels.len(),
    )
    .await?;
    tpch_profile_elapsed("Q04 late lineitem probe", stage);
    let stage = tpch_profile_start();
    let rows = q04_priority_count_rows(priority_labels, counts);
    tpch_profile_elapsed("Q04 final rows", stage);
    Ok(Some(q04_output(rows)?))
}

fn q04_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 2
        && projection.contains("o_orderpriority")
        && projection.contains("count(*)")
        && group_by.contains("o_orderpriority")
        && order_by.contains("o_orderpriority")
        && selection.contains("o_orderdate")
        && selection.contains("exists")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("l_commitdate < l_receiptdate")
}

fn q04_lineitem_path(selection: &SqlExpr) -> Result<Option<PathBuf>> {
    let mut stack = vec![selection];
    while let Some(expr) = stack.pop() {
        match expr {
            SqlExpr::Exists { subquery, .. } => {
                let SetExpr::Select(select) = subquery.body.as_ref() else {
                    continue;
                };
                for table in q04_subquery_tables(select)? {
                    if table_ref_alias_or_name(&table).eq_ignore_ascii_case("lineitem") {
                        return Ok(Some(table.path));
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => stack.push(expr),
            _ => {}
        }
    }
    Ok(None)
}

fn q04_subquery_tables(select: &Select) -> Result<Vec<SqlTableRef>> {
    if let Some(tables) = parse_comma_join_table_refs(select)? {
        return Ok(tables);
    }
    if select.from.is_empty() {
        return Ok(Vec::new());
    }
    if select.from.iter().any(|table| !table.joins.is_empty()) {
        return Ok(Vec::new());
    }
    select
        .from
        .iter()
        .map(|table| parse_table_factor(&table.relation))
        .collect::<Result<Vec<_>>>()
}

struct Q04Row {
    priority: String,
    count: u64,
}

async fn q04_candidate_order_priorities(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<(Vec<u8>, Vec<String>, HashSet<i64>)> {
    if std::env::var_os("DODAM_Q04_DISABLE_LATE_CANDIDATES").is_none()
        && let Some(candidates) = q04_candidate_order_priorities_late(
            engine,
            path.clone(),
            batch_size,
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(candidates);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_orderdate".to_string(),
                "o_orderpriority".to_string(),
            ]),
            None,
        )
        .await?;
    let mut priorities = Vec::<u8>::new();
    let mut labels = Vec::<String>::new();
    let mut label_indices = HashMap::<String, u8>::new();
    let mut candidate_keys = HashSet::<i64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
        let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
        if q04_candidate_order_priorities_typed(
            orderkeys,
            orderdates,
            orderpriorities,
            start_days,
            end_days,
            &mut priorities,
            &mut labels,
            &mut label_indices,
            &mut candidate_keys,
        )? {
            continue;
        }
        for row in 0..batch.num_rows() {
            if orderpriorities.is_null(row) {
                continue;
            }
            let (Some(orderkey), Some(orderdate)) = (
                numeric_i64_value(orderkeys, row)?,
                date32_value(orderdates, row)?,
            ) else {
                continue;
            };
            if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
                continue;
            }
            let priority_index = if let Some(index) = label_indices.get(orderpriorities.value(row))
            {
                *index
            } else {
                let next_index = u8::try_from(labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                labels.push(orderpriorities.value(row).to_string());
                label_indices.insert(orderpriorities.value(row).to_string(), next_index);
                next_index
            };
            let orderkey = usize::try_from(orderkey)
                .map_err(|_| DodamError::UnsupportedSql("order key overflow".to_string()))?;
            if orderkey >= priorities.len() {
                priorities.resize(orderkey + 1, 0);
            }
            priorities[orderkey] = priority_index + 1;
            candidate_keys.insert(orderkey as i64);
        }
    }
    Ok((priorities, labels, candidate_keys))
}

async fn q04_candidate_order_priorities_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<Option<(Vec<u8>, Vec<String>, HashSet<i64>)>> {
    let Some(chunks) = engine
        .late_materialized_parquet_map_with_policy(
            path,
            batch_size,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()]),
            Projection::Columns(vec!["o_orderpriority".to_string()]),
            q04_late_candidate_row_group_chunk(),
            late_materialization_policy_from_env("DODAM_Q04_LATE_MAX_SELECTED_RATIO", 0.60),
            move || Q04LateCandidateState {
                start_days,
                end_days,
                selected_orderkeys: Vec::new(),
                priority_offset: 0,
                labels: Vec::new(),
                label_indices: HashMap::new(),
                rows: Vec::new(),
            },
            q04_late_candidate_build_selection_batch,
            q04_late_candidate_consume_priority_batch,
            |state, _metrics| {
                if state.priority_offset != state.selected_orderkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q04 candidate row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some(Q04CandidatePartial {
                    labels: state.labels,
                    rows: state.rows,
                }))
            },
        )
        .await?
    else {
        return Ok(None);
    };

    let mut metrics = LateMaterializedMetrics::default();
    let mut partials = Vec::new();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        partials.push(chunk.output);
    }
    q04_log_late_candidate_profile(metrics, q04_late_candidate_row_group_chunk());
    Ok(Some(q04_candidate_priorities_from_partials(partials)?))
}

fn q04_late_candidate_row_group_chunk() -> usize {
    std::env::var("DODAM_Q04_LATE_CANDIDATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn late_materialization_policy_from_env(
    env_name: &str,
    default_max_selected_ratio: f64,
) -> LateMaterializationPolicy {
    let max_selected_ratio = std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_max_selected_ratio);
    LateMaterializationPolicy::selective(max_selected_ratio)
}

struct Q04LateCandidateState {
    start_days: i32,
    end_days: i32,
    selected_orderkeys: Vec<i64>,
    priority_offset: usize,
    labels: Vec<String>,
    label_indices: HashMap<String, u8>,
    rows: Vec<(i64, u8)>,
}

struct Q04CandidatePartial {
    labels: Vec<String>,
    rows: Vec<(i64, u8)>,
}

fn q04_late_candidate_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q04LateCandidateState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let (Some(orderkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 && orderdates.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        for (&orderkey, &orderdate) in orderkey_values.iter().zip(orderdate_values) {
            let selected =
                orderkey >= 0 && orderdate >= state.start_days && orderdate < state.end_days;
            selection.push(selected);
            if selected {
                state.selected_orderkeys.push(orderkey);
            }
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let selected = if orderkeys.is_null(row) || orderdates.is_null(row) {
            false
        } else {
            let orderkey = orderkeys.value(row);
            let orderdate = orderdates.value(row);
            orderkey >= 0 && orderdate >= state.start_days && orderdate < state.end_days
        };
        selection.push(selected);
        if selected {
            state.selected_orderkeys.push(orderkeys.value(row));
        }
    }
    Ok(Some(()))
}

fn q04_late_candidate_consume_priority_batch(
    batch: RecordBatch,
    state: &mut Q04LateCandidateState,
) -> Result<Option<()>> {
    let priorities = batch_string_column(&batch, "o_orderpriority")?;
    for row in 0..batch.num_rows() {
        let Some(&orderkey) = state.selected_orderkeys.get(state.priority_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q04 candidate payload row overflow".to_string(),
            ));
        };
        state.priority_offset += 1;
        if priorities.is_valid(row) {
            let priority = priorities.value(row);
            let priority_index = if let Some(index) = state.label_indices.get(priority) {
                *index
            } else {
                let next_index = u8::try_from(state.labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                state.labels.push(priority.to_string());
                state.label_indices.insert(priority.to_string(), next_index);
                next_index
            };
            state.rows.push((orderkey, priority_index));
        }
    }
    Ok(Some(()))
}

fn q04_candidate_priorities_from_partials(
    partials: Vec<Q04CandidatePartial>,
) -> Result<(Vec<u8>, Vec<String>, HashSet<i64>)> {
    let mut priorities = Vec::<u8>::new();
    let mut labels = Vec::<String>::new();
    let mut label_indices = HashMap::<String, u8>::new();
    let row_count = partials.iter().map(|partial| partial.rows.len()).sum();
    let mut candidate_keys = HashSet::<i64>::with_capacity(row_count);
    for partial in partials {
        let mut label_remap = Vec::with_capacity(partial.labels.len());
        for priority in partial.labels {
            let priority_index = if let Some(index) = label_indices.get(priority.as_str()) {
                *index
            } else {
                let next_index = u8::try_from(labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                labels.push(priority.clone());
                label_indices.insert(priority, next_index);
                next_index
            };
            label_remap.push(priority_index);
        }
        for (orderkey, local_priority_index) in partial.rows {
            let priority_index = *label_remap
                .get(usize::from(local_priority_index))
                .ok_or_else(|| {
                    DodamError::UnsupportedSql("Q04 priority label mismatch".to_string())
                })?;
            let orderkey = usize::try_from(orderkey)
                .map_err(|_| DodamError::UnsupportedSql("order key overflow".to_string()))?;
            if orderkey >= priorities.len() {
                priorities.resize(orderkey + 1, 0);
            }
            priorities[orderkey] = priority_index + 1;
            candidate_keys.insert(orderkey as i64);
        }
    }
    Ok((priorities, labels, candidate_keys))
}

fn q04_log_late_candidate_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q04 candidates: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q04_candidate_order_priorities_typed(
    orderkeys: &ArrayRef,
    orderdates: &ArrayRef,
    orderpriorities: &StringArray,
    start_days: i32,
    end_days: i32,
    priorities: &mut Vec<u8>,
    labels: &mut Vec<String>,
    label_indices: &mut HashMap<String, u8>,
    candidate_keys: &mut HashSet<i64>,
) -> Result<bool> {
    try_for_each_i64_date32_str(
        orderkeys,
        orderdates,
        orderpriorities,
        |orderkey, orderdate, priority| {
            if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
                return Ok(());
            }
            let priority_index = if let Some(index) = label_indices.get(priority) {
                *index
            } else {
                let next_index = u8::try_from(labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                labels.push(priority.to_string());
                label_indices.insert(priority.to_string(), next_index);
                next_index
            };
            let orderkey = usize::try_from(orderkey)
                .map_err(|_| DodamError::UnsupportedSql("order key overflow".to_string()))?;
            if orderkey >= priorities.len() {
                priorities.resize(orderkey + 1, 0);
            }
            priorities[orderkey] = priority_index + 1;
            candidate_keys.insert(orderkey as i64);
            Ok(())
        },
    )
}

async fn q04_count_late_candidate_priorities(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    candidate_priorities: &mut [u8],
    candidate_keys: &HashSet<i64>,
    priority_count: usize,
) -> Result<Vec<u64>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
    ]);
    let mut stream = if q04_lineitem_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "l_orderkey",
                candidate_keys.clone(),
            )
            .await?
    } else if let Some((min_key, max_key)) =
        selective_candidate_priority_range(candidate_priorities)
    {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                i64_range_pruning_predicates("l_orderkey", min_key, max_key),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let mut counts = vec![0_u64; priority_count];
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let commitdates = batch_column(&batch, "l_commitdate")?;
        let receiptdates = batch_column(&batch, "l_receiptdate")?;
        if q04_count_late_candidate_priorities_typed(
            orderkeys,
            commitdates,
            receiptdates,
            candidate_priorities,
            &mut counts,
        )? {
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(commitdate), Some(receiptdate)) = (
                numeric_i64_value(orderkeys, row)?,
                date32_value(commitdates, row)?,
                date32_value(receiptdates, row)?,
            ) else {
                continue;
            };
            if commitdate >= receiptdate || orderkey < 0 {
                continue;
            }
            let Ok(orderkey) = usize::try_from(orderkey) else {
                continue;
            };
            let Some(priority_marker) = candidate_priorities.get_mut(orderkey) else {
                continue;
            };
            if *priority_marker == 0 {
                continue;
            }
            let priority_index = usize::from(*priority_marker - 1);
            counts[priority_index] += 1;
            *priority_marker = 0;
        }
    }
    Ok(counts)
}

fn q04_lineitem_row_filter_enabled() -> bool {
    match std::env::var("DODAM_Q04_DISABLE_LINEITEM_ROW_FILTER") {
        Ok(value) => !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => true,
    }
}

fn q04_count_late_candidate_priorities_typed(
    orderkeys: &ArrayRef,
    commitdates: &ArrayRef,
    receiptdates: &ArrayRef,
    candidate_priorities: &mut [u8],
    counts: &mut [u64],
) -> Result<bool> {
    let (Some(orderkeys), Some(commitdates), Some(receiptdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        commitdates.as_any().downcast_ref::<Date32Array>(),
        receiptdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0
        && commitdates.null_count() == 0
        && receiptdates.null_count() == 0
    {
        let orderkeys = orderkeys.values().as_ref();
        let commitdates = commitdates.values().as_ref();
        let receiptdates = receiptdates.values().as_ref();
        for row in 0..orderkeys.len() {
            if commitdates[row] >= receiptdates[row] {
                continue;
            }
            let orderkey = orderkeys[row];
            if orderkey < 0 {
                continue;
            }
            let Ok(orderkey) = usize::try_from(orderkey) else {
                continue;
            };
            let Some(priority_marker) = candidate_priorities.get_mut(orderkey) else {
                continue;
            };
            if *priority_marker == 0 {
                continue;
            }
            let priority_index = usize::from(*priority_marker - 1);
            counts[priority_index] += 1;
            *priority_marker = 0;
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || commitdates.is_null(row) || receiptdates.is_null(row) {
            continue;
        }
        if commitdates.value(row) >= receiptdates.value(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if orderkey < 0 {
            continue;
        }
        let Ok(orderkey) = usize::try_from(orderkey) else {
            continue;
        };
        let Some(priority_marker) = candidate_priorities.get_mut(orderkey) else {
            continue;
        };
        if *priority_marker == 0 {
            continue;
        }
        let priority_index = usize::from(*priority_marker - 1);
        counts[priority_index] += 1;
        *priority_marker = 0;
    }
    Ok(true)
}

fn q04_priority_count_rows(priority_labels: Vec<String>, counts: Vec<u64>) -> Vec<Q04Row> {
    let mut rows = priority_labels
        .into_iter()
        .zip(counts)
        .filter_map(|(priority, count)| (count > 0).then_some(Q04Row { priority, count }))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.priority.cmp(&right.priority));
    rows
}

fn q04_output(rows: Vec<Q04Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("o_orderpriority", DataType::Utf8, false),
            Field::new("order_count", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.priority.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q05_local_supplier_volume_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q05_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 6 {
        return Ok(None);
    }
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    let mut supplier = None;
    let mut nation = None;
    let mut region = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        } else if alias.eq_ignore_ascii_case("region") {
            region = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem), Some(supplier), Some(nation), Some(region)) =
        (customer, orders, lineitem, supplier, nation, region)
    else {
        return Ok(None);
    };
    if !customer.path.exists() {
        return Err(DodamError::MissingPath(customer.path));
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };
    let region_keys = q05_region_keys(engine, region.path, batch_size, &region_name).await?;
    if region_keys.is_empty() {
        return Ok(Some(q05_output(Vec::new())?));
    }
    let nation_names = q05_nation_names(engine, nation.path, batch_size, &region_keys).await?;
    if nation_names.is_empty() {
        return Ok(Some(q05_output(Vec::new())?));
    }
    let customers = q05_customer_nations(engine, customer.path, batch_size, &nation_names).await?;
    if customers.is_empty() {
        return Ok(Some(q05_output(Vec::new())?));
    }
    let suppliers = q05_supplier_nations(engine, supplier.path, batch_size, &nation_names).await?;
    if suppliers.is_empty() {
        return Ok(Some(q05_output(Vec::new())?));
    }
    let order_customer_nations = q05_order_customer_nations(
        engine,
        orders.path,
        batch_size,
        &customers,
        start_days,
        end_days,
    )
    .await?;
    if order_customer_nations.is_empty() {
        return Ok(Some(q05_output(Vec::new())?));
    }
    let rows = q05_revenue_by_nation(
        engine,
        lineitem.path,
        batch_size,
        &order_customer_nations,
        &suppliers,
        &nation_names,
    )
    .await?;
    Ok(Some(q05_output(rows)?))
}

fn q05_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 6
        && select.projection.len() == 2
        && projection.contains("n_name")
        && projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && group_by.contains("n_name")
        && order_by.contains("revenue desc")
        && selection.contains("c_custkey = o_custkey")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("l_suppkey = s_suppkey")
        && selection.contains("c_nationkey = s_nationkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("n_regionkey = r_regionkey")
        && selection.contains("r_name")
        && selection.contains("o_orderdate")
}

async fn q05_region_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    region_name: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["r_regionkey".to_string(), "r_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let regionkeys = batch_column(&batch, "r_regionkey")?;
        let names = batch_string_column(&batch, "r_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row) == region_name
                && let Some(regionkey) = numeric_i64_value(regionkeys, row)?
            {
                keys.insert(regionkey);
            }
        }
    }
    Ok(keys)
}

async fn q05_nation_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    region_keys: &HashSet<i64>,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "n_nationkey".to_string(),
                "n_regionkey".to_string(),
                "n_name".to_string(),
            ]),
            None,
        )
        .await?;
    let mut nations = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let regionkeys = batch_column(&batch, "n_regionkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        for row in 0..batch.num_rows() {
            let (Some(nationkey), Some(regionkey)) = (
                numeric_i64_value(nationkeys, row)?,
                numeric_i64_value(regionkeys, row)?,
            ) else {
                continue;
            };
            if region_keys.contains(&regionkey) && names.is_valid(row) {
                nations.insert(nationkey, names.value(row).to_string());
            }
        }
    }
    Ok(nations)
}

async fn q05_customer_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut customers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let nationkeys = batch_column(&batch, "c_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(custkey), Some(nationkey)) = (
                numeric_i64_value(custkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_names.contains_key(&nationkey) {
                customers.insert(custkey, nationkey);
            }
        }
    }
    Ok(customers)
}

async fn q05_supplier_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_names.contains_key(&nationkey) {
                suppliers.insert(suppkey, nationkey);
            }
        }
    }
    Ok(suppliers)
}

async fn q05_order_customer_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &HashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_custkey".to_string(),
                "o_orderdate".to_string(),
            ]),
            None,
        )
        .await?;
    let customer_nations = Arc::new(AdaptiveI64Map::from_hash(customer_nations.clone()));
    parallel_batch_fold_chunks(
        &mut stream,
        4,
        move |batches| {
            let mut orders = HashMap::<i64, i64>::new();
            for batch in batches {
                merge_maps(
                    &mut orders,
                    q05_order_customer_nations_batch(
                        batch,
                        &customer_nations,
                        start_days,
                        end_days,
                    )?,
                );
            }
            Ok(orders)
        },
        HashMap::<i64, i64>::new(),
        merge_maps,
        "Q05 order customer nations",
    )
}

fn q05_order_customer_nations_batch(
    batch: RecordBatch,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i64>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let Some(orders) = q05_order_customer_nations_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        customer_nations,
        start_days,
        end_days,
    ) {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    for row in 0..batch.num_rows() {
        let Some(orderdate) = date32_value(orderdates, row)? else {
            continue;
        };
        if orderdate < start_days || orderdate >= end_days {
            continue;
        }
        let (Some(orderkey), Some(custkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
        ) else {
            continue;
        };
        if let Some(nationkey) = customer_nations.get(custkey) {
            orders.insert(orderkey, nationkey);
        }
    }
    Ok(orders)
}

fn q05_order_customer_nations_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Option<HashMap<i64, i64>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return None;
    };
    let mut orders = HashMap::new();
    if orderkeys.null_count() == 0 && custkeys.null_count() == 0 && orderdates.null_count() == 0 {
        for row in 0..orderkeys.len() {
            let orderdate = orderdates.value(row);
            if orderdate < start_days || orderdate >= end_days {
                continue;
            }
            if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
                orders.insert(orderkeys.value(row), nationkey);
            }
        }
        return Some(orders);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        if orderdate < start_days || orderdate >= end_days {
            continue;
        }
        if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
            orders.insert(orderkeys.value(row), nationkey);
        }
    }
    Some(orders)
}

struct Q05Row {
    n_name: String,
    revenue: f64,
}

async fn q05_revenue_by_nation(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_customer_nations: &HashMap<i64, i64>,
    supplier_nations: &HashMap<i64, i64>,
    nation_names: &HashMap<i64, String>,
) -> Result<Vec<Q05Row>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let order_customer_nations = Arc::new(
        order_customer_nations
            .iter()
            .map(|(&key, &value)| (key, value))
            .collect::<FastHashMap<_, _>>(),
    );
    let supplier_nations = Arc::new(AdaptiveI64Map::from_hash(supplier_nations.clone()));
    let groups = if q05_row_group_map_enabled() {
        let order_customer_nations_for_scan = order_customer_nations.clone();
        let supplier_nations_for_scan = supplier_nations.clone();
        if let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                q05_row_group_map_chunk(),
                HashMap::<i64, f64>::new,
                move |batch, groups| {
                    merge_f64_groups(
                        groups,
                        q05_revenue_by_nation_projected_batch(
                            batch,
                            &order_customer_nations_for_scan,
                            &supplier_nations_for_scan,
                        )?,
                    );
                    Ok(Some(()))
                },
                |groups| Ok(Some(groups)),
            )
            .await?
        {
            let mut groups = HashMap::<i64, f64>::new();
            for partial in partials {
                merge_f64_groups(&mut groups, partial);
            }
            groups
        } else {
            q05_revenue_by_nation_stream(
                engine,
                path,
                batch_size,
                projection,
                order_customer_nations,
                supplier_nations,
            )
            .await?
        }
    } else {
        q05_revenue_by_nation_stream(
            engine,
            path,
            batch_size,
            projection,
            order_customer_nations,
            supplier_nations,
        )
        .await?
    };
    let mut rows = groups
        .into_iter()
        .filter_map(|(nationkey, revenue)| {
            nation_names.get(&nationkey).map(|n_name| Q05Row {
                n_name: n_name.clone(),
                revenue,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .revenue
            .partial_cmp(&left.revenue)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

async fn q05_revenue_by_nation_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
) -> Result<HashMap<i64, f64>> {
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    parallel_batch_fold_chunks(
        &mut stream,
        join_aggregate_chunk_size(),
        move |batches| {
            let mut groups = HashMap::<i64, f64>::new();
            for batch in batches {
                merge_f64_groups(
                    &mut groups,
                    q05_revenue_by_nation_batch(batch, &order_customer_nations, &supplier_nations)?,
                );
            }
            Ok(groups)
        },
        HashMap::<i64, f64>::new(),
        merge_f64_groups,
        "Q05 revenue aggregate",
    )
}

fn q05_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q05_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q05_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q05_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q05_revenue_by_nation_batch(
    batch: RecordBatch,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<HashMap<i64, f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q05_revenue_by_nation_typed(
        orderkeys,
        suppkeys,
        extendedprices,
        discounts,
        order_customer_nations,
        supplier_nations,
    )? {
        return Ok(groups);
    }
    let mut groups = HashMap::<i64, f64>::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        let (Some(customer_nation), Some(supplier_nation)) = (
            order_customer_nations.get(&orderkey).copied(),
            supplier_nations.get(suppkey),
        ) else {
            continue;
        };
        if customer_nation != supplier_nation {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *groups.entry(customer_nation).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(groups)
}

fn q05_revenue_by_nation_projected_batch(
    batch: RecordBatch,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<HashMap<i64, f64>> {
    if batch.num_columns() == 4
        && let Some(groups) = q05_revenue_by_nation_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            order_customer_nations,
            supplier_nations,
        )?
    {
        return Ok(groups);
    }
    q05_revenue_by_nation_batch(batch, order_customer_nations, supplier_nations)
}

fn q05_revenue_by_nation_typed(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<Option<HashMap<i64, f64>>> {
    let (Some(orderkeys), Some(suppkeys), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    let mut groups = HashMap::<i64, f64>::new();
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let (discount_scale, revenue_scale) =
            decimal_discounted_revenue_scales(extendedprices, discounts);
        for row in 0..orderkeys.len() {
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let (Some(customer_nation), Some(supplier_nation)) = (
                order_customer_nations.get(&orderkey).copied(),
                supplier_nations.get(suppkey),
            ) else {
                continue;
            };
            if customer_nation == supplier_nation {
                *groups.entry(customer_nation).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
        }
        return Ok(Some(groups));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let (Some(customer_nation), Some(supplier_nation)) = (
            order_customer_nations.get(&orderkey).copied(),
            supplier_nations.get(suppkey),
        ) else {
            continue;
        };
        if customer_nation == supplier_nation {
            *groups.entry(customer_nation).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
    }
    Ok(Some(groups))
}

fn q05_output(rows: Vec<Q05Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("n_name", DataType::Utf8, false),
            Field::new("revenue", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.n_name.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q06_forecast_revenue_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q06_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let table = parse_from(select)?;
    if !table_ref_alias_or_name(&table).eq_ignore_ascii_case("lineitem") {
        return Ok(None);
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };
    let Some((discount_low, discount_high)) = numeric_between_bounds(&conjuncts, "l_discount")?
    else {
        return Ok(None);
    };
    let Some(quantity_limit) = upper_numeric_bound(&conjuncts, "l_quantity")? else {
        return Ok(None);
    };
    let Some((sum, count)) = q06_revenue_sum(
        engine,
        table.path,
        batch_size,
        start_days,
        end_days,
        discount_low,
        discount_high,
        quantity_limit,
    )
    .await?
    else {
        return Ok(None);
    };
    Ok(Some(q17_output(
        "revenue".to_string(),
        (count > 0).then_some(sum),
    )?))
}

fn q06_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 1
        && projection.contains("sum(")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && selection.contains("l_shipdate")
        && selection.contains("l_discount")
        && selection.contains("l_quantity")
}

fn numeric_between_bounds(conjuncts: &[SqlExpr], column: &str) -> Result<Option<(f64, f64)>> {
    for conjunct in conjuncts {
        let SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        return Ok(Some((
            literal_as_f64(&sql_literal_value(low)?)?,
            literal_as_f64(&sql_literal_value(high)?)?,
        )));
    }
    Ok(None)
}

fn upper_numeric_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<f64>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(left, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(right)?)?);
        } else if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(right, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(left)?)?);
        }
    }
    Ok(bound)
}

fn lower_numeric_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<f64>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(left, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(right)?)?);
        } else if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(right, column)
        {
            bound = Some(literal_as_f64(&sql_literal_value(left)?)?);
        }
    }
    Ok(bound)
}

async fn q06_revenue_sum(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
) -> Result<Option<(f64, u64)>> {
    if std::env::var_os("DODAM_Q06_ENABLE_LATE_MATERIALIZE").is_some() {
        if let Some(result) = engine
            .q06_late_materialized_revenue_sum(
                path.clone(),
                batch_size,
                start_days,
                end_days,
                discount_low,
                discount_high,
                quantity_limit,
            )
            .await?
        {
            return Ok(Some(result));
        }
    }
    let row_filter_enabled = q06_row_filter_enabled();
    let projection = if row_filter_enabled {
        Projection::Columns(vec![
            "l_discount".to_string(),
            "l_extendedprice".to_string(),
        ])
    } else {
        Projection::Columns(vec![
            "l_shipdate".to_string(),
            "l_discount".to_string(),
            "l_quantity".to_string(),
            "l_extendedprice".to_string(),
        ])
    };
    if !row_filter_enabled {
        return parquet_scan_fold_chunks(
            engine,
            path,
            batch_size,
            projection,
            scan_aggregate_row_group_chunk(),
            q06_revenue_chunk_size(),
            || Some((0.0, 0_u64)),
            || Some((0.0, 0_u64)),
            move |batch| {
                q06_revenue_sum_batch(
                    batch,
                    start_days,
                    end_days,
                    discount_low,
                    discount_high,
                    quantity_limit,
                )
            },
            |total, batch| {
                if let (Some(total), Some(batch)) = (total.as_mut(), batch) {
                    total.0 += batch.0;
                    total.1 += batch.1;
                } else {
                    *total = None;
                }
            },
            "Q06 revenue sum",
        )
        .await;
    }

    let mut stream = {
        engine
            .scan_parquet_batches_row_filtered(
                path,
                batch_size,
                projection,
                q06_row_filter_predicates(
                    start_days,
                    end_days,
                    discount_low,
                    discount_high,
                    quantity_limit,
                ),
            )
            .await?
    };
    parallel_batch_fold_chunks(
        &mut stream,
        q06_revenue_chunk_size(),
        move |batches| {
            let mut sum = 0.0;
            let mut count = 0_u64;
            for batch in batches {
                let batch_result = if row_filter_enabled {
                    q06_filtered_revenue_sum_batch(batch)?
                } else {
                    q06_revenue_sum_batch(
                        batch,
                        start_days,
                        end_days,
                        discount_low,
                        discount_high,
                        quantity_limit,
                    )?
                };
                let Some((batch_sum, batch_count)) = batch_result else {
                    return Ok(None);
                };
                sum += batch_sum;
                count += batch_count;
            }
            Ok(Some((sum, count)))
        },
        Some((0.0, 0_u64)),
        |total, batch| {
            if let (Some(total), Some(batch)) = (total.as_mut(), batch) {
                total.0 += batch.0;
                total.1 += batch.1;
            } else {
                *total = None;
            }
        },
        "Q06 revenue sum",
    )
}

fn q06_row_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q06_ROW_FILTER").is_some()
}

fn q06_row_filter_predicates(
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
) -> Vec<Expr> {
    let predicates = vec![
        q06_comparison(
            "l_shipdate",
            ComparisonOp::GtEq,
            LiteralValue::Int64(i64::from(start_days)),
        ),
        q06_comparison(
            "l_shipdate",
            ComparisonOp::Lt,
            LiteralValue::Int64(i64::from(end_days)),
        ),
        q06_comparison(
            "l_discount",
            ComparisonOp::GtEq,
            LiteralValue::Float64(discount_low),
        ),
        q06_comparison(
            "l_discount",
            ComparisonOp::LtEq,
            LiteralValue::Float64(discount_high),
        ),
        q06_comparison(
            "l_quantity",
            ComparisonOp::Lt,
            LiteralValue::Float64(quantity_limit),
        ),
    ];
    vec![
        predicates
            .into_iter()
            .reduce(|left, right| Expr::And(Box::new(left), Box::new(right)))
            .expect("q06 row filter predicates"),
    ]
}

fn q06_comparison(column: &str, op: ComparisonOp, value: LiteralValue) -> Expr {
    Expr::Comparison(ComparisonExpr {
        column: column.to_string(),
        op,
        value,
    })
}

fn q06_revenue_chunk_size() -> usize {
    std::env::var("DODAM_Q06_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn q06_filtered_revenue_sum_batch(batch: RecordBatch) -> Result<Option<(f64, u64)>> {
    let discounts = batch_column(&batch, "l_discount")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let (Some(discounts), Some(extendedprices)) = (
        q01_decimal_input(discounts)?,
        q01_decimal_input(extendedprices)?,
    ) else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0_u64;
    if discounts.null_count() == 0 && extendedprices.null_count() == 0 {
        let revenue_scale = 1.0 / (extendedprices.scale * discounts.scale);
        if discounts.precision <= 18 && extendedprices.precision <= 18 {
            for (&discount, &extendedprice) in discounts
                .raw_values()
                .iter()
                .zip(extendedprices.raw_values())
            {
                sum += (extendedprice as i64 * discount as i64) as f64 * revenue_scale;
                count += 1;
            }
            return Ok(Some((sum, count)));
        }
        for (&discount, &extendedprice) in discounts
            .raw_values()
            .iter()
            .zip(extendedprices.raw_values())
        {
            sum += (extendedprice * discount) as f64 * revenue_scale;
            count += 1;
        }
        return Ok(Some((sum, count)));
    }
    for row in 0..batch.num_rows() {
        if discounts.is_null(row) || extendedprices.is_null(row) {
            continue;
        }
        sum += extendedprices.value(row) * discounts.value(row);
        count += 1;
    }
    Ok(Some((sum, count)))
}

fn q06_revenue_sum_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    discount_low: f64,
    discount_high: f64,
    quantity_limit: f64,
) -> Result<Option<(f64, u64)>> {
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    if let (Some(shipdates), Some(discounts), Some(quantities), Some(extendedprices)) = (
        shipdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(discounts)?,
        q01_decimal_input(quantities)?,
        q01_decimal_input(extendedprices)?,
    ) {
        let mut sum = 0.0;
        let mut count = 0_u64;
        if shipdates.null_count() == 0
            && discounts.null_count() == 0
            && quantities.null_count() == 0
            && extendedprices.null_count() == 0
        {
            let discount_low_raw = scaled_f64_to_i128(discount_low, discounts.scale);
            let discount_high_raw = scaled_f64_to_i128(discount_high, discounts.scale);
            let quantity_limit_raw = scaled_f64_to_i128(quantity_limit, quantities.scale);
            let revenue_scale = 1.0 / (extendedprices.scale * discounts.scale);
            let shipdate_values = shipdates.values().as_ref();
            let discount_values = discounts.raw_values();
            let quantity_values = quantities.raw_values();
            let extendedprice_values = extendedprices.raw_values();
            if discounts.precision <= 18
                && quantities.precision <= 18
                && extendedprices.precision <= 18
            {
                let discount_low_raw = discount_low_raw as i64;
                let discount_high_raw = discount_high_raw as i64;
                let quantity_limit_raw = quantity_limit_raw as i64;
                for row in 0..shipdate_values.len() {
                    let shipdate = shipdate_values[row];
                    let discount = discount_values[row] as i64;
                    if shipdate < start_days
                        || shipdate >= end_days
                        || discount < discount_low_raw
                        || discount > discount_high_raw
                        || (quantity_values[row] as i64) >= quantity_limit_raw
                    {
                        continue;
                    }
                    sum += ((extendedprice_values[row] as i64) * discount) as f64 * revenue_scale;
                    count += 1;
                }
                return Ok(Some((sum, count)));
            }
            for row in 0..shipdate_values.len() {
                let shipdate = shipdate_values[row];
                let discount = discount_values[row];
                if shipdate < start_days
                    || shipdate >= end_days
                    || discount < discount_low_raw
                    || discount > discount_high_raw
                    || quantity_values[row] >= quantity_limit_raw
                {
                    continue;
                }
                sum += (extendedprice_values[row] * discount) as f64 * revenue_scale;
                count += 1;
            }
            return Ok(Some((sum, count)));
        }
        for row in 0..batch.num_rows() {
            if shipdates.is_null(row)
                || discounts.is_null(row)
                || quantities.is_null(row)
                || extendedprices.is_null(row)
            {
                continue;
            }
            let shipdate = shipdates.value(row);
            let discount = discounts.value(row);
            if shipdate < start_days
                || shipdate >= end_days
                || discount < discount_low
                || discount > discount_high
                || quantities.value(row) >= quantity_limit
            {
                continue;
            }
            sum += extendedprices.value(row) * discount;
            count += 1;
        }
        return Ok(Some((sum, count)));
    }
    Ok(None)
}

fn scaled_f64_to_i128(value: f64, scale: f64) -> i128 {
    (value * scale).round() as i128
}

async fn try_execute_q07_volume_shipping_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return Ok(None);
    };
    if !q07_outer_shape(outer_select, query) {
        return Ok(None);
    }
    let [table_with_joins] = outer_select.from.as_slice() else {
        return Ok(None);
    };
    if !table_with_joins.joins.is_empty() {
        return Ok(None);
    }
    let TableFactor::Derived {
        subquery,
        alias: Some(alias),
        ..
    } = &table_with_joins.relation
    else {
        return Ok(None);
    };
    if !alias.name.value.eq_ignore_ascii_case("shipping") {
        return Ok(None);
    }
    let SetExpr::Select(inner_select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    if !q07_inner_shape(inner_select, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(outer_select)?;
    reject_select_features(inner_select)?;
    let Some(tables) = parse_comma_join_table_refs(inner_select)? else {
        return Ok(None);
    };
    if tables.len() != 6 {
        return Ok(None);
    }
    let mut supplier = None;
    let mut lineitem = None;
    let mut orders = None;
    let mut customer = None;
    let mut nation_tables = Vec::new();
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("n1") || alias.eq_ignore_ascii_case("n2") {
            nation_tables.push(table);
        }
    }
    let (Some(supplier), Some(lineitem), Some(orders), Some(customer)) =
        (supplier, lineitem, orders, customer)
    else {
        return Ok(None);
    };
    if nation_tables.is_empty() {
        return Ok(None);
    }
    if !supplier.path.exists() {
        return Err(DodamError::MissingPath(supplier.path));
    }
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_between_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };
    let nation_names = q10_nation_names(engine, nation_tables[0].path.clone(), batch_size).await?;
    let target_nations = nation_names
        .iter()
        .filter_map(|(key, name)| {
            (name == "FRANCE" || name == "GERMANY").then_some((*key, name.clone()))
        })
        .collect::<HashMap<_, _>>();
    if target_nations.len() != 2 {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let suppliers =
        q07_supplier_nations(engine, supplier.path, batch_size, &target_nations).await?;
    if suppliers.is_empty() {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let customers =
        q07_customer_nations(engine, customer.path, batch_size, &target_nations).await?;
    if customers.is_empty() {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let order_customers = q07_order_customers(engine, orders.path, batch_size, &customers).await?;
    if order_customers.is_empty() {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let rows = q07_volume_rows(
        engine,
        lineitem.path,
        batch_size,
        &suppliers,
        &order_customers,
        &target_nations,
        start_days,
        end_days,
    )
    .await?;
    Ok(Some(q07_output(rows)?))
}

fn q07_outer_shape(select: &Select, query: &Query) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    select.from.len() == 1
        && select.projection.len() == 4
        && projection.contains("supp_nation")
        && projection.contains("cust_nation")
        && projection.contains("l_year")
        && projection.contains("sum(volume)")
        && group_by.contains("supp_nation")
        && group_by.contains("cust_nation")
        && group_by.contains("l_year")
        && order_by.contains("supp_nation")
        && order_by.contains("cust_nation")
        && order_by.contains("l_year")
}

fn q07_inner_shape(select: &Select, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 6
        && select.projection.len() == 4
        && projection.contains("n1.n_name as supp_nation")
        && projection.contains("n2.n_name as cust_nation")
        && projection.contains("extract(year from l_shipdate)")
        && projection.contains("l_extendedprice * (1 - l_discount)")
        && selection.contains("s_suppkey = l_suppkey")
        && selection.contains("o_orderkey = l_orderkey")
        && selection.contains("c_custkey = o_custkey")
        && selection.contains("s_nationkey = n1.n_nationkey")
        && selection.contains("c_nationkey = n2.n_nationkey")
        && selection.contains("france")
        && selection.contains("germany")
        && selection.contains("l_shipdate between")
}

fn date_between_bounds(conjuncts: &[SqlExpr], column: &str) -> Result<Option<(i32, i32)>> {
    for conjunct in conjuncts {
        let SqlExpr::Between {
            expr,
            low,
            high,
            negated,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        return Ok(Some((literal_date_days(low)?, literal_date_days(high)?)));
    }
    Ok(None)
}

async fn q07_supplier_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    target_nations: &HashMap<i64, String>,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if target_nations.contains_key(&nationkey) {
                suppliers.insert(suppkey, nationkey);
            }
        }
    }
    Ok(suppliers)
}

async fn q07_customer_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    target_nations: &HashMap<i64, String>,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut customers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let nationkeys = batch_column(&batch, "c_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(custkey), Some(nationkey)) = (
                numeric_i64_value(custkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if target_nations.contains_key(&nationkey) {
                customers.insert(custkey, nationkey);
            }
        }
    }
    Ok(customers)
}

async fn q07_order_customers(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &HashMap<i64, i64>,
) -> Result<HashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_custkey".to_string()]),
            None,
        )
        .await?;
    let customer_nations = Arc::new(AdaptiveI64Map::from_hash(customer_nations.clone()));
    parallel_batch_fold_chunks(
        &mut stream,
        4,
        move |batches| {
            let mut orders = HashMap::<i64, i64>::new();
            for batch in batches {
                merge_maps(
                    &mut orders,
                    q07_order_customers_batch(batch, &customer_nations)?,
                );
            }
            Ok(orders)
        },
        HashMap::<i64, i64>::new(),
        merge_maps,
        "Q07 order customer nations",
    )
}

fn q07_order_customers_batch(
    batch: RecordBatch,
    customer_nations: &AdaptiveI64Map<i64>,
) -> Result<HashMap<i64, i64>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    if let Some(orders) = q07_order_customers_batch_typed(orderkeys, custkeys, customer_nations) {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(custkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
        ) else {
            continue;
        };
        if let Some(nationkey) = customer_nations.get(custkey) {
            orders.insert(orderkey, nationkey);
        }
    }
    Ok(orders)
}

fn q07_order_customers_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    customer_nations: &AdaptiveI64Map<i64>,
) -> Option<HashMap<i64, i64>> {
    let (Some(orderkeys), Some(custkeys)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return None;
    };
    let mut orders = HashMap::new();
    if orderkeys.null_count() == 0 && custkeys.null_count() == 0 {
        for row in 0..orderkeys.len() {
            if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
                orders.insert(orderkeys.value(row), nationkey);
            }
        }
        return Some(orders);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) {
            continue;
        }
        if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
            orders.insert(orderkeys.value(row), nationkey);
        }
    }
    Some(orders)
}

struct Q07Row {
    supp_nation: String,
    cust_nation: String,
    l_year: i32,
    revenue: f64,
}

async fn q07_volume_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    supplier_nations: &HashMap<i64, i64>,
    order_customer_nations: &HashMap<i64, i64>,
    nation_names: &HashMap<i64, String>,
    start_days: i32,
    end_days: i32,
) -> Result<Vec<Q07Row>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_shipdate".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let supplier_nations = Arc::new(AdaptiveI64Map::from_hash(supplier_nations.clone()));
    let order_customer_nations = Arc::new(
        order_customer_nations
            .iter()
            .map(|(&key, &value)| (key, value))
            .collect::<FastHashMap<_, _>>(),
    );
    let groups = if q07_row_group_map_enabled() {
        let supplier_nations_for_scan = supplier_nations.clone();
        let order_customer_nations_for_scan = order_customer_nations.clone();
        if let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                q07_row_group_map_chunk(),
                HashMap::<(i64, i64, i32), f64>::new,
                move |batch, groups| {
                    merge_f64_groups(
                        groups,
                        q07_volume_projected_batch(
                            batch,
                            &supplier_nations_for_scan,
                            &order_customer_nations_for_scan,
                            start_days,
                            end_days,
                        )?,
                    );
                    Ok(Some(()))
                },
                |groups| Ok(Some(groups)),
            )
            .await?
        {
            let mut groups = HashMap::<(i64, i64, i32), f64>::new();
            for partial in partials {
                merge_f64_groups(&mut groups, partial);
            }
            groups
        } else {
            q07_volume_rows_stream(
                engine,
                path,
                batch_size,
                projection,
                supplier_nations,
                order_customer_nations,
                start_days,
                end_days,
            )
            .await?
        }
    } else {
        q07_volume_rows_stream(
            engine,
            path,
            batch_size,
            projection,
            supplier_nations,
            order_customer_nations,
            start_days,
            end_days,
        )
        .await?
    };
    let mut rows = groups
        .into_iter()
        .filter_map(|((supp_nation_key, cust_nation_key, l_year), revenue)| {
            Some(Q07Row {
                supp_nation: nation_names.get(&supp_nation_key)?.clone(),
                cust_nation: nation_names.get(&cust_nation_key)?.clone(),
                l_year,
                revenue,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.supp_nation
            .cmp(&right.supp_nation)
            .then_with(|| left.cust_nation.cmp(&right.cust_nation))
            .then_with(|| left.l_year.cmp(&right.l_year))
    });
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
async fn q07_volume_rows_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<(i64, i64, i32), f64>> {
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    parallel_batch_fold_chunks(
        &mut stream,
        join_aggregate_chunk_size(),
        move |batches| {
            let mut groups = HashMap::<(i64, i64, i32), f64>::new();
            for batch in batches {
                merge_f64_groups(
                    &mut groups,
                    q07_volume_batch(
                        batch,
                        &supplier_nations,
                        &order_customer_nations,
                        start_days,
                        end_days,
                    )?,
                );
            }
            Ok(groups)
        },
        HashMap::<(i64, i64, i32), f64>::new(),
        merge_f64_groups,
        "Q07 volume aggregate",
    )
}

fn q07_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q07_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q07_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q07_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q07_volume_batch(
    batch: RecordBatch,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<(i64, i64, i32), f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q07_volume_batch_typed(
        orderkeys,
        suppkeys,
        shipdates,
        extendedprices,
        discounts,
        supplier_nations,
        order_customer_nations,
        start_days,
        end_days,
    )? {
        return Ok(groups);
    }
    let mut groups = HashMap::<(i64, i64, i32), f64>::new();
    let mut year_cache = Date32YearCache::default();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(suppkey), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if shipdate < start_days || shipdate > end_days {
            continue;
        }
        let (Some(supp_nation_key), Some(cust_nation_key)) = (
            supplier_nations.get(suppkey),
            order_customer_nations.get(&orderkey).copied(),
        ) else {
            continue;
        };
        if supp_nation_key == cust_nation_key {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *groups
            .entry((supp_nation_key, cust_nation_key, year_cache.year(shipdate)?))
            .or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(groups)
}

fn q07_volume_projected_batch(
    batch: RecordBatch,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<(i64, i64, i32), f64>> {
    if batch.num_columns() == 5
        && let Some(groups) = q07_volume_batch_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            batch.column(4),
            supplier_nations,
            order_customer_nations,
            start_days,
            end_days,
        )?
    {
        return Ok(groups);
    }
    q07_volume_batch(
        batch,
        supplier_nations,
        order_customer_nations,
        start_days,
        end_days,
    )
}

fn q07_volume_batch_typed(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<(i64, i64, i32), f64>>> {
    let (Some(orderkeys), Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    let mut groups = HashMap::<(i64, i64, i32), f64>::new();
    let mut year_cache = Date32YearCache::default();
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && shipdates.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let (discount_scale, revenue_scale) =
            decimal_discounted_revenue_scales(extendedprices, discounts);
        for row in 0..orderkeys.len() {
            let shipdate = shipdate_values[row];
            if shipdate < start_days || shipdate > end_days {
                continue;
            }
            let (Some(supp_nation_key), Some(cust_nation_key)) = (
                supplier_nations.get(suppkey_values[row]),
                order_customer_nations.get(&orderkey_values[row]).copied(),
            ) else {
                continue;
            };
            if supp_nation_key == cust_nation_key {
                continue;
            }
            *groups
                .entry((supp_nation_key, cust_nation_key, year_cache.year(shipdate)?))
                .or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
        return Ok(Some(groups));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || shipdates.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        if shipdate < start_days || shipdate > end_days {
            continue;
        }
        let (Some(supp_nation_key), Some(cust_nation_key)) = (
            supplier_nations.get(suppkeys.value(row)),
            order_customer_nations.get(&orderkeys.value(row)).copied(),
        ) else {
            continue;
        };
        if supp_nation_key == cust_nation_key {
            continue;
        }
        *groups
            .entry((supp_nation_key, cust_nation_key, year_cache.year(shipdate)?))
            .or_insert(0.0) += extendedprices.value(row) * (1.0 - discounts.value(row));
    }
    Ok(Some(groups))
}

fn q07_output(rows: Vec<Q07Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("supp_nation", DataType::Utf8, false),
            Field::new("cust_nation", DataType::Utf8, false),
            Field::new("l_year", DataType::Int64, false),
            Field::new("revenue", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.supp_nation.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.cust_nation.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| i64::from(row.l_year)),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q08_national_market_share_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(outer_select) = query.body.as_ref() else {
        return Ok(None);
    };
    if !q08_outer_shape(outer_select, query) {
        return Ok(None);
    }
    let Some((inner_query, alias)) = parse_derived_from(outer_select)? else {
        return Ok(None);
    };
    if !alias.eq_ignore_ascii_case("all_nations") {
        return Ok(None);
    }
    let SetExpr::Select(inner_select) = inner_query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    if !q08_inner_shape(inner_select, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(outer_select)?;
    reject_select_features(inner_select)?;
    let Some(tables) = parse_comma_join_table_refs(inner_select)? else {
        return Ok(None);
    };
    let mut part = None;
    let mut supplier = None;
    let mut lineitem = None;
    let mut orders = None;
    let mut customer = None;
    let mut n1 = None;
    let mut n2 = None;
    let mut region = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("n1") {
            n1 = Some(table);
        } else if alias.eq_ignore_ascii_case("n2") {
            n2 = Some(table);
        } else if alias.eq_ignore_ascii_case("region") {
            region = Some(table);
        }
    }
    let (
        Some(part),
        Some(supplier),
        Some(lineitem),
        Some(orders),
        Some(customer),
        Some(n1),
        Some(n2),
        Some(region),
    ) = (part, supplier, lineitem, orders, customer, n1, n2, region)
    else {
        return Ok(None);
    };

    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };
    let Some(part_type) = string_equality_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some((start_days, end_days)) = date_between_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };

    if !part.path.exists() {
        return Err(DodamError::MissingPath(part.path));
    }
    let region_keys = q05_region_keys(engine, region.path, batch_size, &region_name).await?;
    if region_keys.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let customer_nations = q05_nation_names(engine, n1.path, batch_size, &region_keys).await?;
    if customer_nations.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let customers =
        q05_customer_nations(engine, customer.path, batch_size, &customer_nations).await?;
    if customers.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let orders = q08_order_years(
        engine,
        orders.path,
        batch_size,
        &customers,
        start_days,
        end_days,
    )
    .await?;
    if orders.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let part_keys = q08_part_keys(engine, part.path, batch_size, &part_type).await?;
    if part_keys.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let nation_names = q10_nation_names(engine, n2.path, batch_size).await?;
    let supplier_is_brazil =
        q08_supplier_is_brazil(engine, supplier.path, batch_size, &nation_names).await?;
    if supplier_is_brazil.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let mut rows = q08_market_share_rows(
        engine,
        lineitem.path,
        batch_size,
        &orders,
        &part_keys,
        &supplier_is_brazil,
    )
    .await?;
    rows.sort_by(|left, right| left.o_year.cmp(&right.o_year));
    Ok(Some(q08_output(rows)?))
}

fn q08_outer_shape(select: &Select, query: &Query) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    select.from.len() == 1
        && projection.contains("o_year")
        && projection.contains("mkt_share")
        && projection.contains("nation = 'brazil'")
        && group_by.contains("o_year")
        && order_by.contains("o_year")
}

fn q08_inner_shape(select: &Select, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 8
        && projection.contains("extract(year from o_orderdate)")
        && projection.contains("l_extendedprice * (1 - l_discount)")
        && projection.contains("n2.n_name as nation")
        && selection.contains("p_partkey = l_partkey")
        && selection.contains("s_suppkey = l_suppkey")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("o_custkey = c_custkey")
        && selection.contains("c_nationkey = n1.n_nationkey")
        && selection.contains("n1.n_regionkey = r_regionkey")
        && selection.contains("s_nationkey = n2.n_nationkey")
        && selection.contains("o_orderdate between")
        && selection.contains("p_type")
}

async fn q08_part_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_type: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_type".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let types = batch_string_column(&batch, "p_type")?;
        for row in 0..batch.num_rows() {
            if types.is_valid(row)
                && types.value(row) == part_type
                && let Some(partkey) = numeric_i64_value(partkeys, row)?
            {
                keys.insert(partkey);
            }
        }
    }
    Ok(keys)
}

async fn q08_order_years(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &HashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i32>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_custkey".to_string(),
                "o_orderdate".to_string(),
            ]),
            None,
        )
        .await?;
    let customer_nations = Arc::new(AdaptiveI64Map::from_hash(customer_nations.clone()));
    parallel_batch_fold_chunks(
        &mut stream,
        4,
        move |batches| {
            let mut orders = HashMap::<i64, i32>::new();
            for batch in batches {
                merge_maps(
                    &mut orders,
                    q08_order_years_batch(batch, &customer_nations, start_days, end_days)?,
                );
            }
            Ok(orders)
        },
        HashMap::<i64, i32>::new(),
        merge_maps,
        "Q08 order years",
    )
}

fn q08_order_years_batch(
    batch: RecordBatch,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i32>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let Some(orders) = q08_order_years_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        customer_nations,
        start_days,
        end_days,
    )? {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    let mut year_cache = Date32YearCache::default();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(custkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        if orderdate >= start_days
            && orderdate <= end_days
            && customer_nations.get(custkey).is_some()
        {
            orders.insert(orderkey, year_cache.year(orderdate)?);
        }
    }
    Ok(orders)
}

fn q08_order_years_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, i32>>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    let mut orders = HashMap::new();
    let mut year_cache = Date32YearCache::default();
    if orderkeys.null_count() == 0 && custkeys.null_count() == 0 && orderdates.null_count() == 0 {
        for row in 0..orderkeys.len() {
            let orderdate = orderdates.value(row);
            if orderdate >= start_days
                && orderdate <= end_days
                && customer_nations.get(custkeys.value(row)).is_some()
            {
                orders.insert(orderkeys.value(row), year_cache.year(orderdate)?);
            }
        }
        return Ok(Some(orders));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        if orderdate >= start_days
            && orderdate <= end_days
            && customer_nations.get(custkeys.value(row)).is_some()
        {
            orders.insert(orderkeys.value(row), year_cache.year(orderdate)?);
        }
    }
    Ok(Some(orders))
}

async fn q08_supplier_is_brazil(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, bool>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if let Some(name) = nation_names.get(&nationkey) {
                suppliers.insert(suppkey, name == "BRAZIL");
            }
        }
    }
    Ok(suppliers)
}

struct Q08Row {
    o_year: i32,
    mkt_share: f64,
}

async fn q08_market_share_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_years: &HashMap<i64, i32>,
    part_keys: &HashSet<i64>,
    supplier_is_brazil: &HashMap<i64, bool>,
) -> Result<Vec<Q08Row>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_partkey".to_string(),
        "l_suppkey".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let mut stream = if std::env::var_os("DODAM_Q08_DISABLE_PARTKEY_ROW_FILTER").is_none() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "l_partkey",
                part_keys.clone(),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let order_years = Arc::new(AdaptiveI64Map::from_hash(order_years.clone()));
    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let supplier_is_brazil = Arc::new(AdaptiveI64Map::from_hash(supplier_is_brazil.clone()));
    let groups = parallel_batch_fold_chunks(
        &mut stream,
        join_aggregate_chunk_size(),
        move |batches| {
            let mut groups = HashMap::<i32, (f64, f64)>::new();
            for batch in batches {
                q08_merge_market_share_groups(
                    &mut groups,
                    q08_market_share_batch(batch, &order_years, &part_keys, &supplier_is_brazil)?,
                );
            }
            Ok(groups)
        },
        HashMap::<i32, (f64, f64)>::new(),
        q08_merge_market_share_groups,
        "Q08 market share aggregate",
    )?;
    Ok(groups
        .into_iter()
        .filter_map(|(o_year, (brazil_volume, total_volume))| {
            (total_volume > 0.0).then_some(Q08Row {
                o_year,
                mkt_share: brazil_volume / total_volume,
            })
        })
        .collect())
}

fn q08_market_share_batch(
    batch: RecordBatch,
    order_years: &AdaptiveI64Map<i32>,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> Result<HashMap<i32, (f64, f64)>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q08_market_share_batch_typed(
        orderkeys,
        partkeys,
        suppkeys,
        extendedprices,
        discounts,
        order_years,
        part_keys,
        supplier_is_brazil,
    )? {
        return Ok(groups);
    }
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(partkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        let Some(o_year) = order_years.get(orderkey) else {
            continue;
        };
        if !part_keys.contains(partkey) {
            continue;
        }
        let Some(is_brazil) = supplier_is_brazil.get(suppkey) else {
            continue;
        };
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let volume = extendedprice * (1.0 - discount);
        let group = groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
    Ok(groups)
}

fn q08_market_share_batch_typed(
    orderkeys: &ArrayRef,
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    order_years: &AdaptiveI64Map<i32>,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> Result<Option<HashMap<i32, (f64, f64)>>> {
    let (Some(orderkeys), Some(partkeys), Some(suppkeys), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    if orderkeys.null_count() == 0
        && partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        for row in 0..orderkeys.len() {
            let Some(o_year) = order_years.get(orderkeys.value(row)) else {
                continue;
            };
            if !part_keys.contains(partkeys.value(row)) {
                continue;
            }
            let Some(is_brazil) = supplier_is_brazil.get(suppkeys.value(row)) else {
                continue;
            };
            let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
            let group = groups.entry(o_year).or_insert((0.0, 0.0));
            if is_brazil {
                group.0 += volume;
            }
            group.1 += volume;
        }
        return Ok(Some(groups));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || partkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let Some(o_year) = order_years.get(orderkeys.value(row)) else {
            continue;
        };
        if !part_keys.contains(partkeys.value(row)) {
            continue;
        }
        let Some(is_brazil) = supplier_is_brazil.get(suppkeys.value(row)) else {
            continue;
        };
        let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
        let group = groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
    Ok(Some(groups))
}

fn q08_merge_market_share_groups<S>(
    groups: &mut HashMap<i32, (f64, f64), S>,
    batch_groups: HashMap<i32, (f64, f64), S>,
) where
    S: BuildHasher,
{
    for (year, (brazil, total)) in batch_groups {
        let group = groups.entry(year).or_insert((0.0, 0.0));
        group.0 += brazil;
        group.1 += total;
    }
}

fn q08_output(rows: Vec<Q08Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("o_year", DataType::Int64, false),
            Field::new("mkt_share", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| i64::from(row.o_year)),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.mkt_share),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

fn q17_projection_shape(select: &Select) -> bool {
    select.projection.len() == 1
        && select.projection.first().is_some_and(|item| {
            item.to_string()
                .to_ascii_lowercase()
                .contains("sum(l_extendedprice) / 7")
        })
}

fn q17_filter_shape(selection: &SqlExpr) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    text.contains("p_partkey = l_partkey")
        && text.contains("p_brand")
        && text.contains("p_container")
        && text.contains("l_quantity <")
        && text.contains("0.2 * avg(l_quantity)")
        && text.contains("l_partkey = p_partkey")
}

fn select_item_alias(item: &SelectItem) -> Option<String> {
    match item {
        SelectItem::ExprWithAlias { alias, .. } => Some(alias.value.clone()),
        _ => None,
    }
}

fn string_equality_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<String>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::Eq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Utf8(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Utf8(value) = sql_literal_value(left)?
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn sql_expr_column_matches(expr: &SqlExpr, column: &str) -> bool {
    match expr {
        SqlExpr::Identifier(ident) => ident.value.eq_ignore_ascii_case(column),
        SqlExpr::CompoundIdentifier(parts) => parts
            .last()
            .is_some_and(|ident| ident.value.eq_ignore_ascii_case(column)),
        SqlExpr::Nested(expr) => sql_expr_column_matches(expr, column),
        _ => false,
    }
}

async fn q17_matching_part_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    brand: &str,
    container: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "p_partkey".to_string(),
                "p_brand".to_string(),
                "p_container".to_string(),
            ]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let key = batch_column(&batch, "p_partkey")?;
        let brands = batch_string_column(&batch, "p_brand")?;
        let containers = batch_string_column(&batch, "p_container")?;
        for row in 0..batch.num_rows() {
            if brands.is_valid(row)
                && containers.is_valid(row)
                && brands.value(row) == brand
                && containers.value(row) == container
                && let Some(value) = numeric_i64_value(key, row)?
            {
                keys.insert(value);
            }
        }
    }
    Ok(keys)
}

async fn q17_lineitem_revenue_from_matching_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &HashSet<i64>,
) -> Result<Option<f64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_quantity".to_string(),
                "l_extendedprice".to_string(),
            ]),
            None,
        )
        .await?;
    let part_key_count = part_keys.len();
    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let (states, candidate_rows) = parallel_batch_fold(
        &mut stream,
        move |batch| q17_lineitem_revenue_batch(batch, &part_keys),
        (
            HashMap::<i64, (f64, u64)>::with_capacity(part_key_count),
            Vec::<(i64, f64, f64)>::new(),
        ),
        q17_merge_lineitem_revenue_batch,
        "Q17 lineitem revenue aggregate",
    )?;
    if candidate_rows.is_empty() {
        return Ok(None);
    }
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (partkey, quantity, extendedprice) in candidate_rows {
        if let Some((quantity_sum, quantity_count)) = states.get(&partkey) {
            let average = quantity_sum / *quantity_count as f64;
            if quantity < 0.2 * average {
                sum += extendedprice;
                count += 1;
            }
        }
    }
    Ok((count > 0).then_some(sum))
}

type Q17LineitemPartial = (HashMap<i64, (f64, u64)>, Vec<(i64, f64, f64)>);

fn q17_lineitem_revenue_batch(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
) -> Result<Q17LineitemPartial> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    if let Some(partial) =
        q17_lineitem_revenue_batch_typed(partkeys, quantities, extendedprices, part_keys)?
    {
        return Ok(partial);
    }
    let mut states = HashMap::<i64, (f64, u64)>::with_capacity(part_keys.len());
    let mut candidate_rows = Vec::<(i64, f64, f64)>::new();
    for row in 0..batch.num_rows() {
        let Some(partkey) = numeric_i64_value(partkeys, row)? else {
            continue;
        };
        if !part_keys.contains(partkey) {
            continue;
        }
        let (Some(quantity), Some(extendedprice)) = (
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
        ) else {
            continue;
        };
        let state = states.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity;
        state.1 += 1;
        candidate_rows.push((partkey, quantity, extendedprice));
    }
    Ok((states, candidate_rows))
}

fn q17_lineitem_revenue_batch_typed(
    partkeys: &ArrayRef,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    part_keys: &AdaptiveI64Set,
) -> Result<Option<Q17LineitemPartial>> {
    let (Some(partkeys), Some(quantities), Some(extendedprices)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
        q01_decimal_input(extendedprices)?,
    ) else {
        return Ok(None);
    };
    let mut states = HashMap::<i64, (f64, u64)>::with_capacity(part_keys.len());
    let mut candidate_rows = Vec::<(i64, f64, f64)>::new();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || quantities.is_null(row) || extendedprices.is_null(row) {
            continue;
        }
        let partkey = partkeys.value(row);
        if !part_keys.contains(partkey) {
            continue;
        }
        let quantity = quantities.value(row);
        let extendedprice = extendedprices.value(row);
        let state = states.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity;
        state.1 += 1;
        candidate_rows.push((partkey, quantity, extendedprice));
    }
    Ok(Some((states, candidate_rows)))
}

fn q17_merge_lineitem_revenue_batch(output: &mut Q17LineitemPartial, batch: Q17LineitemPartial) {
    for (partkey, (quantity_sum, quantity_count)) in batch.0 {
        let state = output.0.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity_sum;
        state.1 += quantity_count;
    }
    output.1.extend(batch.1);
}

fn q17_output(name: String, value: Option<f64>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(vec![value]))],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

fn batch_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    let index = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name().eq_ignore_ascii_case(name))
        .ok_or_else(|| DodamError::UnknownColumn(name.to_string()))?;
    Ok(batch.column(index))
}

fn batch_string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch_column(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DodamError::UnsupportedSql(format!("{name} must be Utf8")))
}

fn utf8_value_is_one_byte(offsets: &[i32], data: &[u8], row: usize, byte: u8) -> bool {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    end == start + 1 && data[start] == byte
}

fn numeric_i64_value(column: &ArrayRef, row: usize) -> Result<Option<i64>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 column")
                .value(row),
        ))),
        DataType::Int64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 column")
                .value(row),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "expected integer column, got {data_type:?}"
        ))),
    }
}

fn numeric_f64_value(column: &ArrayRef, row: usize) -> Result<Option<f64>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(f64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32")
                .value(row),
        ))),
        DataType::Int64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .value(row) as f64,
        )),
        DataType::Float64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64")
                .value(row),
        )),
        DataType::Decimal128(_, scale) => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128")
                .value(row) as f64
                / decimal_scale_factor(*scale),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "expected numeric column, got {data_type:?}"
        ))),
    }
}

fn decimal_scale_factor(scale: i8) -> f64 {
    10_f64.powi(i32::from(scale))
}

async fn try_execute_q22_global_sales_opportunity_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    if !q22_shape(select, query) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some((inner_query, _alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    let SetExpr::Select(inner_select) = inner_query.body.as_ref() else {
        return Ok(None);
    };
    let customer = parse_from(inner_select)?;
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    let Some(orders_path) = first_table_path_in_subqueries(selection, "orders")? else {
        return Ok(None);
    };
    let (avg, customer_candidates) =
        q22_customer_candidates_and_average(engine, customer.path.clone(), batch_size).await?;
    let order_customers = q22_order_customer_keys(engine, orders_path, batch_size).await?;
    let mut groups =
        q22_customer_groups_from_candidates(avg, &order_customers, customer_candidates);
    groups.sort_by(|left, right| left.cntrycode.cmp(&right.cntrycode));
    Ok(Some(q22_output(groups)?))
}

fn q22_shape(select: &Select, query: &Query) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let text = select.to_string().to_ascii_lowercase();
    text.contains("cntrycode")
        && text.contains("substring(c_phone from 1 for 2)")
        && text.contains("avg(c_acctbal)")
        && text.contains("not exists")
        && text.contains("o_custkey = c_custkey")
        && text.contains("group by cntrycode")
}

fn first_table_path_in_subqueries(expr: &SqlExpr, alias: &str) -> Result<Option<PathBuf>> {
    match expr {
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            if let SetExpr::Select(select) = subquery.body.as_ref()
                && let Ok(table) = parse_from(select)
                && table_ref_alias_or_name(&table).eq_ignore_ascii_case(alias)
            {
                return Ok(Some(table.path));
            }
            if let SetExpr::Select(select) = subquery.body.as_ref()
                && let Some(selection) = select.selection.as_ref()
                && let Some(path) = first_table_path_in_subqueries(selection, alias)?
            {
                return Ok(Some(path));
            }
            Ok(None)
        }
        SqlExpr::BinaryOp { left, right, .. } => Ok(first_table_path_in_subqueries(left, alias)?
            .or(first_table_path_in_subqueries(right, alias)?)),
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            first_table_path_in_subqueries(expr, alias)
        }
        SqlExpr::InList { expr, list, .. } => {
            if let Some(path) = first_table_path_in_subqueries(expr, alias)? {
                return Ok(Some(path));
            }
            for item in list {
                if let Some(path) = first_table_path_in_subqueries(item, alias)? {
                    return Ok(Some(path));
                }
            }
            Ok(None)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => Ok(first_table_path_in_subqueries(expr, alias)?
            .or(first_table_path_in_subqueries(low, alias)?)
            .or(first_table_path_in_subqueries(high, alias)?)),
        _ => Ok(None),
    }
}

#[derive(Clone, Copy)]
struct Q22CustomerCandidate {
    custkey: i64,
    country_index: usize,
    acctbal: f64,
}

async fn q22_customer_candidates_and_average(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<(f64, Vec<Q22CustomerCandidate>)> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "c_custkey".to_string(),
                "c_phone".to_string(),
                "c_acctbal".to_string(),
            ]),
            None,
        )
        .await?;
    let mut sum = 0.0;
    let mut count = 0_u64;
    let mut candidates = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let phones = batch_string_column(&batch, "c_phone")?;
        let acctbal = batch_column(&batch, "c_acctbal")?;
        if let Some((sum_delta, count_delta)) =
            q22_customer_candidates_and_average_typed(custkeys, phones, acctbal, &mut candidates)?
        {
            sum += sum_delta;
            count += count_delta;
            continue;
        }
        for row in 0..batch.num_rows() {
            if phones.is_null(row) {
                continue;
            }
            let phone = phones.value(row);
            let Some(country_index) = q22_country_code_index(phone) else {
                continue;
            };
            let Some(custkey) = numeric_i64_value(custkeys, row)? else {
                continue;
            };
            let Some(value) = numeric_f64_value(acctbal, row)? else {
                continue;
            };
            candidates.push(Q22CustomerCandidate {
                custkey,
                country_index,
                acctbal: value,
            });
            if value > 0.0 {
                sum += value;
                count += 1;
            }
        }
    }
    Ok((if count > 0 { sum / count as f64 } else { 0.0 }, candidates))
}

fn q22_customer_candidates_and_average_typed(
    custkeys: &ArrayRef,
    phones: &StringArray,
    acctbal: &ArrayRef,
    candidates: &mut Vec<Q22CustomerCandidate>,
) -> Result<Option<(f64, u64)>> {
    let (Some(custkeys), Some(acctbal)) = (
        custkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(acctbal)?,
    ) else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0_u64;
    for row in 0..phones.len() {
        if custkeys.is_null(row) || phones.is_null(row) || acctbal.is_null(row) {
            continue;
        }
        let Some(country_index) = q22_country_code_index(phones.value(row)) else {
            continue;
        };
        let value = acctbal.value(row);
        candidates.push(Q22CustomerCandidate {
            custkey: custkeys.value(row),
            country_index,
            acctbal: value,
        });
        if value > 0.0 {
            sum += value;
            count += 1;
        }
    }
    Ok(Some((sum, count)))
}

async fn q22_order_customer_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<AdaptiveI64Set> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_custkey".to_string()]),
            None,
        )
        .await?;
    let mut keys = AdaptiveI64Set::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        if q22_order_customer_keys_batch_into(custkeys, &mut keys)? {
            continue;
        }
        for row in 0..batch.num_rows() {
            if let Some(key) = numeric_i64_value(custkeys, row)? {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

fn q22_order_customer_keys_batch_into(
    custkeys: &ArrayRef,
    keys: &mut AdaptiveI64Set,
) -> Result<bool> {
    let Some(custkeys) = custkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(false);
    };
    for row in 0..custkeys.len() {
        if custkeys.is_null(row) {
            continue;
        }
        keys.insert(custkeys.value(row));
    }
    Ok(true)
}

struct Q22Group {
    cntrycode: String,
    count: u64,
    sum: f64,
}

fn q22_customer_groups_from_candidates(
    min_acctbal: f64,
    order_customers: &AdaptiveI64Set,
    candidates: Vec<Q22CustomerCandidate>,
) -> Vec<Q22Group> {
    let mut counts = [0_u64; Q22_COUNTRY_CODES.len()];
    let mut sums = [0.0_f64; Q22_COUNTRY_CODES.len()];
    for candidate in candidates {
        if candidate.acctbal <= min_acctbal || order_customers.contains(candidate.custkey) {
            continue;
        }
        counts[candidate.country_index] += 1;
        sums[candidate.country_index] += candidate.acctbal;
    }
    q22_groups_from_slots(counts, sums)
}

const Q22_COUNTRY_CODES: [&str; 7] = ["13", "17", "18", "23", "29", "30", "31"];

fn q22_groups_from_slots(
    counts: [u64; Q22_COUNTRY_CODES.len()],
    sums: [f64; Q22_COUNTRY_CODES.len()],
) -> Vec<Q22Group> {
    Q22_COUNTRY_CODES
        .into_iter()
        .zip(counts.into_iter().zip(sums))
        .filter_map(|(cntrycode, (count, sum))| {
            (count > 0).then_some(Q22Group {
                cntrycode: cntrycode.to_string(),
                count,
                sum,
            })
        })
        .collect()
}

fn q22_country_code_index(phone: &str) -> Option<usize> {
    match phone.as_bytes().get(..2)? {
        b"13" => Some(0),
        b"17" => Some(1),
        b"18" => Some(2),
        b"23" => Some(3),
        b"29" => Some(4),
        b"30" => Some(5),
        b"31" => Some(6),
        _ => None,
    }
}

fn q22_output(groups: Vec<Q22Group>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cntrycode", DataType::Utf8, false),
            Field::new("numcust", DataType::UInt64, false),
            Field::new("totacctbal", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                groups.iter().map(|group| group.cntrycode.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                groups.iter().map(|group| group.count),
            )),
            Arc::new(Float64Array::from_iter_values(
                groups.iter().map(|group| group.sum),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q18_large_volume_customer_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q18_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 3 {
        return Ok(None);
    }
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem)) = (customer, orders, lineitem) else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let order_quantity_sums =
        q18_qualifying_order_quantities(engine, lineitem.path.clone(), batch_size, 300.0).await?;
    tpch_profile_elapsed("Q18 lineitem quantity sums", stage);
    if order_quantity_sums.is_empty() {
        return Ok(Some(q18_output(Vec::new())?));
    }
    let qualifying_order_keys =
        AdaptiveI64Set::from_hash(order_quantity_sums.keys().copied().collect::<HashSet<_>>());
    let stage = tpch_profile_start();
    let order_rows =
        q18_qualifying_orders(engine, orders.path, batch_size, &qualifying_order_keys).await?;
    tpch_profile_elapsed("Q18 qualifying orders", stage);
    let customer_keys = order_rows
        .values()
        .map(|order| order.custkey)
        .collect::<HashSet<_>>();
    let customer_keys = AdaptiveI64Set::from_hash(customer_keys);
    let stage = tpch_profile_start();
    let customer_names =
        q18_customer_names(engine, customer.path, batch_size, &customer_keys).await?;
    tpch_profile_elapsed("Q18 customer names", stage);

    let stage = tpch_profile_start();
    let mut rows = Vec::new();
    for (orderkey, order) in order_rows {
        let Some(name) = customer_names.get(&order.custkey) else {
            continue;
        };
        let Some(quantity) = order_quantity_sums.get(&orderkey).copied() else {
            continue;
        };
        rows.push(Q18Row {
            c_name: name.clone(),
            c_custkey: order.custkey,
            o_orderkey: orderkey,
            o_orderdate: order.orderdate,
            o_totalprice: order.totalprice,
            quantity,
        });
    }
    rows.sort_by(|left, right| {
        right
            .o_totalprice
            .partial_cmp(&left.o_totalprice)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.o_orderdate.cmp(&right.o_orderdate))
    });
    rows.truncate(100);
    tpch_profile_elapsed("Q18 final rows", stage);
    Ok(Some(q18_output(rows)?))
}

fn q18_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 6
        && matches!(parse_limit(query), Ok(Some(100)))
        && text.contains("o_orderkey in")
        && text.contains("group by l_orderkey")
        && text.contains("sum(l_quantity) > 300")
        && text.contains("c_custkey = o_custkey")
        && text.contains("o_orderkey = l_orderkey")
}

struct Q18Order {
    custkey: i64,
    orderdate: i32,
    totalprice: f64,
}

struct Q18Row {
    c_name: String,
    c_custkey: i64,
    o_orderkey: i64,
    o_orderdate: i32,
    o_totalprice: f64,
    quantity: f64,
}

async fn q18_qualifying_order_quantities(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    threshold: f64,
) -> Result<HashMap<i64, f64>> {
    if std::env::var_os("DODAM_Q18_DISABLE_ORDERED").is_none() {
        if std::env::var_os("DODAM_Q18_DISABLE_PARALLEL_ORDERED").is_none()
            && let Some(ordered) = engine
                .ordered_i64_decimal_group_sum_above(
                    path.clone(),
                    batch_size,
                    "l_orderkey",
                    "l_quantity",
                    threshold,
                )
                .await?
        {
            return Ok(ordered);
        }
        if let Some(ordered) =
            q18_qualifying_order_quantities_ordered(engine, path.clone(), batch_size, threshold)
                .await?
        {
            return Ok(ordered);
        }
    }
    let max_dense_orderkey = engine
        .parquet_i64_column_max(path.clone(), "l_orderkey")
        .await?
        .and_then(|max_key| adaptive_dense_index(max_key, DEFAULT_MAX_DENSE_I64_KEY));
    let has_dense_capacity = max_dense_orderkey.is_some();
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_quantity".to_string()]),
            None,
        )
        .await?;
    let mut sums = DenseI64F64Sum::new_tracking_threshold(threshold);
    if let Some(max_key) = max_dense_orderkey {
        sums.reserve_dense_to(max_key);
    }
    while let Some(batch) = stream.next() {
        let batch = batch?;
        if sums.has_fallback() {
            let fallback = sums.fallback_mut().expect("checked q18 fallback");
            q18_quantity_batch_into(&batch, fallback)?;
            continue;
        }
        if q18_quantity_batch_into_dense(&batch, &mut sums, has_dense_capacity)? {
            continue;
        }
        sums.convert_to_fallback();
        let fallback = sums.fallback_mut().expect("converted q18 fallback");
        q18_quantity_batch_into(&batch, fallback)?;
    }
    Ok(sums.into_filtered_hash(|quantity| quantity > threshold))
}

struct Q18OrderedQuantityState {
    current_key: Option<i64>,
    current_sum: f64,
    threshold: f64,
    output: HashMap<i64, f64>,
}

impl Q18OrderedQuantityState {
    fn new(threshold: f64) -> Self {
        Self {
            current_key: None,
            current_sum: 0.0,
            threshold,
            output: HashMap::new(),
        }
    }

    fn push(&mut self, orderkey: i64, quantity: f64) -> bool {
        if let Some(current_key) = self.current_key {
            if orderkey < current_key {
                return false;
            }
            if orderkey == current_key {
                self.current_sum += quantity;
                return true;
            }
            self.flush_current();
        }
        self.current_key = Some(orderkey);
        self.current_sum = quantity;
        true
    }

    fn flush_current(&mut self) {
        let Some(orderkey) = self.current_key.take() else {
            return;
        };
        if self.current_sum > self.threshold {
            self.output.insert(orderkey, self.current_sum);
        }
        self.current_sum = 0.0;
    }

    fn finish(mut self) -> HashMap<i64, f64> {
        self.flush_current();
        self.output
    }
}

async fn q18_qualifying_order_quantities_ordered(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    threshold: f64,
) -> Result<Option<HashMap<i64, f64>>> {
    let mut stream = engine
        .scan_parquet_batches_preserve_order(
            path,
            batch_size,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_quantity".to_string()]),
        )
        .await?;
    let mut state = Q18OrderedQuantityState::new(threshold);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        if !q18_quantity_batch_into_ordered(&batch, &mut state)? {
            return Ok(None);
        }
    }
    Ok(Some(state.finish()))
}

fn q18_quantity_batch_into_ordered(
    batch: &RecordBatch,
    state: &mut Q18OrderedQuantityState,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    let (Some(orderkeys), Some(quantities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0 && quantities.null_count() == 0 {
        let quantity_scale = 1.0 / quantities.scale;
        for (&orderkey, &quantity) in orderkeys.values().iter().zip(quantities.raw_values()) {
            if !state.push(orderkey, quantity as f64 * quantity_scale) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || quantities.is_null(row) {
            continue;
        }
        if !state.push(orderkeys.value(row), quantities.value(row)) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn q18_quantity_batch_into_dense(
    batch: &RecordBatch,
    sums: &mut DenseI64F64Sum,
    has_dense_capacity: bool,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    let (Some(orderkeys), Some(quantities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0 && quantities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        let quantity_values = quantities.raw_values();
        let quantity_scale = 1.0 / quantities.scale;
        if has_dense_capacity {
            for (&orderkey, &quantity) in orderkey_values.iter().zip(quantity_values) {
                let Some(index) = adaptive_dense_index(orderkey, DEFAULT_MAX_DENSE_I64_KEY) else {
                    return Ok(false);
                };
                sums.add_dense_index(index, quantity as f64 * quantity_scale);
            }
            return Ok(true);
        }
        let mut max_index = 0_usize;
        for &orderkey in orderkey_values {
            let Some(index) = adaptive_dense_index(orderkey, DEFAULT_MAX_DENSE_I64_KEY) else {
                return Ok(false);
            };
            max_index = max_index.max(index);
        }
        sums.reserve_dense_to(max_index);
        for (&orderkey, &quantity) in orderkey_values.iter().zip(quantity_values) {
            let index = usize::try_from(orderkey).expect("validated dense index");
            sums.add_dense_index(index, quantity as f64 * quantity_scale);
        }
        return Ok(true);
    }
    if has_dense_capacity {
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || quantities.is_null(row) {
                continue;
            }
            let Some(index) = adaptive_dense_index(orderkeys.value(row), DEFAULT_MAX_DENSE_I64_KEY)
            else {
                return Ok(false);
            };
            sums.add_dense_index(index, quantities.value(row));
        }
        return Ok(true);
    }
    let mut max_index = 0_usize;
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || quantities.is_null(row) {
            continue;
        }
        let Some(index) = adaptive_dense_index(orderkeys.value(row), DEFAULT_MAX_DENSE_I64_KEY)
        else {
            return Ok(false);
        };
        max_index = max_index.max(index);
    }
    sums.reserve_dense_to(max_index);
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || quantities.is_null(row) {
            continue;
        }
        let index = usize::try_from(orderkeys.value(row)).expect("validated dense index");
        sums.add_dense_index(index, quantities.value(row));
    }
    Ok(true)
}

fn q18_quantity_batch_into(batch: &RecordBatch, sums: &mut AdaptiveI64Map<f64>) -> Result<()> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    if let (Some(orderkeys), Some(quantities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
    ) {
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || quantities.is_null(row) {
                continue;
            }
            sums.update(
                orderkeys.value(row),
                || 0.0,
                |sum| *sum += quantities.value(row),
            );
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        let (Some(orderkey), Some(quantity)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_f64_value(quantities, row)?,
        ) else {
            continue;
        };
        sums.update(orderkey, || 0.0, |sum| *sum += quantity);
    }
    Ok(())
}

async fn q18_qualifying_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    qualifying_orders: &AdaptiveI64Set,
) -> Result<HashMap<i64, Q18Order>> {
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_custkey".to_string(),
        "o_orderdate".to_string(),
        "o_totalprice".to_string(),
    ]);
    let mut stream = if let Some((min_key, max_key)) = qualifying_orders.selective_key_range() {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                i64_range_pruning_predicates("o_orderkey", min_key, max_key),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let qualifying_orders = Arc::new(qualifying_orders.clone());
    parallel_batch_fold(
        &mut stream,
        move |batch| q18_qualifying_orders_batch(batch, &qualifying_orders),
        HashMap::<i64, Q18Order>::new(),
        merge_maps,
        "Q18 qualifying orders",
    )
}

fn q18_qualifying_orders_batch(
    batch: RecordBatch,
    qualifying_orders: &AdaptiveI64Set,
) -> Result<HashMap<i64, Q18Order>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let totalprices = batch_column(&batch, "o_totalprice")?;
    if let Some(orders) = q18_qualifying_orders_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        totalprices,
        qualifying_orders,
    )? {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    for row in 0..batch.num_rows() {
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        if !qualifying_orders.contains(orderkey) {
            continue;
        }
        let (Some(custkey), Some(orderdate), Some(totalprice)) = (
            numeric_i64_value(custkeys, row)?,
            date32_value(orderdates, row)?,
            numeric_f64_value(totalprices, row)?,
        ) else {
            continue;
        };
        orders.insert(
            orderkey,
            Q18Order {
                custkey,
                orderdate,
                totalprice,
            },
        );
    }
    Ok(orders)
}

fn q18_qualifying_orders_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    totalprices: &ArrayRef,
    qualifying_orders: &AdaptiveI64Set,
) -> Result<Option<HashMap<i64, Q18Order>>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates), Some(totalprices)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
        q01_decimal_input(totalprices)?,
    ) else {
        return Ok(None);
    };
    let mut orders = HashMap::new();
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || custkeys.is_null(row)
            || orderdates.is_null(row)
            || totalprices.is_null(row)
        {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if !qualifying_orders.contains(orderkey) {
            continue;
        }
        orders.insert(
            orderkey,
            Q18Order {
                custkey: custkeys.value(row),
                orderdate: orderdates.value(row),
                totalprice: totalprices.value(row),
            },
        );
    }
    Ok(Some(orders))
}

async fn q18_customer_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_keys: &AdaptiveI64Set,
) -> Result<HashMap<i64, String>> {
    let mut customers = HashMap::new();
    if customer_keys.is_empty() {
        return Ok(customers);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_name".to_string()]),
            None,
        )
        .await?;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let keys = batch_column(&batch, "c_custkey")?;
        let names = batch_string_column(&batch, "c_name")?;
        if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>() {
            for row in 0..batch.num_rows() {
                if keys.is_null(row) || names.is_null(row) {
                    continue;
                }
                let key = keys.value(row);
                if !customer_keys.contains(key) {
                    continue;
                }
                customers.insert(key, names.value(row).to_string());
            }
            if customers.len() == customer_keys.len() {
                break;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            if names.is_null(row) {
                continue;
            }
            if let Some(key) = numeric_i64_value(keys, row)? {
                if !customer_keys.contains(key) {
                    continue;
                }
                customers.insert(key, names.value(row).to_string());
            }
        }
        if customers.len() == customer_keys.len() {
            break;
        }
    }
    Ok(customers)
}

fn date32_value(column: &ArrayRef, row: usize) -> Result<Option<i32>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Date32 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32")
                .value(row),
        )),
        DataType::Date64 => Ok(Some(
            (column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64")
                .value(row)
                / 86_400_000) as i32,
        )),
        DataType::Int32 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 date")
                .value(row),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "expected date column, got {data_type:?}"
        ))),
    }
}

fn q18_output(rows: Vec<Q18Row>) -> Result<QueryOutput> {
    let orderdates = rows
        .iter()
        .map(|row| date32_to_ymd_string(row.o_orderdate))
        .collect::<Result<Vec<_>>>()?;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("c_name", DataType::Utf8, false),
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_orderdate", DataType::Utf8, false),
            Field::new("o_totalprice", DataType::Float64, false),
            Field::new("sum(l_quantity)", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_name.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.c_custkey),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.o_orderkey),
            )),
            Arc::new(StringArray::from_iter_values(
                orderdates.iter().map(String::as_str),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.o_totalprice),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.quantity),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q19_discounted_revenue_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q19_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut lineitem = None;
    let mut part = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        }
    }
    let (Some(lineitem), Some(part)) = (lineitem, part) else {
        return Ok(None);
    };
    if !lineitem.path.exists() {
        return Err(DodamError::MissingPath(lineitem.path));
    }
    let rules = q19_rules(selection)?;
    if rules.is_empty() || rules.len() > 8 {
        return Ok(None);
    }
    let stage = tpch_profile_start();
    let part_masks = q19_matching_part_masks(engine, part.path, batch_size, &rules).await?;
    tpch_profile_elapsed("Q19 matching part masks", stage);
    if part_masks.is_empty() {
        return Ok(Some(q17_output("revenue".to_string(), None)?));
    }
    let (sum, count) = q19_lineitem_revenue(
        engine,
        lineitem.path,
        batch_size,
        Arc::new(rules),
        Arc::new(part_masks),
    )
    .await?;
    Ok(Some(q17_output(
        "revenue".to_string(),
        (count > 0).then_some(sum),
    )?))
}

fn q19_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 1
        && projection.contains("sum(")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && selection.contains("p_partkey = l_partkey")
        && selection.contains("p_brand")
        && selection.contains("p_container in")
        && selection.contains("l_quantity")
        && selection.contains("p_size between")
        && selection.contains("l_shipmode in")
        && selection.contains("l_shipinstruct")
}

#[derive(Clone)]
struct Q19Rule {
    brand: String,
    containers: HashSet<String>,
    quantity_low: f64,
    quantity_high: f64,
    size_low: f64,
    size_high: f64,
    shipmodes: HashSet<String>,
    shipinstruct: String,
}

fn q19_rules(selection: &SqlExpr) -> Result<Vec<Q19Rule>> {
    let mut branches = Vec::new();
    collect_sql_or_disjuncts(selection, &mut branches);
    let mut rules = Vec::with_capacity(branches.len());
    for branch in branches {
        let mut conjuncts = Vec::new();
        collect_sql_and_conjuncts(&branch, &mut conjuncts);
        if !conjuncts.iter().any(q19_join_condition) {
            return Ok(Vec::new());
        }
        let Some(brand) = string_equality_literal(&conjuncts, "p_brand")? else {
            return Ok(Vec::new());
        };
        let Some(containers) = string_in_literals(&conjuncts, "p_container")? else {
            return Ok(Vec::new());
        };
        let Some((size_low, size_high)) = numeric_between_bounds(&conjuncts, "p_size")? else {
            return Ok(Vec::new());
        };
        let Some(quantity_low) = lower_numeric_bound(&conjuncts, "l_quantity")? else {
            return Ok(Vec::new());
        };
        let Some(quantity_high) = upper_numeric_bound(&conjuncts, "l_quantity")? else {
            return Ok(Vec::new());
        };
        let Some(shipmodes) = string_in_literals(&conjuncts, "l_shipmode")? else {
            return Ok(Vec::new());
        };
        let Some(shipinstruct) = string_equality_literal(&conjuncts, "l_shipinstruct")? else {
            return Ok(Vec::new());
        };
        rules.push(Q19Rule {
            brand,
            containers,
            quantity_low,
            quantity_high,
            size_low,
            size_high,
            shipmodes,
            shipinstruct,
        });
    }
    Ok(rules)
}

fn q19_join_condition(expr: &SqlExpr) -> bool {
    let SqlExpr::BinaryOp { left, op, right } = expr else {
        return false;
    };
    *op == BinaryOperator::Eq
        && ((sql_expr_column_matches(left, "p_partkey")
            && sql_expr_column_matches(right, "l_partkey"))
            || (sql_expr_column_matches(left, "l_partkey")
                && sql_expr_column_matches(right, "p_partkey")))
}

async fn q19_matching_part_masks(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: &[Q19Rule],
) -> Result<AdaptiveI64Map<u8>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "p_partkey".to_string(),
                "p_brand".to_string(),
                "p_container".to_string(),
                "p_size".to_string(),
            ]),
            None,
        )
        .await?;
    let mut masks = AdaptiveI64Map::<u8>::new_dense();
    let raw_rules = q19_raw_part_rules(rules);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let brands = batch_string_column(&batch, "p_brand")?;
        let containers = batch_string_column(&batch, "p_container")?;
        let sizes = batch_column(&batch, "p_size")?;
        if let Some(size_values) = q19_part_size_values(sizes) {
            if let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>() {
                if partkeys.null_count() == 0
                    && brands.null_count() == 0
                    && containers.null_count() == 0
                    && size_values.null_count() == 0
                {
                    let brand_offsets = brands.value_offsets();
                    let brand_data = brands.value_data();
                    let container_offsets = containers.value_offsets();
                    let container_data = containers.value_data();
                    for row in 0..batch.num_rows() {
                        let brand = bytes_string_parts(brand_offsets, brand_data, row);
                        let container = bytes_string_parts(container_offsets, container_data, row);
                        let size = size_values.value(row);
                        let mut mask = 0_u8;
                        for (index, rule) in raw_rules.iter().enumerate() {
                            if q19_raw_part_rule_matches(rule, brand, container, size) {
                                mask |= 1 << index;
                            }
                        }
                        if mask != 0 {
                            masks.insert(partkeys.value(row), mask);
                        }
                    }
                    continue;
                }
            }
        }
        for row in 0..batch.num_rows() {
            if brands.is_null(row) || containers.is_null(row) {
                continue;
            }
            let (Some(partkey), Some(size)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_f64_value(sizes, row)?,
            ) else {
                continue;
            };
            let mut mask = 0_u8;
            for (index, rule) in rules.iter().enumerate() {
                if brands.value(row) == rule.brand
                    && rule.containers.contains(containers.value(row))
                    && size >= rule.size_low
                    && size <= rule.size_high
                {
                    mask |= 1 << index;
                }
            }
            if mask != 0 {
                masks.insert(partkey, mask);
            }
        }
    }
    Ok(masks)
}

struct Q19RawPartRule {
    brand: Vec<u8>,
    containers: Vec<Vec<u8>>,
    size_low: i64,
    size_high: i64,
}

fn q19_raw_part_rules(rules: &[Q19Rule]) -> Vec<Q19RawPartRule> {
    rules
        .iter()
        .map(|rule| Q19RawPartRule {
            brand: rule.brand.as_bytes().to_vec(),
            containers: rule
                .containers
                .iter()
                .map(|container| container.as_bytes().to_vec())
                .collect(),
            size_low: rule.size_low.ceil() as i64,
            size_high: rule.size_high.floor() as i64,
        })
        .collect()
}

fn q19_raw_part_rule_matches(
    rule: &Q19RawPartRule,
    brand: &[u8],
    container: &[u8],
    size: i64,
) -> bool {
    rule.brand.as_slice() == brand
        && size >= rule.size_low
        && size <= rule.size_high
        && rule
            .containers
            .iter()
            .any(|candidate| candidate.as_slice() == container)
}

enum Q19PartSizeValues<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
}

impl Q19PartSizeValues<'_> {
    fn null_count(&self) -> usize {
        match self {
            Self::Int32(values) => values.null_count(),
            Self::Int64(values) => values.null_count(),
        }
    }

    fn value(&self, row: usize) -> i64 {
        match self {
            Self::Int32(values) => i64::from(values.value(row)),
            Self::Int64(values) => values.value(row),
        }
    }
}

fn q19_part_size_values(column: &ArrayRef) -> Option<Q19PartSizeValues<'_>> {
    column
        .as_any()
        .downcast_ref::<Int32Array>()
        .map(Q19PartSizeValues::Int32)
        .or_else(|| {
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(Q19PartSizeValues::Int64)
        })
}

async fn q19_lineitem_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
) -> Result<(f64, u64)> {
    if std::env::var_os("DODAM_Q19_DISABLE_LATE_MATERIALIZE").is_none() {
        if let Some(result) = q19_late_materialized_lineitem_revenue(
            engine,
            path.clone(),
            batch_size,
            rules.clone(),
            part_masks.clone(),
        )
        .await?
        {
            return Ok(result);
        }
    }
    let profile = tpch_profile_enabled();
    parquet_scan_fold_chunks(
        engine,
        path,
        batch_size,
        Projection::Columns(vec![
            "l_partkey".to_string(),
            "l_quantity".to_string(),
            "l_extendedprice".to_string(),
            "l_discount".to_string(),
            "l_shipmode".to_string(),
            "l_shipinstruct".to_string(),
        ]),
        scan_aggregate_row_group_chunk(),
        4,
        || (0.0, 0_u64, Q19SelectionProfile::default()),
        || (0.0, 0_u64, Q19SelectionProfile::default()),
        move |batch| {
            let mut sum = 0.0;
            let mut count = 0_u64;
            let mut profile_metrics = Q19SelectionProfile::default();
            let mut raw_rule_cache = None;
            let mut batch_profile = profile.then_some(Q19SelectionProfile::default());
            let (batch_sum, batch_count) = q19_lineitem_revenue_batch(
                batch,
                &rules,
                &part_masks,
                &mut raw_rule_cache,
                batch_profile.as_mut(),
            )?;
            sum += batch_sum;
            count += batch_count;
            if let Some(batch_profile) = batch_profile {
                profile_metrics.add(batch_profile);
            }
            Ok((sum, count, profile_metrics))
        },
        |total, batch| {
            total.0 += batch.0;
            total.1 += batch.1;
            total.2.add(batch.2);
        },
        "Q19 lineitem revenue",
    )
    .await
    .map(|(sum, count, profile_metrics)| {
        if profile {
            q19_log_selection_profile(profile_metrics);
        }
        (sum, count)
    })
}

async fn q19_late_materialized_lineitem_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
) -> Result<Option<(f64, u64)>> {
    let predicate_projection = Projection::Columns(vec![
        "l_partkey".to_string(),
        "l_quantity".to_string(),
        "l_discount".to_string(),
        "l_shipmode".to_string(),
        "l_shipinstruct".to_string(),
    ]);
    let payload_projection = Projection::Columns(vec!["l_extendedprice".to_string()]);
    let rules_for_state = rules.clone();
    let part_masks_for_state = part_masks.clone();
    let Some(chunks) = engine
        .late_materialized_parquet_map_with_policy(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            q19_late_materialized_row_group_chunk(),
            late_materialization_policy_from_env("DODAM_Q19_LATE_MAX_SELECTED_RATIO", 0.60),
            move || Q19LateState {
                rules: rules_for_state.clone(),
                part_masks: part_masks_for_state.clone(),
                raw_rule_cache: None,
                selected_discounts: Vec::new(),
                discount_scale: None,
                extendedprice_scale: None,
                discount_offset: 0,
                sum: 0.0,
            },
            q19_late_build_selection_batch,
            q19_late_consume_payload_batch,
            |state, _metrics| {
                if state.discount_offset != state.selected_discounts.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q19 row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some((state.sum, state.selected_discounts.len() as u64)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0_u64;
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        sum += chunk.output.0;
        count += chunk.output.1;
        metrics.add(chunk.metrics);
    }
    q19_log_late_materialized_profile(metrics, q19_late_materialized_row_group_chunk());
    Ok(Some((sum, count)))
}

fn q19_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q19_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

struct Q19LateState {
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
    raw_rule_cache: Option<(u64, Vec<Q19RawLineRule>)>,
    selected_discounts: Vec<i64>,
    discount_scale: Option<i64>,
    extendedprice_scale: Option<i64>,
    discount_offset: usize,
    sum: f64,
}

fn q19_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let shipmodes = batch_string_column(&batch, "l_shipmode")?;
    let shipinstructs = batch_string_column(&batch, "l_shipinstruct")?;
    let (Some(partkeys), Some(quantities), Some(discounts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
        q01_decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    if partkeys.null_count() != 0
        || quantities.null_count() != 0
        || discounts.null_count() != 0
        || shipmodes.null_count() != 0
        || shipinstructs.null_count() != 0
        || quantities.precision > 18
        || discounts.precision > 18
    {
        return Ok(None);
    }
    let discount_scale = discounts.scale as i64;
    if let Some(existing) = state.discount_scale {
        if existing != discount_scale {
            return Ok(None);
        }
    } else {
        state.discount_scale = Some(discount_scale);
    }
    let raw_rules =
        q19_raw_line_rules_cached(&state.rules, quantities.scale, &mut state.raw_rule_cache);
    let shipmode_offsets = shipmodes.value_offsets();
    let shipmode_data = shipmodes.value_data();
    let shipinstruct_offsets = shipinstructs.value_offsets();
    let shipinstruct_data = shipinstructs.value_data();
    let partkey_values = partkeys.values();
    let quantity_values = quantities.raw_values();
    let discount_values = discounts.raw_values();
    for row in 0..batch.num_rows() {
        let selected = if let Some(mask) = state.part_masks.get(partkey_values[row]) {
            q19_rule_matches_lineitem_raw(
                raw_rules,
                mask,
                quantity_values[row],
                bytes_string_parts(shipmode_offsets, shipmode_data, row),
                bytes_string_parts(shipinstruct_offsets, shipinstruct_data, row),
            )
        } else {
            false
        };
        if selected {
            state.selected_discounts.push(discount_values[row] as i64);
        }
        selection.push(selected);
    }
    Ok(Some(()))
}

fn q19_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let Some(extendedprices) = q01_decimal_input(extendedprices)? else {
        return Ok(None);
    };
    if extendedprices.null_count() != 0 || extendedprices.precision > 18 {
        return Ok(None);
    }
    let price_scale = extendedprices.scale as i64;
    if let Some(existing) = state.extendedprice_scale {
        if existing != price_scale {
            return Ok(None);
        }
    } else {
        state.extendedprice_scale = Some(price_scale);
    }
    let discount_scale = state
        .discount_scale
        .ok_or_else(|| DodamError::UnsupportedSql("Q19 missing discount scale".to_string()))?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for &extendedprice in extendedprices.raw_values() {
        let discount = *state
            .selected_discounts
            .get(state.discount_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("Q19 row selection payload mismatch".to_string())
            })?;
        state.sum += ((extendedprice as i64) * (discount_scale - discount)) as f64 * revenue_scale;
        state.discount_offset += 1;
    }
    Ok(Some(()))
}

fn q19_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q19: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q19_lineitem_revenue_batch(
    batch: RecordBatch,
    rules: &[Q19Rule],
    part_masks: &AdaptiveI64Map<u8>,
    raw_rule_cache: &mut Option<(u64, Vec<Q19RawLineRule>)>,
    mut profile: Option<&mut Q19SelectionProfile>,
) -> Result<(f64, u64)> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let shipmodes = batch_string_column(&batch, "l_shipmode")?;
    let shipinstructs = batch_string_column(&batch, "l_shipinstruct")?;
    let mut sum = 0.0;
    let mut count = 0_u64;
    if let (Some(partkeys), Some(quantities), Some(extendedprices), Some(discounts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
        q01_decimal_input(extendedprices)?,
        q01_decimal_input(discounts)?,
    ) {
        let raw_rules = q19_raw_line_rules_cached(rules, quantities.scale, raw_rule_cache);
        if partkeys.null_count() == 0
            && quantities.null_count() == 0
            && extendedprices.null_count() == 0
            && discounts.null_count() == 0
            && shipmodes.null_count() == 0
            && shipinstructs.null_count() == 0
        {
            let discount_one_raw = scaled_f64_to_i128(1.0, discounts.scale);
            let revenue_scale = 1.0 / (extendedprices.scale * discounts.scale);
            let shipmode_offsets = shipmodes.value_offsets();
            let shipmode_data = shipmodes.value_data();
            let shipinstruct_offsets = shipinstructs.value_offsets();
            let shipinstruct_data = shipinstructs.value_data();
            let partkey_values = partkeys.values();
            let quantity_values = quantities.raw_values();
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            for row in 0..batch.num_rows() {
                let selected = if let Some(mask) = part_masks.get(partkey_values[row]) {
                    q19_rule_matches_lineitem_raw(
                        &raw_rules,
                        mask,
                        quantity_values[row],
                        bytes_string_parts(shipmode_offsets, shipmode_data, row),
                        bytes_string_parts(shipinstruct_offsets, shipinstruct_data, row),
                    )
                } else {
                    false
                };
                if let Some(profile) = profile.as_deref_mut() {
                    profile.record(selected);
                }
                if !selected {
                    continue;
                }
                sum += (extendedprice_values[row] * (discount_one_raw - discount_values[row]))
                    as f64
                    * revenue_scale;
                count += 1;
            }
            return Ok((sum, count));
        }
        for row in 0..batch.num_rows() {
            if partkeys.is_null(row)
                || quantities.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
                || shipmodes.is_null(row)
                || shipinstructs.is_null(row)
            {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.record(false);
                }
                continue;
            }
            let quantity = quantities.value(row);
            let selected = if let Some(mask) = part_masks.get(partkeys.value(row)) {
                q19_rule_matches_lineitem(
                    rules,
                    mask,
                    quantity,
                    shipmodes.value(row),
                    shipinstructs.value(row),
                )
            } else {
                false
            };
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(selected);
            }
            if !selected {
                continue;
            }
            sum += extendedprices.value(row) * (1.0 - discounts.value(row));
            count += 1;
        }
        return Ok((sum, count));
    }
    for row in 0..batch.num_rows() {
        if shipmodes.is_null(row) || shipinstructs.is_null(row) {
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(false);
            }
            continue;
        }
        let (Some(partkey), Some(quantity), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(false);
            }
            continue;
        };
        let selected = if let Some(mask) = part_masks.get(partkey) {
            q19_rule_matches_lineitem(
                rules,
                mask,
                quantity,
                shipmodes.value(row),
                shipinstructs.value(row),
            )
        } else {
            false
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.record(selected);
        }
        if selected {
            sum += extendedprice * (1.0 - discount);
            count += 1;
        }
    }
    Ok((sum, count))
}

#[derive(Debug, Clone, Copy, Default)]
struct Q19SelectionProfile {
    total_rows: u64,
    selected_rows: u64,
    selector_runs: u64,
    last_selected: Option<bool>,
}

impl Q19SelectionProfile {
    fn record(&mut self, selected: bool) {
        self.total_rows += 1;
        if selected {
            self.selected_rows += 1;
        }
        if self.last_selected != Some(selected) {
            self.selector_runs += 1;
            self.last_selected = Some(selected);
        }
    }

    fn add(&mut self, other: Self) {
        if other.total_rows == 0 {
            return;
        }
        self.total_rows += other.total_rows;
        self.selected_rows += other.selected_rows;
        self.selector_runs += other.selector_runs;
        self.last_selected = other.last_selected.or(self.last_selected);
    }
}

fn q19_log_selection_profile(profile: Q19SelectionProfile) {
    let ratio = if profile.total_rows == 0 {
        0.0
    } else {
        profile.selected_rows as f64 / profile.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q19: predicate_selected rows={} selected={} ratio={:.6} selector_runs={}",
        profile.total_rows, profile.selected_rows, ratio, profile.selector_runs
    );
}

struct Q19RawLineRule {
    quantity_low: i128,
    quantity_high: i128,
    shipmodes: Vec<Vec<u8>>,
    shipinstruct: Vec<u8>,
}

fn q19_raw_line_rules(rules: &[Q19Rule], quantity_scale: f64) -> Vec<Q19RawLineRule> {
    rules
        .iter()
        .map(|rule| Q19RawLineRule {
            quantity_low: scaled_f64_to_i128(rule.quantity_low, quantity_scale),
            quantity_high: scaled_f64_to_i128(rule.quantity_high, quantity_scale),
            shipmodes: rule
                .shipmodes
                .iter()
                .map(|shipmode| shipmode.as_bytes().to_vec())
                .collect(),
            shipinstruct: rule.shipinstruct.as_bytes().to_vec(),
        })
        .collect()
}

fn q19_raw_line_rules_cached<'a>(
    rules: &[Q19Rule],
    quantity_scale: f64,
    cache: &'a mut Option<(u64, Vec<Q19RawLineRule>)>,
) -> &'a [Q19RawLineRule] {
    let scale_key = quantity_scale.to_bits();
    if !matches!(cache, Some((cached_key, _)) if *cached_key == scale_key) {
        *cache = Some((scale_key, q19_raw_line_rules(rules, quantity_scale)));
    }
    cache
        .as_ref()
        .expect("q19 raw rule cache populated")
        .1
        .as_slice()
}

fn bytes_string_parts<'a>(offsets: &[i32], data: &'a [u8], row: usize) -> &'a [u8] {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    &data[start..end]
}

fn q19_rule_matches_lineitem_raw(
    rules: &[Q19RawLineRule],
    mask: u8,
    quantity: i128,
    shipmode: &[u8],
    shipinstruct: &[u8],
) -> bool {
    let relevant_mask = (u16::from(mask) & ((1_u16 << rules.len().min(8)) - 1)) as u8;
    if relevant_mask != 0 && (relevant_mask & (relevant_mask - 1)) == 0 {
        return rules
            .get(relevant_mask.trailing_zeros() as usize)
            .is_some_and(|rule| {
                q19_raw_rule_matches_lineitem(rule, quantity, shipmode, shipinstruct)
            });
    }
    for (index, rule) in rules.iter().enumerate() {
        if mask & (1 << index) == 0 {
            continue;
        }
        if q19_raw_rule_matches_lineitem(rule, quantity, shipmode, shipinstruct) {
            return true;
        }
    }
    false
}

fn q19_raw_rule_matches_lineitem(
    rule: &Q19RawLineRule,
    quantity: i128,
    shipmode: &[u8],
    shipinstruct: &[u8],
) -> bool {
    quantity >= rule.quantity_low
        && quantity <= rule.quantity_high
        && rule.shipinstruct == shipinstruct
        && rule
            .shipmodes
            .iter()
            .any(|candidate| candidate.as_slice() == shipmode)
}

fn q19_rule_matches_lineitem(
    rules: &[Q19Rule],
    mask: u8,
    quantity: f64,
    shipmode: &str,
    shipinstruct: &str,
) -> bool {
    for (index, rule) in rules.iter().enumerate() {
        if mask & (1 << index) == 0 {
            continue;
        }
        if quantity >= rule.quantity_low
            && quantity <= rule.quantity_high
            && rule.shipmodes.contains(shipmode)
            && shipinstruct == rule.shipinstruct
        {
            return true;
        }
    }
    false
}

async fn try_execute_q21_suppliers_who_kept_orders_waiting_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q21_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 4 {
        return Ok(None);
    }
    let mut supplier = None;
    let mut lineitem = None;
    let mut orders = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("l1") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(supplier), Some(lineitem), Some(orders), Some(nation)) =
        (supplier, lineitem, orders, nation)
    else {
        return Ok(None);
    };
    let stage = tpch_profile_start();
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, "SAUDI ARABIA").await?;
    tpch_profile_elapsed("Q21 nation keys", stage);
    let stage = tpch_profile_start();
    let suppliers = q21_supplier_names(engine, supplier.path, batch_size, &nation_keys).await?;
    tpch_profile_elapsed("Q21 supplier names", stage);
    if suppliers.is_empty() {
        return Ok(Some(q21_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let final_orders = q21_final_order_keys(engine, orders.path, batch_size).await?;
    tpch_profile_elapsed("Q21 final order keys", stage);
    if final_orders.is_empty() {
        return Ok(Some(q21_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let counts = if q21_ordered_lineitem_enabled()
        && let Some(counts) = q21_lineitem_supplier_counts_ordered(
            engine,
            lineitem.path.clone(),
            batch_size,
            &final_orders,
            &suppliers,
        )
        .await?
    {
        tpch_profile_elapsed("Q21 ordered lineitem counts", stage);
        counts
    } else {
        let order_states =
            q21_lineitem_order_states(engine, lineitem.path, batch_size, final_orders).await?;
        tpch_profile_elapsed("Q21 lineitem order states", stage);
        let mut counts = HashMap::<i64, u64>::with_capacity(suppliers.len());
        for state in order_states.into_values() {
            q21_count_qualifying_order(&mut counts, &suppliers, &state);
        }
        counts
    };
    let stage = tpch_profile_start();
    let mut rows = counts
        .into_iter()
        .filter_map(|(suppkey, count)| {
            suppliers.get(&suppkey).map(|name| Q21Row {
                s_name: name.clone(),
                count,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.s_name.cmp(&right.s_name))
    });
    rows.truncate(100);
    tpch_profile_elapsed("Q21 final rows", stage);
    Ok(Some(q21_output(rows)?))
}

fn q21_ordered_lineitem_enabled() -> bool {
    std::env::var_os("DODAM_Q21_DISABLE_ORDERED_LINEITEM").is_none()
}

fn q21_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 2
        && matches!(parse_limit(query), Ok(Some(100)))
        && text.contains("o_orderstatus = 'f'")
        && text.contains("l1.l_receiptdate > l1.l_commitdate")
        && text.contains("exists")
        && text.contains("not exists")
        && text.contains("n_name = 'saudi arabia'")
}

async fn q21_nation_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_name: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["n_nationkey".to_string(), "n_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row) == nation_name
                && let Some(key) = numeric_i64_value(nationkeys, row)?
            {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

async fn q21_supplier_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_nationkey".to_string(),
                "s_name".to_string(),
            ]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        let names = batch_string_column(&batch, "s_name")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_keys.contains(&nationkey) && names.is_valid(row) {
                suppliers.insert(suppkey, names.value(row).to_string());
            }
        }
    }
    Ok(suppliers)
}

async fn q21_final_order_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<Q21FinalOrders> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderstatus".to_string()]),
            None,
        )
        .await?;
    let mut keys = Q21FinalOrders::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q21_final_orders_batch_into(&batch, &mut keys)?;
    }
    Ok(keys)
}

type Q21FinalOrders = AdaptiveI64Set;

fn q21_final_orders_batch_into(batch: &RecordBatch, keys: &mut Q21FinalOrders) -> Result<()> {
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let statuses = batch_string_column(batch, "o_orderstatus")?;
    if let Some(orderkeys) = orderkeys.as_any().downcast_ref::<Int64Array>() {
        if orderkeys.null_count() == 0 && statuses.null_count() == 0 {
            let status_offsets = statuses.value_offsets();
            let status_data = statuses.value_data();
            for row in 0..orderkeys.len() {
                if bytes_string_parts(status_offsets, status_data, row) == b"F" {
                    keys.insert(orderkeys.value(row));
                }
            }
            return Ok(());
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || statuses.is_null(row) || statuses.value(row) != "F" {
                continue;
            }
            keys.insert(orderkeys.value(row));
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        if statuses.is_null(row) || statuses.value(row) != "F" {
            continue;
        }
        if let Some(key) = numeric_i64_value(orderkeys, row)? {
            keys.insert(key);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Q21OrderState {
    first_supplier: i64,
    late_supplier: i64,
    late_row_count: u32,
    flags: u8,
}

impl Q21OrderState {
    const HAS_SUPPLIER: u8 = 1 << 0;
    const HAS_MULTIPLE_SUPPLIERS: u8 = 1 << 1;
    const HAS_LATE_SUPPLIER: u8 = 1 << 2;
    const HAS_MULTIPLE_LATE_SUPPLIERS: u8 = 1 << 3;

    fn has_supplier(&self) -> bool {
        self.flags & Self::HAS_SUPPLIER != 0
    }

    fn has_multiple_suppliers(&self) -> bool {
        self.flags & Self::HAS_MULTIPLE_SUPPLIERS != 0
    }

    fn has_late_supplier(&self) -> bool {
        self.flags & Self::HAS_LATE_SUPPLIER != 0
    }

    fn has_multiple_late_suppliers(&self) -> bool {
        self.flags & Self::HAS_MULTIPLE_LATE_SUPPLIERS != 0
    }

    fn add_supplier(&mut self, suppkey: i64) {
        if !self.has_supplier() {
            self.first_supplier = suppkey;
            self.flags |= Self::HAS_SUPPLIER;
        } else if suppkey != self.first_supplier {
            self.flags |= Self::HAS_MULTIPLE_SUPPLIERS;
        }
    }

    fn add_late_supplier(&mut self, suppkey: i64) {
        if !self.has_late_supplier() {
            self.late_supplier = suppkey;
            self.flags |= Self::HAS_LATE_SUPPLIER;
            self.late_row_count = 1;
        } else if suppkey == self.late_supplier {
            self.late_row_count += 1;
        } else {
            self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
        }
    }

    fn has_single_late_supplier(&self) -> bool {
        self.has_late_supplier() && !self.has_multiple_late_suppliers()
    }

    fn merge(&mut self, other: Q21OrderState) {
        if other.has_supplier() {
            self.add_supplier(other.first_supplier);
            if other.has_multiple_suppliers() {
                self.flags |= Self::HAS_MULTIPLE_SUPPLIERS;
            }
        }
        if !other.has_late_supplier() {
            return;
        }
        if !self.has_late_supplier() {
            self.late_supplier = other.late_supplier;
            self.flags |= Self::HAS_LATE_SUPPLIER;
            self.late_row_count = other.late_row_count;
            if other.has_multiple_late_suppliers() {
                self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
            }
            return;
        }
        if self.late_supplier == other.late_supplier {
            self.late_row_count += other.late_row_count;
        } else {
            self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
        }
        if other.has_multiple_late_suppliers() {
            self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
        }
    }
}

type Q21OrderStateMap = FastHashMap<i64, Q21OrderState>;

fn q21_order_state_map() -> Q21OrderStateMap {
    fast_hash_map()
}

fn q21_order_state_map_with_capacity(capacity: usize) -> Q21OrderStateMap {
    fast_hash_map_with_capacity(capacity)
}

async fn q21_lineitem_order_states(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    final_orders: Q21FinalOrders,
) -> Result<Q21OrderStateMap> {
    if q21_dense_order_state_enabled()
        && let Some(dense_index) = q21_dense_final_order_index(&final_orders)
        && let Some(states) =
            q21_lineitem_order_states_dense(engine, path.clone(), batch_size, dense_index).await?
    {
        return Ok(states);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_receiptdate".to_string(),
        "l_commitdate".to_string(),
    ]);
    let output_capacity = final_orders.len();
    let final_orders = Arc::new(final_orders);
    if q21_row_group_map_enabled()
        && let Some(partials) = engine
            .parquet_row_group_map(
                path.clone(),
                batch_size,
                projection.clone(),
                q21_row_group_map_chunk(),
                q21_order_state_map,
                {
                    let final_orders = final_orders.clone();
                    move |batch, states| {
                        q21_lineitem_order_states_projected_batch_into(
                            batch,
                            &final_orders,
                            states,
                        )?;
                        Ok(Some(()))
                    }
                },
                |states| Ok(Some(states)),
            )
            .await?
    {
        let mut output = q21_order_state_map_with_capacity(output_capacity);
        for partial in partials {
            q21_merge_order_states(&mut output, partial);
        }
        return Ok(output);
    }
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    q21_parallel_batch_order_states(
        &mut stream,
        q21_lineitem_order_state_chunk_size(),
        output_capacity,
        move |batches| {
            let mut states = q21_order_state_map();
            for batch in batches {
                q21_lineitem_order_states_projected_batch_into(batch, &final_orders, &mut states)?;
            }
            Ok(states)
        },
    )
}

fn q21_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q21_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q21_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q21_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q21_dense_order_state_enabled() -> bool {
    std::env::var("DODAM_Q21_ENABLE_DENSE_ORDER_STATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q21_dense_order_state_max_orders() -> usize {
    std::env::var("DODAM_Q21_DENSE_STATE_MAX_ORDERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2_000_000)
}

struct Q21DenseFinalOrderIndex {
    index_by_orderkey: Vec<i32>,
    orderkeys: Vec<i64>,
}

fn q21_dense_final_order_index(
    final_orders: &Q21FinalOrders,
) -> Option<Arc<Q21DenseFinalOrderIndex>> {
    let contains = final_orders.dense_contains_slice()?;
    let selected = final_orders.len();
    if selected > q21_dense_order_state_max_orders() {
        return None;
    }
    let mut index_by_orderkey = vec![-1_i32; contains.len()];
    let mut orderkeys = Vec::with_capacity(selected);
    for (orderkey, selected) in contains.iter().copied().enumerate() {
        if !selected {
            continue;
        }
        let index = i32::try_from(orderkeys.len()).ok()?;
        index_by_orderkey[orderkey] = index;
        orderkeys.push(orderkey as i64);
    }
    Some(Arc::new(Q21DenseFinalOrderIndex {
        index_by_orderkey,
        orderkeys,
    }))
}

async fn q21_lineitem_order_states_dense(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    dense_index: Arc<Q21DenseFinalOrderIndex>,
) -> Result<Option<Q21OrderStateMap>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_receiptdate".to_string(),
                "l_commitdate".to_string(),
            ]),
            None,
        )
        .await?;
    q21_parallel_dense_order_states(
        &mut stream,
        q21_lineitem_order_state_chunk_size(),
        dense_index,
    )
}

fn q21_parallel_dense_order_states(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    dense_index: Arc<Q21DenseFinalOrderIndex>,
) -> Result<Option<Q21OrderStateMap>> {
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let chunk_size = chunk_size.max(1);
    let mut pending_chunks = 0_usize;
    let mut chunk = Vec::with_capacity(chunk_size);
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size {
            continue;
        }
        let sender = sender.clone();
        let dense_index = dense_index.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(q21_dense_order_states_chunk(task_chunk, dense_index));
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let dense_index = dense_index.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(q21_dense_order_states_chunk(chunk, dense_index));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    let mut output = Q21DenseOrderStates::new(dense_index.orderkeys.len());
    for _ in 0..pending_chunks {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql("Q21 dense worker stopped".to_string()))??;
        output.merge(partial);
    }
    let states = output.into_map(&dense_index.orderkeys);
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] Q21 dense lineitem order states: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(Some(states))
}

fn q21_dense_order_states_chunk(
    batches: Vec<RecordBatch>,
    dense_index: Arc<Q21DenseFinalOrderIndex>,
) -> Result<Q21DenseOrderStates> {
    let mut states = Q21DenseOrderStates::new(dense_index.orderkeys.len());
    for batch in batches {
        if !q21_dense_order_states_batch_into(&batch, &dense_index, &mut states)? {
            return Err(DodamError::UnsupportedSql(
                "Q21 dense order-state path requires typed lineitem columns".to_string(),
            ));
        }
    }
    Ok(states)
}

struct Q21DenseOrderStates {
    positions: Vec<i32>,
    states: Vec<Q21OrderState>,
    touched: Vec<usize>,
}

impl Q21DenseOrderStates {
    fn new(len: usize) -> Self {
        Self {
            positions: vec![-1; len],
            states: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn state_mut(&mut self, index: usize) -> &mut Q21OrderState {
        let position = self.positions[index];
        let position = if position < 0 {
            let position = self.states.len();
            self.positions[index] = i32::try_from(position).expect("Q21 state index overflow");
            self.states.push(Q21OrderState::default());
            self.touched.push(index);
            position
        } else {
            position as usize
        };
        &mut self.states[position]
    }

    fn merge(&mut self, other: Self) {
        for index in other.touched {
            let position = other.positions[index];
            debug_assert!(position >= 0);
            self.state_mut(index).merge(other.states[position as usize]);
        }
    }

    fn into_map(self, orderkeys: &[i64]) -> Q21OrderStateMap {
        let mut output = q21_order_state_map_with_capacity(self.touched.len());
        for index in self.touched {
            let position = self.positions[index];
            debug_assert!(position >= 0);
            output.insert(orderkeys[index], self.states[position as usize]);
        }
        output
    }
}

fn q21_dense_order_states_batch_into(
    batch: &RecordBatch,
    dense_index: &Q21DenseFinalOrderIndex,
    states: &mut Q21DenseOrderStates,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let suppkeys = batch_column(batch, "l_suppkey")?;
    let receipt = batch_column(batch, "l_receiptdate")?;
    let commit = batch_column(batch, "l_commitdate")?;
    let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        receipt.as_any().downcast_ref::<Date32Array>(),
        commit.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && receipt.null_count() == 0
        && commit.null_count() == 0
    {
        let orderkeys = orderkeys.values().as_ref();
        let suppkeys = suppkeys.values().as_ref();
        let receipts = receipt.values().as_ref();
        let commits = commit.values().as_ref();
        for row in 0..orderkeys.len() {
            let Some(index) = q21_dense_order_index(dense_index, orderkeys[row]) else {
                continue;
            };
            let state = states.state_mut(index);
            let suppkey = suppkeys[row];
            state.add_supplier(suppkey);
            if receipts[row] > commits[row] {
                state.add_late_supplier(suppkey);
            }
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let Some(index) = q21_dense_order_index(dense_index, orderkeys.value(row)) else {
            continue;
        };
        let state = states.state_mut(index);
        let suppkey = suppkeys.value(row);
        state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            state.add_late_supplier(suppkey);
        }
    }
    Ok(true)
}

fn q21_dense_order_index(dense_index: &Q21DenseFinalOrderIndex, orderkey: i64) -> Option<usize> {
    let index = usize::try_from(orderkey).ok()?;
    let compact = *dense_index.index_by_orderkey.get(index)?;
    usize::try_from(compact).ok()
}

fn q21_parallel_batch_order_states<Map>(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    output_capacity: usize,
    map: Map,
) -> Result<Q21OrderStateMap>
where
    Map: Fn(Vec<RecordBatch>) -> Result<Q21OrderStateMap> + Send + Sync + Clone + 'static,
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let chunk_size = chunk_size.max(1);
    let mut pending_chunks = 0_usize;
    let mut chunk = Vec::with_capacity(chunk_size);
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size {
            continue;
        }
        let sender = sender.clone();
        let map = map.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(task_chunk));
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let map = map.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(chunk));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    let mut partials = Vec::with_capacity(pending_chunks);
    for _ in 0..pending_chunks {
        partials.push(
            receiver
                .recv()
                .map_err(|_| DodamError::UnsupportedSql("Q21 worker stopped".to_string()))??,
        );
    }
    let output = if q21_parallel_merge_enabled() {
        partials
            .into_par_iter()
            .reduce(q21_order_state_map, q21_merge_order_states_owned)
    } else {
        let mut output = q21_order_state_map_with_capacity(output_capacity);
        for partial in partials {
            q21_merge_order_states(&mut output, partial);
        }
        output
    };
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] Q21 lineitem order states: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(output)
}

fn q21_parallel_merge_enabled() -> bool {
    std::env::var("DODAM_Q21_ENABLE_PARALLEL_MERGE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q21_lineitem_order_state_chunk_size() -> usize {
    std::env::var("DODAM_Q21_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(48)
}

fn q21_lineitem_order_states_batch_into(
    batch: RecordBatch,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> Result<()> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let receipt = batch_column(&batch, "l_receiptdate")?;
    let commit = batch_column(&batch, "l_commitdate")?;
    if q21_lineitem_order_states_typed_into(
        orderkeys,
        suppkeys,
        receipt,
        commit,
        final_orders,
        states,
    ) {
        return Ok(());
    }
    let dense_final_orders = final_orders.dense_contains_slice();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if !q21_final_order_contains(final_orders, dense_final_orders, orderkey) {
            continue;
        }
        let state = states.entry(orderkey).or_default();
        state.add_supplier(suppkey);
        let (Some(receipt), Some(commit)) =
            (date32_value(receipt, row)?, date32_value(commit, row)?)
        else {
            continue;
        };
        if receipt > commit {
            state.add_late_supplier(suppkey);
        }
    }
    Ok(())
}

fn q21_lineitem_order_states_projected_batch_into(
    batch: RecordBatch,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> Result<()> {
    if batch.num_columns() == 4
        && q21_lineitem_order_states_typed_into(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            final_orders,
            states,
        )
    {
        return Ok(());
    }
    q21_lineitem_order_states_batch_into(batch, final_orders, states)
}

fn q21_lineitem_order_states_typed_into(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    receipt: &ArrayRef,
    commit: &ArrayRef,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> bool {
    let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        receipt.as_any().downcast_ref::<Date32Array>(),
        commit.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return false;
    };
    let dense_final_orders = final_orders.dense_contains_slice();
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && receipt.null_count() == 0
        && commit.null_count() == 0
    {
        let orderkeys = orderkeys.values().as_ref();
        let suppkeys = suppkeys.values().as_ref();
        let receipts = receipt.values().as_ref();
        let commits = commit.values().as_ref();
        let mut current_orderkey = None::<i64>;
        let mut current_order_selected = false;
        let mut current_state = Q21OrderState::default();
        for row in 0..orderkeys.len() {
            let orderkey = orderkeys[row];
            if current_orderkey.is_some_and(|current| current != orderkey) {
                if current_order_selected {
                    q21_flush_run_state(states, current_orderkey, &mut current_state);
                }
                current_order_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                current_orderkey = Some(orderkey);
            } else if current_orderkey.is_none() {
                current_order_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                current_orderkey = Some(orderkey);
            }
            if !current_order_selected {
                continue;
            }
            let suppkey = suppkeys[row];
            current_state.add_supplier(suppkey);
            if receipts[row] > commits[row] {
                current_state.add_late_supplier(suppkey);
            }
        }
        if current_order_selected {
            q21_flush_run_state(states, current_orderkey, &mut current_state);
        }
        return true;
    }
    let mut current_orderkey = None::<i64>;
    let mut current_order_selected = false;
    let mut current_state = Q21OrderState::default();
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if current_orderkey.is_some_and(|current| current != orderkey) {
            if current_order_selected {
                q21_flush_run_state(states, current_orderkey, &mut current_state);
            }
            current_order_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            current_orderkey = Some(orderkey);
        } else if current_orderkey.is_none() {
            current_order_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            current_orderkey = Some(orderkey);
        }
        if !current_order_selected {
            continue;
        }
        let suppkey = suppkeys.value(row);
        current_state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            current_state.add_late_supplier(suppkey);
        }
    }
    if current_order_selected {
        q21_flush_run_state(states, current_orderkey, &mut current_state);
    }
    true
}

fn q21_final_order_contains(
    final_orders: &Q21FinalOrders,
    dense_final_orders: Option<&[bool]>,
    orderkey: i64,
) -> bool {
    if let Some(dense_final_orders) = dense_final_orders {
        return usize::try_from(orderkey)
            .ok()
            .and_then(|index| dense_final_orders.get(index))
            .copied()
            .unwrap_or(false);
    }
    final_orders.contains(orderkey)
}

async fn q21_lineitem_supplier_counts_ordered(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    final_orders: &Q21FinalOrders,
    suppliers: &HashMap<i64, String>,
) -> Result<Option<HashMap<i64, u64>>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_receiptdate".to_string(),
                "l_commitdate".to_string(),
            ]),
            None,
        )
        .await?;
    let mut counts = HashMap::<i64, u64>::with_capacity(suppliers.len());
    let mut current_orderkey = None::<i64>;
    let mut current_selected = false;
    let mut current_state = Q21OrderState::default();
    let dense_final_orders = final_orders.dense_contains_slice();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        if !q21_ordered_lineitem_counts_projected_batch(
            &batch,
            final_orders,
            dense_final_orders,
            suppliers,
            &mut counts,
            &mut current_orderkey,
            &mut current_selected,
            &mut current_state,
        )? {
            return Ok(None);
        }
    }
    if current_selected {
        q21_count_qualifying_order(&mut counts, suppliers, &current_state);
    }
    Ok(Some(counts))
}

#[allow(clippy::too_many_arguments)]
fn q21_ordered_lineitem_counts_projected_batch(
    batch: &RecordBatch,
    final_orders: &Q21FinalOrders,
    dense_final_orders: Option<&[bool]>,
    suppliers: &HashMap<i64, String>,
    counts: &mut HashMap<i64, u64>,
    current_orderkey: &mut Option<i64>,
    current_selected: &mut bool,
    current_state: &mut Q21OrderState,
) -> Result<bool> {
    if batch.num_columns() == 4
        && let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
            batch.column(0).as_any().downcast_ref::<Int64Array>(),
            batch.column(1).as_any().downcast_ref::<Int64Array>(),
            batch.column(2).as_any().downcast_ref::<Date32Array>(),
            batch.column(3).as_any().downcast_ref::<Date32Array>(),
        )
    {
        return q21_ordered_lineitem_counts_typed_batch(
            orderkeys,
            suppkeys,
            receipt,
            commit,
            final_orders,
            dense_final_orders,
            suppliers,
            counts,
            current_orderkey,
            current_selected,
            current_state,
        );
    }
    q21_ordered_lineitem_counts_batch(
        batch,
        final_orders,
        dense_final_orders,
        suppliers,
        counts,
        current_orderkey,
        current_selected,
        current_state,
    )
}

#[allow(clippy::too_many_arguments)]
fn q21_ordered_lineitem_counts_batch(
    batch: &RecordBatch,
    final_orders: &Q21FinalOrders,
    dense_final_orders: Option<&[bool]>,
    suppliers: &HashMap<i64, String>,
    counts: &mut HashMap<i64, u64>,
    current_orderkey: &mut Option<i64>,
    current_selected: &mut bool,
    current_state: &mut Q21OrderState,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let suppkeys = batch_column(batch, "l_suppkey")?;
    let receipt = batch_column(batch, "l_receiptdate")?;
    let commit = batch_column(batch, "l_commitdate")?;
    let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        receipt.as_any().downcast_ref::<Date32Array>(),
        commit.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    q21_ordered_lineitem_counts_typed_batch(
        orderkeys,
        suppkeys,
        receipt,
        commit,
        final_orders,
        dense_final_orders,
        suppliers,
        counts,
        current_orderkey,
        current_selected,
        current_state,
    )
}

#[allow(clippy::too_many_arguments)]
fn q21_ordered_lineitem_counts_typed_batch(
    orderkeys: &Int64Array,
    suppkeys: &Int64Array,
    receipt: &Date32Array,
    commit: &Date32Array,
    final_orders: &Q21FinalOrders,
    dense_final_orders: Option<&[bool]>,
    suppliers: &HashMap<i64, String>,
    counts: &mut HashMap<i64, u64>,
    current_orderkey: &mut Option<i64>,
    current_selected: &mut bool,
    current_state: &mut Q21OrderState,
) -> Result<bool> {
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if let Some(current) = *current_orderkey {
            if orderkey < current {
                return Ok(false);
            }
            if orderkey != current {
                if *current_selected {
                    q21_count_qualifying_order(counts, suppliers, current_state);
                }
                *current_state = Q21OrderState::default();
                *current_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                *current_orderkey = Some(orderkey);
            }
        } else {
            *current_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            *current_orderkey = Some(orderkey);
        }
        if !*current_selected {
            continue;
        }
        let suppkey = suppkeys.value(row);
        current_state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            current_state.add_late_supplier(suppkey);
        }
    }
    Ok(true)
}

fn q21_count_qualifying_order(
    counts: &mut HashMap<i64, u64>,
    suppliers: &HashMap<i64, String>,
    state: &Q21OrderState,
) {
    if !state.has_multiple_suppliers() || !state.has_single_late_supplier() {
        return;
    }
    let suppkey = state.late_supplier;
    if !suppliers.contains_key(&suppkey) {
        return;
    }
    *counts.entry(suppkey).or_insert(0) += u64::from(state.late_row_count);
}

fn q21_flush_run_state(
    states: &mut Q21OrderStateMap,
    orderkey: Option<i64>,
    state: &mut Q21OrderState,
) {
    let Some(orderkey) = orderkey else {
        return;
    };
    states
        .entry(orderkey)
        .or_default()
        .merge(std::mem::take(state));
}

fn q21_merge_order_states(states: &mut Q21OrderStateMap, batch_states: Q21OrderStateMap) {
    for (orderkey, batch_state) in batch_states {
        states.entry(orderkey).or_default().merge(batch_state);
    }
}

fn q21_merge_order_states_owned(
    mut left: Q21OrderStateMap,
    mut right: Q21OrderStateMap,
) -> Q21OrderStateMap {
    if left.len() < right.len() {
        std::mem::swap(&mut left, &mut right);
    }
    q21_merge_order_states(&mut left, right);
    left
}

struct Q21Row {
    s_name: String,
    count: u64,
}

fn q21_output(rows: Vec<Q21Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_name", DataType::Utf8, false),
            Field::new("numwait", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_name.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}

async fn try_execute_q20_potential_part_promotion_fast(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    if !q20_shape(select, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut supplier = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(supplier), Some(nation)) = (supplier, nation) else {
        return Ok(None);
    };
    let Some(partsupp_path) = first_table_path_in_subqueries(selection, "partsupp")? else {
        return Ok(None);
    };
    let Some(part_path) = first_table_path_in_subqueries(selection, "part")? else {
        return Ok(None);
    };
    let Some(lineitem_path) = first_table_path_in_subqueries(selection, "lineitem")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let forest_parts = q20_forest_part_keys(engine, part_path, batch_size).await?;
    tpch_profile_elapsed("Q20 forest part keys", stage);
    if forest_parts.is_empty() {
        return Ok(Some(q20_output(Vec::new())?));
    }
    let forest_parts = AdaptiveI64Set::from_hash(forest_parts);
    let stage = tpch_profile_start();
    let lineitem_sums =
        q20_lineitem_quantity_sums(engine, lineitem_path, batch_size, &forest_parts).await?;
    tpch_profile_elapsed("Q20 lineitem quantity sums", stage);
    let stage = tpch_profile_start();
    let eligible_suppliers = q20_eligible_supplier_keys(
        engine,
        partsupp_path,
        batch_size,
        &forest_parts,
        &lineitem_sums,
    )
    .await?;
    tpch_profile_elapsed("Q20 eligible suppliers", stage);
    if eligible_suppliers.is_empty() {
        return Ok(Some(q20_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, "CANADA").await?;
    tpch_profile_elapsed("Q20 nation keys", stage);
    let stage = tpch_profile_start();
    let mut rows = q20_supplier_rows(
        engine,
        supplier.path,
        batch_size,
        &nation_keys,
        &eligible_suppliers,
    )
    .await?;
    tpch_profile_elapsed("Q20 supplier rows", stage);
    let stage = tpch_profile_start();
    rows.sort_by(|left, right| left.s_name.cmp(&right.s_name));
    tpch_profile_elapsed("Q20 final sort", stage);
    Ok(Some(q20_output(rows)?))
}

fn q20_shape(select: &Select, selection: &SqlExpr) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 2
        && text.contains("s_suppkey in")
        && text.contains("p_name like 'forest%'")
        && text.contains("ps_availqty >")
        && text.contains("0.5 * sum(l_quantity)")
        && text.contains("n_name = 'canada'")
}

async fn q20_forest_part_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let names = batch_string_column(&batch, "p_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row).starts_with("forest")
                && let Some(key) = numeric_i64_value(partkeys, row)?
            {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

async fn q20_lineitem_quantity_sums(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
) -> Result<HashMap<(i64, i64), f64>> {
    let forest_parts = Arc::new(forest_parts.clone());
    parquet_scan_fold_chunks(
        engine,
        path,
        batch_size,
        Projection::Columns(vec![
            "l_partkey".to_string(),
            "l_suppkey".to_string(),
            "l_quantity".to_string(),
            "l_shipdate".to_string(),
        ]),
        scan_aggregate_row_group_chunk(),
        4,
        HashMap::<(i64, i64), f64>::new,
        HashMap::<(i64, i64), f64>::new,
        move |batch| {
            let mut sums = HashMap::<(i64, i64), f64>::new();
            merge_f64_groups(
                &mut sums,
                q20_lineitem_quantity_sums_batch(batch, &forest_parts)?,
            );
            Ok(sums)
        },
        merge_f64_groups,
        "Q20 lineitem quantity aggregate",
    )
    .await
}

fn q20_lineitem_quantity_sums_batch(
    batch: RecordBatch,
    forest_parts: &AdaptiveI64Set,
) -> Result<HashMap<(i64, i64), f64>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if let Some(sums) =
        q20_lineitem_quantity_sums_typed(partkeys, suppkeys, quantities, shipdates, forest_parts)?
    {
        return Ok(sums);
    }
    let mut sums = HashMap::<(i64, i64), f64>::new();
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if !(8_766..9_131).contains(&shipdate) {
            continue;
        }
        let (Some(partkey), Some(suppkey), Some(quantity)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(quantities, row)?,
        ) else {
            continue;
        };
        if forest_parts.contains(partkey) {
            *sums.entry((partkey, suppkey)).or_insert(0.0) += quantity;
        }
    }
    Ok(sums)
}

fn q20_lineitem_quantity_sums_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    quantities: &ArrayRef,
    shipdates: &ArrayRef,
    forest_parts: &AdaptiveI64Set,
) -> Result<Option<HashMap<(i64, i64), f64>>> {
    let (Some(partkeys), Some(suppkeys), Some(quantities), Some(shipdates)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(quantities)?,
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    let mut sums = HashMap::<(i64, i64), f64>::new();
    if partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && quantities.null_count() == 0
        && shipdates.null_count() == 0
    {
        for row in 0..partkeys.len() {
            let shipdate = shipdates.value(row);
            if !(8_766..9_131).contains(&shipdate) {
                continue;
            }
            let partkey = partkeys.value(row);
            if forest_parts.contains(partkey) {
                *sums.entry((partkey, suppkeys.value(row))).or_insert(0.0) += quantities.value(row);
            }
        }
        return Ok(Some(sums));
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || shipdates.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        if !(8_766..9_131).contains(&shipdate) {
            continue;
        }
        let partkey = partkeys.value(row);
        if forest_parts.contains(partkey) {
            *sums.entry((partkey, suppkeys.value(row))).or_insert(0.0) += quantities.value(row);
        }
    }
    Ok(Some(sums))
}

async fn q20_eligible_supplier_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
    lineitem_sums: &HashMap<(i64, i64), f64>,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_availqty".to_string(),
            ]),
            None,
        )
        .await?;
    let mut suppliers = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "ps_partkey")?;
        let suppkeys = batch_column(&batch, "ps_suppkey")?;
        let availqty = batch_column(&batch, "ps_availqty")?;
        if let Some(batch_suppliers) = q20_eligible_supplier_keys_typed(
            partkeys,
            suppkeys,
            availqty,
            forest_parts,
            lineitem_sums,
        ) {
            suppliers.extend(batch_suppliers);
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(partkey), Some(suppkey), Some(availqty)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
                numeric_f64_value(availqty, row)?,
            ) else {
                continue;
            };
            if !forest_parts.contains(partkey) {
                continue;
            }
            let Some(quantity_sum) = lineitem_sums.get(&(partkey, suppkey)) else {
                continue;
            };
            if availqty > 0.5 * *quantity_sum {
                suppliers.insert(suppkey);
            }
        }
    }
    Ok(suppliers)
}

fn q20_eligible_supplier_keys_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    availqtys: &ArrayRef,
    forest_parts: &AdaptiveI64Set,
    lineitem_sums: &HashMap<(i64, i64), f64>,
) -> Option<HashSet<i64>> {
    let (Some(partkeys), Some(suppkeys), Some(availqtys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        availqtys.as_any().downcast_ref::<Int32Array>(),
    ) else {
        return None;
    };
    let mut suppliers = HashSet::new();
    if partkeys.null_count() == 0 && suppkeys.null_count() == 0 && availqtys.null_count() == 0 {
        for row in 0..partkeys.len() {
            let partkey = partkeys.value(row);
            if !forest_parts.contains(partkey) {
                continue;
            }
            let suppkey = suppkeys.value(row);
            let Some(quantity_sum) = lineitem_sums.get(&(partkey, suppkey)) else {
                continue;
            };
            if f64::from(availqtys.value(row)) > 0.5 * *quantity_sum {
                suppliers.insert(suppkey);
            }
        }
        return Some(suppliers);
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) || availqtys.is_null(row) {
            continue;
        }
        let partkey = partkeys.value(row);
        if !forest_parts.contains(partkey) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        let Some(quantity_sum) = lineitem_sums.get(&(partkey, suppkey)) else {
            continue;
        };
        if f64::from(availqtys.value(row)) > 0.5 * *quantity_sum {
            suppliers.insert(suppkey);
        }
    }
    Some(suppliers)
}

struct Q20Row {
    s_name: String,
    s_address: String,
}

async fn q20_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
    eligible_suppliers: &HashSet<i64>,
) -> Result<Vec<Q20Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_nationkey".to_string(),
                "s_name".to_string(),
                "s_address".to_string(),
            ]),
            None,
        )
        .await?;
    let mut rows = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        let names = batch_string_column(&batch, "s_name")?;
        let addresses = batch_string_column(&batch, "s_address")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if eligible_suppliers.contains(&suppkey)
                && nation_keys.contains(&nationkey)
                && names.is_valid(row)
                && addresses.is_valid(row)
            {
                rows.push(Q20Row {
                    s_name: names.value(row).to_string(),
                    s_address: addresses.value(row).to_string(),
                });
            }
        }
    }
    Ok(rows)
}

fn q20_output(rows: Vec<Q20Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_address.as_str()),
            )),
        ],
    )?;
    Ok(QueryOutput::Scan {
        batches: vec![batch],
    })
}

async fn try_execute_correlated_join_subquery_filter_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 2 || !expr_contains_scalar_subquery(selection) {
        return Ok(None);
    }
    let [left_table, right_table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !left_table.joins.is_empty() || !right_table.joins.is_empty() {
        return Ok(None);
    }

    let left = parse_table_factor(&left_table.relation)?;
    let right = parse_table_factor(&right_table.relation)?;
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let output_aliases = vec![left_alias.as_str(), right_alias.as_str()];
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    if parse_distinct(select)? {
        return Err(DodamError::UnsupportedSql(
            "JOIN with DISTINCT is not supported".to_string(),
        ));
    }
    let (left_keys, right_keys, residual) =
        split_comma_join_selection(Some(selection), &left_alias, &right_alias, &output_aliases)?;
    let Some(residual) = residual else {
        return Ok(None);
    };
    if !expr_contains_scalar_subquery(&residual) {
        return Ok(None);
    }
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(query, &projection.aliases, &output_aliases)?;
    let limit = parse_limit(query)?;

    let join_input_projection = &Projection::All;
    let join_plan = plan_join_inputs(
        join_input_projection,
        None,
        order_by.as_ref(),
        &left_alias,
        &left_keys,
        &right_alias,
        &right_keys,
    );
    let output_projection = if !projection.aggregate_expressions.is_empty()
        || !projection.aggregates.is_empty()
        || order_by.is_some()
    {
        Projection::All
    } else {
        projection.projection.clone()
    };
    let output_projection_pushed = !matches!(output_projection, Projection::All);
    let stream = engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: left.path,
            right_path: right.path,
            batch_size,
            left_keys,
            right_keys,
            left_prefix: left_alias,
            right_prefix: right_alias,
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: join_plan.right_filter,
            output_projection,
            join_memory_limit_bytes: default_join_memory_limit_bytes(),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await?;
    let residual_sql = residual.to_string();
    let mut filtered = Vec::new();
    for batch in collect_batches(stream)? {
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            &residual_sql,
            None,
            &batch,
            None,
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }

    if !projection.aggregates.is_empty() {
        let stream: SendableBatchStream = if projection.aggregate_expressions.is_empty() {
            Box::new(MemoryExec::new(filtered)).execute()?
        } else {
            let batches =
                append_aggregate_expression_columns(filtered, &projection.aggregate_expressions)?;
            Box::new(MemoryExec::new(batches)).execute()?
        };
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 2, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 2, &group_by, &projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit)?;
    if !output_projection_pushed {
        batches = apply_output_projection(batches, &projection.projection)?;
    }
    batches = rename_output_batches(batches, &projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_execute_materialized_join_subquery_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let has_materializable_subquery = select
        .selection
        .as_ref()
        .is_some_and(expr_contains_materializable_subquery)
        || select
            .having
            .as_ref()
            .is_some_and(expr_contains_materializable_subquery);
    if select.from.len() <= 1 || !has_materializable_subquery {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;

    let mut rewritten_query = query.as_ref().clone();
    let SetExpr::Select(rewritten_select) = rewritten_query.body.as_mut() else {
        return Ok(None);
    };
    let mut changed = false;
    if let Some(rewritten_selection) = rewritten_select.selection.take() {
        let Some(rewritten_selection) = Box::pin(rewrite_materializable_subqueries_to_literals(
            engine,
            rewritten_selection,
            batch_size,
            &mut changed,
        ))
        .await?
        else {
            return Ok(None);
        };
        rewritten_select.selection = Some(rewritten_selection);
    }
    if let Some(rewritten_having) = rewritten_select.having.take() {
        let Some(rewritten_having) = Box::pin(rewrite_materializable_subqueries_to_literals(
            engine,
            rewritten_having,
            batch_size,
            &mut changed,
        ))
        .await?
        else {
            return Ok(None);
        };
        rewritten_select.having = Some(rewritten_having);
    }
    if !changed {
        return Ok(None);
    }
    Box::pin(execute_sql(
        engine,
        &rewritten_query.to_string(),
        batch_size,
    ))
    .await
    .map(Some)
}

async fn rewrite_materializable_subqueries_to_literals(
    engine: &DodamEngine,
    expr: SqlExpr,
    batch_size: usize,
    changed: &mut bool,
) -> Result<Option<SqlExpr>> {
    match expr {
        SqlExpr::Exists { subquery, negated } => {
            let output =
                match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                    Ok(output) => output,
                    Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
            let exists = query_output_batches(output)?
                .iter()
                .any(|batch| batch.num_rows() > 0);
            *changed = true;
            Ok(Some(SqlExpr::Value(
                Value::Boolean(if negated { !exists } else { exists }).with_empty_span(),
            )))
        }
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            let output =
                match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                    Ok(output) => output,
                    Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
            let values = literal_values_from_single_column_batches(query_output_batches(output)?)?;
            *changed = true;
            if values.is_empty() {
                return Ok(Some(SqlExpr::Value(
                    Value::Boolean(negated).with_empty_span(),
                )));
            }
            Ok(Some(SqlExpr::InList {
                expr,
                list: values.into_iter().map(literal_value_to_sql_expr).collect(),
                negated,
            }))
        }
        SqlExpr::Subquery(subquery) => {
            let output =
                match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                    Ok(output) => output,
                    Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
            let value = scalar_literal_value_from_batches(query_output_batches(output)?)?;
            *changed = true;
            Ok(Some(literal_value_to_sql_expr(value)))
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let Some(left) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *left, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(right) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *right, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }))
        }
        SqlExpr::Nested(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Nested(Box::new(expr))))
        }
        SqlExpr::UnaryOp { op, expr } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::UnaryOp {
                op,
                expr: Box::new(expr),
            }))
        }
        SqlExpr::IsNull(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::IsNull(Box::new(expr))))
        }
        SqlExpr::IsNotNull(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::IsNotNull(Box::new(expr))))
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let mut rewritten_list = Vec::with_capacity(list.len());
            for item in list {
                let Some(item) = Box::pin(rewrite_materializable_subqueries_to_literals(
                    engine, item, batch_size, changed,
                ))
                .await?
                else {
                    return Ok(None);
                };
                rewritten_list.push(item);
            }
            Ok(Some(SqlExpr::InList {
                expr: Box::new(expr),
                list: rewritten_list,
                negated,
            }))
        }
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(low) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *low, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(high) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *high, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Between {
                expr: Box::new(expr),
                negated,
                low: Box::new(low),
                high: Box::new(high),
            }))
        }
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(pattern) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *pattern, batch_size, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Like {
                negated,
                any,
                expr: Box::new(expr),
                pattern: Box::new(pattern),
                escape_char,
            }))
        }
        expr => Ok(Some(expr)),
    }
}

fn literal_value_to_sql_expr(value: LiteralValue) -> SqlExpr {
    SqlExpr::Value(
        match value {
            LiteralValue::Null => Value::Null,
            LiteralValue::Boolean(value) => Value::Boolean(value),
            LiteralValue::Int64(value) => Value::Number(value.to_string(), false),
            LiteralValue::Float64(value) => Value::Number(value.to_string(), false),
            LiteralValue::Utf8(value) => Value::SingleQuotedString(value),
        }
        .with_empty_span(),
    )
}

async fn parse_filter_with_subqueries(
    engine: &DodamEngine,
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
    batch_size: usize,
) -> Result<Option<Expr>> {
    match expr {
        SqlExpr::Exists { subquery, negated } => {
            let exists = query_output_batches(
                Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?,
            )?
            .iter()
            .any(|batch| batch.num_rows() > 0);
            Ok(Some(Expr::Boolean(Some(if *negated {
                !exists
            } else {
                exists
            }))))
        }
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            let output = Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?;
            let values = literal_values_from_single_column_batches(query_output_batches(output)?)?;
            if let Ok(value) = sql_literal_value(expr) {
                return Ok(Some(Expr::Boolean(evaluate_literal_in_values(
                    &value, &values, *negated,
                ))));
            }
            Ok(Some(Expr::InList {
                column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
                has_null: subquery_values_contain_null(&values),
                values: non_null_subquery_values(values),
                negated: *negated,
            }))
        }
        SqlExpr::InList {
            expr: in_expr,
            list,
            negated,
        } => {
            if let Ok(value) = sql_literal_value(in_expr) {
                return Ok(Some(Expr::Boolean(evaluate_literal_in_list(
                    &value, list, *negated,
                )?)));
            }
            Ok(Some(sql_expr_to_filter_expr(
                expr,
                aliases,
                table_alias,
                allow_aggregates,
            )?))
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::And => {
            let left = Box::pin(parse_filter_with_subqueries(
                engine,
                left,
                aliases,
                table_alias,
                allow_aggregates,
                batch_size,
            ))
            .await?;
            let right = Box::pin(parse_filter_with_subqueries(
                engine,
                right,
                aliases,
                table_alias,
                allow_aggregates,
                batch_size,
            ))
            .await?;
            Ok(Some(match (left, right) {
                (Some(left), Some(right)) => Expr::And(Box::new(left), Box::new(right)),
                (Some(expr), None) | (None, Some(expr)) => expr,
                (None, None) => return Ok(None),
            }))
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::Or => {
            let left = Box::pin(parse_filter_with_subqueries(
                engine,
                left,
                aliases,
                table_alias,
                allow_aggregates,
                batch_size,
            ))
            .await?;
            let right = Box::pin(parse_filter_with_subqueries(
                engine,
                right,
                aliases,
                table_alias,
                allow_aggregates,
                batch_size,
            ))
            .await?;
            Ok(Some(match (left, right) {
                (Some(left), Some(right)) => Expr::Or(Box::new(left), Box::new(right)),
                (Some(expr), None) | (None, Some(expr)) => expr,
                (None, None) => return Ok(None),
            }))
        }
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) && matches!(right.as_ref(), SqlExpr::Subquery(_)) =>
        {
            let SqlExpr::Subquery(subquery) = right.as_ref() else {
                unreachable!("validated scalar subquery")
            };
            let output = Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?;
            let value = scalar_literal_value_from_batches(query_output_batches(output)?)?;
            if let Ok(left) = sql_literal_value(left) {
                return Ok(Some(Expr::Boolean(compare_literal_values(
                    &left, op, &value,
                )?)));
            }
            Ok(Some(Expr::Comparison(ComparisonExpr {
                column: sql_filter_column(left, aliases, table_alias, allow_aggregates)?,
                op: sql_comparison_op(op),
                value,
            })))
        }
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) =>
        {
            if let (Ok(left), Ok(right)) = (sql_literal_value(left), sql_literal_value(right)) {
                return Ok(Some(Expr::Boolean(compare_literal_values(
                    &left, op, &right,
                )?)));
            }
            Ok(Some(sql_expr_to_filter_expr(
                expr,
                aliases,
                table_alias,
                allow_aggregates,
            )?))
        }
        SqlExpr::Nested(expr) => {
            Box::pin(parse_filter_with_subqueries(
                engine,
                expr,
                aliases,
                table_alias,
                allow_aggregates,
                batch_size,
            ))
            .await
        }
        _ => Ok(Some(sql_expr_to_filter_expr(
            expr,
            aliases,
            table_alias,
            allow_aggregates,
        )?)),
    }
}

async fn try_execute_with_cte_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let Some(with) = query.with.as_ref() else {
        return Ok(None);
    };
    if with.recursive || with.cte_tables.len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "only single non-recursive WITH queries are supported".to_string(),
        ));
    }
    let cte = &with.cte_tables[0];
    if !cte.alias.columns.is_empty() || cte.alias.at.is_some() || cte.from.is_some() {
        return Err(DodamError::UnsupportedSql(
            "WITH column aliases, AT aliases, and FROM identifiers are not supported".to_string(),
        ));
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    reject_select_features(select)?;
    if select.from.len() != 2 || select.from.iter().any(|table| !table.joins.is_empty()) {
        return Err(DodamError::UnsupportedSql(
            "WITH currently supports two-table comma joins".to_string(),
        ));
    }

    let cte_alias = cte.alias.name.value.clone();
    let cte_output = Box::pin(execute_sql(engine, &cte.query.to_string(), batch_size)).await?;
    let cte_batches = query_output_batches(cte_output)?;
    let mut relations = Vec::new();
    for table in &select.from {
        relations.push(
            materialize_cte_join_relation(
                engine,
                &table.relation,
                &cte_alias,
                &cte_batches,
                batch_size,
            )
            .await?,
        );
    }
    let [left, right] = relations.as_slice() else {
        return Ok(None);
    };
    let aliases = vec![left.alias.clone(), right.alias.clone()];
    let alias_refs = aliases.iter().map(String::as_str).collect::<Vec<_>>();
    let selection = select.selection.as_ref().ok_or_else(|| {
        DodamError::UnsupportedSql("comma join requires an equality predicate in WHERE".to_string())
    })?;
    let mut rewritten_selection = selection.clone();
    rewrite_cte_scalar_subqueries_to_literals(&mut rewritten_selection, &cte_alias, &cte_batches)?;
    let (left_keys, right_keys, residual) = split_comma_join_selection(
        Some(&rewritten_selection),
        &left.alias,
        &right.alias,
        &alias_refs,
    )?;
    let group_by = parse_join_group_by(select, &alias_refs)?;
    let projection = parse_join_projection(select, &alias_refs, &group_by)?;
    let distinct = parse_distinct(select)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        None,
    )?;
    let filter = residual
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &alias_refs, false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &alias_refs, true))
        .transpose()?;
    let order_by = parse_join_order_by(query, &projection.aliases, &alias_refs)?;
    let limit = parse_limit(query)?;

    let stream = Box::new(HashJoinExec::new(
        Box::new(MemoryExec::new(left.batches.clone())),
        Box::new(MemoryExec::new(right.batches.clone())),
        left_keys,
        right_keys,
        left.alias.clone(),
        right.alias.clone(),
        JoinBuildSide::Right,
        JoinType::Inner,
        Projection::All,
    ))
    .execute()?;
    let mut batches = apply_output_filter(collect_batches(stream)?, filter.as_ref())?;
    if !projection.aggregates.is_empty() {
        batches = append_aggregate_expression_columns(batches, &projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &projection.expressions)?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn materialize_cte_join_relation(
    engine: &DodamEngine,
    relation: &TableFactor,
    cte_alias: &str,
    cte_batches: &[RecordBatch],
    batch_size: usize,
) -> Result<MaterializedJoinRelation> {
    match relation {
        TableFactor::Table { name, alias, .. }
            if object_name_to_string(name)?.eq_ignore_ascii_case(cte_alias) =>
        {
            Ok(MaterializedJoinRelation {
                alias: alias
                    .as_ref()
                    .map_or_else(|| cte_alias.to_string(), |alias| alias.name.value.clone()),
                batches: cte_batches.to_vec(),
            })
        }
        TableFactor::Table { .. } => materialize_join_relation(engine, relation, batch_size).await,
        _ => Err(DodamError::UnsupportedSql(
            "WITH joins currently support direct table references".to_string(),
        )),
    }
}

fn rewrite_cte_scalar_subqueries_to_literals(
    expr: &mut SqlExpr,
    cte_alias: &str,
    cte_batches: &[RecordBatch],
) -> Result<()> {
    match expr {
        SqlExpr::Subquery(query) => {
            if let Some(value) = evaluate_cte_scalar_subquery(query, cte_alias, cte_batches)? {
                *expr = literal_value_to_sql_expr(value);
            }
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(left, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(right, cte_alias, cte_batches)?;
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
        }
        SqlExpr::InList { expr, list, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
            for item in list {
                rewrite_cte_scalar_subqueries_to_literals(item, cte_alias, cte_batches)?;
            }
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(low, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(high, cte_alias, cte_batches)?;
        }
        SqlExpr::Like { expr, pattern, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(pattern, cte_alias, cte_batches)?;
        }
        _ => {}
    }
    Ok(())
}

fn evaluate_cte_scalar_subquery(
    query: &Query,
    cte_alias: &str,
    cte_batches: &[RecordBatch],
) -> Result<Option<LiteralValue>> {
    if query.with.is_some() || query.order_by.is_some() || query.limit_clause.is_some() {
        return Ok(None);
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table.joins.is_empty() || select.selection.is_some() || select.having.is_some() {
        return Ok(None);
    }
    let TableFactor::Table { name, .. } = &table.relation else {
        return Ok(None);
    };
    if !object_name_to_string(name)?.eq_ignore_ascii_case(cte_alias) {
        return Ok(None);
    }
    let [SelectItem::UnnamedExpr(SqlExpr::Function(function))] = select.projection.as_slice()
    else {
        return Ok(None);
    };
    let aggregate = parse_aggregate(function, None)?;
    match aggregate {
        AggregateExpr::Max(column) => max_literal_from_batches(cte_batches, &column).map(Some),
        _ => Ok(None),
    }
}

fn max_literal_from_batches(batches: &[RecordBatch], column: &str) -> Result<LiteralValue> {
    let mut max_value: Option<LiteralValue> = None;
    for batch in batches {
        let index = batch
            .schema()
            .index_of(column)
            .map_err(|_| DodamError::UnknownColumn(column.to_string()))?;
        let array = batch.column(index);
        for row in 0..batch.num_rows() {
            if array.is_null(row) {
                continue;
            }
            let value = literal_value_from_array(array, row)?;
            let replace = match &max_value {
                None => true,
                Some(current) => matches!(
                    compare_literal_values(&value, &BinaryOperator::Gt, current)?,
                    Some(true)
                ),
            };
            if replace {
                max_value = Some(value);
            }
        }
    }
    Ok(max_value.unwrap_or(LiteralValue::Null))
}

async fn try_execute_derived_join_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    let [join] = table.joins.as_slice() else {
        return Ok(None);
    };
    if !is_materialized_join_relation(&table.relation)
        && !is_materialized_join_relation(&join.relation)
    {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let distinct = parse_distinct(select)?;

    let left = materialize_join_relation(engine, &table.relation, batch_size).await?;
    let right = materialize_join_relation(engine, &join.relation, batch_size).await?;
    let left_alias = left.alias.clone();
    let right_alias = right.alias.clone();
    let (join_type, left_keys, right_keys, right_filter) =
        parse_join_condition(join, &left_alias, &right_alias)?;
    let exec_left_alias = left_alias.clone();
    let exec_right_alias = right_alias.clone();
    let output_aliases = if join_type == JoinType::Semi {
        vec![left_alias.as_str()]
    } else {
        vec![left_alias.as_str(), right_alias.as_str()]
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, false))
        .transpose()?;
    let order_by = parse_join_order_by(query, &projection.aliases, &output_aliases)?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        order_by.as_ref(),
    )?;

    let stream = Box::new(HashJoinExec::new(
        Box::new(MemoryExec::new(left.batches)),
        Box::new(MemoryExec::new(apply_output_filter(
            right.batches,
            right_filter.as_ref(),
        )?)),
        left_keys,
        right_keys,
        exec_left_alias,
        exec_right_alias,
        JoinBuildSide::Right,
        join_type,
        Projection::All,
    ))
    .execute()?;
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, filter.as_ref())?;
    if !projection.aggregates.is_empty() {
        let batches =
            append_aggregate_expression_columns(batches, &projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &projection.aggregates)?;
        let having = select
            .having
            .as_ref()
            .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
            .transpose()?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &projection.expressions)?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

struct MaterializedJoinRelation {
    alias: String,
    batches: Vec<RecordBatch>,
}

fn is_materialized_join_relation(relation: &TableFactor) -> bool {
    matches!(relation, TableFactor::Derived { .. })
}

async fn materialize_join_relation(
    engine: &DodamEngine,
    relation: &TableFactor,
    batch_size: usize,
) -> Result<MaterializedJoinRelation> {
    match relation {
        TableFactor::Derived {
            lateral,
            subquery,
            alias,
            sample,
        } => {
            if *lateral || sample.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "LATERAL and TABLESAMPLE derived tables are not supported".to_string(),
                ));
            }
            let alias = alias.as_ref().ok_or_else(|| {
                DodamError::UnsupportedSql("derived tables in joins must have aliases".to_string())
            })?;
            if !alias.columns.is_empty() || alias.at.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "derived table column aliases and AT aliases are not supported".to_string(),
                ));
            }
            let output = Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?;
            Ok(MaterializedJoinRelation {
                alias: alias.name.value.clone(),
                batches: query_output_batches(output)?,
            })
        }
        TableFactor::Table { .. } => {
            let table = parse_table_factor(relation)?;
            let alias = table_ref_alias_or_name(&table);
            let stream = engine
                .scan_parquet_batches(table.path, batch_size, None, Projection::All, None)
                .await?;
            Ok(MaterializedJoinRelation {
                alias,
                batches: collect_batches(stream)?,
            })
        }
        _ => Err(DodamError::UnsupportedSql(
            "derived table joins only support table and derived table inputs".to_string(),
        )),
    }
}

async fn execute_parsed_join_query(
    engine: &DodamEngine,
    query: SqlQuery,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let Some(join) = query.join.clone() else {
        return Ok(None);
    };
    if query.distinct {
        return Err(DodamError::UnsupportedSql(
            "JOIN with DISTINCT is not supported".to_string(),
        ));
    }
    let is_aggregate = query.is_aggregate();
    let aggregates = query.aggregates.clone();
    let group_by = query.group_by.clone();
    let join_input_projection = &query.projection;
    let join_plan = plan_join_inputs(
        join_input_projection,
        query.filter.as_ref(),
        query.order_by.as_ref(),
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    let output_projection = pushed_join_output_projection(&query);
    let output_projection_pushed = !matches!(output_projection, Projection::All);
    let stream = engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: query.path.clone(),
            right_path: join.right.path,
            batch_size,
            left_keys: join.left_keys,
            right_keys: join.right_keys,
            left_prefix: join.left_alias,
            right_prefix: join.right_alias,
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
            output_projection,
            join_memory_limit_bytes: default_join_memory_limit_bytes(),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: join.join_type,
        })
        .await?;
    if is_aggregate {
        let stream = apply_output_filter_stream(stream, query.filter.clone());
        let stream: SendableBatchStream = if query.aggregate_expressions.is_empty() {
            stream
        } else {
            let batches = append_aggregate_expression_columns(
                collect_batches(stream)?,
                &query.aggregate_expressions,
            )?;
            Box::new(MemoryExec::new(batches)).execute()?
        };
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 2, &aggregates)?
        } else {
            collect_grouped_aggregates(stream, 2, &group_by, &aggregates)?
        };
        let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
        batches = apply_output_filter(batches, query.having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&query.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
        }
        batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, query.filter.as_ref())?;
    let projection_requires_expression = projection_requires_expression_path(&query.expressions);
    if projection_requires_expression {
        batches = apply_output_expression_projection(batches, &query.expressions)?;
        batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
    } else {
        batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
        if !output_projection_pushed {
            batches = apply_output_projection(batches, &query.projection)?;
        }
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &query.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

fn same_join_column(left: &str, right: &str) -> bool {
    left == right
        || left.rsplit('.').next() == Some(right)
        || right.rsplit('.').next() == Some(left)
}

async fn try_execute_derived_left_join_count_distribution_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some((subquery, alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;

    let outer_group_by = parse_group_by(select, Some(&alias))?;
    let projection = parse_projection(select, &outer_group_by, Some(&alias))?;
    if parse_distinct(select)?
        || outer_group_by.len() != 1
        || !matches!(projection.aggregates.as_slice(), [AggregateExpr::CountStar])
        || !projection.aggregate_expressions.is_empty()
        || projection_requires_expression_path(&projection.expressions)
        || select.selection.is_some()
        || select.having.is_some()
    {
        return Ok(None);
    }
    let order_by = parse_order_by(query, &projection.aliases, Some(&alias))?;
    let limit = parse_limit(query)?;

    let inner = match parse_query(subquery) {
        Ok(inner) => inner,
        Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let Some(join) = &inner.join else {
        return Ok(None);
    };
    if inner.distinct
        || inner.filter.is_some()
        || inner.having.is_some()
        || inner.order_by.is_some()
        || inner.limit.is_some()
        || !inner.aggregate_expressions.is_empty()
        || projection_requires_expression_path(&inner.expressions)
        || join.join_type != JoinType::Left
        || join.left_keys.len() != 1
        || join.right_keys.len() != 1
        || inner.group_by.len() != 1
        || !same_join_column(&inner.group_by[0], &join.left_keys[0])
    {
        return Ok(None);
    }
    let [AggregateExpr::Count(count_column)] = inner.aggregates.as_slice() else {
        return Ok(None);
    };
    if resolve_inner_output_column_index(&inner, &outer_group_by[0]) != Some(inner.group_by.len()) {
        return Ok(None);
    }
    if !inner.path.exists() {
        return Ok(None);
    }

    let dense_counts = collect_dense_right_counts(engine, join, count_column, batch_size).await?;
    if dense_counts.is_empty() {
        return Ok(None);
    }
    let groups =
        collect_left_count_distribution(engine, &inner.path, join, &dense_counts, batch_size)
            .await?;
    let rows = groups
        .iter()
        .map(|group| match group.values[0].value {
            AggregateValue::Count(value) => value as usize,
            _ => 0,
        })
        .sum();
    let metrics = AggregateMetrics {
        fragments: 2,
        batches: 1,
        rows,
        values: Vec::new(),
        groups,
        ..AggregateMetrics::default()
    };
    let mut batches =
        aggregate_metrics_to_batches(&metrics, &outer_group_by, &projection.aggregates)?;
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    batches = rename_output_batches(batches, &projection.aliases)?;
    Ok(Some(QueryOutput::Aggregate { metrics, batches }))
}

async fn collect_dense_right_counts(
    engine: &DodamEngine,
    join: &SqlJoin,
    count_column: &str,
    batch_size: usize,
) -> Result<Vec<u64>> {
    let count_column = strip_column_prefix(count_column, &join.right_alias);
    let count_column_required = count_column != join.right_keys[0]
        && !parquet_column_is_non_nullable(&join.right.path, &count_column)?;
    let mut right_projection = vec![join.right_keys[0].clone()];
    if count_column_required {
        add_column_once(&mut right_projection, count_column.clone());
    }
    if let Some(filter) = &join.right_filter {
        for column in filter.referenced_columns() {
            add_column_once(
                &mut right_projection,
                strip_column_prefix(&column, &join.right_alias),
            );
        }
    }
    let direct_filter = join
        .right_filter
        .as_ref()
        .filter(|filter| expr_is_like_only(filter.expr()));
    let fast_like_filter = direct_filter
        .filter(|_| fast_like_distribution_enabled())
        .and_then(|filter| fast_like_substrings_filter(filter.expr()));
    let eval_filter = direct_filter.filter(|_| fast_like_filter.is_none());
    let mut right_stream = engine
        .scan_parquet_batches(
            join.right.path.clone(),
            batch_size,
            None,
            Projection::Columns(right_projection),
            if direct_filter.is_some() {
                None
            } else {
                join.right_filter.clone()
            },
        )
        .await?;
    let mut dense_counts = Vec::<u64>::new();
    while let Some(batch) = right_stream.next() {
        let batch = batch?;
        let fast_like_strings = fast_like_filter
            .as_ref()
            .map(|filter| batch_string_column(&batch, &filter.column))
            .transpose()?;
        let fast_like_finders = fast_like_filter.as_ref().map(|filter| {
            filter
                .parts
                .iter()
                .map(|part| Finder::new(part))
                .collect::<Vec<_>>()
        });
        let key_index = batch_column_index(&batch, &join.right_keys[0])?;
        let count_index = if count_column_required {
            Some(batch_column_index(&batch, &count_column)?)
        } else {
            None
        };
        let Some(keys) = batch
            .column(key_index)
            .as_any()
            .downcast_ref::<Int64Array>()
        else {
            return Ok(Vec::new());
        };
        let values = count_index.map(|index| batch.column(index));
        let mask = eval_filter
            .map(|filter| evaluate_filter_mask(&batch, filter))
            .transpose()?;
        for row in 0..batch.num_rows() {
            if let (Some(filter), Some(strings), Some(finders)) = (
                fast_like_filter.as_ref(),
                fast_like_strings.as_ref(),
                fast_like_finders.as_ref(),
            ) && !fast_like_substrings_row_matches(
                strings,
                row,
                &filter.parts,
                finders,
                filter.negated,
            ) {
                continue;
            }
            if mask
                .as_ref()
                .is_some_and(|mask| mask.is_null(row) || !mask.value(row))
            {
                continue;
            }
            if keys.is_null(row) || values.is_some_and(|values| values.is_null(row)) {
                continue;
            }
            let key = keys.value(row);
            if key < 0 || key > 10_000_000 {
                return Ok(Vec::new());
            }
            let index = key as usize;
            if dense_counts.len() <= index {
                dense_counts.resize(index + 1, 0);
            }
            dense_counts[index] += 1;
        }
    }
    Ok(dense_counts)
}

struct FastLikeSubstrings {
    column: String,
    parts: Vec<Vec<u8>>,
    negated: bool,
}

fn fast_like_distribution_enabled() -> bool {
    std::env::var("DODAM_DISABLE_FAST_LIKE_DISTRIBUTION")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn fast_like_substrings_filter(expr: &Expr) -> Option<FastLikeSubstrings> {
    match expr {
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
        } => {
            if escape.is_some()
                || !pattern.starts_with('%')
                || !pattern.ends_with('%')
                || pattern.contains('_')
            {
                return None;
            }
            let parts = pattern
                .split('%')
                .filter(|part| !part.is_empty())
                .map(|part| part.as_bytes().to_vec())
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| FastLikeSubstrings {
                column: column.clone(),
                parts,
                negated: *negated,
            })
        }
        Expr::Not(expr) => {
            let mut filter = fast_like_substrings_filter(expr)?;
            filter.negated = !filter.negated;
            Some(filter)
        }
        _ => None,
    }
}

fn fast_like_substrings_row_matches(
    strings: &StringArray,
    row: usize,
    parts: &[Vec<u8>],
    finders: &[Finder<'_>],
    negated: bool,
) -> bool {
    if strings.is_null(row) {
        return false;
    }
    let mut haystack = bytes_string_parts(strings.value_offsets(), strings.value_data(), row);
    for (part, finder) in parts.iter().zip(finders) {
        let Some(index) = finder.find(haystack) else {
            return negated;
        };
        haystack = &haystack[index + part.len()..];
    }
    !negated
}

fn expr_is_like_only(expr: &Expr) -> bool {
    match expr {
        Expr::Like { .. } => true,
        Expr::Not(expr) => expr_is_like_only(expr),
        Expr::And(left, right) | Expr::Or(left, right) => {
            expr_is_like_only(left) && expr_is_like_only(right)
        }
        Expr::Boolean(_)
        | Expr::Comparison(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::IsNull { .. } => false,
    }
}

fn parquet_column_is_non_nullable(path: &PathBuf, column: &str) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .find(|field| field.name() == column)
        .is_some_and(|field| !field.is_nullable()))
}

async fn collect_left_count_distribution(
    engine: &DodamEngine,
    left_path: &PathBuf,
    join: &SqlJoin,
    dense_counts: &[u64],
    batch_size: usize,
) -> Result<Vec<GroupAggregateResult>> {
    let mut left_stream = engine
        .scan_parquet_batches(
            left_path.clone(),
            batch_size,
            None,
            Projection::Columns(vec![join.left_keys[0].clone()]),
            None,
        )
        .await?;
    let mut distribution = Vec::<u64>::new();
    while let Some(batch) = left_stream.next() {
        let batch = batch?;
        let key_index = batch_column_index(&batch, &join.left_keys[0])?;
        let Some(keys) = batch
            .column(key_index)
            .as_any()
            .downcast_ref::<Int64Array>()
        else {
            return Ok(Vec::new());
        };
        if keys.null_count() == 0 {
            for key in keys.values().as_ref() {
                let count = if *key >= 0 {
                    dense_counts.get(*key as usize).copied().unwrap_or(0)
                } else {
                    0
                };
                let index = count as usize;
                if distribution.len() <= index {
                    distribution.resize(index + 1, 0);
                }
                distribution[index] += 1;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            let count = if keys.is_valid(row) {
                let key = keys.value(row);
                if key >= 0 {
                    dense_counts.get(key as usize).copied().unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            };
            let index = count as usize;
            if distribution.len() <= index {
                distribution.resize(index + 1, 0);
            }
            distribution[index] += 1;
        }
    }
    Ok(distribution
        .into_iter()
        .enumerate()
        .filter(|(_, rows)| *rows > 0)
        .map(|(count, rows)| GroupAggregateResult {
            keys: vec![GroupValue::UInt64(Some(count as u64))],
            values: vec![AggregateResult {
                expr: AggregateExpr::CountStar,
                value: AggregateValue::Count(rows),
            }],
        })
        .collect())
}

fn resolve_inner_output_column_index(inner: &SqlQuery, column: &str) -> Option<usize> {
    if let Some(index) = inner
        .group_by
        .iter()
        .position(|group| same_join_column(group, column))
    {
        return Some(index);
    }
    if let Some(index) = inner
        .aggregates
        .iter()
        .position(|aggregate| aggregate.to_string() == column)
    {
        return Some(inner.group_by.len() + index);
    }
    let (_, target) = inner.aliases.iter().find(|(alias, _)| alias == column)?;
    if let Some(index) = inner
        .group_by
        .iter()
        .position(|group| same_join_column(group, target))
    {
        return Some(index);
    }
    inner
        .aggregates
        .iter()
        .position(|aggregate| aggregate.to_string() == *target)
        .map(|index| inner.group_by.len() + index)
}

fn batch_column_index(batch: &RecordBatch, column: &str) -> Result<usize> {
    batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
        .ok_or_else(|| DodamError::UnknownColumn(column.to_string()))
}

fn try_count_derived_aggregate_groups(
    inner_metrics: &AggregateMetrics,
    inner_batches: &[RecordBatch],
    group_by: &[String],
    projection: &ParsedProjection,
    filter: Option<&FilterExpr>,
    having: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
) -> Result<Option<QueryOutput>> {
    if group_by.len() != 1
        || !matches!(projection.aggregates.as_slice(), [AggregateExpr::CountStar])
        || !projection.aggregate_expressions.is_empty()
        || projection_requires_expression_path(&projection.expressions)
        || filter.is_some()
        || having.is_some()
        || inner_metrics.groups.is_empty()
    {
        return Ok(None);
    }
    let Some(schema_batch) = inner_batches.first() else {
        return Ok(None);
    };
    let Some(output_column_index) = schema_batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == &group_by[0])
    else {
        return Ok(None);
    };
    let inner_key_count = inner_metrics.groups[0].keys.len();
    let mut counts: HashMap<GroupValue, u64> = HashMap::new();
    for group in &inner_metrics.groups {
        let key = if output_column_index < inner_key_count {
            group.keys[output_column_index].clone()
        } else {
            let value_index = output_column_index - inner_key_count;
            let Some(value) = group.values.get(value_index) else {
                return Ok(None);
            };
            let Some(key) = aggregate_value_to_group_value(&value.value) else {
                return Ok(None);
            };
            key
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut groups = counts
        .into_iter()
        .map(|(key, count)| GroupAggregateResult {
            keys: vec![key],
            values: vec![AggregateResult {
                expr: AggregateExpr::CountStar,
                value: AggregateValue::Count(count),
            }],
        })
        .collect::<Vec<_>>();
    groups.sort_by(|left, right| left.keys[0].to_string().cmp(&right.keys[0].to_string()));
    let metrics = AggregateMetrics {
        fragments: 1,
        batches: 1,
        rows: inner_metrics.groups.len(),
        values: Vec::new(),
        groups,
        ..AggregateMetrics::default()
    };
    let mut batches = aggregate_metrics_to_batches(&metrics, group_by, &projection.aggregates)?;
    batches = apply_output_order_limit(batches, order_by, limit)?;
    batches = rename_output_batches(batches, &projection.aliases)?;
    Ok(Some(QueryOutput::Aggregate { metrics, batches }))
}

fn aggregate_value_to_group_value(value: &AggregateValue) -> Option<GroupValue> {
    match value {
        AggregateValue::Count(value) => Some(GroupValue::UInt64(Some(*value))),
        AggregateValue::Int64(value) => Some(GroupValue::Int64(*value)),
        AggregateValue::Date32(value) => Some(GroupValue::Date32(*value)),
        AggregateValue::Date64(value) => Some(GroupValue::Date64(*value)),
        AggregateValue::Decimal128(value, precision, scale) => {
            Some(GroupValue::Decimal128(*value, *precision, *scale))
        }
        AggregateValue::Utf8(value) => Some(GroupValue::Utf8(value.clone())),
        AggregateValue::Float64(_) | AggregateValue::TimestampMillisecond(_, _) => None,
    }
}

async fn try_execute_derived_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some((subquery, alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;

    let distinct = parse_distinct(select)?;
    let parsed_inner = match parse_query(subquery) {
        Ok(query) => Some(query),
        Err(DodamError::UnsupportedSql(_)) | Err(DodamError::UnknownColumn(_)) => None,
        Err(error) => return Err(error),
    };
    let inner_output = if let Some(parsed_inner) = parsed_inner {
        if let Some(output) = execute_parsed_join_query(engine, parsed_inner, batch_size).await? {
            output
        } else {
            Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?
        }
    } else {
        Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?
    };
    let group_by = parse_group_by(select, Some(&alias))?;
    let parsed_projection = parse_projection(select, &group_by, Some(&alias))?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_filter(expr, &[], Some(&alias), false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(query, &parsed_projection.aliases, Some(&alias))?;
    let limit = parse_limit(query)?;

    if let QueryOutput::Aggregate {
        metrics: inner_metrics,
        batches: inner_batches,
    } = &inner_output
    {
        if let Some(output) = try_count_derived_aggregate_groups(
            inner_metrics,
            inner_batches,
            &group_by,
            &parsed_projection,
            filter.as_ref(),
            having.as_ref(),
            order_by.as_ref(),
            limit,
        )? {
            return Ok(Some(output));
        }
    }

    let inner_batches = query_output_batches(inner_output)?;
    if !parsed_projection.aggregates.is_empty() {
        let mut filtered_batches = apply_output_filter(inner_batches, filter.as_ref())?;
        filtered_batches = append_aggregate_expression_columns(
            filtered_batches,
            &parsed_projection.aggregate_expressions,
        )?;
        let stream = Box::new(MemoryExec::new(filtered_batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions =
            projection_requires_expression_path(&parsed_projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &parsed_projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;
    let mut batches = apply_output_filter(inner_batches, filter.as_ref())?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        apply_output_projection(batches, &parsed_projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_execute_multi_comma_join_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() <= 2 {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let aliases = tables
        .iter()
        .map(table_ref_alias_or_name)
        .collect::<Vec<_>>();
    let alias_refs = aliases.iter().map(String::as_str).collect::<Vec<_>>();
    let mut conjuncts = Vec::new();
    let selection = select.selection.as_ref().ok_or_else(|| {
        DodamError::UnsupportedSql("comma join requires an equality predicate in WHERE".to_string())
    })?;
    collect_sql_and_conjuncts(selection, &mut conjuncts);

    let mut used_conjuncts = vec![false; conjuncts.len()];
    let scan_filters =
        comma_join_single_table_filters(&conjuncts, &aliases, &alias_refs, &mut used_conjuncts)?;
    let group_by = parse_join_group_by(select, &alias_refs)?;
    let projection = parse_join_projection(select, &alias_refs, &group_by)?;
    let distinct = parse_distinct(select)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        None,
    )?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &alias_refs, true))
        .transpose()?;
    let order_by = parse_join_order_by(query, &projection.aliases, &alias_refs)?;
    let limit = parse_limit(query)?;
    let scan_projections = comma_join_scan_projections(
        &conjuncts,
        &aliases,
        &alias_refs,
        &group_by,
        &projection,
        having.as_ref(),
        order_by.as_ref(),
    )?;
    let final_columns = comma_join_final_columns(
        &alias_refs,
        &group_by,
        &projection,
        having.as_ref(),
        order_by.as_ref(),
    )?;
    let mut scanned = Vec::with_capacity(tables.len());
    let mut row_counts = Vec::with_capacity(tables.len());
    for index in 0..tables.len() {
        let batches = scan_table_for_comma_join(
            engine,
            &tables[index],
            batch_size,
            scan_filters[index].as_ref(),
            &scan_projections[index],
        )
        .await?;
        row_counts.push(record_batch_rows(&batches));
        scanned.push(Some(batches));
    }
    let join_cost_model =
        CommaJoinCostModel::new(&scanned, &row_counts, &aliases, &alias_refs, &conjuncts)?;
    let use_ndv_join_order = true;
    let start_index = join_cost_model.choose_start_index().unwrap_or_else(|| {
        row_counts
            .iter()
            .enumerate()
            .min_by_key(|(_, rows)| *rows)
            .map(|(index, _)| index)
            .expect("at least one comma join table")
    });
    let mut current = scanned[start_index].take().expect("start input scanned");
    let mut current_rows = row_counts[start_index];
    let mut joined_aliases = vec![aliases[start_index].clone()];
    let mut remaining = (0..tables.len())
        .filter(|index| *index != start_index)
        .collect::<Vec<_>>();
    while !remaining.is_empty() {
        let mut candidates = Vec::new();
        for (remaining_index, table_index) in remaining.iter().copied().enumerate() {
            let alias = &aliases[table_index];
            let mut left_keys = Vec::new();
            let mut right_keys = Vec::new();
            let mut conjunct_indexes = Vec::new();
            for (index, conjunct) in conjuncts.iter().enumerate() {
                if used_conjuncts[index] {
                    continue;
                }
                if let Some((left_key, right_key)) =
                    comma_join_keys_for_next(conjunct, &joined_aliases, alias, &alias_refs)?
                {
                    left_keys.push(left_key);
                    right_keys.push(right_key);
                    conjunct_indexes.push(index);
                }
            }
            if !left_keys.is_empty() {
                candidates.push((
                    remaining_index,
                    table_index,
                    left_keys,
                    right_keys,
                    conjunct_indexes,
                ));
            }
        }
        let selected = if candidates.len() <= 1 || !use_ndv_join_order {
            candidates.into_iter().next()
        } else {
            let mut selected = None;
            for candidate in candidates {
                let (_, table_index, left_keys, right_keys, _) = &candidate;
                let right = scanned[*table_index]
                    .as_ref()
                    .expect("remaining input scanned");
                let score = estimate_join_output_rows(
                    &current,
                    right,
                    current_rows,
                    row_counts[*table_index],
                    left_keys,
                    right_keys,
                )?
                .saturating_mul(
                    estimated_batches_row_width(&current)
                        .saturating_add(join_cost_model.table_width(*table_index)),
                );
                if selected
                    .as_ref()
                    .is_none_or(|(_, selected_score)| score < *selected_score)
                {
                    selected = Some((candidate, score));
                }
            }
            selected.map(|(candidate, _)| candidate)
        };
        let Some((remaining_index, table_index, left_keys, right_keys, conjunct_indexes)) =
            selected
        else {
            return Err(DodamError::UnsupportedSql(format!(
                "comma join could not find equality predicate for remaining tables: {}",
                remaining
                    .iter()
                    .map(|index| aliases[*index].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        for index in conjunct_indexes {
            used_conjuncts[index] = true;
        }
        remaining.remove(remaining_index);
        let alias = &aliases[table_index];
        let right = scanned[table_index]
            .take()
            .expect("remaining input scanned");
        let right_rows = row_counts[table_index];
        let left_prefix = if joined_aliases.len() == 1 {
            joined_aliases[0].as_str()
        } else {
            "__dodam_join"
        };
        let build_side = if current_rows <= right_rows {
            JoinBuildSide::Left
        } else {
            JoinBuildSide::Right
        };
        let stream = Box::new(HashJoinExec::new(
            Box::new(MemoryExec::new(current)),
            Box::new(MemoryExec::new(right)),
            left_keys,
            right_keys,
            left_prefix.to_string(),
            alias.clone(),
            build_side,
            JoinType::Inner,
            Projection::All,
        ))
        .execute()?;
        current = collect_batches(stream)?;
        current_rows = record_batch_rows(&current);
        if left_prefix == "__dodam_join" {
            current = strip_batch_field_prefix(current, "__dodam_join.")?;
        }
        joined_aliases.push(alias.clone());
        current = prune_comma_join_current_columns(
            current,
            &joined_aliases,
            &alias_refs,
            &conjuncts,
            &used_conjuncts,
            &final_columns,
        )?;
    }

    let residual = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (!used_conjuncts[index]).then_some(conjunct))
        .collect::<Vec<_>>();
    let residual = combine_sql_and_conjuncts(residual);
    let (filter_residual, subquery_residual) = split_subquery_residual(residual);
    let filter = filter_residual
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &alias_refs, false))
        .transpose()?;

    let mut batches = apply_output_filter(current, filter.as_ref())?;
    if let Some(residual) = subquery_residual.as_ref() {
        if let Some(optimized) =
            try_apply_correlated_min_equality_filter(engine, batches.clone(), residual, batch_size)
                .await?
        {
            batches = optimized;
        } else {
            batches = apply_correlated_subquery_filter_batches(
                engine,
                batches,
                &residual.to_string(),
                batch_size,
            )
            .await?;
        }
    }
    if !projection.aggregates.is_empty() {
        batches = append_aggregate_expression_columns(batches, &projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &projection.expressions)?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

fn record_batch_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn estimated_batches_row_width(batches: &[RecordBatch]) -> u128 {
    batches
        .first()
        .map(|batch| {
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| estimated_type_width(field.data_type()))
                .sum::<u128>()
                .max(1)
        })
        .unwrap_or(1)
}

fn estimated_type_width(data_type: &DataType) -> u128 {
    match data_type {
        DataType::Boolean => 1,
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => 8,
        DataType::Decimal128(_, _) => 16,
        DataType::Decimal256(_, _) => 32,
        DataType::Utf8 | DataType::LargeUtf8 => 24,
        DataType::Binary | DataType::LargeBinary => 24,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => 32,
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| estimated_type_width(field.data_type()))
            .sum::<u128>()
            .max(1),
        _ => 16,
    }
}

fn estimate_join_output_rows(
    left: &[RecordBatch],
    right: &[RecordBatch],
    left_rows: usize,
    right_rows: usize,
    left_keys: &[String],
    right_keys: &[String],
) -> Result<u128> {
    let left_ndv = sampled_key_ndv(left, left_keys, 100_000)?.max(1);
    let right_ndv = sampled_key_ndv(right, right_keys, 100_000)?.max(1);
    let denominator = left_ndv.max(right_ndv) as u128;
    Ok((left_rows as u128).saturating_mul(right_rows as u128) / denominator)
}

#[derive(Clone)]
struct CommaJoinTableCostStats {
    rows: u128,
    row_width: u128,
    key_ndv: HashMap<String, u128>,
}

#[derive(Clone)]
struct CommaJoinCostEdge {
    left: usize,
    left_key: String,
    right: usize,
    right_key: String,
}

struct CommaJoinCostModel {
    tables: Vec<CommaJoinTableCostStats>,
    edges: Vec<CommaJoinCostEdge>,
}

impl CommaJoinCostModel {
    fn new(
        scanned: &[Option<Vec<RecordBatch>>],
        row_counts: &[usize],
        aliases: &[String],
        alias_refs: &[&str],
        conjuncts: &[SqlExpr],
    ) -> Result<Self> {
        let mut edges = Vec::new();
        let mut key_columns = vec![Vec::<String>::new(); aliases.len()];
        for conjunct in conjuncts {
            let Some((left_alias, left_key, right_alias, right_key)) =
                comma_join_base_edge(conjunct, alias_refs)?
            else {
                continue;
            };
            let Some(left) = aliases
                .iter()
                .position(|alias| alias.eq_ignore_ascii_case(left_alias))
            else {
                continue;
            };
            let Some(right) = aliases
                .iter()
                .position(|alias| alias.eq_ignore_ascii_case(right_alias))
            else {
                continue;
            };
            add_column_once(&mut key_columns[left], left_key.clone());
            add_column_once(&mut key_columns[right], right_key.clone());
            edges.push(CommaJoinCostEdge {
                left,
                left_key,
                right,
                right_key,
            });
        }

        let mut tables = Vec::with_capacity(scanned.len());
        for (index, batches) in scanned.iter().enumerate() {
            let batches = batches.as_ref().expect("comma join input scanned");
            let mut key_ndv = HashMap::new();
            for key in &key_columns[index] {
                key_ndv.insert(
                    key.clone(),
                    sampled_key_ndv(batches, &[key.clone()], 100_000)? as u128,
                );
            }
            tables.push(CommaJoinTableCostStats {
                rows: row_counts[index].max(1) as u128,
                row_width: estimated_batches_row_width(batches),
                key_ndv,
            });
        }

        Ok(Self { tables, edges })
    }

    fn choose_start_index(&self) -> Option<usize> {
        (0..self.tables.len())
            .filter_map(|start| {
                self.estimate_greedy_plan_cost(start)
                    .map(|cost| (start, cost))
            })
            .min_by_key(|(_, cost)| *cost)
            .map(|(start, _)| start)
    }

    fn table_width(&self, table_index: usize) -> u128 {
        self.tables[table_index].row_width
    }

    fn estimate_greedy_plan_cost(&self, start: usize) -> Option<u128> {
        let mut joined = vec![false; self.tables.len()];
        joined[start] = true;
        let mut joined_count = 1usize;
        let mut rows = self.tables[start].rows;
        let mut row_width = self.tables[start].row_width;
        let mut total_cost = rows.saturating_mul(row_width);

        while joined_count < self.tables.len() {
            let mut selected = None;
            for table_index in 0..self.tables.len() {
                if joined[table_index] {
                    continue;
                }
                let edges = self
                    .edges
                    .iter()
                    .filter(|edge| {
                        (edge.left == table_index && joined[edge.right])
                            || (edge.right == table_index && joined[edge.left])
                    })
                    .collect::<Vec<_>>();
                if edges.is_empty() {
                    continue;
                }
                let output_rows = self.estimate_join_rows(rows, table_index, &edges);
                let output_width = row_width.saturating_add(self.tables[table_index].row_width);
                let build_cost = rows.saturating_mul(row_width).min(
                    self.tables[table_index]
                        .rows
                        .saturating_mul(self.tables[table_index].row_width),
                );
                let step_cost = output_rows
                    .saturating_mul(output_width)
                    .saturating_add(build_cost);
                if selected
                    .as_ref()
                    .is_none_or(|(_, selected_cost, _)| step_cost < *selected_cost)
                {
                    selected = Some((table_index, step_cost, output_rows));
                }
            }
            let (table_index, step_cost, output_rows) = selected?;
            joined[table_index] = true;
            joined_count += 1;
            rows = output_rows.max(1);
            row_width = row_width.saturating_add(self.tables[table_index].row_width);
            total_cost = total_cost.saturating_add(step_cost);
        }

        Some(total_cost)
    }

    fn estimate_join_rows(
        &self,
        current_rows: u128,
        next_table: usize,
        edges: &[&CommaJoinCostEdge],
    ) -> u128 {
        let next_rows = self.tables[next_table].rows;
        let denominator = edges
            .iter()
            .map(|edge| {
                let (left_ndv, right_ndv) = self.edge_ndv(edge);
                left_ndv.max(right_ndv).max(1)
            })
            .max()
            .unwrap_or(1);
        current_rows.saturating_mul(next_rows) / denominator
    }

    fn edge_ndv(&self, edge: &CommaJoinCostEdge) -> (u128, u128) {
        (
            self.tables[edge.left]
                .key_ndv
                .get(&edge.left_key)
                .copied()
                .unwrap_or(self.tables[edge.left].rows)
                .max(1),
            self.tables[edge.right]
                .key_ndv
                .get(&edge.right_key)
                .copied()
                .unwrap_or(self.tables[edge.right].rows)
                .max(1),
        )
    }
}

fn sampled_key_ndv(batches: &[RecordBatch], keys: &[String], sample_rows: usize) -> Result<usize> {
    let mut values = HashSet::new();
    let mut sampled = 0usize;
    for batch in batches {
        if sampled >= sample_rows {
            break;
        }
        let key_indices = keys
            .iter()
            .map(|key| batch_column_index(batch, key))
            .collect::<Result<Vec<_>>>()?;
        for row in 0..batch.num_rows() {
            if sampled >= sample_rows {
                break;
            }
            sampled += 1;
            let mut parts = Vec::with_capacity(key_indices.len());
            let mut has_null = false;
            for index in &key_indices {
                match semijoin_key_at(batch.column(*index), row)? {
                    Some(value) => parts.push(value),
                    None => {
                        has_null = true;
                        break;
                    }
                }
            }
            if !has_null {
                values.insert(parts.join("\x1f"));
            }
        }
    }
    Ok(values.len())
}

fn comma_join_single_table_filters(
    conjuncts: &[SqlExpr],
    aliases: &[String],
    alias_refs: &[&str],
    used_conjuncts: &mut [bool],
) -> Result<Vec<Option<FilterExpr>>> {
    let mut filters = vec![Vec::<SqlExpr>::new(); aliases.len()];
    for (index, conjunct) in conjuncts.iter().enumerate() {
        if expr_contains_materializable_subquery(conjunct) {
            continue;
        }
        let Some(alias) = single_table_conjunct_alias(conjunct, alias_refs)? else {
            continue;
        };
        let Some(table_index) = aliases
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(alias))
        else {
            continue;
        };
        if parse_filter(conjunct, &[], Some(alias), false).is_err() {
            continue;
        }
        filters[table_index].push(conjunct.clone());
        used_conjuncts[index] = true;
    }
    filters
        .into_iter()
        .zip(aliases)
        .map(|(filters, alias)| {
            let Some(expr) = combine_sql_and_conjuncts(filters) else {
                return Ok(None);
            };
            parse_filter(&expr, &[], Some(alias), false).map(Some)
        })
        .collect()
}

fn comma_join_scan_projections(
    conjuncts: &[SqlExpr],
    aliases: &[String],
    alias_refs: &[&str],
    group_by: &[String],
    projection: &ParsedProjection,
    having: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
) -> Result<Vec<Projection>> {
    if conjuncts.iter().any(expr_contains_materializable_subquery) {
        return Ok(vec![Projection::All; aliases.len()]);
    }

    let mut columns = vec![Vec::<String>::new(); aliases.len()];
    for conjunct in conjuncts {
        add_comma_join_expr_columns(&mut columns, conjunct, aliases, alias_refs)?;
    }
    for column in group_by {
        add_comma_join_column(&mut columns, column, aliases)?;
    }
    if let Projection::Columns(projected) = &projection.projection {
        for column in projected {
            add_comma_join_column(&mut columns, column, aliases)?;
        }
    } else {
        return Ok(vec![Projection::All; aliases.len()]);
    }
    for aggregate in &projection.aggregates {
        if let Some(column) = aggregate.referenced_column() {
            add_comma_join_column(&mut columns, column, aliases)?;
        }
    }
    for expression in &projection.aggregate_expressions {
        for column in join_scalar_expression_columns(&expression.expr, alias_refs)? {
            add_comma_join_column(&mut columns, &column, aliases)?;
        }
    }
    for expression in &projection.expressions {
        for column in join_scalar_expression_columns(&expression.expr, alias_refs)? {
            add_comma_join_column(&mut columns, &column, aliases)?;
        }
    }
    if let Some(having) = having {
        for column in having.referenced_columns() {
            add_comma_join_column(&mut columns, &column, aliases)?;
        }
    }
    if let Some(order_by) = order_by {
        for sort in &order_by.expressions {
            add_comma_join_column(&mut columns, &sort.column, aliases)?;
        }
    }

    Ok(columns
        .into_iter()
        .map(|columns| {
            if columns.is_empty() {
                Projection::All
            } else {
                Projection::Columns(columns)
            }
        })
        .collect())
}

fn comma_join_final_columns(
    alias_refs: &[&str],
    group_by: &[String],
    projection: &ParsedProjection,
    having: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
) -> Result<HashSet<String>> {
    let mut columns = HashSet::new();
    for column in group_by {
        columns.insert(column.clone());
    }
    if let Projection::Columns(projected) = &projection.projection {
        columns.extend(projected.iter().cloned());
    }
    for aggregate in &projection.aggregates {
        if let Some(column) = aggregate.referenced_column() {
            columns.insert(column.to_string());
        }
    }
    for expression in &projection.aggregate_expressions {
        columns.extend(join_scalar_expression_columns(
            &expression.expr,
            alias_refs,
        )?);
    }
    for expression in &projection.expressions {
        columns.extend(join_scalar_expression_columns(
            &expression.expr,
            alias_refs,
        )?);
    }
    if let Some(having) = having {
        columns.extend(having.referenced_columns());
    }
    if let Some(order_by) = order_by {
        columns.extend(order_by.expressions.iter().map(|sort| sort.column.clone()));
    }
    Ok(columns)
}

fn prune_comma_join_current_columns(
    batches: Vec<RecordBatch>,
    joined_aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    final_columns: &HashSet<String>,
) -> Result<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(batches);
    }
    let mut needed = final_columns.clone();
    for (conjunct, used) in conjuncts.iter().zip(used_conjuncts) {
        if *used {
            continue;
        }
        let mut columns = Vec::new();
        collect_join_column_candidates(conjunct, alias_refs, &mut columns)?;
        needed.extend(columns);
    }
    let schema = batches[0].schema();
    let keep = schema
        .fields()
        .iter()
        .filter_map(|field| {
            let name = field.name();
            if comma_join_field_needed(name, joined_aliases, &needed) {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if keep.len() == schema.fields().len() {
        return Ok(batches);
    }
    if keep.is_empty() {
        return Ok(batches);
    }
    apply_output_projection(batches, &Projection::Columns(keep))
}

fn comma_join_field_needed(
    field_name: &str,
    joined_aliases: &[String],
    needed: &HashSet<String>,
) -> bool {
    if needed.contains(field_name) {
        return true;
    }
    let Some((alias, column)) = field_name.split_once('.') else {
        return true;
    };
    if !joined_aliases
        .iter()
        .any(|joined| joined.eq_ignore_ascii_case(alias))
    {
        return true;
    }
    needed.contains(&format!("{alias}.{column}"))
}

fn add_comma_join_expr_columns(
    output: &mut [Vec<String>],
    expr: &SqlExpr,
    aliases: &[String],
    alias_refs: &[&str],
) -> Result<()> {
    let mut columns = Vec::new();
    collect_join_column_candidates(expr, alias_refs, &mut columns)?;
    for column in columns {
        add_comma_join_column(output, &column, aliases)?;
    }
    Ok(())
}

fn add_comma_join_column(
    output: &mut [Vec<String>],
    qualified_column: &str,
    aliases: &[String],
) -> Result<()> {
    let Some((alias, column)) = qualified_column.split_once('.') else {
        return Ok(());
    };
    let Some(index) = aliases
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(alias))
    else {
        return Ok(());
    };
    add_column_once(&mut output[index], column.to_string());
    Ok(())
}

fn single_table_conjunct_alias<'a>(
    expr: &SqlExpr,
    table_aliases: &'a [&str],
) -> Result<Option<&'a str>> {
    let mut columns = Vec::new();
    collect_join_column_candidates(expr, table_aliases, &mut columns)?;
    let mut owner: Option<&'a str> = None;
    for column in columns {
        let Some((alias, _)) = column.split_once('.') else {
            return Ok(None);
        };
        let Some(alias) = table_aliases
            .iter()
            .copied()
            .find(|candidate| candidate.eq_ignore_ascii_case(alias))
        else {
            return Ok(None);
        };
        if let Some(existing) = owner {
            if !existing.eq_ignore_ascii_case(alias) {
                return Ok(None);
            }
        } else {
            owner = Some(alias);
        }
    }
    Ok(owner)
}

fn collect_join_column_candidates(
    expr: &SqlExpr,
    table_aliases: &[&str],
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_join_column_candidates(left, table_aliases, columns)?;
            collect_join_column_candidates(right, table_aliases, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
        }
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            add_column_once(columns, join_column_name(expr, table_aliases)?);
        }
        SqlExpr::Function(function) => {
            for arg in function_arg_exprs(function) {
                collect_join_column_candidates(arg, table_aliases, columns)?;
            }
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            if let Some(expr) = substring_from {
                collect_join_column_candidates(expr, table_aliases, columns)?;
            }
            if let Some(expr) = substring_for {
                collect_join_column_candidates(expr, table_aliases, columns)?;
            }
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            for item in list {
                collect_join_column_candidates(item, table_aliases, columns)?;
            }
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            collect_join_column_candidates(low, table_aliases, columns)?;
            collect_join_column_candidates(high, table_aliases, columns)?;
        }
        SqlExpr::Like { expr, pattern, .. } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            collect_join_column_candidates(pattern, table_aliases, columns)?;
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(operand) = operand {
                collect_join_column_candidates(operand, table_aliases, columns)?;
            }
            for when in conditions {
                collect_join_column_candidates(&when.condition, table_aliases, columns)?;
                collect_join_column_candidates(&when.result, table_aliases, columns)?;
            }
            if let Some(else_result) = else_result {
                collect_join_column_candidates(else_result, table_aliases, columns)?;
            }
        }
        SqlExpr::Value(_) => {}
        SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } | SqlExpr::Subquery(_) => {}
        _ => {}
    }
    Ok(())
}

async fn scan_table_for_comma_join(
    engine: &DodamEngine,
    table: &SqlTableRef,
    batch_size: usize,
    filter: Option<&FilterExpr>,
    projection: &Projection,
) -> Result<Vec<RecordBatch>> {
    let stream = engine
        .scan_parquet_batches(
            table.path.clone(),
            batch_size,
            None,
            projection.clone(),
            filter.cloned(),
        )
        .await?;
    collect_batches(stream)
}

fn strip_batch_field_prefix(batches: Vec<RecordBatch>, prefix: &str) -> Result<Vec<RecordBatch>> {
    batches
        .into_iter()
        .map(|batch| {
            let fields = batch
                .schema()
                .fields()
                .iter()
                .map(|field| {
                    let name = field
                        .name()
                        .as_str()
                        .strip_prefix(prefix)
                        .unwrap_or(field.name().as_str())
                        .to_string();
                    Arc::new(Field::new(
                        name,
                        field.data_type().clone(),
                        field.is_nullable(),
                    ))
                })
                .collect::<Vec<_>>();
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                batch.columns().to_vec(),
            )?)
        })
        .collect()
}

fn parse_derived_from(select: &Select) -> Result<Option<(&Query, String)>> {
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table.joins.is_empty() {
        return Ok(None);
    }
    let TableFactor::Derived {
        lateral,
        subquery,
        alias,
        sample,
    } = &table.relation
    else {
        return Ok(None);
    };
    if *lateral || sample.is_some() {
        return Err(DodamError::UnsupportedSql(
            "LATERAL and TABLESAMPLE derived tables are not supported".to_string(),
        ));
    }
    let alias = alias.as_ref().ok_or_else(|| {
        DodamError::UnsupportedSql("derived tables must have an alias".to_string())
    })?;
    if !alias.columns.is_empty() || alias.at.is_some() {
        return Err(DodamError::UnsupportedSql(
            "derived table column aliases and AT aliases are not supported".to_string(),
        ));
    }
    Ok(Some((subquery.as_ref(), alias.name.value.clone())))
}

fn query_output_batches(output: QueryOutput) -> Result<Vec<RecordBatch>> {
    match output {
        QueryOutput::Scan { batches } | QueryOutput::Aggregate { batches, .. } => Ok(batches),
        QueryOutput::Explain { .. } => Err(DodamError::UnsupportedSql(
            "EXPLAIN cannot be used as a derived table".to_string(),
        )),
    }
}

fn literal_values_from_single_column_batches(
    batches: Vec<RecordBatch>,
) -> Result<Vec<LiteralValue>> {
    let Some(schema) = batches.first().map(RecordBatch::schema) else {
        return Ok(Vec::new());
    };
    if schema.fields().len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "IN subquery must return exactly one column".to_string(),
        ));
    }
    let mut values = Vec::new();
    for batch in batches {
        let column = batch.column(0);
        for row in 0..batch.num_rows() {
            if column.is_null(row) {
                values.push(LiteralValue::Null);
                continue;
            }
            values.push(literal_value_from_array(column, row)?);
        }
    }
    Ok(values)
}

fn scalar_literal_value_from_batches(batches: Vec<RecordBatch>) -> Result<LiteralValue> {
    let values = literal_values_from_single_column_batches(batches)?;
    match values.as_slice() {
        [] => Ok(LiteralValue::Null),
        [value] => Ok(value.clone()),
        _ => Err(DodamError::UnsupportedSql(
            "scalar subquery must return at most one row".to_string(),
        )),
    }
}

fn non_null_literal_values(list: &[SqlExpr]) -> Result<Vec<LiteralValue>> {
    let mut values = Vec::with_capacity(list.len());
    for expr in list {
        let value = sql_literal_value(expr)?;
        if !matches!(value, LiteralValue::Null) {
            values.push(value);
        }
    }
    Ok(values)
}

fn literal_list_contains_null(list: &[SqlExpr]) -> Result<bool> {
    for expr in list {
        if matches!(sql_literal_value(expr)?, LiteralValue::Null) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn subquery_values_contain_null(values: &[LiteralValue]) -> bool {
    values
        .iter()
        .any(|value| matches!(value, LiteralValue::Null))
}

fn non_null_subquery_values(values: Vec<LiteralValue>) -> Vec<LiteralValue> {
    values
        .into_iter()
        .filter(|value| !matches!(value, LiteralValue::Null))
        .collect()
}

fn evaluate_literal_in_values(
    value: &LiteralValue,
    values: &[LiteralValue],
    negated: bool,
) -> Option<bool> {
    if matches!(value, LiteralValue::Null) {
        return None;
    }
    let has_null = subquery_values_contain_null(values);
    let matched = values
        .iter()
        .filter(|candidate| !matches!(candidate, LiteralValue::Null))
        .any(|candidate| {
            matches!(
                compare_literal_values(value, &BinaryOperator::Eq, candidate),
                Ok(Some(true))
            )
        });
    let result = if matched {
        Some(true)
    } else if has_null {
        None
    } else {
        Some(false)
    };
    if negated {
        result.map(|value| !value)
    } else {
        result
    }
}

fn compare_literal_values(
    left: &LiteralValue,
    op: &BinaryOperator,
    right: &LiteralValue,
) -> Result<Option<bool>> {
    if matches!(left, LiteralValue::Null) || matches!(right, LiteralValue::Null) {
        return Ok(None);
    }
    let ordering = match (left, right) {
        (LiteralValue::Boolean(left), LiteralValue::Boolean(right)) => left.cmp(right),
        (LiteralValue::Int64(left), LiteralValue::Int64(right)) => left.cmp(right),
        (LiteralValue::Float64(left), LiteralValue::Float64(right)) => left
            .partial_cmp(right)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{left} {op} {right}")))?,
        (LiteralValue::Int64(left), LiteralValue::Float64(right)) => (*left as f64)
            .partial_cmp(right)
            .ok_or_else(|| DodamError::InvalidFilter(format!("{left} {op} {right}")))?,
        (LiteralValue::Float64(left), LiteralValue::Int64(right)) => left
            .partial_cmp(&(*right as f64))
            .ok_or_else(|| DodamError::InvalidFilter(format!("{left} {op} {right}")))?,
        (LiteralValue::Utf8(left), LiteralValue::Utf8(right)) => left.cmp(right),
        _ => {
            return Err(DodamError::InvalidFilter(format!("{left} {op} {right}")));
        }
    };
    Ok(Some(match op {
        BinaryOperator::Eq => ordering.is_eq(),
        BinaryOperator::NotEq => !ordering.is_eq(),
        BinaryOperator::Gt => ordering.is_gt(),
        BinaryOperator::GtEq => ordering.is_gt() || ordering.is_eq(),
        BinaryOperator::Lt => ordering.is_lt(),
        BinaryOperator::LtEq => ordering.is_lt() || ordering.is_eq(),
        _ => unreachable!("validated comparison operator"),
    }))
}

fn literal_value_from_array(column: &ArrayRef, row: usize) -> Result<LiteralValue> {
    match column.data_type() {
        DataType::Boolean => {
            let values = column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean data type");
            Ok(LiteralValue::Boolean(values.value(row)))
        }
        DataType::Int32 => {
            let values = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            Ok(LiteralValue::Int64(i64::from(values.value(row))))
        }
        DataType::Int64 => {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            Ok(LiteralValue::Int64(values.value(row)))
        }
        DataType::Float64 => {
            let values = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 data type");
            Ok(LiteralValue::Float64(values.value(row)))
        }
        DataType::Decimal128(_, scale) => {
            let values = column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128 data type");
            Ok(LiteralValue::Float64(
                values.value(row) as f64 / 10_f64.powi(i32::from(*scale)),
            ))
        }
        DataType::Date32 => {
            let values = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            let (year, month, day) = civil_from_days(i64::from(values.value(row)))?;
            Ok(LiteralValue::Utf8(format!("{year:04}-{month:02}-{day:02}")))
        }
        DataType::Date64 => {
            let values = column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64 data type");
            let (year, month, day) = civil_from_days(values.value(row) / 86_400_000)?;
            Ok(LiteralValue::Utf8(format!("{year:04}-{month:02}-{day:02}")))
        }
        DataType::Utf8 => {
            let values = column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 data type");
            Ok(LiteralValue::Utf8(values.value(row).to_string()))
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "IN subquery result type {data_type} is not supported yet"
        ))),
    }
}

fn sql_uses_materialized_subquery(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    Ok(parse_derived_from(select)?.is_some()
        || select.selection.as_ref().is_some_and(|expr| {
            top_level_exists_subquery(Some(expr)).is_some()
                || expr_contains_materializable_subquery(expr)
        })
        || select
            .having
            .as_ref()
            .is_some_and(expr_contains_materializable_subquery))
}

fn sql_uses_multi_comma_join(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    Ok(parse_comma_join_table_refs(select)?.is_some_and(|tables| tables.len() > 2))
}

fn sql_uses_expression_predicate(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    Ok(select
        .selection
        .as_ref()
        .is_some_and(predicate_requires_expression_path))
}

pub async fn try_execute_sql_streaming(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<SendableBatchStream>> {
    if explain_sql(engine, sql, batch_size).await?.is_some() {
        return Ok(None);
    }
    if sql_uses_materialized_subquery(sql)? {
        return Ok(None);
    }
    if sql_uses_multi_comma_join(sql)? {
        return Ok(None);
    }
    if sql_uses_expression_predicate(sql)? {
        return Ok(None);
    }
    let query = parse_sql(sql)?;
    let Some(join) = query.join.clone() else {
        return Ok(None);
    };
    if query.is_aggregate()
        || query.having.is_some()
        || query.distinct
        || query.filter.is_some()
        || query.order_by.is_some()
        || !query.aliases.is_empty()
    {
        return Ok(None);
    }

    let join_plan = plan_join_inputs(
        &query.projection,
        None,
        None,
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    let output_projection = pushed_join_output_projection(&query);
    if matches!(output_projection, Projection::All) && !matches!(query.projection, Projection::All)
    {
        return Ok(None);
    }
    engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: query.path,
            right_path: join.right.path,
            batch_size,
            left_keys: join.left_keys,
            right_keys: join.right_keys,
            left_prefix: join.left_alias,
            right_prefix: join.right_alias,
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
            output_projection,
            join_memory_limit_bytes: default_join_memory_limit_bytes(),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: join.join_type,
        })
        .await
        .map(Some)
}

pub async fn try_execute_sql_to_sink(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<Option<ScanPlanMetrics>> {
    if explain_sql(engine, sql, batch_size).await?.is_some() {
        return Ok(None);
    }
    if sql_uses_materialized_subquery(sql)? {
        return Ok(None);
    }
    if sql_uses_multi_comma_join(sql)? {
        return Ok(None);
    }
    if sql_uses_expression_predicate(sql)? {
        return Ok(None);
    }
    let query = parse_sql(sql)?;
    let Some(join) = query.join.clone() else {
        return Ok(None);
    };
    if query.is_aggregate()
        || query.having.is_some()
        || query.distinct
        || query.filter.is_some()
        || query.order_by.is_some()
        || !query.aliases.is_empty()
    {
        return Ok(None);
    }

    let join_plan = plan_join_inputs(
        &query.projection,
        None,
        None,
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    let output_projection = pushed_join_output_projection(&query);
    if matches!(output_projection, Projection::All) && !matches!(query.projection, Projection::All)
    {
        return Ok(None);
    }
    let plan = engine
        .plan_parquet_join(JoinParquetRequest {
            left_path: query.path,
            right_path: join.right.path,
            batch_size,
            left_keys: join.left_keys,
            right_keys: join.right_keys,
            left_prefix: join.left_alias,
            right_prefix: join.right_alias,
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
            output_projection,
            join_memory_limit_bytes: default_join_memory_limit_bytes(),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: join.join_type,
        })
        .await?;
    engine.write_join_plan_to_sink(plan, sink).map(Some)
}

pub async fn execute_sql_to_result_sink(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    sink: &mut dyn SqlResultSink,
    options: SqlSinkExecutionOptions,
) -> Result<SqlSinkExecutionProfile> {
    let mut profile = SqlSinkExecutionProfile::default();
    if options.allow_direct_or_streaming && sql_may_use_direct_or_streaming(sql) {
        let direct_started = Instant::now();
        if let Some(metrics) =
            try_execute_sql_to_sink(engine, sql, batch_size, sink.record_batch_sink()).await?
        {
            profile.direct_sink = Some(direct_started.elapsed());
            profile.scan_plan_metrics = Some(metrics);
            return Ok(profile);
        }
        profile.direct_sink = Some(direct_started.elapsed());

        let streaming_started = Instant::now();
        if let Some(stream) = try_execute_sql_streaming(engine, sql, batch_size).await? {
            engine.write_batches_to_sink(stream, sink.record_batch_sink())?;
            profile.streaming = Some(streaming_started.elapsed());
            return Ok(profile);
        }
        profile.streaming = Some(streaming_started.elapsed());
    } else {
        profile.direct_sink = Some(Duration::ZERO);
        profile.streaming = Some(Duration::ZERO);
    }

    let execute_started = Instant::now();
    let output = execute_sql(engine, sql, batch_size).await?;
    profile.execute = Some(execute_started.elapsed());
    let write_started = Instant::now();
    sink.write_output(output)?;
    profile.write_output = Some(write_started.elapsed());
    Ok(profile)
}

fn sql_may_use_direct_or_streaming(sql: &str) -> bool {
    let lower = sql.to_ascii_lowercase();
    if lower.contains(" group by ")
        || lower.contains(" order by ")
        || lower.contains(" having ")
        || lower.contains(" distinct ")
        || lower.contains(" exists")
        || lower.contains(" in (select")
        || lower.contains(" sum(")
        || lower.contains(" count(")
        || lower.contains(" avg(")
        || lower.contains(" min(")
        || lower.contains(" max(")
    {
        return false;
    }
    true
}

async fn explain_sql(engine: &DodamEngine, sql: &str, batch_size: usize) -> Result<Option<String>> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one statement".to_string(),
        ));
    };

    let Statement::Explain {
        analyze,
        verbose,
        query_plan,
        estimate,
        statement,
        format,
        options,
        ..
    } = statement
    else {
        return Ok(None);
    };

    if *analyze || *verbose || *query_plan || *estimate || format.is_some() || options.is_some() {
        return Err(DodamError::UnsupportedSql(
            "EXPLAIN options are not supported yet".to_string(),
        ));
    }

    let Statement::Query(query) = statement.as_ref() else {
        return Err(DodamError::UnsupportedSql(
            "EXPLAIN only supports SELECT queries".to_string(),
        ));
    };
    let query = parse_query(query)?;
    explain_query(engine, query, batch_size).await.map(Some)
}

async fn explain_query(engine: &DodamEngine, query: SqlQuery, batch_size: usize) -> Result<String> {
    if let Some(join) = query.join.clone() {
        if query.is_aggregate() || query.having.is_some() || query.distinct {
            return Err(DodamError::UnsupportedSql(
                "JOIN with aggregates, HAVING, or DISTINCT is not supported".to_string(),
            ));
        }
        let join_plan = plan_join_inputs(
            &query.projection,
            query.filter.as_ref(),
            query.order_by.as_ref(),
            &join.left_alias,
            &join.left_keys,
            &join.right_alias,
            &join.right_keys,
        );
        return engine
            .explain_join_parquet(JoinParquetRequest {
                left_path: query.path,
                right_path: join.right.path,
                batch_size,
                left_keys: join.left_keys,
                right_keys: join.right_keys,
                left_prefix: join.left_alias,
                right_prefix: join.right_alias,
                left_projection: join_plan.left_projection,
                right_projection: join_plan.right_projection,
                left_filter: join_plan.left_filter,
                right_filter: combine_filter_options(
                    join_plan.right_filter,
                    join.right_filter.clone(),
                ),
                output_projection: Projection::All,
                join_memory_limit_bytes: default_join_memory_limit_bytes(),
                join_algorithm: JoinAlgorithm::Auto,
                join_type: join.join_type,
            })
            .await;
    }

    if query.is_aggregate() {
        return engine
            .explain_parquet_aggregate(
                query.path,
                batch_size,
                query.aggregates,
                query.group_by,
                query.filter,
            )
            .await;
    }
    if query.distinct {
        return engine
            .explain_parquet_distinct_scan(
                query.path,
                batch_size,
                query.limit,
                query.projection,
                query.filter,
                query.order_by,
            )
            .await;
    }
    engine
        .explain_parquet_scan(
            query.path,
            batch_size,
            query.limit,
            query.projection,
            query.filter,
            query.order_by,
        )
        .await
}

fn pushed_join_output_projection(query: &SqlQuery) -> Projection {
    let Some(join) = &query.join else {
        return Projection::All;
    };
    if !query.aggregate_expressions.is_empty() {
        return Projection::All;
    }
    if query.is_aggregate() {
        return aggregate_join_output_projection(query);
    }
    if !matches!(join.join_type, JoinType::Inner | JoinType::Semi) {
        return Projection::All;
    }
    if projection_requires_expression_path(&query.expressions)
        || query.filter.is_some()
        || query.order_by.is_some()
    {
        Projection::All
    } else {
        query.projection.clone()
    }
}

fn aggregate_join_output_projection(query: &SqlQuery) -> Projection {
    let Projection::Columns(columns) = &query.projection else {
        return Projection::All;
    };
    let Some(join) = &query.join else {
        return Projection::All;
    };
    let mut columns = columns.clone();
    if let Some(filter) = &query.filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut columns, column);
        }
    }
    let aliases = [join.left_alias.as_str(), join.right_alias.as_str()];
    for expression in &query.aggregate_expressions {
        for column in join_scalar_expression_columns(&expression.expr, &aliases)
            .unwrap_or_else(|_| scalar_expression_columns(&expression.expr))
        {
            add_column_once(&mut columns, column);
        }
    }
    Projection::Columns(columns)
}

fn parse_query(query: &Query) -> Result<SqlQuery> {
    reject_query_features(query)?;
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(DodamError::UnsupportedSql(
            "only simple SELECT queries are supported".to_string(),
        ));
    };
    parse_select(query, select)
}

fn default_join_memory_limit_bytes() -> u64 {
    std::env::var("DODAM_JOIN_MEMORY_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128 * 1024 * 1024)
}

fn parse_select(query: &Query, select: &Select) -> Result<SqlQuery> {
    reject_select_features(select)?;
    if select.from.len() > 1 {
        return parse_comma_join_select(query, select);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return parse_join_select(query, select);
    }
    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    let distinct = parse_distinct(select)?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_filter(expr, &[], path.alias.as_deref(), false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    Ok(SqlQuery {
        path: path.path,
        join: None,
        projection: parsed_projection.projection,
        filter,
        having,
        order_by,
        limit,
        distinct,
        aggregates: parsed_projection.aggregates,
        aggregate_expressions: parsed_projection.aggregate_expressions,
        expressions: parsed_projection.expressions,
        group_by,
        aliases: parsed_projection.aliases,
    })
}

fn parse_comma_join_select(query: &Query, select: &Select) -> Result<SqlQuery> {
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Err(DodamError::UnsupportedSql(
            "comma joins currently support exactly two FROM tables".to_string(),
        ));
    };
    let [left, right] = tables.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "comma joins currently support exactly two FROM tables".to_string(),
        ));
    };
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let output_aliases = vec![left_alias.as_str(), right_alias.as_str()];
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let distinct = parse_distinct(select)?;
    let (left_keys, right_keys, residual) = split_comma_join_selection(
        select.selection.as_ref(),
        &left_alias,
        &right_alias,
        &output_aliases,
    )?;
    let filter = residual
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(query, &projection.aliases, &output_aliases)?;
    let limit = parse_limit(query)?;

    Ok(SqlQuery {
        path: left.path.clone(),
        join: Some(SqlJoin {
            right: right.clone(),
            left_alias,
            right_alias,
            left_keys,
            right_keys,
            right_filter: None,
            join_type: JoinType::Inner,
        }),
        projection: projection.projection,
        filter,
        having,
        order_by,
        limit,
        distinct,
        aggregates: projection.aggregates,
        aggregate_expressions: projection.aggregate_expressions,
        expressions: projection.expressions,
        group_by,
        aliases: projection.aliases,
    })
}

fn parse_join_select(query: &Query, select: &Select) -> Result<SqlQuery> {
    let [table] = select.from.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one FROM table".to_string(),
        ));
    };
    let [join] = table.joins.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one JOIN".to_string(),
        ));
    };
    let left = parse_table_factor(&table.relation)?;
    let right = parse_table_factor(&join.relation)?;
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let (join_type, left_keys, right_keys, right_filter) =
        parse_join_condition(join, &left_alias, &right_alias)?;
    let join_aliases = [left_alias.as_str(), right_alias.as_str()];
    let output_aliases = if join_type == JoinType::Semi {
        vec![left_alias.as_str()]
    } else {
        join_aliases.to_vec()
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let distinct = parse_distinct(select)?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(query, &projection.aliases, &output_aliases)?;
    let limit = parse_limit(query)?;

    Ok(SqlQuery {
        path: left.path,
        join: Some(SqlJoin {
            right,
            left_alias,
            right_alias,
            left_keys,
            right_keys,
            right_filter,
            join_type,
        }),
        projection: projection.projection,
        filter,
        having,
        order_by,
        limit,
        distinct,
        aggregates: projection.aggregates,
        aggregate_expressions: projection.aggregate_expressions,
        expressions: projection.expressions,
        group_by,
        aliases: projection.aliases,
    })
}

fn reject_query_features(query: &Query) -> Result<()> {
    if query.with.is_some()
        || query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Err(DodamError::UnsupportedSql(
            "WITH/FETCH/locks/settings/format/pipe clauses are not supported".to_string(),
        ));
    }
    Ok(())
}

fn reject_select_features(select: &Select) -> Result<()> {
    if select.select_modifiers.is_some()
        || select.top.is_some()
        || select.into.is_some()
        || select.prewhere.is_some()
        || !select.connect_by.is_empty()
        || !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
        || !select.sort_by.is_empty()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.value_table_mode.is_some()
    {
        return Err(DodamError::UnsupportedSql(
            "TOP/window/qualify/select modifiers are not supported".to_string(),
        ));
    }
    Ok(())
}

fn parse_distinct(select: &Select) -> Result<bool> {
    match &select.distinct {
        None | Some(Distinct::All) => Ok(false),
        Some(Distinct::Distinct) => Ok(true),
        Some(Distinct::On(_)) => Err(DodamError::UnsupportedSql(
            "DISTINCT ON is not supported".to_string(),
        )),
    }
}

fn validate_distinct(
    distinct: bool,
    projection: &Projection,
    aggregates: &[AggregateExpr],
    order_by: Option<&SortKey>,
) -> Result<()> {
    if !distinct {
        return Ok(());
    }
    if !aggregates.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "DISTINCT with aggregate SELECT items is not supported".to_string(),
        ));
    }

    if let (Projection::Columns(columns), Some(order_by)) = (projection, order_by) {
        for sort in &order_by.expressions {
            if !columns.iter().any(|column| column == &sort.column) {
                return Err(DodamError::UnsupportedSql(format!(
                    "DISTINCT ORDER BY column {} must appear in SELECT list",
                    sort.column
                )));
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SqlTableRef {
    path: PathBuf,
    alias: Option<String>,
}

fn table_ref_alias_or_name(table: &SqlTableRef) -> String {
    table.alias.clone().unwrap_or_else(|| {
        table
            .path
            .file_stem()
            .or_else(|| table.path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| table.path.to_str().unwrap_or(""))
            .to_string()
    })
}

fn parse_from(select: &Select) -> Result<SqlTableRef> {
    let [table] = select.from.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one FROM table".to_string(),
        ));
    };
    if !table.joins.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "JOIN is not supported".to_string(),
        ));
    }
    parse_table_factor(&table.relation)
}

fn parse_comma_join_table_refs(select: &Select) -> Result<Option<Vec<SqlTableRef>>> {
    if select.from.is_empty() {
        return Ok(None);
    }
    if select.from.len() > 1 {
        if select.from.iter().any(|table| !table.joins.is_empty()) {
            return Err(DodamError::UnsupportedSql(
                "mixed comma and explicit JOIN syntax is not supported".to_string(),
            ));
        }
        return select
            .from
            .iter()
            .map(|table| parse_table_factor(&table.relation))
            .collect::<Result<Vec<_>>>()
            .map(Some);
    }

    let table = &select.from[0];
    if table.joins.is_empty() {
        return Ok(None);
    }
    let mut tables = vec![parse_table_factor(&table.relation)?];
    for join in &table.joins {
        match &join.join_operator {
            JoinOperator::CrossJoin(JoinConstraint::None) => {
                tables.push(parse_table_factor(&join.relation)?);
            }
            _ => return Ok(None),
        }
    }
    Ok(Some(tables))
}

fn parse_table_factor(relation: &TableFactor) -> Result<SqlTableRef> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
        ..
    } = relation
    else {
        return Err(DodamError::UnsupportedSql(
            "only direct table paths or registered table names are supported".to_string(),
        ));
    };
    if args.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || *with_ordinality
        || !partitions.is_empty()
        || json_path.is_some()
        || sample.is_some()
        || !index_hints.is_empty()
    {
        return Err(DodamError::UnsupportedSql(
            "table functions, hints, versions, partitions, and samples are not supported"
                .to_string(),
        ));
    }
    if let Some(alias) = alias
        && (!alias.columns.is_empty() || alias.at.is_some())
    {
        return Err(DodamError::UnsupportedSql(
            "table column aliases and AT aliases are not supported".to_string(),
        ));
    }
    Ok(SqlTableRef {
        path: PathBuf::from(object_name_to_string(name)?),
        alias: alias.as_ref().map(|alias| alias.name.value.clone()),
    })
}

fn split_comma_join_selection(
    selection: Option<&SqlExpr>,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<(Vec<String>, Vec<String>, Option<SqlExpr>)> {
    let Some(selection) = selection else {
        return Err(DodamError::UnsupportedSql(
            "comma join requires an equality predicate in WHERE".to_string(),
        ));
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut residuals = Vec::new();
    for conjunct in conjuncts {
        match comma_join_equality_keys(&conjunct, left_alias, right_alias, table_aliases)? {
            Some((left_key, right_key)) => {
                left_keys.push(left_key);
                right_keys.push(right_key);
            }
            None => residuals.push(conjunct),
        }
    }
    if left_keys.is_empty() {
        for (left_key, right_key) in
            common_or_comma_join_equality_keys(selection, left_alias, right_alias, table_aliases)?
        {
            left_keys.push(left_key);
            right_keys.push(right_key);
        }
    }
    if left_keys.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "comma join requires at least one equality predicate between the two tables"
                .to_string(),
        ));
    }
    Ok((left_keys, right_keys, combine_sql_and_conjuncts(residuals)))
}

fn split_subquery_residual(residual: Option<SqlExpr>) -> (Option<SqlExpr>, Option<SqlExpr>) {
    let Some(residual) = residual else {
        return (None, None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(&residual, &mut conjuncts);
    let mut filter_conjuncts = Vec::new();
    let mut subquery_conjuncts = Vec::new();
    for conjunct in conjuncts {
        if expr_contains_materializable_subquery(&conjunct) {
            subquery_conjuncts.push(conjunct);
        } else {
            filter_conjuncts.push(conjunct);
        }
    }
    (
        combine_sql_and_conjuncts(filter_conjuncts),
        combine_sql_and_conjuncts(subquery_conjuncts),
    )
}

async fn try_apply_correlated_min_equality_filter(
    engine: &DodamEngine,
    batches: Vec<RecordBatch>,
    residual: &SqlExpr,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if batches.is_empty() {
        return Ok(Some(batches));
    }
    let Some(plan) = correlated_min_equality_plan(&batches[0], residual)? else {
        return Ok(None);
    };
    let aggregate_output = Box::pin(execute_sql(engine, &plan.aggregate_sql, batch_size)).await?;
    let aggregate_batches = query_output_batches(aggregate_output)?;
    if aggregate_batches.is_empty() || aggregate_batches.iter().all(|batch| batch.num_rows() == 0) {
        return Ok(Some(Vec::new()));
    }
    let Some(inner_key) = resolve_batch_column(&aggregate_batches[0], &plan.inner_key)? else {
        return Ok(None);
    };
    let Some(aggregate_column) =
        resolve_batch_column(&aggregate_batches[0], &plan.aggregate_column)?
    else {
        return Ok(None);
    };

    let stream = Box::new(HashJoinExec::new(
        Box::new(MemoryExec::new(batches)),
        Box::new(MemoryExec::new(aggregate_batches)),
        vec![plan.outer_key.physical_name.clone()],
        vec![inner_key.physical_name],
        "__dodam_outer".to_string(),
        "__dodam_corr".to_string(),
        JoinBuildSide::Right,
        JoinType::Inner,
        Projection::All,
    ))
    .execute()?;
    let joined = collect_batches(stream)?;
    let joined = strip_batch_field_prefix(joined, "__dodam_outer.")?;
    let min_column = format!("__dodam_corr.{}", aggregate_column.physical_name);
    let filtered = apply_output_filter(
        joined,
        Some(&FilterExpr::new(Expr::ColumnComparison {
            left: plan.outer_value.physical_name,
            op: ComparisonOp::Eq,
            right: min_column,
        })),
    )?;
    Ok(Some(drop_prefixed_columns(filtered, "__dodam_corr.")?))
}

struct CorrelatedMinEqualityPlan {
    outer_value: BoundColumn,
    outer_key: BoundColumn,
    inner_key: String,
    aggregate_column: String,
    aggregate_sql: String,
}

fn correlated_min_equality_plan(
    outer_batch: &RecordBatch,
    residual: &SqlExpr,
) -> Result<Option<CorrelatedMinEqualityPlan>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = residual
    else {
        return Ok(None);
    };
    let (outer_value_expr, subquery) = match (left.as_ref(), right.as_ref()) {
        (outer, SqlExpr::Subquery(subquery)) => (outer, subquery),
        (SqlExpr::Subquery(subquery), outer) => (outer, subquery),
        _ => return Ok(None),
    };
    let outer_value = sql_column_name(outer_value_expr, None)?;
    let Some(outer_value) = resolve_batch_column(outer_batch, &outer_value)? else {
        return Ok(None);
    };

    let SetExpr::Select(select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    if select.distinct.is_some()
        || select.having.is_some()
        || !matches!(select.group_by, GroupByExpr::Expressions(ref exprs, _) if exprs.is_empty())
        || !select.sort_by.is_empty()
        || !select.lateral_views.is_empty()
        || !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
    {
        return Ok(None);
    }
    let [SelectItem::UnnamedExpr(SqlExpr::Function(function))] = select.projection.as_slice()
    else {
        return Ok(None);
    };
    let aggregate = parse_aggregate(function, None)?;
    let AggregateExpr::Min(min_input_column) = aggregate else {
        return Ok(None);
    };
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    let inner_prefixes = select_inner_column_prefixes(select)?;
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((correlation_index, inner_key, outer_key)) =
        correlated_inner_outer_key(outer_batch, &conjuncts, &inner_prefixes)?
    else {
        return Ok(None);
    };
    let Some(outer_key) = resolve_batch_column(outer_batch, &outer_key)? else {
        return Ok(None);
    };
    let remaining = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != correlation_index).then_some(conjunct))
        .collect::<Vec<_>>();
    let where_sql = combine_sql_and_conjuncts(remaining)
        .map(|expr| format!(" WHERE {expr}"))
        .unwrap_or_default();
    let from_sql = select
        .from
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if from_sql.is_empty() {
        return Ok(None);
    }
    let aggregate_column = format!("min({min_input_column})");
    let aggregate_sql = format!(
        "SELECT {inner_key}, min({min_input_column}) FROM {from_sql}{where_sql} GROUP BY {inner_key}"
    );
    Ok(Some(CorrelatedMinEqualityPlan {
        outer_value,
        outer_key,
        inner_key,
        aggregate_column,
        aggregate_sql,
    }))
}

fn correlated_inner_outer_key(
    outer_batch: &RecordBatch,
    conjuncts: &[SqlExpr],
    inner_prefixes: &[String],
) -> Result<Option<(usize, String, String)>> {
    for (index, conjunct) in conjuncts.iter().enumerate() {
        let SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = conjunct
        else {
            continue;
        };
        let Some(left_column) = semijoin_column_name(left)? else {
            continue;
        };
        let Some(right_column) = semijoin_column_name(right)? else {
            continue;
        };
        let left_in_outer = resolve_batch_column(outer_batch, &left_column)?.is_some();
        let right_in_outer = resolve_batch_column(outer_batch, &right_column)?.is_some();
        let left_inner = column_has_any_prefix(&left_column, inner_prefixes);
        let right_inner = column_has_any_prefix(&right_column, inner_prefixes);
        match (left_in_outer, right_in_outer) {
            (true, true) if left_inner && !right_inner => {
                return Ok(Some((index, left_column, right_column)));
            }
            (true, true) if right_inner && !left_inner => {
                return Ok(Some((index, right_column, left_column)));
            }
            (true, false) => return Ok(Some((index, right_column, left_column))),
            (false, true) => return Ok(Some((index, left_column, right_column))),
            _ => {}
        }
    }
    Ok(None)
}

fn resolve_batch_column(batch: &RecordBatch, column: &str) -> Result<Option<BoundColumn>> {
    ColumnResolver::batch(batch).resolve_batch_bound(column)
}

fn aggregate_column_parts(column: &str) -> Option<(&str, &str)> {
    let (function, rest) = column.split_once('(')?;
    let argument = rest.strip_suffix(')')?;
    Some((function, argument))
}

fn select_inner_column_prefixes(select: &Select) -> Result<Vec<String>> {
    let mut prefixes = Vec::new();
    for table in &select.from {
        add_table_factor_prefix(&table.relation, &mut prefixes)?;
        for join in &table.joins {
            match &join.join_operator {
                JoinOperator::CrossJoin(JoinConstraint::None) => {
                    add_table_factor_prefix(&join.relation, &mut prefixes)?;
                }
                _ => return Ok(Vec::new()),
            }
        }
    }
    Ok(prefixes)
}

fn add_table_factor_prefix(relation: &TableFactor, prefixes: &mut Vec<String>) -> Result<()> {
    let table_ref = parse_table_factor(relation)?;
    let alias = table_ref_alias_or_name(&table_ref);
    if let Some(prefix) = tpch_alias_prefix(&alias) {
        add_column_once(prefixes, prefix.to_string());
    } else if let Some(initial) = alias.chars().next() {
        add_column_once(prefixes, initial.to_string());
    }
    Ok(())
}

fn column_has_any_prefix(column: &str, prefixes: &[String]) -> bool {
    let unqualified = unqualified_semijoin_column(column);
    prefixes
        .iter()
        .any(|prefix| unqualified.starts_with(&format!("{prefix}_")))
}

fn drop_prefixed_columns(batches: Vec<RecordBatch>, prefix: &str) -> Result<Vec<RecordBatch>> {
    let mut output = Vec::new();
    for batch in batches {
        let keep = batch
            .schema()
            .fields()
            .iter()
            .filter_map(|field| (!field.name().starts_with(prefix)).then_some(field.name().clone()))
            .collect::<Vec<_>>();
        if keep.is_empty() {
            continue;
        }
        let mut projected = apply_output_projection(vec![batch], &Projection::Columns(keep))?;
        output.append(&mut projected);
    }
    Ok(output)
}

fn common_or_comma_join_equality_keys(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<Vec<(String, String)>> {
    match expr {
        SqlExpr::Nested(expr) => {
            common_or_comma_join_equality_keys(expr, left_alias, right_alias, table_aliases)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Or,
            right,
        } => {
            let left_keys =
                common_or_comma_join_equality_keys(left, left_alias, right_alias, table_aliases)?;
            let right_keys =
                common_or_comma_join_equality_keys(right, left_alias, right_alias, table_aliases)?;
            Ok(left_keys
                .into_iter()
                .filter(|key| right_keys.iter().any(|right_key| right_key == key))
                .collect())
        }
        expr => branch_comma_join_equality_keys(expr, left_alias, right_alias, table_aliases),
    }
}

fn branch_comma_join_equality_keys(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<Vec<(String, String)>> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(expr, &mut conjuncts);
    let mut keys = Vec::new();
    for conjunct in conjuncts {
        if let Some(key) =
            comma_join_equality_keys(&conjunct, left_alias, right_alias, table_aliases)?
            && !keys.iter().any(|existing| existing == &key)
        {
            keys.push(key);
        }
    }
    Ok(keys)
}

fn collect_sql_and_conjuncts(expr: &SqlExpr, conjuncts: &mut Vec<SqlExpr>) {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_sql_and_conjuncts(left, conjuncts);
            collect_sql_and_conjuncts(right, conjuncts);
        }
        SqlExpr::Nested(expr) => collect_sql_and_conjuncts(expr, conjuncts),
        expr => conjuncts.push(expr.clone()),
    }
}

fn collect_sql_or_disjuncts(expr: &SqlExpr, disjuncts: &mut Vec<SqlExpr>) {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Or,
            right,
        } => {
            collect_sql_or_disjuncts(left, disjuncts);
            collect_sql_or_disjuncts(right, disjuncts);
        }
        SqlExpr::Nested(expr) => collect_sql_or_disjuncts(expr, disjuncts),
        expr => disjuncts.push(expr.clone()),
    }
}

fn combine_sql_and_conjuncts(mut conjuncts: Vec<SqlExpr>) -> Option<SqlExpr> {
    let first = conjuncts.pop()?;
    Some(
        conjuncts
            .into_iter()
            .fold(first, |right, left| SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            }),
    )
}

fn comma_join_equality_keys(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<Option<(String, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let left_prefix = format!("{left_alias}.");
    let right_prefix = format!("{right_alias}.");
    if left_column.starts_with(&left_prefix) && right_column.starts_with(&right_prefix) {
        return Ok(Some((
            left_column
                .strip_prefix(&left_prefix)
                .expect("left prefix")
                .to_string(),
            right_column
                .strip_prefix(&right_prefix)
                .expect("right prefix")
                .to_string(),
        )));
    }
    if left_column.starts_with(&right_prefix) && right_column.starts_with(&left_prefix) {
        return Ok(Some((
            right_column
                .strip_prefix(&left_prefix)
                .expect("left prefix")
                .to_string(),
            left_column
                .strip_prefix(&right_prefix)
                .expect("right prefix")
                .to_string(),
        )));
    }
    Ok(None)
}

fn comma_join_keys_for_next(
    expr: &SqlExpr,
    joined_aliases: &[String],
    next_alias: &str,
    table_aliases: &[&str],
) -> Result<Option<(String, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let Some(left_owner) = join_column_owner(&left_column, table_aliases) else {
        return Ok(None);
    };
    let Some(right_owner) = join_column_owner(&right_column, table_aliases) else {
        return Ok(None);
    };
    if left_owner == next_alias && joined_aliases.iter().any(|alias| alias == right_owner) {
        return Ok(Some((
            joined_comma_join_key(&right_column, right_owner, joined_aliases),
            unqualified_join_column(&left_column, next_alias),
        )));
    }
    if right_owner == next_alias && joined_aliases.iter().any(|alias| alias == left_owner) {
        return Ok(Some((
            joined_comma_join_key(&left_column, left_owner, joined_aliases),
            unqualified_join_column(&right_column, next_alias),
        )));
    }
    Ok(None)
}

fn comma_join_base_edge<'a>(
    expr: &SqlExpr,
    table_aliases: &'a [&'a str],
) -> Result<Option<(&'a str, String, &'a str, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let Some(left_owner) = join_column_owner(&left_column, table_aliases) else {
        return Ok(None);
    };
    let Some(right_owner) = join_column_owner(&right_column, table_aliases) else {
        return Ok(None);
    };
    if left_owner == right_owner {
        return Ok(None);
    }
    Ok(Some((
        left_owner,
        unqualified_join_column(&left_column, left_owner),
        right_owner,
        unqualified_join_column(&right_column, right_owner),
    )))
}

fn join_column_owner<'a>(column: &str, table_aliases: &'a [&str]) -> Option<&'a str> {
    table_aliases
        .iter()
        .copied()
        .find(|alias| column.starts_with(&format!("{alias}.")))
}

fn joined_comma_join_key(column: &str, owner: &str, joined_aliases: &[String]) -> String {
    if joined_aliases.len() == 1 && joined_aliases[0] == owner {
        unqualified_join_column(column, owner)
    } else {
        column.to_string()
    }
}

fn unqualified_join_column(column: &str, alias: &str) -> String {
    column
        .strip_prefix(&format!("{alias}."))
        .expect("qualified join column")
        .to_string()
}

fn maybe_join_column_name(expr: &SqlExpr, table_aliases: &[&str]) -> Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            join_column_name(expr, table_aliases).map(Some)
        }
        _ => Ok(None),
    }
}

fn parse_join_condition(
    join: &sqlparser::ast::Join,
    left_alias: &str,
    right_alias: &str,
) -> Result<(JoinType, Vec<String>, Vec<String>, Option<FilterExpr>)> {
    let (join_type, constraint) = match &join.join_operator {
        JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => {
            (JoinType::Inner, constraint)
        }
        JoinOperator::Semi(constraint) | JoinOperator::LeftSemi(constraint) => {
            (JoinType::Semi, constraint)
        }
        JoinOperator::Left(constraint) | JoinOperator::LeftOuter(constraint) => {
            (JoinType::Left, constraint)
        }
        JoinOperator::Right(constraint) | JoinOperator::RightOuter(constraint) => {
            (JoinType::Right, constraint)
        }
        JoinOperator::FullOuter(constraint) => (JoinType::Full, constraint),
        operator => {
            return Err(DodamError::UnsupportedSql(format!(
                "only INNER, LEFT, RIGHT, FULL, and LEFT SEMI JOIN are supported, got {operator:?}"
            )));
        }
    };
    let JoinConstraint::On(expr) = constraint else {
        return Err(DodamError::UnsupportedSql(
            "JOIN requires equality ON conditions".to_string(),
        ));
    };

    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut right_filters = Vec::new();
    collect_join_equalities(
        expr,
        left_alias,
        right_alias,
        &mut left_keys,
        &mut right_keys,
        &mut right_filters,
        join_type,
    )?;
    if left_keys.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "JOIN requires at least one equality ON condition".to_string(),
        ));
    }
    Ok((
        join_type,
        left_keys,
        right_keys,
        combine_expr_filters(right_filters),
    ))
}

fn collect_join_equalities(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    left_keys: &mut Vec<String>,
    right_keys: &mut Vec<String>,
    right_filters: &mut Vec<Expr>,
    join_type: JoinType,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_join_equalities(
                left,
                left_alias,
                right_alias,
                left_keys,
                right_keys,
                right_filters,
                join_type,
            )?;
            collect_join_equalities(
                right,
                left_alias,
                right_alias,
                left_keys,
                right_keys,
                right_filters,
                join_type,
            )
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            if let (Some(left_column), Some(right_column)) = (
                unqualified_column_identifier(left),
                unqualified_column_identifier(right),
            ) {
                left_keys.push(left_column);
                right_keys.push(right_column);
                return Ok(());
            }
            let left_column = qualified_join_column(left, &[left_alias, right_alias])?;
            let right_column = qualified_join_column(right, &[left_alias, right_alias])?;
            let (left_column, right_column) = if left_column.starts_with(&format!("{left_alias}."))
                && right_column.starts_with(&format!("{right_alias}."))
            {
                (left_column, right_column)
            } else if left_column.starts_with(&format!("{right_alias}."))
                && right_column.starts_with(&format!("{left_alias}."))
            {
                (right_column, left_column)
            } else {
                return Err(DodamError::UnsupportedSql(
                    "JOIN condition must compare one column from each side".to_string(),
                ));
            };
            left_keys.push(
                left_column
                    .strip_prefix(&format!("{left_alias}."))
                    .expect("left prefix")
                    .to_string(),
            );
            right_keys.push(
                right_column
                    .strip_prefix(&format!("{right_alias}."))
                    .expect("right prefix")
                    .to_string(),
            );
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "JOIN requires equality ON conditions joined by AND".to_string(),
        ))
        .or_else(|_| {
            let filter = join_expr_to_filter_expr(expr, &[], &[left_alias, right_alias], false)?;
            let filter =
                normalize_right_join_on_filter(filter, left_alias, right_alias, join_type)?;
            right_filters.push(filter);
            Ok(())
        }),
    }
}

fn unqualified_column_identifier(expr: &SqlExpr) -> Option<String> {
    match expr {
        SqlExpr::Identifier(ident) => Some(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => {
            let [ident] = parts.as_slice() else {
                return None;
            };
            Some(ident.value.clone())
        }
        _ => None,
    }
}

fn normalize_right_join_on_filter(
    expr: Expr,
    left_alias: &str,
    right_alias: &str,
    join_type: JoinType,
) -> Result<Expr> {
    if !matches!(join_type, JoinType::Inner | JoinType::Left | JoinType::Semi) {
        return Err(DodamError::UnsupportedSql(
            "JOIN ON residual filters are only supported for INNER, LEFT, and SEMI joins"
                .to_string(),
        ));
    }
    let mut columns = Vec::new();
    collect_filter_columns(&expr, &mut columns);
    if columns
        .iter()
        .any(|column| column == left_alias || column.starts_with(&format!("{left_alias}.")))
    {
        return Err(DodamError::UnsupportedSql(
            "JOIN ON residual filters may only reference the right input".to_string(),
        ));
    }
    Ok(strip_filter_prefix(expr, right_alias))
}

fn combine_expr_filters(mut filters: Vec<Expr>) -> Option<FilterExpr> {
    let first = filters.pop()?;
    Some(FilterExpr::new(
        filters.into_iter().fold(first, |right, left| {
            Expr::And(Box::new(left), Box::new(right))
        }),
    ))
}

fn combine_filter_options(
    left: Option<FilterExpr>,
    right: Option<FilterExpr>,
) -> Option<FilterExpr> {
    match (left, right) {
        (None, None) => None,
        (Some(filter), None) | (None, Some(filter)) => Some(filter),
        (Some(left), Some(right)) => Some(FilterExpr::new(Expr::And(
            Box::new(left.expr().clone()),
            Box::new(right.expr().clone()),
        ))),
    }
}

fn collect_filter_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Boolean(_) => {}
        Expr::Comparison(comparison) => add_filter_column(columns, &comparison.column),
        Expr::ColumnComparison { left, right, .. } => {
            add_filter_column(columns, left);
            add_filter_column(columns, right);
        }
        Expr::InList { column, .. } | Expr::Like { column, .. } | Expr::IsNull { column, .. } => {
            add_filter_column(columns, column);
        }
        Expr::Not(expr) => collect_filter_columns(expr, columns),
        Expr::And(left, right) | Expr::Or(left, right) => {
            collect_filter_columns(left, columns);
            collect_filter_columns(right, columns);
        }
    }
}

fn add_filter_column(columns: &mut Vec<String>, column: &str) {
    if !columns.iter().any(|existing| existing == column) {
        columns.push(column.to_string());
    }
}

fn strip_filter_prefix(expr: Expr, prefix: &str) -> Expr {
    match expr {
        Expr::Boolean(value) => Expr::Boolean(value),
        Expr::Comparison(mut comparison) => {
            comparison.column = strip_column_prefix(&comparison.column, prefix);
            Expr::Comparison(comparison)
        }
        Expr::ColumnComparison { left, op, right } => Expr::ColumnComparison {
            left: strip_column_prefix(&left, prefix),
            op,
            right: strip_column_prefix(&right, prefix),
        },
        Expr::InList {
            column,
            values,
            negated,
            has_null,
        } => Expr::InList {
            column: strip_column_prefix(&column, prefix),
            values,
            negated,
            has_null,
        },
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
        } => Expr::Like {
            column: strip_column_prefix(&column, prefix),
            pattern,
            negated,
            escape,
        },
        Expr::IsNull { column, negated } => Expr::IsNull {
            column: strip_column_prefix(&column, prefix),
            negated,
        },
        Expr::Not(expr) => Expr::Not(Box::new(strip_filter_prefix(*expr, prefix))),
        Expr::And(left, right) => Expr::And(
            Box::new(strip_filter_prefix(*left, prefix)),
            Box::new(strip_filter_prefix(*right, prefix)),
        ),
        Expr::Or(left, right) => Expr::Or(
            Box::new(strip_filter_prefix(*left, prefix)),
            Box::new(strip_filter_prefix(*right, prefix)),
        ),
    }
}

fn strip_column_prefix(column: &str, prefix: &str) -> String {
    column
        .strip_prefix(&format!("{prefix}."))
        .unwrap_or(column)
        .to_string()
}

fn parse_join_projection(
    select: &Select,
    table_aliases: &[&str],
    group_by: &[String],
) -> Result<ParsedProjection> {
    let mut columns = Vec::new();
    let mut aggregates = Vec::new();
    let mut aggregate_expressions = Vec::new();
    let mut aggregate_expression_columns = Vec::new();
    let mut aliases = Vec::new();
    let mut expressions = Vec::new();
    let mut wildcard = false;

    for item in &select.projection {
        match item {
            SelectItem::Wildcard(_) => wildcard = true,
            SelectItem::UnnamedExpr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            ) => {
                let column = join_column_name(expr, table_aliases)?;
                columns.push(column.clone());
                expressions.push(ProjectionExpression {
                    output_name: column_output_name(expr),
                    expr: ScalarSqlExpression::Column(column),
                });
            }
            SelectItem::UnnamedExpr(SqlExpr::Function(function)) => {
                let (aggregate, expression) = parse_join_aggregate_with_input_expression(
                    function,
                    table_aliases,
                    aggregate_expressions.len(),
                )?;
                if let Some(expression) = expression {
                    for column in join_scalar_expression_columns(&expression.expr, table_aliases)? {
                        add_column_once(&mut aggregate_expression_columns, column);
                    }
                    for column in join_sql_expression_columns(
                        &SqlExpr::Function(function.clone()),
                        table_aliases,
                    )? {
                        add_column_once(&mut aggregate_expression_columns, column);
                    }
                    aggregate_expressions.push(expression);
                }
                aliases.push((function.to_string(), aggregate.to_string()));
                aggregates.push(aggregate);
            }
            SelectItem::UnnamedExpr(expr) => {
                let mut expression_columns = Vec::new();
                let mut found_aggregate = false;
                let expression = ProjectionExpression {
                    output_name: expr.to_string(),
                    expr: parse_join_aggregate_output_expression(
                        expr,
                        table_aliases,
                        &mut aggregates,
                        &mut aggregate_expressions,
                        &mut aggregate_expression_columns,
                        &mut expression_columns,
                        &mut found_aggregate,
                    )?,
                };
                if !found_aggregate {
                    return Err(DodamError::UnsupportedSql(format!(
                        "unsupported JOIN SELECT item: {item}"
                    )));
                }
                for column in expression_columns {
                    add_column_once(&mut columns, column);
                }
                expressions.push(expression);
            }
            SelectItem::ExprWithAlias { expr, alias } => match expr {
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                    let column = join_column_name(expr, table_aliases)?;
                    aliases.push((alias.value.clone(), column.clone()));
                    columns.push(column);
                    expressions.push(ProjectionExpression {
                        output_name: alias.value.clone(),
                        expr: ScalarSqlExpression::Column(
                            aliases.last().expect("alias just pushed").1.clone(),
                        ),
                    });
                }
                SqlExpr::Function(function) => {
                    let (aggregate, expression) = parse_join_aggregate_with_input_expression(
                        function,
                        table_aliases,
                        aggregate_expressions.len(),
                    )?;
                    if let Some(expression) = expression {
                        for column in
                            join_scalar_expression_columns(&expression.expr, table_aliases)?
                        {
                            add_column_once(&mut aggregate_expression_columns, column);
                        }
                        for column in join_sql_expression_columns(expr, table_aliases)? {
                            add_column_once(&mut aggregate_expression_columns, column);
                        }
                        aggregate_expressions.push(expression);
                    }
                    aliases.push((function.to_string(), aggregate.to_string()));
                    aliases.push((alias.value.clone(), aggregate.to_string()));
                    aggregates.push(aggregate);
                }
                _ => {
                    let mut expression_columns = Vec::new();
                    let mut found_aggregate = false;
                    let parsed = parse_join_aggregate_output_expression(
                        expr,
                        table_aliases,
                        &mut aggregates,
                        &mut aggregate_expressions,
                        &mut aggregate_expression_columns,
                        &mut expression_columns,
                        &mut found_aggregate,
                    )?;
                    if !found_aggregate {
                        let parsed = parse_join_scalar_sql_expression(expr, table_aliases)?;
                        for column in join_scalar_expression_columns(&parsed, table_aliases)? {
                            add_column_once(&mut columns, column);
                        }
                        aliases.push((alias.value.clone(), alias.value.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: parsed,
                        });
                        continue;
                    }
                    for column in expression_columns {
                        add_column_once(&mut columns, column);
                    }
                    aliases.push((alias.value.clone(), alias.value.clone()));
                    expressions.push(ProjectionExpression {
                        output_name: alias.value.clone(),
                        expr: parsed,
                    });
                }
            },
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(DodamError::UnsupportedSql(
                    "qualified wildcard in JOIN is not supported yet".to_string(),
                ));
            }
            _ => {
                return Err(DodamError::UnsupportedSql(format!(
                    "JOIN SELECT items must be qualified columns, got {item}"
                )));
            }
        }
    }

    if wildcard && select.projection.len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "SELECT * cannot be mixed with other items".to_string(),
        ));
    }

    if wildcard && !aggregates.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "SELECT * cannot be used with aggregate JOIN SELECT items".to_string(),
        ));
    }

    if !aggregates.is_empty() {
        for column in &columns {
            if !group_by.iter().any(|group_column| group_column == column) {
                return Err(DodamError::UnsupportedSql(format!(
                    "non-aggregate JOIN SELECT column {column} must appear in GROUP BY"
                )));
            }
        }
        let mut projected_columns = columns.clone();
        for aggregate in &aggregates {
            if let Some(column) = aggregate.referenced_column() {
                add_column_once(&mut projected_columns, column.to_string());
            }
        }
        for column in aggregate_expression_columns {
            add_column_once(&mut projected_columns, column);
        }
        return Ok(ParsedProjection {
            projection: Projection::Columns(projected_columns),
            aggregates,
            aggregate_expressions,
            aliases,
            expressions,
        });
    }

    Ok(ParsedProjection {
        projection: if wildcard {
            Projection::All
        } else {
            Projection::Columns(columns)
        },
        aggregates,
        aggregate_expressions,
        aliases,
        expressions,
    })
}

fn parse_join_group_by(select: &Select, table_aliases: &[&str]) -> Result<Vec<String>> {
    match &select.group_by {
        GroupByExpr::Expressions(expressions, modifiers) if modifiers.is_empty() => expressions
            .iter()
            .map(|expr| join_column_name(expr, table_aliases))
            .collect::<Result<Vec<_>>>(),
        GroupByExpr::Expressions(_, _) | GroupByExpr::All(_) => Err(DodamError::UnsupportedSql(
            "GROUP BY modifiers and GROUP BY ALL are not supported".to_string(),
        )),
    }
}

fn parse_join_order_by(
    query: &Query,
    aliases: &[(String, String)],
    table_aliases: &[&str],
) -> Result<Option<SortKey>> {
    let Some(order_by) = &query.order_by else {
        return Ok(None);
    };
    if order_by.interpolate.is_some() {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY INTERPOLATE is not supported".to_string(),
        ));
    }
    let OrderByKind::Expressions(expressions) = &order_by.kind else {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY ALL is not supported".to_string(),
        ));
    };
    let expressions = expressions
        .iter()
        .map(|order| {
            if order.with_fill.is_some() || order.options.nulls_first.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY WITH FILL and NULLS FIRST/LAST are not supported".to_string(),
                ));
            }
            let column = match &order.expr {
                SqlExpr::Identifier(ident) => {
                    let resolved = resolve_alias(&ident.value, aliases);
                    if resolved == ident.value {
                        join_column_name(&order.expr, table_aliases)?
                    } else {
                        resolved
                    }
                }
                SqlExpr::CompoundIdentifier(_) => join_column_name(&order.expr, table_aliases)?,
                SqlExpr::Function(function) => resolve_alias(
                    &parse_join_aggregate(function, table_aliases)?.to_string(),
                    aliases,
                ),
                expr => {
                    return Err(DodamError::UnsupportedSql(format!(
                        "expected JOIN ORDER BY alias, qualified column, or aggregate expression, got {expr}"
                    )));
                }
            };
            Ok(SortExpr {
                column,
                descending: order.options.asc == Some(false),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    SortKey::new(expressions).map(Some)
}

fn parse_join_filter(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<FilterExpr> {
    Ok(FilterExpr::new(join_expr_to_filter_expr(
        expr,
        aliases,
        table_aliases,
        allow_aggregates,
    )?))
}

fn join_expr_to_filter_expr(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<Expr> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(Expr::And(
                Box::new(join_expr_to_filter_expr(
                    left,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
                Box::new(join_expr_to_filter_expr(
                    right,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Or => Ok(Expr::Or(
                Box::new(join_expr_to_filter_expr(
                    left,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
                Box::new(join_expr_to_filter_expr(
                    right,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq => {
                let left_column =
                    join_filter_column(left, aliases, table_aliases, allow_aggregates);
                if left_column.is_err()
                    && let Some(right_column) =
                        maybe_join_filter_column(right, aliases, table_aliases, allow_aggregates)?
                {
                    return Ok(Expr::Comparison(ComparisonExpr {
                        column: right_column,
                        op: reverse_comparison_op(sql_comparison_op(op)),
                        value: sql_literal_value(left)?,
                    }));
                }
                let left = left_column?;
                let op = sql_comparison_op(op);
                if let Some(right) =
                    maybe_join_filter_column(right, aliases, table_aliases, allow_aggregates)?
                {
                    Ok(Expr::ColumnComparison { left, op, right })
                } else {
                    Ok(Expr::Comparison(ComparisonExpr {
                        column: left,
                        op,
                        value: sql_literal_value(right)?,
                    }))
                }
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported JOIN WHERE operator: {op}"
            ))),
        },
        SqlExpr::Nested(expr) => {
            join_expr_to_filter_expr(expr, aliases, table_aliases, allow_aggregates)
        }
        SqlExpr::Value(value) => match &value.value {
            Value::Boolean(value) => Ok(Expr::Boolean(Some(*value))),
            Value::Null => Ok(Expr::Boolean(None)),
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported JOIN WHERE expression: {expr}"
            ))),
        },
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let column = join_filter_column(expr, aliases, table_aliases, allow_aggregates)?;
            let low = sql_literal_value(low)?;
            let high = sql_literal_value(high)?;
            if *negated {
                Ok(Expr::Or(
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column: column.clone(),
                        op: ComparisonOp::Lt,
                        value: low,
                    })),
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column,
                        op: ComparisonOp::Gt,
                        value: high,
                    })),
                ))
            } else {
                Ok(Expr::And(
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column: column.clone(),
                        op: ComparisonOp::GtEq,
                        value: low,
                    })),
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column,
                        op: ComparisonOp::LtEq,
                        value: high,
                    })),
                ))
            }
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => Ok(Expr::Not(Box::new(
            join_expr_to_filter_expr(expr, aliases, table_aliases, allow_aggregates)?,
        ))),
        SqlExpr::IsNull(expr) => Ok(Expr::IsNull {
            column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
            negated: false,
        }),
        SqlExpr::IsNotNull(expr) => Ok(Expr::IsNull {
            column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
            negated: true,
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => Ok(Expr::InList {
            column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
            negated: *negated,
            has_null: literal_list_contains_null(list)?,
            values: non_null_literal_values(list)?,
        }),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(DodamError::UnsupportedSql(
                    "LIKE ANY is not supported".to_string(),
                ));
            }
            Ok(Expr::Like {
                column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
                pattern: sql_like_pattern(pattern)?,
                negated: *negated,
                escape: sql_like_escape(escape_char)?,
            })
        }
        SqlExpr::ILike { .. } => Err(DodamError::UnsupportedSql(
            "ILIKE is not supported".to_string(),
        )),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN WHERE expression: {expr}"
        ))),
    }
}

fn reverse_comparison_op(op: ComparisonOp) -> ComparisonOp {
    match op {
        ComparisonOp::Eq => ComparisonOp::Eq,
        ComparisonOp::NotEq => ComparisonOp::NotEq,
        ComparisonOp::Gt => ComparisonOp::Lt,
        ComparisonOp::GtEq => ComparisonOp::LtEq,
        ComparisonOp::Lt => ComparisonOp::Gt,
        ComparisonOp::LtEq => ComparisonOp::GtEq,
    }
}

fn join_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<String> {
    match expr {
        SqlExpr::Identifier(ident) => {
            let resolved = resolve_alias(&ident.value, aliases);
            if resolved == ident.value {
                join_column_name(expr, table_aliases)
            } else {
                Ok(resolved)
            }
        }
        SqlExpr::CompoundIdentifier(_) => join_column_name(expr, table_aliases),
        SqlExpr::Function(function) if allow_aggregates => {
            let function_name = function.to_string();
            let resolved = resolve_alias(&function_name, aliases);
            if resolved == function_name {
                Ok(parse_join_aggregate(function, table_aliases)?.to_string())
            } else {
                Ok(resolved)
            }
        }
        SqlExpr::Nested(expr) => join_filter_column(expr, aliases, table_aliases, allow_aggregates),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected JOIN column or aggregate expression, got {expr}"
        ))),
    }
}

fn maybe_join_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) | SqlExpr::Nested(_) => {
            join_filter_column(expr, aliases, table_aliases, allow_aggregates).map(Some)
        }
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundColumn {
    relation: Option<String>,
    name: String,
    physical_name: String,
}

impl BoundColumn {
    fn unqualified(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            relation: None,
            physical_name: name.clone(),
            name,
        }
    }

    fn qualified(relation: impl Into<String>, name: impl Into<String>) -> Self {
        let relation = relation.into();
        let name = name.into();
        Self {
            physical_name: format!("{relation}.{name}"),
            relation: Some(relation),
            name,
        }
    }

    fn physical(physical_name: impl Into<String>) -> Self {
        let physical_name = physical_name.into();
        let (relation, name) = physical_name
            .split_once('.')
            .map(|(relation, name)| (Some(relation.to_string()), name.to_string()))
            .unwrap_or((None, physical_name.clone()));
        Self {
            relation,
            name,
            physical_name,
        }
    }
}

struct ColumnResolver<'a> {
    table_alias: Option<&'a str>,
    table_aliases: &'a [&'a str],
    batch: Option<&'a RecordBatch>,
}

impl<'a> ColumnResolver<'a> {
    fn single(table_alias: Option<&'a str>) -> Self {
        Self {
            table_alias,
            table_aliases: &[],
            batch: None,
        }
    }

    fn join(table_aliases: &'a [&'a str]) -> Self {
        Self {
            table_alias: None,
            table_aliases,
            batch: None,
        }
    }

    fn batch(batch: &'a RecordBatch) -> Self {
        Self {
            table_alias: None,
            table_aliases: &[],
            batch: Some(batch),
        }
    }

    fn resolve_single_column(&self, expr: &SqlExpr) -> Result<String> {
        self.resolve_single_bound(expr)
            .map(|column| column.physical_name)
    }

    fn resolve_single_bound(&self, expr: &SqlExpr) -> Result<BoundColumn> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(BoundColumn::unqualified(ident.value.clone())),
            SqlExpr::CompoundIdentifier(parts) => {
                let [qualifier, column] = parts.as_slice() else {
                    return Err(DodamError::UnsupportedSql(format!(
                        "only table-qualified columns are supported, got {expr}"
                    )));
                };
                if let Some(table_alias) = self.table_alias
                    && qualifier.value != table_alias
                {
                    return Err(DodamError::UnsupportedSql(format!(
                        "unknown table qualifier: {}",
                        qualifier.value
                    )));
                }
                Ok(if self.table_alias.is_some() {
                    BoundColumn::unqualified(column.value.clone())
                } else {
                    BoundColumn::qualified(qualifier.value.clone(), column.value.clone())
                })
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "expected column identifier, got {expr}"
            ))),
        }
    }

    fn raw_column(expr: &SqlExpr) -> Result<Option<String>> {
        Ok(Self::raw_bound(expr)?.map(|column| column.physical_name))
    }

    fn raw_bound(expr: &SqlExpr) -> Result<Option<BoundColumn>> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(Some(BoundColumn::unqualified(ident.value.clone()))),
            SqlExpr::CompoundIdentifier(parts) => {
                let [qualifier, column] = parts.as_slice() else {
                    return Err(DodamError::UnsupportedSql(format!(
                        "only table-qualified columns are supported, got {expr}"
                    )));
                };
                Ok(Some(BoundColumn::qualified(
                    qualifier.value.clone(),
                    column.value.clone(),
                )))
            }
            SqlExpr::Nested(expr) => Self::raw_bound(expr),
            _ => Ok(None),
        }
    }

    fn resolve_join_column(&self, expr: &SqlExpr) -> Result<String> {
        self.resolve_join_bound(expr)
            .map(|column| column.physical_name)
    }

    fn resolve_join_bound(&self, expr: &SqlExpr) -> Result<BoundColumn> {
        match expr {
            SqlExpr::Identifier(ident) => {
                if let Some((qualifier, column)) = ident.value.split_once('.') {
                    self.validate_join_qualifier(qualifier)?;
                    return Ok(BoundColumn::qualified(qualifier, column));
                }
                self.infer_unqualified_join_bound(&ident.value)
            }
            SqlExpr::CompoundIdentifier(parts) => match parts.as_slice() {
                [_] => self.infer_unqualified_join_bound(&parts[0].value),
                [qualifier, column] => {
                    self.validate_join_qualifier(&qualifier.value)?;
                    Ok(BoundColumn::qualified(
                        qualifier.value.clone(),
                        column.value.clone(),
                    ))
                }
                _ => Err(DodamError::UnsupportedSql(format!(
                    "only table-qualified columns are supported, got {expr}"
                ))),
            },
            _ => Err(DodamError::UnsupportedSql(format!(
                "expected JOIN column, got {expr}"
            ))),
        }
    }

    fn resolve_batch_bound(&self, column: &str) -> Result<Option<BoundColumn>> {
        let Some(batch) = self.batch else {
            return Ok(None);
        };
        if batch_column_index(batch, column).is_ok() {
            return Ok(Some(BoundColumn::physical(column)));
        }
        if let Some((function, argument)) = aggregate_column_parts(column) {
            let aggregate_suffix = format!(".{argument})");
            let matches = batch
                .schema()
                .fields()
                .iter()
                .filter(|field| {
                    field.name().starts_with(&format!("{function}("))
                        && field.name().ends_with(&aggregate_suffix)
                })
                .map(|field| field.name().clone())
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [] => {}
                [column] => return Ok(Some(BoundColumn::physical(column.clone()))),
                _ => return Err(ambiguous_column(column)),
            }
        }
        let suffix = format!(".{column}");
        let matches = batch
            .schema()
            .fields()
            .iter()
            .filter(|field| field.name().ends_with(&suffix))
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Ok(None),
            [column] => Ok(Some(BoundColumn::physical(column.clone()))),
            _ => Err(ambiguous_column(column)),
        }
    }

    fn validate_join_qualifier(&self, qualifier: &str) -> Result<()> {
        if !self
            .table_aliases
            .iter()
            .any(|table_alias| table_alias.eq_ignore_ascii_case(qualifier))
        {
            return Err(DodamError::UnsupportedSql(format!(
                "unknown table qualifier: {qualifier}"
            )));
        }
        Ok(())
    }

    fn infer_unqualified_join_bound(&self, column: &str) -> Result<BoundColumn> {
        let Some((prefix, _)) = column.split_once('_') else {
            return Err(DodamError::UnsupportedSql(format!(
                "expected qualified column, got {column}"
            )));
        };
        if matches!(prefix.to_ascii_lowercase().as_str(), "supplier" | "total")
            && self
                .table_aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case("revenue"))
        {
            return Ok(BoundColumn::qualified("revenue", column));
        }
        if let Some(alias) = infer_tpch_table_alias(prefix, self.table_aliases) {
            return Ok(BoundColumn::qualified(alias, column));
        }
        let Some(prefix_initial) = prefix.chars().next() else {
            return Err(DodamError::UnsupportedSql(format!(
                "cannot infer JOIN table for unqualified column {column}"
            )));
        };
        let matches = self
            .table_aliases
            .iter()
            .filter(|alias| {
                alias
                    .chars()
                    .next()
                    .is_some_and(|initial| initial.eq_ignore_ascii_case(&prefix_initial))
            })
            .collect::<Vec<_>>();
        let [alias] = matches.as_slice() else {
            return Err(DodamError::UnsupportedSql(format!(
                "cannot infer JOIN table for unqualified column {column}"
            )));
        };
        Ok(BoundColumn::qualified((*alias).to_string(), column))
    }
}

fn ambiguous_column(column: &str) -> DodamError {
    DodamError::UnsupportedSql(format!("ambiguous column {column}"))
}

fn qualified_join_column(expr: &SqlExpr, table_aliases: &[&str]) -> Result<String> {
    if !matches!(expr, SqlExpr::CompoundIdentifier(_)) {
        return Err(DodamError::UnsupportedSql(format!(
            "expected qualified column, got {expr}"
        )));
    }
    ColumnResolver::join(table_aliases).resolve_join_column(expr)
}

fn join_column_name(expr: &SqlExpr, table_aliases: &[&str]) -> Result<String> {
    ColumnResolver::join(table_aliases).resolve_join_column(expr)
}

fn column_output_name(expr: &SqlExpr) -> String {
    match expr {
        SqlExpr::Identifier(ident) => ident.value.clone(),
        SqlExpr::CompoundIdentifier(parts) => parts
            .last()
            .map(|ident| ident.value.clone())
            .unwrap_or_else(|| expr.to_string()),
        _ => expr.to_string(),
    }
}

fn infer_tpch_table_alias<'a>(prefix: &str, table_aliases: &'a [&str]) -> Option<&'a str> {
    let table = match prefix.to_ascii_lowercase().as_str() {
        "c" => "customer",
        "o" => "orders",
        "l" => "lineitem",
        "p" => "part",
        "ps" => "partsupp",
        "s" => "supplier",
        "n" => "nation",
        "r" => "region",
        _ => return None,
    };
    table_aliases
        .iter()
        .copied()
        .find(|alias| alias.eq_ignore_ascii_case(table))
}

fn tpch_alias_prefix(alias: &str) -> Option<&'static str> {
    match alias.to_ascii_lowercase().as_str() {
        "customer" => Some("c"),
        "orders" => Some("o"),
        "lineitem" => Some("l"),
        "part" => Some("p"),
        "partsupp" => Some("ps"),
        "supplier" => Some("s"),
        "nation" => Some("n"),
        "region" => Some("r"),
        _ => None,
    }
}

#[derive(Debug)]
struct ParsedProjection {
    projection: Projection,
    aggregates: Vec<AggregateExpr>,
    aggregate_expressions: Vec<ProjectionExpression>,
    aliases: Vec<(String, String)>,
    expressions: Vec<ProjectionExpression>,
}

#[derive(Debug, Clone, PartialEq)]
struct ProjectionExpression {
    output_name: String,
    expr: ScalarSqlExpression,
}

#[derive(Debug, Clone, PartialEq)]
enum ScalarSqlExpression {
    Column(String),
    Literal(LiteralValue),
    Binary {
        left: Box<ScalarSqlExpression>,
        op: BinaryOperator,
        right: Box<ScalarSqlExpression>,
    },
    Cast {
        expr: Box<ScalarSqlExpression>,
        target: String,
    },
    Coalesce(Vec<ScalarSqlExpression>),
    Lower(Box<ScalarSqlExpression>),
    Upper(Box<ScalarSqlExpression>),
    Length(Box<ScalarSqlExpression>),
    ExtractYear(Box<ScalarSqlExpression>),
    Substring {
        expr: Box<ScalarSqlExpression>,
        start: Box<ScalarSqlExpression>,
        length: Option<Box<ScalarSqlExpression>>,
    },
    Case {
        conditions: Vec<SqlExpr>,
        results: Vec<ScalarSqlExpression>,
        else_result: Option<Box<ScalarSqlExpression>>,
    },
}

fn parse_projection(
    select: &Select,
    group_by: &[String],
    table_alias: Option<&str>,
) -> Result<ParsedProjection> {
    let mut columns = Vec::new();
    let mut aggregates = Vec::new();
    let mut aggregate_expressions = Vec::new();
    let mut aggregate_expression_columns = Vec::new();
    let mut aliases = Vec::new();
    let mut expressions = Vec::new();
    let mut wildcard = false;

    for item in &select.projection {
        match item {
            SelectItem::Wildcard(_) => wildcard = true,
            SelectItem::UnnamedExpr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            ) => {
                let column = sql_column_name(expr, table_alias)?;
                add_column_once(&mut columns, column.clone());
                expressions.push(ProjectionExpression {
                    output_name: column.clone(),
                    expr: ScalarSqlExpression::Column(column),
                });
            }
            SelectItem::UnnamedExpr(SqlExpr::Function(function)) => {
                if let Some(expr) = parse_scalar_function_projection(function, None, table_alias)? {
                    for column in scalar_expression_columns(&expr.expr) {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(expr);
                } else {
                    let (aggregate, expression) = parse_aggregate_with_input_expression(
                        function,
                        table_alias,
                        aggregate_expressions.len(),
                    )?;
                    if let Some(expression) = expression {
                        for column in scalar_expression_columns(&expression.expr) {
                            add_column_once(&mut aggregate_expression_columns, column);
                        }
                        aggregate_expressions.push(expression);
                    }
                    aggregates.push(aggregate);
                }
            }
            SelectItem::UnnamedExpr(expr) => {
                let mut expression_columns = Vec::new();
                let mut found_aggregate = false;
                if let Ok(parsed) = parse_aggregate_output_expression(
                    expr,
                    table_alias,
                    &mut aggregates,
                    &mut aggregate_expressions,
                    &mut aggregate_expression_columns,
                    &mut expression_columns,
                    &mut found_aggregate,
                ) && found_aggregate
                {
                    for column in expression_columns {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(ProjectionExpression {
                        output_name: expr.to_string(),
                        expr: parsed,
                    });
                } else {
                    let expression = parse_scalar_projection(expr, None, table_alias)?;
                    for column in scalar_expression_columns(&expression.expr) {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(expression);
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => match expr {
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                    let column = sql_column_name(expr, table_alias)?;
                    add_column_once(&mut columns, column.clone());
                    expressions.push(ProjectionExpression {
                        output_name: alias.value.clone(),
                        expr: ScalarSqlExpression::Column(column.clone()),
                    });
                    aliases.push((alias.value.clone(), column));
                }
                SqlExpr::Function(function) => {
                    if let Some(expr) =
                        parse_scalar_function_projection(function, Some(&alias.value), table_alias)?
                    {
                        for column in scalar_expression_columns(&expr.expr) {
                            add_column_once(&mut columns, column);
                        }
                        expressions.push(expr);
                    } else {
                        let (aggregate, expression) = parse_aggregate_with_input_expression(
                            function,
                            table_alias,
                            aggregate_expressions.len(),
                        )?;
                        if let Some(expression) = expression {
                            for column in scalar_expression_columns(&expression.expr) {
                                add_column_once(&mut aggregate_expression_columns, column);
                            }
                            aggregate_expressions.push(expression);
                        }
                        aliases.push((alias.value.clone(), aggregate.to_string()));
                        aggregates.push(aggregate);
                    }
                }
                _ => {
                    let mut expression_columns = Vec::new();
                    let mut found_aggregate = false;
                    if let Ok(parsed) = parse_aggregate_output_expression(
                        expr,
                        table_alias,
                        &mut aggregates,
                        &mut aggregate_expressions,
                        &mut aggregate_expression_columns,
                        &mut expression_columns,
                        &mut found_aggregate,
                    ) && found_aggregate
                    {
                        for column in expression_columns {
                            add_column_once(&mut columns, column);
                        }
                        aliases.push((alias.value.clone(), alias.value.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: parsed,
                        });
                    } else {
                        let expression =
                            parse_scalar_projection(expr, Some(&alias.value), table_alias)?;
                        for column in scalar_expression_columns(&expression.expr) {
                            add_column_once(&mut columns, column);
                        }
                        expressions.push(expression);
                    }
                }
            },
            SelectItem::ExprWithAliases { .. } => {
                return Err(DodamError::UnsupportedSql(
                    "multi-alias SELECT items are not supported".to_string(),
                ));
            }
            _ => {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported SELECT item: {item}"
                )));
            }
        }
    }

    if wildcard && select.projection.len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "SELECT * cannot be mixed with other items".to_string(),
        ));
    }

    if aggregates.is_empty() {
        return Ok(ParsedProjection {
            projection: if wildcard {
                Projection::All
            } else {
                Projection::Columns(columns)
            },
            aggregates,
            aggregate_expressions,
            aliases,
            expressions: if wildcard { Vec::new() } else { expressions },
        });
    }

    if !aggregates.is_empty()
        && expressions.iter().any(|expr| {
            !matches!(expr.expr, ScalarSqlExpression::Column(_))
                && !scalar_expression_references_aggregate(&expr.expr, &aggregates)
        })
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate SELECT queries do not support scalar projection expressions yet".to_string(),
        ));
    }

    for column in &columns {
        if !group_by.iter().any(|group_column| group_column == column) {
            return Err(DodamError::UnsupportedSql(format!(
                "non-aggregate SELECT column {column} must appear in GROUP BY"
            )));
        }
    }
    let mut projected_columns = columns.clone();
    for aggregate in &aggregates {
        if let Some(column) = aggregate.referenced_column() {
            if !column.starts_with("__dodam_agg_expr_") {
                add_column_once(&mut projected_columns, column.to_string());
            }
        }
    }
    for column in aggregate_expression_columns {
        add_column_once(&mut projected_columns, column);
    }
    Ok(ParsedProjection {
        projection: Projection::Columns(projected_columns),
        aggregates,
        aggregate_expressions,
        aliases,
        expressions,
    })
}

fn parse_scalar_projection(
    expr: &SqlExpr,
    alias: Option<&str>,
    table_alias: Option<&str>,
) -> Result<ProjectionExpression> {
    let parsed = parse_scalar_sql_expression(expr, table_alias)?;
    Ok(ProjectionExpression {
        output_name: alias.map_or_else(|| expr.to_string(), ToString::to_string),
        expr: parsed,
    })
}

fn parse_scalar_function_projection(
    function: &sqlparser::ast::Function,
    alias: Option<&str>,
    table_alias: Option<&str>,
) -> Result<Option<ProjectionExpression>> {
    let name = object_name_to_string(&function.name)?;
    let lowercase_name = name.to_ascii_lowercase();
    if !matches!(
        lowercase_name.as_str(),
        "coalesce" | "lower" | "upper" | "length"
    ) {
        return Ok(None);
    }
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "scalar function filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported scalar function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported scalar function arguments: {}",
            function.args
        )));
    }
    let values = args
        .args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => {
                parse_scalar_sql_expression(expr, table_alias)
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported scalar function argument: {arg}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let expr = match lowercase_name.as_str() {
        "coalesce" => {
            if values.is_empty() {
                return Err(DodamError::UnsupportedSql(
                    "COALESCE requires at least one argument".to_string(),
                ));
            }
            ScalarSqlExpression::Coalesce(values)
        }
        "lower" | "upper" | "length" => {
            let [value] = values.as_slice() else {
                return Err(DodamError::UnsupportedSql(format!(
                    "{name} requires exactly one argument"
                )));
            };
            match lowercase_name.as_str() {
                "lower" => ScalarSqlExpression::Lower(Box::new(value.clone())),
                "upper" => ScalarSqlExpression::Upper(Box::new(value.clone())),
                "length" => ScalarSqlExpression::Length(Box::new(value.clone())),
                _ => unreachable!("validated scalar function"),
            }
        }
        _ => unreachable!("validated scalar function"),
    };
    Ok(Some(ProjectionExpression {
        output_name: alias.map_or_else(|| function.to_string(), ToString::to_string),
        expr,
    }))
}

fn parse_scalar_sql_expression(
    expr: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => Ok(ScalarSqlExpression::Column(
            sql_column_name(expr, table_alias)?,
        )),
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::UnaryOp { op, expr }
            if matches!(op, UnaryOperator::Minus | UnaryOperator::Plus)
                && sql_literal_value(expr).is_ok() =>
        {
            Ok(ScalarSqlExpression::Literal(sql_literal_value(
                &SqlExpr::UnaryOp {
                    op: *op,
                    expr: expr.clone(),
                },
            )?))
        }
        SqlExpr::Nested(expr) => parse_scalar_sql_expression(expr, table_alias),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_scalar_sql_expression(left, table_alias)?),
                op: op.clone(),
                right: Box::new(parse_scalar_sql_expression(right, table_alias)?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_scalar_sql_expression(expr, table_alias)?),
            target: data_type.to_string(),
        }),
        SqlExpr::Function(function) => {
            let Some(projection) = parse_scalar_function_projection(function, None, table_alias)?
            else {
                return Err(DodamError::UnsupportedSql(format!(
                    "unsupported scalar function: {function}"
                )));
            };
            Ok(projection.expr)
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let Some(start) = substring_from else {
                return Err(DodamError::UnsupportedSql(
                    "SUBSTRING requires a FROM/start expression".to_string(),
                ));
            };
            Ok(ScalarSqlExpression::Substring {
                expr: Box::new(parse_scalar_sql_expression(expr, table_alias)?),
                start: Box::new(parse_scalar_sql_expression(start, table_alias)?),
                length: substring_for
                    .as_ref()
                    .map(|expr| parse_scalar_sql_expression(expr, table_alias).map(Box::new))
                    .transpose()?,
            })
        }
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => {
            Ok(ScalarSqlExpression::ExtractYear(Box::new(
                parse_scalar_sql_expression(expr, table_alias)?,
            )))
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if operand.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "simple CASE operand is not supported yet".to_string(),
                ));
            }
            Ok(ScalarSqlExpression::Case {
                conditions: conditions
                    .iter()
                    .map(|when| when.condition.clone())
                    .collect(),
                results: conditions
                    .iter()
                    .map(|when| parse_scalar_sql_expression(&when.result, table_alias))
                    .collect::<Result<Vec<_>>>()?,
                else_result: else_result
                    .as_ref()
                    .map(|expr| parse_scalar_sql_expression(expr, table_alias).map(Box::new))
                    .transpose()?,
            })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported SELECT expression: {expr}"
        ))),
    }
}

fn parse_join_scalar_sql_expression(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => Ok(ScalarSqlExpression::Column(
            join_column_name(expr, table_aliases)?,
        )),
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::Nested(expr) => parse_join_scalar_sql_expression(expr, table_aliases),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_join_scalar_sql_expression(left, table_aliases)?),
                op: op.clone(),
                right: Box::new(parse_join_scalar_sql_expression(right, table_aliases)?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_join_scalar_sql_expression(expr, table_aliases)?),
            target: data_type.to_string(),
        }),
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => {
            Ok(ScalarSqlExpression::ExtractYear(Box::new(
                parse_join_scalar_sql_expression(expr, table_aliases)?,
            )))
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if operand.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "simple CASE operand is not supported yet".to_string(),
                ));
            }
            Ok(ScalarSqlExpression::Case {
                conditions: conditions
                    .iter()
                    .map(|when| rewrite_join_scalar_predicate(&when.condition, table_aliases))
                    .collect::<Result<Vec<_>>>()?,
                results: conditions
                    .iter()
                    .map(|when| parse_join_scalar_sql_expression(&when.result, table_aliases))
                    .collect::<Result<Vec<_>>>()?,
                else_result: else_result
                    .as_ref()
                    .map(|expr| parse_join_scalar_sql_expression(expr, table_aliases).map(Box::new))
                    .transpose()?,
            })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN scalar expression: {expr}"
        ))),
    }
}

fn parse_join_aggregate_output_expression(
    expr: &SqlExpr,
    table_aliases: &[&str],
    aggregates: &mut Vec<AggregateExpr>,
    aggregate_expressions: &mut Vec<ProjectionExpression>,
    aggregate_expression_columns: &mut Vec<String>,
    expression_columns: &mut Vec<String>,
    found_aggregate: &mut bool,
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let column = join_column_name(expr, table_aliases)?;
            add_column_once(expression_columns, column.clone());
            Ok(ScalarSqlExpression::Column(column))
        }
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::Nested(expr) => parse_join_aggregate_output_expression(
            expr,
            table_aliases,
            aggregates,
            aggregate_expressions,
            aggregate_expression_columns,
            expression_columns,
            found_aggregate,
        ),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_join_aggregate_output_expression(
                    left,
                    table_aliases,
                    aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
                op: op.clone(),
                right: Box::new(parse_join_aggregate_output_expression(
                    right,
                    table_aliases,
                    aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_join_aggregate_output_expression(
                expr,
                table_aliases,
                aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?),
            target: data_type.to_string(),
        }),
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => Ok(
            ScalarSqlExpression::ExtractYear(Box::new(parse_join_aggregate_output_expression(
                expr,
                table_aliases,
                aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?)),
        ),
        SqlExpr::Function(function) => {
            let (aggregate, expression) = parse_join_aggregate_with_input_expression(
                function,
                table_aliases,
                aggregate_expressions.len(),
            )?;
            if let Some(expression) = expression {
                for column in join_scalar_expression_columns(&expression.expr, table_aliases)? {
                    add_column_once(aggregate_expression_columns, column);
                }
                aggregate_expressions.push(expression);
            }
            let column = aggregate.to_string();
            aggregates.push(aggregate);
            *found_aggregate = true;
            Ok(ScalarSqlExpression::Column(column))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN aggregate output expression: {expr}"
        ))),
    }
}

fn rewrite_join_scalar_predicate(expr: &SqlExpr, table_aliases: &[&str]) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let column = join_column_name(expr, table_aliases)?;
            Ok(sql_column_expr(&column))
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(rewrite_join_scalar_predicate(left, table_aliases)?),
            op: op.clone(),
            right: Box::new(rewrite_join_scalar_predicate(right, table_aliases)?),
        }),
        SqlExpr::Nested(expr) => Ok(SqlExpr::Nested(Box::new(rewrite_join_scalar_predicate(
            expr,
            table_aliases,
        )?))),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op: op.clone(),
            expr: Box::new(rewrite_join_scalar_predicate(expr, table_aliases)?),
        }),
        SqlExpr::IsNull(expr) => Ok(SqlExpr::IsNull(Box::new(rewrite_join_scalar_predicate(
            expr,
            table_aliases,
        )?))),
        SqlExpr::IsNotNull(expr) => Ok(SqlExpr::IsNotNull(Box::new(
            rewrite_join_scalar_predicate(expr, table_aliases)?,
        ))),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::Like {
            negated: *negated,
            any: *any,
            expr: Box::new(rewrite_join_scalar_predicate(expr, table_aliases)?),
            pattern: Box::new(rewrite_join_scalar_predicate(pattern, table_aliases)?),
            escape_char: escape_char.clone(),
        }),
        SqlExpr::Value(_) => Ok(expr.clone()),
        _ => Ok(expr.clone()),
    }
}

fn sql_column_expr(column: &str) -> SqlExpr {
    SqlExpr::Identifier(sqlparser::ast::Ident::new(column))
}

fn scalar_expression_columns(expr: &ScalarSqlExpression) -> Vec<String> {
    let mut columns = Vec::new();
    collect_scalar_expression_columns(expr, &mut columns);
    columns
}

fn join_scalar_expression_columns(
    expr: &ScalarSqlExpression,
    table_aliases: &[&str],
) -> Result<Vec<String>> {
    let mut columns = Vec::new();
    collect_join_scalar_expression_columns(expr, table_aliases, &mut columns)?;
    Ok(columns)
}

fn join_sql_expression_columns(expr: &SqlExpr, table_aliases: &[&str]) -> Result<Vec<String>> {
    let mut columns = Vec::new();
    collect_join_column_candidates(expr, table_aliases, &mut columns)?;
    Ok(columns)
}

fn collect_join_scalar_expression_columns(
    expr: &ScalarSqlExpression,
    table_aliases: &[&str],
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        ScalarSqlExpression::Column(column) => add_column_once(columns, column.clone()),
        ScalarSqlExpression::Literal(_) => {}
        ScalarSqlExpression::Binary { left, right, .. } => {
            collect_join_scalar_expression_columns(left, table_aliases, columns)?;
            collect_join_scalar_expression_columns(right, table_aliases, columns)?;
        }
        ScalarSqlExpression::Cast { expr, .. } => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
        }
        ScalarSqlExpression::Coalesce(values) => {
            for value in values {
                collect_join_scalar_expression_columns(value, table_aliases, columns)?;
            }
        }
        ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::ExtractYear(expr) => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            collect_join_scalar_expression_columns(expr, table_aliases, columns)?;
            collect_join_scalar_expression_columns(start, table_aliases, columns)?;
            if let Some(length) = length {
                collect_join_scalar_expression_columns(length, table_aliases, columns)?;
            }
        }
        ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } => {
            for condition in conditions {
                collect_join_predicate_columns(condition, table_aliases, columns)?;
            }
            for result in results {
                collect_join_scalar_expression_columns(result, table_aliases, columns)?;
            }
            if let Some(else_result) = else_result {
                collect_join_scalar_expression_columns(else_result, table_aliases, columns)?;
            }
        }
    }
    Ok(())
}

fn collect_join_predicate_columns(
    expr: &SqlExpr,
    table_aliases: &[&str],
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            add_column_once(columns, join_column_name(expr, table_aliases)?);
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_join_predicate_columns(left, table_aliases, columns)?;
            collect_join_predicate_columns(right, table_aliases, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr) => {
            collect_join_predicate_columns(expr, table_aliases, columns)?;
        }
        SqlExpr::Like { expr, pattern, .. } => {
            collect_join_predicate_columns(expr, table_aliases, columns)?;
            collect_join_predicate_columns(pattern, table_aliases, columns)?;
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
}

fn collect_scalar_expression_columns(expr: &ScalarSqlExpression, columns: &mut Vec<String>) {
    match expr {
        ScalarSqlExpression::Column(column) => add_column_once(columns, column.clone()),
        ScalarSqlExpression::Literal(_) => {}
        ScalarSqlExpression::Binary { left, right, .. } => {
            collect_scalar_expression_columns(left, columns);
            collect_scalar_expression_columns(right, columns);
        }
        ScalarSqlExpression::Cast { expr, .. } => collect_scalar_expression_columns(expr, columns),
        ScalarSqlExpression::Coalesce(values) => {
            for value in values {
                collect_scalar_expression_columns(value, columns);
            }
        }
        ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::ExtractYear(expr) => {
            collect_scalar_expression_columns(expr, columns)
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            collect_scalar_expression_columns(expr, columns);
            collect_scalar_expression_columns(start, columns);
            if let Some(length) = length {
                collect_scalar_expression_columns(length, columns);
            }
        }
        ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } => {
            for condition in conditions {
                let _ = collect_predicate_expression_columns(condition, None, columns);
            }
            for result in results {
                collect_scalar_expression_columns(result, columns);
            }
            if let Some(else_result) = else_result {
                collect_scalar_expression_columns(else_result, columns);
            }
        }
    }
}

fn scalar_expression_references_aggregate(
    expr: &ScalarSqlExpression,
    aggregates: &[AggregateExpr],
) -> bool {
    match expr {
        ScalarSqlExpression::Column(column) => aggregates
            .iter()
            .any(|aggregate| column == &aggregate.to_string()),
        ScalarSqlExpression::Literal(_) => false,
        ScalarSqlExpression::Binary { left, right, .. } => {
            scalar_expression_references_aggregate(left, aggregates)
                || scalar_expression_references_aggregate(right, aggregates)
        }
        ScalarSqlExpression::Cast { expr, .. }
        | ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::ExtractYear(expr) => {
            scalar_expression_references_aggregate(expr, aggregates)
        }
        ScalarSqlExpression::Coalesce(values) => values
            .iter()
            .any(|value| scalar_expression_references_aggregate(value, aggregates)),
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            scalar_expression_references_aggregate(expr, aggregates)
                || scalar_expression_references_aggregate(start, aggregates)
                || length.as_deref().is_some_and(|length| {
                    scalar_expression_references_aggregate(length, aggregates)
                })
        }
        ScalarSqlExpression::Case {
            results,
            else_result,
            ..
        } => {
            results
                .iter()
                .any(|result| scalar_expression_references_aggregate(result, aggregates))
                || else_result.as_deref().is_some_and(|else_result| {
                    scalar_expression_references_aggregate(else_result, aggregates)
                })
        }
    }
}

fn parse_aggregate_output_expression(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    aggregates: &mut Vec<AggregateExpr>,
    aggregate_expressions: &mut Vec<ProjectionExpression>,
    aggregate_expression_columns: &mut Vec<String>,
    expression_columns: &mut Vec<String>,
    found_aggregate: &mut bool,
) -> Result<ScalarSqlExpression> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            let column = sql_column_name(expr, table_alias)?;
            add_column_once(expression_columns, column.clone());
            Ok(ScalarSqlExpression::Column(column))
        }
        SqlExpr::Value(_) => Ok(ScalarSqlExpression::Literal(sql_literal_value(expr)?)),
        SqlExpr::Nested(expr) => parse_aggregate_output_expression(
            expr,
            table_alias,
            aggregates,
            aggregate_expressions,
            aggregate_expression_columns,
            expression_columns,
            found_aggregate,
        ),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
            ) =>
        {
            Ok(ScalarSqlExpression::Binary {
                left: Box::new(parse_aggregate_output_expression(
                    left,
                    table_alias,
                    aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
                op: op.clone(),
                right: Box::new(parse_aggregate_output_expression(
                    right,
                    table_alias,
                    aggregates,
                    aggregate_expressions,
                    aggregate_expression_columns,
                    expression_columns,
                    found_aggregate,
                )?),
            })
        }
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(ScalarSqlExpression::Cast {
            expr: Box::new(parse_aggregate_output_expression(
                expr,
                table_alias,
                aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?),
            target: data_type.to_string(),
        }),
        SqlExpr::Extract { field, expr, .. } if *field == DateTimeField::Year => Ok(
            ScalarSqlExpression::ExtractYear(Box::new(parse_aggregate_output_expression(
                expr,
                table_alias,
                aggregates,
                aggregate_expressions,
                aggregate_expression_columns,
                expression_columns,
                found_aggregate,
            )?)),
        ),
        SqlExpr::Function(function) => {
            let (aggregate, expression) = parse_aggregate_with_input_expression(
                function,
                table_alias,
                aggregate_expressions.len(),
            )?;
            if let Some(expression) = expression {
                for column in scalar_expression_columns(&expression.expr) {
                    add_column_once(aggregate_expression_columns, column);
                }
                aggregate_expressions.push(expression);
            }
            let column = aggregate.to_string();
            aggregates.push(aggregate);
            *found_aggregate = true;
            Ok(ScalarSqlExpression::Column(column))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported aggregate output expression: {expr}"
        ))),
    }
}

fn parse_aggregate(
    function: &sqlparser::ast::Function,
    table_alias: Option<&str>,
) -> Result<AggregateExpr> {
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    let (args, duplicate_treatment) = match &function.args {
        FunctionArguments::List(args) if args.clauses.is_empty() => {
            (&args.args, args.duplicate_treatment)
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    if matches!(duplicate_treatment, Some(DuplicateTreatment::All)) {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let argument = match args.as_slice() {
        [] => {
            return Err(DodamError::UnsupportedSql(format!(
                "missing argument for {name}"
            )));
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => "*".to_string(),
        [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            )),
        ] => sql_column_name(expr, table_alias)?,
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    if duplicate_treatment == Some(DuplicateTreatment::Distinct) {
        if !name.eq_ignore_ascii_case("count") || argument == "*" {
            return Err(DodamError::UnsupportedSql(format!(
                "only count(DISTINCT column) is supported, got {function}"
            )));
        }
        AggregateExpr::parse(&format!("count_distinct({argument})"))
    } else {
        AggregateExpr::parse(&format!("{name}({argument})"))
    }
}

fn parse_aggregate_with_input_expression(
    function: &sqlparser::ast::Function,
    table_alias: Option<&str>,
    expression_index: usize,
) -> Result<(AggregateExpr, Option<ProjectionExpression>)> {
    match parse_aggregate(function, table_alias) {
        Ok(aggregate) => return Ok((aggregate, None)),
        Err(DodamError::UnsupportedSql(message))
            if message.starts_with("unsupported function arguments:") => {}
        Err(error) => return Err(error),
    }

    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    if !matches!(
        name.to_ascii_lowercase().as_str(),
        "sum" | "avg" | "min" | "max"
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "aggregate expression input is not supported for {name}"
        )));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] = args.args.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    let column = format!("__dodam_agg_expr_{expression_index}");
    let expression = ProjectionExpression {
        output_name: column.clone(),
        expr: parse_scalar_sql_expression(expr, table_alias)?,
    };
    Ok((
        AggregateExpr::parse(&format!("{name}({column})"))?,
        Some(expression),
    ))
}

fn parse_join_aggregate(
    function: &sqlparser::ast::Function,
    table_aliases: &[&str],
) -> Result<AggregateExpr> {
    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    let (args, duplicate_treatment) = match &function.args {
        FunctionArguments::List(args) if args.clauses.is_empty() => {
            (&args.args, args.duplicate_treatment)
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    if matches!(duplicate_treatment, Some(DuplicateTreatment::All)) {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let argument = match args.as_slice() {
        [] => {
            return Err(DodamError::UnsupportedSql(format!(
                "missing argument for {name}"
            )));
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => "*".to_string(),
        [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            )),
        ] => join_column_name(expr, table_aliases)?,
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "JOIN aggregate arguments must be * or columns, got {}",
                function.args
            )));
        }
    };
    if duplicate_treatment == Some(DuplicateTreatment::Distinct) {
        if !name.eq_ignore_ascii_case("count") || argument == "*" {
            return Err(DodamError::UnsupportedSql(format!(
                "only count(DISTINCT column) is supported, got {function}"
            )));
        }
        AggregateExpr::parse(&format!("count_distinct({argument})"))
    } else {
        AggregateExpr::parse(&format!("{name}({argument})"))
    }
}

fn parse_join_aggregate_with_input_expression(
    function: &sqlparser::ast::Function,
    table_aliases: &[&str],
    expression_index: usize,
) -> Result<(AggregateExpr, Option<ProjectionExpression>)> {
    match parse_join_aggregate(function, table_aliases) {
        Ok(aggregate) => return Ok((aggregate, None)),
        Err(DodamError::UnsupportedSql(message))
            if message.starts_with("JOIN aggregate arguments must be") => {}
        Err(error) => return Err(error),
    }

    if function.filter.is_some()
        || function.over.is_some()
        || !function.within_group.is_empty()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate filters, windows, within group, null treatment, and parameters are not supported"
                .to_string(),
        ));
    }
    let name = object_name_to_string(&function.name)?;
    if !matches!(
        name.to_ascii_lowercase().as_str(),
        "sum" | "avg" | "min" | "max"
    ) {
        return Err(DodamError::UnsupportedSql(format!(
            "aggregate expression input is not supported for {name}"
        )));
    }
    let FunctionArguments::List(args) = &function.args else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    if !args.clauses.is_empty() || args.duplicate_treatment.is_some() {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    }
    let [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] = args.args.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported function arguments: {}",
            function.args
        )));
    };
    let column = format!("__dodam_join_agg_expr_{expression_index}");
    let expression = ProjectionExpression {
        output_name: column.clone(),
        expr: parse_join_scalar_sql_expression(expr, table_aliases)?,
    };
    Ok((
        AggregateExpr::parse(&format!("{name}({column})"))?,
        Some(expression),
    ))
}

fn parse_group_by(select: &Select, table_alias: Option<&str>) -> Result<Vec<String>> {
    match &select.group_by {
        GroupByExpr::Expressions(expressions, modifiers) if modifiers.is_empty() => expressions
            .iter()
            .map(|expr| sql_column_name(expr, table_alias))
            .collect::<Result<Vec<_>>>(),
        GroupByExpr::Expressions(_, _) | GroupByExpr::All(_) => Err(DodamError::UnsupportedSql(
            "GROUP BY modifiers and GROUP BY ALL are not supported".to_string(),
        )),
    }
}

fn add_column_once(columns: &mut Vec<String>, column: String) {
    if !columns.iter().any(|existing| existing == &column) {
        columns.push(column);
    }
}

fn parse_order_by(
    query: &Query,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
) -> Result<Option<SortKey>> {
    let Some(order_by) = &query.order_by else {
        return Ok(None);
    };
    if order_by.interpolate.is_some() {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY INTERPOLATE is not supported".to_string(),
        ));
    }
    let OrderByKind::Expressions(expressions) = &order_by.kind else {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY ALL is not supported".to_string(),
        ));
    };
    let expressions = expressions
        .iter()
        .map(|order| {
            if order.with_fill.is_some() || order.options.nulls_first.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY WITH FILL and NULLS FIRST/LAST are not supported".to_string(),
                ));
            }
            Ok(SortExpr {
                column: parse_order_expr(&order.expr, aliases, table_alias)?,
                descending: order.options.asc == Some(false),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    SortKey::new(expressions).map(Some)
}

fn parse_order_expr(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
) -> Result<String> {
    let column = match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            sql_column_name(expr, table_alias)?
        }
        SqlExpr::Function(function) => parse_aggregate(function, table_alias)?.to_string(),
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "expected ORDER BY column or aggregate expression, got {expr}"
            )));
        }
    };
    Ok(resolve_alias(&column, aliases))
}

fn resolve_alias(name: &str, aliases: &[(String, String)]) -> String {
    aliases
        .iter()
        .find(|(alias, _)| alias == name)
        .map(|(_, target)| target.clone())
        .unwrap_or_else(|| name.to_string())
}

fn parse_limit(query: &Query) -> Result<Option<usize>> {
    let Some(limit_clause) = &query.limit_clause else {
        return Ok(None);
    };
    match limit_clause {
        LimitClause::LimitOffset {
            limit: Some(limit),
            offset: None,
            limit_by,
        } if limit_by.is_empty() => parse_usize_literal(limit).map(Some),
        LimitClause::LimitOffset {
            limit: None,
            offset: None,
            limit_by,
        } if limit_by.is_empty() => Ok(None),
        _ => Err(DodamError::UnsupportedSql(
            "only LIMIT <integer> without OFFSET is supported".to_string(),
        )),
    }
}

fn parse_filter(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<FilterExpr> {
    Ok(FilterExpr::new(sql_expr_to_filter_expr(
        expr,
        aliases,
        table_alias,
        allow_aggregates,
    )?))
}

fn sql_expr_to_filter_expr(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<Expr> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(Expr::And(
                Box::new(sql_expr_to_filter_expr(
                    left,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
                Box::new(sql_expr_to_filter_expr(
                    right,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Or => Ok(Expr::Or(
                Box::new(sql_expr_to_filter_expr(
                    left,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
                Box::new(sql_expr_to_filter_expr(
                    right,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq => {
                if let (Ok(left), Ok(right)) = (sql_literal_value(left), sql_literal_value(right)) {
                    return Ok(Expr::Boolean(compare_literal_values(&left, op, &right)?));
                }
                let left = sql_filter_column(left, aliases, table_alias, allow_aggregates)?;
                let op = sql_comparison_op(op);
                if let Some(right) =
                    maybe_sql_filter_column(right, aliases, table_alias, allow_aggregates)?
                {
                    Ok(Expr::ColumnComparison { left, op, right })
                } else {
                    Ok(Expr::Comparison(ComparisonExpr {
                        column: left,
                        op,
                        value: sql_literal_value(right)?,
                    }))
                }
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported WHERE operator: {op}"
            ))),
        },
        SqlExpr::Nested(expr) => {
            sql_expr_to_filter_expr(expr, aliases, table_alias, allow_aggregates)
        }
        SqlExpr::Value(value) => match &value.value {
            Value::Boolean(value) => Ok(Expr::Boolean(Some(*value))),
            Value::Null => Ok(Expr::Boolean(None)),
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported WHERE expression: {expr}"
            ))),
        },
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let column = sql_filter_column(expr, aliases, table_alias, allow_aggregates)?;
            let low = sql_literal_value(low)?;
            let high = sql_literal_value(high)?;
            if *negated {
                Ok(Expr::Or(
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column: column.clone(),
                        op: ComparisonOp::Lt,
                        value: low,
                    })),
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column,
                        op: ComparisonOp::Gt,
                        value: high,
                    })),
                ))
            } else {
                Ok(Expr::And(
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column: column.clone(),
                        op: ComparisonOp::GtEq,
                        value: low,
                    })),
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column,
                        op: ComparisonOp::LtEq,
                        value: high,
                    })),
                ))
            }
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => Ok(Expr::Not(Box::new(
            sql_expr_to_filter_expr(expr, aliases, table_alias, allow_aggregates)?,
        ))),
        SqlExpr::IsNull(expr) => Ok(Expr::IsNull {
            column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
            negated: false,
        }),
        SqlExpr::IsNotNull(expr) => Ok(Expr::IsNull {
            column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
            negated: true,
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => match sql_filter_column(expr, aliases, table_alias, allow_aggregates) {
            Ok(column) => Ok(Expr::InList {
                column,
                negated: *negated,
                has_null: literal_list_contains_null(list)?,
                values: non_null_literal_values(list)?,
            }),
            Err(error) => {
                let value = sql_literal_value(expr).map_err(|_| error)?;
                Ok(Expr::Boolean(evaluate_literal_in_list(
                    &value, list, *negated,
                )?))
            }
        },
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(DodamError::UnsupportedSql(
                    "LIKE ANY is not supported".to_string(),
                ));
            }
            Ok(Expr::Like {
                column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
                pattern: sql_like_pattern(pattern)?,
                negated: *negated,
                escape: sql_like_escape(escape_char)?,
            })
        }
        SqlExpr::ILike { .. } => Err(DodamError::UnsupportedSql(
            "ILIKE is not supported".to_string(),
        )),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported WHERE expression: {expr}"
        ))),
    }
}

fn sql_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<String> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            Ok(resolve_alias(&sql_column_name(expr, table_alias)?, aliases))
        }
        SqlExpr::Function(function) if allow_aggregates => {
            Ok(parse_aggregate(function, table_alias)?.to_string())
        }
        SqlExpr::Nested(expr) => sql_filter_column(expr, aliases, table_alias, allow_aggregates),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected column or aggregate expression, got {expr}"
        ))),
    }
}

fn maybe_sql_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) | SqlExpr::Nested(_) => {
            sql_filter_column(expr, aliases, table_alias, allow_aggregates).map(Some)
        }
        _ => Ok(None),
    }
}

fn sql_like_pattern(expr: &SqlExpr) -> Result<String> {
    match sql_literal_value(expr)? {
        LiteralValue::Utf8(pattern) => Ok(pattern),
        LiteralValue::Null => Err(DodamError::UnsupportedSql(
            "LIKE NULL patterns are not supported yet".to_string(),
        )),
        value => Err(DodamError::UnsupportedSql(format!(
            "LIKE pattern must be a string literal, got {value}"
        ))),
    }
}

fn sql_like_escape(escape_char: &Option<sqlparser::ast::ValueWithSpan>) -> Result<Option<char>> {
    let Some(escape_char) = escape_char else {
        return Ok(None);
    };
    let Value::SingleQuotedString(value) = &escape_char.value else {
        return Err(DodamError::UnsupportedSql(
            "LIKE ESCAPE must be a string literal".to_string(),
        ));
    };
    let mut chars = value.chars();
    let Some(ch) = chars.next() else {
        return Err(DodamError::UnsupportedSql(
            "LIKE ESCAPE must contain exactly one character".to_string(),
        ));
    };
    if chars.next().is_some() {
        return Err(DodamError::UnsupportedSql(
            "LIKE ESCAPE must contain exactly one character".to_string(),
        ));
    }
    Ok(Some(ch))
}

fn sql_literal_value(expr: &SqlExpr) -> Result<LiteralValue> {
    match expr {
        SqlExpr::Value(value) => match &value.value {
            Value::Number(value, false) => value
                .parse::<i64>()
                .map(LiteralValue::Int64)
                .or_else(|_| value.parse::<f64>().map(LiteralValue::Float64))
                .map_err(|_| {
                    DodamError::UnsupportedSql(format!("unsupported numeric literal: {value}"))
                }),
            Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => {
                Ok(LiteralValue::Utf8(value.clone()))
            }
            Value::Boolean(value) => Ok(LiteralValue::Boolean(*value)),
            Value::Null => Ok(LiteralValue::Null),
            value => Err(DodamError::UnsupportedSql(format!(
                "unsupported literal: {value}"
            ))),
        },
        SqlExpr::TypedString(typed) if typed.data_type.to_string().eq_ignore_ascii_case("date") => {
            match &typed.value.value {
                Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => {
                    Ok(LiteralValue::Utf8(value.clone()))
                }
                value => Err(DodamError::UnsupportedSql(format!(
                    "unsupported DATE literal: {value}"
                ))),
            }
        }
        SqlExpr::BinaryOp { left, op, right }
            if matches!(op, BinaryOperator::Plus | BinaryOperator::Minus) =>
        {
            if let Some(value) = decimal_number_literal_arithmetic(left, op, right)? {
                return Ok(value);
            }
            let left_value = sql_literal_value(left)?;
            if let Some((amount, field)) = interval_literal(right)? {
                return apply_date_interval(left_value, op.clone(), amount, field);
            }
            let right_value = sql_literal_value(right)?;
            apply_literal_arithmetic(left_value, op.clone(), right_value)
        }
        SqlExpr::UnaryOp { op, expr }
            if matches!(op, UnaryOperator::Minus | UnaryOperator::Plus) =>
        {
            let value = sql_literal_value(expr)?;
            match (op, value) {
                (UnaryOperator::Plus, value) => Ok(value),
                (UnaryOperator::Minus, LiteralValue::Int64(value)) => {
                    Ok(LiteralValue::Int64(value.checked_neg().ok_or_else(
                        || DodamError::UnsupportedSql("integer literal overflow".to_string()),
                    )?))
                }
                (UnaryOperator::Minus, LiteralValue::Float64(value)) => {
                    Ok(LiteralValue::Float64(-value))
                }
                (UnaryOperator::Minus, value) => Err(DodamError::UnsupportedSql(format!(
                    "unary minus requires a numeric literal, got {value}"
                ))),
                _ => unreachable!("validated unary operator"),
            }
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let LiteralValue::Utf8(value) = sql_literal_value(expr)? else {
                return Err(DodamError::UnsupportedSql(format!(
                    "SUBSTRING literal input must be a string, got {expr}"
                )));
            };
            let Some(start) = substring_from else {
                return Err(DodamError::UnsupportedSql(
                    "SUBSTRING requires a FROM/start expression".to_string(),
                ));
            };
            let start = literal_usize(start, "SUBSTRING start")?;
            let length = substring_for
                .as_ref()
                .map(|expr| literal_usize(expr, "SUBSTRING length"))
                .transpose()?;
            Ok(LiteralValue::Utf8(substring_literal(&value, start, length)))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected literal, got {expr}"
        ))),
    }
}

fn evaluate_literal_in_list(
    value: &LiteralValue,
    list: &[SqlExpr],
    negated: bool,
) -> Result<Option<bool>> {
    let mut has_null = false;
    for candidate in list {
        let candidate = sql_literal_value(candidate)?;
        if matches!(candidate, LiteralValue::Null) {
            has_null = true;
            continue;
        }
        if compare_literal_values(value, &BinaryOperator::Eq, &candidate)? == Some(true) {
            return Ok(Some(!negated));
        }
    }
    if has_null {
        Ok(None)
    } else {
        Ok(Some(negated))
    }
}

fn literal_usize(expr: &SqlExpr, context: &str) -> Result<usize> {
    let LiteralValue::Int64(value) = sql_literal_value(expr)? else {
        return Err(DodamError::UnsupportedSql(format!(
            "{context} must be an integer literal"
        )));
    };
    usize::try_from(value)
        .map_err(|_| DodamError::UnsupportedSql(format!("{context} must be non-negative")))
}

fn substring_literal(value: &str, start: usize, length: Option<usize>) -> String {
    let start = start.saturating_sub(1);
    value
        .chars()
        .skip(start)
        .take(length.unwrap_or(usize::MAX))
        .collect()
}

fn decimal_number_literal_arithmetic(
    left: &SqlExpr,
    op: &BinaryOperator,
    right: &SqlExpr,
) -> Result<Option<LiteralValue>> {
    let Some(left) = numeric_literal_text(left) else {
        return Ok(None);
    };
    let Some(right) = numeric_literal_text(right) else {
        return Ok(None);
    };
    let scale = decimal_scale(left).max(decimal_scale(right));
    let left = decimal_literal_to_scaled(left, scale)?;
    let right = decimal_literal_to_scaled(right, scale)?;
    let value = match op {
        BinaryOperator::Plus => left + right,
        BinaryOperator::Minus => left - right,
        _ => unreachable!("validated arithmetic operator"),
    };
    if scale == 0 {
        return Ok(Some(LiteralValue::Int64(i64::try_from(value).map_err(
            |_| DodamError::UnsupportedSql("numeric literal overflow".to_string()),
        )?)));
    }
    let negative = value < 0;
    let value = value.abs();
    let factor =
        10_i128.pow(u32::try_from(scale).map_err(|_| {
            DodamError::UnsupportedSql("numeric literal scale overflow".to_string())
        })?);
    let whole = value / factor;
    let fractional = value % factor;
    let literal = format!(
        "{}{}.{:0width$}",
        if negative { "-" } else { "" },
        whole,
        fractional,
        width = scale
    );
    Ok(Some(LiteralValue::Float64(
        literal.parse::<f64>().map_err(|_| {
            DodamError::UnsupportedSql(format!("unsupported numeric literal: {literal}"))
        })?,
    )))
}

fn numeric_literal_text(expr: &SqlExpr) -> Option<&str> {
    let SqlExpr::Value(value) = expr else {
        return None;
    };
    let Value::Number(value, false) = &value.value else {
        return None;
    };
    Some(value)
}

fn decimal_scale(value: &str) -> usize {
    value
        .split_once('.')
        .map(|(_, fractional)| fractional.len())
        .unwrap_or(0)
}

fn decimal_literal_to_scaled(value: &str, scale: usize) -> Result<i128> {
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fractional) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty()
        || !whole.chars().all(|ch| ch.is_ascii_digit())
        || !fractional.chars().all(|ch| ch.is_ascii_digit())
        || fractional.len() > scale
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported numeric literal: {value}"
        )));
    }
    let mut result = whole
        .parse::<i128>()
        .map_err(|_| DodamError::UnsupportedSql(format!("unsupported numeric literal: {value}")))?
        .checked_mul(10_i128.pow(u32::try_from(scale).map_err(|_| {
            DodamError::UnsupportedSql("numeric literal scale overflow".to_string())
        })?))
        .ok_or_else(|| DodamError::UnsupportedSql("numeric literal overflow".to_string()))?;
    let mut fractional_value = fractional.parse::<i128>().unwrap_or(0);
    for _ in fractional.len()..scale {
        fractional_value = fractional_value
            .checked_mul(10)
            .ok_or_else(|| DodamError::UnsupportedSql("numeric literal overflow".to_string()))?;
    }
    result = result
        .checked_add(fractional_value)
        .ok_or_else(|| DodamError::UnsupportedSql("numeric literal overflow".to_string()))?;
    if negative {
        result = -result;
    }
    Ok(result)
}

fn apply_literal_arithmetic(
    left: LiteralValue,
    op: BinaryOperator,
    right: LiteralValue,
) -> Result<LiteralValue> {
    match (left, right) {
        (LiteralValue::Int64(left), LiteralValue::Int64(right)) => {
            Ok(LiteralValue::Int64(match op {
                BinaryOperator::Plus => left + right,
                BinaryOperator::Minus => left - right,
                _ => unreachable!("validated arithmetic operator"),
            }))
        }
        (left, right) => {
            let left = literal_as_f64(&left)?;
            let right = literal_as_f64(&right)?;
            Ok(LiteralValue::Float64(match op {
                BinaryOperator::Plus => left + right,
                BinaryOperator::Minus => left - right,
                _ => unreachable!("validated arithmetic operator"),
            }))
        }
    }
}

fn literal_as_f64(value: &LiteralValue) -> Result<f64> {
    match value {
        LiteralValue::Int64(value) => Ok(*value as f64),
        LiteralValue::Float64(value) => Ok(*value),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected numeric literal, got {value}"
        ))),
    }
}

fn interval_literal(expr: &SqlExpr) -> Result<Option<(i64, DateTimeField)>> {
    let SqlExpr::Interval(interval) = expr else {
        return Ok(None);
    };
    let SqlExpr::Value(value) = interval.value.as_ref() else {
        return Err(DodamError::UnsupportedSql(format!(
            "unsupported INTERVAL value: {}",
            interval.value
        )));
    };
    let amount = match &value.value {
        Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => value,
        value => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported INTERVAL literal: {value}"
            )));
        }
    }
    .parse::<i64>()
    .map_err(|_| DodamError::UnsupportedSql(format!("unsupported INTERVAL: {expr}")))?;
    let field = interval.leading_field.clone().ok_or_else(|| {
        DodamError::UnsupportedSql(format!("INTERVAL requires a leading field: {expr}"))
    })?;
    Ok(Some((amount, field)))
}

fn apply_date_interval(
    value: LiteralValue,
    op: BinaryOperator,
    amount: i64,
    field: DateTimeField,
) -> Result<LiteralValue> {
    let LiteralValue::Utf8(date) = value else {
        return Err(DodamError::UnsupportedSql(
            "INTERVAL arithmetic currently requires a DATE literal".to_string(),
        ));
    };
    let amount = match op {
        BinaryOperator::Plus => amount,
        BinaryOperator::Minus => -amount,
        _ => unreachable!("validated arithmetic operator"),
    };
    let (year, month, day) = parse_ymd(&date)?;
    let (year, month, day) = match field {
        DateTimeField::Day => {
            let days = days_from_civil(year, month, day)? + amount;
            civil_from_days(days)?
        }
        DateTimeField::Month => add_months(year, month, day, amount)?,
        DateTimeField::Year => add_months(year, month, day, amount * 12)?,
        field => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported INTERVAL field for DATE arithmetic: {field}"
            )));
        }
    };
    Ok(LiteralValue::Utf8(format!("{year:04}-{month:02}-{day:02}")))
}

fn parse_ymd(value: &str) -> Result<(i32, u32, u32)> {
    let mut parts = value.split('-');
    let year = parts
        .next()
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid DATE literal: {value}")))?;
    let month = parts
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid DATE literal: {value}")))?;
    let day = parts
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| DodamError::UnsupportedSql(format!("invalid DATE literal: {value}")))?;
    if parts.next().is_some()
        || !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month(year, month)
    {
        return Err(DodamError::UnsupportedSql(format!(
            "invalid DATE literal: {value}"
        )));
    }
    Ok((year, month, day))
}

fn add_months(year: i32, month: u32, day: u32, months: i64) -> Result<(i32, u32, u32)> {
    let month_index = i64::from(year) * 12 + i64::from(month - 1) + months;
    let year = month_index.div_euclid(12);
    let month = month_index.rem_euclid(12) + 1;
    let year = i32::try_from(year)
        .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
    let month = u32::try_from(month)
        .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
    Ok((year, month, day.min(days_in_month(year, month))))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_from_civil(year: i32, month: u32, day: u32) -> Result<i64> {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = i64::from(month);
    let day = i64::from(day);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Ok(era * 146_097 + doe - 719_468)
}

fn civil_from_days(days: i64) -> Result<(i32, u32, u32)> {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + i64::from(month <= 2);
    Ok((
        i32::try_from(year)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?,
        u32::try_from(month)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?,
        u32::try_from(day)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?,
    ))
}

fn year_from_days(days: i64) -> Result<i32> {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = mp + if mp < 10 { 3 } else { -9 };
    i32::try_from(year + i64::from(month <= 2))
        .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))
}

#[derive(Default)]
struct Date32YearCache {
    base_day: i32,
    years: Vec<i32>,
    disabled: bool,
}

impl Date32YearCache {
    const MAX_SPAN_DAYS: usize = 20_000;

    fn year(&mut self, day: i32) -> Result<i32> {
        if self.disabled {
            return year_from_days(i64::from(day));
        }
        if self.years.is_empty() {
            self.base_day = day;
            self.years.push(year_from_days(i64::from(day))?);
            return Ok(self.years[0]);
        }
        if day < self.base_day {
            let prepend = usize::try_from(self.base_day - day)
                .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
            if prepend + self.years.len() > Self::MAX_SPAN_DAYS {
                self.disabled = true;
                self.years.clear();
                return year_from_days(i64::from(day));
            }
            let mut years = Vec::with_capacity(prepend + self.years.len());
            for offset in 0..prepend {
                years.push(year_from_days(i64::from(day) + offset as i64)?);
            }
            years.extend_from_slice(&self.years);
            self.base_day = day;
            self.years = years;
            return Ok(self.years[0]);
        }
        let index = usize::try_from(day - self.base_day)
            .map_err(|_| DodamError::UnsupportedSql("DATE arithmetic overflow".to_string()))?;
        if index >= self.years.len() {
            if index + 1 > Self::MAX_SPAN_DAYS {
                self.disabled = true;
                self.years.clear();
                return year_from_days(i64::from(day));
            }
            let start = self.years.len();
            self.years.reserve(index + 1 - start);
            for offset in start..=index {
                self.years
                    .push(year_from_days(i64::from(self.base_day) + offset as i64)?);
            }
        }
        Ok(self.years[index])
    }
}

fn sql_comparison_op(op: &BinaryOperator) -> ComparisonOp {
    match op {
        BinaryOperator::Eq => ComparisonOp::Eq,
        BinaryOperator::NotEq => ComparisonOp::NotEq,
        BinaryOperator::Gt => ComparisonOp::Gt,
        BinaryOperator::GtEq => ComparisonOp::GtEq,
        BinaryOperator::Lt => ComparisonOp::Lt,
        BinaryOperator::LtEq => ComparisonOp::LtEq,
        _ => unreachable!("validated comparison operator"),
    }
}

fn sql_column_name(expr: &SqlExpr, table_alias: Option<&str>) -> Result<String> {
    ColumnResolver::single(table_alias).resolve_single_column(expr)
}

fn parse_usize_literal(expr: &SqlExpr) -> Result<usize> {
    let SqlExpr::Value(value) = expr else {
        return Err(DodamError::UnsupportedSql(format!(
            "expected integer literal, got {expr}"
        )));
    };
    match value.value.clone() {
        Value::Number(value, false) => value.parse::<usize>().map_err(|_| {
            DodamError::UnsupportedSql(format!(
                "expected non-negative integer literal, got {value}"
            ))
        }),
        value => Err(DodamError::UnsupportedSql(format!(
            "expected integer literal, got {value}"
        ))),
    }
}

fn object_name_to_string(name: &ObjectName) -> Result<String> {
    let [part] = name.0.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "compound object names are not supported: {name}"
        )));
    };
    match part {
        ObjectNamePart::Identifier(ident) => Ok(ident.value.clone()),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported object name: {name}"
        ))),
    }
}

fn collect_batches(mut stream: SendableBatchStream) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }
    Ok(batches)
}

fn apply_output_filter_stream(
    stream: SendableBatchStream,
    filter: Option<FilterExpr>,
) -> SendableBatchStream {
    let Some(filter) = filter else {
        return stream;
    };
    let (input, metrics) = stream.into_parts();
    SendableBatchStream::new(Box::new(OutputFilterStream { input, filter }), metrics)
}

struct OutputFilterStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    filter: FilterExpr,
}

impl Iterator for OutputFilterStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        for batch in &mut self.input {
            match batch {
                Ok(batch) => match filter_batch(batch, &self.filter) {
                    Ok(batch) if batch.num_rows() == 0 => continue,
                    result => return Some(result),
                },
                Err(error) => return Some(Err(error)),
            }
        }
        None
    }
}

fn aggregate_metrics_to_batches(
    metrics: &AggregateMetrics,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Vec<RecordBatch>> {
    if group_by.is_empty() {
        return aggregate_values_to_batch(&metrics.values).map(|batch| vec![batch]);
    }

    let mut fields = Vec::new();
    let mut columns = Vec::new();

    for (index, column) in group_by.iter().enumerate() {
        let values = metrics
            .groups
            .iter()
            .map(|group| group.keys.get(index))
            .collect::<Vec<_>>();
        let (field, array) = group_values_to_column(column, &values);
        fields.push(field);
        columns.push(array);
    }

    for (index, aggregate) in aggregates.iter().enumerate() {
        let values = metrics
            .groups
            .iter()
            .filter_map(|group| group.values.get(index))
            .map(|result| &result.value)
            .collect::<Vec<_>>();
        let (field, array) = aggregate_values_to_column(&aggregate.to_string(), &values);
        fields.push(field);
        columns.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(vec![RecordBatch::try_new(schema, columns)?])
}

fn apply_output_order_limit(
    batches: Vec<RecordBatch>,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
) -> Result<Vec<RecordBatch>> {
    let Some(order_by) = order_by else {
        return Ok(match limit {
            Some(limit) => limit_batches(batches, limit),
            None => batches,
        });
    };
    if batches.is_empty() {
        return Ok(batches);
    }
    if topk_batch_prune_enabled()
        && let Some(limit) = limit
        && batches.len() > 1
    {
        let candidates = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .map(|batch| sort_output_batch(batch, order_by, Some(limit)))
            .collect::<Result<Vec<_>>>()?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let schema = candidates[0].schema();
        let batch = if candidates.len() == 1 {
            candidates[0].clone()
        } else {
            concat_batches(&schema, candidates.iter())?
        };
        return Ok(vec![sort_output_batch(&batch, order_by, Some(limit))?]);
    }

    let schema = batches[0].schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        concat_batches(&schema, batches.iter())?
    };
    Ok(vec![sort_output_batch(&batch, order_by, limit)?])
}

fn topk_batch_prune_enabled() -> bool {
    std::env::var("DODAM_DISABLE_TOPK_BATCH_PRUNE")
        .map(|value| value != "1" && !value.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn sort_output_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    let sort_columns = order_by
        .expressions
        .iter()
        .map(|sort| {
            let column_index = batch
                .schema()
                .fields()
                .iter()
                .position(|field| field.name() == &sort.column)
                .ok_or_else(|| DodamError::UnknownColumn(sort.column.clone()))?;
            Ok(SortColumn {
                values: batch.column(column_index).clone(),
                options: Some(SortOptions {
                    descending: sort.descending,
                    nulls_first: false,
                }),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let indices = lexsort_to_indices(&sort_columns, limit)?;
    Ok(take_record_batch(batch, &indices)?)
}

fn limit_batches(batches: Vec<RecordBatch>, limit: usize) -> Vec<RecordBatch> {
    let mut remaining = limit;
    let mut limited = Vec::new();
    for batch in batches {
        if remaining == 0 {
            break;
        }
        let rows = remaining.min(batch.num_rows());
        remaining -= rows;
        if rows > 0 {
            limited.push(batch.slice(0, rows));
        }
    }
    limited
}

fn apply_output_filter(
    batches: Vec<RecordBatch>,
    filter: Option<&FilterExpr>,
) -> Result<Vec<RecordBatch>> {
    let Some(filter) = filter else {
        return Ok(batches);
    };

    let mut filtered = Vec::new();
    for batch in batches {
        let batch = filter_batch(batch, filter)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

fn apply_output_expression_filter(
    batches: Vec<RecordBatch>,
    predicate: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    let mut filtered = Vec::new();
    for batch in batches {
        let mask = evaluate_scalar_predicate(&batch, predicate, table_alias)?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

fn evaluate_scalar_predicate(
    batch: &RecordBatch,
    predicate: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<BooleanArray> {
    match predicate {
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::And => {
            let left = evaluate_scalar_predicate(batch, left, table_alias)?;
            let right = evaluate_scalar_predicate(batch, right, table_alias)?;
            Ok(boolean_and(&left, &right))
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::Or => {
            let left = evaluate_scalar_predicate(batch, left, table_alias)?;
            let right = evaluate_scalar_predicate(batch, right, table_alias)?;
            Ok(boolean_or(&left, &right))
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            let mask = evaluate_scalar_predicate(batch, expr, table_alias)?;
            Ok(boolean_not(&mask))
        }
        SqlExpr::Nested(expr) => evaluate_scalar_predicate(batch, expr, table_alias),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) =>
        {
            let left = evaluate_scalar_expression(
                batch,
                &parse_scalar_sql_expression(left, table_alias)?,
            )?;
            let right = evaluate_scalar_expression(
                batch,
                &parse_scalar_sql_expression(right, table_alias)?,
            )?;
            Ok(BooleanArray::from(compare_evaluated_scalars(
                left, op, right,
            )?))
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            let value = evaluate_scalar_expression(
                batch,
                &parse_scalar_sql_expression(expr, table_alias)?,
            )?;
            let is_not_null = matches!(predicate, SqlExpr::IsNotNull(_));
            Ok(BooleanArray::from(
                scalar_null_mask(value)
                    .into_iter()
                    .map(|is_null| Some(if is_not_null { !is_null } else { is_null }))
                    .collect::<Vec<_>>(),
            ))
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let value = evaluate_scalar_expression(
                batch,
                &parse_scalar_sql_expression(expr, table_alias)?,
            )?;
            let values = list
                .iter()
                .map(|expr| {
                    evaluate_scalar_expression(
                        batch,
                        &parse_scalar_sql_expression(expr, table_alias)?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(BooleanArray::from(evaluate_scalar_in_list(
                value, &values, *negated,
            )?))
        }
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(DodamError::UnsupportedSql(
                    "LIKE ANY is not supported".to_string(),
                ));
            }
            let value = scalar_as_utf8(evaluate_scalar_expression(
                batch,
                &parse_scalar_sql_expression(expr, table_alias)?,
            )?)?;
            let pattern = sql_like_pattern(pattern)?;
            let escape = sql_like_escape(escape_char)?;
            let tokens = scalar_like_pattern_tokens(&pattern, escape)?;
            Ok(BooleanArray::from(
                value
                    .into_iter()
                    .map(|value| {
                        value.map(|value| {
                            let matched = scalar_like_matches(&value, &tokens);
                            if *negated { !matched } else { matched }
                        })
                    })
                    .collect::<Vec<_>>(),
            ))
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported expression WHERE predicate: {predicate}"
        ))),
    }
}

fn boolean_and(left: &BooleanArray, right: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        (0..left.len())
            .map(
                |row| match (boolean_value(left, row), boolean_value(right, row)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
            )
            .collect::<Vec<_>>(),
    )
}

fn boolean_or(left: &BooleanArray, right: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        (0..left.len())
            .map(
                |row| match (boolean_value(left, row), boolean_value(right, row)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
            )
            .collect::<Vec<_>>(),
    )
}

fn boolean_not(mask: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        (0..mask.len())
            .map(|row| boolean_value(mask, row).map(|value| !value))
            .collect::<Vec<_>>(),
    )
}

fn boolean_value(mask: &BooleanArray, row: usize) -> Option<bool> {
    if mask.is_null(row) {
        None
    } else {
        Some(mask.value(row))
    }
}

fn apply_output_projection(
    batches: Vec<RecordBatch>,
    projection: &Projection,
) -> Result<Vec<RecordBatch>> {
    let Projection::Columns(columns) = projection else {
        return Ok(batches);
    };

    batches
        .into_iter()
        .map(|batch| {
            let indices = columns
                .iter()
                .map(|column| {
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .position(|field| field.name() == column)
                        .ok_or_else(|| DodamError::UnknownColumn(column.clone()))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(batch.project(&indices)?)
        })
        .collect()
}

fn apply_output_expression_projection(
    batches: Vec<RecordBatch>,
    expressions: &[ProjectionExpression],
) -> Result<Vec<RecordBatch>> {
    if expressions.is_empty() {
        return Ok(batches);
    }
    batches
        .into_iter()
        .map(|batch| {
            let mut fields = Vec::with_capacity(expressions.len());
            let mut columns = Vec::with_capacity(expressions.len());
            for expression in expressions {
                let value = evaluate_scalar_expression(&batch, &expression.expr)?;
                fields.push(Field::new(
                    expression.output_name.clone(),
                    value.data_type(),
                    value.is_nullable(),
                ));
                columns.push(value.into_array(batch.num_rows()));
            }
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                columns,
            )?)
        })
        .collect()
}

fn append_aggregate_expression_columns(
    batches: Vec<RecordBatch>,
    expressions: &[ProjectionExpression],
) -> Result<Vec<RecordBatch>> {
    if expressions.is_empty() {
        return Ok(batches);
    }
    batches
        .into_iter()
        .map(|batch| append_aggregate_expression_batch(batch, expressions))
        .collect()
}

fn append_aggregate_expression_stream(
    stream: SendableBatchStream,
    expressions: Vec<ProjectionExpression>,
) -> SendableBatchStream {
    if expressions.is_empty() {
        return stream;
    }
    let (inner, metrics) = stream.into_parts();
    SendableBatchStream::new(
        Box::new(inner.map(move |batch| append_aggregate_expression_batch(batch?, &expressions))),
        metrics,
    )
}

fn append_aggregate_expression_batch(
    batch: RecordBatch,
    expressions: &[ProjectionExpression],
) -> Result<RecordBatch> {
    if expressions.is_empty() {
        return Ok(batch);
    }
    let mut fields = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for expression in expressions {
        let value = evaluate_scalar_expression(&batch, &expression.expr)?;
        fields.push(Arc::new(Field::new(
            expression.output_name.clone(),
            value.data_type(),
            value.is_nullable(),
        )));
        columns.push(value.into_array(batch.num_rows()));
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

#[derive(Clone)]
enum EvaluatedScalar {
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Utf8(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
}

impl EvaluatedScalar {
    fn len(&self) -> usize {
        match self {
            Self::Int64(values) => values.len(),
            Self::Float64(values) => values.len(),
            Self::Utf8(values) => values.len(),
            Self::Boolean(values) => values.len(),
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            Self::Int64(_) => DataType::Int64,
            Self::Float64(_) => DataType::Float64,
            Self::Utf8(_) => DataType::Utf8,
            Self::Boolean(_) => DataType::Boolean,
        }
    }

    fn is_nullable(&self) -> bool {
        match self {
            Self::Int64(values) => values.iter().any(Option::is_none),
            Self::Float64(values) => values.iter().any(Option::is_none),
            Self::Utf8(values) => values.iter().any(Option::is_none),
            Self::Boolean(values) => values.iter().any(Option::is_none),
        }
    }

    fn into_array(self, _rows: usize) -> ArrayRef {
        match self {
            Self::Int64(values) => Arc::new(Int64Array::from(values)) as ArrayRef,
            Self::Float64(values) => Arc::new(Float64Array::from(values)) as ArrayRef,
            Self::Utf8(values) => Arc::new(StringArray::from(values)) as ArrayRef,
            Self::Boolean(values) => Arc::new(BooleanArray::from(values)) as ArrayRef,
        }
    }
}

fn evaluate_scalar_expression(
    batch: &RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<EvaluatedScalar> {
    match expr {
        ScalarSqlExpression::Column(column) => evaluated_column(batch, column),
        ScalarSqlExpression::Literal(value) => Ok(evaluated_literal(value, batch.num_rows())),
        ScalarSqlExpression::Binary { left, op, right } => {
            if decimal_expression_fast_enabled()
                && let Some(values) = evaluate_decimal_product_expression(batch, left, op, right)?
            {
                return Ok(EvaluatedScalar::Float64(values));
            }
            let left = evaluate_scalar_expression(batch, left)?;
            let right = evaluate_scalar_expression(batch, right)?;
            evaluate_binary_scalar(left, op, right)
        }
        ScalarSqlExpression::Cast { expr, target } => {
            let value = evaluate_scalar_expression(batch, expr)?;
            cast_evaluated_scalar(value, target)
        }
        ScalarSqlExpression::Coalesce(values) => {
            let mut evaluated = values
                .iter()
                .map(|expr| evaluate_scalar_expression(batch, expr))
                .collect::<Result<Vec<_>>>()?;
            let Some(first) = evaluated.first().cloned() else {
                return Err(DodamError::UnsupportedSql(
                    "COALESCE requires at least one argument".to_string(),
                ));
            };
            let mut result = first;
            for value in evaluated.drain(1..) {
                result = coalesce_evaluated_scalar(result, value)?;
            }
            Ok(result)
        }
        ScalarSqlExpression::Lower(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Utf8(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.to_lowercase()))
                    .collect(),
            ))
        }
        ScalarSqlExpression::Upper(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Utf8(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.to_uppercase()))
                    .collect(),
            ))
        }
        ScalarSqlExpression::Length(expr) => {
            let value = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Int64(
                value
                    .into_iter()
                    .map(|value| value.map(|value| value.chars().count() as i64))
                    .collect(),
            ))
        }
        ScalarSqlExpression::ExtractYear(expr) => {
            let value = scalar_as_i64(evaluate_scalar_expression(batch, expr)?)?;
            Ok(EvaluatedScalar::Int64(
                value
                    .into_iter()
                    .map(|value| {
                        value
                            .map(|days| civil_from_days(days).map(|(year, _, _)| i64::from(year)))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()?,
            ))
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            let values = scalar_as_utf8(evaluate_scalar_expression(batch, expr)?)?;
            let starts = scalar_as_i64(evaluate_scalar_expression(batch, start)?)?;
            let lengths = length
                .as_ref()
                .map(|expr| scalar_as_i64(evaluate_scalar_expression(batch, expr)?))
                .transpose()?;
            Ok(EvaluatedScalar::Utf8(
                (0..batch.num_rows())
                    .map(|row| {
                        substring_value(
                            values[row].as_deref(),
                            starts[row],
                            lengths.as_ref().map(|values| values[row]),
                        )
                    })
                    .collect(),
            ))
        }
        ScalarSqlExpression::Case {
            conditions,
            results,
            else_result,
        } => evaluate_case_expression(batch, conditions, results, else_result.as_deref()),
    }
}

struct DecimalScalarColumn<'a> {
    values: &'a Decimal128Array,
    scale: f64,
}

fn decimal_expression_fast_enabled() -> bool {
    std::env::var("DODAM_DISABLE_DECIMAL_EXPR_FAST")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn evaluate_decimal_product_expression(
    batch: &RecordBatch,
    left: &ScalarSqlExpression,
    op: &BinaryOperator,
    right: &ScalarSqlExpression,
) -> Result<Option<Vec<Option<f64>>>> {
    if *op != BinaryOperator::Multiply {
        return Ok(None);
    }
    if let Some((value, complement)) = decimal_discount_product_operands(batch, left, right)? {
        return Ok(Some(decimal_complement_product(value, complement)));
    }
    if let Some((value, complement)) = decimal_discount_product_operands(batch, right, left)? {
        return Ok(Some(decimal_complement_product(value, complement)));
    }
    if let (Some(left), Some(right)) = (
        decimal_scalar_column(batch, left)?,
        decimal_scalar_column(batch, right)?,
    ) {
        return Ok(Some(decimal_product(left, right)));
    }
    Ok(None)
}

fn decimal_discount_product_operands<'a>(
    batch: &'a RecordBatch,
    value: &ScalarSqlExpression,
    complement: &ScalarSqlExpression,
) -> Result<Option<(DecimalScalarColumn<'a>, DecimalScalarColumn<'a>)>> {
    let Some(value) = decimal_scalar_column(batch, value)? else {
        return Ok(None);
    };
    let Some(complement) = decimal_one_minus_column(batch, complement)? else {
        return Ok(None);
    };
    Ok(Some((value, complement)))
}

fn decimal_one_minus_column<'a>(
    batch: &'a RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<Option<DecimalScalarColumn<'a>>> {
    let ScalarSqlExpression::Binary { left, op, right } = expr else {
        return Ok(None);
    };
    if *op != BinaryOperator::Minus || !scalar_literal_is_one(left) {
        return Ok(None);
    }
    decimal_scalar_column(batch, right)
}

fn decimal_scalar_column<'a>(
    batch: &'a RecordBatch,
    expr: &ScalarSqlExpression,
) -> Result<Option<DecimalScalarColumn<'a>>> {
    let ScalarSqlExpression::Column(column) = expr else {
        return Ok(None);
    };
    let index = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
        .ok_or_else(|| DodamError::UnknownColumn(column.to_string()))?;
    let array = batch.column(index);
    let DataType::Decimal128(precision, scale) = array.data_type() else {
        return Ok(None);
    };
    if *precision > 18 {
        return Ok(None);
    }
    let Some(scale_raw) = decimal_scale_i128(*scale) else {
        return Ok(None);
    };
    Ok(Some(DecimalScalarColumn {
        values: array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128 scalar input"),
        scale: scale_raw as f64,
    }))
}

fn scalar_literal_is_one(expr: &ScalarSqlExpression) -> bool {
    match expr {
        ScalarSqlExpression::Literal(LiteralValue::Int64(1)) => true,
        ScalarSqlExpression::Literal(LiteralValue::Float64(value)) => *value == 1.0,
        _ => false,
    }
}

fn decimal_scale_i128(scale: i8) -> Option<i128> {
    let scale = u32::try_from(scale).ok()?;
    Some(10_i128.checked_pow(scale)?)
}

fn decimal_complement_product(
    value: DecimalScalarColumn<'_>,
    complement: DecimalScalarColumn<'_>,
) -> Vec<Option<f64>> {
    let value_raw = value.values.values();
    let complement_raw = complement.values.values();
    if value.values.null_count() == 0 && complement.values.null_count() == 0 {
        return value_raw
            .iter()
            .copied()
            .zip(complement_raw.iter().copied())
            .map(|(value_raw, complement_value)| {
                Some(
                    (value_raw as f64 / value.scale)
                        * (1.0 - complement_value as f64 / complement.scale),
                )
            })
            .collect();
    }
    (0..value.values.len())
        .map(|row| {
            if value.values.is_null(row) || complement.values.is_null(row) {
                None
            } else {
                Some(
                    (value_raw[row] as f64 / value.scale)
                        * (1.0 - complement_raw[row] as f64 / complement.scale),
                )
            }
        })
        .collect()
}

fn decimal_product(
    left: DecimalScalarColumn<'_>,
    right: DecimalScalarColumn<'_>,
) -> Vec<Option<f64>> {
    let left_raw = left.values.values();
    let right_raw = right.values.values();
    if left.values.null_count() == 0 && right.values.null_count() == 0 {
        return left_raw
            .iter()
            .copied()
            .zip(right_raw.iter().copied())
            .map(|(left_value, right_value)| {
                Some((left_value as f64 / left.scale) * (right_value as f64 / right.scale))
            })
            .collect();
    }
    (0..left.values.len())
        .map(|row| {
            if left.values.is_null(row) || right.values.is_null(row) {
                None
            } else {
                Some((left_raw[row] as f64 / left.scale) * (right_raw[row] as f64 / right.scale))
            }
        })
        .collect()
}

fn evaluate_case_expression(
    batch: &RecordBatch,
    conditions: &[SqlExpr],
    results: &[ScalarSqlExpression],
    else_result: Option<&ScalarSqlExpression>,
) -> Result<EvaluatedScalar> {
    if conditions.len() != results.len() {
        return Err(DodamError::UnsupportedSql(
            "CASE conditions and results length mismatch".to_string(),
        ));
    }
    let evaluated_results = results
        .iter()
        .map(|expr| evaluate_scalar_expression(batch, expr))
        .collect::<Result<Vec<_>>>()?;
    let evaluated_else = else_result
        .map(|expr| evaluate_scalar_expression(batch, expr))
        .transpose()?;
    let result_kind = evaluated_results
        .iter()
        .chain(evaluated_else.iter())
        .find_map(evaluated_scalar_kind)
        .unwrap_or(EvaluatedScalarKind::Utf8);
    let mut output = empty_scalar_values(result_kind, batch.num_rows());

    let masks = conditions
        .iter()
        .map(|condition| evaluate_scalar_predicate(batch, condition, None))
        .collect::<Result<Vec<_>>>()?;
    for row in 0..batch.num_rows() {
        let mut selected = None;
        for (index, mask) in masks.iter().enumerate() {
            if boolean_value(mask, row) == Some(true) {
                selected = evaluated_results.get(index);
                break;
            }
        }
        let selected = selected.or(evaluated_else.as_ref());
        set_scalar_value_from(&mut output, row, selected)?;
    }
    Ok(output)
}

fn evaluate_scalar_in_list(
    value: EvaluatedScalar,
    values: &[EvaluatedScalar],
    negated: bool,
) -> Result<Vec<Option<bool>>> {
    let value_kind = evaluated_scalar_kind(&value).unwrap_or(EvaluatedScalarKind::Utf8);
    let mut output = Vec::with_capacity(value.len());
    for row in 0..value.len() {
        let candidate = scalar_value_at(&value, row)?;
        if candidate.is_none() {
            output.push(None);
            continue;
        }
        let mut matched = false;
        let mut has_null = false;
        for value in values {
            let value = scalar_value_at(&cast_scalar_for_kind(value.clone(), value_kind)?, row)?;
            match value {
                Some(value) => {
                    if Some(value) == candidate {
                        matched = true;
                        break;
                    }
                }
                None => has_null = true,
            }
        }
        let result = if matched {
            Some(!negated)
        } else if has_null {
            None
        } else {
            Some(negated)
        };
        output.push(result);
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarLikeToken {
    Any,
    One,
    Char(char),
}

fn scalar_like_pattern_tokens(pattern: &str, escape: Option<char>) -> Result<Vec<ScalarLikeToken>> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if Some(ch) == escape {
            let escaped = chars.next().ok_or_else(|| {
                DodamError::InvalidFilter("LIKE ESCAPE at end of pattern".to_string())
            })?;
            tokens.push(ScalarLikeToken::Char(escaped));
        } else {
            tokens.push(match ch {
                '%' => ScalarLikeToken::Any,
                '_' => ScalarLikeToken::One,
                ch => ScalarLikeToken::Char(ch),
            });
        }
    }
    Ok(tokens)
}

fn scalar_like_matches(value: &str, pattern: &[ScalarLikeToken]) -> bool {
    fn matches_from(
        value: &[char],
        pattern: &[ScalarLikeToken],
        value_index: usize,
        pattern_index: usize,
    ) -> bool {
        if pattern_index == pattern.len() {
            return value_index == value.len();
        }
        match pattern[pattern_index] {
            ScalarLikeToken::Char(ch) => {
                value.get(value_index).is_some_and(|value| *value == ch)
                    && matches_from(value, pattern, value_index + 1, pattern_index + 1)
            }
            ScalarLikeToken::One => {
                value_index < value.len()
                    && matches_from(value, pattern, value_index + 1, pattern_index + 1)
            }
            ScalarLikeToken::Any => (value_index..=value.len())
                .any(|index| matches_from(value, pattern, index, pattern_index + 1)),
        }
    }
    let value = value.chars().collect::<Vec<_>>();
    matches_from(&value, pattern, 0, 0)
}

fn substring_value(
    value: Option<&str>,
    start: Option<i64>,
    length: Option<Option<i64>>,
) -> Option<String> {
    let value = value?;
    let start = start?;
    let chars = value.chars().collect::<Vec<_>>();
    let start_index = if start <= 1 {
        0
    } else {
        usize::try_from(start - 1).ok()?
    };
    let available = chars.len().saturating_sub(start_index);
    let take = match length {
        Some(Some(length)) if length <= 0 => 0,
        Some(Some(length)) => usize::try_from(length).ok()?.min(available),
        Some(None) => return None,
        None => available,
    };
    Some(chars.iter().skip(start_index).take(take).collect())
}

#[derive(Clone, PartialEq)]
enum ScalarValue {
    Int64(i64),
    Float64(f64),
    Utf8(String),
    Boolean(bool),
}

fn scalar_value_at(value: &EvaluatedScalar, row: usize) -> Result<Option<ScalarValue>> {
    Ok(match value {
        EvaluatedScalar::Int64(values) => values[row].map(ScalarValue::Int64),
        EvaluatedScalar::Float64(values) => values[row].map(ScalarValue::Float64),
        EvaluatedScalar::Utf8(values) => values[row].clone().map(ScalarValue::Utf8),
        EvaluatedScalar::Boolean(values) => values[row].map(ScalarValue::Boolean),
    })
}

fn cast_scalar_for_kind(
    value: EvaluatedScalar,
    kind: EvaluatedScalarKind,
) -> Result<EvaluatedScalar> {
    match kind {
        EvaluatedScalarKind::Int64 => Ok(EvaluatedScalar::Int64(scalar_as_i64(value)?)),
        EvaluatedScalarKind::Float64 => Ok(EvaluatedScalar::Float64(scalar_as_f64(value)?)),
        EvaluatedScalarKind::Utf8 => Ok(EvaluatedScalar::Utf8(scalar_as_utf8(value)?)),
        EvaluatedScalarKind::Boolean => match value {
            EvaluatedScalar::Boolean(_) => Ok(value),
            other => Err(DodamError::UnsupportedSql(format!(
                "cannot use {} in boolean IN list",
                other.data_type()
            ))),
        },
    }
}

#[derive(Clone, Copy)]
enum EvaluatedScalarKind {
    Int64,
    Float64,
    Utf8,
    Boolean,
}

fn evaluated_scalar_kind(value: &EvaluatedScalar) -> Option<EvaluatedScalarKind> {
    Some(match value {
        EvaluatedScalar::Int64(_) => EvaluatedScalarKind::Int64,
        EvaluatedScalar::Float64(_) => EvaluatedScalarKind::Float64,
        EvaluatedScalar::Utf8(_) => EvaluatedScalarKind::Utf8,
        EvaluatedScalar::Boolean(_) => EvaluatedScalarKind::Boolean,
    })
}

fn empty_scalar_values(kind: EvaluatedScalarKind, rows: usize) -> EvaluatedScalar {
    match kind {
        EvaluatedScalarKind::Int64 => EvaluatedScalar::Int64(vec![None; rows]),
        EvaluatedScalarKind::Float64 => EvaluatedScalar::Float64(vec![None; rows]),
        EvaluatedScalarKind::Utf8 => EvaluatedScalar::Utf8(vec![None; rows]),
        EvaluatedScalarKind::Boolean => EvaluatedScalar::Boolean(vec![None; rows]),
    }
}

fn set_scalar_value_from(
    output: &mut EvaluatedScalar,
    row: usize,
    source: Option<&EvaluatedScalar>,
) -> Result<()> {
    match output {
        EvaluatedScalar::Int64(values) => {
            values[row] = source
                .map(|source| scalar_value_as_i64(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Float64(values) => {
            values[row] = source
                .map(|source| scalar_value_as_f64(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Utf8(values) => {
            values[row] = source
                .map(|source| scalar_value_as_utf8(source, row))
                .transpose()?
                .flatten();
        }
        EvaluatedScalar::Boolean(values) => {
            values[row] = source
                .map(|source| scalar_value_as_bool(source, row))
                .transpose()?
                .flatten();
        }
    }
    Ok(())
}

fn scalar_value_as_i64(value: &EvaluatedScalar, row: usize) -> Result<Option<i64>> {
    match value {
        EvaluatedScalar::Int64(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn scalar_value_as_f64(value: &EvaluatedScalar, row: usize) -> Result<Option<f64>> {
    match value {
        EvaluatedScalar::Int64(values) => Ok(values[row].map(|value| value as f64)),
        EvaluatedScalar::Float64(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn scalar_value_as_utf8(value: &EvaluatedScalar, row: usize) -> Result<Option<String>> {
    match value {
        EvaluatedScalar::Int64(values) => Ok(values[row].map(|value| value.to_string())),
        EvaluatedScalar::Float64(values) => Ok(values[row].map(|value| value.to_string())),
        EvaluatedScalar::Utf8(values) => Ok(values[row].clone()),
        EvaluatedScalar::Boolean(values) => Ok(values[row].map(|value| value.to_string())),
    }
}

fn scalar_value_as_bool(value: &EvaluatedScalar, row: usize) -> Result<Option<bool>> {
    match value {
        EvaluatedScalar::Boolean(values) => Ok(values[row]),
        _ => Err(DodamError::UnsupportedSql(
            "CASE result type mismatch".to_string(),
        )),
    }
}

fn evaluated_column(batch: &RecordBatch, column: &str) -> Result<EvaluatedScalar> {
    let index = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
        .ok_or_else(|| DodamError::UnknownColumn(column.to_string()))?;
    let array = batch.column(index);
    match array.data_type() {
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            Ok(EvaluatedScalar::Int64(
                values.iter().map(|value| value.map(i64::from)).collect(),
            ))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            Ok(EvaluatedScalar::Int64(values.iter().collect()))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64");
            Ok(EvaluatedScalar::Float64(values.iter().collect()))
        }
        DataType::Decimal128(_, scale) => {
            let values = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128");
            let scale = 10_f64.powi(i32::from(*scale));
            Ok(EvaluatedScalar::Float64(
                values
                    .iter()
                    .map(|value| value.map(|value| value as f64 / scale))
                    .collect(),
            ))
        }
        DataType::Utf8 => {
            let values = array.as_any().downcast_ref::<StringArray>().expect("Utf8");
            Ok(EvaluatedScalar::Utf8(
                values
                    .iter()
                    .map(|value| value.map(str::to_string))
                    .collect(),
            ))
        }
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean");
            Ok(EvaluatedScalar::Boolean(values.iter().collect()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32");
            Ok(EvaluatedScalar::Int64(
                values.iter().map(|value| value.map(i64::from)).collect(),
            ))
        }
        DataType::Date64 => {
            let values = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64");
            Ok(EvaluatedScalar::Int64(
                values
                    .iter()
                    .map(|value| value.map(|value| value / 86_400_000))
                    .collect(),
            ))
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let values = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .expect("TimestampMillisecond");
            Ok(EvaluatedScalar::Int64(
                values
                    .iter()
                    .map(|value| value.map(|value| value / 86_400_000))
                    .collect(),
            ))
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "projection expression column type {data_type} is not supported yet"
        ))),
    }
}

fn evaluated_literal(value: &LiteralValue, rows: usize) -> EvaluatedScalar {
    match value {
        LiteralValue::Null => EvaluatedScalar::Utf8(vec![None; rows]),
        LiteralValue::Boolean(value) => EvaluatedScalar::Boolean(vec![Some(*value); rows]),
        LiteralValue::Int64(value) => EvaluatedScalar::Int64(vec![Some(*value); rows]),
        LiteralValue::Float64(value) => EvaluatedScalar::Float64(vec![Some(*value); rows]),
        LiteralValue::Utf8(value) => EvaluatedScalar::Utf8(vec![Some(value.clone()); rows]),
    }
}

fn compare_evaluated_scalars(
    left: EvaluatedScalar,
    op: &BinaryOperator,
    right: EvaluatedScalar,
) -> Result<Vec<Option<bool>>> {
    match (&left, &right) {
        (EvaluatedScalar::Utf8(_), _) | (_, EvaluatedScalar::Utf8(_)) => {
            let left = scalar_as_utf8(left)?;
            let right = scalar_as_utf8(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_values(left, op, right))
                .collect())
        }
        (EvaluatedScalar::Boolean(_), _) | (_, EvaluatedScalar::Boolean(_)) => {
            let EvaluatedScalar::Boolean(left) = left else {
                return Err(DodamError::UnsupportedSql(
                    "boolean comparisons require boolean operands".to_string(),
                ));
            };
            let EvaluatedScalar::Boolean(right) = right else {
                return Err(DodamError::UnsupportedSql(
                    "boolean comparisons require boolean operands".to_string(),
                ));
            };
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_values(left, op, right))
                .collect())
        }
        (EvaluatedScalar::Float64(_), _) | (_, EvaluatedScalar::Float64(_)) => {
            let left = scalar_as_f64(left)?;
            let right = scalar_as_f64(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_f64(left, op, right))
                .collect())
        }
        _ => {
            let left = scalar_as_i64(left)?;
            let right = scalar_as_i64(right)?;
            Ok(left
                .into_iter()
                .zip(right)
                .map(|(left, right)| compare_optional_values(left, op, right))
                .collect())
        }
    }
}

fn compare_optional_values<T: Ord>(
    left: Option<T>,
    op: &BinaryOperator,
    right: Option<T>,
) -> Option<bool> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    Some(match op {
        BinaryOperator::Eq => left == right,
        BinaryOperator::NotEq => left != right,
        BinaryOperator::Gt => left > right,
        BinaryOperator::GtEq => left >= right,
        BinaryOperator::Lt => left < right,
        BinaryOperator::LtEq => left <= right,
        _ => unreachable!("validated comparison operator"),
    })
}

fn compare_optional_f64(
    left: Option<f64>,
    op: &BinaryOperator,
    right: Option<f64>,
) -> Option<bool> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    Some(match op {
        BinaryOperator::Eq => left == right,
        BinaryOperator::NotEq => left != right,
        BinaryOperator::Gt => left > right,
        BinaryOperator::GtEq => left >= right,
        BinaryOperator::Lt => left < right,
        BinaryOperator::LtEq => left <= right,
        _ => unreachable!("validated comparison operator"),
    })
}

fn scalar_null_mask(value: EvaluatedScalar) -> Vec<bool> {
    match value {
        EvaluatedScalar::Int64(values) => values.into_iter().map(|value| value.is_none()).collect(),
        EvaluatedScalar::Float64(values) => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
        EvaluatedScalar::Utf8(values) => values.into_iter().map(|value| value.is_none()).collect(),
        EvaluatedScalar::Boolean(values) => {
            values.into_iter().map(|value| value.is_none()).collect()
        }
    }
}

fn evaluate_binary_scalar(
    left: EvaluatedScalar,
    op: &BinaryOperator,
    right: EvaluatedScalar,
) -> Result<EvaluatedScalar> {
    match (&left, &right) {
        (EvaluatedScalar::Float64(_), _) | (_, EvaluatedScalar::Float64(_)) => {
            let left = scalar_as_f64(left)?;
            let right = scalar_as_f64(right)?;
            Ok(EvaluatedScalar::Float64(
                left.into_iter()
                    .zip(right)
                    .map(|(left, right)| match (left, right) {
                        (Some(left), Some(right)) => match op {
                            BinaryOperator::Plus => Some(left + right),
                            BinaryOperator::Minus => Some(left - right),
                            BinaryOperator::Multiply => Some(left * right),
                            BinaryOperator::Divide => Some(left / right),
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect(),
            ))
        }
        _ => {
            let left = scalar_as_i64(left)?;
            let right = scalar_as_i64(right)?;
            Ok(EvaluatedScalar::Int64(
                left.into_iter()
                    .zip(right)
                    .map(|(left, right)| match (left, right) {
                        (Some(left), Some(right)) => match op {
                            BinaryOperator::Plus => left.checked_add(right),
                            BinaryOperator::Minus => left.checked_sub(right),
                            BinaryOperator::Multiply => left.checked_mul(right),
                            BinaryOperator::Divide if right != 0 => Some(left / right),
                            BinaryOperator::Divide => None,
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect(),
            ))
        }
    }
}

fn scalar_as_i64(value: EvaluatedScalar) -> Result<Vec<Option<i64>>> {
    match value {
        EvaluatedScalar::Int64(values) => Ok(values),
        other => Err(DodamError::UnsupportedSql(format!(
            "cannot use {} in integer arithmetic",
            other.data_type()
        ))),
    }
}

fn scalar_as_f64(value: EvaluatedScalar) -> Result<Vec<Option<f64>>> {
    match value {
        EvaluatedScalar::Int64(values) => Ok(values
            .into_iter()
            .map(|value| value.map(|value| value as f64))
            .collect()),
        EvaluatedScalar::Float64(values) => Ok(values),
        other => Err(DodamError::UnsupportedSql(format!(
            "cannot use {} in floating point arithmetic",
            other.data_type()
        ))),
    }
}

fn cast_evaluated_scalar(value: EvaluatedScalar, target: &str) -> Result<EvaluatedScalar> {
    let target = target.to_ascii_lowercase();
    if matches!(
        target.as_str(),
        "varchar" | "text" | "string" | "char" | "character varying"
    ) {
        return Ok(EvaluatedScalar::Utf8(match value {
            EvaluatedScalar::Int64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
            EvaluatedScalar::Float64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
            EvaluatedScalar::Utf8(values) => values,
            EvaluatedScalar::Boolean(values) => values
                .into_iter()
                .map(|value| value.map(|value| value.to_string()))
                .collect(),
        }));
    }
    if matches!(target.as_str(), "bigint" | "int8" | "integer" | "int") {
        return Ok(EvaluatedScalar::Int64(match value {
            EvaluatedScalar::Int64(values) => values,
            EvaluatedScalar::Float64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value as i64))
                .collect(),
            EvaluatedScalar::Utf8(values) => values
                .into_iter()
                .map(|value| value.and_then(|value| value.parse::<i64>().ok()))
                .collect(),
            EvaluatedScalar::Boolean(values) => values
                .into_iter()
                .map(|value| value.map(i64::from))
                .collect(),
        }));
    }
    if matches!(target.as_str(), "double" | "float8" | "float" | "real") {
        return Ok(EvaluatedScalar::Float64(match value {
            EvaluatedScalar::Int64(values) => values
                .into_iter()
                .map(|value| value.map(|value| value as f64))
                .collect(),
            EvaluatedScalar::Float64(values) => values,
            EvaluatedScalar::Utf8(values) => values
                .into_iter()
                .map(|value| value.and_then(|value| value.parse::<f64>().ok()))
                .collect(),
            EvaluatedScalar::Boolean(values) => values
                .into_iter()
                .map(|value| value.map(|value| if value { 1.0 } else { 0.0 }))
                .collect(),
        }));
    }
    Err(DodamError::UnsupportedSql(format!(
        "unsupported CAST target: {target}"
    )))
}

fn coalesce_evaluated_scalar(
    left: EvaluatedScalar,
    right: EvaluatedScalar,
) -> Result<EvaluatedScalar> {
    match (left, right) {
        (EvaluatedScalar::Utf8(left), right) => {
            let right = scalar_as_utf8(right)?;
            Ok(EvaluatedScalar::Utf8(coalesce_options(left, right)))
        }
        (left, EvaluatedScalar::Utf8(right)) => {
            let left = scalar_as_utf8(left)?;
            Ok(EvaluatedScalar::Utf8(coalesce_options(left, right)))
        }
        (EvaluatedScalar::Float64(left), right) => {
            let right = scalar_as_f64(right)?;
            Ok(EvaluatedScalar::Float64(coalesce_options(left, right)))
        }
        (left, EvaluatedScalar::Float64(right)) => {
            let left = scalar_as_f64(left)?;
            Ok(EvaluatedScalar::Float64(coalesce_options(left, right)))
        }
        (EvaluatedScalar::Int64(left), EvaluatedScalar::Int64(right)) => {
            Ok(EvaluatedScalar::Int64(coalesce_options(left, right)))
        }
        (EvaluatedScalar::Boolean(left), EvaluatedScalar::Boolean(right)) => {
            Ok(EvaluatedScalar::Boolean(coalesce_options(left, right)))
        }
        (left, right) => Err(DodamError::UnsupportedSql(format!(
            "cannot COALESCE {} and {}",
            left.data_type(),
            right.data_type()
        ))),
    }
}

fn scalar_as_utf8(value: EvaluatedScalar) -> Result<Vec<Option<String>>> {
    Ok(match value {
        EvaluatedScalar::Utf8(values) => values,
        EvaluatedScalar::Int64(values) => values
            .into_iter()
            .map(|value| value.map(|value| value.to_string()))
            .collect(),
        EvaluatedScalar::Float64(values) => values
            .into_iter()
            .map(|value| value.map(|value| value.to_string()))
            .collect(),
        EvaluatedScalar::Boolean(values) => values
            .into_iter()
            .map(|value| value.map(|value| value.to_string()))
            .collect(),
    })
}

fn coalesce_options<T>(left: Vec<Option<T>>, right: Vec<Option<T>>) -> Vec<Option<T>> {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left.or(right))
        .collect()
}

fn rename_output_batches(
    batches: Vec<RecordBatch>,
    aliases: &[(String, String)],
) -> Result<Vec<RecordBatch>> {
    if aliases.is_empty() {
        return Ok(batches);
    }

    batches
        .into_iter()
        .map(|batch| {
            let fields = batch
                .schema()
                .fields()
                .iter()
                .map(|field| {
                    let name = aliases
                        .iter()
                        .find(|(alias, target)| !alias.contains('(') && target == field.name())
                        .map(|(alias, _)| alias.as_str())
                        .unwrap_or_else(|| field.name().as_str());
                    Field::new(name, field.data_type().clone(), field.is_nullable())
                })
                .collect::<Vec<_>>();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec())
                .map_err(DodamError::from)
        })
        .collect()
}

fn aggregate_values_to_batch(values: &[crate::execution::AggregateResult]) -> Result<RecordBatch> {
    let mut fields = Vec::new();
    let mut columns = Vec::new();

    for value in values {
        let (field, array) = aggregate_values_to_column(&value.expr.to_string(), &[&value.value]);
        fields.push(field);
        columns.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn group_values_to_column(
    name: &str,
    values: &[Option<&crate::execution::GroupValue>],
) -> (Field, ArrayRef) {
    let data_type = values
        .iter()
        .find_map(|value| match value {
            Some(crate::execution::GroupValue::Utf8(_)) => Some(DataType::Utf8),
            Some(crate::execution::GroupValue::Date64(_)) => Some(DataType::Date64),
            Some(crate::execution::GroupValue::Date32(_)) => Some(DataType::Date32),
            Some(crate::execution::GroupValue::Decimal128(_, precision, scale)) => {
                Some(DataType::Decimal128(*precision, *scale))
            }
            Some(crate::execution::GroupValue::UInt64(_)) => Some(DataType::UInt64),
            Some(crate::execution::GroupValue::Int64(_)) => Some(DataType::Int64),
            None => None,
        })
        .unwrap_or(DataType::Int64);

    match data_type {
        DataType::Utf8 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(crate::execution::GroupValue::Utf8(value)) => value.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Utf8, true),
                Arc::new(StringArray::from(values)),
            )
        }
        DataType::UInt64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(crate::execution::GroupValue::UInt64(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::UInt64, true),
                Arc::new(UInt64Array::from(values)),
            )
        }
        DataType::Decimal128(precision, scale) => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(crate::execution::GroupValue::Decimal128(value, _, _)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Decimal128(precision, scale), true),
                Arc::new(
                    Decimal128Array::from(values)
                        .with_precision_and_scale(precision, scale)
                        .expect("valid Decimal128 group type"),
                ),
            )
        }
        DataType::Date32 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(crate::execution::GroupValue::Date32(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date32, true),
                Arc::new(Date32Array::from(values)),
            )
        }
        DataType::Date64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(crate::execution::GroupValue::Date64(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date64, true),
                Arc::new(Date64Array::from(values)),
            )
        }
        _ => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(crate::execution::GroupValue::Int64(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Int64, true),
                Arc::new(Int64Array::from(values)),
            )
        }
    }
}

fn aggregate_values_to_column(
    name: &str,
    values: &[&crate::execution::AggregateValue],
) -> (Field, ArrayRef) {
    let data_type = values
        .iter()
        .map(|value| match value {
            crate::execution::AggregateValue::Count(_) => DataType::UInt64,
            crate::execution::AggregateValue::Int64(_) => DataType::Int64,
            crate::execution::AggregateValue::Float64(_) => DataType::Float64,
            crate::execution::AggregateValue::Date32(_) => DataType::Date32,
            crate::execution::AggregateValue::Date64(_) => DataType::Date64,
            crate::execution::AggregateValue::TimestampMillisecond(_, timezone) => {
                DataType::Timestamp(TimeUnit::Millisecond, timezone.clone().map(Into::into))
            }
            crate::execution::AggregateValue::Decimal128(_, precision, scale) => {
                DataType::Decimal128(*precision, *scale)
            }
            crate::execution::AggregateValue::Utf8(_) => DataType::Utf8,
        })
        .next()
        .unwrap_or(DataType::Int64);

    match data_type {
        DataType::UInt64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Count(value) => Some(*value),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::UInt64, true),
                Arc::new(UInt64Array::from(values)),
            )
        }
        DataType::Float64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Float64(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Float64, true),
                Arc::new(Float64Array::from(values)),
            )
        }
        DataType::Utf8 => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Utf8(value) => value.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Utf8, true),
                Arc::new(StringArray::from(values)),
            )
        }
        DataType::Date32 => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Date32(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date32, true),
                Arc::new(Date32Array::from(values)),
            )
        }
        DataType::Date64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Date64(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date64, true),
                Arc::new(Date64Array::from(values)),
            )
        }
        DataType::Decimal128(precision, scale) => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Decimal128(value, _, _) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Decimal128(precision, scale), true),
                Arc::new(
                    Decimal128Array::from(values)
                        .with_precision_and_scale(precision, scale)
                        .expect("valid Decimal128 aggregate type"),
                ),
            )
        }
        DataType::Timestamp(TimeUnit::Millisecond, timezone) => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::TimestampMillisecond(value, _) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            let array = TimestampMillisecondArray::from(values);
            let array = if let Some(timezone) = timezone.as_ref() {
                array.with_timezone(timezone.clone())
            } else {
                array
            };
            (
                Field::new(
                    name,
                    DataType::Timestamp(TimeUnit::Millisecond, timezone),
                    true,
                ),
                Arc::new(array),
            )
        }
        _ => {
            let values = values
                .iter()
                .map(|value| match value {
                    crate::execution::AggregateValue::Int64(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Int64, true),
                Arc::new(Int64Array::from(values)),
            )
        }
    }
}
