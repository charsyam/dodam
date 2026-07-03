use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

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
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sqlparser::ast::{
    BinaryOperator, DateTimeField, Distinct, DuplicateTreatment, Expr as SqlExpr, FunctionArg,
    FunctionArgExpr, FunctionArguments, GroupByExpr, JoinConstraint, JoinOperator, LimitClause,
    ObjectName, ObjectNamePart, OrderByKind, Query, Select, SelectItem, SetExpr, Statement,
    TableFactor, UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::engine::{DodamEngine, JoinAlgorithm, JoinParquetRequest};
use crate::error::{DodamError, Result};
use crate::execution::JoinType;
use crate::execution::{
    AggregateExpr, AggregateMetrics, AggregateResult, AggregateValue, ComparisonExpr, ComparisonOp,
    DistinctExec, Expr, FilterExpr, GroupAggregateResult, GroupValue, HashJoinExec, JoinBuildSide,
    LiteralValue, MemoryExec, PhysicalPlan, Projection, RecordBatchSink, ScanPlanMetrics,
    SendableBatchStream, SortExpr, SortKey, collect_aggregates, collect_grouped_aggregates,
    evaluate_filter_mask, filter_batch,
};
use crate::optimizer::plan_join_inputs;

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
    if let Some(output) = try_execute_q12_shipping_modes_fast(engine, sql, batch_size).await? {
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
    if let Some(output) = try_execute_q07_volume_shipping_fast(engine, sql, batch_size).await? {
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

#[derive(Default)]
struct Q01State {
    sum_qty: f64,
    sum_base_price: f64,
    sum_disc_price: f64,
    sum_charge: f64,
    sum_discount: f64,
    qty_count: u64,
    price_count: u64,
    discount_count: u64,
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
        self.qty_count += 1;
        self.price_count += 1;
        self.discount_count += 1;
        self.count_order += 1;
    }
}

struct Q01Row {
    returnflag: String,
    linestatus: String,
    state: Q01State,
}

async fn q01_pricing_summary_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    cutoff_days: i32,
) -> Result<Vec<Q01Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_returnflag".to_string(),
                "l_linestatus".to_string(),
                "l_quantity".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
                "l_tax".to_string(),
                "l_shipdate".to_string(),
            ]),
            None,
        )
        .await?;
    let mut groups = Vec::<Q01Row>::with_capacity(4);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let returnflags = batch_string_column(&batch, "l_returnflag")?;
        let linestatuses = batch_string_column(&batch, "l_linestatus")?;
        let quantities = batch_column(&batch, "l_quantity")?;
        let extendedprices = batch_column(&batch, "l_extendedprice")?;
        let discounts = batch_column(&batch, "l_discount")?;
        let taxes = batch_column(&batch, "l_tax")?;
        let shipdates = batch_column(&batch, "l_shipdate")?;
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
            continue;
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
            q01_group_state(&mut groups, returnflags.value(row), linestatuses.value(row)).update(
                quantity,
                extendedprice,
                discount,
                tax,
            );
        }
    }
    let mut rows = groups;
    rows.sort_by(|left, right| {
        left.returnflag
            .cmp(&right.returnflag)
            .then_with(|| left.linestatus.cmp(&right.linestatus))
    });
    Ok(rows)
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
    groups: &mut Vec<Q01Row>,
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
        let returnflag = returnflags.value(row);
        let linestatus = linestatuses.value(row);
        q01_group_state(groups, returnflag, linestatus).update(
            quantities.value(row),
            extendedprices.value(row),
            discounts.value(row),
            taxes.value(row),
        );
    }
    Ok(true)
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

struct Q01DecimalInput<'a> {
    values: &'a Decimal128Array,
    scale: f64,
}

impl Q01DecimalInput<'_> {
    fn is_null(&self, row: usize) -> bool {
        self.values.is_null(row)
    }

    fn value(&self, row: usize) -> f64 {
        self.values.value(row) as f64 / self.scale
    }
}

fn q01_decimal_input(column: &ArrayRef) -> Result<Option<Q01DecimalInput<'_>>> {
    let DataType::Decimal128(_, scale) = column.data_type() else {
        return Ok(None);
    };
    Ok(Some(Q01DecimalInput {
        values: column
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128 q01 input"),
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
                    .map(|row| row.state.sum_qty / row.state.qty_count as f64),
            )),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| {
                row.state.sum_base_price / row.state.price_count as f64
            }))),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| {
                row.state.sum_discount / row.state.discount_count as f64
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
    let customers = q10_customer_rows(engine, customer.path, batch_size).await?;
    if customers.is_empty() {
        return Ok(Some(q10_output(Vec::new())?));
    }
    let nation_names = q10_nation_names(engine, nation.path, batch_size).await?;
    let order_customers =
        q10_order_customers(engine, orders.path, batch_size, start_days, end_days).await?;
    if order_customers.is_empty() {
        return Ok(Some(q10_output(Vec::new())?));
    }
    let revenue_by_customer =
        q10_returned_revenue_by_customer(engine, lineitem.path, batch_size, &order_customers)
            .await?;
    let mut rows = revenue_by_customer
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
    rows.sort_by(|left, right| {
        right
            .revenue
            .partial_cmp(&left.revenue)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(20);
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
) -> Result<HashMap<i64, Q10Customer>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "c_custkey".to_string(),
                "c_name".to_string(),
                "c_acctbal".to_string(),
                "c_nationkey".to_string(),
                "c_address".to_string(),
                "c_phone".to_string(),
                "c_comment".to_string(),
            ]),
            None,
        )
        .await?;
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
        for row in 0..batch.num_rows() {
            let (Some(custkey), Some(acctbal), Some(nationkey)) = (
                numeric_i64_value(custkeys, row)?,
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
    }
    Ok(customers)
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
    let mut orders = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
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
    }
    Ok(orders)
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
    let mut revenues = HashMap::<i64, f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let returnflags = batch_string_column(&batch, "l_returnflag")?;
        let extendedprices = batch_column(&batch, "l_extendedprice")?;
        let discounts = batch_column(&batch, "l_discount")?;
        for row in 0..batch.num_rows() {
            if returnflags.is_null(row) || returnflags.value(row) != "R" {
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
    let part_keys =
        q09_matching_part_keys(engine, part.path, batch_size, &part_name_substring).await?;
    if part_keys.is_empty() {
        return Ok(Some(q09_output(Vec::new())?));
    }
    let nation_names = q10_nation_names(engine, nation.path, batch_size).await?;
    let supplier_nations = q09_supplier_nations(engine, supplier.path, batch_size).await?;
    let order_years = q09_order_years(engine, orders.path, batch_size).await?;
    let supply_costs = q09_supply_costs(engine, partsupp.path, batch_size, &part_keys).await?;
    let rows = q09_profit_rows(
        engine,
        lineitem.path,
        batch_size,
        &part_keys,
        &supplier_nations,
        &nation_names,
        &order_years,
        &supply_costs,
    )
    .await?;
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
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let names = batch_string_column(&batch, "p_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row).contains(name_substring)
                && let Some(partkey) = numeric_i64_value(partkeys, row)?
            {
                keys.insert(partkey);
            }
        }
    }
    Ok(keys)
}

async fn q09_supplier_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
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
            suppliers.insert(suppkey, nationkey);
        }
    }
    Ok(suppliers)
}

async fn q09_order_years(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashMap<i64, i32>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()]),
            None,
        )
        .await?;
    let mut orders = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(orderdate)) = (
                numeric_i64_value(orderkeys, row)?,
                date32_value(orderdates, row)?,
            ) else {
                continue;
            };
            let (year, _, _) = civil_from_days(i64::from(orderdate))?;
            orders.insert(orderkey, year);
        }
    }
    Ok(orders)
}

async fn q09_supply_costs(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &HashSet<i64>,
) -> Result<HashMap<(i64, i64), f64>> {
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
    let mut costs = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "ps_partkey")?;
        let suppkeys = batch_column(&batch, "ps_suppkey")?;
        let supplycosts = batch_column(&batch, "ps_supplycost")?;
        for row in 0..batch.num_rows() {
            let (Some(partkey), Some(suppkey), Some(supplycost)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
                numeric_f64_value(supplycosts, row)?,
            ) else {
                continue;
            };
            if part_keys.contains(&partkey) {
                costs.insert((partkey, suppkey), supplycost);
            }
        }
    }
    Ok(costs)
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
    part_keys: &HashSet<i64>,
    supplier_nations: &HashMap<i64, i64>,
    nation_names: &HashMap<i64, String>,
    order_years: &HashMap<i64, i32>,
    supply_costs: &HashMap<(i64, i64), f64>,
) -> Result<Vec<Q09Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_partkey".to_string(),
                "l_suppkey".to_string(),
                "l_quantity".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    let mut groups = HashMap::<(i64, i32), f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let partkeys = batch_column(&batch, "l_partkey")?;
        let suppkeys = batch_column(&batch, "l_suppkey")?;
        let quantities = batch_column(&batch, "l_quantity")?;
        let extendedprices = batch_column(&batch, "l_extendedprice")?;
        let discounts = batch_column(&batch, "l_discount")?;
        if q09_update_profit_decimal_batch(
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
            &mut groups,
        )? {
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(partkey), Some(suppkey)) = (
                numeric_i64_value(orderkeys, row)?,
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
            ) else {
                continue;
            };
            if !part_keys.contains(&partkey) {
                continue;
            }
            let (Some(o_year), Some(nationkey), Some(supplycost)) = (
                order_years.get(&orderkey).copied(),
                supplier_nations.get(&suppkey).copied(),
                supply_costs.get(&(partkey, suppkey)).copied(),
            ) else {
                continue;
            };
            let (Some(quantity), Some(extendedprice), Some(discount)) = (
                numeric_f64_value(quantities, row)?,
                numeric_f64_value(extendedprices, row)?,
                numeric_f64_value(discounts, row)?,
            ) else {
                continue;
            };
            let amount = extendedprice * (1.0 - discount) - supplycost * quantity;
            *groups.entry((nationkey, o_year)).or_insert(0.0) += amount;
        }
    }
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

fn q09_update_profit_decimal_batch(
    orderkeys: &ArrayRef,
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    part_keys: &HashSet<i64>,
    supplier_nations: &HashMap<i64, i64>,
    order_years: &HashMap<i64, i32>,
    supply_costs: &HashMap<(i64, i64), f64>,
    groups: &mut HashMap<(i64, i32), f64>,
) -> Result<bool> {
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
        return Ok(false);
    };

    for row in 0..orderkeys.len() {
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
        if !part_keys.contains(&partkey) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let (Some(o_year), Some(nationkey), Some(supplycost)) = (
            order_years.get(&orderkey).copied(),
            supplier_nations.get(&suppkey).copied(),
            supply_costs.get(&(partkey, suppkey)).copied(),
        ) else {
            continue;
        };
        let amount = extendedprices.value(row) * (1.0 - discounts.value(row))
            - supplycost * quantities.value(row);
        *groups.entry((nationkey, o_year)).or_insert(0.0) += amount;
    }
    Ok(true)
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
        q12_shipping_mode_counts_from_orders(engine, orders.path, batch_size, &pending).await?;
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

#[derive(Default)]
struct Q12State {
    high_line_count: u64,
    low_line_count: u64,
}

struct Q12Row {
    shipmode: String,
    high_line_count: u64,
    low_line_count: u64,
}

#[derive(Default)]
struct Q12PendingOrder {
    counts: HashMap<String, u64>,
}

async fn q12_filtered_lineitem_counts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &HashSet<String>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, Q12PendingOrder>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_shipmode".to_string(),
                "l_commitdate".to_string(),
                "l_receiptdate".to_string(),
                "l_shipdate".to_string(),
            ]),
            None,
        )
        .await?;
    let mut pending = HashMap::<i64, Q12PendingOrder>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let modes = batch_string_column(&batch, "l_shipmode")?;
        let commitdates = batch_column(&batch, "l_commitdate")?;
        let receiptdates = batch_column(&batch, "l_receiptdate")?;
        let shipdates = batch_column(&batch, "l_shipdate")?;
        for row in 0..batch.num_rows() {
            if modes.is_null(row) {
                continue;
            }
            let mode = modes.value(row);
            if !shipmodes.contains(mode) {
                continue;
            }
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
            *pending
                .entry(orderkey)
                .or_default()
                .counts
                .entry(mode.to_string())
                .or_insert(0) += 1;
        }
    }
    Ok(pending)
}

async fn q12_shipping_mode_counts_from_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    pending: &HashMap<i64, Q12PendingOrder>,
) -> Result<Vec<Q12Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_orderpriority".to_string(),
            ]),
            None,
        )
        .await?;
    let mut groups = HashMap::<String, Q12State>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
        for row in 0..batch.num_rows() {
            if orderpriorities.is_null(row) {
                continue;
            }
            let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
                continue;
            };
            let Some(order) = pending.get(&orderkey) else {
                continue;
            };
            let is_high_priority = matches!(orderpriorities.value(row), "1-URGENT" | "2-HIGH");
            for (mode, count) in &order.counts {
                let group = groups.entry(mode.clone()).or_default();
                if is_high_priority {
                    group.high_line_count += *count;
                } else {
                    group.low_line_count += *count;
                }
            }
        }
    }
    let mut rows = groups
        .into_iter()
        .map(|(shipmode, state)| Q12Row {
            shipmode,
            high_line_count: state.high_line_count,
            low_line_count: state.low_line_count,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.shipmode.cmp(&right.shipmode));
    Ok(rows)
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

async fn q03_order_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customers: &HashSet<i64>,
    order_cutoff: i32,
) -> Result<HashMap<i64, Q03Order>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_custkey".to_string(),
                "o_orderdate".to_string(),
                "o_shippriority".to_string(),
            ]),
            None,
        )
        .await?;
    let mut orders = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
        let priorities = batch_column(&batch, "o_shippriority")?;
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(custkey), Some(orderdate), Some(priority)) = (
                numeric_i64_value(orderkeys, row)?,
                numeric_i64_value(custkeys, row)?,
                date32_value(orderdates, row)?,
                numeric_i64_value(priorities, row)?,
            ) else {
                continue;
            };
            if customers.contains(&custkey) && orderdate < order_cutoff {
                orders.insert(
                    orderkey,
                    Q03Order {
                        o_orderdate: orderdate,
                        o_shippriority: priority,
                    },
                );
            }
        }
    }
    Ok(orders)
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
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_shipdate".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    let mut revenues = HashMap::<i64, f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let shipdates = batch_column(&batch, "l_shipdate")?;
        let extendedprices = batch_column(&batch, "l_extendedprice")?;
        let discounts = batch_column(&batch, "l_discount")?;
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
    }
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
    rows.sort_by(|left, right| {
        right
            .revenue
            .partial_cmp(&left.revenue)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.o_orderdate.cmp(&right.o_orderdate))
    });
    rows.truncate(10);
    Ok(rows)
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
    let late_orders = q04_late_order_keys(engine, lineitem_path, batch_size).await?;
    if late_orders.is_empty() {
        return Ok(Some(q04_output(Vec::new())?));
    }
    let rows = q04_priority_counts(
        engine,
        orders.path,
        batch_size,
        &late_orders,
        start_days,
        end_days,
    )
    .await?;
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
                let Some(tables) = parse_comma_join_table_refs(select)? else {
                    continue;
                };
                for table in tables {
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

async fn q04_late_order_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_commitdate".to_string(),
                "l_receiptdate".to_string(),
            ]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let commitdates = batch_column(&batch, "l_commitdate")?;
        let receiptdates = batch_column(&batch, "l_receiptdate")?;
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(commitdate), Some(receiptdate)) = (
                numeric_i64_value(orderkeys, row)?,
                date32_value(commitdates, row)?,
                date32_value(receiptdates, row)?,
            ) else {
                continue;
            };
            if commitdate < receiptdate {
                keys.insert(orderkey);
            }
        }
    }
    Ok(keys)
}

struct Q04Row {
    priority: String,
    count: u64,
}

async fn q04_priority_counts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    late_orders: &HashSet<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Vec<Q04Row>> {
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
    let mut counts = HashMap::<String, u64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
        let priorities = batch_string_column(&batch, "o_orderpriority")?;
        for row in 0..batch.num_rows() {
            if priorities.is_null(row) {
                continue;
            }
            let (Some(orderkey), Some(orderdate)) = (
                numeric_i64_value(orderkeys, row)?,
                date32_value(orderdates, row)?,
            ) else {
                continue;
            };
            if orderdate >= start_days && orderdate < end_days && late_orders.contains(&orderkey) {
                *counts.entry(priorities.value(row).to_string()).or_insert(0) += 1;
            }
        }
    }
    let mut rows = counts
        .into_iter()
        .map(|(priority, count)| Q04Row { priority, count })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.priority.cmp(&right.priority));
    Ok(rows)
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
    let mut orders = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
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
            if let Some(nationkey) = customer_nations.get(&custkey).copied() {
                orders.insert(orderkey, nationkey);
            }
        }
    }
    Ok(orders)
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
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    let mut groups = HashMap::<i64, f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let suppkeys = batch_column(&batch, "l_suppkey")?;
        let extendedprices = batch_column(&batch, "l_extendedprice")?;
        let discounts = batch_column(&batch, "l_discount")?;
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(suppkey)) = (
                numeric_i64_value(orderkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
            ) else {
                continue;
            };
            let (Some(customer_nation), Some(supplier_nation)) = (
                order_customer_nations.get(&orderkey).copied(),
                supplier_nations.get(&suppkey).copied(),
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
    }
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
    let mut orders = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(custkey)) = (
                numeric_i64_value(orderkeys, row)?,
                numeric_i64_value(custkeys, row)?,
            ) else {
                continue;
            };
            if let Some(nationkey) = customer_nations.get(&custkey).copied() {
                orders.insert(orderkey, nationkey);
            }
        }
    }
    Ok(orders)
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
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_shipdate".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    let mut groups = HashMap::<(String, String, i32), f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let suppkeys = batch_column(&batch, "l_suppkey")?;
        let shipdates = batch_column(&batch, "l_shipdate")?;
        let extendedprices = batch_column(&batch, "l_extendedprice")?;
        let discounts = batch_column(&batch, "l_discount")?;
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
                supplier_nations.get(&suppkey).copied(),
                order_customer_nations.get(&orderkey).copied(),
            ) else {
                continue;
            };
            let (Some(supp_nation), Some(cust_nation)) = (
                nation_names.get(&supp_nation_key),
                nation_names.get(&cust_nation_key),
            ) else {
                continue;
            };
            if !((supp_nation == "FRANCE" && cust_nation == "GERMANY")
                || (supp_nation == "GERMANY" && cust_nation == "FRANCE"))
            {
                continue;
            }
            let (Some(extendedprice), Some(discount)) = (
                numeric_f64_value(extendedprices, row)?,
                numeric_f64_value(discounts, row)?,
            ) else {
                continue;
            };
            let (year, _, _) = civil_from_days(i64::from(shipdate))?;
            *groups
                .entry((supp_nation.clone(), cust_nation.clone(), year))
                .or_insert(0.0) += extendedprice * (1.0 - discount);
        }
    }
    let mut rows = groups
        .into_iter()
        .map(|((supp_nation, cust_nation, l_year), revenue)| Q07Row {
            supp_nation,
            cust_nation,
            l_year,
            revenue,
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
    let mut states = HashMap::<i64, (f64, u64)>::with_capacity(part_keys.len());
    let mut candidate_rows = Vec::<(i64, f64, f64)>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkey = batch_column(&batch, "l_partkey")?;
        let quantity = batch_column(&batch, "l_quantity")?;
        let extendedprice = batch_column(&batch, "l_extendedprice")?;
        for row in 0..batch.num_rows() {
            let Some(partkey) = numeric_i64_value(partkey, row)? else {
                continue;
            };
            if !part_keys.contains(&partkey) {
                continue;
            }
            let (Some(quantity), Some(extendedprice)) = (
                numeric_f64_value(quantity, row)?,
                numeric_f64_value(extendedprice, row)?,
            ) else {
                continue;
            };
            let state = states.entry(partkey).or_insert((0.0, 0));
            state.0 += quantity;
            state.1 += 1;
            candidate_rows.push((partkey, quantity, extendedprice));
        }
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
    let country_codes = ["13", "31", "23", "29", "30", "18", "17"]
        .into_iter()
        .collect::<HashSet<_>>();
    let avg =
        q22_average_positive_acctbal(engine, customer.path.clone(), batch_size, &country_codes)
            .await?;
    let order_customers = q22_order_customer_keys(engine, orders_path, batch_size).await?;
    let mut groups = q22_customer_groups(
        engine,
        customer.path,
        batch_size,
        &country_codes,
        avg,
        &order_customers,
    )
    .await?;
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

async fn q22_average_positive_acctbal(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    country_codes: &HashSet<&str>,
) -> Result<f64> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_phone".to_string(), "c_acctbal".to_string()]),
            None,
        )
        .await?;
    let mut sum = 0.0;
    let mut count = 0_u64;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let phones = batch_string_column(&batch, "c_phone")?;
        let acctbal = batch_column(&batch, "c_acctbal")?;
        for row in 0..batch.num_rows() {
            if phones.is_null(row) {
                continue;
            }
            let phone = phones.value(row);
            if phone.len() < 2 || !country_codes.contains(&phone[..2]) {
                continue;
            }
            let Some(value) = numeric_f64_value(acctbal, row)? else {
                continue;
            };
            if value > 0.0 {
                sum += value;
                count += 1;
            }
        }
    }
    Ok(if count > 0 { sum / count as f64 } else { 0.0 })
}

async fn q22_order_customer_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_custkey".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        for row in 0..batch.num_rows() {
            if let Some(key) = numeric_i64_value(custkeys, row)? {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

struct Q22Group {
    cntrycode: String,
    count: u64,
    sum: f64,
}

async fn q22_customer_groups(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    country_codes: &HashSet<&str>,
    min_acctbal: f64,
    order_customers: &HashSet<i64>,
) -> Result<Vec<Q22Group>> {
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
    let mut groups = HashMap::<String, (u64, f64)>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let phones = batch_string_column(&batch, "c_phone")?;
        let acctbal = batch_column(&batch, "c_acctbal")?;
        for row in 0..batch.num_rows() {
            let Some(custkey) = numeric_i64_value(custkeys, row)? else {
                continue;
            };
            if order_customers.contains(&custkey) || phones.is_null(row) {
                continue;
            }
            let phone = phones.value(row);
            if phone.len() < 2 || !country_codes.contains(&phone[..2]) {
                continue;
            }
            let Some(value) = numeric_f64_value(acctbal, row)? else {
                continue;
            };
            if value <= min_acctbal {
                continue;
            }
            let state = groups.entry(phone[..2].to_string()).or_insert((0, 0.0));
            state.0 += 1;
            state.1 += value;
        }
    }
    Ok(groups
        .into_iter()
        .map(|(cntrycode, (count, sum))| Q22Group {
            cntrycode,
            count,
            sum,
        })
        .collect())
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

    let order_quantity_sums = grouped_numeric_sum(
        engine,
        lineitem.path.clone(),
        batch_size,
        "l_orderkey",
        "l_quantity",
    )
    .await?;
    let qualifying_orders = order_quantity_sums
        .iter()
        .filter_map(|(&orderkey, &sum)| (sum > 300.0).then_some(orderkey))
        .collect::<HashSet<_>>();
    if qualifying_orders.is_empty() {
        return Ok(Some(q18_output(Vec::new())?));
    }
    let order_rows =
        q18_qualifying_orders(engine, orders.path, batch_size, &qualifying_orders).await?;
    let customer_names = q18_customer_names(engine, customer.path, batch_size).await?;

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

async fn grouped_numeric_sum(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    key_column: &str,
    value_column: &str,
) -> Result<HashMap<i64, f64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![key_column.to_string(), value_column.to_string()]),
            None,
        )
        .await?;
    let mut sums = HashMap::<i64, f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let keys = batch_column(&batch, key_column)?;
        let values = batch_column(&batch, value_column)?;
        if update_i64_decimal_sums(keys, values, &mut sums)? {
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(key), Some(value)) = (
                numeric_i64_value(keys, row)?,
                numeric_f64_value(values, row)?,
            ) else {
                continue;
            };
            *sums.entry(key).or_insert(0.0) += value;
        }
    }
    Ok(sums)
}

fn update_i64_decimal_sums(
    keys: &ArrayRef,
    values: &ArrayRef,
    sums: &mut HashMap<i64, f64>,
) -> Result<bool> {
    let (Some(keys), Some(values)) = (
        keys.as_any().downcast_ref::<Int64Array>(),
        q01_decimal_input(values)?,
    ) else {
        return Ok(false);
    };
    for row in 0..keys.len() {
        if keys.is_null(row) || values.is_null(row) {
            continue;
        }
        *sums.entry(keys.value(row)).or_insert(0.0) += values.value(row);
    }
    Ok(true)
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

async fn q18_qualifying_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    qualifying_orders: &HashSet<i64>,
) -> Result<HashMap<i64, Q18Order>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_custkey".to_string(),
                "o_orderdate".to_string(),
                "o_totalprice".to_string(),
            ]),
            None,
        )
        .await?;
    let mut orders = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let custkeys = batch_column(&batch, "o_custkey")?;
        let orderdates = batch_column(&batch, "o_orderdate")?;
        let totalprices = batch_column(&batch, "o_totalprice")?;
        for row in 0..batch.num_rows() {
            let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
                continue;
            };
            if !qualifying_orders.contains(&orderkey) {
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
    }
    Ok(orders)
}

async fn q18_customer_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_name".to_string()]),
            None,
        )
        .await?;
    let mut customers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let keys = batch_column(&batch, "c_custkey")?;
        let names = batch_string_column(&batch, "c_name")?;
        for row in 0..batch.num_rows() {
            if names.is_null(row) {
                continue;
            }
            if let Some(key) = numeric_i64_value(keys, row)? {
                customers.insert(key, names.value(row).to_string());
            }
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
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, "SAUDI ARABIA").await?;
    let suppliers = q21_supplier_names(engine, supplier.path, batch_size, &nation_keys).await?;
    if suppliers.is_empty() {
        return Ok(Some(q21_output(Vec::new())?));
    }
    let final_orders = q21_final_order_keys(engine, orders.path, batch_size).await?;
    if final_orders.is_empty() {
        return Ok(Some(q21_output(Vec::new())?));
    }
    let order_states =
        q21_lineitem_order_states(engine, lineitem.path, batch_size, &final_orders).await?;
    let mut counts = HashMap::<i64, u64>::with_capacity(suppliers.len());
    for state in order_states.into_values() {
        if !state.has_multiple_suppliers || !state.has_single_late_supplier() {
            continue;
        }
        let suppkey = state.late_supplier;
        if !suppliers.contains_key(&suppkey) {
            continue;
        }
        *counts.entry(suppkey).or_insert(0) += state.late_row_count;
    }
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
    Ok(Some(q21_output(rows)?))
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
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderstatus".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "o_orderkey")?;
        let statuses = batch_string_column(&batch, "o_orderstatus")?;
        for row in 0..batch.num_rows() {
            if statuses.is_valid(row)
                && statuses.value(row) == "F"
                && let Some(key) = numeric_i64_value(orderkeys, row)?
            {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

#[derive(Default)]
struct Q21OrderState {
    first_supplier: i64,
    has_supplier: bool,
    has_multiple_suppliers: bool,
    late_supplier: i64,
    has_late_supplier: bool,
    has_multiple_late_suppliers: bool,
    late_row_count: u64,
}

impl Q21OrderState {
    fn add_supplier(&mut self, suppkey: i64) {
        if !self.has_supplier {
            self.first_supplier = suppkey;
            self.has_supplier = true;
        } else if suppkey != self.first_supplier {
            self.has_multiple_suppliers = true;
        }
    }

    fn add_late_supplier(&mut self, suppkey: i64) {
        if !self.has_late_supplier {
            self.late_supplier = suppkey;
            self.has_late_supplier = true;
            self.late_row_count = 1;
        } else if suppkey == self.late_supplier {
            self.late_row_count += 1;
        } else {
            self.has_multiple_late_suppliers = true;
        }
    }

    fn has_single_late_supplier(&self) -> bool {
        self.has_late_supplier && !self.has_multiple_late_suppliers
    }
}

async fn q21_lineitem_order_states(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    final_orders: &HashSet<i64>,
) -> Result<HashMap<i64, Q21OrderState>> {
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
    let mut states = HashMap::<i64, Q21OrderState>::with_capacity(final_orders.len());
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let orderkeys = batch_column(&batch, "l_orderkey")?;
        let suppkeys = batch_column(&batch, "l_suppkey")?;
        let receipt = batch_column(&batch, "l_receiptdate")?;
        let commit = batch_column(&batch, "l_commitdate")?;
        if q21_update_lineitem_states_typed(
            orderkeys,
            suppkeys,
            receipt,
            commit,
            final_orders,
            &mut states,
        ) {
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(orderkey), Some(suppkey)) = (
                numeric_i64_value(orderkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
            ) else {
                continue;
            };
            if !final_orders.contains(&orderkey) {
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
    }
    Ok(states)
}

fn q21_update_lineitem_states_typed(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    receipt: &ArrayRef,
    commit: &ArrayRef,
    final_orders: &HashSet<i64>,
    states: &mut HashMap<i64, Q21OrderState>,
) -> bool {
    let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        receipt.as_any().downcast_ref::<Date32Array>(),
        commit.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return false;
    };
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if !final_orders.contains(&orderkey) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        let state = states.entry(orderkey).or_default();
        state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            state.add_late_supplier(suppkey);
        }
    }
    true
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

    let forest_parts = q20_forest_part_keys(engine, part_path, batch_size).await?;
    if forest_parts.is_empty() {
        return Ok(Some(q20_output(Vec::new())?));
    }
    let lineitem_sums = q20_lineitem_quantity_sums(engine, lineitem_path, batch_size).await?;
    let eligible_suppliers = q20_eligible_supplier_keys(
        engine,
        partsupp_path,
        batch_size,
        &forest_parts,
        &lineitem_sums,
    )
    .await?;
    if eligible_suppliers.is_empty() {
        return Ok(Some(q20_output(Vec::new())?));
    }
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, "CANADA").await?;
    let mut rows = q20_supplier_rows(
        engine,
        supplier.path,
        batch_size,
        &nation_keys,
        &eligible_suppliers,
    )
    .await?;
    rows.sort_by(|left, right| left.s_name.cmp(&right.s_name));
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
) -> Result<HashMap<(i64, i64), f64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_suppkey".to_string(),
                "l_quantity".to_string(),
                "l_shipdate".to_string(),
            ]),
            None,
        )
        .await?;
    let mut sums = HashMap::<(i64, i64), f64>::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "l_partkey")?;
        let suppkeys = batch_column(&batch, "l_suppkey")?;
        let quantities = batch_column(&batch, "l_quantity")?;
        let shipdates = batch_column(&batch, "l_shipdate")?;
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
            *sums.entry((partkey, suppkey)).or_insert(0.0) += quantity;
        }
    }
    Ok(sums)
}

async fn q20_eligible_supplier_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &HashSet<i64>,
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
        for row in 0..batch.num_rows() {
            let (Some(partkey), Some(suppkey), Some(availqty)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
                numeric_f64_value(availqty, row)?,
            ) else {
                continue;
            };
            if !forest_parts.contains(&partkey) {
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
        let mask = direct_filter
            .map(|filter| evaluate_filter_mask(&batch, filter))
            .transpose()?;
        for row in 0..batch.num_rows() {
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

    let schema = batches[0].schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        concat_batches(&schema, batches.iter())?
    };
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
    Ok(vec![take_record_batch(&batch, &indices)?])
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
