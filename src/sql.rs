use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float64Array, Int32Array, Int64Array,
    StringArray, TimestampMillisecondArray, UInt64Array,
};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_ord::sort::{SortColumn, SortOptions, lexsort_to_indices};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use sqlparser::ast::{
    BinaryOperator, DateTimeField, Distinct, Expr as SqlExpr, FunctionArg, FunctionArgExpr,
    FunctionArguments, GroupByExpr, JoinConstraint, JoinOperator, LimitClause, ObjectName,
    ObjectNamePart, OrderByKind, Query, Select, SelectItem, SetExpr, Statement, TableFactor,
    UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::engine::{DodamEngine, JoinAlgorithm, JoinParquetRequest};
use crate::error::{DodamError, Result};
use crate::execution::JoinType;
use crate::execution::{
    AggregateExpr, AggregateMetrics, ComparisonExpr, ComparisonOp, DistinctExec, Expr, FilterExpr,
    HashJoinExec, JoinBuildSide, LiteralValue, MemoryExec, PhysicalPlan, Projection,
    RecordBatchSink, ScanPlanMetrics, SendableBatchStream, SortExpr, SortKey, collect_aggregates,
    collect_grouped_aggregates, filter_batch,
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
    if let Some(output) = try_execute_derived_join_sql(engine, sql, batch_size).await? {
        return Ok(output);
    }
    if let Some(output) = try_execute_derived_sql(engine, sql, batch_size).await? {
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
        let join_plan = plan_join_inputs(
            &query.projection,
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
                right_filter: join_plan.right_filter,
                output_projection,
                join_memory_limit_bytes: default_join_memory_limit_bytes(),
                join_algorithm: JoinAlgorithm::Auto,
                join_type: join.join_type,
            })
            .await?;
        if is_aggregate {
            let stream = apply_output_filter_stream(stream, query.filter.clone());
            let metrics = if group_by.is_empty() {
                collect_aggregates(stream, 2, &aggregates)?
            } else {
                collect_grouped_aggregates(stream, 2, &group_by, &aggregates)?
            };
            let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
            batches = apply_output_filter(batches, query.having.as_ref())?;
            batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
            batches = rename_output_batches(batches, &query.aliases)?;
            return Ok(QueryOutput::Aggregate { metrics, batches });
        }
        let mut batches = collect_batches(stream)?;
        batches = apply_output_filter(batches, query.filter.as_ref())?;
        batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
        if !output_projection_pushed {
            batches = apply_output_projection(batches, &query.projection)?;
        }
        batches = rename_output_batches(batches, &query.aliases)?;
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
            let batches = append_aggregate_expression_columns(
                collect_batches(stream)?,
                &query.aggregate_expressions,
            )?;
            let stream = Box::new(MemoryExec::new(batches)).execute()?;
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
        batches = apply_output_order_limit(batches, query.order_by.as_ref(), query.limit)?;
        batches = rename_output_batches(batches, &query.aliases)?;
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
    let path = parse_from(select)?;
    let Some(outer_alias) = path.alias.as_deref() else {
        return Ok(None);
    };
    let selection_sql = selection.to_string();
    if !subquery_references_outer_alias(&selection_sql, outer_alias) {
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
            "correlated subquery filters currently support only non-aggregate SELECT queries"
                .to_string(),
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
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            &selection_sql,
            outer_alias,
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

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit)?;
    batches = apply_output_projection(batches, &parsed_projection.projection)?;
    batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn evaluate_correlated_subquery_filter_mask(
    engine: &DodamEngine,
    selection_sql: &str,
    outer_alias: &str,
    batch: &RecordBatch,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<BooleanArray> {
    let mut values = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let bound_sql = bind_outer_row_references(selection_sql, outer_alias, batch, row)?;
        let bound_expr = parse_sql_expr_fragment(&bound_sql)?;
        let row_batch = batch.slice(row, 1);
        let expr = Box::pin(parse_filter_with_subqueries(
            engine,
            &bound_expr,
            &[],
            table_alias,
            false,
            batch_size,
        ))
        .await?;
        let matches = if let Some(expr) = expr {
            filter_batch(row_batch, &FilterExpr::new(expr))?.num_rows() > 0
        } else {
            true
        };
        values.push(Some(matches));
    }
    Ok(BooleanArray::from(values))
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
        let bound_sql = bind_outer_row_references(subquery_sql, outer_alias, batch, row)?;
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
    outer_alias: &str,
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
        sql = sql.replace(
            &format!("{outer_alias}.{}", field.name()),
            &sql_literal(&literal),
        );
    }
    Ok(sql)
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
    let filter = Box::pin(parse_filter_with_subqueries(
        engine,
        select.selection.as_ref().expect("selection checked"),
        &[],
        path.alias.as_deref(),
        false,
        batch_size,
    ))
    .await?
    .map(FilterExpr::new);
    let order_by = parse_order_by(query, &parsed_projection.aliases, path.alias.as_deref())?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let stream = if distinct {
        engine
            .scan_parquet_distinct_batches(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
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
                parsed_projection.projection,
                filter,
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
                filter,
            )
            .await?
    };
    let batches = rename_output_batches(collect_batches(stream)?, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
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
    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        return Ok(None);
    }
    if parse_distinct(select)? {
        return Err(DodamError::UnsupportedSql(
            "projection expressions currently support only non-aggregate SELECT queries"
                .to_string(),
        ));
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
        _ => false,
    }
}

fn scalar_predicate_side_requires_expression(expr: &SqlExpr) -> bool {
    !matches!(
        expr,
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) | SqlExpr::Value(_)
    )
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
        SqlExpr::Cast { expr, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
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
    let (join_type, left_keys, right_keys) = parse_join_condition(join, &left_alias, &right_alias)?;
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
        Box::new(MemoryExec::new(right.batches)),
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        batches = rename_output_batches(batches, &projection.aliases)?;
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    batches = apply_output_projection(batches, &projection.projection)?;
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    batches = rename_output_batches(batches, &projection.aliases)?;
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
            let alias = table.alias.clone().ok_or_else(|| {
                DodamError::UnsupportedSql("JOIN inputs must have table aliases".to_string())
            })?;
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
    let inner_output = Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?;
    let inner_batches = query_output_batches(inner_output)?;
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

    if !parsed_projection.aggregates.is_empty() {
        let filtered_batches = apply_output_filter(inner_batches, filter.as_ref())?;
        let stream = Box::new(MemoryExec::new(filtered_batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;
    let mut batches = apply_output_filter(inner_batches, filter.as_ref())?;
    batches = apply_output_projection(batches, &parsed_projection.projection)?;
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit)?;
    batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
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
        }))
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
            right_filter: join_plan.right_filter,
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
            right_filter: join_plan.right_filter,
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
                right_filter: join_plan.right_filter,
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
    if !matches!(join.join_type, JoinType::Inner | JoinType::Semi) {
        return Projection::All;
    }
    if query.is_aggregate() {
        return aggregate_join_output_projection(query);
    }
    if query.filter.is_some() || query.order_by.is_some() {
        Projection::All
    } else {
        query.projection.clone()
    }
}

fn aggregate_join_output_projection(query: &SqlQuery) -> Projection {
    let Projection::Columns(columns) = &query.projection else {
        return Projection::All;
    };
    let mut columns = columns.clone();
    if let Some(filter) = &query.filter {
        for column in filter.referenced_columns() {
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
        group_by,
        aliases: parsed_projection.aliases,
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
    let left_alias = left.alias.clone().ok_or_else(|| {
        DodamError::UnsupportedSql("JOIN inputs must have table aliases".to_string())
    })?;
    let right_alias = right.alias.clone().ok_or_else(|| {
        DodamError::UnsupportedSql("JOIN inputs must have table aliases".to_string())
    })?;
    let (join_type, left_keys, right_keys) = parse_join_condition(join, &left_alias, &right_alias)?;
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

fn parse_join_condition(
    join: &sqlparser::ast::Join,
    left_alias: &str,
    right_alias: &str,
) -> Result<(JoinType, Vec<String>, Vec<String>)> {
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
    collect_join_equalities(
        expr,
        left_alias,
        right_alias,
        &mut left_keys,
        &mut right_keys,
    )?;
    Ok((join_type, left_keys, right_keys))
}

fn collect_join_equalities(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    left_keys: &mut Vec<String>,
    right_keys: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_join_equalities(left, left_alias, right_alias, left_keys, right_keys)?;
            collect_join_equalities(right, left_alias, right_alias, left_keys, right_keys)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
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
        )),
    }
}

fn parse_join_projection(
    select: &Select,
    table_aliases: &[&str],
    group_by: &[String],
) -> Result<ParsedProjection> {
    let mut columns = Vec::new();
    let mut aggregates = Vec::new();
    let mut aliases = Vec::new();
    let mut wildcard = false;

    for item in &select.projection {
        match item {
            SelectItem::Wildcard(_) => wildcard = true,
            SelectItem::UnnamedExpr(expr @ SqlExpr::CompoundIdentifier(_)) => {
                columns.push(qualified_join_column(expr, table_aliases)?);
            }
            SelectItem::UnnamedExpr(SqlExpr::Function(function)) => {
                aggregates.push(parse_join_aggregate(function, table_aliases)?);
            }
            SelectItem::ExprWithAlias { expr, alias } => match expr {
                SqlExpr::CompoundIdentifier(_) => {
                    let column = qualified_join_column(expr, table_aliases)?;
                    aliases.push((alias.value.clone(), column.clone()));
                    columns.push(column);
                }
                SqlExpr::Function(function) => {
                    let aggregate = parse_join_aggregate(function, table_aliases)?;
                    aliases.push((alias.value.clone(), aggregate.to_string()));
                    aggregates.push(aggregate);
                }
                _ => {
                    return Err(DodamError::UnsupportedSql(format!(
                        "unsupported JOIN SELECT item: {item}"
                    )));
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
        return Ok(ParsedProjection {
            projection: Projection::Columns(projected_columns),
            aggregates,
            aggregate_expressions: Vec::new(),
            aliases,
            expressions: Vec::new(),
        });
    }

    Ok(ParsedProjection {
        projection: if wildcard {
            Projection::All
        } else {
            Projection::Columns(columns)
        },
        aggregates,
        aggregate_expressions: Vec::new(),
        aliases,
        expressions: Vec::new(),
    })
}

fn parse_join_group_by(select: &Select, table_aliases: &[&str]) -> Result<Vec<String>> {
    match &select.group_by {
        GroupByExpr::Expressions(expressions, modifiers) if modifiers.is_empty() => expressions
            .iter()
            .map(|expr| qualified_join_column(expr, table_aliases))
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
                SqlExpr::Identifier(ident) => resolve_alias(&ident.value, aliases),
                SqlExpr::CompoundIdentifier(_) => {
                    qualified_join_column(&order.expr, table_aliases)?
                }
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
            | BinaryOperator::LtEq => Ok(Expr::Comparison(ComparisonExpr {
                column: join_filter_column(left, aliases, table_aliases, allow_aggregates)?,
                op: sql_comparison_op(op),
                value: sql_literal_value(right)?,
            })),
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported JOIN WHERE operator: {op}"
            ))),
        },
        SqlExpr::Nested(expr) => {
            join_expr_to_filter_expr(expr, aliases, table_aliases, allow_aggregates)
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
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN WHERE expression: {expr}"
        ))),
    }
}

fn join_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<String> {
    match expr {
        SqlExpr::Identifier(ident) => Ok(resolve_alias(&ident.value, aliases)),
        SqlExpr::CompoundIdentifier(_) => qualified_join_column(expr, table_aliases),
        SqlExpr::Function(function) if allow_aggregates => {
            Ok(parse_join_aggregate(function, table_aliases)?.to_string())
        }
        SqlExpr::Nested(expr) => join_filter_column(expr, aliases, table_aliases, allow_aggregates),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected JOIN column or aggregate expression, got {expr}"
        ))),
    }
}

fn qualified_join_column(expr: &SqlExpr, table_aliases: &[&str]) -> Result<String> {
    let SqlExpr::CompoundIdentifier(parts) = expr else {
        return Err(DodamError::UnsupportedSql(format!(
            "expected qualified column, got {expr}"
        )));
    };
    let [qualifier, column] = parts.as_slice() else {
        return Err(DodamError::UnsupportedSql(format!(
            "only table-qualified columns are supported, got {expr}"
        )));
    };
    if !table_aliases
        .iter()
        .any(|table_alias| *table_alias == qualifier.value)
    {
        return Err(DodamError::UnsupportedSql(format!(
            "unknown table qualifier: {}",
            qualifier.value
        )));
    }
    Ok(format!("{}.{}", qualifier.value, column.value))
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
                let expression = parse_scalar_projection(expr, None, table_alias)?;
                for column in scalar_expression_columns(&expression.expr) {
                    add_column_once(&mut columns, column);
                }
                expressions.push(expression);
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
                    let expression =
                        parse_scalar_projection(expr, Some(&alias.value), table_alias)?;
                    for column in scalar_expression_columns(&expression.expr) {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(expression);
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
        && expressions
            .iter()
            .any(|expr| !matches!(expr.expr, ScalarSqlExpression::Column(_)))
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

fn scalar_expression_columns(expr: &ScalarSqlExpression) -> Vec<String> {
    let mut columns = Vec::new();
    collect_scalar_expression_columns(expr, &mut columns);
    columns
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
        | ScalarSqlExpression::Length(expr) => collect_scalar_expression_columns(expr, columns),
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
    let args = match &function.args {
        FunctionArguments::List(args)
            if args.clauses.is_empty() && args.duplicate_treatment.is_none() =>
        {
            &args.args
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
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
    AggregateExpr::parse(&format!("{name}({argument})"))
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
    let args = match &function.args {
        FunctionArguments::List(args)
            if args.clauses.is_empty() && args.duplicate_treatment.is_none() =>
        {
            &args.args
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "unsupported function arguments: {}",
                function.args
            )));
        }
    };
    let argument = match args.as_slice() {
        [] => {
            return Err(DodamError::UnsupportedSql(format!(
                "missing argument for {name}"
            )));
        }
        [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => "*".to_string(),
        [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr @ SqlExpr::CompoundIdentifier(_)))] => {
            qualified_join_column(expr, table_aliases)?
        }
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "JOIN aggregate arguments must be * or qualified columns, got {}",
                function.args
            )));
        }
    };
    AggregateExpr::parse(&format!("{name}({argument})"))
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
            | BinaryOperator::LtEq => Ok(Expr::Comparison(ComparisonExpr {
                column: sql_filter_column(left, aliases, table_alias, allow_aggregates)?,
                op: sql_comparison_op(op),
                value: sql_literal_value(right)?,
            })),
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported WHERE operator: {op}"
            ))),
        },
        SqlExpr::Nested(expr) => {
            sql_expr_to_filter_expr(expr, aliases, table_alias, allow_aggregates)
        }
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
        } => Ok(Expr::InList {
            column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
            negated: *negated,
            has_null: literal_list_contains_null(list)?,
            values: non_null_literal_values(list)?,
        }),
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
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected literal, got {expr}"
        ))),
    }
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
    match expr {
        SqlExpr::Identifier(ident) => Ok(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => {
            let [qualifier, column] = parts.as_slice() else {
                return Err(DodamError::UnsupportedSql(format!(
                    "only table-qualified columns are supported, got {expr}"
                )));
            };
            if let Some(table_alias) = table_alias
                && qualifier.value != table_alias
            {
                return Err(DodamError::UnsupportedSql(format!(
                    "unknown table qualifier: {}",
                    qualifier.value
                )));
            }
            Ok(column.value.clone())
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected column identifier, got {expr}"
        ))),
    }
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
        .map(|batch| {
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
        })
        .collect()
}

#[derive(Clone)]
enum EvaluatedScalar {
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Utf8(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
}

impl EvaluatedScalar {
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
                        .find(|(_, target)| target == field.name())
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
