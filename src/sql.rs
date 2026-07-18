use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBufferBuilder, BooleanBuilder, Date32Array, Date64Array,
    Decimal128Array, DictionaryArray, Float64Array, Int32Array, Int64Array, ListArray, StringArray,
    StructArray, TimestampMillisecondArray, UInt32Array, UInt64Array, make_array,
};
use arrow::buffer::NullBuffer;
use arrow::compute::filter_record_batch;
use arrow::compute::kernels::boolean::{is_not_null, is_null};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use arrow_ord::sort::{SortColumn, SortOptions, lexsort_to_indices};
use arrow_row::{OwnedRow, RowConverter, SortField};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use memchr::memmem::Finder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use sqlparser::ast::{
    AccessExpr, BinaryOperator, CeilFloorKind, DateTimeField, Distinct, DuplicateTreatment,
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, JoinConstraint,
    JoinOperator, LimitClause, ObjectName, ObjectNamePart, OrderByKind, Query, Select, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, SetOperator, SetQuantifier, Statement, Subscript,
    TableFactor, TableWithJoins, UnaryOperator, Value, WindowType,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::cost::{
    DecimalRangeSelectivityInput, ExpressionAggregateLateChunkCostInput,
    FusedSelectedAggregateCostInput, OrderedPrimitiveChunkCostInput, OrderedPrimitiveSinkCostInput,
    PipelineMemoryCostInput, PipelineMemoryStrategy, PrimitiveOrderLimitCostInput,
    PrimitiveOrderLimitStrategy, ProjectionSelectivityCostInput, SelectedPayloadDecision,
    SelectedPayloadSpreadCostInput, SqlRuleCostInput, StreamingLeftDeepJoinCostInput,
    WorkerCostInput, choose_expression_aggregate_late_row_group_chunk,
    choose_fused_selected_aggregate, choose_late_materialization_projection_selected_ratio,
    choose_ordered_primitive_row_group_chunk, choose_ordered_primitive_sink,
    choose_parallel_workers, choose_pipeline_memory_strategy,
    choose_primitive_order_limit_strategy, choose_selected_payload_by_spread,
    choose_streaming_left_deep_join, estimate_sql_rule_cost,
};
use crate::dense::{
    AdaptiveI64Map, AdaptiveI64Set, DenseAtomicU8, DenseI64BoolLookup, DenseI64F64Sum,
    DenseI64I32Map, PackedU32PairDistinct, SortedI64Lookup, adaptive_dense_index,
};
use crate::engine::{
    DirectPrimitiveBatchLocation, DodamEngine, JoinAlgorithm, JoinParquetRequest,
    LateMaterializationPolicy, LateMaterializedMetrics, LateSelectionBuilder,
    OrderedRowGroupBoundary, OrderedRowGroupChunk, RowGroupBatchScanProfile,
    direct_selection_fold_enabled, merge_ordered_row_group_chunks,
};
use crate::error::{DodamError, Result};
use crate::execution::JoinType;
use crate::execution::{
    AggregateExpr, AggregateMetrics, AggregateResult, AggregateValue, CoalesceKeyCountSumCollector,
    ComparisonExpr, ComparisonOp, DecimalDateRangeFilter, DecimalInput, DistinctExec, Expr,
    FilterExpr, GroupAggregateResult, GroupKeyExpr, GroupKeyLiteral, GroupValue, HashJoinExec,
    JoinBuildSide, LiteralValue, MemoryExec, PartitionedHashJoinExec, PartitionedHashJoinOptions,
    PhysicalPlan, PredicateSet, PrimitiveBatch, PrimitiveColumn, PrimitiveColumnValues, Projection,
    RecordBatchSink, ScanPlanMetrics, SendableBatchStream, SortExpr, SortKey, StripPrefixExec,
    aggregate_metrics_to_batches, collect_aggregates, collect_grouped_aggregates,
    collect_grouped_aggregates_with_key_exprs, decimal_discounted_revenue_raw,
    decimal_discounted_revenue_raw_i64, decimal_discounted_revenue_scales, decimal_input,
    evaluate_filter_mask, evaluate_projected_view_filter_mask, filter_batch, scan_projection,
    try_for_each_i64_i64_date32,
};
use crate::hash::{FastHashMap, FastHashSet, fast_hash_map, fast_hash_map_with_capacity};
use crate::optimizer::{
    ColumnRangeStats, JoinAggregateLookupFusionCostInput, JoinInputPlan, LogicalJoinEdge,
    LogicalJoinGraph, LogicalJoinPlanTree, LogicalJoinTableStats,
    estimate_join_aggregate_lookup_fusion_cost as estimate_optimizer_join_aggregate_lookup_fusion_cost,
    plan_join_inputs,
};
use crate::storage::{
    DirectColumnScanMetrics, DirectI64I32I32ScanMetrics, DirectOrderedPrimitiveBatch,
    DirectOrderedPrimitiveColumnValues, DirectPrimitiveColumnScanMetrics,
    DirectPrimitiveColumnSpec, DirectPrimitiveColumnType, DirectSelectedPrimitiveColumnPageView,
    DirectSelectedPrimitivePageBatch, PrimitiveRowGroupMinMax, read_i32_le_unchecked,
};
use crate::vector::{
    BatchConsumer, BatchView, Date32VectorView, Decimal128VectorView, DictionaryI32View,
    DictionaryStringValues, I32VectorView, I64VectorView, SelectionVector, Utf8VectorView,
    consume_record_batch, dictionary_i32_view_match_flags, dictionary_i32_view_match_index,
    read_i32_le_unaligned, read_i64_le_unaligned, store_i64_keys_matching_dictionary_target,
    store_i64_keys_matching_utf8_target,
};

use aggregate_parser::{
    filtered_aggregate_spec_from_function, filtered_join_aggregate_spec_from_function,
    parse_aggregate, parse_aggregate_output_expression, parse_aggregate_with_input_expression,
    parse_join_aggregate, parse_join_aggregate_output_expression,
    parse_join_aggregate_with_input_expression,
};
mod aggregate_parser;
mod batch_streams;
mod bilateral_shipping_volume;
mod column_resolver;
mod comma_join;
mod correlated_avg_threshold;
mod derived_count;
mod derived_prefix_avg_antijoin;
mod direct_join_sink;
mod discounted_revenue_predicate;
mod explain;
mod expression_aggregate;
mod filter_parser;
mod group_by;
mod grouped_sum_semijoin;
mod join_coalesce;
mod join_condition;
mod join_filter_parser;
mod join_lookup_fusion;
mod join_projection;
mod join_projection_plan;
mod late_materialization;
mod literal_values;
mod literals;
mod metadata_predicate;
mod nation_market_share;
mod native_filtered;
mod order_priority_exists;
mod output_order;
mod output_utils;
mod parallel_folds;
mod parts_supplier_group;
mod predicate_pushdown;
mod prefix_supplier_threshold;
mod pricing_summary;
mod primitive_buffers;
mod primitive_selection;
mod profiling;
mod profit_by_nation_year;
mod projection_expression;
mod projection_parser;
mod projection_types;
mod projection_utils;
mod query_features;
mod query_modifiers;
mod query_parser;
mod query_routing;
mod regional_supplier_revenue;
mod returned_customer_revenue;
mod rule_registry;
mod scalar_eval;
mod scalar_output;
mod scalar_parser;
mod scan_decimal_aggregate;
mod semijoin;
mod semijoin_exists;
mod semijoin_tuple;
mod set_operations;
mod set_query;
mod set_sink;
mod shipping_order_priority;
mod shipping_priority_counts;
mod shipping_priority_revenue;
mod subquery_rewrite;
mod supplier_stock_threshold;
mod supplier_wait_antijoin;
mod table_refs;
mod tpch_rules;
mod types;
mod window;
mod with_cte;

use batch_streams::{
    MonotonicOrderState, OrderedLimitCollector, apply_output_filter_stream, collect_batches,
    collect_expression_filtered_limit_batches, collect_ordered_stream_limit_batches,
    collect_verified_monotonic_order_limit_batches,
};
use bilateral_shipping_volume::*;
use column_resolver::{
    BoundColumn, ColumnResolver, aggregate_column_parts, batch_column_index,
    batch_projected_column_index, infer_tpch_table_alias, join_column_name, object_name_to_string,
    projected_column_index, resolve_batch_column, sql_column_name,
};
use comma_join::*;
use correlated_avg_threshold::*;
use derived_count::*;
use derived_prefix_avg_antijoin::*;
use direct_join_sink::plan_direct_join_sink_request;
use discounted_revenue_predicate::*;
use explain::explain_sql;
use expression_aggregate::{
    append_aggregate_expression_columns, boolean_array_no_nulls_from_len,
    collect_aggregates_with_optional_expression_views, compare_i32,
    expression_aggregate_output_limit, expression_aggregate_row_at_time_fallback_enabled,
    group_key_exprs_for_aggregate, push_boolean_mask_selection, simple_case_literal_group_key,
    try_collect_expression_aggregate_fused_dictionary_selected,
    try_collect_expression_aggregate_late_materialized,
    try_collect_expression_aggregate_row_group_map, try_collect_expression_aggregate_scan_fold,
};
use filter_parser::{parse_filter, sql_expr_to_filter_expr, sql_filter_column};
use group_by::{
    group_by_expressions, group_by_synthetic_column, group_expression_bindings,
    join_group_expression_bindings, parse_group_by, physical_projection_columns,
    projected_group_expression, projection_ordinal_targets, qualified_wildcard_name,
};
use grouped_sum_semijoin::*;
use join_coalesce::*;
use join_condition::{
    collect_filter_columns, collect_sql_and_conjuncts, collect_sql_or_disjuncts,
    combine_expr_filters, combine_filter_options, combine_sql_and_conjuncts,
    combine_sql_and_disjuncts, comma_join_base_edge, comma_join_equality_keys,
    comma_join_keys_for_next, common_or_comma_join_equality_keys, join_column_owner,
    joined_comma_join_key, maybe_join_column_name, parse_join_condition, strip_column_prefix,
    unqualified_join_column,
};
use join_filter_parser::{join_expr_to_filter_expr, parse_join_filter, parse_join_filter_plan};
use join_lookup_fusion::{
    choose_join_aggregate_lookup_fusion, execute_join_aggregate_lookup_fusion,
    join_aggregate_lookup_fusion_disabled, plan_join_aggregate_lookup_fusion,
};
use join_projection::{parse_join_group_by, parse_join_order_by, parse_join_projection};
use join_projection_plan::{
    join_input_projection_with_expression_filter, pushed_join_output_projection,
};
use late_materialization::*;
use literal_values::{
    compare_literal_values, evaluate_literal_in_values, literal_list_contains_null,
    literal_value_from_array, literal_values_from_single_column_batches, non_null_literal_values,
    non_null_subquery_values, query_output_batches, scalar_literal_value_from_batches,
    subquery_values_contain_null,
};
use literals::{
    Date32YearCache, civil_from_days, days_from_civil, evaluate_literal_in_list, literal_as_f64,
    parse_date32_days, parse_decimal_cast_target, parse_decimal_literal_to_scaled,
    parse_timestamp_millis_value, parse_usize_literal, parse_ymd, sql_comparison_op,
    sql_like_escape, sql_like_pattern, sql_literal_value,
};
use metadata_predicate::simplify_filtered_aggregates_with_parquet_stats;
use nation_market_share::*;
use native_filtered::{
    NativeFilteredAggregateSpec, collect_native_filtered_aggregates,
    legacy_case_filtered_aggregate_specs, native_filtered_input_kind,
};
use order_priority_exists::*;
use output_order::{
    apply_aggregate_output_order_limit, apply_output_expression_projection_order_limit,
    apply_output_order_limit, output_batches_satisfy_order,
};
use output_utils::{apply_output_distinct, limit_batches};
use parallel_folds::*;
use parts_supplier_group::*;
use predicate_pushdown::{
    PredicateParserKind, safe_expression_pushdown_filter, split_subquery_and_expression_filters,
};
use prefix_supplier_threshold::*;
use pricing_summary::*;
use primitive_buffers::{
    NullFreePrimitiveColumn, PrimitiveColumnOutput, PrimitiveFilterValues,
    direct_ordered_primitive_batch_to_primitive_batch, gather_null_free_primitive_column,
    overwrite_direct_selected_page_value, overwrite_null_free_primitive_value,
    overwrite_primitive_batch_value, overwrite_primitive_output_slot,
    primitive_column_matches_direct_type, primitive_column_values_key, primitive_empty_batch,
    primitive_output_batch_from_columns, primitive_output_data_type, primitive_output_len,
    primitive_topk_key, push_direct_selected_page_value, push_null_free_primitive_row,
    push_null_free_primitive_value, push_primitive_batch_value, push_primitive_output_slot,
};
use primitive_selection::{
    ordered_sink_profile_enabled, primitive_ordered_selected_batch,
    primitive_topk_filter_i32_positions, primitive_topk_filter_i64_positions,
    primitive_topk_filter_positions_into, primitive_topk_filter_positions_with_min_key_into,
    primitive_topk_sequence_base, reserve_selected_positions, row_at_time_fallback_enabled,
};
use profiling::{
    generic_profile_elapsed, generic_profile_start, semijoin_profile_enabled, sql_elapsed_nanos,
    sql_nanos_to_millis, tpch_profile_elapsed, tpch_profile_enabled, tpch_profile_start,
};
use profit_by_nation_year::*;
use projection_expression::{
    monotonic_order_limit_scan_enabled, monotonic_stream_limit_column,
    small_dynamic_in_list_row_filter_limit, try_execute_projection_expression_sql,
};
use projection_parser::{
    add_filtered_aggregate_projection_columns, column_output_name, parse_projection,
    tpch_alias_prefix,
};
use projection_types::{
    GroupExpressionBinding, ParsedProjection, ProjectionExpression, ScalarSqlExpression,
};
use projection_utils::{
    add_column_once, add_projection_column_once, add_projection_columns, apply_output_projection,
    apply_qualified_wildcard_projection, output_batch_column_index,
    projection_expressions_are_plain_columns, projection_requires_expression_path,
};
use query_features::{
    expr_contains_materializable_subquery, parse_distinct, query_contains_set_operation,
    reject_query_features, reject_select_features, validate_distinct,
};
use query_modifiers::{
    alias_target, parse_limit, parse_offset, parse_order_by, resolve_alias,
    resolve_order_by_ordinal, scan_limit_with_offset,
};
use query_parser::{parse_query, split_comma_join_selection, split_subquery_residual};
use query_routing::{
    plan_direct_join_sink_request_relaxed, sql_select_has_explicit_join,
    sql_uses_materialized_subquery, sql_uses_multi_comma_join, sql_uses_set_operation,
};
use regional_supplier_revenue::*;
use returned_customer_revenue::*;
use rule_registry::{sql_rule_shape_mismatch_error, try_execute_registered_sql_rules};
use scalar_eval::{
    EvaluatedScalar, ScalarValue, apply_output_expression_filter,
    apply_output_expression_projection, apply_output_filter, apply_output_join_expression_filter,
    boolean_and, boolean_not, boolean_or, decimal_scale_i128, evaluate_scalar_expression,
    evaluate_scalar_predicate, evaluated_array, evaluated_column, literal_as_date32_for_type,
    literal_as_decimal128_for_type, literal_as_i64_for_type, reverse_binary_operator,
    scalar_as_f64, scalar_value_as_i64, scalar_value_at, validate_decimal_precision,
};
use scalar_output::{
    coalesce_options, drop_prefixed_columns, format_date32_days, format_decimal128_value,
    format_f64_for_sql_varchar, format_timestamp_millis, rename_output_batches,
    strip_batch_field_prefix,
};
use scalar_parser::{
    case_conditions_from_operand, join_scalar_expression_columns, join_sql_expression_columns,
    parse_join_scalar_function_projection, parse_join_scalar_sql_expression,
    parse_join_struct_field_access, parse_scalar_function_projection, parse_scalar_sql_expression,
    parse_struct_field_access, rewrite_join_scalar_predicate, scalar_expression_columns,
    scalar_expression_references_aggregate, sql_column_expr,
};
use scan_decimal_aggregate::try_collect_filtered_decimal_product_sum_scan_fold;
use semijoin::{
    apply_correlated_subquery_filter_batches, evaluate_correlated_subquery_filter_mask,
    rewrite_uncorrelated_scalar_subqueries_to_literals, semijoin_key_at,
    try_apply_correlated_min_equality_filter, try_execute_correlated_exists_subquery_sql,
    try_execute_correlated_in_pair_semijoin_sql, try_execute_correlated_subquery_filter_sql,
    try_execute_in_subquery_sql,
};
use semijoin_exists::{
    top_level_exists_subquery, try_execute_correlated_exists_semijoin_sql,
    try_execute_exists_subquery_sql,
};
use semijoin_tuple::*;
use set_operations::{
    align_union_all_batch_schema, append_disjoint_literal_values, append_union_all_batches,
    append_unique_literal_values, apply_all_row_set_operation, apply_distinct_row_set_operation,
    except_literal_values, intersect_literal_values, literal_values_to_unique_i64,
    union_all_operand_sql_with_child_topk, union_child_topk_for_quantifier,
    union_quantifier_is_distinct, validate_union_all_batches,
};
use set_query::*;
use set_sink::{
    append_same_source_union_all_filter_batches, same_source_union_primitive_chunk_size,
    try_execute_set_operation_sql_to_sink, write_same_source_primitive_batch_to_sink,
};
use shipping_order_priority::*;
use shipping_priority_counts::*;
use shipping_priority_revenue::*;
use subquery_rewrite::{
    try_execute_correlated_join_subquery_filter_sql, try_execute_materialized_join_subquery_sql,
};
use supplier_stock_threshold::*;
use supplier_wait_antijoin::*;
use table_refs::{
    SqlTableRef, named_comma_join_tables, parse_comma_join_table_refs, parse_derived_from,
    parse_from, parse_multi_input_join_table_refs_and_conjuncts, parse_select_table_refs,
    parse_table_factor, select_inner_column_prefixes, table_ref_alias_or_name,
};
pub use types::{
    QueryOutput, SqlExecutionOptions, SqlResultSink, SqlSinkExecutionOptions,
    SqlSinkExecutionProfile,
};
use window::try_execute_window_sql;
use with_cte::try_execute_with_cte_sql;

const DEFAULT_MAX_DENSE_I64_KEY: usize = 20_000_000;
const DEFAULT_Q09_ORDER_YEAR_DENSE_BYTES: usize = 384 * 1024 * 1024;
const MAX_SQL_EXTERNAL_JOIN_PARTITIONS: usize = 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct SqlQuery {
    path: PathBuf,
    join: Option<SqlJoin>,
    projection: Projection,
    filter: Option<FilterExpr>,
    expression_filter: Option<SqlExpr>,
    having: Option<FilterExpr>,
    order_by: Option<SortKey>,
    limit: Option<usize>,
    offset: usize,
    distinct: bool,
    aggregates: Vec<AggregateExpr>,
    filtered_aggregates: Vec<NativeFilteredAggregateSpec>,
    aggregate_expressions: Vec<ProjectionExpression>,
    expressions: Vec<ProjectionExpression>,
    group_by: Vec<String>,
    aliases: Vec<(String, String)>,
    qualified_wildcards: Vec<String>,
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
    execute_sql_with_options(engine, sql, batch_size, SqlExecutionOptions::default()).await
}

pub async fn execute_sql_with_options(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<QueryOutput> {
    if let Some(plan) = explain_sql(engine, sql, batch_size).await? {
        return Ok(QueryOutput::Explain { plan });
    }
    if let Some(output) = match try_execute_set_operation_sql(engine, sql, batch_size).await {
        Ok(output) => output,
        Err(DodamError::UnsupportedSql(message)) if sql_rule_shape_mismatch_error(&message) => None,
        Err(error) => return Err(error),
    } {
        return Ok(output);
    }
    if let Some(output) = match try_execute_window_sql(engine, sql, batch_size).await {
        Ok(output) => output,
        Err(DodamError::UnsupportedSql(message)) if sql_rule_shape_mismatch_error(&message) => None,
        Err(error) => return Err(error),
    } {
        return Ok(output);
    }
    if let Some(output) = try_execute_registered_sql_rules(engine, sql, batch_size, options).await?
    {
        return Ok(output);
    }
    let query = parse_sql(sql)?;
    if let Some(join) = query.join.clone() {
        let is_aggregate = query.is_aggregate();
        let aggregates = query.aggregates.clone();
        let group_by = query.group_by.clone();
        let join_input_projection = if query.qualified_wildcards.is_empty() {
            join_input_projection_with_expression_filter(&query)?
        } else {
            Projection::All
        };
        let projection_requires_expression =
            projection_requires_expression_path(&query.expressions);
        let input_order_by = if projection_requires_expression {
            None
        } else {
            query.order_by.as_ref()
        };
        let join_plan = plan_join_inputs(
            &join_input_projection,
            query.filter.as_ref(),
            input_order_by,
            &join.left_alias,
            &join.left_keys,
            &join.right_alias,
            &join.right_keys,
        );
        if let Some(join_graph) =
            build_logical_explicit_join_graph(engine, &query, &join, &join_plan)?
        {
            log_multi_input_join_optimizer_plan(
                "explicit_join_order",
                &join_graph,
                join_graph.choose_best_plan().as_ref(),
            );
        }
        if is_aggregate
            && let Some(output) = try_execute_join_coalesce_count_sum_aggregate(
                engine, &query, &join, &join_plan, batch_size,
            )
            .await?
        {
            return Ok(output);
        }
        let output_projection = pushed_join_output_projection(&query)?;
        let output_projection_is_final = output_projection == query.projection;
        let stream = engine
            .join_parquet_batches(JoinParquetRequest {
                left_path: query.path.clone(),
                right_path: join.right.path,
                batch_size,
                left_keys: join.left_keys,
                right_keys: join.right_keys,
                left_prefix: join.left_alias.clone(),
                right_prefix: join.right_alias.clone(),
                left_projection: join_plan.left_projection,
                right_projection: join_plan.right_projection,
                left_filter: join_plan.left_filter,
                right_filter: combine_filter_options(
                    join_plan.right_filter,
                    join.right_filter.clone(),
                ),
                output_projection,
                join_memory_limit_bytes: join_memory_limit_bytes(options),
                join_algorithm: JoinAlgorithm::Auto,
                join_type: join.join_type,
            })
            .await?;
        if is_aggregate {
            let stream = apply_output_filter_stream(stream, query.filter.clone());
            let stream: SendableBatchStream =
                if let Some(expression_filter) = query.expression_filter.as_ref() {
                    Box::new(MemoryExec::new(apply_output_join_expression_filter(
                        collect_batches(stream)?,
                        expression_filter,
                        &[join.left_alias.as_str(), join.right_alias.as_str()],
                    )?))
                    .execute()?
                } else {
                    stream
                };
            let metrics = collect_aggregates_with_optional_expression_views(
                stream,
                2,
                &group_by,
                &aggregates,
                &query.filtered_aggregates,
                &query.aggregate_expressions,
            )?;
            let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
            batches = apply_output_filter(batches, query.having.as_ref())?;
            let has_output_expressions = projection_requires_expression_path(&query.expressions);
            if has_output_expressions {
                batches = apply_output_expression_projection(batches, &query.expressions)?;
            }
            batches = apply_aggregate_output_order_limit(
                batches,
                query.order_by.as_ref(),
                query.limit,
                query.offset,
                &metrics,
                &group_by,
            )?;
            if !has_output_expressions {
                batches = rename_output_batches(batches, &query.aliases)?;
            }
            return Ok(QueryOutput::Aggregate { metrics, batches });
        }
        let mut batches = collect_batches(stream)?;
        batches = apply_output_filter(batches, query.filter.as_ref())?;
        if let Some(expression_filter) = query.expression_filter.as_ref() {
            batches = apply_output_join_expression_filter(
                batches,
                expression_filter,
                &[join.left_alias.as_str(), join.right_alias.as_str()],
            )?;
        }
        if projection_requires_expression {
            if query.distinct {
                batches = apply_output_expression_projection(batches, &query.expressions)?;
                batches = apply_output_distinct(batches, true)?;
                batches = apply_output_order_limit(
                    batches,
                    query.order_by.as_ref(),
                    query.limit,
                    query.offset,
                )?;
            } else {
                batches = apply_output_expression_projection_order_limit(
                    batches,
                    &query.expressions,
                    query.order_by.as_ref(),
                    query.limit,
                    query.offset,
                )?;
            }
        } else {
            if query.distinct {
                if !output_projection_is_final && query.qualified_wildcards.is_empty() {
                    batches = apply_output_projection(batches, &query.projection)?;
                }
                batches = apply_output_distinct(batches, true)?;
                batches = apply_output_order_limit(
                    batches,
                    query.order_by.as_ref(),
                    query.limit,
                    query.offset,
                )?;
            } else {
                batches = apply_output_order_limit(
                    batches,
                    query.order_by.as_ref(),
                    query.limit,
                    query.offset,
                )?;
                if !output_projection_is_final && query.qualified_wildcards.is_empty() {
                    batches = apply_output_projection(batches, &query.projection)?;
                }
            }
        }
        if !query.qualified_wildcards.is_empty() {
            batches = apply_qualified_wildcard_projection(
                batches,
                &query.qualified_wildcards,
                &query.projection,
            )?;
        }
        if !projection_requires_expression {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(QueryOutput::Scan { batches });
    }

    if query.is_aggregate() {
        let aggregates = query.aggregates.clone();
        let group_by = query.group_by.clone();
        let metrics = if let Some(metrics) =
            try_collect_direct_monotonic_count_distinct(engine, &query, batch_size)?
        {
            metrics
        } else if !query.aggregate_expressions.is_empty() || !query.filtered_aggregates.is_empty() {
            if let Some(metrics) = try_collect_filtered_decimal_product_sum_scan_fold(
                engine,
                query.path.clone(),
                batch_size,
                query.filter.clone(),
                &aggregates,
                &query.aggregate_expressions,
            )
            .await?
            {
                metrics
            } else if let Some(metrics) =
                try_collect_expression_aggregate_fused_dictionary_selected(
                    engine,
                    query.path.clone(),
                    batch_size,
                    query.filter.clone(),
                    &group_by,
                    &aggregates,
                    &query.aggregate_expressions,
                    query.order_by.is_some(),
                    expression_aggregate_output_limit(
                        &group_by,
                        query.order_by.as_ref(),
                        query.limit,
                        query.offset,
                    ),
                )
                .await?
            {
                metrics
            } else if let Some(metrics) = try_collect_expression_aggregate_late_materialized(
                engine,
                query.path.clone(),
                batch_size,
                query.filter.clone(),
                &group_by,
                &aggregates,
                &query.aggregate_expressions,
                query.order_by.is_some(),
                expression_aggregate_output_limit(
                    &group_by,
                    query.order_by.as_ref(),
                    query.limit,
                    query.offset,
                ),
            )
            .await?
            {
                metrics
            } else if let Some(metrics) = try_collect_expression_aggregate_scan_fold(
                engine,
                query.path.clone(),
                batch_size,
                query.projection.clone(),
                query.filter.clone(),
                &group_by,
                &aggregates,
                &query.aggregate_expressions,
                query.order_by.is_some(),
                expression_aggregate_output_limit(
                    &group_by,
                    query.order_by.as_ref(),
                    query.limit,
                    query.offset,
                ),
            )
            .await?
            {
                metrics
            } else if let Some(metrics) = try_collect_expression_aggregate_row_group_map(
                engine,
                query.path.clone(),
                batch_size,
                query.projection.clone(),
                query.filter.clone(),
                &group_by,
                &aggregates,
                &query.aggregate_expressions,
            )
            .await?
            {
                metrics
            } else {
                let filtered_aggregates = simplify_filtered_aggregates_with_parquet_stats(
                    engine,
                    &query.path,
                    &query.filtered_aggregates,
                )?;
                let stream = engine
                    .scan_parquet_batches(
                        query.path,
                        batch_size,
                        None,
                        query.projection.clone(),
                        query.filter,
                    )
                    .await?;
                collect_aggregates_with_optional_expression_views(
                    stream,
                    1,
                    &group_by,
                    &aggregates,
                    &filtered_aggregates,
                    &query.aggregate_expressions,
                )?
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
        batches = apply_aggregate_output_order_limit(
            batches,
            query.order_by.as_ref(),
            query.limit,
            query.offset,
            &metrics,
            &group_by,
        )?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(QueryOutput::Aggregate { metrics, batches });
    }

    if query.distinct
        && let Some(mut batches) = try_execute_direct_distinct_scan(
            engine,
            DirectDistinctScan {
                path: query.path.clone(),
                projection: query.projection.clone(),
                aliases: query.aliases.clone(),
                filter: query.filter.clone(),
            },
            batch_size,
        )?
    {
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        return Ok(QueryOutput::Scan { batches });
    }

    if let Some(batches) =
        try_execute_monotonic_row_group_order_limit_scan(engine, &query, batch_size).await?
    {
        let batches = rename_output_batches(batches, &query.aliases)?;
        return Ok(QueryOutput::Scan { batches });
    }

    if !query.distinct
        && monotonic_order_limit_scan_enabled()
        && query.limit.is_some()
        && Path::new(&query.path).exists()
        && let Some(column) = monotonic_stream_limit_column(query.order_by.as_ref())
        && engine
            .parquet_row_groups_monotonic_by_column(query.path.clone(), &column)
            .await?
        && engine
            .parquet_column_monotonic_by_scan(query.path.clone(), &column, batch_size)
            .await?
    {
        let stream = engine
            .scan_parquet_filtered_batches_preserve_order(
                query.path.clone(),
                batch_size,
                query.projection.clone(),
                query.filter.clone(),
            )
            .await?;
        let batches = collect_ordered_stream_limit_batches(stream, query.limit, query.offset)?;
        let batches = rename_output_batches(batches, &query.aliases)?;
        return Ok(QueryOutput::Scan { batches });
    }

    let post_scan_order_by =
        if prefer_post_scan_primitive_desc_topk(engine, &query, query.order_by.as_ref())? {
            query.order_by.clone()
        } else {
            None
        };
    let stream = if query.distinct {
        engine
            .scan_parquet_distinct_batches(
                query.path,
                batch_size,
                scan_limit_with_offset(query.limit, query.offset)?,
                query.projection,
                query.filter,
                query.order_by,
            )
            .await?
    } else if post_scan_order_by.is_some() {
        engine
            .scan_parquet_batches(query.path, batch_size, None, query.projection, query.filter)
            .await?
    } else if let Some(order_by) = query.order_by {
        engine
            .scan_parquet_ordered_batches_by(
                query.path,
                batch_size,
                scan_limit_with_offset(query.limit, query.offset)?,
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
                scan_limit_with_offset(query.limit, query.offset)?,
                query.projection,
                query.filter,
            )
            .await?
    };
    let batches = apply_output_order_limit(
        collect_batches(stream)?,
        post_scan_order_by.as_ref(),
        query.limit,
        query.offset,
    )?;
    let batches = rename_output_batches(batches, &query.aliases)?;
    Ok(QueryOutput::Scan { batches })
}

async fn try_execute_monotonic_row_group_order_limit_scan(
    engine: &DodamEngine,
    query: &SqlQuery,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if query.distinct || query.expression_filter.is_some() || query.limit.is_none() {
        return Ok(None);
    }
    if !Path::new(&query.path).exists() {
        return Ok(None);
    }
    let Some(order_by) = query.order_by.as_ref() else {
        return Ok(None);
    };
    let Some(order_column) = monotonic_stream_limit_column(Some(order_by)) else {
        return Ok(None);
    };
    if !monotonic_order_limit_scan_enabled()
        || !monotonic_row_group_order_limit_scan_enabled()
        || !engine
            .parquet_row_groups_monotonic_by_column(query.path.clone(), &order_column)
            .await?
    {
        return Ok(None);
    }

    let mut scan_projection = scan_projection(&query.projection, query.filter.as_ref());
    add_projection_column_once(&mut scan_projection, order_column.clone());
    let row_groups = engine.parquet_row_group_count(&query.path)?;
    let mut output = Vec::new();
    let mut order_state = MonotonicOrderState::default();
    let mut limiter = OrderedLimitCollector::new(query.limit, query.offset);

    for row_group in 0..row_groups {
        let batches = engine
            .scan_parquet_row_group_batches(
                query.path.clone(),
                batch_size,
                scan_projection.clone(),
                vec![row_group],
            )
            .await?;
        for batch in batches {
            let batch = if let Some(filter) = query.filter.as_ref() {
                filter_batch(batch, filter)?
            } else {
                batch
            };
            if batch.num_rows() == 0 {
                continue;
            }
            if !order_state.consume_batch(&batch, &order_column)? {
                return Ok(None);
            }
            limiter.push_batch(batch, &mut output);
        }
        if limiter.is_complete() {
            let output = apply_output_projection(output, &query.projection)?;
            return Ok(Some(output));
        }
    }

    let output = apply_output_projection(output, &query.projection)?;
    Ok(Some(output))
}

fn monotonic_row_group_order_limit_scan_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_MONOTONIC_ROW_GROUP_ORDER_LIMIT_SCAN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn prefer_post_scan_primitive_desc_topk(
    engine: &DodamEngine,
    query: &SqlQuery,
    order_by: Option<&SortKey>,
) -> Result<bool> {
    if query.distinct
        || query.limit.is_none()
        || query.offset != 0
        || !Path::new(&query.path).exists()
    {
        return Ok(false);
    }
    let Some(order_by) = order_by else {
        return Ok(false);
    };
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(false);
    };
    if !sort.descending || sort.nulls_first {
        return Ok(false);
    }
    let Projection::Columns(columns) = &query.projection else {
        return Ok(false);
    };
    if !columns.iter().any(|column| column == &sort.column) {
        return Ok(false);
    }
    let Some(column_types) = engine
        .parquet_direct_primitive_column_types(&query.path, std::slice::from_ref(&sort.column))?
    else {
        return Ok(false);
    };
    Ok(
        choose_primitive_order_limit_strategy(PrimitiveOrderLimitCostInput {
            has_limit: query.limit.is_some(),
            offset: query.offset,
            sort_keys: order_by.expressions.len(),
            descending: sort.descending,
            nulls_first: sort.nulls_first,
            sort_key_projected: true,
            sort_key_is_i64: matches!(column_types.as_slice(), [DirectPrimitiveColumnType::I64]),
        }) == PrimitiveOrderLimitStrategy::PostScanTopK,
    )
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
        SqlExpr::BinaryOp { left, right, .. } => {
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
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            scalar_predicate_side_requires_expression(expr)
                || scalar_predicate_side_requires_expression(low)
                || scalar_predicate_side_requires_expression(high)
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            scalar_predicate_side_requires_expression(expr)
                || scalar_predicate_side_requires_expression(pattern)
        }
        _ => false,
    }
}

fn scalar_predicate_side_requires_expression(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Identifier(_) => false,
        SqlExpr::CompoundIdentifier(parts) => parts.len() > 1,
        SqlExpr::Function(_)
        | SqlExpr::Substring { .. }
        | SqlExpr::Cast { .. }
        | SqlExpr::Case { .. }
        | SqlExpr::CompoundFieldAccess { .. } => true,
        _ => sql_literal_value(expr).is_err(),
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
            if let Some((column, _)) = parse_struct_field_access(expr, table_alias)? {
                add_column_once(columns, column);
            } else {
                add_column_once(columns, sql_column_name(expr, table_alias)?);
            }
        }
        SqlExpr::CompoundFieldAccess { .. } => {
            for column in
                scalar_expression_columns(&parse_scalar_sql_expression(expr, table_alias)?)
            {
                add_column_once(columns, column);
            }
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
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
            collect_predicate_expression_columns(pattern, table_alias, columns)?;
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
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            expr_contains_scalar_subquery(expr) || expr_contains_scalar_subquery(pattern)
        }
        _ => false,
    }
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

fn should_use_i64_set_row_filter(
    default_enabled: bool,
    disable_env: &str,
    enable_env: Option<&str>,
    key_count: usize,
    projected_columns: usize,
) -> bool {
    if env_flag_enabled(disable_env) {
        return false;
    }
    let enabled = enable_env.is_some_and(env_flag_enabled) || default_enabled;
    enabled
        && key_count > 0
        && key_count <= i64_set_row_filter_max_keys()
        && projected_columns >= i64_set_row_filter_min_projected_columns()
}

fn should_use_i64_set_row_filter_for_keys(
    default_enabled: bool,
    disable_env: &str,
    enable_env: Option<&str>,
    keys: &HashSet<i64>,
    projected_columns: usize,
) -> bool {
    let forced_enabled = enable_env.is_some_and(env_flag_enabled);
    if !should_use_i64_set_row_filter(
        default_enabled,
        disable_env,
        enable_env,
        keys.len(),
        projected_columns,
    ) {
        return false;
    }
    if forced_enabled {
        return true;
    }
    let Some((min_key, max_key)) = raw_i64_key_range(keys.iter().copied()) else {
        return false;
    };
    let Some(width) = max_key
        .checked_sub(min_key)
        .and_then(|width| width.checked_add(1))
        .and_then(|width| usize::try_from(width).ok())
    else {
        return false;
    };
    if width == 0 {
        return false;
    }
    let density = keys.len() as f64 / width as f64;
    density <= i64_set_row_filter_max_density()
        || keys.len() <= i64_set_row_filter_always_allow_keys()
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn i64_set_row_filter_max_keys() -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_MAX_KEYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1_000_000)
}

fn i64_set_row_filter_min_projected_columns() -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_MIN_PROJECTED_COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

fn i64_set_row_filter_max_density() -> f64 {
    std::env::var("DODAM_I64_SET_ROW_FILTER_MAX_DENSITY")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.35)
}

fn i64_set_row_filter_always_allow_keys() -> usize {
    std::env::var("DODAM_I64_SET_ROW_FILTER_ALWAYS_ALLOW_KEYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096)
}

fn raw_i64_key_range(keys: impl IntoIterator<Item = i64>) -> Option<(i64, i64)> {
    let mut iter = keys.into_iter();
    let first = iter.next()?;
    let mut min_key = first;
    let mut max_key = first;
    for key in iter {
        min_key = min_key.min(key);
        max_key = max_key.max(key);
    }
    Some((min_key, max_key))
}

fn projection_column_count(projection: &Projection) -> usize {
    match projection {
        Projection::All => usize::MAX,
        Projection::Columns(columns) => columns.len(),
    }
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

fn numeric_i64_equality_literal(conjuncts: &[SqlExpr], column: &str) -> Result<Option<i64>> {
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if *op != BinaryOperator::Eq {
            continue;
        }
        if sql_expr_column_matches(left, column) {
            if let LiteralValue::Int64(value) = sql_literal_value(right)? {
                return Ok(Some(value));
            }
        } else if sql_expr_column_matches(right, column)
            && let LiteralValue::Int64(value) = sql_literal_value(left)?
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

fn date32_to_ymd_string(days: i32) -> Result<String> {
    let (year, month, day) = civil_from_days(i64::from(days))?;
    Ok(format!("{year:04}-{month:02}-{day:02}"))
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

fn scaled_f64_to_i128(value: f64, scale: f64) -> i128 {
    (value * scale).round() as i128
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

fn single_f64_aggregate_output(name: String, value: Option<f64>) -> Result<QueryOutput> {
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

fn bytes_string_parts<'a>(offsets: &[i32], data: &'a [u8], row: usize) -> &'a [u8] {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    &data[start..end]
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
            if predicate_requires_expression_path(expr) {
                return Ok(None);
            }
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
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            let Some(expr) = Box::pin(parse_filter_with_subqueries(
                engine,
                expr,
                aliases,
                table_alias,
                allow_aggregates,
                batch_size,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(Expr::Not(Box::new(expr))))
        }
        SqlExpr::IsNull(inner) | SqlExpr::IsNotNull(inner) => {
            if let Ok(value) = sql_literal_value(inner) {
                let is_null = matches!(value, LiteralValue::Null);
                let is_not_null = matches!(expr, SqlExpr::IsNotNull(_));
                return Ok(Some(Expr::Boolean(Some(if is_not_null {
                    !is_null
                } else {
                    is_null
                }))));
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
            if predicate_requires_expression_path(expr) {
                return Ok(None);
            }
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
        _ if predicate_requires_expression_path(expr) => Ok(None),
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
    options: SqlExecutionOptions,
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

    let left = materialize_join_relation(engine, &table.relation, batch_size, options).await?;
    let right = materialize_join_relation(engine, &join.relation, batch_size, options).await?;
    let left_alias = left.alias.clone();
    let right_alias = right.alias.clone();
    let (join_type, left_keys, right_keys, right_filter) =
        parse_join_condition(join, &left_alias, &right_alias)?;
    let derived_join_graph =
        build_logical_materialized_join_graph(&left, &right, &left_keys, &right_keys)?;
    let derived_join_plan = derived_join_graph.choose_best_plan();
    log_multi_input_join_optimizer_plan(
        "derived_join_order",
        &derived_join_graph,
        derived_join_plan.as_ref(),
    );
    let exec_left_alias = left_alias.clone();
    let exec_right_alias = right_alias.clone();
    let output_aliases = if join_type == JoinType::Semi {
        vec![left_alias.as_str()]
    } else {
        vec![left_alias.as_str(), right_alias.as_str()]
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let (filter, expression_filter) = parse_join_filter_plan(
        select.selection.as_ref(),
        &projection.aliases,
        &output_aliases,
        false,
    )?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
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
        choose_materialized_join_build_side(&derived_join_graph).unwrap_or(JoinBuildSide::Right),
        join_type,
        Projection::All,
    ))
    .execute()?;
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, filter.as_ref())?;
    if let Some(expression_filter) = expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(batches, expression_filter, &output_aliases)?;
    }
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection_order_limit(
            batches,
            &projection.expressions,
            order_by.as_ref(),
            limit,
            0,
        )?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    if !projection_requires_expression {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    }
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
    options: SqlExecutionOptions,
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
            let output = Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await?;
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

fn build_logical_materialized_join_graph(
    left: &MaterializedJoinRelation,
    right: &MaterializedJoinRelation,
    left_keys: &[String],
    right_keys: &[String],
) -> Result<LogicalJoinGraph> {
    let left_rows = record_batch_rows(&left.batches).max(1);
    let right_rows = record_batch_rows(&right.batches).max(1);
    let tables = vec![
        logical_materialized_table_stats(&left.batches, left_rows, left_keys)?,
        logical_materialized_table_stats(&right.batches, right_rows, right_keys)?,
    ];
    let edges = left_keys
        .iter()
        .zip(right_keys)
        .map(|(left_key, right_key)| LogicalJoinEdge {
            left: 0,
            left_key: left_key.clone(),
            right: 1,
            right_key: right_key.clone(),
        })
        .collect();
    Ok(LogicalJoinGraph { tables, edges })
}

fn logical_materialized_table_stats(
    batches: &[RecordBatch],
    rows: usize,
    keys: &[String],
) -> Result<LogicalJoinTableStats> {
    let mut key_ndv = HashMap::new();
    let mut column_ranges = HashMap::new();
    for key in keys {
        key_ndv.insert(
            key.clone(),
            sampled_key_ndv(batches, std::slice::from_ref(key), 100_000)? as u128,
        );
        if let Some(range) = primitive_column_range_stats(batches, key)? {
            column_ranges.insert(key.clone(), range);
        }
    }
    Ok(LogicalJoinTableStats {
        base_rows: rows.max(1) as u128,
        rows: rows.max(1) as u128,
        row_width: estimated_batches_row_width(batches).max(1),
        key_ndv,
        column_ranges,
    })
}

fn choose_materialized_join_build_side(join_graph: &LogicalJoinGraph) -> Option<JoinBuildSide> {
    let left = join_graph.tables.first()?;
    let right = join_graph.tables.get(1)?;
    let left_cost = left.rows.saturating_mul(left.row_width.max(1));
    let right_cost = right.rows.saturating_mul(right.row_width.max(1));
    Some(if left_cost <= right_cost {
        JoinBuildSide::Left
    } else {
        JoinBuildSide::Right
    })
}

async fn execute_parsed_join_query(
    engine: &DodamEngine,
    query: SqlQuery,
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<Option<QueryOutput>> {
    let Some(join) = query.join.clone() else {
        return Ok(None);
    };
    let is_aggregate = query.is_aggregate();
    let aggregates = query.aggregates.clone();
    let group_by = query.group_by.clone();
    let join_input_projection = if query.qualified_wildcards.is_empty() {
        join_input_projection_with_expression_filter(&query)?
    } else {
        Projection::All
    };
    let join_plan = plan_join_inputs(
        &join_input_projection,
        query.filter.as_ref(),
        query.order_by.as_ref(),
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    if let Some(join_graph) = build_logical_explicit_join_graph(engine, &query, &join, &join_plan)?
    {
        log_multi_input_join_optimizer_plan(
            "explicit_join_order",
            &join_graph,
            join_graph.choose_best_plan().as_ref(),
        );
    }
    if is_aggregate
        && let Some(output) = try_execute_join_coalesce_count_sum_aggregate(
            engine, &query, &join, &join_plan, batch_size,
        )
        .await?
    {
        return Ok(Some(output));
    }
    let output_projection = pushed_join_output_projection(&query)?;
    let output_projection_is_final = output_projection == query.projection;
    let stream = engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: query.path.clone(),
            right_path: join.right.path,
            batch_size,
            left_keys: join.left_keys,
            right_keys: join.right_keys,
            left_prefix: join.left_alias.clone(),
            right_prefix: join.right_alias.clone(),
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
            output_projection,
            join_memory_limit_bytes: join_memory_limit_bytes(options),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: join.join_type,
        })
        .await?;
    if is_aggregate {
        let stream = apply_output_filter_stream(stream, query.filter.clone());
        let stream: SendableBatchStream =
            if let Some(expression_filter) = query.expression_filter.as_ref() {
                Box::new(MemoryExec::new(apply_output_join_expression_filter(
                    collect_batches(stream)?,
                    expression_filter,
                    &[join.left_alias.as_str(), join.right_alias.as_str()],
                )?))
                .execute()?
            } else {
                stream
            };
        let metrics = collect_aggregates_with_optional_expression_views(
            stream,
            2,
            &group_by,
            &aggregates,
            &query.filtered_aggregates,
            &query.aggregate_expressions,
        )?;
        let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
        batches = apply_output_filter(batches, query.having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&query.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
        }
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, query.filter.as_ref())?;
    if let Some(expression_filter) = query.expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(
            batches,
            expression_filter,
            &[join.left_alias.as_str(), join.right_alias.as_str()],
        )?;
    }
    let projection_requires_expression = projection_requires_expression_path(&query.expressions);
    if projection_requires_expression {
        batches = apply_output_expression_projection_order_limit(
            batches,
            &query.expressions,
            query.order_by.as_ref(),
            query.limit,
            query.offset,
        )?;
    } else {
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        if !output_projection_is_final && query.qualified_wildcards.is_empty() {
            batches = apply_output_projection(batches, &query.projection)?;
        }
    }
    if query.distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    if !query.qualified_wildcards.is_empty() {
        batches = apply_qualified_wildcard_projection(
            batches,
            &query.qualified_wildcards,
            &query.projection,
        )?;
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &query.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

fn build_logical_explicit_join_graph(
    engine: &DodamEngine,
    query: &SqlQuery,
    join: &SqlJoin,
    join_plan: &JoinInputPlan,
) -> Result<Option<LogicalJoinGraph>> {
    if join.left_keys.is_empty() || join.left_keys.len() != join.right_keys.len() {
        return Ok(None);
    }
    let left_rows = engine
        .parquet_total_row_count(&query.path)
        .unwrap_or(0)
        .max(1) as u128;
    let right_rows = engine
        .parquet_total_row_count(&join.right.path)
        .unwrap_or(0)
        .max(1) as u128;
    let left_key_ndv = join
        .left_keys
        .iter()
        .map(|key| (key.clone(), left_rows))
        .collect::<HashMap<_, _>>();
    let right_key_ndv = join
        .right_keys
        .iter()
        .map(|key| (key.clone(), right_rows))
        .collect::<HashMap<_, _>>();
    let edges = join
        .left_keys
        .iter()
        .zip(&join.right_keys)
        .map(|(left_key, right_key)| LogicalJoinEdge {
            left: 0,
            left_key: left_key.clone(),
            right: 1,
            right_key: right_key.clone(),
        })
        .collect();
    Ok(Some(LogicalJoinGraph {
        tables: vec![
            LogicalJoinTableStats {
                base_rows: left_rows,
                rows: left_rows,
                row_width: estimated_projection_width(&join_plan.left_projection),
                key_ndv: left_key_ndv,
                column_ranges: HashMap::new(),
            },
            LogicalJoinTableStats {
                base_rows: right_rows,
                rows: right_rows,
                row_width: estimated_projection_width(&join_plan.right_projection),
                key_ndv: right_key_ndv,
                column_ranges: HashMap::new(),
            },
        ],
        edges,
    }))
}

fn estimated_projection_width(projection: &Projection) -> u128 {
    match projection {
        Projection::Columns(columns) => (columns.len() as u128).saturating_mul(16).max(1),
        Projection::All => 128,
    }
}

async fn try_execute_derived_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
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
    let inner_has_materializable_subquery = match subquery.body.as_ref() {
        SetExpr::Select(inner_select) => inner_select
            .selection
            .as_ref()
            .is_some_and(expr_contains_materializable_subquery),
        _ => false,
    };
    let parsed_inner = match parse_query(subquery) {
        Ok(query) => Some(query),
        Err(DodamError::UnsupportedSql(_))
        | Err(DodamError::UnknownColumn(_))
        | Err(DodamError::UnknownTableQualifier(_)) => None,
        Err(error) => return Err(error),
    };
    let inner_output = if let Some(parsed_inner) = parsed_inner {
        if !inner_has_materializable_subquery
            && let Some(output) =
                execute_parsed_join_query(engine, parsed_inner, batch_size, options).await?
        {
            output
        } else {
            Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await?
        }
    } else {
        Box::pin(execute_sql_with_options(
            engine,
            &subquery.to_string(),
            batch_size,
            options,
        ))
        .await?
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
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        Some(&alias),
    )?;
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
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
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_execute_multi_comma_join_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
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
    let Some((tables, mut conjuncts)) = parse_multi_input_join_table_refs_and_conjuncts(select)?
    else {
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
    if let Some(selection) = select.selection.as_ref() {
        collect_sql_and_conjuncts(selection, &mut conjuncts);
    }
    if conjuncts.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "multi-input join requires an equality predicate".to_string(),
        ));
    }

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
    let rewritten_having = if let Some(expr) = select.having.as_ref() {
        Some(
            Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                engine,
                expr.clone(),
                batch_size,
            ))
            .await?,
        )
    } else {
        None
    };
    let having = rewritten_having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &alias_refs, true))
        .transpose()?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &alias_refs,
    )?;
    let limit = parse_limit(query)?;
    let memory_limit_bytes = join_memory_limit_bytes(options);
    let lookup_fusion_plan = if join_aggregate_lookup_fusion_disabled() {
        None
    } else {
        plan_join_aggregate_lookup_fusion(
            tables.len(),
            &aliases,
            &alias_refs,
            &conjuncts,
            &used_conjuncts,
            &group_by,
            &projection,
            distinct,
            having.as_ref(),
        )?
    };
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
    if let Some(lookup_fusion_plan) = lookup_fusion_plan.as_ref() {
        let metadata_graph = build_logical_multi_join_graph_from_metadata(
            engine,
            &tables,
            &aliases,
            &alias_refs,
            &conjuncts,
            &scan_projections,
        )?;
        let graph_plan = metadata_graph.choose_best_plan();
        log_multi_input_join_optimizer_plan(
            "join_aggregate_lookup_fusion_graph",
            &metadata_graph,
            graph_plan.as_ref(),
        );
        if choose_join_aggregate_lookup_fusion(
            lookup_fusion_plan,
            &metadata_graph,
            graph_plan.as_ref(),
        ) {
            if let Some(output) = execute_join_aggregate_lookup_fusion(
                engine,
                &tables,
                &scan_filters,
                &group_by,
                &projection,
                order_by.as_ref(),
                limit,
                batch_size,
                lookup_fusion_plan,
            )
            .await?
            {
                return Ok(Some(output));
            }
        }
    }
    let mut scanned = Vec::with_capacity(tables.len());
    let mut row_counts = Vec::with_capacity(tables.len());
    let mut base_row_counts = Vec::with_capacity(tables.len());
    for index in 0..tables.len() {
        base_row_counts.push(
            engine
                .parquet_total_row_count(&tables[index].path)
                .unwrap_or_else(|_| 0),
        );
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
    let join_graph = build_logical_comma_join_graph(
        &scanned,
        &row_counts,
        &base_row_counts,
        &aliases,
        &alias_refs,
        &conjuncts,
    )?;
    let join_plan = join_graph.choose_best_plan();
    log_multi_input_join_optimizer_plan("multi_input_join_order", &join_graph, join_plan.as_ref());
    let current = if let Some(tree) =
        choose_bushy_multi_input_join_execution_tree(&join_graph, join_plan.as_ref())
    {
        execute_bushy_comma_join_tree(
            &tree,
            &mut scanned,
            &row_counts,
            &aliases,
            &alias_refs,
            &conjuncts,
            &mut used_conjuncts,
            &final_columns,
            memory_limit_bytes,
        )?
        .batches
    } else {
        execute_left_deep_comma_join(
            scanned,
            &row_counts,
            &aliases,
            &alias_refs,
            &conjuncts,
            &mut used_conjuncts,
            &final_columns,
            &join_graph,
            join_plan.as_ref(),
            memory_limit_bytes,
        )?
    };

    let residual = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (!used_conjuncts[index]).then_some(conjunct))
        .collect::<Vec<_>>();
    let residual = combine_sql_and_conjuncts(residual);
    let (filter_residual, subquery_residual) = split_subquery_residual(residual);
    let (filter, expression_filter) = parse_join_filter_plan(
        filter_residual.as_ref(),
        &projection.aliases,
        &alias_refs,
        false,
    )?;

    let mut batches = apply_output_filter(current, filter.as_ref())?;
    if let Some(expression_filter) = expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(batches, expression_filter, &alias_refs)?;
    }
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection_order_limit(
            batches,
            &projection.expressions,
            order_by.as_ref(),
            limit,
            0,
        )?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    if !projection_requires_expression {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

pub async fn try_execute_sql_streaming(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
) -> Result<Option<SendableBatchStream>> {
    if explain_sql(engine, sql, batch_size).await?.is_some() {
        return Ok(None);
    }
    let Some(request) = plan_direct_join_sink_request_relaxed(sql, batch_size)? else {
        return Ok(None);
    };
    engine.join_parquet_batches(request).await.map(Some)
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
    if let Some(metrics) =
        try_execute_set_operation_sql_to_sink(engine, sql, batch_size, sink).await?
    {
        return Ok(Some(metrics));
    }
    let Some(request) = plan_direct_join_sink_request_relaxed(sql, batch_size)? else {
        return Ok(None);
    };
    let plan = engine.plan_parquet_join(request).await?;
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
    if options.allow_direct_or_streaming && explain_sql(engine, sql, batch_size).await?.is_none() {
        let direct_started = Instant::now();
        if let Some(metrics) =
            try_execute_set_operation_sql_to_sink(engine, sql, batch_size, sink.record_batch_sink())
                .await?
        {
            profile.direct_sink = Some(direct_started.elapsed());
            profile.scan_plan_metrics = Some(metrics);
            return Ok(profile);
        }
        if let Some(request) = plan_direct_join_sink_request_relaxed(sql, batch_size)? {
            let plan = engine.plan_parquet_join(request).await?;
            let metrics = engine.write_join_plan_to_sink(plan, sink.record_batch_sink())?;
            profile.direct_sink = Some(direct_started.elapsed());
            profile.scan_plan_metrics = Some(metrics);
            return Ok(profile);
        }
        profile.direct_sink = Some(direct_started.elapsed());
        profile.streaming = Some(Duration::ZERO);
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

fn default_join_memory_limit_bytes() -> u64 {
    std::env::var("DODAM_JOIN_MEMORY_LIMIT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128 * 1024 * 1024)
}

fn join_memory_limit_bytes(options: SqlExecutionOptions) -> u64 {
    options
        .join_memory_limit_bytes
        .filter(|value| *value > 0)
        .unwrap_or_else(default_join_memory_limit_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int32_batch(column: &str, values: Vec<i32>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                column,
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(values))],
        )
        .expect("test batch")
    }

    fn two_int32_batch(
        left_column: &str,
        left_values: Vec<i32>,
        right_column: &str,
        right_values: Vec<i32>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(left_column, DataType::Int32, false),
                Field::new(right_column, DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from(left_values)),
                Arc::new(Int32Array::from(right_values)),
            ],
        )
        .expect("test batch")
    }

    #[test]
    fn logical_multi_join_graph_uses_shared_stats_builder() {
        let scanned = vec![
            Some(vec![int32_batch("id", vec![1, 2, 3, 4])]),
            Some(vec![two_int32_batch(
                "id",
                vec![1, 2],
                "region_id",
                vec![1, 1],
            )]),
            Some(vec![int32_batch("region_id", vec![1])]),
        ];
        let row_counts = vec![4, 2, 1];
        let base_row_counts = vec![10, 2, 1];
        let key_columns = vec![
            vec!["id".to_string()],
            vec!["id".to_string(), "region_id".to_string()],
            vec!["region_id".to_string()],
        ];
        let graph = build_logical_multi_join_graph(
            &scanned,
            &row_counts,
            &base_row_counts,
            &key_columns,
            vec![
                LogicalJoinEdge {
                    left: 0,
                    left_key: "id".to_string(),
                    right: 1,
                    right_key: "id".to_string(),
                },
                LogicalJoinEdge {
                    left: 1,
                    left_key: "region_id".to_string(),
                    right: 2,
                    right_key: "region_id".to_string(),
                },
            ],
        )
        .expect("graph");

        assert_eq!(graph.tables.len(), 3);
        assert_eq!(graph.tables[0].base_rows, 10);
        assert_eq!(graph.tables[0].rows, 4);
        assert_eq!(graph.tables[0].key_ndv.get("id"), Some(&4));
        assert!(graph.choose_best_plan().is_some());
        assert!(graph.choose_exhaustive_bushy_plan().is_some());
    }

    #[test]
    fn direct_join_sink_request_accepts_plain_projected_join() {
        let sql = "SELECT l.l_orderkey, o.o_orderdate \
                   FROM '/tmp/lineitem.parquet' l \
                   JOIN '/tmp/orders.parquet' o \
                   ON l.l_orderkey = o.o_orderkey";

        let request = plan_direct_join_sink_request(sql, 1024)
            .expect("query should parse")
            .expect("plain projected join should use direct sink");

        assert_eq!(request.left_keys, vec!["l_orderkey"]);
        assert_eq!(request.right_keys, vec!["o_orderkey"]);
        assert_eq!(
            request.left_projection,
            Projection::Columns(vec!["l_orderkey".to_string()])
        );
        assert_eq!(
            request.right_projection,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()])
        );
    }

    #[test]
    fn direct_join_sink_request_accepts_fully_pushed_side_filter() {
        let sql = "SELECT l.l_orderkey, o.o_orderdate \
                   FROM '/tmp/lineitem.parquet' l \
                   JOIN '/tmp/orders.parquet' o \
                   ON l.l_orderkey = o.o_orderkey \
                   WHERE o.o_orderstatus = 'O'";

        let request = plan_direct_join_sink_request(sql, 1024)
            .expect("query should parse")
            .expect("side-filtered projected join should use direct sink");

        assert_eq!(request.left_filter, None);
        assert_eq!(
            request.right_filter,
            Some(FilterExpr::new(Expr::Comparison(ComparisonExpr {
                column: "o_orderstatus".to_string(),
                op: ComparisonOp::Eq,
                value: LiteralValue::Utf8("O".to_string()),
            })))
        );
    }

    #[test]
    fn direct_join_sink_request_rejects_materialized_join_shapes() {
        for sql in [
            "SELECT l.l_orderkey, count(*) \
             FROM '/tmp/lineitem.parquet' l \
             JOIN '/tmp/orders.parquet' o \
             ON l.l_orderkey = o.o_orderkey \
             GROUP BY l.l_orderkey",
            "SELECT l.l_orderkey \
             FROM '/tmp/lineitem.parquet' l \
             JOIN '/tmp/orders.parquet' o \
             ON l.l_orderkey = o.o_orderkey \
             ORDER BY l.l_orderkey",
            "SELECT l.l_orderkey AS key \
             FROM '/tmp/lineitem.parquet' l \
             JOIN '/tmp/orders.parquet' o \
             ON l.l_orderkey = o.o_orderkey",
        ] {
            assert!(
                plan_direct_join_sink_request(sql, 1024)
                    .expect("query should parse")
                    .is_none(),
                "{sql}"
            );
        }
    }

    #[test]
    fn same_source_union_all_plan_accepts_disjoint_equality_filters() {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(
            &dialect,
            "SELECT id, bucket, value FROM '/tmp/facts.parquet' WHERE bucket = 1 \
             UNION ALL \
             SELECT id, bucket, value FROM '/tmp/facts.parquet' WHERE bucket = 7",
        )
        .expect("query should parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };

        let mut operands = Vec::new();
        assert!(
            collect_union_all_operand_queries(query.body.as_ref(), &mut operands)
                .expect("operands should parse")
        );
        let plan =
            same_source_disjoint_union_all_plan(&operands).expect("shared scan should be planned");

        assert_eq!(plan.path, PathBuf::from("/tmp/facts.parquet"));
        assert_eq!(
            plan.projection,
            Projection::Columns(vec![
                "id".to_string(),
                "bucket".to_string(),
                "value".to_string()
            ])
        );
        assert!(matches!(plan.filter.expr(), Expr::InList { .. }));
    }

    #[test]
    fn same_source_union_all_plan_accepts_disjoint_in_filters() {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(
            &dialect,
            "SELECT id, bucket, value FROM '/tmp/facts.parquet' WHERE bucket IN (1, 3) \
             UNION ALL \
             SELECT id, bucket, value FROM '/tmp/facts.parquet' WHERE bucket IN (7, 9)",
        )
        .expect("query should parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };

        let mut operands = Vec::new();
        assert!(
            collect_union_all_operand_queries(query.body.as_ref(), &mut operands)
                .expect("operands should parse")
        );
        let plan =
            same_source_disjoint_union_all_plan(&operands).expect("shared scan should be planned");

        let Expr::InList { values, .. } = plan.filter.expr() else {
            panic!("expected in-list filter");
        };
        assert_eq!(values.len(), 4);
    }

    #[test]
    fn same_source_union_all_plan_rejects_overlapping_in_filters() {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(
            &dialect,
            "SELECT id, bucket, value FROM '/tmp/facts.parquet' WHERE bucket IN (1, 3) \
             UNION ALL \
             SELECT id, bucket, value FROM '/tmp/facts.parquet' WHERE bucket IN (3, 7)",
        )
        .expect("query should parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };

        let mut operands = Vec::new();
        assert!(
            collect_union_all_operand_queries(query.body.as_ref(), &mut operands)
                .expect("operands should parse")
        );
        assert!(same_source_disjoint_union_all_plan(&operands).is_none());
        assert!(same_source_union_all_filter_scan_plan(&operands).is_some());
    }

    #[test]
    fn expression_pushdown_derives_coalesce_prefilter() {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(
            &dialect,
            "SELECT id FROM '/tmp/facts.parquet' WHERE COALESCE(bucket, 7) = 7",
        )
        .expect("query should parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };
        let SetExpr::Select(select) = query.body.as_ref() else {
            panic!("expected select");
        };
        let selection = select.selection.as_ref().expect("selection");
        let filter = safe_expression_pushdown_filter(selection, None, PredicateParserKind::Single)
            .expect("pushdown should parse")
            .expect("coalesce should produce a prefilter");

        assert_eq!(
            filter,
            FilterExpr::new(Expr::Or(
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "bucket".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Int64(7),
                })),
                Box::new(Expr::IsNull {
                    column: "bucket".to_string(),
                    negated: false,
                }),
            ))
        );
    }

    #[test]
    fn join_expression_filter_returns_side_prefilter_and_residual() {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(
            &dialect,
            "SELECT l.l_orderkey FROM '/tmp/lineitem.parquet' l \
             JOIN '/tmp/orders.parquet' o ON l.l_orderkey = o.o_orderkey \
             WHERE COALESCE(o.o_orderstatus, 'O') = 'O'",
        )
        .expect("query should parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };
        let SetExpr::Select(select) = query.body.as_ref() else {
            panic!("expected select");
        };
        let selection = select.selection.as_ref().expect("selection");
        let aliases = [("l_orderkey".to_string(), "l_orderkey".to_string())];
        let table_aliases = ["l", "o"];
        let (filter, residual) =
            parse_join_filter_plan(Some(selection), &aliases, &table_aliases, false)
                .expect("join filter should parse");

        assert!(residual.is_some());
        assert_eq!(
            filter,
            Some(FilterExpr::new(Expr::Or(
                Box::new(Expr::Comparison(ComparisonExpr {
                    column: "o.o_orderstatus".to_string(),
                    op: ComparisonOp::Eq,
                    value: LiteralValue::Utf8("O".to_string()),
                })),
                Box::new(Expr::IsNull {
                    column: "o.o_orderstatus".to_string(),
                    negated: false,
                }),
            )))
        );
    }
}
