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
mod derived_query;
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
mod rule_helpers;
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
use derived_query::*;
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
    PredicateParserKind, collect_predicate_expression_columns, expr_contains_scalar_subquery,
    function_arg_exprs, predicate_expression_columns, predicate_requires_expression_path,
    safe_expression_pushdown_filter, split_subquery_and_expression_filters,
    unqualified_column_matches_table_alias,
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
use rule_helpers::*;
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
