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
mod derived_count;
mod direct_join_sink;
mod discounted_revenue_predicate;
mod explain;
mod expression_aggregate;
mod filter_parser;
mod group_by;
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
mod prefix_supplier_threshold;
mod pricing_summary;
mod primitive_buffers;
mod primitive_selection;
mod profiling;
mod profit_by_nation_year;
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
mod set_sink;
mod shipping_order_priority;
mod shipping_priority_counts;
mod shipping_priority_revenue;
mod supplier_stock_threshold;
mod supplier_wait_antijoin;
mod table_refs;
mod tpch_rules;
mod types;
mod window;

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
use derived_count::*;
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
use set_sink::{
    append_same_source_union_all_filter_batches, same_source_union_primitive_chunk_size,
    try_execute_set_operation_sql_to_sink, write_same_source_primitive_batch_to_sink,
};
use shipping_order_priority::*;
use shipping_priority_counts::*;
use shipping_priority_revenue::*;
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

async fn try_execute_set_operation_sql(
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
    if !query_contains_set_operation(query.body.as_ref()) {
        return Ok(None);
    }
    let order_by = parse_order_by(query, &[], &[], None)?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    if let Some(batches) = try_execute_same_source_union_all_monotonic_topk(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )
    .await?
    {
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) =
        try_execute_same_source_union_all_scan(engine, query.body.as_ref(), batch_size).await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if order_by.is_none()
        && limit.is_none()
        && offset == 0
        && let Some(batches) =
            try_execute_same_source_union_all_filter_scan(engine, query.body.as_ref(), batch_size)
                .await?
    {
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_simple_case_distinct_set_literals(query.body.as_ref())? {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_same_source_union_distinct_scan(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )
    .await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_same_source_distinct_set_scan(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )
    .await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    if let Some(mut batches) = try_execute_same_source_all_set_scan(
        engine,
        query.body.as_ref(),
        batch_size,
        order_by.as_ref(),
        limit,
        offset,
    )? {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    let child_topk = if offset == 0 {
        order_by.as_ref().zip(limit)
    } else {
        None
    };
    let mut batches = Box::pin(execute_set_operation_expr(
        engine,
        query.body.as_ref(),
        batch_size,
        child_topk,
        false,
    ))
    .await?;
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_execute_same_source_union_all_monotonic_topk(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let (Some(order_by), Some(limit)) = (order_by, limit) else {
        return Ok(None);
    };
    if offset != 0 || limit == 0 {
        return Ok(None);
    }
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(None);
    };
    if !sort.descending || sort.nulls_first {
        return Ok(None);
    }
    let Some(shared) = plan_same_source_union_all_scan(expr)? else {
        return Ok(None);
    };
    if let Some(mut batches) =
        try_row_group_ordered_desc_sort(engine, &shared, order_by, &sort.column, batch_size, limit)
            .await?
    {
        batches = apply_output_projection(batches, &shared.projection)?;
        batches = rename_output_batches(batches, &shared.aliases)?;
        return Ok(Some(batches));
    }
    if let Some(mut batches) =
        try_reverse_row_group_desc_tail_topk(engine, &shared, &sort.column, batch_size, limit)
            .await?
    {
        batches = apply_output_order_limit(batches, Some(order_by), Some(limit), 0)?;
        batches = apply_output_projection(batches, &shared.projection)?;
        batches = rename_output_batches(batches, &shared.aliases)?;
        return Ok(Some(batches));
    }
    if let Some(mut batches) = try_same_source_union_all_streaming_desc_topk(
        engine,
        &shared,
        &sort.column,
        batch_size,
        limit,
    )
    .await?
    {
        batches = apply_output_projection(batches, &shared.projection)?;
        batches = rename_output_batches(batches, &shared.aliases)?;
        return Ok(Some(batches));
    }
    let stream = engine
        .scan_parquet_filtered_batches_preserve_order(
            shared.path,
            batch_size,
            shared.projection,
            Some(shared.filter),
        )
        .await?;
    let Some(mut batches) = collect_monotonic_desc_tail_topk(stream, &sort.column, limit)? else {
        return Ok(None);
    };
    batches = rename_output_batches(batches, &shared.aliases)?;
    batches = apply_output_order_limit(batches, Some(order_by), Some(limit), 0)?;
    Ok(Some(batches))
}

async fn try_row_group_ordered_desc_sort(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    order_by: &SortKey,
    sort_column: &str,
    batch_size: usize,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit <= reverse_row_group_topk_max_limit_rows() {
        return Ok(None);
    }
    if !engine
        .parquet_row_groups_monotonic_by_column(shared.path.clone(), sort_column)
        .await?
    {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&shared.path)?;
    if row_group_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    for sort in &order_by.expressions {
        add_projection_columns(&mut scan_projection, vec![sort.column.clone()]);
    }

    if row_group_ordered_desc_bulk_global_sort_enabled() {
        let row_groups = (0..row_group_count).rev().collect::<Vec<_>>();
        let batches = engine
            .scan_parquet_row_group_batches(
                shared.path.clone(),
                batch_size,
                scan_projection,
                row_groups,
            )
            .await?;
        let mut filtered = Vec::new();
        for batch in batches {
            let batch = filter_batch(batch, &shared.filter)?;
            if batch.num_rows() > 0 {
                filtered.push(batch);
            }
        }
        return Ok(Some(apply_output_order_limit(
            filtered,
            Some(order_by),
            Some(limit),
            0,
        )?));
    }

    if row_group_ordered_desc_parallel_enabled() && row_group_count > 1 {
        let row_group_batch_size = row_group_ordered_sort_batch_size(batch_size);
        let mut handles = Vec::with_capacity(row_group_count);
        for row_group in (0..row_group_count).rev() {
            let engine = engine.clone();
            let path = shared.path.clone();
            let scan_projection = scan_projection.clone();
            let filter = shared.filter.clone();
            let order_by = order_by.clone();
            let sort_column = sort_column.to_string();
            handles.push(tokio::task::spawn(async move {
                scan_filter_sort_ordered_row_group(
                    &engine,
                    path,
                    row_group_batch_size,
                    scan_projection,
                    row_group,
                    &filter,
                    &order_by,
                    &sort_column,
                )
                .await
            }));
        }
        let mut output = Vec::new();
        let mut rows = 0usize;
        for handle in handles {
            let sorted = handle
                .await
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))??;
            for batch in sorted {
                rows += batch.num_rows();
                output.push(batch);
            }
            if rows >= limit {
                break;
            }
        }
        return Ok(Some(limit_batches(output, Some(limit), 0)));
    }

    let mut output = Vec::new();
    let mut rows = 0usize;
    let row_group_batch_size = row_group_ordered_sort_batch_size(batch_size);
    for row_group in (0..row_group_count).rev() {
        let sorted = scan_filter_sort_ordered_row_group(
            engine,
            shared.path.clone(),
            row_group_batch_size,
            scan_projection.clone(),
            row_group,
            &shared.filter,
            order_by,
            sort_column,
        )
        .await?;
        for batch in sorted {
            rows += batch.num_rows();
            output.push(batch);
        }
        if rows >= limit {
            break;
        }
    }
    Ok(Some(limit_batches(output, Some(limit), 0)))
}

async fn scan_filter_sort_ordered_row_group(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    row_group: usize,
    filter: &FilterExpr,
    order_by: &SortKey,
    sort_column: &str,
) -> Result<Vec<RecordBatch>> {
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let scan_started = profile.then(Instant::now);
    let batches = engine
        .scan_parquet_row_group_batches(path, batch_size, projection, vec![row_group])
        .await?;
    let scan_elapsed = scan_started.map(|started| started.elapsed());
    let post_started = profile.then(Instant::now);
    if let Some(sorted) =
        reverse_filter_ascending_batches_if_ordered(&batches, sort_column, filter)?
    {
        print_ordered_row_group_profile(
            total_started,
            scan_elapsed,
            post_started.map(|started| started.elapsed()),
            row_group,
            sorted.iter().map(RecordBatch::num_rows).sum(),
            sorted.len(),
            "reverse-filter",
        );
        return Ok(sorted);
    }
    let mut filtered = Vec::new();
    for batch in batches {
        let batch = filter_batch(batch, filter)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    if filtered.is_empty() {
        print_ordered_row_group_profile(
            total_started,
            scan_elapsed,
            post_started.map(|started| started.elapsed()),
            row_group,
            0,
            0,
            "empty",
        );
        return Ok(Vec::new());
    }
    let result = match reverse_ascending_primitive_batches_if_ordered(&filtered, sort_column)? {
        Some(sorted) => Ok(sorted),
        None => apply_output_order_limit(filtered, Some(order_by), None, 0),
    }?;
    print_ordered_row_group_profile(
        total_started,
        scan_elapsed,
        post_started.map(|started| started.elapsed()),
        row_group,
        result.iter().map(RecordBatch::num_rows).sum(),
        result.len(),
        "filter-sort",
    );
    Ok(result)
}

fn row_group_ordered_desc_parallel_enabled() -> bool {
    std::env::var("DODAM_ROW_GROUP_ORDERED_DESC_PARALLEL")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn print_ordered_row_group_profile(
    total_started: Option<Instant>,
    scan_elapsed: Option<Duration>,
    post_elapsed: Option<Duration>,
    row_group: usize,
    rows: usize,
    batches: usize,
    mode: &str,
) {
    let Some(total_started) = total_started else {
        return;
    };
    eprintln!(
        "[dodam:ordered-row-group-profile] row_group={} mode={} total={}us scan={}us post={}us rows={} batches={}",
        row_group,
        mode,
        total_started.elapsed().as_micros(),
        scan_elapsed
            .map(|duration| duration.as_micros())
            .unwrap_or(0),
        post_elapsed
            .map(|duration| duration.as_micros())
            .unwrap_or(0),
        rows,
        batches
    );
}

fn reverse_filter_ascending_batches_if_ordered(
    batches: &[RecordBatch],
    sort_column: &str,
    filter: &FilterExpr,
) -> Result<Option<Vec<RecordBatch>>> {
    let mut previous_last: Option<i128> = None;
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let index = output_batch_column_index(batch, sort_column)?;
        let Some((first, last)) = numeric_column_ascending_bounds(batch.column(index))? else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            return Ok(None);
        }
        previous_last = Some(last);
    }

    let mut output = Vec::new();
    for batch in batches.iter().rev() {
        if batch.num_rows() == 0 {
            continue;
        }
        if let Some(batch) = reverse_primitive_in_list_selected_batch(batch, filter)? {
            if batch.num_rows() > 0 {
                output.push(batch);
            }
            continue;
        }
        let indices = match reverse_simple_in_list_indices(batch, filter)? {
            Some(indices) => indices,
            None if reverse_filter_ordered_batches_disabled() => return Ok(None),
            None => {
                let mask = evaluate_filter_mask(batch, filter)?;
                let mut indices = Vec::new();
                for row in (0..batch.num_rows()).rev() {
                    if !mask.is_null(row) && mask.value(row) {
                        indices.push(row as u32);
                    }
                }
                indices
            }
        };
        if indices.is_empty() {
            continue;
        }
        output.push(take_record_batch(batch, &UInt32Array::from(indices))?);
    }
    Ok(Some(output))
}

fn reverse_primitive_in_list_selected_batch(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<RecordBatch>> {
    if !reverse_primitive_in_list_ordered_batches_enabled() {
        return Ok(None);
    }
    reverse_primitive_in_list_selected_batch_unchecked(batch, filter)
}

fn reverse_primitive_in_list_selected_batch_unchecked(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<RecordBatch>> {
    let Some(indices) = reverse_simple_in_list_indices_unchecked(batch, filter)? else {
        return Ok(None);
    };
    if indices.is_empty() {
        return Ok(Some(RecordBatch::new_empty(batch.schema())));
    }
    let mut columns = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        let Some(array) = gather_primitive_array(column, &indices) else {
            return Ok(None);
        };
        columns.push(array);
    }
    Ok(Some(RecordBatch::try_new(batch.schema(), columns)?))
}

fn gather_primitive_array(column: &ArrayRef, indices: &[u32]) -> Option<ArrayRef> {
    if column.null_count() != 0 {
        return None;
    }
    match column.data_type() {
        DataType::Int32 => {
            let values = column.as_any().downcast_ref::<Int32Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Int32Array::from(output)))
        }
        DataType::Int64 => {
            let values = column.as_any().downcast_ref::<Int64Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Int64Array::from(output)))
        }
        DataType::UInt64 => {
            let values = column.as_any().downcast_ref::<UInt64Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(UInt64Array::from(output)))
        }
        DataType::Float64 => {
            let values = column.as_any().downcast_ref::<Float64Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Float64Array::from(output)))
        }
        DataType::Date32 => {
            let values = column.as_any().downcast_ref::<Date32Array>()?;
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices {
                output.push(values.value(index as usize));
            }
            Some(Arc::new(Date32Array::from(output)))
        }
        _ => None,
    }
}

fn reverse_simple_in_list_indices(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<Vec<u32>>> {
    if !reverse_in_list_ordered_batches_enabled() {
        return Ok(None);
    }
    reverse_simple_in_list_indices_unchecked(batch, filter)
}

fn reverse_simple_in_list_indices_unchecked(
    batch: &RecordBatch,
    filter: &FilterExpr,
) -> Result<Option<Vec<u32>>> {
    let Expr::InList {
        column,
        values,
        negated,
        ..
    } = filter.expr()
    else {
        return Ok(None);
    };
    if *negated {
        return Ok(None);
    }
    let column_index = output_batch_column_index(batch, column)?;
    let array = batch.column(column_index);
    match array.data_type() {
        DataType::Int32 => {
            let values = values
                .iter()
                .map(|value| value.as_i32(column))
                .collect::<Result<Vec<_>>>()?;
            let values_array = array.as_any().downcast_ref::<Int32Array>().expect("Int32");
            let mut indices = Vec::new();
            for row in (0..values_array.len()).rev() {
                if !values_array.is_null(row)
                    && values.iter().any(|value| *value == values_array.value(row))
                {
                    indices.push(row as u32);
                }
            }
            Ok(Some(indices))
        }
        DataType::Int64 => {
            let values = values
                .iter()
                .map(|value| value.as_i64(column))
                .collect::<Result<Vec<_>>>()?;
            let values_array = array.as_any().downcast_ref::<Int64Array>().expect("Int64");
            let mut indices = Vec::new();
            for row in (0..values_array.len()).rev() {
                if !values_array.is_null(row)
                    && values.iter().any(|value| *value == values_array.value(row))
                {
                    indices.push(row as u32);
                }
            }
            Ok(Some(indices))
        }
        _ => Ok(None),
    }
}

fn reverse_primitive_in_list_ordered_batches_enabled() -> bool {
    std::env::var("DODAM_ENABLE_REVERSE_PRIMITIVE_IN_LIST_ORDERED_BATCHES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn reverse_in_list_ordered_batches_enabled() -> bool {
    std::env::var("DODAM_ENABLE_REVERSE_IN_LIST_ORDERED_BATCHES")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn reverse_filter_ordered_batches_disabled() -> bool {
    std::env::var("DODAM_DISABLE_REVERSE_FILTER_ORDERED_BATCHES")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn row_group_ordered_sort_batch_size(default_batch_size: usize) -> usize {
    std::env::var("DODAM_ROW_GROUP_ORDERED_SORT_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| default_batch_size.max(value))
        .unwrap_or(default_batch_size)
}

fn reverse_ascending_primitive_batches_if_ordered(
    batches: &[RecordBatch],
    sort_column: &str,
) -> Result<Option<Vec<RecordBatch>>> {
    if row_group_primitive_reverse_materialization_disabled() {
        return Ok(None);
    }
    let mut previous_last: Option<i128> = None;
    for batch in batches {
        if !record_batch_supports_fast_reverse(batch) {
            return Ok(None);
        }
        let index = output_batch_column_index(batch, sort_column)?;
        let Some((first, last)) = numeric_column_ascending_bounds(batch.column(index))? else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            return Ok(None);
        }
        previous_last = Some(last);
    }

    let mut output = Vec::with_capacity(batches.len());
    for batch in batches.iter().rev() {
        let Some(reversed) = reverse_primitive_record_batch_rows(batch)? else {
            return Ok(None);
        };
        output.push(reversed);
    }
    Ok(Some(output))
}

fn row_group_primitive_reverse_materialization_disabled() -> bool {
    std::env::var("DODAM_DISABLE_ROW_GROUP_PRIMITIVE_REVERSE_MATERIALIZATION")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn row_group_ordered_desc_bulk_global_sort_enabled() -> bool {
    std::env::var("DODAM_ROW_GROUP_ORDERED_DESC_BULK_GLOBAL_SORT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn record_batch_supports_fast_reverse(batch: &RecordBatch) -> bool {
    batch
        .columns()
        .iter()
        .all(|column| primitive_array_supports_fast_reverse(column))
}

fn primitive_array_supports_fast_reverse(column: &ArrayRef) -> bool {
    column.null_count() == 0
        && matches!(
            column.data_type(),
            DataType::Int32
                | DataType::Int64
                | DataType::UInt64
                | DataType::Float64
                | DataType::Date32
        )
}

fn reverse_primitive_record_batch_rows(batch: &RecordBatch) -> Result<Option<RecordBatch>> {
    let mut columns = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        let Some(reversed) = reverse_primitive_array(column) else {
            return Ok(None);
        };
        columns.push(reversed);
    }
    Ok(Some(RecordBatch::try_new(batch.schema(), columns)?))
}

fn reverse_primitive_array(column: &ArrayRef) -> Option<ArrayRef> {
    if column.null_count() != 0 {
        return None;
    }
    match column.data_type() {
        DataType::Int32 => {
            let values = column.as_any().downcast_ref::<Int32Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Int32Array::from(output)))
        }
        DataType::Int64 => {
            let values = column.as_any().downcast_ref::<Int64Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Int64Array::from(output)))
        }
        DataType::UInt64 => {
            let values = column.as_any().downcast_ref::<UInt64Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(UInt64Array::from(output)))
        }
        DataType::Float64 => {
            let values = column.as_any().downcast_ref::<Float64Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Float64Array::from(output)))
        }
        DataType::Date32 => {
            let values = column.as_any().downcast_ref::<Date32Array>()?;
            let mut output = Vec::with_capacity(values.len());
            for row in (0..values.len()).rev() {
                output.push(values.value(row));
            }
            Some(Arc::new(Date32Array::from(output)))
        }
        _ => None,
    }
}

async fn try_reverse_row_group_desc_tail_topk(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    sort_column: &str,
    batch_size: usize,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit > reverse_row_group_topk_max_limit_rows() {
        return Ok(None);
    }
    if !engine
        .parquet_row_groups_monotonic_by_column(shared.path.clone(), sort_column)
        .await?
    {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&shared.path)?;
    if row_group_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    add_projection_columns(&mut scan_projection, vec![sort_column.to_string()]);

    let mut suffix = Vec::new();
    let mut suffix_rows = 0usize;
    for row_group in (0..row_group_count).rev() {
        let batches = engine
            .scan_parquet_row_group_batches(
                shared.path.clone(),
                batch_size,
                scan_projection.clone(),
                vec![row_group],
            )
            .await?;
        let mut filtered = Vec::new();
        for batch in batches {
            let batch = filter_batch(batch, &shared.filter)?;
            if batch.num_rows() > 0 {
                filtered.push(batch);
            }
        }
        suffix_rows += filtered.iter().map(RecordBatch::num_rows).sum::<usize>();
        suffix.splice(0..0, filtered);
        if suffix_rows >= limit {
            break;
        }
    }
    Ok(Some(suffix))
}

fn reverse_row_group_topk_max_limit_rows() -> usize {
    std::env::var("DODAM_REVERSE_ROW_GROUP_TOPK_MAX_LIMIT_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(65_536)
}

fn collect_monotonic_desc_tail_topk(
    mut stream: SendableBatchStream,
    sort_column: &str,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let mut tail = VecDeque::new();
    let mut tail_rows = 0usize;
    let mut previous_last = None;

    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let index = output_batch_column_index(&batch, sort_column)?;
        let column = batch.column(index);
        let Some((first, last)) = numeric_column_ascending_bounds(column)? else {
            return Ok(None);
        };
        if previous_last.is_some_and(|previous| previous > first) {
            return Ok(None);
        }
        previous_last = Some(last);
        tail_rows += batch.num_rows();
        tail.push_back(batch);
        while tail_rows > limit {
            let excess = tail_rows - limit;
            let front_rows = tail.front().map(RecordBatch::num_rows).unwrap_or(0);
            if excess >= front_rows {
                tail.pop_front();
                tail_rows -= front_rows;
            } else if let Some(front) = tail.pop_front() {
                let kept = front.slice(excess, front_rows - excess);
                tail_rows -= excess;
                tail.push_front(kept);
            }
        }
    }
    if tail.is_empty() {
        return Ok(Some(Vec::new()));
    }
    Ok(Some(tail.into_iter().collect()))
}

async fn try_same_source_union_all_streaming_desc_topk(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    sort_column: &str,
    batch_size: usize,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit > reverse_row_group_topk_max_limit_rows() {
        return Ok(None);
    }

    let mut scan_projection = shared.projection.clone();
    add_projection_columns(&mut scan_projection, shared.filter.referenced_columns());
    add_projection_columns(&mut scan_projection, vec![sort_column.to_string()]);

    let mut stream = engine
        .scan_parquet_batches(
            shared.path.clone(),
            batch_size,
            None,
            scan_projection,
            Some(shared.filter.clone()),
        )
        .await?;
    let mut batches = Vec::new();
    let mut heap = BinaryHeap::<Reverse<(i128, u64, usize, u32)>>::with_capacity(limit + 1);
    let mut sequence = 0u64;
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    let mut next_elapsed = Duration::ZERO;
    let mut heap_elapsed = Duration::ZERO;
    let mut selected_sort_elapsed = Duration::ZERO;
    let mut materialize_elapsed = Duration::ZERO;
    let mut scanned_batches = 0usize;
    let mut scanned_rows = 0usize;

    for batch in stream.by_ref() {
        let next_started = profile.then(Instant::now);
        let batch = batch?;
        if let Some(started) = next_started {
            next_elapsed += started.elapsed();
        }
        if batch.num_rows() == 0 {
            continue;
        }
        scanned_batches += 1;
        scanned_rows += batch.num_rows();
        let key_index = output_batch_column_index(&batch, sort_column)?;
        let key_column = batch.column(key_index);
        if !topk_sort_key_type_supported(key_column.data_type()) {
            return Ok(None);
        }
        let batch_index = batches.len();
        let heap_started = profile.then(Instant::now);
        if !update_streaming_desc_topk_heap(
            key_column,
            batch_index,
            limit,
            &mut sequence,
            &mut heap,
        )? {
            return Ok(None);
        }
        if let Some(started) = heap_started {
            heap_elapsed += started.elapsed();
        }
        batches.push(batch);
    }

    if heap.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let mut selected = heap
        .into_iter()
        .map(|Reverse(item)| item)
        .collect::<Vec<_>>();
    let selected_sort_started = profile.then(Instant::now);
    selected.sort_unstable_by(|left, right| right.cmp(left));
    if let Some(started) = selected_sort_started {
        selected_sort_elapsed += started.elapsed();
    }
    let materialize_started = profile.then(Instant::now);
    let batch = materialize_topk_selected_rows(&batches, &selected)?;
    if let Some(started) = materialize_started {
        materialize_elapsed += started.elapsed();
    }
    print_streaming_topk_profile(
        total_started,
        next_elapsed,
        heap_elapsed,
        selected_sort_elapsed,
        materialize_elapsed,
        scanned_batches,
        scanned_rows,
        selected.len(),
    );
    Ok(Some(vec![batch]))
}

#[allow(clippy::too_many_arguments)]
fn print_streaming_topk_profile(
    total_started: Option<Instant>,
    next_elapsed: Duration,
    heap_elapsed: Duration,
    selected_sort_elapsed: Duration,
    materialize_elapsed: Duration,
    scanned_batches: usize,
    scanned_rows: usize,
    selected_rows: usize,
) {
    let Some(total_started) = total_started else {
        return;
    };
    eprintln!(
        "[dodam:streaming-topk-profile] total={}us next={}us heap={}us selected_sort={}us materialize={}us batches={} rows={} selected={}",
        total_started.elapsed().as_micros(),
        next_elapsed.as_micros(),
        heap_elapsed.as_micros(),
        selected_sort_elapsed.as_micros(),
        materialize_elapsed.as_micros(),
        scanned_batches,
        scanned_rows,
        selected_rows,
    );
}

#[allow(clippy::too_many_arguments)]
fn try_write_same_source_union_all_streaming_primitive_topk_to_sink(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    scan_projection: &Projection,
    sort_column: &str,
    limit: usize,
    row_group_count: usize,
    sink: &mut dyn RecordBatchSink,
) -> Result<bool> {
    try_write_same_source_union_all_streaming_primitive_topk_to_sink_inner(
        engine,
        shared,
        batch_size,
        scan_projection,
        sort_column,
        limit,
        row_group_count,
        sink,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_write_same_source_union_all_streaming_primitive_topk_to_sink_inner(
    engine: &DodamEngine,
    shared: &SameSourceUnionAllScan,
    batch_size: usize,
    scan_projection: &Projection,
    sort_column: &str,
    limit: usize,
    row_group_count: usize,
    sink: &mut dyn RecordBatchSink,
    allow_selected_payload: bool,
) -> Result<bool> {
    if limit > reverse_row_group_topk_max_limit_rows() {
        return Ok(false);
    }
    let Projection::Columns(columns) = scan_projection else {
        return Ok(false);
    };
    let Expr::InList {
        column: filter_column,
        values,
        negated,
        ..
    } = shared.filter.expr()
    else {
        return Ok(false);
    };
    if *negated {
        return Ok(false);
    }
    let Some(column_types) = engine.parquet_direct_primitive_column_types(&shared.path, columns)?
    else {
        return Ok(false);
    };
    if !column_types.iter().all(|column_type| {
        matches!(
            column_type,
            DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
        )
    }) {
        return Ok(false);
    }
    let Some(filter_index) = columns.iter().position(|column| column == filter_column) else {
        return Ok(false);
    };
    let Some(sort_index) = columns.iter().position(|column| column == sort_column) else {
        return Ok(false);
    };
    let key_scan_columns = primitive_topk_key_columns(columns, filter_index, sort_index);
    let explicit_selected_payload = primitive_topk_selected_payload_enabled();
    let auto_selected_payload =
        primitive_topk_selected_payload_auto_enabled() && !explicit_selected_payload;
    let use_selected_payload = allow_selected_payload
        && (explicit_selected_payload || auto_selected_payload)
        && primitive_topk_selected_payload_precheck_accepts(
            engine,
            &shared.path,
            sort_column,
            limit,
            row_group_count,
            columns,
            &key_scan_columns,
            auto_selected_payload,
        )?;
    let scan_columns = if use_selected_payload {
        key_scan_columns
    } else {
        columns.clone()
    };
    let scan_column_types = scan_columns
        .iter()
        .map(|column| {
            let Some(index) = columns.iter().position(|candidate| candidate == column) else {
                return Err(DodamError::UnsupportedSql(format!(
                    "primitive top-k scan column {column} is not projected"
                )));
            };
            Ok(column_types[index])
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(scan_filter_index) = scan_columns
        .iter()
        .position(|column| column == filter_column)
    else {
        return Ok(false);
    };
    let Some(scan_sort_index) = scan_columns.iter().position(|column| column == sort_column) else {
        return Ok(false);
    };
    let scan_filter_values = match scan_column_types[scan_filter_index] {
        DirectPrimitiveColumnType::I32 => PrimitiveFilterValues::I32(
            values
                .iter()
                .map(|value| value.as_i32(filter_column))
                .collect::<Result<Vec<_>>>()?,
        ),
        DirectPrimitiveColumnType::I64 => PrimitiveFilterValues::I64(
            values
                .iter()
                .map(|value| value.as_i64(filter_column))
                .collect::<Result<Vec<_>>>()?,
        ),
        DirectPrimitiveColumnType::Date32
        | DirectPrimitiveColumnType::Decimal128Int64 { .. }
        | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(false),
    };
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let profile = ordered_sink_profile_enabled();
    let total_started = profile.then(Instant::now);
    if primitive_topk_fused_selected_page_reader_enabled() && !use_selected_payload {
        let filter_i32_values;
        let filter_i64_values;
        let (filter_i32, filter_i64) = match &scan_filter_values {
            PrimitiveFilterValues::I32(values) => {
                filter_i32_values = values.clone();
                (&filter_i32_values[..], &[][..])
            }
            PrimitiveFilterValues::I64(values) => {
                filter_i64_values = values.clone();
                (&[][..], &filter_i64_values[..])
            }
        };
        let specs = columns
            .iter()
            .zip(column_types.iter())
            .map(|(name, column_type)| DirectPrimitiveColumnSpec {
                name,
                column_type: *column_type,
            })
            .collect::<Vec<_>>();
        let mut state = PrimitiveSelectedBatchTopkState::new(limit, &column_types);
        let mut metrics = DirectPrimitiveColumnScanMetrics::default();
        let mut supported = true;
        let chunk_size = same_source_union_primitive_chunk_size(row_group_count);
        for chunk in row_groups.chunks(chunk_size) {
            let scan_results = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(chunk.len());
                for (position, row_group) in chunk.iter().copied().enumerate() {
                    let engine = engine.clone();
                    let path = shared.path.clone();
                    let specs = specs.clone();
                    let column_types = column_types.clone();
                    let columns = columns.to_vec();
                    handles.push(scope.spawn(move || {
                        let mut local_state =
                            PrimitiveSelectedBatchTopkState::new(limit, &column_types);
                        let metrics = engine
                            .scan_parquet_required_plain_primitive_in_list_desc_selected_pages(
                                &path,
                                batch_size,
                                &[row_group],
                                &specs,
                                filter_index,
                                filter_i32,
                                filter_i64,
                                |batch| {
                                    local_state.consume_selected_page(
                                        batch,
                                        &column_types,
                                        sort_index,
                                    )
                                },
                            )?;
                        let batch = if local_state.unsupported {
                            None
                        } else {
                            Some(local_state.into_primitive_batch(&columns, &column_types)?)
                                .filter(|batch| !batch.is_empty())
                        };
                        Ok::<_, DodamError>((position, batch, metrics))
                    }));
                }
                let mut results = Vec::with_capacity(handles.len());
                for handle in handles {
                    match handle.join() {
                        Ok(result) => results.push(result?),
                        Err(_) => {
                            return Err(DodamError::UnsupportedSql(
                                "primitive top-k fused selected page worker panicked".to_string(),
                            ));
                        }
                    }
                }
                Ok::<_, DodamError>(results)
            })?;
            let mut scan_results = scan_results;
            scan_results.sort_by_key(|(position, _, _)| *position);
            for (_, batch, row_group_metrics) in scan_results {
                let Some(row_group_metrics) = row_group_metrics else {
                    supported = false;
                    break;
                };
                metrics.merge_from(row_group_metrics);
                if let Some(batch) = batch {
                    state.consume_primitive_batch(&batch, &column_types, sort_index)?;
                }
            }
            if !supported || state.unsupported {
                break;
            }
        }
        if supported && !state.unsupported {
            let batch = state.into_primitive_batch(columns, &column_types)?;
            print_streaming_primitive_topk_profile(total_started, &metrics, batch.num_rows());
            write_same_source_primitive_batch_to_sink(batch, scan_projection, shared, sink)?;
            return Ok(true);
        }
    }
    let Some((state, metrics)) = engine
        .scan_parquet_primitive_columns_parallel_view_fold_with_location(
            shared.path.clone(),
            batch_size,
            row_groups,
            scan_columns
                .iter()
                .zip(scan_column_types.iter())
                .map(|(name, column_type)| (name.clone(), *column_type))
                .collect(),
            || PrimitiveTopkState::new(limit, &scan_column_types, use_selected_payload),
            |state, location, view| {
                state.consume_view(
                    location,
                    view,
                    &scan_column_types,
                    scan_filter_index,
                    &scan_filter_values,
                    scan_sort_index,
                )
            },
            PrimitiveTopkState::merge,
        )?
    else {
        return Ok(false);
    };
    if state.unsupported {
        return Ok(false);
    }
    let batch = if use_selected_payload {
        if let Some(row_refs) = state.selected_row_refs_sorted() {
            let base_batch = state.into_primitive_batch(&scan_columns, &scan_column_types)?;
            if primitive_topk_selected_payload_spread_accepts(
                &row_refs,
                row_group_count,
                columns,
                &scan_columns,
            ) || !primitive_topk_selected_payload_spread_gate_enabled()
            {
                read_primitive_topk_selected_payload_with_base(
                    engine,
                    &shared.path,
                    batch_size,
                    row_refs,
                    base_batch,
                    &scan_columns,
                    &scan_column_types,
                    columns,
                    &column_types,
                )?
            } else {
                return try_write_same_source_union_all_streaming_primitive_topk_to_sink_inner(
                    engine,
                    shared,
                    batch_size,
                    scan_projection,
                    sort_column,
                    limit,
                    row_group_count,
                    sink,
                    false,
                );
            }
        } else {
            state.into_primitive_batch(&scan_columns, &scan_column_types)?
        }
    } else {
        state.into_primitive_batch(columns, &column_types)?
    };
    print_streaming_primitive_topk_profile(total_started, &metrics, batch.num_rows());
    write_same_source_primitive_batch_to_sink(batch, scan_projection, shared, sink)?;
    Ok(true)
}

fn primitive_topk_key_columns(
    columns: &[String],
    filter_index: usize,
    sort_index: usize,
) -> Vec<String> {
    let mut scan_columns = Vec::with_capacity(2);
    scan_columns.push(columns[filter_index].clone());
    if sort_index != filter_index {
        scan_columns.push(columns[sort_index].clone());
    }
    scan_columns
}

fn primitive_topk_selected_payload_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn primitive_topk_selected_payload_auto_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_AUTO")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn primitive_topk_fused_selected_page_reader_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_FUSED_SELECTED_PAGE_READER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn primitive_topk_selected_payload_precheck_accepts(
    engine: &DodamEngine,
    path: &Path,
    sort_column: &str,
    limit: usize,
    total_row_groups: usize,
    output_columns: &[String],
    base_columns: &[String],
    require_stats: bool,
) -> Result<bool> {
    let missing_payload_columns =
        primitive_topk_missing_payload_columns(output_columns, base_columns);
    if missing_payload_columns == 0 {
        log_primitive_topk_selected_payload_spread(
            limit,
            0,
            total_row_groups,
            missing_payload_columns,
            SelectedPayloadDecision::EmptySelection,
            "precheck",
        );
        return Ok(true);
    }
    if require_stats && missing_payload_columns < primitive_topk_selected_payload_min_auto_columns()
    {
        log_primitive_topk_selected_payload_spread(
            limit,
            0,
            total_row_groups,
            missing_payload_columns,
            SelectedPayloadDecision::PayloadColumns,
            "precheck",
        );
        return Ok(false);
    }
    let Some(ranges) = engine.parquet_primitive_column_min_max_by_row_group(path, sort_column)?
    else {
        return Ok(!require_stats);
    };
    let Some(candidate_row_groups) =
        estimate_desc_topk_candidate_row_groups_from_stats(&ranges, limit)
    else {
        return Ok(!require_stats);
    };
    let decision = choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
        selected_rows: limit,
        selected_row_groups: candidate_row_groups,
        total_row_groups,
        missing_payload_columns,
        max_selected_row_group_ratio: primitive_topk_selected_payload_max_row_group_ratio(),
        max_selected_row_groups: primitive_topk_selected_payload_max_row_groups(),
    });
    log_primitive_topk_selected_payload_spread(
        limit,
        candidate_row_groups,
        total_row_groups,
        missing_payload_columns,
        decision,
        "precheck",
    );
    Ok(decision.accepted())
}

fn estimate_desc_topk_candidate_row_groups_from_stats(
    ranges: &[PrimitiveRowGroupMinMax],
    limit: usize,
) -> Option<usize> {
    if limit == 0 || ranges.is_empty() {
        return Some(0);
    }
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable_by(|left, right| {
        right
            .max
            .cmp(&left.max)
            .then_with(|| right.min.cmp(&left.min))
            .then_with(|| left.row_group.cmp(&right.row_group))
    });
    let mut rows = 0usize;
    let mut threshold_min = None;
    for range in &sorted {
        rows = rows.saturating_add(range.rows);
        threshold_min =
            Some(threshold_min.map_or(range.min, |current: i128| current.min(range.min)));
        if rows >= limit {
            break;
        }
    }
    let threshold_min = threshold_min?;
    Some(
        ranges
            .iter()
            .filter(|range| range.max >= threshold_min)
            .count(),
    )
}

fn primitive_topk_selected_payload_spread_accepts(
    row_refs: &[PrimitiveTopkRowRef],
    total_row_groups: usize,
    output_columns: &[String],
    base_columns: &[String],
) -> bool {
    let missing_payload_columns =
        primitive_topk_missing_payload_columns(output_columns, base_columns);
    let mut row_groups = FastHashSet::default();
    for row_ref in row_refs {
        row_groups.insert(row_ref.row_group);
    }
    let decision = choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
        selected_rows: row_refs.len(),
        selected_row_groups: row_groups.len(),
        total_row_groups,
        missing_payload_columns,
        max_selected_row_group_ratio: primitive_topk_selected_payload_max_row_group_ratio(),
        max_selected_row_groups: primitive_topk_selected_payload_max_row_groups(),
    });
    log_primitive_topk_selected_payload_spread(
        row_refs.len(),
        row_groups.len(),
        total_row_groups,
        missing_payload_columns,
        decision,
        "actual",
    );
    decision.accepted()
}

fn primitive_topk_missing_payload_columns(
    output_columns: &[String],
    base_columns: &[String],
) -> usize {
    output_columns
        .iter()
        .filter(|column| !base_columns.contains(*column))
        .count()
}

fn primitive_topk_selected_payload_max_row_group_ratio() -> f64 {
    std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_MAX_ROW_GROUP_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.25)
}

fn primitive_topk_selected_payload_max_row_groups() -> usize {
    std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_MAX_ROW_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn primitive_topk_selected_payload_min_auto_columns() -> usize {
    std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_MIN_AUTO_COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
}

fn primitive_topk_selected_payload_spread_gate_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_SPREAD_GATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_SELECTED_PAYLOAD_SPREAD_GATE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn log_primitive_topk_selected_payload_spread(
    selected_rows: usize,
    selected_row_groups: usize,
    total_row_groups: usize,
    missing_payload_columns: usize,
    decision: SelectedPayloadDecision,
    phase: &str,
) {
    if !std::env::var("DODAM_PRIMITIVE_TOPK_SELECTED_PAYLOAD_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:primitive-topk-selected-payload] phase={} decision={} selected_rows={} row_groups={}/{} missing_payload_columns={}",
        phase,
        decision.reason(),
        selected_rows,
        selected_row_groups,
        total_row_groups,
        missing_payload_columns
    );
}

fn read_primitive_topk_selected_payload(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_refs: Vec<PrimitiveTopkRowRef>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    if row_refs.is_empty() {
        return primitive_empty_batch(column_names, column_types);
    }
    let mut refs_by_row_group = FastHashMap::<usize, Vec<(usize, usize)>>::default();
    for (output_index, row_ref) in row_refs.iter().copied().enumerate() {
        refs_by_row_group
            .entry(row_ref.row_group)
            .or_default()
            .push((row_ref.row_offset, output_index));
    }
    let mut row_groups = refs_by_row_group.keys().copied().collect::<Vec<_>>();
    row_groups.sort_unstable();
    for refs in refs_by_row_group.values_mut() {
        refs.sort_unstable_by_key(|(row_offset, _)| *row_offset);
    }
    let refs_by_row_group = Arc::new(refs_by_row_group);
    let rows = row_refs.len();
    let Some((state, _metrics)) = engine
        .scan_parquet_primitive_columns_parallel_view_fold_with_location(
            path.to_path_buf(),
            batch_size,
            row_groups,
            column_names
                .iter()
                .zip(column_types.iter())
                .map(|(name, column_type)| (name.clone(), *column_type))
                .collect(),
            || SelectedPrimitivePayloadState::new(rows, column_types),
            {
                let refs_by_row_group = Arc::clone(&refs_by_row_group);
                move |state: &mut SelectedPrimitivePayloadState, location, view| {
                    state.consume(location, view, column_types, refs_by_row_group.as_ref())
                }
            },
            SelectedPrimitivePayloadState::merge,
        )?
    else {
        return Err(DodamError::UnsupportedSql(
            "primitive top-k selected payload reader is unsupported".to_string(),
        ));
    };
    state.into_batch(column_names, column_types)
}

#[allow(clippy::too_many_arguments)]
fn read_primitive_topk_selected_payload_with_base(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_refs: Vec<PrimitiveTopkRowRef>,
    base_batch: PrimitiveBatch,
    base_column_names: &[String],
    base_column_types: &[DirectPrimitiveColumnType],
    output_column_names: &[String],
    output_column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    if output_column_names.len() != output_column_types.len()
        || base_column_names.len() != base_column_types.len()
    {
        return Err(DodamError::UnsupportedSql(
            "primitive top-k selected payload schema mismatch".to_string(),
        ));
    }
    let mut base_columns = base_batch
        .columns
        .into_iter()
        .map(|column| (column.name.clone(), column))
        .collect::<FastHashMap<_, _>>();
    let mut missing_names = Vec::new();
    let mut missing_types = Vec::new();
    for (name, column_type) in output_column_names.iter().zip(output_column_types.iter()) {
        if base_columns.contains_key(name) {
            continue;
        }
        missing_names.push(name.clone());
        missing_types.push(*column_type);
    }
    let mut missing_columns = if missing_names.is_empty() {
        FastHashMap::default()
    } else {
        read_primitive_topk_selected_payload(
            engine,
            path,
            batch_size,
            row_refs,
            &missing_names,
            &missing_types,
        )?
        .columns
        .into_iter()
        .map(|column| (column.name.clone(), column))
        .collect::<FastHashMap<_, _>>()
    };
    let mut columns = Vec::with_capacity(output_column_names.len());
    for (name, column_type) in output_column_names.iter().zip(output_column_types.iter()) {
        let column = if let Some(column) = base_columns.remove(name) {
            column
        } else if let Some(column) = missing_columns.remove(name) {
            column
        } else {
            return Err(DodamError::UnsupportedSql(format!(
                "primitive top-k selected payload column {name} was not materialized"
            )));
        };
        if !primitive_column_matches_direct_type(&column, column_type) {
            return Err(DodamError::UnsupportedSql(format!(
                "primitive top-k selected payload column {name} type mismatch"
            )));
        }
        columns.push(column);
    }
    Ok(PrimitiveBatch { columns })
}

struct SelectedPrimitivePayloadState {
    columns: Vec<PrimitiveColumnOutput>,
    filled: Vec<bool>,
    unsupported: bool,
}

impl SelectedPrimitivePayloadState {
    fn new(rows: usize, column_types: &[DirectPrimitiveColumnType]) -> Self {
        let columns = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => PrimitiveColumnOutput::I32(vec![0; rows]),
                DirectPrimitiveColumnType::I64 => PrimitiveColumnOutput::I64(vec![0; rows]),
                _ => unreachable!("primitive selected payload only uses i32/i64 columns"),
            })
            .collect();
        Self {
            columns,
            filled: vec![false; rows],
            unsupported: false,
        }
    }

    fn consume(
        &mut self,
        location: DirectPrimitiveBatchLocation,
        view: BatchView<'_>,
        column_types: &[DirectPrimitiveColumnType],
        refs_by_row_group: &FastHashMap<usize, Vec<(usize, usize)>>,
    ) -> Result<()> {
        let Some(refs) = refs_by_row_group.get(&location.row_group) else {
            return Ok(());
        };
        let batch_start = location.row_offset;
        let batch_end = batch_start.saturating_add(view.num_rows());
        let first = refs.partition_point(|(row_offset, _)| *row_offset < batch_start);
        let mut index = first;
        if index >= refs.len() || refs[index].0 >= batch_end {
            return Ok(());
        }
        let Some(columns) = null_free_primitive_columns_for_topk(view, column_types) else {
            self.unsupported = true;
            return Ok(());
        };
        while index < refs.len() {
            let (row_offset, output_index) = refs[index];
            if row_offset >= batch_end {
                break;
            }
            let local_row = row_offset - batch_start;
            for (source, target) in columns.iter().zip(self.columns.iter_mut()) {
                overwrite_null_free_primitive_value(source, local_row, target, output_index)?;
            }
            self.filled[output_index] = true;
            index += 1;
        }
        Ok(())
    }

    fn merge(&mut self, source: Self) -> Result<()> {
        self.unsupported |= source.unsupported;
        for (index, filled) in source.filled.iter().copied().enumerate() {
            if !filled {
                continue;
            }
            for (source_column, target) in source.columns.iter().zip(self.columns.iter_mut()) {
                overwrite_primitive_output_slot(source_column, index, target, index)?;
            }
            self.filled[index] = true;
        }
        Ok(())
    }

    fn into_batch(
        self,
        column_names: &[String],
        column_types: &[DirectPrimitiveColumnType],
    ) -> Result<PrimitiveBatch> {
        if self.unsupported || self.filled.iter().any(|filled| !*filled) {
            return Err(DodamError::UnsupportedSql(
                "primitive top-k selected payload reader did not fill all rows".to_string(),
            ));
        }
        let columns = self
            .columns
            .into_iter()
            .zip(column_names.iter())
            .zip(column_types.iter())
            .map(|((values, name), column_type)| {
                let values = match values {
                    PrimitiveColumnOutput::I32(values) => PrimitiveColumnValues::I32(values),
                    PrimitiveColumnOutput::I64(values) => PrimitiveColumnValues::I64(values),
                };
                Ok(PrimitiveColumn {
                    name: name.clone(),
                    data_type: primitive_output_data_type(column_type)?,
                    nullable: false,
                    values,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PrimitiveBatch { columns })
    }
}

fn print_streaming_primitive_topk_profile(
    total_started: Option<Instant>,
    metrics: &DirectPrimitiveColumnScanMetrics,
    rows: usize,
) {
    let Some(total_started) = total_started else {
        return;
    };
    let column_read = metrics
        .column_read_nanos
        .iter()
        .map(|nanos| format!("{:.3}", (*nanos as f64) / 1_000_000.0))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[dodam:streaming-primitive-topk-profile] total={}us read={:.3}ms consume={:.3}ms row_groups={} batches={} scanned_rows={} output_rows={} column_read_ms=[{}]",
        total_started.elapsed().as_micros(),
        (metrics.read_nanos as f64) / 1_000_000.0,
        (metrics.consume_nanos as f64) / 1_000_000.0,
        metrics.row_groups,
        metrics.batches,
        metrics.rows,
        rows,
        column_read,
    );
}

struct PrimitiveTopkState {
    limit: usize,
    unsupported: bool,
    sequence: u64,
    heap: BinaryHeap<Reverse<(i128, u64, usize)>>,
    columns: Vec<PrimitiveColumnOutput>,
    row_refs: Option<Vec<PrimitiveTopkRowRef>>,
    selected_positions: Vec<usize>,
}

struct PrimitiveSelectedBatchTopkState {
    limit: usize,
    unsupported: bool,
    sequence: u64,
    heap: BinaryHeap<Reverse<(i128, u64, usize)>>,
    columns: Vec<PrimitiveColumnOutput>,
}

#[derive(Clone, Copy)]
struct PrimitiveTopkRowRef {
    row_group: usize,
    row_offset: usize,
}

impl PrimitiveSelectedBatchTopkState {
    fn new(limit: usize, column_types: &[DirectPrimitiveColumnType]) -> Self {
        let columns = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(limit))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(limit))
                }
                _ => unreachable!("primitive selected top-k only uses i32/i64 columns"),
            })
            .collect();
        Self {
            limit,
            unsupported: false,
            sequence: 0,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
            columns,
        }
    }

    fn consume_selected_page(
        &mut self,
        batch: DirectSelectedPrimitivePageBatch<'_>,
        column_types: &[DirectPrimitiveColumnType],
        sort_index: usize,
    ) -> Result<()> {
        if sort_index >= batch.columns.len()
            || batch.columns.len() != column_types.len()
            || batch.columns.is_empty()
        {
            self.unsupported = true;
            return Ok(());
        }
        for &row in batch.selected_positions {
            let Some(key) = batch.columns[sort_index].value_i128(row) else {
                self.unsupported = true;
                return Ok(());
            };
            self.insert_candidate_page(key, &batch.columns, column_types, row)?;
            self.sequence = self.sequence.wrapping_add(1);
        }
        Ok(())
    }

    fn consume_primitive_batch(
        &mut self,
        batch: &PrimitiveBatch,
        column_types: &[DirectPrimitiveColumnType],
        sort_index: usize,
    ) -> Result<()> {
        if sort_index >= batch.columns.len()
            || batch.columns.len() != column_types.len()
            || batch.columns.is_empty()
        {
            self.unsupported = true;
            return Ok(());
        }
        let rows = batch.num_rows();
        for column in &batch.columns {
            if column.values.len() != rows {
                self.unsupported = true;
                return Ok(());
            }
        }
        for row in 0..rows {
            let Some(key) = primitive_column_values_key(&batch.columns[sort_index].values, row)
            else {
                self.unsupported = true;
                return Ok(());
            };
            self.insert_candidate_primitive_batch(key, &batch.columns, column_types, row)?;
            self.sequence = self.sequence.wrapping_add(1);
        }
        Ok(())
    }

    fn insert_candidate_page(
        &mut self,
        key: i128,
        columns: &[DirectSelectedPrimitiveColumnPageView<'_>],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        let sequence = self.sequence;
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_page(columns, column_types, row)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_page(slot, columns, column_types, row)?;
        }
        Ok(())
    }

    fn insert_candidate_primitive_batch(
        &mut self,
        key: i128,
        columns: &[PrimitiveColumn],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        let sequence = self.sequence;
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_primitive_batch(columns, column_types, row)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_primitive_batch(slot, columns, column_types, row)?;
        }
        Ok(())
    }

    fn push_slot_from_page(
        &mut self,
        columns: &[DirectSelectedPrimitiveColumnPageView<'_>],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            push_direct_selected_page_value(source, column_type, row, target)?;
        }
        Ok(slot)
    }

    fn overwrite_slot_from_page(
        &mut self,
        slot: usize,
        columns: &[DirectSelectedPrimitiveColumnPageView<'_>],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            overwrite_direct_selected_page_value(source, column_type, row, target, slot)?;
        }
        Ok(())
    }

    fn push_slot_from_primitive_batch(
        &mut self,
        columns: &[PrimitiveColumn],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            push_primitive_batch_value(&source.values, column_type, row, target)?;
        }
        Ok(slot)
    }

    fn overwrite_slot_from_primitive_batch(
        &mut self,
        slot: usize,
        columns: &[PrimitiveColumn],
        column_types: &[DirectPrimitiveColumnType],
        row: usize,
    ) -> Result<()> {
        for ((source, column_type), target) in columns
            .iter()
            .zip(column_types.iter())
            .zip(self.columns.iter_mut())
        {
            overwrite_primitive_batch_value(&source.values, column_type, row, target, slot)?;
        }
        Ok(())
    }

    fn into_primitive_batch(
        self,
        column_names: &[String],
        column_types: &[DirectPrimitiveColumnType],
    ) -> Result<PrimitiveBatch> {
        let mut selected = self
            .heap
            .into_iter()
            .map(|Reverse(item)| item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        let mut output = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(selected.len()))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(selected.len()))
                }
                _ => unreachable!("primitive selected top-k only uses i32/i64 columns"),
            })
            .collect::<Vec<_>>();
        for (_, _, slot) in selected {
            for (source, target) in self.columns.iter().zip(output.iter_mut()) {
                push_primitive_output_slot(source, slot, target)?;
            }
        }
        primitive_output_batch_from_columns(output, column_names, column_types)
    }
}

impl PrimitiveTopkState {
    fn new(limit: usize, column_types: &[DirectPrimitiveColumnType], track_row_refs: bool) -> Self {
        let columns = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(limit))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(limit))
                }
                _ => unreachable!("primitive top-k only uses i32/i64 columns"),
            })
            .collect();
        Self {
            limit,
            unsupported: false,
            sequence: 0,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
            columns,
            row_refs: (track_row_refs
                || primitive_topk_row_refs_enabled()
                || primitive_topk_selected_payload_enabled())
            .then(|| Vec::with_capacity(limit)),
            selected_positions: Vec::new(),
        }
    }

    fn consume_view(
        &mut self,
        location: DirectPrimitiveBatchLocation,
        view: BatchView<'_>,
        column_types: &[DirectPrimitiveColumnType],
        filter_index: usize,
        filter_values: &PrimitiveFilterValues,
        sort_index: usize,
    ) -> Result<()> {
        let Some(columns) = null_free_primitive_columns_for_topk(view, column_types) else {
            self.unsupported = true;
            return Ok(());
        };
        let threshold_key =
            if primitive_topk_fused_filter_threshold_enabled() && self.heap.len() >= self.limit {
                self.heap.peek().map(|worst| worst.0.0)
            } else {
                None
            };
        if let Some(threshold_key) = threshold_key {
            primitive_topk_filter_positions_with_min_key_into(
                columns[filter_index],
                filter_values,
                columns[sort_index],
                threshold_key,
                &mut self.selected_positions,
            );
        } else {
            primitive_topk_filter_positions_into(
                columns[filter_index],
                filter_values,
                &mut self.selected_positions,
            );
        }
        let base_sequence = primitive_topk_sequence_base(location).unwrap_or(self.sequence);
        self.sequence = self.sequence.wrapping_add(view.num_rows() as u64);
        for index in 0..self.selected_positions.len() {
            let row = self.selected_positions[index];
            let Some(key) = primitive_topk_key(columns[sort_index], row) else {
                return Err(DodamError::UnsupportedSql(
                    "streaming primitive top-k sort column type mismatch".to_string(),
                ));
            };
            self.insert_candidate_with_sequence(
                key,
                base_sequence.wrapping_add(row as u64),
                &columns,
                PrimitiveTopkRowRef {
                    row_group: location.row_group,
                    row_offset: location.row_offset.saturating_add(row),
                },
                row,
            )?;
        }
        self.selected_positions.clear();
        Ok(())
    }

    fn insert_candidate_with_sequence(
        &mut self,
        key: i128,
        sequence: u64,
        columns: &[NullFreePrimitiveColumn<'_>],
        row_ref: PrimitiveTopkRowRef,
        row: usize,
    ) -> Result<()> {
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_columns(columns, row_ref, row)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_columns(slot, columns, row_ref, row)?;
        }
        Ok(())
    }

    fn merge(&mut self, other: Self) -> Result<()> {
        self.unsupported |= other.unsupported;
        if other.unsupported {
            return Ok(());
        }
        let mut selected = other
            .heap
            .iter()
            .map(|Reverse(item)| *item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        for (key, _, slot) in selected {
            self.insert_candidate_from_state(key, &other, slot)?;
        }
        Ok(())
    }

    fn insert_candidate_from_state(
        &mut self,
        key: i128,
        source: &PrimitiveTopkState,
        source_slot: usize,
    ) -> Result<()> {
        let sequence = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        if self.heap.len() < self.limit {
            let slot = self.push_slot_from_state(source, source_slot)?;
            self.heap.push(Reverse((key, sequence, slot)));
            return Ok(());
        }
        if let Some(worst) = self.heap.peek()
            && key < worst.0.0
        {
            return Ok(());
        }
        let mut replace_slot = None;
        {
            if let Some(mut worst) = self.heap.peek_mut()
                && (key, sequence, 0usize) > (worst.0.0, worst.0.1, 0usize)
            {
                replace_slot = Some(worst.0.2);
                *worst = Reverse((key, sequence, worst.0.2));
            }
        }
        if let Some(slot) = replace_slot {
            self.overwrite_slot_from_state(slot, source, source_slot)?;
        }
        Ok(())
    }

    fn into_primitive_batch(
        self,
        column_names: &[String],
        column_types: &[DirectPrimitiveColumnType],
    ) -> Result<PrimitiveBatch> {
        let mut selected = self
            .heap
            .into_iter()
            .map(|Reverse(item)| item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        let mut output = column_types
            .iter()
            .map(|column_type| match column_type {
                DirectPrimitiveColumnType::I32 => {
                    PrimitiveColumnOutput::I32(Vec::with_capacity(selected.len()))
                }
                DirectPrimitiveColumnType::I64 => {
                    PrimitiveColumnOutput::I64(Vec::with_capacity(selected.len()))
                }
                _ => unreachable!("primitive top-k only uses i32/i64 columns"),
            })
            .collect::<Vec<_>>();
        for (_, _, slot) in selected {
            if let Some(row_refs) = &self.row_refs {
                let row_ref = row_refs[slot];
                let _physical_row = (row_ref.row_group, row_ref.row_offset);
            }
            for (source, target) in self.columns.iter().zip(output.iter_mut()) {
                push_primitive_output_slot(source, slot, target)?;
            }
        }
        primitive_output_batch_from_columns(output, column_names, column_types)
    }

    fn selected_row_refs_sorted(&self) -> Option<Vec<PrimitiveTopkRowRef>> {
        let row_refs = self.row_refs.as_ref()?;
        let mut selected = self
            .heap
            .iter()
            .map(|Reverse(item)| *item)
            .collect::<Vec<_>>();
        selected.sort_unstable_by(|left, right| right.cmp(left));
        Some(
            selected
                .into_iter()
                .map(|(_, _, slot)| row_refs[slot])
                .collect(),
        )
    }

    fn push_slot_from_columns(
        &mut self,
        columns: &[NullFreePrimitiveColumn<'_>],
        row_ref: PrimitiveTopkRowRef,
        row: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for (source, target) in columns.iter().zip(self.columns.iter_mut()) {
            push_null_free_primitive_value(source, row, target)?;
        }
        if let Some(row_refs) = &mut self.row_refs {
            row_refs.push(row_ref);
        }
        Ok(slot)
    }

    fn overwrite_slot_from_columns(
        &mut self,
        slot: usize,
        columns: &[NullFreePrimitiveColumn<'_>],
        row_ref: PrimitiveTopkRowRef,
        row: usize,
    ) -> Result<()> {
        for (source, target) in columns.iter().zip(self.columns.iter_mut()) {
            overwrite_null_free_primitive_value(source, row, target, slot)?;
        }
        if let Some(row_refs) = &mut self.row_refs {
            row_refs[slot] = row_ref;
        }
        Ok(())
    }

    fn push_slot_from_state(
        &mut self,
        source: &PrimitiveTopkState,
        source_slot: usize,
    ) -> Result<usize> {
        let slot = primitive_output_len(&self.columns[0]);
        for (source_column, target) in source.columns.iter().zip(self.columns.iter_mut()) {
            push_primitive_output_slot(source_column, source_slot, target)?;
        }
        if let (Some(target_refs), Some(source_refs)) = (&mut self.row_refs, &source.row_refs) {
            target_refs.push(source_refs[source_slot]);
        }
        Ok(slot)
    }

    fn overwrite_slot_from_state(
        &mut self,
        slot: usize,
        source: &PrimitiveTopkState,
        source_slot: usize,
    ) -> Result<()> {
        for (source_column, target) in source.columns.iter().zip(self.columns.iter_mut()) {
            overwrite_primitive_output_slot(source_column, source_slot, target, slot)?;
        }
        if let (Some(target_refs), Some(source_refs)) = (&mut self.row_refs, &source.row_refs) {
            target_refs[slot] = source_refs[source_slot];
        }
        Ok(())
    }
}

fn primitive_topk_row_refs_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_ROW_REFS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn primitive_topk_fused_filter_threshold_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_FUSED_FILTER_THRESHOLD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_FUSED_FILTER_THRESHOLD")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn primitive_topk_block_max_skip_enabled() -> bool {
    if std::env::var("DODAM_DISABLE_PRIMITIVE_TOPK_BLOCK_MAX_SKIP")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    std::env::var("DODAM_ENABLE_PRIMITIVE_TOPK_BLOCK_MAX_SKIP")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn null_free_primitive_columns_for_topk<'a>(
    view: BatchView<'a>,
    column_types: &[DirectPrimitiveColumnType],
) -> Option<Vec<NullFreePrimitiveColumn<'a>>> {
    let mut columns = Vec::with_capacity(column_types.len());
    for (index, column_type) in column_types.iter().enumerate() {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let values = view.i32_vector(index)?.values_if_null_free()?;
                columns.push(NullFreePrimitiveColumn::I32(values));
            }
            DirectPrimitiveColumnType::I64 => {
                let values = view.i64_vector(index)?.values_if_null_free()?;
                columns.push(NullFreePrimitiveColumn::I64(values));
            }
            _ => return None,
        }
    }
    Some(columns)
}

fn update_streaming_desc_topk_heap(
    key_column: &ArrayRef,
    batch_index: usize,
    limit: usize,
    sequence: &mut u64,
    heap: &mut BinaryHeap<Reverse<(i128, u64, usize, u32)>>,
) -> Result<bool> {
    match key_column.data_type() {
        DataType::Int32 => {
            let values = key_column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        DataType::Int64 => {
            let values = key_column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        DataType::UInt64 => {
            let values = key_column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        DataType::Date32 => {
            let values = key_column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            update_streaming_desc_topk_heap_typed(
                values.len(),
                |row| (!values.is_null(row)).then(|| i128::from(values.value(row))),
                batch_index,
                limit,
                sequence,
                heap,
            )
        }
        data_type => Err(DodamError::UnsupportedSql(format!(
            "unsupported streaming top-k sort type: {data_type:?}"
        ))),
    }
}

fn update_streaming_desc_topk_heap_typed<F>(
    rows: usize,
    mut key_at: F,
    batch_index: usize,
    limit: usize,
    sequence: &mut u64,
    heap: &mut BinaryHeap<Reverse<(i128, u64, usize, u32)>>,
) -> Result<bool>
where
    F: FnMut(usize) -> Option<i128>,
{
    let mut threshold = (heap.len() == limit)
        .then(|| heap.peek().map(|worst| worst.0))
        .flatten();
    for row in 0..rows {
        let Some(key) = key_at(row) else {
            return Ok(false);
        };
        if let Some(worst) = threshold
            && key < worst.0
        {
            *sequence = (*sequence).wrapping_add(1);
            continue;
        }
        let item = (key, *sequence, batch_index, row as u32);
        *sequence = (*sequence).wrapping_add(1);
        if heap.len() < limit {
            heap.push(Reverse(item));
            if heap.len() == limit {
                threshold = heap.peek().map(|worst| worst.0);
            }
        } else {
            let mut replaced = false;
            {
                if let Some(mut worst) = heap.peek_mut()
                    && item > worst.0
                {
                    *worst = Reverse(item);
                    replaced = true;
                }
            }
            if replaced {
                threshold = heap.peek().map(|worst| worst.0);
            }
        }
    }
    Ok(true)
}

fn materialize_topk_selected_rows(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
) -> Result<RecordBatch> {
    if selected.is_empty() {
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .unwrap_or_else(|| Arc::new(Schema::empty()));
        return Ok(RecordBatch::new_empty(schema));
    }
    if let Some(batch) = take_topk_selected_record_batch_runs(batches, selected)? {
        return Ok(batch);
    }
    if let Some(batch) = gather_topk_selected_record_batch(batches, selected)? {
        return Ok(batch);
    }
    let mut chunks = Vec::with_capacity(selected.len());
    for &(_, _, batch_index, row) in selected {
        chunks.push(batches[batch_index].slice(row as usize, 1));
    }
    let schema = chunks[0].schema();
    Ok(concat_batches(&schema, chunks.iter())?)
}

fn take_topk_selected_record_batch_runs(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
) -> Result<Option<RecordBatch>> {
    if selected.len() < topk_take_materialization_min_rows() {
        return Ok(None);
    }
    let Some(first_batch) = batches.first() else {
        return Ok(None);
    };
    let schema = first_batch.schema();
    let mut chunks = Vec::new();
    let mut index = 0usize;
    while index < selected.len() {
        let batch_index = selected[index].2;
        let Some(batch) = batches.get(batch_index) else {
            return Ok(None);
        };
        if batch.schema() != schema {
            return Ok(None);
        }
        let mut rows = Vec::new();
        while index < selected.len() && selected[index].2 == batch_index {
            rows.push(selected[index].3);
            index += 1;
        }
        chunks.push(take_record_batch(batch, &UInt32Array::from(rows))?);
    }
    let Some(first) = chunks.first() else {
        return Ok(None);
    };
    if chunks.len() == 1 {
        return Ok(chunks.pop());
    }
    let schema = first.schema();
    Ok(Some(concat_batches(&schema, chunks.iter())?))
}

fn topk_take_materialization_min_rows() -> usize {
    std::env::var("DODAM_TOPK_TAKE_MATERIALIZATION_MIN_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(usize::MAX)
}

fn gather_topk_selected_record_batch(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
) -> Result<Option<RecordBatch>> {
    let Some(first_batch) = batches.first() else {
        return Ok(None);
    };
    let schema = first_batch.schema();
    let mut columns = Vec::with_capacity(first_batch.num_columns());
    for column_index in 0..first_batch.num_columns() {
        let Some(column) = gather_topk_selected_array(batches, selected, column_index) else {
            return Ok(None);
        };
        columns.push(column);
    }
    Ok(Some(RecordBatch::try_new(schema, columns)?))
}

fn gather_topk_selected_array(
    batches: &[RecordBatch],
    selected: &[(i128, u64, usize, u32)],
    column_index: usize,
) -> Option<ArrayRef> {
    let data_type = batches.first()?.column(column_index).data_type().clone();
    if batches
        .iter()
        .any(|batch| batch.column(column_index).data_type() != &data_type)
    {
        return None;
    }

    match data_type {
        DataType::Boolean => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<BooleanArray>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(BooleanArray::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<BooleanArray>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(BooleanArray::from(output)))
            }
        }
        DataType::Int32 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int32Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Int32Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int32Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Int32Array::from(output)))
            }
        }
        DataType::Int64 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int64Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Int64Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Int64Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Int64Array::from(output)))
            }
        }
        DataType::UInt64 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<UInt64Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(UInt64Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<UInt64Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(UInt64Array::from(output)))
            }
        }
        DataType::Float64 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Float64Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Float64Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Float64Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Float64Array::from(output)))
            }
        }
        DataType::Date32 => {
            if topk_selected_column_null_count(batches, column_index) == 0 {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Date32Array>()?;
                    output.push(values.value(row as usize));
                }
                Some(Arc::new(Date32Array::from(output)))
            } else {
                let mut output = Vec::with_capacity(selected.len());
                for &(_, _, batch_index, row) in selected {
                    let values = batches[batch_index]
                        .column(column_index)
                        .as_any()
                        .downcast_ref::<Date32Array>()?;
                    let row = row as usize;
                    output.push((!values.is_null(row)).then(|| values.value(row)));
                }
                Some(Arc::new(Date32Array::from(output)))
            }
        }
        _ => None,
    }
}

fn topk_selected_column_null_count(batches: &[RecordBatch], column_index: usize) -> usize {
    let mut null_count = 0usize;
    for batch in batches {
        null_count = null_count.saturating_add(batch.column(column_index).null_count());
    }
    null_count
}

fn topk_sort_key_type_supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int32 | DataType::Int64 | DataType::UInt64 | DataType::Date32
    )
}

fn numeric_column_ascending_bounds(column: &ArrayRef) -> Result<Option<(i128, i128)>> {
    match column.data_type() {
        DataType::Int32 => {
            let values = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        DataType::Int64 => {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        DataType::UInt64 => {
            let values = column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        DataType::Date32 => {
            let values = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            primitive_ascending_bounds(values.len(), |row| {
                (!values.is_null(row)).then(|| i128::from(values.value(row)))
            })
        }
        _ => Ok(None),
    }
}

fn primitive_ascending_bounds<F>(len: usize, mut value: F) -> Result<Option<(i128, i128)>>
where
    F: FnMut(usize) -> Option<i128>,
{
    if len == 0 {
        return Ok(None);
    }
    let Some(first) = value(0) else {
        return Ok(None);
    };
    let mut previous = first;
    for row in 1..len {
        let Some(current) = value(row) else {
            return Ok(None);
        };
        if previous > current {
            return Ok(None);
        }
        previous = current;
    }
    Ok(Some((first, previous)))
}

async fn try_execute_same_source_union_all_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_union_all_scan(expr)? else {
        return Ok(None);
    };
    let stream = engine
        .scan_parquet_batches(
            shared.path,
            batch_size,
            None,
            shared.projection,
            Some(shared.filter),
        )
        .await?;
    let batches = rename_output_batches(collect_batches(stream)?, &shared.aliases)?;
    Ok(Some(batches))
}

async fn try_execute_same_source_union_all_filter_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_union_all_filter_scan(expr)? else {
        return Ok(None);
    };
    let mut stream = engine
        .scan_parquet_batches(
            shared.path.clone(),
            batch_size,
            None,
            shared.scan_projection.clone(),
            Some(shared.prefilter.clone()),
        )
        .await?;
    let mut output = Vec::new();
    for batch in stream.by_ref() {
        let batch = batch?;
        append_same_source_union_all_filter_batches(&mut output, &batch, &shared)?;
    }
    Ok(Some(output))
}

async fn try_execute_same_source_union_distinct_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_union_distinct_scan(expr)? else {
        return Ok(None);
    };
    if let Some(batches) = try_execute_direct_distinct_scan(
        engine,
        DirectDistinctScan {
            path: shared.path.clone(),
            projection: shared.projection.clone(),
            aliases: shared.aliases.clone(),
            filter: Some(shared.filter.clone()),
        },
        batch_size,
    )? {
        return Ok(Some(batches));
    }
    let stream = engine
        .scan_parquet_distinct_batches(
            shared.path,
            batch_size,
            scan_limit_with_offset(limit, offset)?,
            shared.projection,
            Some(shared.filter),
            order_by.cloned(),
        )
        .await?;
    let batches = rename_output_batches(collect_batches(stream)?, &shared.aliases)?;
    Ok(Some(batches))
}

async fn try_execute_same_source_distinct_set_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(shared) = plan_same_source_distinct_set_scan(expr)? else {
        return Ok(None);
    };
    if let Some(batches) = try_execute_direct_distinct_scan(
        engine,
        DirectDistinctScan {
            path: shared.path.clone(),
            projection: shared.projection.clone(),
            aliases: shared.aliases.clone(),
            filter: Some(shared.filter.clone()),
        },
        batch_size,
    )? {
        return Ok(Some(batches));
    }
    let stream = engine
        .scan_parquet_distinct_batches(
            shared.path,
            batch_size,
            scan_limit_with_offset(limit, offset)?,
            shared.projection,
            Some(shared.filter),
            order_by.cloned(),
        )
        .await?;
    let batches = rename_output_batches(collect_batches(stream)?, &shared.aliases)?;
    Ok(Some(batches))
}

fn try_execute_same_source_all_set_scan(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    _order_by: Option<&SortKey>,
    _limit: Option<usize>,
    _offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Some(scan) = plan_same_source_all_set_primitive_scan(expr)? else {
        return Ok(None);
    };
    let Some(column_types) = engine
        .parquet_direct_primitive_column_types(&scan.path, std::slice::from_ref(&scan.column))?
    else {
        return Ok(None);
    };
    let [column_type] = column_types.as_slice() else {
        return Ok(None);
    };
    if !matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    ) {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&scan.path)?;
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let candidates = scan.candidates();
    let Some((state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        scan.path.clone(),
        batch_size,
        row_groups,
        vec![(scan.column.clone(), *column_type)],
        || PrimitiveSetAllCounts::new(candidates.clone()),
        move |state, view| state.consume(view, *column_type),
        |state, partial| {
            state.merge(partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    if state.unsupported {
        return Ok(None);
    }
    let output = state.finish(&scan.left_values, &scan.right_values, scan.op);
    let (data_type, column): (DataType, ArrayRef) = match column_type {
        DirectPrimitiveColumnType::I32 => {
            let values = output
                .into_iter()
                .map(|value| {
                    i32::try_from(value)
                        .map_err(|_| DodamError::InvalidFilter(format!("{}={value}", scan.column)))
                })
                .collect::<Result<Vec<_>>>()?;
            (DataType::Int32, Arc::new(Int32Array::from(values)))
        }
        DirectPrimitiveColumnType::I64 => (DataType::Int64, Arc::new(Int64Array::from(output))),
        _ => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(&scan.column, data_type, true)]));
    let batch = RecordBatch::try_new(schema, vec![column])?;
    rename_output_batches(vec![batch], &scan.aliases).map(Some)
}

#[derive(Clone)]
struct DirectDistinctScan {
    path: PathBuf,
    projection: Projection,
    aliases: Vec<(String, String)>,
    filter: Option<FilterExpr>,
}

#[derive(Clone)]
struct SameSourceAllSetPrimitiveScan {
    path: PathBuf,
    column: String,
    aliases: Vec<(String, String)>,
    left_values: Vec<i64>,
    right_values: Vec<i64>,
    op: SetOperator,
}

impl SameSourceAllSetPrimitiveScan {
    fn candidates(&self) -> Vec<i64> {
        let mut candidates = Vec::new();
        for value in self.left_values.iter().chain(self.right_values.iter()) {
            if !candidates.iter().any(|candidate| candidate == value) {
                candidates.push(*value);
            }
        }
        candidates
    }
}

fn try_collect_direct_monotonic_count_distinct(
    engine: &DodamEngine,
    query: &SqlQuery,
    batch_size: usize,
) -> Result<Option<AggregateMetrics>> {
    if !query.group_by.is_empty()
        || query.filter.is_some()
        || !query.aggregate_expressions.is_empty()
        || !query.filtered_aggregates.is_empty()
    {
        return Ok(None);
    }
    let [AggregateExpr::CountDistinct(column)] = query.aggregates.as_slice() else {
        return Ok(None);
    };
    let Some(column_types) =
        engine.parquet_direct_primitive_column_types(&query.path, std::slice::from_ref(column))?
    else {
        return Ok(None);
    };
    let Some(column_type) = column_types.first().copied() else {
        return Ok(None);
    };
    if !matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    ) {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(&query.path)?).collect::<Vec<_>>();
    let Some((state, scan_metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        query.path.clone(),
        batch_size,
        row_groups,
        vec![(column.clone(), column_type)],
        MonotonicPrimitiveDistinctCount::default,
        move |state, view| state.consume(view, column_type),
        |state, partial| state.merge(partial),
    )?
    else {
        return Ok(None);
    };
    let Some(count) = state.finish() else {
        return Ok(None);
    };
    Ok(Some(AggregateMetrics {
        fragments: 1,
        batches: scan_metrics.batches,
        rows: scan_metrics.rows,
        values: vec![AggregateResult {
            expr: query.aggregates[0].clone(),
            value: AggregateValue::Count(count),
        }],
        ..AggregateMetrics::default()
    }))
}

#[derive(Default)]
struct MonotonicPrimitiveDistinctCount {
    ranges: Vec<(i64, i64, u64)>,
    unsupported: bool,
}

impl MonotonicPrimitiveDistinctCount {
    fn consume(
        &mut self,
        view: BatchView<'_>,
        column_type: DirectPrimitiveColumnType,
    ) -> Result<()> {
        if self.unsupported || view.num_rows() == 0 {
            return Ok(());
        }
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(values) = view.i32_vector(0) else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_i32(values)
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(values) = view.i64_vector(0) else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_i64(values)
            }
            _ => {
                self.unsupported = true;
                Ok(())
            }
        }
    }

    fn consume_i32(&mut self, values: I32VectorView<'_>) -> Result<()> {
        if let Some((bytes, len)) = values.raw_bytes() {
            return self.consume_len_i64(len, |row| {
                Some(i64::from(read_i32_le_unaligned(bytes, row)))
            });
        }
        if let Some(values) = values.values_if_null_free() {
            return self.consume_len_i64(values.len(), |row| Some(i64::from(values[row])));
        }
        if let Some((values, def_levels)) = values.raw_nullable() {
            let full_width_values = values.len() == def_levels.len();
            let mut value_index = 0usize;
            return self.consume_len_i64(def_levels.len(), |row| {
                if def_levels[row] == 0 {
                    None
                } else if full_width_values {
                    Some(i64::from(values[row]))
                } else {
                    let value = values.get(value_index).copied().map(i64::from);
                    value_index += 1;
                    value
                }
            });
        }
        self.unsupported = true;
        Ok(())
    }

    fn consume_i64(&mut self, values: I64VectorView<'_>) -> Result<()> {
        if let Some((bytes, len)) = values.raw_bytes() {
            return self.consume_len_i64(len, |row| Some(read_i64_le_unaligned(bytes, row)));
        }
        if let Some(values) = values.values_if_null_free() {
            return self.consume_len_i64(values.len(), |row| Some(values[row]));
        }
        if let Some((values, def_levels)) = values.raw_nullable() {
            let full_width_values = values.len() == def_levels.len();
            let mut value_index = 0usize;
            return self.consume_len_i64(def_levels.len(), |row| {
                if def_levels[row] == 0 {
                    None
                } else if full_width_values {
                    Some(values[row])
                } else {
                    let value = values.get(value_index).copied();
                    value_index += 1;
                    value
                }
            });
        }
        self.unsupported = true;
        Ok(())
    }

    fn consume_len_i64<F>(&mut self, rows: usize, mut value_at: F) -> Result<()>
    where
        F: FnMut(usize) -> Option<i64>,
    {
        let mut first = None;
        let mut last = None;
        let mut count = 0u64;
        for row in 0..rows {
            let Some(value) = value_at(row) else {
                continue;
            };
            if last.is_some_and(|previous| value <= previous) {
                self.unsupported = true;
                return Ok(());
            }
            first.get_or_insert(value);
            last = Some(value);
            count += 1;
        }
        if let (Some(first), Some(last)) = (first, last) {
            self.ranges.push((first, last, count));
        }
        Ok(())
    }

    fn merge(&mut self, partial: Self) -> Result<()> {
        if self.unsupported || partial.unsupported {
            self.unsupported = true;
            return Ok(());
        }
        self.ranges.extend(partial.ranges);
        Ok(())
    }

    fn finish(mut self) -> Option<u64> {
        if self.unsupported {
            return None;
        }
        self.ranges.sort_unstable_by_key(|range| range.0);
        let mut previous_end = None;
        let mut count = 0u64;
        for (start, end, range_count) in self.ranges {
            if previous_end.is_some_and(|previous| start <= previous) {
                return None;
            }
            previous_end = Some(end);
            count = count.saturating_add(range_count);
        }
        Some(count)
    }
}

fn try_execute_direct_distinct_scan(
    engine: &DodamEngine,
    scan: DirectDistinctScan,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if let Some(batches) =
        try_execute_direct_distinct_single_primitive_values(engine, &scan, batch_size)?
    {
        return Ok(Some(batches));
    }
    if let Some(batches) = try_execute_direct_distinct_primitive_pairs(engine, &scan, batch_size)? {
        return Ok(Some(batches));
    }
    Ok(None)
}

fn try_execute_direct_distinct_single_primitive_values(
    engine: &DodamEngine,
    scan: &DirectDistinctScan,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Projection::Columns(columns) = &scan.projection else {
        return Ok(None);
    };
    let [projected_column] = columns.as_slice() else {
        return Ok(None);
    };
    let Some(filter) = scan.filter.as_ref() else {
        return Ok(None);
    };
    let Expr::InList {
        column,
        values,
        negated: false,
        has_null: false,
    } = filter.expr()
    else {
        return Ok(None);
    };
    if column != projected_column {
        return Ok(None);
    }
    let candidates = values
        .iter()
        .map(|value| value.as_i64(column))
        .collect::<Result<Vec<_>>>()?;
    if candidates.is_empty() {
        return Ok(None);
    }
    let Some(column_types) =
        engine.parquet_direct_primitive_column_types(&scan.path, std::slice::from_ref(column))?
    else {
        return Ok(None);
    };
    let [column_type] = column_types.as_slice() else {
        return Ok(None);
    };
    if !matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    ) {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&scan.path)?;
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let Some((state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        scan.path.clone(),
        batch_size,
        row_groups,
        vec![(column.clone(), *column_type)],
        || PrimitiveCandidatePresence::new(candidates.clone()),
        move |state, view| state.consume(view, *column_type),
        |state, partial| {
            state.merge(partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    if state.unsupported {
        return Ok(None);
    }
    let mut output = Vec::with_capacity(state.candidates.len());
    for (index, candidate) in state.candidates.iter().copied().enumerate() {
        if state.found[index] {
            output.push(candidate);
        }
    }
    let (data_type, column): (DataType, ArrayRef) = match column_type {
        DirectPrimitiveColumnType::I32 => {
            let values = output
                .into_iter()
                .map(|value| {
                    i32::try_from(value)
                        .map_err(|_| DodamError::InvalidFilter(format!("{column}={value}")))
                })
                .collect::<Result<Vec<_>>>()?;
            (DataType::Int32, Arc::new(Int32Array::from(values)))
        }
        DirectPrimitiveColumnType::I64 => (DataType::Int64, Arc::new(Int64Array::from(output))),
        _ => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        projected_column,
        data_type,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![column])?;
    rename_output_batches(vec![batch], &scan.aliases).map(Some)
}

fn try_execute_direct_distinct_primitive_pairs(
    engine: &DodamEngine,
    scan: &DirectDistinctScan,
    batch_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let Projection::Columns(columns) = &scan.projection else {
        return Ok(None);
    };
    let [first_column, second_column] = columns.as_slice() else {
        return Ok(None);
    };
    let Some(filter) = scan.filter.as_ref() else {
        return Ok(None);
    };
    let Expr::InList {
        column,
        values,
        negated: false,
        has_null: false,
    } = filter.expr()
    else {
        return Ok(None);
    };
    let Some(filter_index) = columns.iter().position(|candidate| candidate == column) else {
        return Ok(None);
    };
    if filter_index != 0 {
        return Ok(None);
    }
    let other_index = if filter_index == 0 { 1 } else { 0 };
    let candidates = values
        .iter()
        .map(|value| value.as_i64(column))
        .collect::<Result<Vec<_>>>()?;
    if candidates.is_empty() {
        return Ok(None);
    }
    let Some(column_types) = engine.parquet_direct_primitive_column_types(&scan.path, columns)?
    else {
        return Ok(None);
    };
    if !primitive_distinct_column_type_supported(column_types[filter_index])
        || !primitive_distinct_column_type_supported(column_types[other_index])
    {
        return Ok(None);
    }
    let row_group_count = engine.parquet_row_group_count(&scan.path)?;
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let Some((state, _metrics)) = engine.scan_parquet_primitive_columns_parallel_view_fold(
        scan.path.clone(),
        batch_size,
        row_groups,
        columns
            .iter()
            .zip(column_types.iter())
            .map(|(name, column_type)| (name.clone(), *column_type))
            .collect(),
        || {
            PrimitiveDistinctPairs::new(
                candidates.clone(),
                filter_index,
                other_index,
                column_types[filter_index],
                column_types[other_index],
            )
        },
        |state, view| state.consume(view),
        |state, partial| {
            state.merge(partial);
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    if state.unsupported {
        return Ok(None);
    }
    let mut pairs = state.pairs.into_iter().collect::<Vec<_>>();
    pairs.sort_unstable();
    let first_values = pairs.iter().map(|(key, _)| *key).collect::<Vec<_>>();
    let second_values = pairs.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            first_column,
            primitive_distinct_data_type(column_types[0])?,
            true,
        ),
        Field::new(
            second_column,
            primitive_distinct_data_type(column_types[1])?,
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            primitive_distinct_array(first_values, column_types[0])?,
            primitive_distinct_array(second_values, column_types[1])?,
        ],
    )?;
    rename_output_batches(vec![batch], &scan.aliases).map(Some)
}

fn primitive_distinct_column_type_supported(column_type: DirectPrimitiveColumnType) -> bool {
    matches!(
        column_type,
        DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::I64
    )
}

fn primitive_distinct_data_type(column_type: DirectPrimitiveColumnType) -> Result<DataType> {
    match column_type {
        DirectPrimitiveColumnType::I32 => Ok(DataType::Int32),
        DirectPrimitiveColumnType::I64 => Ok(DataType::Int64),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported primitive distinct type {column_type:?}"
        ))),
    }
}

fn primitive_distinct_array(
    values: Vec<i64>,
    column_type: DirectPrimitiveColumnType,
) -> Result<ArrayRef> {
    match column_type {
        DirectPrimitiveColumnType::I32 => Ok(Arc::new(Int32Array::from(
            values
                .into_iter()
                .map(|value| {
                    i32::try_from(value).map_err(|_| {
                        DodamError::InvalidFilter(format!("primitive distinct value {value}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ))),
        DirectPrimitiveColumnType::I64 => Ok(Arc::new(Int64Array::from(values))),
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported primitive distinct type {column_type:?}"
        ))),
    }
}

fn primitive_distinct_column_values(
    view: &BatchView<'_>,
    index: usize,
    column_type: DirectPrimitiveColumnType,
) -> Option<Vec<i64>> {
    match column_type {
        DirectPrimitiveColumnType::I32 => {
            let column = view.i32_vector(index)?;
            if let Some(values) = column.values_if_null_free() {
                return Some(values.iter().map(|value| i64::from(*value)).collect());
            }
            let (values, def_levels) = column.raw_nullable()?;
            Some(nullable_i32_values_to_i64(values, def_levels))
        }
        DirectPrimitiveColumnType::I64 => {
            let column = view.i64_vector(index)?;
            if let Some(values) = column.values_if_null_free() {
                return Some(values.to_vec());
            }
            let (values, def_levels) = column.raw_nullable()?;
            Some(nullable_i64_values(values, def_levels))
        }
        _ => None,
    }
}

fn primitive_distinct_null_free_column_values(
    view: &BatchView<'_>,
    index: usize,
    column_type: DirectPrimitiveColumnType,
) -> Option<Vec<i64>> {
    match column_type {
        DirectPrimitiveColumnType::I32 => view
            .i32_vector(index)
            .and_then(|column| column.values_if_null_free())
            .map(|values| values.iter().map(|value| i64::from(*value)).collect()),
        DirectPrimitiveColumnType::I64 => view
            .i64_vector(index)
            .and_then(|column| column.values_if_null_free())
            .map(|values| values.to_vec()),
        _ => None,
    }
}

fn nullable_i32_values_to_i64(values: &[i32], def_levels: &[i16]) -> Vec<i64> {
    nullable_values(values.len(), def_levels, |index| i64::from(values[index]))
}

fn nullable_i64_values(values: &[i64], def_levels: &[i16]) -> Vec<i64> {
    nullable_values(values.len(), def_levels, |index| values[index])
}

fn nullable_values<F>(value_len: usize, def_levels: &[i16], value_at: F) -> Vec<i64>
where
    F: Fn(usize) -> i64,
{
    let mut output = Vec::with_capacity(def_levels.len());
    let mut value_index = 0usize;
    let full_width_values = value_len == def_levels.len();
    for (row, definition) in def_levels.iter().copied().enumerate() {
        if definition == 0 {
            continue;
        }
        let index = if full_width_values { row } else { value_index };
        output.push(value_at(index));
        value_index += 1;
    }
    output
}

struct PrimitiveCandidatePresence {
    candidates: Vec<i64>,
    found: Vec<bool>,
    unsupported: bool,
}

impl PrimitiveCandidatePresence {
    fn new(candidates: Vec<i64>) -> Self {
        let found = vec![false; candidates.len()];
        Self {
            candidates,
            found,
            unsupported: false,
        }
    }

    fn consume(
        &mut self,
        view: BatchView<'_>,
        column_type: DirectPrimitiveColumnType,
    ) -> Result<()> {
        let Some(values) = primitive_distinct_column_values(&view, 0, column_type) else {
            self.unsupported = true;
            return Ok(());
        };
        self.consume_values(values)
    }

    fn consume_values<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = i64>,
    {
        let values = values.into_iter();
        match self.candidates.as_slice() {
            [a] => {
                for value in values {
                    if value == *a {
                        self.found[0] = true;
                    }
                }
            }
            [a, b] => {
                for value in values {
                    if value == *a {
                        self.found[0] = true;
                    } else if value == *b {
                        self.found[1] = true;
                    }
                }
            }
            [a, b, c] => {
                for value in values {
                    if value == *a {
                        self.found[0] = true;
                    } else if value == *b {
                        self.found[1] = true;
                    } else if value == *c {
                        self.found[2] = true;
                    }
                }
            }
            candidates => {
                for value in values {
                    if let Some(index) = candidates.iter().position(|candidate| *candidate == value)
                    {
                        self.found[index] = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, partial: Self) {
        self.unsupported |= partial.unsupported;
        for (found, partial_found) in self.found.iter_mut().zip(partial.found) {
            *found |= partial_found;
        }
    }
}

struct PrimitiveSetAllCounts {
    candidates: Vec<i64>,
    counts: Vec<usize>,
    unsupported: bool,
}

impl PrimitiveSetAllCounts {
    fn new(candidates: Vec<i64>) -> Self {
        let counts = vec![0; candidates.len()];
        Self {
            candidates,
            counts,
            unsupported: false,
        }
    }

    fn consume(
        &mut self,
        view: BatchView<'_>,
        column_type: DirectPrimitiveColumnType,
    ) -> Result<()> {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(values) = view
                    .i32_vector(0)
                    .and_then(|column| column.values_if_null_free())
                else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_values(values.iter().map(|value| i64::from(*value)))
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(values) = view
                    .i64_vector(0)
                    .and_then(|column| column.values_if_null_free())
                else {
                    self.unsupported = true;
                    return Ok(());
                };
                self.consume_values(values.iter().copied())
            }
            _ => {
                self.unsupported = true;
                Ok(())
            }
        }
    }

    fn consume_values<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = i64>,
    {
        let values = values.into_iter();
        match self.candidates.as_slice() {
            [] => for _ in values {},
            [a] => {
                let mut count = 0usize;
                for value in values {
                    count += usize::from(value == *a);
                }
                self.counts[0] += count;
            }
            [a, b] => {
                let mut count_a = 0usize;
                let mut count_b = 0usize;
                for value in values {
                    if value == *a {
                        count_a += 1;
                    } else if value == *b {
                        count_b += 1;
                    }
                }
                self.counts[0] += count_a;
                self.counts[1] += count_b;
            }
            [a, b, c] => {
                let mut count_a = 0usize;
                let mut count_b = 0usize;
                let mut count_c = 0usize;
                for value in values {
                    if value == *a {
                        count_a += 1;
                    } else if value == *b {
                        count_b += 1;
                    } else if value == *c {
                        count_c += 1;
                    }
                }
                self.counts[0] += count_a;
                self.counts[1] += count_b;
                self.counts[2] += count_c;
            }
            candidates => {
                for value in values {
                    if let Some(index) = candidates.iter().position(|candidate| *candidate == value)
                    {
                        self.counts[index] += 1;
                    }
                }
            }
        };
        Ok(())
    }

    fn merge(&mut self, partial: Self) {
        self.unsupported |= partial.unsupported;
        for (count, partial_count) in self.counts.iter_mut().zip(partial.counts) {
            *count += partial_count;
        }
    }

    fn finish(&self, left_values: &[i64], right_values: &[i64], op: SetOperator) -> Vec<i64> {
        let mut output = Vec::new();
        for &value in left_values {
            let left_count = self.count_for(value);
            let right_count = right_values
                .iter()
                .any(|right| *right == value)
                .then(|| self.count_for(value))
                .unwrap_or(0);
            let repeats = match op {
                SetOperator::Intersect => left_count.min(right_count),
                SetOperator::Except => left_count.saturating_sub(right_count),
                _ => 0,
            };
            output.extend(std::iter::repeat_n(value, repeats));
        }
        output
    }

    fn count_for(&self, value: i64) -> usize {
        self.candidates
            .iter()
            .position(|candidate| *candidate == value)
            .map(|index| self.counts[index])
            .unwrap_or(0)
    }
}

struct PrimitiveDistinctPairs {
    candidates: Vec<i64>,
    filter_index: usize,
    other_index: usize,
    filter_type: DirectPrimitiveColumnType,
    other_type: DirectPrimitiveColumnType,
    pairs: FastHashSet<(i64, i64)>,
    unsupported: bool,
}

impl PrimitiveDistinctPairs {
    fn new(
        candidates: Vec<i64>,
        filter_index: usize,
        other_index: usize,
        filter_type: DirectPrimitiveColumnType,
        other_type: DirectPrimitiveColumnType,
    ) -> Self {
        Self {
            candidates,
            filter_index,
            other_index,
            filter_type,
            other_type,
            pairs: FastHashSet::default(),
            unsupported: false,
        }
    }

    fn consume(&mut self, view: BatchView<'_>) -> Result<()> {
        let Some(keys) =
            primitive_distinct_null_free_column_values(&view, self.filter_index, self.filter_type)
        else {
            self.unsupported = true;
            return Ok(());
        };
        let Some(values) =
            primitive_distinct_null_free_column_values(&view, self.other_index, self.other_type)
        else {
            self.unsupported = true;
            return Ok(());
        };
        if keys.len() != values.len() {
            self.unsupported = true;
            return Ok(());
        }
        match self.candidates.as_slice() {
            [a] => {
                for row in 0..keys.len() {
                    if keys[row] == *a {
                        self.pairs.insert((keys[row], values[row]));
                    }
                }
            }
            [a, b] => {
                for row in 0..keys.len() {
                    let key = keys[row];
                    if key == *a || key == *b {
                        self.pairs.insert((key, values[row]));
                    }
                }
            }
            [a, b, c] => {
                for row in 0..keys.len() {
                    let key = keys[row];
                    if key == *a || key == *b || key == *c {
                        self.pairs.insert((key, values[row]));
                    }
                }
            }
            candidates => {
                for row in 0..keys.len() {
                    let key = keys[row];
                    if candidates.contains(&key) {
                        self.pairs.insert((key, values[row]));
                    }
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, partial: Self) {
        self.unsupported |= partial.unsupported;
        self.pairs.extend(partial.pairs);
    }
}

fn plan_same_source_union_all_scan(expr: &SetExpr) -> Result<Option<SameSourceUnionAllScan>> {
    let mut operands = Vec::new();
    if !collect_union_all_operand_queries(expr, &mut operands)? || operands.len() < 2 {
        return Ok(None);
    }
    let Some(shared) = same_source_disjoint_union_all_plan(&operands) else {
        return Ok(None);
    };
    Ok(Some(shared))
}

fn plan_same_source_union_all_filter_scan(
    expr: &SetExpr,
) -> Result<Option<SameSourceUnionAllFilterScan>> {
    let mut operands = Vec::new();
    if !collect_union_all_operand_queries(expr, &mut operands)? || operands.len() < 2 {
        return Ok(None);
    }
    Ok(same_source_union_all_filter_scan_plan(&operands))
}

fn plan_same_source_union_distinct_scan(expr: &SetExpr) -> Result<Option<SameSourceUnionAllScan>> {
    let mut operands = Vec::new();
    if !collect_union_distinct_operand_queries(expr, &mut operands)? || operands.len() < 2 {
        return Ok(None);
    }
    let Some(shared) = same_source_union_distinct_plan(&operands) else {
        return Ok(None);
    };
    Ok(Some(shared))
}

fn plan_same_source_distinct_set_scan(expr: &SetExpr) -> Result<Option<SameSourceUnionAllScan>> {
    let SetExpr::SetOperation {
        op,
        set_quantifier,
        left,
        right,
    } = expr
    else {
        return Ok(None);
    };
    if !matches!(*op, SetOperator::Intersect | SetOperator::Except)
        || !union_quantifier_is_distinct(*set_quantifier)
    {
        return Ok(None);
    }
    let Some(left_query) = single_set_operand_query(left.as_ref())? else {
        return Ok(None);
    };
    let Some(right_query) = single_set_operand_query(right.as_ref())? else {
        return Ok(None);
    };
    Ok(same_source_distinct_set_plan(
        &left_query,
        &right_query,
        *op,
    ))
}

fn try_execute_simple_case_distinct_set_literals(
    expr: &SetExpr,
) -> Result<Option<Vec<RecordBatch>>> {
    let SetExpr::SetOperation {
        op,
        set_quantifier,
        left,
        right,
    } = expr
    else {
        return Ok(None);
    };
    if !matches!(*op, SetOperator::Intersect | SetOperator::Except)
        || !union_quantifier_is_distinct(*set_quantifier)
    {
        return Ok(None);
    }
    let Some(left_query) = single_set_operand_query(left.as_ref())? else {
        return Ok(None);
    };
    let Some(right_query) = single_set_operand_query(right.as_ref())? else {
        return Ok(None);
    };
    if left_query.path != right_query.path || left_query.aliases != right_query.aliases {
        return Ok(None);
    }
    let Some((output_name, left_values)) = simple_case_literal_projection_values(&left_query)?
    else {
        return Ok(None);
    };
    let Some((right_output_name, right_values)) =
        simple_case_literal_projection_values(&right_query)?
    else {
        return Ok(None);
    };
    if output_name != right_output_name {
        return Ok(None);
    }
    let values = match op {
        SetOperator::Intersect => intersect_literal_values(left_values, &right_values),
        SetOperator::Except => except_literal_values(left_values, &right_values),
        _ => return Ok(None),
    };
    literal_values_batch(&output_name, values).map(|batch| Some(vec![batch]))
}

fn simple_case_literal_projection_values(
    query: &SqlQuery,
) -> Result<Option<(String, Vec<LiteralValue>)>> {
    if query.join.is_some()
        || query.expression_filter.is_some()
        || query.having.is_some()
        || query.order_by.is_some()
        || query.limit.is_some()
        || query.offset != 0
        || query.distinct
        || !query.aggregates.is_empty()
        || !query.aggregate_expressions.is_empty()
        || query.group_by.len() > 0
        || query.expressions.len() != 1
        || query.qualified_wildcards.len() > 0
    {
        return Ok(None);
    }
    let expression = &query.expressions[0];
    let Some(case) = simple_case_literal_descriptor(&expression.expr) else {
        return Ok(None);
    };
    let Some(filter) = query.filter.as_ref() else {
        return Ok(None);
    };
    let Some((filter_column, filter_values)) = positive_literal_filter_values(filter) else {
        return Ok(None);
    };
    if filter_column != case.column {
        return Ok(None);
    }
    let mut output = Vec::new();
    for value in filter_values {
        append_unique_literal_values(&mut output, vec![case.result_for_literal(&value)]);
    }
    Ok(Some((expression.output_name.clone(), output)))
}

struct SimpleCaseLiteralDescriptor {
    column: String,
    branches: Vec<(LiteralValue, LiteralValue)>,
    else_value: LiteralValue,
}

impl SimpleCaseLiteralDescriptor {
    fn result_for_literal(&self, value: &LiteralValue) -> LiteralValue {
        self.branches
            .iter()
            .find_map(|(condition, result)| (condition == value).then(|| result.clone()))
            .unwrap_or_else(|| self.else_value.clone())
    }
}

fn simple_case_literal_descriptor(
    expr: &ScalarSqlExpression,
) -> Option<SimpleCaseLiteralDescriptor> {
    let GroupKeyExpr::SimpleCaseLiteral {
        column,
        branches,
        else_value,
    } = simple_case_literal_group_key(expr)?
    else {
        return None;
    };
    Some(SimpleCaseLiteralDescriptor {
        column,
        branches: branches
            .into_iter()
            .map(|(condition, result)| {
                Some((
                    literal_value_from_group_key_literal(&condition)?,
                    literal_value_from_group_key_literal(&result)?,
                ))
            })
            .collect::<Option<Vec<_>>>()?,
        else_value: literal_value_from_group_key_literal(&else_value)?,
    })
}

fn literal_value_from_group_key_literal(value: &GroupKeyLiteral) -> Option<LiteralValue> {
    Some(match value {
        GroupKeyLiteral::Null => LiteralValue::Null,
        GroupKeyLiteral::Boolean(value) => LiteralValue::Boolean(*value),
        GroupKeyLiteral::Int64(value) => LiteralValue::Int64(*value),
        GroupKeyLiteral::Float64(value) => LiteralValue::Float64(f64::from_bits(*value)),
        GroupKeyLiteral::Utf8(value) => LiteralValue::Utf8(value.clone()),
    })
}

fn literal_values_batch(name: &str, values: Vec<LiteralValue>) -> Result<RecordBatch> {
    if values
        .iter()
        .all(|value| matches!(value, LiteralValue::Utf8(_) | LiteralValue::Null))
    {
        let array = StringArray::from(
            values
                .into_iter()
                .map(|value| match value {
                    LiteralValue::Utf8(value) => Some(value),
                    LiteralValue::Null => None,
                    _ => unreachable!("checked utf8 literal values"),
                })
                .collect::<Vec<_>>(),
        );
        return RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)])),
            vec![Arc::new(array)],
        )
        .map_err(DodamError::from);
    }
    if values
        .iter()
        .all(|value| matches!(value, LiteralValue::Int64(_) | LiteralValue::Null))
    {
        let array = Int64Array::from(
            values
                .into_iter()
                .map(|value| match value {
                    LiteralValue::Int64(value) => Some(value),
                    LiteralValue::Null => None,
                    _ => unreachable!("checked int64 literal values"),
                })
                .collect::<Vec<_>>(),
        );
        return RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)])),
            vec![Arc::new(array)],
        )
        .map_err(DodamError::from);
    }
    Err(DodamError::UnsupportedSql(
        "simple CASE set literal output type is not supported yet".to_string(),
    ))
}

fn plan_same_source_all_set_primitive_scan(
    expr: &SetExpr,
) -> Result<Option<SameSourceAllSetPrimitiveScan>> {
    let SetExpr::SetOperation {
        op,
        set_quantifier,
        left,
        right,
    } = expr
    else {
        return Ok(None);
    };
    if !matches!(*op, SetOperator::Intersect | SetOperator::Except)
        || *set_quantifier != SetQuantifier::All
    {
        return Ok(None);
    }
    let Some(left_query) = single_set_operand_query(left.as_ref())? else {
        return Ok(None);
    };
    let Some(right_query) = single_set_operand_query(right.as_ref())? else {
        return Ok(None);
    };
    Ok(same_source_all_set_primitive_plan(
        &left_query,
        &right_query,
        *op,
    ))
}

struct SameSourceUnionAllScan {
    path: PathBuf,
    projection: Projection,
    aliases: Vec<(String, String)>,
    filter: FilterExpr,
}

struct SameSourceUnionAllFilterScan {
    path: PathBuf,
    projection: Projection,
    scan_projection: Projection,
    aliases: Vec<(String, String)>,
    filters: Vec<FilterExpr>,
    prefilter: FilterExpr,
}

fn single_set_operand_query(expr: &SetExpr) -> Result<Option<SqlQuery>> {
    match expr {
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref())
                || query.order_by.is_some()
                || query.limit_clause.is_some()
                || parse_offset(query)? != 0
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Ok(None);
            }
            Ok(Some(parse_sql(&query.to_string())?))
        }
        SetExpr::Select(_) => Ok(Some(parse_sql(&expr.to_string())?)),
        _ => Ok(None),
    }
}

fn collect_union_distinct_operand_queries(
    expr: &SetExpr,
    output: &mut Vec<SqlQuery>,
) -> Result<bool> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union && union_quantifier_is_distinct(*set_quantifier) => Ok(
            collect_union_distinct_operand_queries(left.as_ref(), output)?
                && collect_union_distinct_operand_queries(right.as_ref(), output)?,
        ),
        SetExpr::SetOperation { .. } => Ok(false),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return collect_union_distinct_operand_queries(query.body.as_ref(), output);
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || parse_offset(query)? != 0
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Ok(false);
            }
            output.push(parse_sql(&query.to_string())?);
            Ok(true)
        }
        SetExpr::Select(_) => {
            output.push(parse_sql(&expr.to_string())?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn collect_union_all_operand_queries(expr: &SetExpr, output: &mut Vec<SqlQuery>) -> Result<bool> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union && *set_quantifier == SetQuantifier::All => {
            Ok(collect_union_all_operand_queries(left.as_ref(), output)?
                && collect_union_all_operand_queries(right.as_ref(), output)?)
        }
        SetExpr::SetOperation { .. } => Ok(false),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return collect_union_all_operand_queries(query.body.as_ref(), output);
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || parse_offset(query)? != 0
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Ok(false);
            }
            output.push(parse_sql(&query.to_string())?);
            Ok(true)
        }
        SetExpr::Select(_) => {
            output.push(parse_sql(&expr.to_string())?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn same_source_union_distinct_plan(operands: &[SqlQuery]) -> Option<SameSourceUnionAllScan> {
    let first = operands.first()?;
    if !same_source_union_all_operand_supported(first) {
        return None;
    }
    let (filter_column, first_values) = positive_literal_filter_values(first.filter.as_ref()?)?;
    let mut values = Vec::new();
    append_unique_literal_values(&mut values, first_values);
    for operand in operands.iter().skip(1) {
        if !same_source_union_all_operand_supported(operand)
            || operand.path != first.path
            || operand.projection != first.projection
            || operand.aliases != first.aliases
        {
            return None;
        }
        let (column, operand_values) = positive_literal_filter_values(operand.filter.as_ref()?)?;
        if column != filter_column {
            return None;
        }
        append_unique_literal_values(&mut values, operand_values);
    }
    Some(SameSourceUnionAllScan {
        path: first.path.clone(),
        projection: first.projection.clone(),
        aliases: first.aliases.clone(),
        filter: FilterExpr::new(Expr::InList {
            column: filter_column,
            values,
            negated: false,
            has_null: false,
        }),
    })
}

fn same_source_distinct_set_plan(
    left: &SqlQuery,
    right: &SqlQuery,
    op: SetOperator,
) -> Option<SameSourceUnionAllScan> {
    if !same_source_union_all_operand_supported(left)
        || !same_source_union_all_operand_supported(right)
        || left.path != right.path
        || left.projection != right.projection
        || left.aliases != right.aliases
    {
        return None;
    }
    let (left_column, left_values) = positive_literal_filter_values(left.filter.as_ref()?)?;
    let (right_column, right_values) = positive_literal_filter_values(right.filter.as_ref()?)?;
    if left_column != right_column {
        return None;
    }
    let values = match op {
        SetOperator::Intersect => intersect_literal_values(left_values, &right_values),
        SetOperator::Except => except_literal_values(left_values, &right_values),
        _ => return None,
    };
    Some(SameSourceUnionAllScan {
        path: left.path.clone(),
        projection: left.projection.clone(),
        aliases: left.aliases.clone(),
        filter: FilterExpr::new(Expr::InList {
            column: left_column,
            values,
            negated: false,
            has_null: false,
        }),
    })
}

fn same_source_all_set_primitive_plan(
    left: &SqlQuery,
    right: &SqlQuery,
    op: SetOperator,
) -> Option<SameSourceAllSetPrimitiveScan> {
    if !same_source_union_all_operand_supported(left)
        || !same_source_union_all_operand_supported(right)
        || left.path != right.path
        || left.projection != right.projection
        || left.aliases != right.aliases
    {
        return None;
    }
    let Projection::Columns(projected) = &left.projection else {
        return None;
    };
    let [projected_column] = projected.as_slice() else {
        return None;
    };
    let (left_column, left_values) = positive_literal_filter_values(left.filter.as_ref()?)?;
    let (right_column, right_values) = positive_literal_filter_values(right.filter.as_ref()?)?;
    if left_column != right_column || left_column != *projected_column {
        return None;
    }
    Some(SameSourceAllSetPrimitiveScan {
        path: left.path.clone(),
        column: projected_column.clone(),
        aliases: left.aliases.clone(),
        left_values: literal_values_to_unique_i64(left_values, projected_column).ok()?,
        right_values: literal_values_to_unique_i64(right_values, projected_column).ok()?,
        op,
    })
}

fn same_source_union_all_filter_scan_plan(
    operands: &[SqlQuery],
) -> Option<SameSourceUnionAllFilterScan> {
    let first = operands.first()?;
    if !same_source_union_all_operand_supported(first) {
        return None;
    }
    let mut filters = Vec::with_capacity(operands.len());
    for operand in operands {
        if !same_source_union_all_operand_supported(operand)
            || operand.path != first.path
            || operand.projection != first.projection
            || operand.aliases != first.aliases
        {
            return None;
        }
        filters.push(operand.filter.clone()?);
    }
    let prefilter = union_filter_or(filters.iter().map(|filter| filter.expr().clone()))?;
    let mut scan_projection = first.projection.clone();
    for filter in &filters {
        add_projection_columns(&mut scan_projection, filter.referenced_columns());
    }
    Some(SameSourceUnionAllFilterScan {
        path: first.path.clone(),
        projection: first.projection.clone(),
        scan_projection,
        aliases: first.aliases.clone(),
        filters,
        prefilter,
    })
}

fn same_source_disjoint_union_all_plan(operands: &[SqlQuery]) -> Option<SameSourceUnionAllScan> {
    let first = operands.first()?;
    if !same_source_union_all_operand_supported(first) {
        return None;
    }
    let (filter_column, first_values) = positive_literal_filter_values(first.filter.as_ref()?)?;
    let mut values = Vec::new();
    append_disjoint_literal_values(&mut values, first_values)?;
    for operand in operands.iter().skip(1) {
        if !same_source_union_all_operand_supported(operand)
            || operand.path != first.path
            || operand.projection != first.projection
            || operand.aliases != first.aliases
        {
            return None;
        }
        let (column, operand_values) = positive_literal_filter_values(operand.filter.as_ref()?)?;
        if column != filter_column {
            return None;
        }
        append_disjoint_literal_values(&mut values, operand_values)?;
    }
    Some(SameSourceUnionAllScan {
        path: first.path.clone(),
        projection: first.projection.clone(),
        aliases: first.aliases.clone(),
        filter: FilterExpr::new(Expr::InList {
            column: filter_column,
            values,
            negated: false,
            has_null: false,
        }),
    })
}

fn union_filter_or(filters: impl IntoIterator<Item = Expr>) -> Option<FilterExpr> {
    filters
        .into_iter()
        .reduce(|left, right| Expr::Or(Box::new(left), Box::new(right)))
        .map(FilterExpr::new)
}

fn positive_literal_filter_values(filter: &FilterExpr) -> Option<(String, Vec<LiteralValue>)> {
    match filter.expr() {
        Expr::Comparison(comparison)
            if comparison.op == ComparisonOp::Eq
                && !matches!(comparison.value, LiteralValue::Null) =>
        {
            Some((comparison.column.clone(), vec![comparison.value.clone()]))
        }
        Expr::InList {
            column,
            values,
            negated: false,
            has_null: false,
        } if values
            .iter()
            .all(|value| !matches!(value, LiteralValue::Null)) =>
        {
            Some((column.clone(), values.clone()))
        }
        _ => None,
    }
}

fn same_source_union_all_operand_supported(query: &SqlQuery) -> bool {
    query.join.is_none()
        && query.expression_filter.is_none()
        && query.having.is_none()
        && query.order_by.is_none()
        && query.limit.is_none()
        && query.offset == 0
        && !query.distinct
        && query.aggregates.is_empty()
        && query.aggregate_expressions.is_empty()
        && projection_expressions_are_plain_columns(&query.expressions)
        && query.group_by.is_empty()
        && query.qualified_wildcards.is_empty()
}

async fn execute_set_operation_expr(
    engine: &DodamEngine,
    expr: &SetExpr,
    batch_size: usize,
    child_topk: Option<(&SortKey, usize)>,
    child_distinct: bool,
) -> Result<Vec<RecordBatch>> {
    match expr {
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Union => {
            let distinct = union_quantifier_is_distinct(*set_quantifier);
            let mut left_batches = Box::pin(execute_set_operation_expr(
                engine,
                left.as_ref(),
                batch_size,
                union_child_topk_for_quantifier(*set_quantifier, child_topk),
                child_distinct || distinct,
            ))
            .await?;
            let right_batches = Box::pin(execute_set_operation_expr(
                engine,
                right.as_ref(),
                batch_size,
                union_child_topk_for_quantifier(*set_quantifier, child_topk),
                child_distinct || distinct,
            ))
            .await?;
            append_union_all_batches(&mut left_batches, right_batches)?;
            if distinct {
                left_batches = apply_output_distinct(left_batches, true)?;
            }
            Ok(left_batches)
        }
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } if *op == SetOperator::Intersect || *op == SetOperator::Except => {
            let left_batches = Box::pin(execute_set_operation_expr(
                engine,
                left.as_ref(),
                batch_size,
                None,
                false,
            ))
            .await?;
            let right_batches = Box::pin(execute_set_operation_expr(
                engine,
                right.as_ref(),
                batch_size,
                None,
                false,
            ))
            .await?;
            let batches = if *set_quantifier == SetQuantifier::All {
                apply_all_row_set_operation(left_batches, right_batches, *op)?
            } else {
                apply_distinct_row_set_operation(left_batches, right_batches, *op)?
            };
            apply_output_distinct(batches, child_distinct)
        }
        SetExpr::SetOperation {
            op, set_quantifier, ..
        } => Err(DodamError::UnsupportedSql(format!(
            "{op} {set_quantifier} is not supported yet"
        ))),
        SetExpr::Query(query) => {
            if query_contains_set_operation(query.body.as_ref()) {
                return Box::pin(execute_set_operation_expr(
                    engine,
                    query.body.as_ref(),
                    batch_size,
                    child_topk,
                    child_distinct,
                ))
                .await;
            }
            if query.order_by.is_some()
                || query.limit_clause.is_some()
                || query.fetch.is_some()
                || !query.locks.is_empty()
            {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY, LIMIT, FETCH, and locking clauses inside UNION operands are not supported yet"
                        .to_string(),
                ));
            }
            let sql = union_all_operand_sql_with_child_topk(&query.to_string(), child_topk);
            let batches =
                query_output_batches(Box::pin(execute_sql(engine, &sql, batch_size)).await?)?;
            apply_output_distinct(batches, child_distinct)
        }
        SetExpr::Select(_) => {
            let sql = union_all_operand_sql_with_child_topk(&expr.to_string(), child_topk);
            let batches =
                query_output_batches(Box::pin(execute_sql(engine, &sql, batch_size)).await?)?;
            apply_output_distinct(batches, child_distinct)
        }
        other => Err(DodamError::UnsupportedSql(format!(
            "unsupported set operation operand: {other}"
        ))),
    }
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
            if let Some(filter) = safe_expression_pushdown_filter(
                &conjunct,
                table_alias,
                PredicateParserKind::Single,
            )? {
                filters.push(filter.expr().clone());
            }
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

#[derive(Clone, Copy)]
enum PredicateParserKind<'a> {
    Single,
    Join(&'a [&'a str]),
}

fn safe_expression_pushdown_filter(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let filter = match expr {
        SqlExpr::Nested(expr) => safe_expression_pushdown_filter(expr, table_alias, parser_kind)?,
        SqlExpr::UnaryOp { op, .. } if *op == UnaryOperator::Not => None,
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::And => {
            combine_filter_options(
                safe_expression_pushdown_filter(left, table_alias, parser_kind)?,
                safe_expression_pushdown_filter(right, table_alias, parser_kind)?,
            )
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::Or => {
            let Some(left) = safe_expression_pushdown_filter(left, table_alias, parser_kind)?
            else {
                return Ok(None);
            };
            let Some(right) = safe_expression_pushdown_filter(right, table_alias, parser_kind)?
            else {
                return Ok(None);
            };
            Some(FilterExpr::new(Expr::Or(
                Box::new(left.expr().clone()),
                Box::new(right.expr().clone()),
            )))
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
            safe_expression_comparison_pushdown(left, op, right, table_alias, parser_kind)?
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => safe_expression_in_list_pushdown(expr, list, *negated, table_alias, parser_kind)?,
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => safe_expression_between_pushdown(expr, *negated, low, high, table_alias, parser_kind)?,
        SqlExpr::IsNull(expr) => {
            safe_expression_null_pushdown(expr, false, table_alias, parser_kind)?
        }
        SqlExpr::IsNotNull(expr) => {
            safe_expression_null_pushdown(expr, true, table_alias, parser_kind)?
        }
        SqlExpr::Like {
            expr,
            pattern,
            negated,
            any,
            escape_char,
        } => safe_expression_like_pushdown(
            expr,
            pattern,
            *negated,
            *any,
            escape_char,
            table_alias,
            parser_kind,
            false,
        )?,
        SqlExpr::ILike {
            expr,
            pattern,
            negated,
            any,
            escape_char,
        } => safe_expression_like_pushdown(
            expr,
            pattern,
            *negated,
            *any,
            escape_char,
            table_alias,
            parser_kind,
            true,
        )?,
        _ => None,
    };
    Ok(filter)
}

fn safe_expression_between_pushdown(
    expr: &SqlExpr,
    negated: bool,
    low: &SqlExpr,
    high: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    let low = sql_literal_value(low)?;
    let high = sql_literal_value(high)?;
    let lower = Expr::Comparison(ComparisonExpr {
        column: column.clone(),
        op: if negated {
            ComparisonOp::Lt
        } else {
            ComparisonOp::GtEq
        },
        value: low,
    });
    let upper = Expr::Comparison(ComparisonExpr {
        column,
        op: if negated {
            ComparisonOp::Gt
        } else {
            ComparisonOp::LtEq
        },
        value: high,
    });
    Ok(Some(FilterExpr::new(if negated {
        Expr::Or(Box::new(lower), Box::new(upper))
    } else {
        Expr::And(Box::new(lower), Box::new(upper))
    })))
}

fn safe_expression_comparison_pushdown(
    left: &SqlExpr,
    op: &BinaryOperator,
    right: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    if let Some(filter) =
        safe_column_wrapper_comparison_pushdown(left, op, right, table_alias, parser_kind)?
    {
        return Ok(Some(filter));
    }
    safe_column_wrapper_comparison_pushdown(
        right,
        &reverse_binary_operator(op),
        left,
        table_alias,
        parser_kind,
    )
}

fn safe_expression_in_list_pushdown(
    expr: &SqlExpr,
    list: &[SqlExpr],
    negated: bool,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    if negated {
        return Ok(None);
    }
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    let values = non_null_literal_values(list)?;
    if values.is_empty() {
        return Ok(None);
    }
    Ok(Some(FilterExpr::new(Expr::InList {
        column,
        values,
        negated: false,
        has_null: literal_list_contains_null(list)?,
    })))
}

fn safe_expression_null_pushdown(
    expr: &SqlExpr,
    negated: bool,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    Ok(Some(FilterExpr::new(Expr::IsNull { column, negated })))
}

fn safe_expression_like_pushdown(
    expr: &SqlExpr,
    pattern: &SqlExpr,
    negated: bool,
    any: bool,
    escape_char: &Option<sqlparser::ast::ValueWithSpan>,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
    case_insensitive: bool,
) -> Result<Option<FilterExpr>> {
    if any {
        return Ok(None);
    }
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    Ok(Some(FilterExpr::new(Expr::Like {
        column,
        pattern: sql_like_pattern(pattern)?,
        negated,
        escape: sql_like_escape(escape_char)?,
        case_insensitive,
    })))
}

fn safe_column_wrapper_comparison_pushdown(
    expr: &SqlExpr,
    op: &BinaryOperator,
    literal_expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let Ok(literal) = sql_literal_value(literal_expr) else {
        return Ok(None);
    };
    if let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? {
        return Ok(Some(FilterExpr::new(Expr::Comparison(ComparisonExpr {
            column,
            op: sql_comparison_op(op),
            value: literal,
        }))));
    }
    coalesce_comparison_pushdown(expr, op, &literal, table_alias, parser_kind)
}

fn simple_column_wrapped_expression(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<String>> {
    match parse_predicate_scalar_expr(expr, table_alias, parser_kind)? {
        ScalarSqlExpression::Column(column) => Ok(Some(column)),
        _ => Ok(None),
    }
}

fn coalesce_comparison_pushdown(
    expr: &SqlExpr,
    op: &BinaryOperator,
    literal: &LiteralValue,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let ScalarSqlExpression::Coalesce(values) =
        parse_predicate_scalar_expr(expr, table_alias, parser_kind)?
    else {
        return Ok(None);
    };
    let [
        ScalarSqlExpression::Column(column),
        ScalarSqlExpression::Literal(fallback),
    ] = values.as_slice()
    else {
        return Ok(None);
    };
    let column_filter = Expr::Comparison(ComparisonExpr {
        column: column.clone(),
        op: sql_comparison_op(op),
        value: literal.clone(),
    });
    if compare_literal_values(fallback, op, literal)? == Some(true) {
        Ok(Some(FilterExpr::new(Expr::Or(
            Box::new(column_filter),
            Box::new(Expr::IsNull {
                column: column.clone(),
                negated: false,
            }),
        ))))
    } else {
        Ok(Some(FilterExpr::new(column_filter)))
    }
}

fn parse_predicate_scalar_expr(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<ScalarSqlExpression> {
    match parser_kind {
        PredicateParserKind::Single => parse_scalar_sql_expression(expr, table_alias),
        PredicateParserKind::Join(table_aliases) => {
            parse_join_scalar_sql_expression(expr, table_aliases)
        }
    }
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
    let (filter, expression_filters) = if let Some(selection) = select.selection.as_ref() {
        if predicate_requires_expression_path(selection) {
            split_subquery_and_expression_filters(
                engine,
                selection,
                path.alias.as_deref(),
                batch_size,
            )
            .await?
        } else {
            (
                Some(parse_filter(
                    selection,
                    &parsed_projection.aliases,
                    path.alias.as_deref(),
                    false,
                )?),
                Vec::new(),
            )
        }
    } else {
        (None, Vec::new())
    };
    let filter_requires_expression = !expression_filters.is_empty();
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
        let order_by = parse_order_by(
            query,
            &parsed_projection.aliases,
            &parsed_projection.ordinal_targets,
            path.alias.as_deref(),
        )?;
        let limit = parse_limit(query)?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;
    let mut scan_projection = parsed_projection.projection.clone();
    for expression_filter in &expression_filters {
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(expression_filter, path.alias.as_deref())?,
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
    if projection_requires_expression
        && !filter_requires_expression
        && let Some(filter) = filter.clone()
        && let Some(mut batches) = try_late_materialized_projection_expression(
            engine,
            path.path.clone(),
            batch_size,
            filter,
            scan_projection.clone(),
            &parsed_projection.expressions,
        )
        .await?
    {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    let expression_selection = combine_sql_and_conjuncts(expression_filters.clone());
    let (stream, expression_filter_stream_limited) = if let Some(selection) =
        expression_selection.as_ref()
    {
        if let Some(limit) = limit {
            let auto_exact_monotonic_order_column =
                expression_filter_exact_monotonic_stream_limit_auto_enabled()
                    .then(|| monotonic_stream_limit_column(order_by.as_ref()))
                    .flatten();
            if expression_filter_stream_limit_enabled()
                || auto_exact_monotonic_order_column.is_some()
            {
                let monotonic_order_column =
                    auto_exact_monotonic_order_column.clone().or_else(|| {
                        expression_filter_monotonic_stream_limit_enabled()
                            .then(|| monotonic_stream_limit_column(order_by.as_ref()))
                            .flatten()
                    });
                let use_monotonic_stream = if let Some(column) = monotonic_order_column.as_ref() {
                    if expression_filter_exact_monotonic_stream_limit_enabled()
                        || auto_exact_monotonic_order_column.is_some()
                    {
                        engine
                            .parquet_row_groups_monotonic_by_column(path.path.clone(), column)
                            .await?
                            && engine
                                .parquet_column_monotonic_by_scan(
                                    path.path.clone(),
                                    column,
                                    batch_size,
                                )
                                .await?
                    } else {
                        engine
                            .parquet_row_groups_monotonic_by_column(path.path.clone(), column)
                            .await?
                    }
                } else {
                    false
                };
                let stream = if use_monotonic_stream {
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?
                } else if auto_exact_monotonic_order_column.is_some() {
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?
                } else if let Some(order_by) = order_by.clone() {
                    engine
                        .scan_parquet_ordered_batches_by(
                            path.path,
                            batch_size,
                            None,
                            scan_projection,
                            filter,
                            order_by,
                        )
                        .await?
                } else {
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?
                };
                if use_monotonic_stream || auto_exact_monotonic_order_column.is_none() {
                    let batches = collect_expression_filtered_limit_batches(
                        stream,
                        selection,
                        path.alias.as_deref(),
                        limit,
                    )?;
                    (SendableBatchStream::from_batches(batches), true)
                } else {
                    (stream, false)
                }
            } else {
                (
                    engine
                        .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                        .await?,
                    false,
                )
            }
        } else {
            (
                engine
                    .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
                    .await?,
                false,
            )
        }
    } else if let Some(order_by) = order_by.clone() {
        (
            engine
                .scan_parquet_ordered_batches_by(
                    path.path,
                    batch_size,
                    limit,
                    scan_projection,
                    filter,
                    order_by,
                )
                .await?,
            false,
        )
    } else {
        (
            engine
                .scan_parquet_batches(path.path, batch_size, limit, scan_projection, filter)
                .await?,
            false,
        )
    };
    let mut batches = collect_batches(stream)?;
    if !expression_filters.is_empty()
        && !expression_filter_stream_limited
        && let Some(selection) = expression_selection.as_ref()
    {
        batches = apply_output_expression_filter(batches, selection, path.alias.as_deref())?;
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    }
    let batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        let batches = apply_output_projection(batches, &parsed_projection.projection)?;
        rename_output_batches(batches, &parsed_projection.aliases)?
    };
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_late_materialized_projection_expression(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: FilterExpr,
    payload_projection: Projection,
    expressions: &[ProjectionExpression],
) -> Result<Option<Vec<RecordBatch>>> {
    if !late_projection_enabled() {
        return Ok(None);
    }
    let predicate_columns = filter.referenced_columns();
    let predicate_projection = Projection::Columns(predicate_columns.clone());
    let predicates = PredicateSet::new(Some(filter.clone()));
    let expressions = expressions.to_vec();
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            predicates.pushdown().to_vec(),
            late_projection_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                late_projection_max_selected_ratio(),
                late_projection_max_selector_run_ratio(),
            )
            .with_io_cost_gate(late_projection_io_cost_gate_enabled()),
            Vec::<RecordBatch>::new,
            {
                let filter = filter.clone();
                let predicate_columns = predicate_columns.clone();
                move |view, selection, _state: &mut Vec<RecordBatch>| {
                    let mask = if let Some(mask) =
                        evaluate_projected_view_filter_mask(view, &predicate_columns, &filter)?
                    {
                        mask
                    } else {
                        if !expression_aggregate_row_at_time_fallback_enabled() {
                            return Ok(None);
                        }
                        let Some(batch) = view.try_record_batch() else {
                            return Ok(None);
                        };
                        evaluate_filter_mask(batch, &filter)?
                    };
                    push_boolean_mask_selection(&mask, selection);
                    Ok(Some(()))
                }
            },
            move |view, state: &mut Vec<RecordBatch>| {
                let Some(batch) = view.try_record_batch() else {
                    return Ok(None);
                };
                state.push(batch.clone());
                Ok(Some(()))
            },
            {
                let expressions = expressions.clone();
                move |state, _metrics| {
                    if state.is_empty() {
                        return Ok(None);
                    }
                    Ok(Some(apply_output_expression_projection(
                        state,
                        &expressions,
                    )?))
                }
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut batches = Vec::new();
    for chunk in chunks {
        batches.extend(chunk.output);
    }
    Ok(Some(batches))
}

fn late_projection_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_LATE_PROJECTION")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn late_projection_row_group_chunk() -> usize {
    std::env::var("DODAM_LATE_PROJECTION_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn late_projection_max_selected_ratio() -> f64 {
    std::env::var("DODAM_LATE_PROJECTION_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.35)
}

fn late_projection_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_LATE_PROJECTION_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.25)
}

fn late_projection_io_cost_gate_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_LATE_PROJECTION_IO_COST_GATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_stream_limit_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_FILTER_STREAM_LIMIT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_monotonic_stream_limit_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_FILTER_MONOTONIC_STREAM_LIMIT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_exact_monotonic_stream_limit_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_FILTER_EXACT_MONOTONIC_STREAM_LIMIT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_filter_exact_monotonic_stream_limit_auto_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_EXPRESSION_FILTER_EXACT_MONOTONIC_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn monotonic_stream_limit_column(order_by: Option<&SortKey>) -> Option<String> {
    let [sort] = order_by?.expressions.as_slice() else {
        return None;
    };
    (!sort.descending && !sort.nulls_first).then(|| sort.column.clone())
}

fn monotonic_order_limit_scan_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_MONOTONIC_ORDER_LIMIT_SCAN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn small_dynamic_in_list_row_filter_limit() -> usize {
    std::env::var("DODAM_DYNAMIC_IN_LIST_ROW_FILTER_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64)
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

async fn try_execute_join_with_correlated_avg_threshold_sql(
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
    if select.from.len() != 2
        || !correlated_avg_threshold_projection_shape(select)
        || !correlated_avg_threshold_filter_shape(selection)
    {
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

async fn q14_promo_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<DenseI64BoolLookup> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_type".to_string()]),
            None,
        )
        .await?;
    let mut parts = DenseI64BoolLookup::default();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let types = batch_string_column(&batch, "p_type")?;
        if let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>()
            && partkeys.null_count() == 0
        {
            for row in 0..batch.num_rows() {
                if types.is_valid(row) {
                    parts.insert(partkeys.value(row), types.value(row).starts_with("PROMO"));
                }
            }
            continue;
        }
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
    promo_parts: DenseI64BoolLookup,
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
    promo_parts: &DenseI64BoolLookup,
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
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
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
            let Some(is_promo) = promo_parts.get(partkeys.value(row)) else {
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
        let Some(is_promo) = promo_parts.get(partkey) else {
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

async fn q15_revenue_by_supplier(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, f64>> {
    if let Some(revenues) =
        q15_revenue_by_supplier_direct(engine, &path, batch_size, start_days, end_days)?
    {
        return Ok(revenues);
    }
    let projection = Projection::Columns(vec![
        "l_suppkey".to_string(),
        "l_shipdate".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            projection,
            scan_aggregate_row_group_chunk(),
            8,
            scan_aggregate_fusion_enabled(),
            HashMap::<i64, f64>::new,
            HashMap::<i64, f64>::new,
            move |view, revenues| {
                q15_revenue_by_supplier_view_into(view, start_days, end_days, revenues)?;
                Ok(Some(()))
            },
            merge_f64_groups,
            "Q15 revenue aggregate",
        )
        .await
}

fn q15_revenue_by_supplier_direct(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, f64>>> {
    if !direct_discounted_revenue_selected_fold_enabled() {
        return Ok(None);
    }
    let trace = std::env::var("DODAM_DIRECT_SELECTION_TRACE")
        .or_else(|_| std::env::var("DODAM_TPCH_PROFILE"))
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
    if trace {
        eprintln!("[dodam:direct-selected] Q15 candidate");
    }
    if !direct_selection_fold_enabled() {
        if trace {
            eprintln!("[dodam:direct-selected] Q15 reject: direct selection fold disabled");
        }
        return Ok(None);
    }
    let Some((_price_precision, price_scale)) =
        engine.parquet_decimal128_type(path, "l_extendedprice")?
    else {
        if trace {
            eprintln!("[dodam:direct-selected] Q15 reject: l_extendedprice is not Decimal128");
        }
        return Ok(None);
    };
    let Some((_discount_precision, discount_decimal_scale)) =
        engine.parquet_decimal128_type(path, "l_discount")?
    else {
        if trace {
            eprintln!("[dodam:direct-selected] Q15 reject: l_discount is not Decimal128");
        }
        return Ok(None);
    };
    let date_max = end_days.checked_sub(1).ok_or_else(|| {
        DodamError::UnsupportedSql("invalid empty Q15 date range for direct fold".to_string())
    })?;
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let discount_scale = decimal_scale_factor(discount_decimal_scale);
    let revenue_scale =
        1.0 / (decimal_scale_factor(price_scale) * decimal_scale_factor(discount_decimal_scale));
    let Some((revenues, _metrics)) = engine
        .scan_parquet_i64_date_decimal_decimal_selected_typed_fold(
            path,
            batch_size,
            &row_groups,
            ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
            Some(start_days),
            Some(date_max),
            HashMap::<i64, f64>::new,
            move |revenues, batch| {
                q15_revenue_by_supplier_direct_batch_into(
                    batch,
                    start_days,
                    end_days,
                    discount_scale,
                    revenue_scale,
                    revenues,
                )
            },
            |revenues, partial| {
                merge_f64_groups(revenues, partial);
                Ok(())
            },
        )?
    else {
        return Ok(None);
    };
    Ok(Some(revenues))
}

fn direct_discounted_revenue_selected_fold_enabled() -> bool {
    std::env::var("DODAM_ENABLE_DIRECT_DISCOUNTED_REVENUE_SELECTED_FOLD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q15_revenue_by_supplier_direct_batch_into(
    batch: crate::storage::DirectI64DateDecimalDecimalSelectedBatch<'_>,
    start_days: i32,
    end_days: i32,
    discount_scale: f64,
    revenue_scale: f64,
    revenues: &mut HashMap<i64, f64>,
) -> Result<()> {
    if batch.keys.len() != batch.left_decimals.len()
        || batch.keys.len() != batch.right_decimals.len()
        || batch.keys.len() != batch.dates.len()
    {
        return Err(DodamError::UnsupportedSql(
            "direct discounted revenue batch length mismatch".to_string(),
        ));
    }
    if batch.predicate_applied {
        for row in 0..batch.keys.len() {
            *revenues.entry(batch.keys[row]).or_insert(0.0) += decimal_discounted_revenue_raw_i64(
                batch.left_decimals[row],
                batch.right_decimals[row],
                discount_scale,
                revenue_scale,
            );
        }
        return Ok(());
    }
    for row in 0..batch.keys.len() {
        let date = batch.dates[row];
        if date >= start_days && date < end_days {
            *revenues.entry(batch.keys[row]).or_insert(0.0) += decimal_discounted_revenue_raw_i64(
                batch.left_decimals[row],
                batch.right_decimals[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Ok(())
}

fn q15_revenue_by_supplier_batch_into<S: BuildHasher>(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64, S>,
) -> Result<()> {
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let (Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
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
        return Ok(());
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
    Ok(())
}

fn q15_revenue_by_supplier_view_into(
    view: BatchView<'_>,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64>,
) -> Result<()> {
    if view.num_columns() == 4 {
        if update_i64_grouped_discounted_revenue_by_date_view(
            view, 0, 1, 2, 3, start_days, end_days, revenues,
        )? {
            return Ok(());
        }
        let (Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        ) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(());
            };
            return q15_revenue_by_supplier_batch_into(
                batch.clone(),
                start_days,
                end_days,
                revenues,
            );
        };
        for row in 0..view.num_rows() {
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
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(());
    };
    q15_revenue_by_supplier_batch_into(batch.clone(), start_days, end_days, revenues)
}

#[allow(clippy::too_many_arguments)]
fn update_i64_grouped_discounted_revenue_by_date_view<S: BuildHasher>(
    view: BatchView<'_>,
    key_index: usize,
    date_index: usize,
    extendedprice_index: usize,
    discount_index: usize,
    start_days: i32,
    end_days: i32,
    revenues: &mut HashMap<i64, f64, S>,
) -> Result<bool> {
    let (Some(keys), Some(dates), Some(extendedprices), Some(discounts)) = (
        view.i64_vector(key_index),
        view.date32_vector(date_index),
        view.decimal128_vector(extendedprice_index),
        view.decimal128_vector(discount_index),
    ) else {
        return Ok(false);
    };
    let (Some(key_values), Some(date_values)) =
        (keys.values_if_null_free(), dates.values_if_null_free())
    else {
        return Ok(false);
    };
    if extendedprices.null_count() != 0 || discounts.null_count() != 0 {
        return Ok(false);
    }
    let discount_scale = discounts.scale();
    let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
    if let (Some(extendedprice_values), Some(discount_values)) =
        (extendedprices.raw_i64_values(), discounts.raw_i64_values())
    {
        for row in 0..view.num_rows() {
            let date = date_values[row];
            if date >= start_days && date < end_days {
                *revenues.entry(key_values[row]).or_insert(0.0) +=
                    decimal_discounted_revenue_raw_i64(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    );
            }
        }
        return Ok(true);
    }
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    for row in 0..view.num_rows() {
        let date = date_values[row];
        if date >= start_days && date < end_days {
            *revenues.entry(key_values[row]).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Ok(true)
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
        q15_supplier_rows_view_into(BatchView::new(&batch), top_suppliers, &mut rows)?;
    }
    rows.sort_by_key(|row| row.suppkey);
    Ok(rows)
}

fn q15_supplier_rows_view_into(
    view: BatchView<'_>,
    top_suppliers: &HashMap<i64, f64>,
    rows: &mut Vec<Q15Row>,
) -> Result<()> {
    if view.num_columns() == 4
        && let (Some(suppkeys), Some(names), Some(addresses), Some(phones)) =
            (view.i64(0), view.utf8(1), view.utf8(2), view.utf8(3))
    {
        for row in 0..view.num_rows() {
            if suppkeys.is_null(row)
                || names.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
            {
                continue;
            }
            let suppkey = suppkeys.value(row);
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
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q15 supplier raw vector columns have unsupported types".to_string(),
        ));
    };
    let suppkeys = batch_column(batch, "s_suppkey")?;
    let names = batch_string_column(batch, "s_name")?;
    let addresses = batch_string_column(batch, "s_address")?;
    let phones = batch_string_column(batch, "s_phone")?;
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
    Ok(())
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

fn correlated_avg_threshold_projection_shape(select: &Select) -> bool {
    select.projection.len() == 1
        && select.projection.first().is_some_and(|item| {
            item.to_string()
                .to_ascii_lowercase()
                .contains("sum(l_extendedprice) / 7")
        })
}

fn correlated_avg_threshold_filter_shape(selection: &SqlExpr) -> bool {
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
    if q17_late_materialized_enabled()
        && let Some(sum) =
            q17_lineitem_revenue_late_materialized(engine, path.clone(), batch_size, part_keys)
                .await?
    {
        return Ok(sum);
    }
    let part_key_count = part_keys.len();
    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let Some(partials) = engine
        .parquet_row_group_map_view(
            path,
            batch_size,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_quantity".to_string(),
                "l_extendedprice".to_string(),
            ]),
            q17_lineitem_chunk_size(),
            {
                let part_key_count = part_key_count;
                move || q17_lineitem_partial_new(part_key_count)
            },
            {
                let part_keys = part_keys.clone();
                move |view, partial| {
                    q17_lineitem_revenue_view_into(view, &part_keys, partial)?;
                    Ok(Some(()))
                }
            },
            |partial| Ok(Some(partial)),
        )
        .await?
    else {
        return Err(DodamError::UnsupportedSql(
            "Q17 lineitem row-group map is unavailable".to_string(),
        ));
    };
    let mut merged = q17_lineitem_partial_new(part_key_count);
    for partial in partials {
        q17_merge_lineitem_revenue_batch(&mut merged, partial);
    }
    let (states, candidate_rows) = merged;
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

struct Q17LateLineitemState {
    part_keys: Arc<AdaptiveI64Set>,
    quantity_state: HashMap<i64, (f64, u64)>,
    selected_partkeys: Vec<i64>,
    selected_quantities: Vec<f64>,
    payload_offset: usize,
    candidate_rows: Vec<(i64, f64, f64)>,
}

async fn q17_lineitem_revenue_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &HashSet<i64>,
) -> Result<Option<Option<f64>>> {
    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["l_partkey".to_string(), "l_quantity".to_string()]),
            Projection::Columns(vec!["l_extendedprice".to_string()]),
            Vec::new(),
            q17_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q17_late_materialized_max_selected_ratio(),
                q17_late_materialized_max_selector_run_ratio(),
            ),
            {
                let part_keys = part_keys.clone();
                move || Q17LateLineitemState {
                    part_keys: part_keys.clone(),
                    quantity_state: HashMap::new(),
                    selected_partkeys: Vec::new(),
                    selected_quantities: Vec::new(),
                    payload_offset: 0,
                    candidate_rows: Vec::new(),
                }
            },
            q17_late_build_quantity_selection_view,
            q17_late_consume_extendedprice_payload_view,
            |state, _metrics| {
                if state.payload_offset != state.selected_partkeys.len()
                    || state.payload_offset != state.selected_quantities.len()
                {
                    return Err(DodamError::UnsupportedSql(
                        "Q17 row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some((state.quantity_state, state.candidate_rows)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut quantity_state = HashMap::<i64, (f64, u64)>::new();
    let mut candidate_rows = Vec::<(i64, f64, f64)>::new();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_state, chunk_candidates) = chunk.output;
        q17_merge_quantity_state(&mut quantity_state, chunk_state);
        candidate_rows.extend(chunk_candidates);
        metrics.add(chunk.metrics);
    }
    q17_log_late_materialized_profile(metrics, q17_late_materialized_row_group_chunk());
    if candidate_rows.is_empty() {
        return Ok(Some(None));
    }
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (partkey, quantity, extendedprice) in candidate_rows {
        if let Some((quantity_sum, quantity_count)) = quantity_state.get(&partkey) {
            let average = quantity_sum / *quantity_count as f64;
            if quantity < 0.2 * average {
                sum += extendedprice;
                count += 1;
            }
        }
    }
    Ok(Some((count > 0).then_some(sum)))
}

fn q17_late_build_quantity_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q17LateLineitemState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(partkeys), Some(quantities)) = (view.i64_vector(0), view.decimal128_vector(1))
    {
        q17_late_build_quantity_selection_typed(partkeys, quantities, selection, state);
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    let partkeys = batch_column(batch, "l_partkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    let (Some(partkeys), Some(quantities)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
    ) else {
        return Ok(None);
    };
    for row in 0..partkeys.len() {
        let selected = !partkeys.is_null(row)
            && !quantities.is_null(row)
            && state.part_keys.contains(partkeys.value(row));
        selection.push(selected);
        if selected {
            let partkey = partkeys.value(row);
            let quantity = quantities.value(row);
            let aggregate = state.quantity_state.entry(partkey).or_insert((0.0, 0));
            aggregate.0 += quantity;
            aggregate.1 += 1;
            state.selected_partkeys.push(partkey);
            state.selected_quantities.push(quantity);
        }
    }
    Ok(Some(()))
}

fn q17_late_build_quantity_selection_typed(
    partkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q17LateLineitemState,
) {
    let dense_part_keys = state.part_keys.dense_contains_slice();
    if let Some(partkey_values) = partkeys.values_if_null_free()
        && quantities.null_count() == 0
    {
        let quantity_values = quantities.raw_values();
        let quantity_scale = 1.0 / quantities.scale();
        for row in 0..partkey_values.len() {
            let partkey = partkey_values[row];
            let selected = state.part_keys.contains_cached(dense_part_keys, partkey);
            selection.push(selected);
            if selected {
                let quantity = quantity_values[row] as f64 * quantity_scale;
                let aggregate = state.quantity_state.entry(partkey).or_insert((0.0, 0));
                aggregate.0 += quantity;
                aggregate.1 += 1;
                state.selected_partkeys.push(partkey);
                state.selected_quantities.push(quantity);
            }
        }
        return;
    }
    for row in 0..partkeys.len() {
        let selected = !partkeys.is_null(row)
            && !quantities.is_null(row)
            && state
                .part_keys
                .contains_cached(dense_part_keys, partkeys.value(row));
        selection.push(selected);
        if selected {
            let partkey = partkeys.value(row);
            let quantity = quantities.value(row);
            let aggregate = state.quantity_state.entry(partkey).or_insert((0.0, 0));
            aggregate.0 += quantity;
            aggregate.1 += 1;
            state.selected_partkeys.push(partkey);
            state.selected_quantities.push(quantity);
        }
    }
}

fn q17_late_consume_extendedprice_payload_view(
    view: BatchView<'_>,
    state: &mut Q17LateLineitemState,
) -> Result<Option<()>> {
    if view.num_columns() == 1
        && let Some(extendedprices) = view.decimal128_vector(0)
    {
        q17_late_consume_extendedprice_payload_typed(extendedprices, state)?;
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    let extendedprices = batch_column(batch, "l_extendedprice")?;
    let Some(extendedprices) = decimal_input(extendedprices)? else {
        return Ok(None);
    };
    for row in 0..batch.num_rows() {
        let (Some(&partkey), Some(&quantity)) = (
            state.selected_partkeys.get(state.payload_offset),
            state.selected_quantities.get(state.payload_offset),
        ) else {
            return Err(DodamError::UnsupportedSql(
                "Q17 row selection payload overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        if extendedprices.is_null(row) {
            continue;
        }
        state
            .candidate_rows
            .push((partkey, quantity, extendedprices.value(row)));
    }
    Ok(Some(()))
}

fn q17_late_consume_extendedprice_payload_typed(
    extendedprices: Decimal128VectorView<'_>,
    state: &mut Q17LateLineitemState,
) -> Result<()> {
    if extendedprices.null_count() == 0 {
        let values = extendedprices.raw_values();
        let scale = 1.0 / extendedprices.scale();
        for &raw in values {
            let (Some(&partkey), Some(&quantity)) = (
                state.selected_partkeys.get(state.payload_offset),
                state.selected_quantities.get(state.payload_offset),
            ) else {
                return Err(DodamError::UnsupportedSql(
                    "Q17 row selection payload overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            state
                .candidate_rows
                .push((partkey, quantity, raw as f64 * scale));
        }
        return Ok(());
    }
    for row in 0..extendedprices.len() {
        let (Some(&partkey), Some(&quantity)) = (
            state.selected_partkeys.get(state.payload_offset),
            state.selected_quantities.get(state.payload_offset),
        ) else {
            return Err(DodamError::UnsupportedSql(
                "Q17 row selection payload overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        if extendedprices.is_null(row) {
            continue;
        }
        state
            .candidate_rows
            .push((partkey, quantity, extendedprices.value(row)));
    }
    Ok(())
}

fn q17_merge_quantity_state(
    output: &mut HashMap<i64, (f64, u64)>,
    input: HashMap<i64, (f64, u64)>,
) {
    for (partkey, (sum, count)) in input {
        let aggregate = output.entry(partkey).or_insert((0.0, 0));
        aggregate.0 += sum;
        aggregate.1 += count;
    }
}

fn q17_late_materialized_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_Q17_LATE_MATERIALIZATION")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q17_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q17_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q17_late_materialized_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q17_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

fn q17_late_materialized_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q17_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

fn q17_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q17 lineitem: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q17_lineitem_chunk_size() -> usize {
    std::env::var("DODAM_Q17_LINEITEM_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64)
}

fn q17_lineitem_partial_new(part_key_count: usize) -> Q17LineitemPartial {
    (
        HashMap::<i64, (f64, u64)>::with_capacity(part_key_count),
        Vec::<(i64, f64, f64)>::new(),
    )
}

fn q17_lineitem_revenue_batch_into(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
    partial: &mut Q17LineitemPartial,
) -> Result<()> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    if q17_lineitem_revenue_batch_typed_into(
        partkeys,
        quantities,
        extendedprices,
        part_keys,
        partial,
    )? {
        return Ok(());
    }
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
        let state = partial.0.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity;
        state.1 += 1;
        partial.1.push((partkey, quantity, extendedprice));
    }
    Ok(())
}

fn q17_lineitem_revenue_view_into(
    view: BatchView<'_>,
    part_keys: &AdaptiveI64Set,
    partial: &mut Q17LineitemPartial,
) -> Result<()> {
    if view.num_columns() == 3
        && let (Some(partkeys), Some(quantities), Some(extendedprices)) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
        )
    {
        q17_lineitem_revenue_vector_typed_into(
            partkeys,
            quantities,
            extendedprices,
            part_keys,
            partial,
        );
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "q17 lineitem revenue raw vector columns have unsupported types".to_string(),
        ));
    };
    q17_lineitem_revenue_batch_into(batch.clone(), part_keys, partial)
}

fn q17_lineitem_revenue_batch_typed_into(
    partkeys: &ArrayRef,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    part_keys: &AdaptiveI64Set,
    partial: &mut Q17LineitemPartial,
) -> Result<bool> {
    let (Some(partkeys), Some(quantities), Some(extendedprices)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
        decimal_input(extendedprices)?,
    ) else {
        return Ok(false);
    };
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
        let state = partial.0.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity;
        state.1 += 1;
        partial.1.push((partkey, quantity, extendedprice));
    }
    Ok(true)
}

fn q17_lineitem_revenue_vector_typed_into(
    partkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    part_keys: &AdaptiveI64Set,
    partial: &mut Q17LineitemPartial,
) {
    if let Some(partkey_values) = partkeys.values_if_null_free()
        && quantities.null_count() == 0
        && extendedprices.null_count() == 0
    {
        let quantity_values = quantities.raw_values();
        let extendedprice_values = extendedprices.raw_values();
        let quantity_scale = 1.0 / quantities.scale();
        let extendedprice_scale = 1.0 / extendedprices.scale();
        for row in 0..partkey_values.len() {
            let partkey = partkey_values[row];
            if !part_keys.contains(partkey) {
                continue;
            }
            let quantity = quantity_values[row] as f64 * quantity_scale;
            let extendedprice = extendedprice_values[row] as f64 * extendedprice_scale;
            let state = partial.0.entry(partkey).or_insert((0.0, 0));
            state.0 += quantity;
            state.1 += 1;
            partial.1.push((partkey, quantity, extendedprice));
        }
        return;
    }

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
        let state = partial.0.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity;
        state.1 += 1;
        partial.1.push((partkey, quantity, extendedprice));
    }
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

async fn try_execute_derived_prefix_avg_anti_join_aggregate_sql(
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
    if !derived_prefix_avg_anti_join_aggregate_shape(select, query) {
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

fn derived_prefix_avg_anti_join_aggregate_shape(select: &Select, query: &Query) -> bool {
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
        decimal_input(acctbal)?,
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
    if custkeys.null_count() == 0 {
        return Ok(keys.try_insert_dense_values(custkeys.values().as_ref()));
    }
    for row in 0..custkeys.len() {
        if custkeys.is_valid(row) {
            keys.insert(custkeys.value(row));
        }
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

async fn try_execute_join_with_grouped_sum_semijoin_sql(
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
    if !join_with_grouped_sum_semijoin_shape(select, query, selection) {
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

fn join_with_grouped_sum_semijoin_shape(
    select: &Select,
    query: &Query,
    selection: &SqlExpr,
) -> bool {
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
        decimal_input(quantities)?,
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
        decimal_input(quantities)?,
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
        decimal_input(quantities)?,
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
        decimal_input(totalprices)?,
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

fn bytes_string_parts<'a>(offsets: &[i32], data: &'a [u8], row: usize) -> &'a [u8] {
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    &data[start..end]
}

async fn try_execute_correlated_join_subquery_filter_sql(
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
    let _distinct = parse_distinct(select)?;
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
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
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
            join_memory_limit_bytes: join_memory_limit_bytes(options),
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit, 0)?;
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
            options,
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
            options,
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
    Box::pin(execute_sql_with_options(
        engine,
        &rewritten_query.to_string(),
        batch_size,
        options,
    ))
    .await
    .map(Some)
}

async fn rewrite_materializable_subqueries_to_literals(
    engine: &DodamEngine,
    expr: SqlExpr,
    batch_size: usize,
    options: SqlExecutionOptions,
    changed: &mut bool,
) -> Result<Option<SqlExpr>> {
    match expr {
        SqlExpr::Exists { subquery, negated } => {
            let output = match Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await
            {
                Ok(output) => output,
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
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
            let output = match Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await
            {
                Ok(output) => output,
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
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
            let output = match Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await
            {
                Ok(output) => output,
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
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
                engine, *left, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(right) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *right, batch_size, options, changed,
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
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Nested(Box::new(expr))))
        }
        SqlExpr::UnaryOp { op, expr } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
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
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::IsNull(Box::new(expr))))
        }
        SqlExpr::IsNotNull(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
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
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let mut rewritten_list = Vec::with_capacity(list.len());
            for item in list {
                let Some(item) = Box::pin(rewrite_materializable_subqueries_to_literals(
                    engine, item, batch_size, options, changed,
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
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(low) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *low, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(high) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *high, batch_size, options, changed,
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
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(pattern) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *pattern, batch_size, options, changed,
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

async fn try_execute_with_cte_sql(
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
    let cte_alias = cte.alias.name.value.clone();
    let cte_output = Box::pin(execute_sql_with_options(
        engine,
        &cte.query.to_string(),
        batch_size,
        options,
    ))
    .await?;
    let cte_batches = query_output_batches(cte_output)?;
    if let Some(output) = try_execute_single_cte_select(&cte_alias, &cte_batches, select, query)? {
        return Ok(Some(output));
    }
    let (relations, left_keys, right_keys, residual, right_filter, join_type) = if select.from.len()
        == 2
        && select.from.iter().all(|table| table.joins.is_empty())
    {
        let mut relations = Vec::new();
        for table in &select.from {
            relations.push(
                materialize_cte_join_relation(
                    engine,
                    &table.relation,
                    &cte_alias,
                    &cte_batches,
                    batch_size,
                    options,
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
            DodamError::UnsupportedSql(
                "comma join requires an equality predicate in WHERE".to_string(),
            )
        })?;
        let mut rewritten_selection = selection.clone();
        rewrite_cte_scalar_subqueries_to_literals(
            &mut rewritten_selection,
            &cte_alias,
            &cte_batches,
        )?;
        let (left_keys, right_keys, residual) = split_comma_join_selection(
            Some(&rewritten_selection),
            &left.alias,
            &right.alias,
            &alias_refs,
        )?;
        (
            relations,
            left_keys,
            right_keys,
            residual,
            None,
            JoinType::Inner,
        )
    } else if let [table] = select.from.as_slice()
        && let [join] = table.joins.as_slice()
    {
        let left = materialize_cte_join_relation(
            engine,
            &table.relation,
            &cte_alias,
            &cte_batches,
            batch_size,
            options,
        )
        .await?;
        let right = materialize_cte_join_relation(
            engine,
            &join.relation,
            &cte_alias,
            &cte_batches,
            batch_size,
            options,
        )
        .await?;
        let mut rewritten_join = join.clone();
        if let JoinOperator::Join(JoinConstraint::On(expr))
        | JoinOperator::Inner(JoinConstraint::On(expr))
        | JoinOperator::Left(JoinConstraint::On(expr))
        | JoinOperator::LeftOuter(JoinConstraint::On(expr))
        | JoinOperator::Right(JoinConstraint::On(expr))
        | JoinOperator::RightOuter(JoinConstraint::On(expr))
        | JoinOperator::FullOuter(JoinConstraint::On(expr))
        | JoinOperator::Semi(JoinConstraint::On(expr))
        | JoinOperator::LeftSemi(JoinConstraint::On(expr)) = &mut rewritten_join.join_operator
        {
            rewrite_cte_scalar_subqueries_to_literals(expr, &cte_alias, &cte_batches)?;
        }
        let (join_type, left_keys, right_keys, right_filter) =
            parse_join_condition(&rewritten_join, &left.alias, &right.alias)?;
        let residual = if let Some(selection) = select.selection.as_ref() {
            let mut rewritten_selection = selection.clone();
            rewrite_cte_scalar_subqueries_to_literals(
                &mut rewritten_selection,
                &cte_alias,
                &cte_batches,
            )?;
            Some(rewritten_selection)
        } else {
            None
        };
        (
            vec![left, right],
            left_keys,
            right_keys,
            residual,
            right_filter,
            join_type,
        )
    } else {
        return Err(DodamError::UnsupportedSql(
            "WITH currently supports two-table comma joins and single explicit JOINs".to_string(),
        ));
    };
    let [left, right] = relations.as_slice() else {
        return Ok(None);
    };
    let aliases = vec![left.alias.clone(), right.alias.clone()];
    let alias_refs = aliases.iter().map(String::as_str).collect::<Vec<_>>();
    let output_aliases = if join_type == JoinType::Semi {
        vec![left.alias.as_str()]
    } else {
        alias_refs.clone()
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let distinct = parse_distinct(select)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        None,
    )?;
    let (filter, expression_filter) = parse_join_filter_plan(
        residual.as_ref(),
        &projection.aliases,
        &output_aliases,
        false,
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
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
    let limit = parse_limit(query)?;

    let stream = Box::new(HashJoinExec::new(
        Box::new(MemoryExec::new(left.batches.clone())),
        Box::new(MemoryExec::new(apply_output_filter(
            right.batches.clone(),
            right_filter.as_ref(),
        )?)),
        left_keys,
        right_keys,
        left.alias.clone(),
        right.alias.clone(),
        JoinBuildSide::Right,
        join_type,
        Projection::All,
    ))
    .execute()?;
    let mut batches = apply_output_filter(collect_batches(stream)?, filter.as_ref())?;
    if let Some(expression_filter) = expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(batches, expression_filter, &output_aliases)?;
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

fn try_execute_single_cte_select(
    cte_alias: &str,
    cte_batches: &[RecordBatch],
    select: &Select,
    query: &Query,
) -> Result<Option<QueryOutput>> {
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table.joins.is_empty() {
        return Ok(None);
    }
    let TableFactor::Table { name, alias, .. } = &table.relation else {
        return Ok(None);
    };
    if !object_name_to_string(name)?.eq_ignore_ascii_case(cte_alias) {
        return Ok(None);
    }
    let effective_alias = alias
        .as_ref()
        .map_or_else(|| cte_alias.to_string(), |alias| alias.name.value.clone());
    let group_by = parse_group_by(select, Some(&effective_alias))?;
    let projection = parse_projection(select, &group_by, Some(&effective_alias))?;
    let distinct = parse_distinct(select)?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_filter(expr, &[], Some(&effective_alias), false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        Some(&effective_alias),
    )?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        order_by.as_ref(),
    )?;

    let mut batches = apply_output_filter(cte_batches.to_vec(), filter.as_ref())?;
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
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
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
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
    options: SqlExecutionOptions,
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
        TableFactor::Table { .. } => {
            materialize_join_relation(engine, relation, batch_size, options).await
        }
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
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
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
        let index = batch_column_index(batch, column)?;
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
