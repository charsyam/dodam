use super::*;

pub(super) fn append_aggregate_expression_columns(
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

pub(super) fn collect_aggregates_with_optional_expression_views(
    stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    filtered_aggregates: &[NativeFilteredAggregateSpec],
    expressions: &[ProjectionExpression],
) -> Result<AggregateMetrics> {
    let started = generic_profile_start();
    if group_by.len() <= 1 && !filtered_aggregates.is_empty() {
        let metrics = collect_native_filtered_aggregates(
            stream,
            fragments,
            group_by,
            filtered_aggregates.to_vec(),
        )?;
        generic_profile_elapsed("aggregate filtered", started);
        return Ok(metrics);
    }
    if group_by.len() <= 1
        && let Some(specs) = legacy_case_filtered_aggregate_specs(aggregates, expressions)
    {
        let metrics = collect_native_filtered_aggregates(stream, fragments, group_by, specs)?;
        generic_profile_elapsed("aggregate legacy case filtered", started);
        return Ok(metrics);
    }
    let started = generic_profile_start();
    if group_by.is_empty() {
        let stream = append_aggregate_expression_stream(stream, expressions.to_vec());
        let metrics = collect_aggregates(stream, fragments, aggregates)?;
        generic_profile_elapsed("aggregate global append/fold", started);
        return Ok(metrics);
    }
    if let Some(group_key_exprs) = group_key_exprs_for_aggregate(group_by, aggregates, expressions)
    {
        let metrics = collect_grouped_aggregates_with_key_exprs(
            stream,
            fragments,
            &group_key_exprs,
            aggregates,
        )?
        .expect("expression-view aggregate precondition");
        generic_profile_elapsed("aggregate grouped expression-view", started);
        return Ok(metrics);
    }
    let stream = append_aggregate_expression_stream(stream, expressions.to_vec());
    let metrics = collect_grouped_aggregates(stream, fragments, group_by, aggregates)?;
    generic_profile_elapsed("aggregate grouped append/fold", started);
    Ok(metrics)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn try_collect_expression_aggregate_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: Option<FilterExpr>,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
    ordered_output: bool,
    output_limit: Option<usize>,
) -> Result<Option<AggregateMetrics>> {
    if !expression_aggregate_late_materialized_enabled() {
        return Ok(None);
    }
    let Some(filter) = filter else {
        return Ok(None);
    };
    let Some(group_keys) = group_key_exprs_for_aggregate(group_by, aggregates, expressions) else {
        return Ok(None);
    };
    if CoalesceKeyCountSumCollector::new(&group_keys, aggregates).is_none() {
        return Ok(None);
    }
    let payload_projection =
        expression_aggregate_payload_projection(&group_keys, aggregates, expressions);
    let Projection::Columns(payload_projected_columns) = &payload_projection else {
        return Ok(None);
    };
    let payload_projected_columns = payload_projected_columns.clone();
    let predicate_columns = filter.referenced_columns();
    let predicate_projection = Projection::Columns(predicate_columns.clone());
    let aggregates = aggregates.to_vec();
    let group_keys_for_state = group_keys.clone();
    let aggregates_for_state = aggregates.clone();
    let Some(partials) = engine
        .late_materialized_parquet_map_pruned_with_policy_view_dictionary_columns(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            expression_aggregate_dictionary_columns(&group_keys),
            Vec::new(),
            expression_aggregate_late_row_group_chunk(ordered_output, output_limit),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                expression_aggregate_late_max_selected_ratio(),
                expression_aggregate_late_max_selector_run_ratio(),
            )
            .with_io_cost_gate(!ordered_output && expression_aggregate_late_io_cost_gate_enabled()),
            move || {
                CoalesceKeyCountSumCollector::new_with_composite_hash(
                    &group_keys_for_state,
                    &aggregates_for_state,
                    !ordered_output,
                )
                .expect("expression aggregate late materialization precondition")
            },
            {
                let filter = filter.clone();
                let predicate_columns = predicate_columns.clone();
                move |view, selection, _collector: &mut CoalesceKeyCountSumCollector| {
                    if push_projected_view_filter_selection(
                        view,
                        &predicate_columns,
                        &filter,
                        selection,
                    )? {
                        return Ok(Some(()));
                    }
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
            {
                let payload_projected_columns = payload_projected_columns.clone();
                move |view, collector: &mut CoalesceKeyCountSumCollector| {
                    if expression_aggregate_row_at_time_fallback_enabled() {
                        collector.consume_projected_view(view, &payload_projected_columns)?;
                    } else if !collector
                        .try_consume_projected_view_vectorized(view, &payload_projected_columns)?
                    {
                        return Ok(None);
                    }
                    Ok(Some(()))
                }
            },
            |collector, _metrics| Ok(Some(collector)),
        )
        .await?
    else {
        return Ok(None);
    };
    let collectors = partials
        .into_iter()
        .map(|partial| {
            log_expression_aggregate_late_profile(&partial.metrics);
            partial.output
        })
        .collect::<Vec<_>>();
    Ok(Some(
        CoalesceKeyCountSumCollector::merge_partials_with_order_and_output(
            collectors,
            1,
            &aggregates,
            Some(group_by),
            ordered_output,
            output_limit,
        )?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn try_collect_expression_aggregate_fused_dictionary_selected(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: Option<FilterExpr>,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
    ordered_output: bool,
    output_limit: Option<usize>,
) -> Result<Option<AggregateMetrics>> {
    let Some(filter) = filter else {
        return Ok(None);
    };
    let Some(group_keys) = group_key_exprs_for_aggregate(group_by, aggregates, expressions) else {
        return Ok(None);
    };
    let [
        GroupKeyExpr::Column(first_column),
        GroupKeyExpr::Column(second_column),
        GroupKeyExpr::CoalesceLiteral {
            column: dictionary_column,
            fallback: GroupKeyLiteral::Utf8(fallback),
        },
    ] = group_keys.as_slice()
    else {
        return Ok(None);
    };
    let [AggregateExpr::CountStar, AggregateExpr::Sum(sum_column)] = aggregates else {
        return Ok(None);
    };
    let Some((decimal_column, _precision, _decimal_scale, decimal_filter)) =
        expression_aggregate_decimal_filter(engine, &path, &filter).await?
    else {
        return Ok(None);
    };
    let forced_fused_selected = fused_dictionary_selected_aggregate_enabled();
    let auto_fused_selected = !forced_fused_selected
        && fused_dictionary_selected_aggregate_auto_accepts(
            engine,
            &path,
            &decimal_column,
            &decimal_filter,
        )?;
    if !forced_fused_selected && !auto_fused_selected {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(&path)?).collect::<Vec<_>>();
    let group_keys_for_state = group_keys.clone();
    let aggregates_for_state = aggregates.to_vec();
    let decimal_min = option_i128_to_i64_local(decimal_filter.decimal_min)?;
    let decimal_max = option_i128_to_i64_local(decimal_filter.decimal_max)?;
    let Some((collector, scan_metrics)) = engine
        .scan_parquet_i32_i32_dictionary_i64_decimal_selected_fold(
            path,
            batch_size,
            &row_groups,
            [
                first_column.as_str(),
                second_column.as_str(),
                dictionary_column.as_str(),
                sum_column.as_str(),
                decimal_column.as_str(),
            ],
            fallback.as_bytes(),
            decimal_min,
            decimal_max,
            auto_fused_selected.then(fused_dictionary_selected_auto_workers),
            || {
                CoalesceKeyCountSumCollector::new_with_composite_hash(
                    &group_keys_for_state,
                    &aggregates_for_state,
                    !ordered_output,
                )
                .expect("fused dictionary selected aggregate precondition")
            },
            |collector, batch| {
                if let Some((chunks, sums, selected_rows)) = batch.sum_chunk_ranges() {
                    collector.consume_i32_date_dictionary_i64_sum_chunk_ranges(
                        batch.first(),
                        batch.second(),
                        batch.dictionary_ids(),
                        batch.dictionary(),
                        chunks,
                        sums,
                        selected_rows,
                    )
                } else if let Some(selection) = batch.selection() {
                    collector.consume_i32_date_dictionary_i64_masked(
                        batch.first(),
                        batch.second(),
                        batch.dictionary_ids(),
                        batch.dictionary(),
                        batch.sums(),
                        selection,
                    )
                } else {
                    collector.consume_i32_date_dictionary_i64_slices(
                        batch.first(),
                        batch.second(),
                        batch.dictionary_ids(),
                        batch.dictionary(),
                        batch.sums(),
                    )
                }
            },
            |collector, partial| collector.merge_partial(partial),
        )?
    else {
        return Ok(None);
    };
    log_fused_dictionary_selected_profile(&scan_metrics);
    Ok(Some(
        CoalesceKeyCountSumCollector::merge_partials_with_order_and_output(
            vec![collector],
            1,
            aggregates,
            Some(group_by),
            ordered_output,
            output_limit,
        )?,
    ))
}

fn log_fused_dictionary_selected_profile(metrics: &DirectPrimitiveColumnScanMetrics) {
    if !std::env::var("DODAM_COALESCE_AGG_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:coalesce-agg-profile] fused_selected row_groups={} batches={} rows={} selected={} runs={} read={:.3}ms consume={:.3}ms predicate={:.3}ms payload={:.3}ms dictionary={:.3}ms full_payload_batches={} selected_payload_batches={} selected_read_rows={} selected_skipped_rows={}",
        metrics.row_groups,
        metrics.batches,
        metrics.rows,
        metrics.selected_rows,
        metrics.selected_runs,
        sql_nanos_to_millis(metrics.read_nanos),
        sql_nanos_to_millis(metrics.consume_nanos),
        sql_nanos_to_millis(metrics.selected_predicate_nanos),
        sql_nanos_to_millis(metrics.selected_payload_nanos),
        sql_nanos_to_millis(metrics.selected_dictionary_nanos),
        metrics.full_payload_batches,
        metrics.selected_payload_batches,
        metrics.selected_read_rows,
        metrics.selected_skipped_rows,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn try_collect_expression_aggregate_scan_fold(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    filter: Option<FilterExpr>,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
    ordered_output: bool,
    output_limit: Option<usize>,
) -> Result<Option<AggregateMetrics>> {
    if !expression_aggregate_scan_fold_enabled() {
        return Ok(None);
    }
    let Some(group_keys) = group_key_exprs_for_aggregate(group_by, aggregates, expressions) else {
        return Ok(None);
    };
    let Some(collector) = CoalesceKeyCountSumCollector::new_with_composite_hash(
        &group_keys,
        aggregates,
        !ordered_output,
    ) else {
        return Ok(None);
    };
    let mut projection = projection;
    if let Some(filter) = filter.as_ref() {
        add_projection_columns(&mut projection, filter.referenced_columns());
    }
    let Projection::Columns(projected_columns) = &projection else {
        return Ok(None);
    };
    let projected_columns = projected_columns.clone();
    let aggregates = aggregates.to_vec();
    let dictionary_columns = expression_aggregate_dictionary_columns(&group_keys);
    if !dictionary_columns.is_empty()
        && expression_aggregate_dictionary_scan_fold_enabled()
        && let Some(filter_for_dictionary) = filter.clone()
    {
        let predicates = PredicateSet::new(Some(filter_for_dictionary.clone()));
        let group_keys_for_state = group_keys.clone();
        let aggregates_for_state = aggregates.clone();
        if let Some(partials) = engine
            .parquet_row_group_map_dictionary_columns_pruned_view(
                path.clone(),
                batch_size,
                projection.clone(),
                dictionary_columns,
                predicates.pushdown().to_vec(),
                expression_aggregate_dictionary_scan_fold_row_group_chunk(),
                move || {
                    CoalesceKeyCountSumCollector::new_with_composite_hash(
                        &group_keys_for_state,
                        &aggregates_for_state,
                        !ordered_output,
                    )
                    .expect("expression aggregate dictionary scan-fold precondition")
                },
                {
                    let projected_columns = projected_columns.clone();
                    move |view, collector: &mut CoalesceKeyCountSumCollector| {
                        let mask = if let Some(mask) = evaluate_projected_view_filter_mask(
                            view,
                            &projected_columns,
                            &filter_for_dictionary,
                        )? {
                            mask
                        } else {
                            if !expression_aggregate_row_at_time_fallback_enabled() {
                                return Ok(None);
                            }
                            let Some(batch) = view.try_record_batch() else {
                                return Ok(None);
                            };
                            evaluate_filter_mask(batch, &filter_for_dictionary)?
                        };
                        if mask.true_count() == 0 {
                            return Ok(Some(()));
                        }
                        if expression_aggregate_row_at_time_fallback_enabled() {
                            collector.consume_projected_view_masked_with_record_batch_fallback(
                                view,
                                &projected_columns,
                                &mask,
                            )?;
                        } else if !collector.try_consume_projected_view_masked_vectorized(
                            view,
                            &projected_columns,
                            &mask,
                        )? {
                            return Ok(None);
                        }
                        Ok(Some(()))
                    }
                },
                |collector| Ok(Some(collector)),
            )
            .await?
        {
            return Ok(Some(
                CoalesceKeyCountSumCollector::merge_partials_with_order_and_output(
                    partials,
                    1,
                    &aggregates,
                    Some(group_by),
                    ordered_output,
                    output_limit,
                )?,
            ));
        }
    }
    let collector = engine
        .scan_parquet_batches_fold_view(
            path,
            batch_size,
            None,
            projection.clone(),
            filter,
            collector,
            move |view, collector| collector.consume_projected_view(view, &projected_columns),
            Ok,
        )
        .await?;
    Ok(Some(
        CoalesceKeyCountSumCollector::merge_partials_with_order_and_output(
            vec![collector],
            1,
            &aggregates,
            Some(group_by),
            ordered_output,
            output_limit,
        )?,
    ))
}

pub(super) fn expression_aggregate_output_limit(
    group_by: &[String],
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Option<usize> {
    let limit = limit?;
    if offset != 0 || !aggregate_order_by_prefix_matches(group_by, order_by) {
        return None;
    }
    Some(limit)
}

fn aggregate_order_by_prefix_matches(group_by: &[String], order_by: Option<&SortKey>) -> bool {
    let Some(order_by) = order_by else {
        return false;
    };
    if order_by.expressions.is_empty() || order_by.expressions.len() > group_by.len() {
        return false;
    }
    order_by
        .expressions
        .iter()
        .zip(group_by)
        .all(|(sort, group)| !sort.descending && !sort.nulls_first && sort.column == *group)
}

#[allow(dead_code)]
fn evaluate_filter_mask_for_projected_view(
    view: BatchView<'_>,
    columns: &[String],
    filter: &FilterExpr,
) -> Result<Option<BooleanArray>> {
    evaluate_expr_mask_for_projected_view(view, columns, filter.expr())
}

fn push_projected_view_filter_selection(
    view: BatchView<'_>,
    columns: &[String],
    filter: &FilterExpr,
    selection: &mut LateSelectionBuilder,
) -> Result<bool> {
    if std::env::var("DODAM_DISABLE_DIRECT_LATE_SELECTION")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return Ok(false);
    }
    push_projected_view_expr_selection(view, columns, filter.expr(), selection)
}

pub(super) fn push_boolean_mask_selection(
    mask: &BooleanArray,
    selection: &mut LateSelectionBuilder,
) {
    if mask.true_count() == 0 {
        selection.push_repeated(mask.len(), false);
        return;
    }
    if mask.true_count() == mask.len() && mask.null_count() == 0 {
        selection.push_repeated(mask.len(), true);
        return;
    }
    selection.push_selected_offsets(
        mask.len(),
        (0..mask.len()).filter(|row| mask.is_valid(*row) && mask.value(*row)),
    );
}

fn push_projected_view_expr_selection(
    view: BatchView<'_>,
    columns: &[String],
    expr: &Expr,
    selection: &mut LateSelectionBuilder,
) -> Result<bool> {
    match expr {
        Expr::Boolean(value) => {
            let value = value.unwrap_or(false);
            selection.push_repeated(view.num_rows(), value);
            Ok(true)
        }
        Expr::Comparison(comparison) => {
            push_comparison_selection_for_projected_view(view, columns, comparison, selection)
        }
        Expr::InList {
            column,
            values,
            negated,
            has_null,
        } => push_in_list_selection_for_projected_view(
            view, columns, column, values, *negated, *has_null, selection,
        ),
        Expr::Not(_)
        | Expr::And(_, _)
        | Expr::Or(_, _)
        | Expr::ColumnComparison { .. }
        | Expr::Like { .. }
        | Expr::IsNull { .. } => Ok(false),
    }
}

fn push_comparison_selection_for_projected_view(
    view: BatchView<'_>,
    columns: &[String],
    comparison: &ComparisonExpr,
    selection: &mut LateSelectionBuilder,
) -> Result<bool> {
    if matches!(comparison.value, LiteralValue::Null) {
        selection.push_repeated(view.num_rows(), false);
        return Ok(true);
    }
    let Some(index) = projected_view_column_index(columns, &comparison.column) else {
        return Ok(false);
    };
    if let Some(values) = view.i64_vector(index) {
        let literal = comparison.value.as_i64(&comparison.column)?;
        if let Some(raw) = values.values_if_null_free() {
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    compare_i64(*value, comparison.op, literal).then_some(row)
                }),
            );
        } else if let Some((data, len)) = values.raw_bytes() {
            selection.push_selected_offsets(
                len,
                (0..len).filter(|row| {
                    compare_i64(read_i64_le_unaligned(data, *row), comparison.op, literal)
                }),
            );
        } else {
            selection.push_selected_offsets(
                values.len(),
                (0..values.len()).filter(|row| {
                    !values.is_null(*row) && compare_i64(values.value(*row), comparison.op, literal)
                }),
            );
        }
        return Ok(true);
    }
    if let Some(values) = view.i32_vector(index) {
        let literal = comparison.value.as_i32(&comparison.column)?;
        if let Some(raw) = values.values_if_null_free() {
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    compare_i32(*value, comparison.op, literal).then_some(row)
                }),
            );
        } else if let Some((data, len)) = values.raw_bytes() {
            selection.push_selected_offsets(
                len,
                (0..len).filter(|row| {
                    compare_i32(read_i32_le_unaligned(data, *row), comparison.op, literal)
                }),
            );
        } else {
            selection.push_selected_offsets(
                values.len(),
                (0..values.len()).filter(|row| {
                    !values.is_null(*row) && compare_i32(values.value(*row), comparison.op, literal)
                }),
            );
        }
        return Ok(true);
    }
    if let Some(values) = view.date32_vector(index) {
        let Some(literal) = literal_as_date32_for_type(&comparison.value)? else {
            return Ok(false);
        };
        if let Some(raw) = values.values_if_null_free() {
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    compare_i32(*value, comparison.op, literal).then_some(row)
                }),
            );
        } else if let Some((data, len)) = values.raw_bytes() {
            selection.push_selected_offsets(
                len,
                (0..len).filter(|row| {
                    compare_i32(read_i32_le_unaligned(data, *row), comparison.op, literal)
                }),
            );
        } else {
            selection.push_selected_offsets(
                values.len(),
                (0..values.len()).filter(|row| {
                    !values.is_null(*row) && compare_i32(values.value(*row), comparison.op, literal)
                }),
            );
        }
        return Ok(true);
    }
    if let Some(values) = view.decimal128_vector(index) {
        let Some(literal) = literal_to_decimal_scaled(&comparison.value, values.scale_i64())?
        else {
            return Ok(false);
        };
        if let Some(raw) = values.raw_i64_values() {
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    (!values.is_null(row)
                        && compare_i128(i128::from(*value), comparison.op, literal))
                    .then_some(row)
                }),
            );
        } else if let Some((data, len)) = values.raw_i64_bytes() {
            selection.push_selected_offsets(
                len,
                (0..len).filter(|row| {
                    compare_i128(
                        i128::from(read_i64_le_unaligned(data, *row)),
                        comparison.op,
                        literal,
                    )
                }),
            );
        } else {
            let raw = values.raw_values();
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    (!values.is_null(row) && compare_i128(*value, comparison.op, literal))
                        .then_some(row)
                }),
            );
        }
        return Ok(true);
    }
    Ok(false)
}

fn push_in_list_selection_for_projected_view(
    view: BatchView<'_>,
    columns: &[String],
    column: &str,
    values: &[LiteralValue],
    negated: bool,
    has_null: bool,
    selection: &mut LateSelectionBuilder,
) -> Result<bool> {
    let Some(index) = projected_view_column_index(columns, column) else {
        return Ok(false);
    };
    if let Some(probe) = view.i64_vector(index) {
        let values = values
            .iter()
            .filter(|value| !matches!(value, LiteralValue::Null))
            .map(|value| value.as_i64(column))
            .collect::<Result<Vec<_>>>()?;
        if let Some(raw) = probe.values_if_null_free() {
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    selected_in_list_result(
                        small_in_list_match_i64(*value, &values),
                        negated,
                        has_null,
                    )
                    .then_some(row)
                }),
            );
        } else {
            selection.push_selected_offsets(
                probe.len(),
                (0..probe.len()).filter(|row| {
                    !probe.is_null(*row)
                        && selected_in_list_result(
                            small_in_list_match_i64(probe.value(*row), &values),
                            negated,
                            has_null,
                        )
                }),
            );
        }
        return Ok(true);
    }
    if let Some(probe) = view.i32_vector(index) {
        let values = values
            .iter()
            .filter(|value| !matches!(value, LiteralValue::Null))
            .map(|value| value.as_i32(column))
            .collect::<Result<Vec<_>>>()?;
        if let Some(raw) = probe.values_if_null_free() {
            selection.push_selected_offsets(
                raw.len(),
                raw.iter().enumerate().filter_map(|(row, value)| {
                    selected_in_list_result(
                        small_in_list_match_i32(*value, &values),
                        negated,
                        has_null,
                    )
                    .then_some(row)
                }),
            );
        } else {
            selection.push_selected_offsets(
                probe.len(),
                (0..probe.len()).filter(|row| {
                    !probe.is_null(*row)
                        && selected_in_list_result(
                            small_in_list_match_i32(probe.value(*row), &values),
                            negated,
                            has_null,
                        )
                }),
            );
        }
        return Ok(true);
    }
    Ok(false)
}

fn evaluate_expr_mask_for_projected_view(
    view: BatchView<'_>,
    columns: &[String],
    expr: &Expr,
) -> Result<Option<BooleanArray>> {
    match expr {
        Expr::Boolean(value) => Ok(Some(BooleanArray::from(vec![*value; view.num_rows()]))),
        Expr::Comparison(comparison) => {
            evaluate_comparison_mask_for_projected_view(view, columns, comparison)
        }
        Expr::InList {
            column,
            values,
            negated,
            has_null,
        } => evaluate_in_list_mask_for_projected_view(
            view, columns, column, values, *negated, *has_null,
        ),
        Expr::Not(expr) => {
            let Some(mask) = evaluate_expr_mask_for_projected_view(view, columns, expr)? else {
                return Ok(None);
            };
            Ok(Some(boolean_not(&mask)))
        }
        Expr::And(left, right) => {
            let Some(left) = evaluate_expr_mask_for_projected_view(view, columns, left)? else {
                return Ok(None);
            };
            let Some(right) = evaluate_expr_mask_for_projected_view(view, columns, right)? else {
                return Ok(None);
            };
            Ok(Some(boolean_and(&left, &right)))
        }
        Expr::Or(left, right) => {
            let Some(left) = evaluate_expr_mask_for_projected_view(view, columns, left)? else {
                return Ok(None);
            };
            let Some(right) = evaluate_expr_mask_for_projected_view(view, columns, right)? else {
                return Ok(None);
            };
            Ok(Some(boolean_or(&left, &right)))
        }
        Expr::ColumnComparison { .. } | Expr::Like { .. } | Expr::IsNull { .. } => Ok(None),
    }
}

fn evaluate_comparison_mask_for_projected_view(
    view: BatchView<'_>,
    columns: &[String],
    comparison: &ComparisonExpr,
) -> Result<Option<BooleanArray>> {
    if matches!(comparison.value, LiteralValue::Null) {
        return Ok(Some(BooleanArray::from(vec![None; view.num_rows()])));
    }
    let Some(index) = projected_view_column_index(columns, &comparison.column) else {
        return Ok(None);
    };
    if let Some(values) = view.i64_vector(index) {
        let value = comparison.value.as_i64(&comparison.column)?;
        return Ok(Some(compare_i64_view(values, comparison.op, value)));
    }
    if let Some(values) = view.i32_vector(index) {
        let value = comparison.value.as_i32(&comparison.column)?;
        return Ok(Some(compare_i32_view(values, comparison.op, value)));
    }
    if let Some(values) = view.date32_vector(index) {
        let Some(value) = literal_as_date32_for_type(&comparison.value)? else {
            return Ok(None);
        };
        return Ok(Some(compare_date32_view(values, comparison.op, value)));
    }
    if let Some(values) = view.decimal128_vector(index) {
        let Some(value) = literal_to_decimal_scaled(&comparison.value, values.scale_i64())? else {
            return Ok(None);
        };
        return Ok(Some(compare_decimal128_view(values, comparison.op, value)));
    }
    Ok(None)
}

fn evaluate_in_list_mask_for_projected_view(
    view: BatchView<'_>,
    columns: &[String],
    column: &str,
    values: &[LiteralValue],
    negated: bool,
    has_null: bool,
) -> Result<Option<BooleanArray>> {
    let Some(index) = projected_view_column_index(columns, column) else {
        return Ok(None);
    };
    if let Some(probe) = view.i64_vector(index) {
        let values = values
            .iter()
            .filter(|value| !matches!(value, LiteralValue::Null))
            .map(|value| value.as_i64(column))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Some(in_list_i64_view(probe, &values, negated, has_null)));
    }
    if let Some(probe) = view.i32_vector(index) {
        let values = values
            .iter()
            .filter(|value| !matches!(value, LiteralValue::Null))
            .map(|value| value.as_i32(column))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Some(in_list_i32_view(probe, &values, negated, has_null)));
    }
    Ok(None)
}

fn projected_view_column_index(columns: &[String], column: &str) -> Option<usize> {
    columns.iter().position(|candidate| candidate == column)
}

fn compare_i64_view(values: I64VectorView<'_>, op: ComparisonOp, literal: i64) -> BooleanArray {
    if let Some(raw) = values.values_if_null_free() {
        return boolean_array_no_nulls_from_len(raw.len(), |row| {
            compare_i64(raw[row], op, literal)
        });
    }
    if let Some((data, len)) = values.raw_bytes() {
        return boolean_array_no_nulls_from_i64_bytes(data, len, |value| {
            compare_i64(value, op, literal)
        });
    }
    let mut output = BooleanBuilder::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            output.append_null();
        } else {
            output.append_value(compare_i64(values.value(row), op, literal));
        }
    }
    output.finish()
}

fn compare_i32_view(values: I32VectorView<'_>, op: ComparisonOp, literal: i32) -> BooleanArray {
    if let Some(raw) = values.values_if_null_free() {
        return boolean_array_no_nulls_from_len(raw.len(), |row| {
            compare_i32(raw[row], op, literal)
        });
    }
    if let Some((data, len)) = values.raw_bytes() {
        return boolean_array_no_nulls_from_i32_bytes(data, len, |value| {
            compare_i32(value, op, literal)
        });
    }
    let mut output = BooleanBuilder::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            output.append_null();
        } else {
            output.append_value(compare_i32(values.value(row), op, literal));
        }
    }
    output.finish()
}

fn compare_date32_view(
    values: Date32VectorView<'_>,
    op: ComparisonOp,
    literal: i32,
) -> BooleanArray {
    if let Some(raw) = values.values_if_null_free() {
        return boolean_array_no_nulls_from_len(raw.len(), |row| {
            compare_i32(raw[row], op, literal)
        });
    }
    if let Some((data, len)) = values.raw_bytes() {
        return boolean_array_no_nulls_from_i32_bytes(data, len, |value| {
            compare_i32(value, op, literal)
        });
    }
    let mut output = BooleanBuilder::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            output.append_null();
        } else {
            output.append_value(compare_i32(values.value(row), op, literal));
        }
    }
    output.finish()
}

fn compare_decimal128_view(
    values: Decimal128VectorView<'_>,
    op: ComparisonOp,
    literal: i128,
) -> BooleanArray {
    let mut output = BooleanBuilder::with_capacity(values_len_decimal128(values));
    if let Some(raw) = values.raw_i64_values() {
        for row in 0..raw.len() {
            if values.is_null(row) {
                output.append_null();
            } else {
                output.append_value(compare_i128(i128::from(raw[row]), op, literal));
            }
        }
    } else if let Some((data, len)) = values.raw_i64_bytes() {
        for row in 0..len {
            output.append_value(compare_i128(
                i128::from(read_i64_le_unaligned(data, row)),
                op,
                literal,
            ));
        }
    } else {
        let raw = values.raw_values();
        for row in 0..raw.len() {
            if values.is_null(row) {
                output.append_null();
            } else {
                output.append_value(compare_i128(raw[row], op, literal));
            }
        }
    }
    output.finish()
}

fn values_len_decimal128(values: Decimal128VectorView<'_>) -> usize {
    values.len()
}

fn in_list_i64_view(
    values: I64VectorView<'_>,
    list: &[i64],
    negated: bool,
    has_null: bool,
) -> BooleanArray {
    if !has_null && let Some(raw) = values.values_if_null_free() {
        return boolean_array_no_nulls_from_len(raw.len(), |row| {
            selected_in_list_result(small_in_list_match_i64(raw[row], list), negated, false)
        });
    }
    if !has_null && let Some((data, len)) = values.raw_bytes() {
        return boolean_array_no_nulls_from_i64_bytes(data, len, |value| {
            selected_in_list_result(small_in_list_match_i64(value, list), negated, false)
        });
    }
    let mut output = BooleanBuilder::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            output.append_null();
            continue;
        }
        append_in_list_result(
            &mut output,
            small_in_list_match_i64(values.value(row), list),
            negated,
            has_null,
        );
    }
    output.finish()
}

fn in_list_i32_view(
    values: I32VectorView<'_>,
    list: &[i32],
    negated: bool,
    has_null: bool,
) -> BooleanArray {
    if !has_null && let Some(raw) = values.values_if_null_free() {
        return boolean_array_no_nulls_from_len(raw.len(), |row| {
            selected_in_list_result(small_in_list_match_i32(raw[row], list), negated, false)
        });
    }
    if !has_null && let Some((data, len)) = values.raw_bytes() {
        return boolean_array_no_nulls_from_i32_bytes(data, len, |value| {
            selected_in_list_result(small_in_list_match_i32(value, list), negated, false)
        });
    }
    let mut output = BooleanBuilder::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            output.append_null();
            continue;
        }
        append_in_list_result(
            &mut output,
            small_in_list_match_i32(values.value(row), list),
            negated,
            has_null,
        );
    }
    output.finish()
}

fn small_in_list_match_i64(value: i64, list: &[i64]) -> bool {
    match list {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => list.contains(&value),
    }
}

fn small_in_list_match_i32(value: i32, list: &[i32]) -> bool {
    match list {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => list.contains(&value),
    }
}

fn append_in_list_result(
    output: &mut BooleanBuilder,
    matched: bool,
    negated: bool,
    has_null: bool,
) {
    if matched {
        output.append_value(!negated);
    } else if has_null {
        output.append_null();
    } else {
        output.append_value(negated);
    }
}

fn selected_in_list_result(matched: bool, negated: bool, has_null: bool) -> bool {
    if matched {
        !negated
    } else if has_null {
        false
    } else {
        negated
    }
}

pub(super) fn boolean_array_no_nulls_from_len(
    len: usize,
    mut value_at: impl FnMut(usize) -> bool,
) -> BooleanArray {
    let mut values = BooleanBufferBuilder::new(len);
    for row in 0..len {
        values.append(value_at(row));
    }
    BooleanArray::new(values.build(), None)
}

fn boolean_array_no_nulls_from_i32_bytes(
    data: &[u8],
    len: usize,
    mut value_at: impl FnMut(i32) -> bool,
) -> BooleanArray {
    debug_assert!(data.len() >= len.saturating_mul(std::mem::size_of::<i32>()));
    let mut values = BooleanBufferBuilder::new(len);
    for row in 0..len {
        values.append(value_at(read_i32_le_unaligned(data, row)));
    }
    BooleanArray::new(values.build(), None)
}

fn boolean_array_no_nulls_from_i64_bytes(
    data: &[u8],
    len: usize,
    mut value_at: impl FnMut(i64) -> bool,
) -> BooleanArray {
    debug_assert!(data.len() >= len.saturating_mul(std::mem::size_of::<i64>()));
    let mut values = BooleanBufferBuilder::new(len);
    for row in 0..len {
        values.append(value_at(read_i64_le_unaligned(data, row)));
    }
    BooleanArray::new(values.build(), None)
}

fn compare_i64(left: i64, op: ComparisonOp, right: i64) -> bool {
    match op {
        ComparisonOp::Eq => left == right,
        ComparisonOp::NotEq => left != right,
        ComparisonOp::Lt => left < right,
        ComparisonOp::LtEq => left <= right,
        ComparisonOp::Gt => left > right,
        ComparisonOp::GtEq => left >= right,
    }
}

pub(super) fn compare_i32(left: i32, op: ComparisonOp, right: i32) -> bool {
    compare_i64(i64::from(left), op, i64::from(right))
}

fn compare_i128(left: i128, op: ComparisonOp, right: i128) -> bool {
    match op {
        ComparisonOp::Eq => left == right,
        ComparisonOp::NotEq => left != right,
        ComparisonOp::Lt => left < right,
        ComparisonOp::LtEq => left <= right,
        ComparisonOp::Gt => left > right,
        ComparisonOp::GtEq => left >= right,
    }
}

fn literal_to_decimal_scaled(
    value: &LiteralValue,
    scale_factor: Option<i64>,
) -> Result<Option<i128>> {
    let Some(scale_factor) = scale_factor else {
        return Ok(None);
    };
    match value {
        LiteralValue::Null => Ok(None),
        LiteralValue::Int64(value) => Ok(i128::from(*value)
            .checked_mul(i128::from(scale_factor))
            .map(Some)
            .unwrap_or(None)),
        LiteralValue::Float64(value) => {
            decimal_literal_to_scaled_factor(&value.to_string(), scale_factor)
        }
        LiteralValue::Utf8(value) => decimal_literal_to_scaled_factor(value, scale_factor),
        LiteralValue::Boolean(_) => Ok(None),
    }
}

fn decimal_literal_to_scaled_factor(value: &str, scale_factor: i64) -> Result<Option<i128>> {
    let Some(scale_digits) = decimal_scale_digits(scale_factor) else {
        return Ok(None);
    };
    let value = value.trim();
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix(['-', '+']).unwrap_or(value);
    let (whole, fractional) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() || !whole.chars().all(|ch| ch.is_ascii_digit()) {
        return Ok(None);
    }
    if !fractional.chars().all(|ch| ch.is_ascii_digit()) {
        return Ok(None);
    }
    let mut scaled = whole
        .parse::<i128>()
        .map_err(|_| DodamError::InvalidCast(format!("decimal literal {value} is out of range")))?;
    scaled = scaled
        .checked_mul(i128::from(scale_factor))
        .ok_or_else(|| DodamError::InvalidCast("decimal literal overflow".to_string()))?;
    let kept = &fractional[..fractional.len().min(scale_digits)];
    let mut frac = if kept.is_empty() {
        0
    } else {
        kept.parse::<i128>().map_err(|_| {
            DodamError::InvalidCast(format!("decimal literal {value} is out of range"))
        })?
    };
    for _ in kept.len()..scale_digits {
        frac *= 10;
    }
    scaled = scaled
        .checked_add(frac)
        .ok_or_else(|| DodamError::InvalidCast("decimal literal overflow".to_string()))?;
    Ok(Some(if negative { -scaled } else { scaled }))
}

fn decimal_scale_digits(mut scale_factor: i64) -> Option<usize> {
    if scale_factor <= 0 {
        return None;
    }
    let mut digits = 0usize;
    while scale_factor > 1 {
        if scale_factor % 10 != 0 {
            return None;
        }
        scale_factor /= 10;
        digits += 1;
    }
    Some(digits)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn try_collect_expression_aggregate_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    filter: Option<FilterExpr>,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Result<Option<AggregateMetrics>> {
    if !expression_aggregate_row_group_map_enabled() {
        return Ok(None);
    }
    let Some(group_keys) = group_key_exprs_for_aggregate(group_by, aggregates, expressions) else {
        return Ok(None);
    };
    let mut projection = projection;
    if let Some(filter) = filter.as_ref() {
        add_projection_columns(&mut projection, filter.referenced_columns());
    }
    let aggregates = aggregates.to_vec();
    let group_keys_for_state = group_keys.clone();
    let aggregates_for_state = aggregates.clone();
    let Some(partials) = engine
        .parquet_row_group_map_pruned(
            path,
            batch_size,
            projection,
            Vec::new(),
            expression_aggregate_row_group_map_chunk(),
            move || {
                CoalesceKeyCountSumCollector::new(&group_keys_for_state, &aggregates_for_state)
                    .expect("expression aggregate row-group map precondition")
            },
            {
                let filter = filter.clone();
                move |batch, collector: &mut CoalesceKeyCountSumCollector| {
                    let batch = if let Some(filter) = filter.as_ref() {
                        filter_batch(batch, filter)?
                    } else {
                        batch
                    };
                    if batch.num_rows() > 0 {
                        collector.consume_batch(&batch)?;
                    }
                    Ok(Some(()))
                }
            },
            |collector| Ok(Some(collector)),
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(CoalesceKeyCountSumCollector::merge_partials(
        partials,
        1,
        &aggregates,
    )?))
}

fn expression_aggregate_row_group_map_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_AGG_ROW_GROUP_MAP")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_aggregate_scan_fold_enabled() -> bool {
    std::env::var("DODAM_DISABLE_EXPRESSION_AGG_SCAN_FOLD")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn expression_aggregate_dictionary_scan_fold_enabled() -> bool {
    std::env::var("DODAM_ENABLE_EXPRESSION_AGG_DICTIONARY_SCAN_FOLD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn fused_dictionary_selected_aggregate_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FUSED_DICTIONARY_SELECTED_AGG")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn fused_dictionary_selected_aggregate_auto_accepts(
    engine: &DodamEngine,
    path: &Path,
    decimal_column: &str,
    decimal_filter: &DecimalDateRangeFilter,
) -> Result<bool> {
    if std::env::var("DODAM_DISABLE_FUSED_DICTIONARY_SELECTED_AUTO")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        log_fused_dictionary_selected_auto_decision("disabled", None, None, None, None, None, None);
        return Ok(false);
    }
    let Some((column_min, column_max)) = engine
        .parquet_i128_column_min_max(path, decimal_column)?
        .or_else(|| {
            engine
                .parquet_i128_column_min_max_relaxed(path, decimal_column)
                .ok()
                .flatten()
        })
    else {
        log_fused_dictionary_selected_auto_decision(
            "missing-stats",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        return Ok(false);
    };
    let Some(estimated_selectivity) = (DecimalRangeSelectivityInput {
        column_min,
        column_max,
        filter_min: decimal_filter.decimal_min,
        filter_max: decimal_filter.decimal_max,
    })
    .estimated_selectivity() else {
        log_fused_dictionary_selected_auto_decision(
            "invalid-selectivity",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        return Ok(false);
    };
    if let Some((candidate_row_groups, total_row_groups, total_rows)) =
        decimal_filter_candidate_row_group_spread(engine, path, decimal_column, decimal_filter)?
    {
        let estimated_selected_rows = ((total_rows as f64) * estimated_selectivity).ceil() as usize;
        let decision = choose_selected_payload_by_spread(SelectedPayloadSpreadCostInput {
            selected_rows: estimated_selected_rows.max(1),
            selected_row_groups: candidate_row_groups,
            total_row_groups,
            missing_payload_columns: 4,
            max_selected_row_group_ratio:
                fused_dictionary_selected_auto_max_selected_row_group_ratio(),
            max_selected_row_groups: fused_dictionary_selected_auto_max_selected_row_groups(),
        });
        if !decision.accepted() {
            if estimated_selectivity <= fused_dictionary_selected_auto_spread_override_ratio() {
                log_fused_dictionary_selected_auto_decision(
                    "spread-override",
                    Some(estimated_selectivity),
                    Some(estimated_selected_rows.max(1)),
                    Some(candidate_row_groups),
                    Some(total_row_groups),
                    Some(total_rows),
                    Some(fused_dictionary_selected_auto_max_estimated_ratio()),
                );
            } else {
                log_fused_dictionary_selected_auto_decision(
                    decision.reason(),
                    Some(estimated_selectivity),
                    Some(estimated_selected_rows.max(1)),
                    Some(candidate_row_groups),
                    Some(total_row_groups),
                    Some(total_rows),
                    Some(fused_dictionary_selected_auto_max_estimated_ratio()),
                );
                return Ok(false);
            }
        }
    }
    let max_selectivity = fused_dictionary_selected_auto_max_estimated_ratio();
    let accepted = choose_fused_selected_aggregate(FusedSelectedAggregateCostInput {
        estimated_selectivity,
        max_selectivity,
    });
    log_fused_dictionary_selected_auto_decision(
        if accepted {
            "accepted"
        } else {
            "estimated-selectivity"
        },
        Some(estimated_selectivity),
        None,
        None,
        None,
        None,
        Some(max_selectivity),
    );
    Ok(accepted)
}

fn fused_dictionary_selected_auto_spread_override_ratio() -> f64 {
    std::env::var("DODAM_FUSED_DICTIONARY_SELECTED_AUTO_SPREAD_OVERRIDE_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.02)
}

fn log_fused_dictionary_selected_auto_decision(
    reason: &str,
    estimated_selectivity: Option<f64>,
    estimated_selected_rows: Option<usize>,
    candidate_row_groups: Option<usize>,
    total_row_groups: Option<usize>,
    total_rows: Option<usize>,
    max_selectivity: Option<f64>,
) {
    if !std::env::var("DODAM_COALESCE_AGG_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:coalesce-agg-profile] fused_selected_auto decision={} estimated_selectivity={} estimated_selected_rows={} row_groups={}/{} total_rows={} max_selectivity={}",
        reason,
        estimated_selectivity
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        estimated_selected_rows
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        candidate_row_groups
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        total_row_groups
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        total_rows
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        max_selectivity
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
    );
}

fn decimal_filter_candidate_row_group_spread(
    engine: &DodamEngine,
    path: &Path,
    decimal_column: &str,
    decimal_filter: &DecimalDateRangeFilter,
) -> Result<Option<(usize, usize, usize)>> {
    let Some(ranges) =
        engine.parquet_primitive_column_min_max_by_row_group(path, decimal_column)?
    else {
        return Ok(None);
    };
    if ranges.is_empty() {
        return Ok(Some((0, 0, 0)));
    }
    let mut candidate_row_groups = 0usize;
    let mut total_rows = 0usize;
    for range in &ranges {
        total_rows = total_rows.saturating_add(range.rows);
        if decimal_filter_row_group_may_match(range, decimal_filter) {
            candidate_row_groups += 1;
        }
    }
    Ok(Some((candidate_row_groups, ranges.len(), total_rows)))
}

fn decimal_filter_row_group_may_match(
    range: &PrimitiveRowGroupMinMax,
    decimal_filter: &DecimalDateRangeFilter,
) -> bool {
    if decimal_filter
        .decimal_min
        .is_some_and(|filter_min| range.max < filter_min)
    {
        return false;
    }
    if decimal_filter
        .decimal_max
        .is_some_and(|filter_max| range.min > filter_max)
    {
        return false;
    }
    true
}

fn fused_dictionary_selected_auto_max_estimated_ratio() -> f64 {
    std::env::var("DODAM_FUSED_DICT_SELECTED_AUTO_MAX_ESTIMATED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.05)
        .clamp(0.0, 1.0)
}

fn fused_dictionary_selected_auto_max_selected_row_group_ratio() -> f64 {
    std::env::var("DODAM_FUSED_DICT_SELECTED_AUTO_MAX_ROW_GROUP_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.50)
        .clamp(0.0, 1.0)
}

fn fused_dictionary_selected_auto_max_selected_row_groups() -> usize {
    std::env::var("DODAM_FUSED_DICT_SELECTED_AUTO_MAX_ROW_GROUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn fused_dictionary_selected_auto_workers() -> usize {
    std::env::var("DODAM_FUSED_DICT_SELECTED_AUTO_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            choose_parallel_workers(WorkerCostInput {
                row_groups: usize::MAX,
                available_parallelism: std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
                max_workers: 12,
            })
        })
}

fn expression_aggregate_dictionary_scan_fold_row_group_chunk() -> usize {
    std::env::var("DODAM_EXPRESSION_AGG_DICTIONARY_SCAN_FOLD_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn expression_aggregate_late_materialized_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_EXPRESSION_AGG_LATE_MATERIALIZE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn expression_aggregate_late_row_group_chunk(
    ordered_output: bool,
    output_limit: Option<usize>,
) -> usize {
    if let Some(value) = std::env::var("DODAM_EXPRESSION_AGG_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return value;
    }
    choose_expression_aggregate_late_row_group_chunk(ExpressionAggregateLateChunkCostInput {
        ordered_output,
        output_limit,
        default_chunk: 2,
        ordered_limit_chunk: 4,
    })
}

fn expression_aggregate_late_max_selected_ratio() -> f64 {
    std::env::var("DODAM_EXPRESSION_AGG_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.75)
}

fn expression_aggregate_late_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_EXPRESSION_AGG_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.90)
}

fn expression_aggregate_late_io_cost_gate_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_EXPRESSION_AGG_LATE_IO_COST_GATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn expression_aggregate_row_at_time_fallback_enabled() -> bool {
    row_at_time_fallback_enabled()
}

fn log_expression_aggregate_late_profile(metrics: &LateMaterializedMetrics) {
    if !std::env::var("DODAM_COALESCE_AGG_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    let run_ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selector_runs as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:coalesce-agg-profile] late rows={} selected={} ratio={:.6} selector_runs={} run_ratio={:.6}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, run_ratio
    );
}

fn expression_aggregate_payload_projection(
    group_keys: &[GroupKeyExpr],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Projection {
    let mut columns = Vec::new();
    for group_key in group_keys {
        match group_key {
            GroupKeyExpr::Column(column) => add_column_once(&mut columns, column.clone()),
            GroupKeyExpr::CoalesceLiteral { column, .. } => {
                add_column_once(&mut columns, column.clone())
            }
            GroupKeyExpr::LowerCoalesceLiteral { column, .. } => {
                add_column_once(&mut columns, column.clone())
            }
            GroupKeyExpr::Int64AddLiteral { column, .. } => {
                add_column_once(&mut columns, column.clone())
            }
            GroupKeyExpr::SimpleCaseLiteral { column, .. } => {
                add_column_once(&mut columns, column.clone())
            }
        }
    }
    for aggregate in aggregates {
        if let AggregateExpr::Sum(column)
        | AggregateExpr::Min(column)
        | AggregateExpr::Max(column)
        | AggregateExpr::Avg(column) = aggregate
        {
            add_column_once(&mut columns, column.clone());
        }
    }
    for expression in expressions {
        add_scalar_expression_columns(&mut columns, &expression.expr);
    }
    Projection::Columns(columns)
}

async fn expression_aggregate_decimal_filter(
    engine: &DodamEngine,
    path: &Path,
    filter: &FilterExpr,
) -> Result<Option<(String, u8, i8, DecimalDateRangeFilter)>> {
    let mut matches = Vec::new();
    for column in filter.referenced_columns() {
        let Some((precision, scale)) = engine.parquet_decimal128_type(path, &column)? else {
            continue;
        };
        let Some(range) =
            DecimalDateRangeFilter::try_new(filter.expr(), &column, "__dodam_no_date__", scale)?
        else {
            continue;
        };
        matches.push((column, precision, scale, range));
    }
    if matches.len() == 1 {
        Ok(matches.pop())
    } else {
        Ok(None)
    }
}

fn option_i128_to_i64_local(value: Option<i128>) -> Result<Option<i64>> {
    value
        .map(|value| {
            i64::try_from(value).map_err(|_| {
                DodamError::UnsupportedSql(
                    "direct selected dictionary decimal bound out of i64 range".to_string(),
                )
            })
        })
        .transpose()
}

fn add_scalar_expression_columns(columns: &mut Vec<String>, expression: &ScalarSqlExpression) {
    match expression {
        ScalarSqlExpression::Column(column)
        | ScalarSqlExpression::StructField { column, .. }
        | ScalarSqlExpression::ListLength { column, .. } => {
            add_column_once(columns, column.clone());
        }
        ScalarSqlExpression::ListIndex { column, index, .. } => {
            add_column_once(columns, column.clone());
            add_scalar_expression_columns(columns, index);
        }
        ScalarSqlExpression::Literal(_) => {}
        ScalarSqlExpression::Binary { left, right, .. } => {
            add_scalar_expression_columns(columns, left);
            add_scalar_expression_columns(columns, right);
        }
        ScalarSqlExpression::Cast { expr, .. }
        | ScalarSqlExpression::Lower(expr)
        | ScalarSqlExpression::Upper(expr)
        | ScalarSqlExpression::Length(expr)
        | ScalarSqlExpression::Trim(expr)
        | ScalarSqlExpression::Abs(expr)
        | ScalarSqlExpression::Round(expr)
        | ScalarSqlExpression::Floor(expr)
        | ScalarSqlExpression::Ceil(expr)
        | ScalarSqlExpression::ExtractYear(expr) => add_scalar_expression_columns(columns, expr),
        ScalarSqlExpression::Coalesce(values) => {
            for value in values {
                add_scalar_expression_columns(columns, value);
            }
        }
        ScalarSqlExpression::Concat(values) => {
            for value in values {
                add_scalar_expression_columns(columns, value);
            }
        }
        ScalarSqlExpression::Replace { expr, from, to } => {
            add_scalar_expression_columns(columns, expr);
            add_scalar_expression_columns(columns, from);
            add_scalar_expression_columns(columns, to);
        }
        ScalarSqlExpression::Substring {
            expr,
            start,
            length,
        } => {
            add_scalar_expression_columns(columns, expr);
            add_scalar_expression_columns(columns, start);
            if let Some(length) = length {
                add_scalar_expression_columns(columns, length);
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
                add_scalar_expression_columns(columns, result);
            }
            if let Some(else_result) = else_result {
                add_scalar_expression_columns(columns, else_result);
            }
        }
    }
}

fn expression_aggregate_dictionary_columns(group_keys: &[GroupKeyExpr]) -> Vec<String> {
    group_keys
        .iter()
        .filter_map(|group_key| match group_key {
            GroupKeyExpr::CoalesceLiteral {
                column,
                fallback: GroupKeyLiteral::Utf8(_),
            } => Some(column.clone()),
            _ => None,
        })
        .collect()
}

fn expression_aggregate_row_group_map_chunk() -> usize {
    std::env::var("DODAM_EXPRESSION_AGG_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(scan_aggregate_row_group_chunk)
}

pub(super) fn group_key_exprs_for_aggregate(
    group_by: &[String],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Option<Vec<GroupKeyExpr>> {
    if !matches!(
        aggregates,
        [AggregateExpr::CountStar, AggregateExpr::Sum(_)]
    ) {
        return None;
    }
    if group_by.len() == 2 && expressions.len() == 2 {
        let mut keys = Vec::with_capacity(2);
        for group_column in group_by {
            let expression = expressions
                .iter()
                .find(|expression| &expression.output_name == group_column)?;
            keys.push(expression_group_key_expr(&expression.expr)?);
        }
        return Some(keys);
    }
    if expressions.len() != 1 {
        return None;
    }
    if group_by.last()? != &expressions[0].output_name {
        return None;
    }
    if group_by.len() == 1
        && let Some(group_key) = simple_case_literal_group_key(&expressions[0].expr)
    {
        return Some(vec![group_key]);
    }
    if !(group_by.len() == 2 || group_by.len() == 3) {
        return None;
    }
    let ScalarSqlExpression::Coalesce(values) = &expressions[0].expr else {
        return None;
    };
    let [left, right] = values.as_slice() else {
        return None;
    };
    let (column, fallback) =
        coalesce_column_literal(left, right).or_else(|| coalesce_column_literal(right, left))?;
    let mut keys = group_by[..group_by.len() - 1]
        .iter()
        .cloned()
        .map(GroupKeyExpr::Column)
        .collect::<Vec<_>>();
    keys.push(GroupKeyExpr::CoalesceLiteral { column, fallback });
    Some(keys)
}

fn expression_group_key_expr(expr: &ScalarSqlExpression) -> Option<GroupKeyExpr> {
    lower_coalesce_group_key_expr(expr).or_else(|| int64_add_literal_group_key_expr(expr))
}

fn lower_coalesce_group_key_expr(expr: &ScalarSqlExpression) -> Option<GroupKeyExpr> {
    let ScalarSqlExpression::Lower(inner) = expr else {
        return None;
    };
    let ScalarSqlExpression::Coalesce(values) = inner.as_ref() else {
        return None;
    };
    let [left, right] = values.as_slice() else {
        return None;
    };
    let (column, fallback) =
        coalesce_column_literal(left, right).or_else(|| coalesce_column_literal(right, left))?;
    let GroupKeyLiteral::Utf8(fallback) = fallback else {
        return None;
    };
    Some(GroupKeyExpr::LowerCoalesceLiteral { column, fallback })
}

fn int64_add_literal_group_key_expr(expr: &ScalarSqlExpression) -> Option<GroupKeyExpr> {
    let ScalarSqlExpression::Binary { left, op, right } = expr else {
        return None;
    };
    if *op != BinaryOperator::Plus {
        return None;
    }
    int64_add_literal_group_key_side(left, right)
        .or_else(|| int64_add_literal_group_key_side(right, left))
}

fn int64_add_literal_group_key_side(
    column_expr: &ScalarSqlExpression,
    literal_expr: &ScalarSqlExpression,
) -> Option<GroupKeyExpr> {
    let ScalarSqlExpression::Column(column) = column_expr else {
        return None;
    };
    let ScalarSqlExpression::Literal(LiteralValue::Int64(value)) = literal_expr else {
        return None;
    };
    Some(GroupKeyExpr::Int64AddLiteral {
        column: column.clone(),
        value: *value,
    })
}

pub(super) fn simple_case_literal_group_key(expr: &ScalarSqlExpression) -> Option<GroupKeyExpr> {
    let ScalarSqlExpression::Case {
        conditions,
        results,
        else_result,
    } = expr
    else {
        return None;
    };
    if conditions.len() != results.len() {
        return None;
    }
    let ScalarSqlExpression::Literal(else_literal) = else_result.as_deref()? else {
        return None;
    };
    let mut column_name = None::<String>;
    let mut branches = Vec::with_capacity(conditions.len());
    for (condition, result) in conditions.iter().zip(results) {
        let (condition_column, condition_literal) = equality_column_literal(condition)?;
        match &column_name {
            Some(column_name) if column_name != &condition_column => return None,
            None => column_name = Some(condition_column),
            _ => {}
        }
        let ScalarSqlExpression::Literal(result_literal) = result else {
            return None;
        };
        branches.push((
            group_key_literal(&condition_literal),
            group_key_literal(result_literal),
        ));
    }
    Some(GroupKeyExpr::SimpleCaseLiteral {
        column: column_name?,
        branches,
        else_value: group_key_literal(else_literal),
    })
}

fn equality_column_literal(expr: &SqlExpr) -> Option<(String, LiteralValue)> {
    let SqlExpr::BinaryOp { left, op, right } = expr else {
        return None;
    };
    if *op != BinaryOperator::Eq {
        return None;
    }
    equality_column_literal_side(left, right).or_else(|| equality_column_literal_side(right, left))
}

fn equality_column_literal_side(
    column_expr: &SqlExpr,
    literal_expr: &SqlExpr,
) -> Option<(String, LiteralValue)> {
    let column = sql_column_name(column_expr, None).ok()?;
    let literal = sql_literal_value(literal_expr).ok()?;
    Some((column, literal))
}

fn coalesce_column_literal(
    column_expr: &ScalarSqlExpression,
    literal_expr: &ScalarSqlExpression,
) -> Option<(String, GroupKeyLiteral)> {
    let ScalarSqlExpression::Column(column) = column_expr else {
        return None;
    };
    let ScalarSqlExpression::Literal(value) = literal_expr else {
        return None;
    };
    Some((column.clone(), group_key_literal(value)))
}

fn group_key_literal(value: &LiteralValue) -> GroupKeyLiteral {
    match value {
        LiteralValue::Null => GroupKeyLiteral::Null,
        LiteralValue::Boolean(value) => GroupKeyLiteral::Boolean(*value),
        LiteralValue::Int64(value) => GroupKeyLiteral::Int64(*value),
        LiteralValue::Float64(value) => GroupKeyLiteral::Float64(value.to_bits()),
        LiteralValue::Utf8(value) => GroupKeyLiteral::Utf8(value.clone()),
    }
}
