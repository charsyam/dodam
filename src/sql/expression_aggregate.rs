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
        if filtered_aggregates.is_empty()
            && let Some((sum_expr, left_column, right_column)) =
                global_sum_product_expression_shape(aggregates, expressions)
        {
            let metrics = collect_global_sum_product_expression(
                stream,
                fragments,
                sum_expr.clone(),
                left_column,
                right_column,
            )?;
            generic_profile_elapsed("aggregate global sum-product direct", started);
            return Ok(metrics);
        }
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

fn collect_global_sum_product_expression(
    mut stream: SendableBatchStream,
    fragments: usize,
    sum_expr: AggregateExpr,
    left_column: &str,
    right_column: &str,
) -> Result<AggregateMetrics> {
    let started = Instant::now();
    let mut rows = 0usize;
    let mut batches = 0usize;
    let mut int_sum = 0i64;
    let mut float_sum = 0f64;
    let mut count = 0u64;
    let mut float_result = false;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let Some(left) = SumProductInput::new(&batch, left_column)? else {
            return Err(DodamError::UnsupportedSql(format!(
                "SUM product input column is not numeric: {left_column}"
            )));
        };
        let Some(right) = SumProductInput::new(&batch, right_column)? else {
            return Err(DodamError::UnsupportedSql(format!(
                "SUM product input column is not numeric: {right_column}"
            )));
        };
        rows = rows.saturating_add(batch.num_rows());
        batches += 1;
        if left.is_integer() && right.is_integer() && !float_result {
            let (batch_sum, batch_count) = left.sum_product_i64(&right, batch.num_rows())?;
            int_sum = int_sum.checked_add(batch_sum).ok_or_else(|| {
                DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
            })?;
            count += batch_count;
        } else {
            if !float_result {
                float_sum = int_sum as f64;
                float_result = true;
            }
            let (batch_sum, batch_count) = left.sum_product_f64(&right, batch.num_rows());
            float_sum += batch_sum;
            count += batch_count;
        }
    }
    let value = if float_result {
        AggregateValue::Float64((count > 0).then_some(float_sum))
    } else if count == 0 {
        AggregateValue::Int64(None)
    } else {
        AggregateValue::Int64(Some(int_sum))
    };
    Ok(AggregateMetrics {
        fragments,
        batches,
        rows,
        values: vec![AggregateResult {
            expr: sum_expr,
            value,
        }],
        aggregate_nanos: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        ..AggregateMetrics::default()
    })
}

fn global_sum_product_expression_shape<'a>(
    aggregates: &'a [AggregateExpr],
    expressions: &'a [ProjectionExpression],
) -> Option<(&'a AggregateExpr, &'a str, &'a str)> {
    let [AggregateExpr::Sum(sum_column)] = aggregates else {
        return None;
    };
    let [expression] = expressions else {
        return None;
    };
    if expression.output_name != *sum_column {
        return None;
    }
    let Some((left_column, right_column)) = sum_product_i64_columns(&expression.expr) else {
        return None;
    };
    Some((&aggregates[0], left_column, right_column))
}

fn sum_product_i64_columns(expr: &ScalarSqlExpression) -> Option<(&str, &str)> {
    let ScalarSqlExpression::Binary { left, op, right } = expr else {
        return None;
    };
    if *op != BinaryOperator::Multiply {
        return None;
    }
    let ScalarSqlExpression::Column(left) = left.as_ref() else {
        return None;
    };
    let ScalarSqlExpression::Column(right) = right.as_ref() else {
        return None;
    };
    Some((left.as_str(), right.as_str()))
}

enum SumProductInput<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float64(&'a Float64Array),
    Decimal128(DecimalInput<'a>),
}

impl<'a> SumProductInput<'a> {
    fn new(batch: &'a RecordBatch, column: &str) -> Result<Option<Self>> {
        let values = batch.column(sum_product_column_index(batch, column)?);
        match values.data_type() {
            DataType::Int32 => Ok(Some(Self::Int32(
                values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 sum-product input"),
            ))),
            DataType::Int64 => Ok(Some(Self::Int64(
                values
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 sum-product input"),
            ))),
            DataType::Float64 => Ok(Some(Self::Float64(
                values
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 sum-product input"),
            ))),
            DataType::Decimal128(_, _) => Ok(decimal_input(values)?.map(Self::Decimal128)),
            _ => Ok(None),
        }
    }

    fn is_integer(&self) -> bool {
        matches!(self, Self::Int32(_) | Self::Int64(_))
    }

    #[inline]
    fn value(&self, row: usize) -> Option<i64> {
        match self {
            Self::Int32(values) => values.is_valid(row).then(|| i64::from(values.value(row))),
            Self::Int64(values) => values.is_valid(row).then(|| values.value(row)),
            Self::Float64(_) => None,
            Self::Decimal128(_) => None,
        }
    }

    #[inline]
    fn value_f64(&self, row: usize) -> Option<f64> {
        match self {
            Self::Int32(values) => values.is_valid(row).then(|| f64::from(values.value(row))),
            Self::Int64(values) => values.is_valid(row).then(|| values.value(row) as f64),
            Self::Float64(values) => values.is_valid(row).then(|| values.value(row)),
            Self::Decimal128(values) => (!values.is_null(row)).then(|| values.value(row)),
        }
    }

    fn sum_product_i64(&self, other: &Self, rows: usize) -> Result<(i64, u64)> {
        match (self, other) {
            (Self::Int32(left), Self::Int32(right))
                if left.null_count() == 0 && right.null_count() == 0 =>
            {
                sum_product_non_null_i32_i32(left.values(), right.values())
            }
            (Self::Int32(left), Self::Int64(right))
                if left.null_count() == 0 && right.null_count() == 0 =>
            {
                sum_product_non_null_i32_i64(left.values(), right.values())
            }
            (Self::Int64(left), Self::Int32(right))
                if left.null_count() == 0 && right.null_count() == 0 =>
            {
                sum_product_non_null_i64_i32(left.values(), right.values())
            }
            (Self::Int64(left), Self::Int64(right))
                if left.null_count() == 0 && right.null_count() == 0 =>
            {
                sum_product_non_null_i64_i64(left.values(), right.values())
            }
            _ => self.sum_product_nullable(other, rows),
        }
    }

    fn sum_product_f64(&self, other: &Self, rows: usize) -> (f64, u64) {
        match (self, other) {
            (Self::Decimal128(left), Self::Decimal128(right))
                if left.null_count() == 0 && right.null_count() == 0 =>
            {
                sum_product_non_null_decimal_decimal(left, right)
            }
            (Self::Float64(left), Self::Float64(right))
                if left.null_count() == 0 && right.null_count() == 0 =>
            {
                sum_product_non_null_f64_f64(left.values(), right.values())
            }
            _ => self.sum_product_nullable_f64(other, rows),
        }
    }

    fn sum_product_nullable(&self, other: &Self, rows: usize) -> Result<(i64, u64)> {
        let mut sum = 0i64;
        let mut count = 0u64;
        for row in 0..rows {
            let Some(left) = self.value(row) else {
                continue;
            };
            let Some(right) = other.value(row) else {
                continue;
            };
            let Some(product) = left.checked_mul(right) else {
                continue;
            };
            sum = sum.checked_add(product).ok_or_else(|| {
                DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
            })?;
            count += 1;
        }
        Ok((sum, count))
    }

    fn sum_product_nullable_f64(&self, other: &Self, rows: usize) -> (f64, u64) {
        let mut sum = 0f64;
        let mut count = 0u64;
        for row in 0..rows {
            let Some(left) = self.value_f64(row) else {
                continue;
            };
            let Some(right) = other.value_f64(row) else {
                continue;
            };
            sum += left * right;
            count += 1;
        }
        (sum, count)
    }
}

fn sum_product_column_index(batch: &RecordBatch, column: &str) -> Result<usize> {
    batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == column)
        .ok_or_else(|| DodamError::UnknownColumn(column.to_string()))
}

fn sum_product_non_null_i32_i32(left: &[i32], right: &[i32]) -> Result<(i64, u64)> {
    let mut sum = 0i64;
    for (&left, &right) in left.iter().zip(right) {
        let product = i64::from(left) * i64::from(right);
        sum = sum.checked_add(product).ok_or_else(|| {
            DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
        })?;
    }
    Ok((sum, left.len() as u64))
}

fn sum_product_non_null_i32_i64(left: &[i32], right: &[i64]) -> Result<(i64, u64)> {
    let mut sum = 0i64;
    let mut count = 0u64;
    for (&left, &right) in left.iter().zip(right) {
        let Some(product) = i64::from(left).checked_mul(right) else {
            continue;
        };
        sum = sum.checked_add(product).ok_or_else(|| {
            DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
        })?;
        count += 1;
    }
    Ok((sum, count))
}

fn sum_product_non_null_i64_i32(left: &[i64], right: &[i32]) -> Result<(i64, u64)> {
    let mut sum = 0i64;
    let mut count = 0u64;
    for (&left, &right) in left.iter().zip(right) {
        let Some(product) = left.checked_mul(i64::from(right)) else {
            continue;
        };
        sum = sum.checked_add(product).ok_or_else(|| {
            DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
        })?;
        count += 1;
    }
    Ok((sum, count))
}

fn sum_product_non_null_i64_i64(left: &[i64], right: &[i64]) -> Result<(i64, u64)> {
    let mut sum = 0i64;
    let mut count = 0u64;
    for (&left, &right) in left.iter().zip(right) {
        let Some(product) = left.checked_mul(right) else {
            continue;
        };
        sum = sum.checked_add(product).ok_or_else(|| {
            DodamError::UnsupportedSql("SUM integer expression overflow".to_string())
        })?;
        count += 1;
    }
    Ok((sum, count))
}

fn sum_product_non_null_decimal_decimal(
    left: &DecimalInput<'_>,
    right: &DecimalInput<'_>,
) -> (f64, u64) {
    let mut sum = 0f64;
    let left_scale = left.scale;
    let right_scale = right.scale;
    for (&left_value, &right_value) in left.raw_values().iter().zip(right.raw_values()) {
        sum += (left_value as f64 / left_scale) * (right_value as f64 / right_scale);
    }
    (sum, left.raw_values().len() as u64)
}

fn sum_product_non_null_f64_f64(left: &[f64], right: &[f64]) -> (f64, u64) {
    let mut sum = 0f64;
    for (&left, &right) in left.iter().zip(right) {
        sum += left * right;
    }
    (sum, left.len() as u64)
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
    if let Some(metrics) = try_collect_two_expression_dictionary_scan_fold(
        engine,
        &path,
        batch_size,
        filter.as_ref(),
        group_by,
        aggregates,
        &group_keys,
        ordered_output,
        output_limit,
    )? {
        return Ok(Some(metrics));
    }
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
    if !dictionary_columns.is_empty() && expression_aggregate_dictionary_scan_fold_enabled() {
        let predicates = PredicateSet::new(filter.clone());
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
                    let filter_for_dictionary = filter.clone();
                    move |view, collector: &mut CoalesceKeyCountSumCollector| {
                        if let Some(filter_for_dictionary) = filter_for_dictionary.as_ref() {
                            let mask = if let Some(mask) = evaluate_projected_view_filter_mask(
                                view,
                                &projected_columns,
                                filter_for_dictionary,
                            )? {
                                mask
                            } else {
                                if !expression_aggregate_row_at_time_fallback_enabled() {
                                    return Ok(None);
                                }
                                let Some(batch) = view.try_record_batch() else {
                                    return Ok(None);
                                };
                                evaluate_filter_mask(batch, filter_for_dictionary)?
                            };
                            if mask.true_count() == 0 {
                                return Ok(Some(()));
                            }
                            if collector.try_consume_projected_view_masked_vectorized(
                                view,
                                &projected_columns,
                                &mask,
                            )? {
                                return Ok(Some(()));
                            }
                            if expression_aggregate_row_at_time_fallback_enabled() {
                                collector
                                    .consume_projected_view_masked_with_record_batch_fallback(
                                        view,
                                        &projected_columns,
                                        &mask,
                                    )?;
                            } else {
                                return Ok(None);
                            }
                            return Ok(Some(()));
                        }
                        if collector
                            .try_consume_projected_view_vectorized(view, &projected_columns)?
                        {
                            return Ok(Some(()));
                        }
                        if expression_aggregate_row_at_time_fallback_enabled() {
                            collector.consume_projected_view(view, &projected_columns)?;
                        } else {
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

#[allow(clippy::too_many_arguments)]
fn try_collect_two_expression_dictionary_scan_fold(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    filter: Option<&FilterExpr>,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    group_keys: &[GroupKeyExpr],
    ordered_output: bool,
    output_limit: Option<usize>,
) -> Result<Option<AggregateMetrics>> {
    if filter.is_some() {
        return Ok(None);
    }
    let [AggregateExpr::CountStar, AggregateExpr::Sum(sum_column)] = aggregates else {
        return Ok(None);
    };
    let (lower_column, lower_fallback, add_column, add_value, lower_first) = match group_keys {
        [
            GroupKeyExpr::LowerCoalesceLiteral { column, fallback },
            GroupKeyExpr::Int64AddLiteral {
                column: add_column,
                value,
            },
        ] => (
            column.as_str(),
            fallback.as_str(),
            add_column.as_str(),
            *value,
            true,
        ),
        [
            GroupKeyExpr::Int64AddLiteral { column, value },
            GroupKeyExpr::LowerCoalesceLiteral {
                column: lower_column,
                fallback,
            },
        ] => (
            lower_column.as_str(),
            fallback.as_str(),
            column.as_str(),
            *value,
            false,
        ),
        _ => return Ok(None),
    };
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let mut state = TwoExpressionDictionaryScanFoldState::new(
        lower_fallback,
        aggregates[0].clone(),
        aggregates[1].clone(),
        lower_first,
    );
    let started = Instant::now();
    let Some(scan_metrics) = engine.scan_parquet_i32_i64_dictionary_id_columns(
        path,
        batch_size,
        &row_groups,
        [add_column, sum_column.as_str(), lower_column],
        |add_values, sum_values, lower_def_levels, lower_ids, dictionary| {
            state.consume_dictionary_ids(
                add_values,
                add_value,
                sum_values,
                lower_def_levels,
                lower_ids,
                dictionary,
            )?;
            Ok(Some(()))
        },
    )?
    else {
        return Ok(None);
    };
    let mut metrics = state.finish(1, group_by, ordered_output, output_limit);
    metrics.batches = scan_metrics.batches;
    metrics.rows = scan_metrics.rows;
    metrics.aggregate_nanos = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    generic_profile_elapsed(
        "aggregate two-expression dictionary scan-fold",
        Some(started),
    );
    Ok(Some(metrics))
}

struct TwoExpressionDictionaryScanFoldState {
    groups: std::collections::HashMap<(String, i64), usize>,
    group_values: Vec<(String, i64, u64, i64)>,
    fallback: String,
    count_expr: AggregateExpr,
    sum_expr: AggregateExpr,
    lower_first: bool,
    dictionary_cache: Vec<String>,
    dictionary_group_slots: Vec<usize>,
    fallback_group_slots: Vec<usize>,
}

impl TwoExpressionDictionaryScanFoldState {
    fn new(
        fallback: &str,
        count_expr: AggregateExpr,
        sum_expr: AggregateExpr,
        lower_first: bool,
    ) -> Self {
        Self {
            groups: std::collections::HashMap::new(),
            group_values: Vec::new(),
            fallback: ascii_lower_if_needed(fallback),
            count_expr,
            sum_expr,
            lower_first,
            dictionary_cache: Vec::new(),
            dictionary_group_slots: Vec::new(),
            fallback_group_slots: Vec::new(),
        }
    }

    fn consume_dictionary_ids(
        &mut self,
        add_values: &[i32],
        add_value: i64,
        sum_values: &[i64],
        lower_def_levels: Option<&[i16]>,
        lower_ids: &[i32],
        dictionary: &[bytes::Bytes],
    ) -> Result<()> {
        if add_values.len() != sum_values.len() {
            return Err(DodamError::UnsupportedSql(
                "two-expression dictionary scan-fold input length mismatch".to_string(),
            ));
        }
        self.refresh_dictionary_cache(dictionary)?;
        let Some((min_key, width)) = add_key_range(add_values, add_value) else {
            return Ok(());
        };
        let Some(dictionary_slots) = dictionary.len().checked_mul(width) else {
            return Err(DodamError::UnsupportedSql(
                "expression aggregate dictionary slot overflow".to_string(),
            ));
        };
        if width > 4096 || dictionary_slots > 1_048_576 {
            return Err(DodamError::UnsupportedSql(
                "expression aggregate dictionary slot range too large".to_string(),
            ));
        }
        self.dictionary_group_slots.clear();
        self.dictionary_group_slots
            .resize(dictionary_slots, usize::MAX);
        self.fallback_group_slots.clear();
        self.fallback_group_slots.resize(width, usize::MAX);
        let mut dictionary_offset = 0usize;
        for row in 0..add_values.len() {
            let add_key = i64::from(add_values[row]).saturating_add(add_value);
            let offset = usize::try_from(add_key - min_key).map_err(|_| {
                DodamError::UnsupportedSql(
                    "expression aggregate add-key offset out of range".to_string(),
                )
            })?;
            let group_id = if lower_def_levels.is_some_and(|levels| levels[row] == 0) {
                if self.fallback_group_slots[offset] != usize::MAX {
                    self.fallback_group_slots[offset]
                } else {
                    let group_id = self.group_id_for_key(self.fallback.clone(), add_key);
                    self.fallback_group_slots[offset] = group_id;
                    group_id
                }
            } else {
                let Some(&dictionary_id) = lower_ids.get(dictionary_offset) else {
                    return Err(DodamError::UnsupportedSql(
                        "dictionary id missing in expression aggregate scan-fold".to_string(),
                    ));
                };
                dictionary_offset += 1;
                let dictionary_id = usize::try_from(dictionary_id).map_err(|_| {
                    DodamError::UnsupportedSql(
                        "negative dictionary id in expression aggregate scan-fold".to_string(),
                    )
                })?;
                if dictionary_id >= self.dictionary_cache.len() {
                    return Err(DodamError::UnsupportedSql(
                        "dictionary id out of range in expression aggregate scan-fold".to_string(),
                    ));
                }
                let slot = dictionary_id
                    .checked_mul(width)
                    .and_then(|slot| slot.checked_add(offset))
                    .ok_or_else(|| {
                        DodamError::UnsupportedSql(
                            "expression aggregate dictionary slot overflow".to_string(),
                        )
                    })?;
                if self.dictionary_group_slots[slot] != usize::MAX {
                    self.dictionary_group_slots[slot]
                } else {
                    let group_id = self
                        .group_id_for_key(self.dictionary_cache[dictionary_id].clone(), add_key);
                    self.dictionary_group_slots[slot] = group_id;
                    group_id
                }
            };
            let entry = &mut self.group_values[group_id];
            entry.2 = entry.2.saturating_add(1);
            entry.3 = entry.3.saturating_add(sum_values[row]);
        }
        if dictionary_offset != lower_ids.len() {
            return Err(DodamError::UnsupportedSql(
                "unused dictionary ids in expression aggregate scan-fold".to_string(),
            ));
        }
        Ok(())
    }

    fn group_id_for_key(&mut self, key: String, add_key: i64) -> usize {
        if let Some(group_id) = self.groups.get(&(key.clone(), add_key)).copied() {
            return group_id;
        }
        let group_id = self.group_values.len();
        self.groups.insert((key.clone(), add_key), group_id);
        self.group_values.push((key, add_key, 0, 0));
        group_id
    }

    fn refresh_dictionary_cache(&mut self, dictionary: &[bytes::Bytes]) -> Result<()> {
        self.dictionary_cache.clear();
        self.dictionary_cache.reserve(dictionary.len());
        for value in dictionary {
            let value = std::str::from_utf8(value.as_ref()).map_err(|_| {
                DodamError::UnsupportedSql(
                    "invalid UTF8 dictionary value in expression aggregate scan-fold".to_string(),
                )
            })?;
            self.dictionary_cache.push(ascii_lower_if_needed(value));
        }
        Ok(())
    }

    fn finish(
        self,
        fragments: usize,
        _group_by: &[String],
        ordered_output: bool,
        output_limit: Option<usize>,
    ) -> AggregateMetrics {
        let mut groups = self
            .group_values
            .into_iter()
            .map(|(lower_key, add_key, count, sum)| {
                let keys = if self.lower_first {
                    vec![
                        GroupValue::Utf8(Some(lower_key)),
                        GroupValue::Int64(Some(add_key)),
                    ]
                } else {
                    vec![
                        GroupValue::Int64(Some(add_key)),
                        GroupValue::Utf8(Some(lower_key)),
                    ]
                };
                GroupAggregateResult {
                    keys,
                    values: vec![
                        AggregateResult {
                            expr: self.count_expr.clone(),
                            value: AggregateValue::Count(count),
                        },
                        AggregateResult {
                            expr: self.sum_expr.clone(),
                            value: AggregateValue::Int64(Some(sum)),
                        },
                    ],
                }
            })
            .collect::<Vec<_>>();
        if ordered_output {
            groups.sort_by(|left, right| {
                compare_group_keys_for_expression_fold(&left.keys, &right.keys)
            });
            if let Some(limit) = output_limit {
                groups.truncate(limit);
            }
        }
        AggregateMetrics {
            fragments,
            groups,
            ..AggregateMetrics::default()
        }
    }
}

fn add_key_range(values: &[i32], add_value: i64) -> Option<(i64, usize)> {
    let min = i64::from(values.iter().copied().min()?).saturating_add(add_value);
    let max = i64::from(values.iter().copied().max()?).saturating_add(add_value);
    let width = max
        .checked_sub(min)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| usize::try_from(value).ok())?;
    Some((min, width))
}

fn ascii_lower_if_needed(value: &str) -> String {
    if value.as_bytes().iter().any(u8::is_ascii_uppercase) {
        value.to_ascii_lowercase()
    } else {
        value.to_string()
    }
}

fn compare_group_keys_for_expression_fold(
    left: &[GroupValue],
    right: &[GroupValue],
) -> std::cmp::Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| match (left, right) {
            (GroupValue::Utf8(left), GroupValue::Utf8(right)) => left.cmp(right),
            (GroupValue::Int64(left), GroupValue::Int64(right)) => left.cmp(right),
            (GroupValue::Utf8(_), _) => std::cmp::Ordering::Greater,
            (GroupValue::Int64(_), _) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Equal,
        })
        .find(|ordering| *ordering != std::cmp::Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
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

pub(super) fn push_projected_view_filter_selection(
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

fn projected_view_column_index(columns: &[String], column: &str) -> Option<usize> {
    columns.iter().position(|candidate| candidate == column)
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
    !std::env::var("DODAM_DISABLE_EXPRESSION_AGG_DICTIONARY_SCAN_FOLD")
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
            }
            | GroupKeyExpr::LowerCoalesceLiteral { column, .. } => Some(column.clone()),
            _ => None,
        })
        .collect()
}

fn expression_aggregate_row_group_map_chunk() -> usize {
    row_group_map_chunk_size(
        "DODAM_EXPRESSION_AGG_ROW_GROUP_CHUNK",
        scan_aggregate_row_group_chunk(),
    )
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
