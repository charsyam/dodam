use super::*;

pub(super) async fn try_execute_join_coalesce_count_sum_aggregate(
    engine: &DodamEngine,
    query: &SqlQuery,
    join: &SqlJoin,
    join_plan: &JoinInputPlan,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let profile = join_profile_enabled_sql();
    let total_started = profile.then(Instant::now);
    if !join_coalesce_count_sum_fusion_enabled() {
        return Ok(None);
    }
    if join.join_type != JoinType::Inner
        || join.left_keys.len() != 1
        || join.right_keys.len() != 1
        || query.having.is_some()
        || query.distinct
        || query.expression_filter.is_some()
    {
        trace_join_coalesce_fusion("reject", "unsupported join shape");
        return Ok(None);
    }
    let [AggregateExpr::CountStar, AggregateExpr::Sum(sum_column)] = query.aggregates.as_slice()
    else {
        trace_join_coalesce_fusion("reject", "unsupported aggregate list");
        return Ok(None);
    };
    let Some(group_keys) = group_key_exprs_for_aggregate(
        &query.group_by,
        &query.aggregates,
        &query.aggregate_expressions,
    ) else {
        trace_join_coalesce_fusion("reject", "unsupported group expression list");
        return Ok(None);
    };
    let [
        GroupKeyExpr::Column(left_group_column),
        GroupKeyExpr::CoalesceLiteral {
            column: right_payload_column,
            fallback: GroupKeyLiteral::Utf8(fallback),
        },
    ] = group_keys.as_slice()
    else {
        trace_join_coalesce_fusion("reject", "unsupported group key shape");
        return Ok(None);
    };
    trace_join_coalesce_fusion(
        "candidate",
        &format!(
            "left_group={} right_payload={} sum={} group_by=[{}] aliases=[{}]",
            left_group_column,
            right_payload_column,
            sum_column,
            query.group_by.join(","),
            query
                .aliases
                .iter()
                .map(|(left, right)| format!("{left}->{right}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    if !join_column_belongs_to(&left_group_column, &join.left_alias)
        || !join_column_belongs_to(sum_column, &join.left_alias)
        || !join_column_belongs_to(&right_payload_column, &join.right_alias)
    {
        trace_join_coalesce_fusion("reject", "columns do not belong to expected join sides");
        return Ok(None);
    }

    let right_key = strip_column_prefix(&join.right_keys[0], &join.right_alias);
    let right_payload = strip_column_prefix(&right_payload_column, &join.right_alias);
    let right_scan_started = profile.then(Instant::now);
    let right_batches = scan_join_side_batches(
        engine,
        &join.right,
        batch_size,
        join_plan.right_filter.as_ref(),
        Projection::Columns(vec![right_key.clone(), right_payload.clone()]),
    )
    .await?;
    let right_scan_nanos = elapsed_optional_nanos(right_scan_started);
    let right_lookup_started = profile.then(Instant::now);
    let Some(right_lookup) =
        build_unique_i64_to_utf8_id_lookup(&right_batches, &right_key, &right_payload)?
    else {
        trace_join_coalesce_fusion("reject", "right join key is not unique");
        return Ok(None);
    };
    let right_lookup_nanos = elapsed_optional_nanos(right_lookup_started);
    let right_values = right_lookup.values;

    let left_key = strip_column_prefix(&join.left_keys[0], &join.left_alias);
    let left_group = strip_column_prefix(&left_group_column, &join.left_alias);
    let left_sum = strip_column_prefix(sum_column, &join.left_alias);
    let left_projection = Projection::Columns(unique_columns([
        left_key.clone(),
        left_group.clone(),
        left_sum.clone(),
    ]));
    let left_scan_started = profile.then(Instant::now);
    let left_late = try_late_join_coalesce_left_aggregate(
        engine,
        query.path.clone(),
        batch_size,
        join_plan.left_filter.as_ref(),
        &left_key,
        &left_group,
        &left_sum,
        right_lookup.lookup.clone(),
    )
    .await?;
    let left_direct = if left_late.is_some() {
        None
    } else {
        try_direct_join_coalesce_left_aggregate(
            engine,
            &query.path,
            batch_size,
            join_plan.left_filter.as_ref(),
            &left_key,
            &left_group,
            &left_sum,
            &right_lookup.lookup,
        )?
    };
    let (groups, rows, batches, left_scan_nanos, aggregate_nanos) =
        if let Some(left_late) = left_late {
            (
                left_late.groups.into_iter().collect::<Vec<_>>(),
                left_late.rows,
                left_late.batches,
                elapsed_optional_nanos(left_scan_started),
                left_late.aggregate_nanos,
            )
        } else if let Some(left_direct) = left_direct {
            (
                left_direct.groups.into_iter().collect::<Vec<_>>(),
                left_direct.rows,
                left_direct.batches,
                elapsed_optional_nanos(left_scan_started),
                left_direct.aggregate_nanos,
            )
        } else {
            let mut left_stream = engine
                .scan_parquet_batches(
                    query.path.clone(),
                    batch_size,
                    None,
                    left_projection,
                    join_plan.left_filter.clone(),
                )
                .await?;
            let aggregate_started = profile.then(Instant::now);
            let right_dense_lookup = right_lookup.lookup.dense_slices();
            let mut groups = JoinCoalesceGroupAccumulator::new();
            let mut rows = 0usize;
            let mut batches = 0usize;
            while let Some(batch) = left_stream.next() {
                let batch = batch?;
                if batch.num_rows() == 0 {
                    continue;
                }
                batches += 1;
                rows = rows.saturating_add(batch.num_rows());
                let key = i64_array_like(&batch, &left_key)?;
                let group = i64_array_like(&batch, &left_group)?;
                let sum = i64_array_like(&batch, &left_sum)?;
                if !key.has_nulls() && !group.has_nulls() && !sum.has_nulls() {
                    for row in 0..batch.num_rows() {
                        if let Some(class_id) = right_lookup
                            .lookup
                            .get_cached(right_dense_lookup, key.value(row))
                        {
                            groups.update_non_null(group.value(row), class_id, sum.value(row));
                        }
                    }
                } else {
                    for row in 0..batch.num_rows() {
                        if key.is_null(row) {
                            continue;
                        }
                        let Some(class_id) = right_lookup
                            .lookup
                            .get_cached(right_dense_lookup, key.value(row))
                        else {
                            continue;
                        };
                        groups.update(
                            (!group.is_null(row)).then(|| group.value(row)),
                            class_id,
                            (!sum.is_null(row)).then(|| sum.value(row)),
                        );
                    }
                }
            }
            let left_scan_nanos = elapsed_optional_nanos(left_scan_started);
            (
                groups.into_entries(),
                rows,
                batches,
                left_scan_nanos,
                elapsed_optional_nanos(aggregate_started),
            )
        };

    let finish_started = profile.then(Instant::now);
    let mut group_results = groups
        .into_iter()
        .map(|((bucket, class_id), (count, sum))| GroupAggregateResult {
            keys: vec![
                GroupValue::Int64(bucket),
                GroupValue::Utf8(Some(
                    right_values[class_id]
                        .as_deref()
                        .unwrap_or(&fallback)
                        .to_string(),
                )),
            ],
            values: vec![
                AggregateResult {
                    expr: AggregateExpr::CountStar,
                    value: AggregateValue::Count(count),
                },
                AggregateResult {
                    expr: AggregateExpr::Sum(sum_column.clone()),
                    value: AggregateValue::Int64(Some(sum)),
                },
            ],
        })
        .collect::<Vec<_>>();
    group_results.sort_by(|left, right| compare_join_fused_group_keys(&left.keys, &right.keys));
    let output =
        join_coalesce_count_sum_batches(&group_results, &query.group_by, &query.aggregates)?;
    let metrics = AggregateMetrics {
        fragments: 2,
        batches,
        rows,
        groups: group_results,
        ..AggregateMetrics::default()
    };
    let mut output = output;
    output = apply_output_order_limit(output, query.order_by.as_ref(), query.limit, query.offset)?;
    output = rename_output_batches(output, &query.aliases)?;
    let finish_nanos = elapsed_optional_nanos(finish_started);
    if profile {
        eprintln!(
            "[dodam:join-fusion-profile] total={:.3}ms right_scan={:.3}ms right_lookup={:.3}ms left_scan={:.3}ms aggregate={:.3}ms finish={:.3}ms left_batches={} right_batches={} left_rows={} groups={}",
            sql_nanos_to_millis(elapsed_optional_nanos(total_started)),
            sql_nanos_to_millis(right_scan_nanos),
            sql_nanos_to_millis(right_lookup_nanos),
            sql_nanos_to_millis(left_scan_nanos),
            sql_nanos_to_millis(aggregate_nanos),
            sql_nanos_to_millis(finish_nanos),
            batches,
            right_batches.len(),
            rows,
            metrics.groups.len(),
        );
    }
    trace_join_coalesce_fusion("accept", "unique right-key coalesce count/sum aggregate");
    Ok(Some(QueryOutput::Aggregate {
        metrics,
        batches: output,
    }))
}

pub(super) fn join_coalesce_count_sum_batches(
    groups: &[GroupAggregateResult],
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Vec<RecordBatch>> {
    if group_by.len() != 2 || aggregates.len() != 2 {
        return aggregate_metrics_to_batches(
            &AggregateMetrics {
                groups: groups.to_vec(),
                ..AggregateMetrics::default()
            },
            group_by,
            aggregates,
        );
    }
    let mut bucket_values = Vec::with_capacity(groups.len());
    let mut class_values = Vec::with_capacity(groups.len());
    let mut count_values = Vec::with_capacity(groups.len());
    let mut sum_values = Vec::with_capacity(groups.len());
    for group in groups {
        match group.keys.as_slice() {
            [GroupValue::Int64(bucket), GroupValue::Utf8(class)] => {
                bucket_values.push(*bucket);
                class_values.push(class.clone());
            }
            _ => {
                return aggregate_metrics_to_batches(
                    &AggregateMetrics {
                        groups: groups.to_vec(),
                        ..AggregateMetrics::default()
                    },
                    group_by,
                    aggregates,
                );
            }
        }
        match group.values.as_slice() {
            [
                AggregateResult {
                    value: AggregateValue::Count(count),
                    ..
                },
                AggregateResult {
                    value: AggregateValue::Int64(sum),
                    ..
                },
            ] => {
                count_values.push(Some(*count));
                sum_values.push(*sum);
            }
            _ => {
                return aggregate_metrics_to_batches(
                    &AggregateMetrics {
                        groups: groups.to_vec(),
                        ..AggregateMetrics::default()
                    },
                    group_by,
                    aggregates,
                );
            }
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new(group_by[0].clone(), DataType::Int64, true),
        Field::new(group_by[1].clone(), DataType::Utf8, true),
        Field::new(aggregates[0].to_string(), DataType::UInt64, true),
        Field::new(aggregates[1].to_string(), DataType::Int64, true),
    ]));
    Ok(vec![RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(bucket_values)),
            Arc::new(StringArray::from(class_values)),
            Arc::new(UInt64Array::from(count_values)),
            Arc::new(Int64Array::from(sum_values)),
        ],
    )?])
}

pub(super) fn join_profile_enabled_sql() -> bool {
    std::env::var("DODAM_JOIN_PROFILE").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(super) fn elapsed_optional_nanos(started: Option<Instant>) -> u64 {
    started
        .map(|started| started.elapsed().as_nanos().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

pub(super) fn join_coalesce_count_sum_fusion_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_JOIN_COALESCE_COUNT_SUM_FUSION")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn trace_join_coalesce_fusion(decision: &str, reason: &str) {
    if std::env::var("DODAM_OPTIMIZER_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        eprintln!(
            "[dodam:optimizer] rule=join_coalesce_count_sum_fusion decision={} reason=\"{}\"",
            decision, reason
        );
    }
}

pub(super) async fn scan_join_side_batches(
    engine: &DodamEngine,
    table: &SqlTableRef,
    batch_size: usize,
    filter: Option<&FilterExpr>,
    projection: Projection,
) -> Result<Vec<RecordBatch>> {
    let stream = engine
        .scan_parquet_batches(
            table.path.clone(),
            batch_size,
            None,
            projection,
            filter.cloned(),
        )
        .await?;
    collect_batches(stream)
}

pub(super) struct DirectJoinCoalesceLeftAggregate {
    groups: Vec<((Option<i64>, usize), (u64, i64))>,
    rows: usize,
    batches: usize,
    aggregate_nanos: u64,
}

pub(super) struct JoinCoalesceLateLeftState {
    filter: DirectI64Filter,
    right_lookup: AdaptiveI64Map<usize>,
    selected_buckets: Vec<i64>,
    selected_positions: Vec<usize>,
    payload_offset: usize,
    groups: JoinCoalesceGroupAccumulator,
    selected_rows: usize,
    batches: usize,
}

pub(super) enum JoinCoalesceGroupAccumulator {
    Small(Vec<((Option<i64>, usize), (u64, i64))>),
    Hash(FastHashMap<(Option<i64>, usize), (u64, i64)>),
}

impl JoinCoalesceGroupAccumulator {
    const SMALL_LIMIT: usize = 64;

    fn new() -> Self {
        Self::Small(Vec::new())
    }

    fn update(&mut self, bucket: Option<i64>, class_id: usize, sum: Option<i64>) {
        self.update_entry((bucket, class_id), sum.unwrap_or(0), sum.is_some());
    }

    fn update_non_null(&mut self, bucket: i64, class_id: usize, sum: i64) {
        self.update_entry((Some(bucket), class_id), sum, true);
    }

    fn add_counts(&mut self, key: (Option<i64>, usize), count: u64, sum: i64) {
        match self {
            Self::Small(groups) => {
                for (existing_key, value) in groups.iter_mut() {
                    if *existing_key == key {
                        value.0 = value.0.saturating_add(count);
                        value.1 = value.1.saturating_add(sum);
                        return;
                    }
                }
                if groups.len() < Self::SMALL_LIMIT {
                    groups.push((key, (count, sum)));
                    return;
                }
                let mut hash = groups.drain(..).collect::<FastHashMap<_, _>>();
                let entry = hash.entry(key).or_insert((0, 0));
                entry.0 = entry.0.saturating_add(count);
                entry.1 = entry.1.saturating_add(sum);
                *self = Self::Hash(hash);
            }
            Self::Hash(groups) => {
                let entry = groups.entry(key).or_insert((0, 0));
                entry.0 = entry.0.saturating_add(count);
                entry.1 = entry.1.saturating_add(sum);
            }
        }
    }

    fn update_entry(&mut self, key: (Option<i64>, usize), sum: i64, has_sum: bool) {
        match self {
            Self::Small(groups) => {
                for (existing_key, value) in groups.iter_mut() {
                    if *existing_key == key {
                        update_join_coalesce_group_value(value, sum, has_sum);
                        return;
                    }
                }
                if groups.len() < Self::SMALL_LIMIT {
                    let mut value = (0, 0);
                    update_join_coalesce_group_value(&mut value, sum, has_sum);
                    groups.push((key, value));
                    return;
                }
                let mut hash = groups.drain(..).collect::<FastHashMap<_, _>>();
                let entry = hash.entry(key).or_insert((0, 0));
                update_join_coalesce_group_value(entry, sum, has_sum);
                *self = Self::Hash(hash);
            }
            Self::Hash(groups) => {
                let entry = groups.entry(key).or_insert((0, 0));
                update_join_coalesce_group_value(entry, sum, has_sum);
            }
        }
    }

    fn into_entries(self) -> Vec<((Option<i64>, usize), (u64, i64))> {
        match self {
            Self::Small(groups) => groups,
            Self::Hash(groups) => groups.into_iter().collect(),
        }
    }
}

pub(super) fn update_join_coalesce_group_value(value: &mut (u64, i64), sum: i64, has_sum: bool) {
    value.0 = value.0.saturating_add(1);
    if has_sum {
        value.1 = value.1.saturating_add(sum);
    }
}

pub(super) async fn try_late_join_coalesce_left_aggregate(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    filter: Option<&FilterExpr>,
    key_column: &str,
    group_column: &str,
    sum_column: &str,
    right_lookup: AdaptiveI64Map<usize>,
) -> Result<Option<DirectJoinCoalesceLeftAggregate>> {
    if !join_coalesce_late_left_enabled() {
        return Ok(None);
    }
    let Some(filter) = direct_join_left_i64_filter(filter, group_column)? else {
        return Ok(None);
    };
    let key_column = key_column.to_string();
    let group_column = group_column.to_string();
    let sum_column = sum_column.to_string();
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec![group_column.clone()]),
            Projection::Columns(vec![key_column.clone(), sum_column.clone()]),
            Vec::new(),
            join_coalesce_late_left_row_group_chunk(),
            LateMaterializationPolicy::selective(join_coalesce_late_left_max_selected_ratio()),
            {
                let right_lookup = right_lookup.clone();
                let filter = filter.clone();
                move || JoinCoalesceLateLeftState {
                    filter: filter.clone(),
                    right_lookup: right_lookup.clone(),
                    selected_buckets: Vec::with_capacity(batch_size / 2),
                    selected_positions: Vec::with_capacity(batch_size / 2),
                    payload_offset: 0,
                    groups: JoinCoalesceGroupAccumulator::new(),
                    selected_rows: 0,
                    batches: 0,
                }
            },
            |view, selection, state| {
                let Some(group) = DirectI64ishVector::from_batch(view, 0) else {
                    return Ok(None);
                };
                state.batches = state.batches.saturating_add(1);
                state.filter.push_selected_positions_and_values(
                    &group,
                    &mut state.selected_positions,
                    &mut state.selected_buckets,
                );
                state.selected_rows = state
                    .selected_rows
                    .saturating_add(state.selected_positions.len());
                selection
                    .push_selected_offsets(group.len(), state.selected_positions.iter().copied());
                Ok(Some(()))
            },
            |view, state| {
                let Some(key) = DirectI64ishVector::from_batch(view, 0) else {
                    return Ok(None);
                };
                let Some(sum) = DirectI64ishVector::from_batch(view, 1) else {
                    return Ok(None);
                };
                let dense_lookup = state.right_lookup.dense_slices();
                if !key.has_nulls() && !sum.has_nulls() {
                    for row in 0..key.len() {
                        let Some(bucket) =
                            state.selected_buckets.get(state.payload_offset).copied()
                        else {
                            return Err(DodamError::UnsupportedSql(
                                "join late payload row overflow".to_string(),
                            ));
                        };
                        state.payload_offset += 1;
                        if let Some(class_id) =
                            state.right_lookup.get_cached(dense_lookup, key.value(row))
                        {
                            state
                                .groups
                                .update_non_null(bucket, class_id, sum.value(row));
                        }
                    }
                } else {
                    for row in 0..key.len() {
                        let Some(bucket) =
                            state.selected_buckets.get(state.payload_offset).copied()
                        else {
                            return Err(DodamError::UnsupportedSql(
                                "join late payload row overflow".to_string(),
                            ));
                        };
                        state.payload_offset += 1;
                        if key.is_null(row) {
                            continue;
                        }
                        let Some(class_id) =
                            state.right_lookup.get_cached(dense_lookup, key.value(row))
                        else {
                            continue;
                        };
                        state.groups.update(
                            Some(bucket),
                            class_id,
                            (!sum.is_null(row)).then(|| sum.value(row)),
                        );
                    }
                }
                if state.payload_offset == state.selected_buckets.len() {
                    state.selected_buckets.clear();
                    state.payload_offset = 0;
                }
                Ok(Some(()))
            },
            |state, _metrics| {
                if state.payload_offset != state.selected_buckets.len() {
                    return Err(DodamError::UnsupportedSql(
                        "join late payload row mismatch".to_string(),
                    ));
                }
                Ok(Some(DirectJoinCoalesceLeftAggregate {
                    groups: state.groups.into_entries().into_iter().collect(),
                    rows: state.selected_rows,
                    batches: state.batches,
                    aggregate_nanos: 0,
                }))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut groups = JoinCoalesceGroupAccumulator::new();
    let mut rows = 0usize;
    let mut batches = 0usize;
    for chunk in chunks {
        rows = rows.saturating_add(chunk.output.rows);
        batches = batches.saturating_add(chunk.output.batches);
        for (key, (count, sum)) in chunk.output.groups {
            groups.add_counts(key, count, sum);
        }
    }
    Ok(Some(DirectJoinCoalesceLeftAggregate {
        groups: groups.into_entries().into_iter().collect(),
        rows,
        batches,
        aggregate_nanos: 0,
    }))
}

pub(super) fn join_coalesce_late_left_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_JOIN_COALESCE_LATE_LEFT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn join_coalesce_late_left_row_group_chunk() -> usize {
    std::env::var("DODAM_JOIN_COALESCE_LATE_LEFT_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

pub(super) fn join_coalesce_late_left_max_selected_ratio() -> f64 {
    std::env::var("DODAM_JOIN_COALESCE_LATE_LEFT_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.50)
}

#[derive(Clone)]
pub(super) enum DirectI64Filter {
    All,
    None,
    TinyIn(Vec<i64>),
    In(FastHashSet<i64>),
}

impl DirectI64Filter {
    fn push_selected_positions_and_values(
        &self,
        value: &DirectI64ishVector<'_>,
        positions: &mut Vec<usize>,
        selected_values: &mut Vec<i64>,
    ) {
        positions.clear();
        match self {
            Self::All => {
                reserve_selected_positions(positions, value.len());
                selected_values.reserve(value.len());
                for row in 0..value.len() {
                    if !value.is_null(row) {
                        positions.push(row);
                        selected_values.push(value.value(row));
                    }
                }
            }
            Self::None => {}
            Self::TinyIn(values) if !value.has_nulls() => {
                if let Some(raw_values) = value.i64_values_if_null_free() {
                    push_i64_tiny_in_positions_and_values(
                        raw_values,
                        values,
                        positions,
                        selected_values,
                    );
                } else if let Some(raw_values) = value.i32_values_if_null_free()
                    && let Some(values) = i64_filter_values_as_i32(values)
                {
                    push_i32_tiny_in_positions_and_values(
                        raw_values,
                        &values,
                        positions,
                        selected_values,
                    );
                }
            }
            Self::TinyIn(values) => {
                for row in 0..value.len() {
                    if !value.is_null(row) {
                        let current = value.value(row);
                        if values.iter().any(|candidate| *candidate == current) {
                            positions.push(row);
                            selected_values.push(current);
                        }
                    }
                }
            }
            Self::In(values) if !value.has_nulls() => {
                if let Some(raw_values) = value.i64_values_if_null_free() {
                    push_i64_hashset_positions_and_values(
                        raw_values,
                        values,
                        positions,
                        selected_values,
                    );
                } else if let Some(raw_values) = value.i32_values_if_null_free() {
                    push_i32_as_i64_hashset_positions_and_values(
                        raw_values,
                        values,
                        positions,
                        selected_values,
                    );
                }
            }
            Self::In(values) => {
                for row in 0..value.len() {
                    if !value.is_null(row) {
                        let current = value.value(row);
                        if values.contains(&current) {
                            positions.push(row);
                            selected_values.push(current);
                        }
                    }
                }
            }
        }
    }

    fn push_selected_positions(&self, value: &DirectI64ishVector<'_>, positions: &mut Vec<usize>) {
        positions.clear();
        match self {
            Self::All => positions.extend(0..value.len()),
            Self::None => {}
            Self::TinyIn(values) if !value.has_nulls() => {
                if let Some(raw_values) = value.i64_values_if_null_free() {
                    primitive_topk_filter_i64_positions(raw_values, values, positions);
                } else if let Some(raw_values) = value.i32_values_if_null_free()
                    && let Some(values) = i64_filter_values_as_i32(values)
                {
                    primitive_topk_filter_i32_positions(raw_values, &values, positions);
                }
            }
            Self::TinyIn(values) => {
                for row in 0..value.len() {
                    if !value.is_null(row)
                        && values
                            .iter()
                            .any(|candidate| *candidate == value.value(row))
                    {
                        positions.push(row);
                    }
                }
            }
            Self::In(values) if !value.has_nulls() => {
                if let Some(raw_values) = value.i64_values_if_null_free() {
                    push_i64_hashset_positions(raw_values, values, positions);
                } else if let Some(raw_values) = value.i32_values_if_null_free() {
                    push_i32_as_i64_hashset_positions(raw_values, values, positions);
                }
            }
            Self::In(values) => {
                for row in 0..value.len() {
                    if !value.is_null(row) && values.contains(&value.value(row)) {
                        positions.push(row);
                    }
                }
            }
        }
    }
}

pub(super) fn push_i64_hashset_positions(
    values: &[i64],
    filter_values: &FastHashSet<i64>,
    selected: &mut Vec<usize>,
) {
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        let v0 = values[row];
        let v1 = values[row + 1];
        let v2 = values[row + 2];
        let v3 = values[row + 3];
        let v4 = values[row + 4];
        let v5 = values[row + 5];
        let v6 = values[row + 6];
        let v7 = values[row + 7];
        if filter_values.contains(&v0) {
            selected.push(row);
        }
        if filter_values.contains(&v1) {
            selected.push(row + 1);
        }
        if filter_values.contains(&v2) {
            selected.push(row + 2);
        }
        if filter_values.contains(&v3) {
            selected.push(row + 3);
        }
        if filter_values.contains(&v4) {
            selected.push(row + 4);
        }
        if filter_values.contains(&v5) {
            selected.push(row + 5);
        }
        if filter_values.contains(&v6) {
            selected.push(row + 6);
        }
        if filter_values.contains(&v7) {
            selected.push(row + 7);
        }
        row += 8;
    }
    while row < values.len() {
        if filter_values.contains(&values[row]) {
            selected.push(row);
        }
        row += 1;
    }
}

pub(super) fn push_i64_hashset_positions_and_values(
    values: &[i64],
    filter_values: &FastHashSet<i64>,
    selected: &mut Vec<usize>,
    selected_values: &mut Vec<i64>,
) {
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        let v0 = values[row];
        let v1 = values[row + 1];
        let v2 = values[row + 2];
        let v3 = values[row + 3];
        let v4 = values[row + 4];
        let v5 = values[row + 5];
        let v6 = values[row + 6];
        let v7 = values[row + 7];
        if filter_values.contains(&v0) {
            selected.push(row);
            selected_values.push(v0);
        }
        if filter_values.contains(&v1) {
            selected.push(row + 1);
            selected_values.push(v1);
        }
        if filter_values.contains(&v2) {
            selected.push(row + 2);
            selected_values.push(v2);
        }
        if filter_values.contains(&v3) {
            selected.push(row + 3);
            selected_values.push(v3);
        }
        if filter_values.contains(&v4) {
            selected.push(row + 4);
            selected_values.push(v4);
        }
        if filter_values.contains(&v5) {
            selected.push(row + 5);
            selected_values.push(v5);
        }
        if filter_values.contains(&v6) {
            selected.push(row + 6);
            selected_values.push(v6);
        }
        if filter_values.contains(&v7) {
            selected.push(row + 7);
            selected_values.push(v7);
        }
        row += 8;
    }
    while row < values.len() {
        let value = values[row];
        if filter_values.contains(&value) {
            selected.push(row);
            selected_values.push(value);
        }
        row += 1;
    }
}

pub(super) fn push_i32_as_i64_hashset_positions(
    values: &[i32],
    filter_values: &FastHashSet<i64>,
    selected: &mut Vec<usize>,
) {
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        let v0 = i64::from(values[row]);
        let v1 = i64::from(values[row + 1]);
        let v2 = i64::from(values[row + 2]);
        let v3 = i64::from(values[row + 3]);
        let v4 = i64::from(values[row + 4]);
        let v5 = i64::from(values[row + 5]);
        let v6 = i64::from(values[row + 6]);
        let v7 = i64::from(values[row + 7]);
        if filter_values.contains(&v0) {
            selected.push(row);
        }
        if filter_values.contains(&v1) {
            selected.push(row + 1);
        }
        if filter_values.contains(&v2) {
            selected.push(row + 2);
        }
        if filter_values.contains(&v3) {
            selected.push(row + 3);
        }
        if filter_values.contains(&v4) {
            selected.push(row + 4);
        }
        if filter_values.contains(&v5) {
            selected.push(row + 5);
        }
        if filter_values.contains(&v6) {
            selected.push(row + 6);
        }
        if filter_values.contains(&v7) {
            selected.push(row + 7);
        }
        row += 8;
    }
    while row < values.len() {
        if filter_values.contains(&i64::from(values[row])) {
            selected.push(row);
        }
        row += 1;
    }
}

pub(super) fn push_i32_as_i64_hashset_positions_and_values(
    values: &[i32],
    filter_values: &FastHashSet<i64>,
    selected: &mut Vec<usize>,
    selected_values: &mut Vec<i64>,
) {
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        let v0 = i64::from(values[row]);
        let v1 = i64::from(values[row + 1]);
        let v2 = i64::from(values[row + 2]);
        let v3 = i64::from(values[row + 3]);
        let v4 = i64::from(values[row + 4]);
        let v5 = i64::from(values[row + 5]);
        let v6 = i64::from(values[row + 6]);
        let v7 = i64::from(values[row + 7]);
        if filter_values.contains(&v0) {
            selected.push(row);
            selected_values.push(v0);
        }
        if filter_values.contains(&v1) {
            selected.push(row + 1);
            selected_values.push(v1);
        }
        if filter_values.contains(&v2) {
            selected.push(row + 2);
            selected_values.push(v2);
        }
        if filter_values.contains(&v3) {
            selected.push(row + 3);
            selected_values.push(v3);
        }
        if filter_values.contains(&v4) {
            selected.push(row + 4);
            selected_values.push(v4);
        }
        if filter_values.contains(&v5) {
            selected.push(row + 5);
            selected_values.push(v5);
        }
        if filter_values.contains(&v6) {
            selected.push(row + 6);
            selected_values.push(v6);
        }
        if filter_values.contains(&v7) {
            selected.push(row + 7);
            selected_values.push(v7);
        }
        row += 8;
    }
    while row < values.len() {
        let value = i64::from(values[row]);
        if filter_values.contains(&value) {
            selected.push(row);
            selected_values.push(value);
        }
        row += 1;
    }
}

pub(super) fn push_i64_tiny_in_positions_and_values(
    values: &[i64],
    filter_values: &[i64],
    selected: &mut Vec<usize>,
    selected_values: &mut Vec<i64>,
) {
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        push_i64_tiny_in_position_value(values, filter_values, row, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 1, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 2, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 3, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 4, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 5, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 6, selected, selected_values);
        push_i64_tiny_in_position_value(values, filter_values, row + 7, selected, selected_values);
        row += 8;
    }
    while row < values.len() {
        push_i64_tiny_in_position_value(values, filter_values, row, selected, selected_values);
        row += 1;
    }
}

pub(super) fn push_i64_tiny_in_position_value(
    values: &[i64],
    filter_values: &[i64],
    row: usize,
    selected: &mut Vec<usize>,
    selected_values: &mut Vec<i64>,
) {
    let value = values[row];
    if filter_values.iter().any(|candidate| *candidate == value) {
        selected.push(row);
        selected_values.push(value);
    }
}

pub(super) fn push_i32_tiny_in_positions_and_values(
    values: &[i32],
    filter_values: &[i32],
    selected: &mut Vec<usize>,
    selected_values: &mut Vec<i64>,
) {
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        push_i32_tiny_in_position_value(values, filter_values, row, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 1, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 2, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 3, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 4, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 5, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 6, selected, selected_values);
        push_i32_tiny_in_position_value(values, filter_values, row + 7, selected, selected_values);
        row += 8;
    }
    while row < values.len() {
        push_i32_tiny_in_position_value(values, filter_values, row, selected, selected_values);
        row += 1;
    }
}

pub(super) fn push_i32_tiny_in_position_value(
    values: &[i32],
    filter_values: &[i32],
    row: usize,
    selected: &mut Vec<usize>,
    selected_values: &mut Vec<i64>,
) {
    let value = values[row];
    if filter_values.iter().any(|candidate| *candidate == value) {
        selected.push(row);
        selected_values.push(i64::from(value));
    }
}

pub(super) enum DirectI64ishVector<'a> {
    I64(I64VectorView<'a>),
    I32(I32VectorView<'a>),
}

impl<'a> DirectI64ishVector<'a> {
    fn from_batch(batch: BatchView<'a>, index: usize) -> Option<Self> {
        batch
            .i64_vector(index)
            .map(Self::I64)
            .or_else(|| batch.i32_vector(index).map(Self::I32))
    }

    fn len(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }

    fn has_nulls(&self) -> bool {
        match self {
            Self::I64(values) => values.values_if_null_free().is_none(),
            Self::I32(values) => values.values_if_null_free().is_none(),
        }
    }

    fn is_null(&self, row: usize) -> bool {
        match self {
            Self::I64(values) => values.is_null(row),
            Self::I32(values) => values.is_null(row),
        }
    }

    fn value(&self, row: usize) -> i64 {
        match self {
            Self::I64(values) => values.value(row),
            Self::I32(values) => i64::from(values.value(row)),
        }
    }

    fn i64_values_if_null_free(&self) -> Option<&'a [i64]> {
        match self {
            Self::I64(values) => values.values_if_null_free(),
            Self::I32(_) => None,
        }
    }

    fn i32_values_if_null_free(&self) -> Option<&'a [i32]> {
        match self {
            Self::I64(_) => None,
            Self::I32(values) => values.values_if_null_free(),
        }
    }
}

pub(super) fn i64_filter_values_as_i32(values: &[i64]) -> Option<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()
}

pub(super) fn try_direct_join_coalesce_left_aggregate(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    filter: Option<&FilterExpr>,
    key_column: &str,
    group_column: &str,
    sum_column: &str,
    right_lookup: &AdaptiveI64Map<usize>,
) -> Result<Option<DirectJoinCoalesceLeftAggregate>> {
    if !direct_join_coalesce_left_aggregate_enabled() {
        return Ok(None);
    }
    let Some(filter) = direct_join_left_i64_filter(filter, group_column)? else {
        return Ok(None);
    };
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let candidates = [
        [
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I64,
        ],
        [
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I64,
        ],
        [
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I64,
        ],
        [
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I64,
        ],
        [
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I32,
        ],
        [
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I32,
        ],
        [
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I64,
            DirectPrimitiveColumnType::I32,
        ],
        [
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I32,
            DirectPrimitiveColumnType::I32,
        ],
    ];
    for [key_type, group_type, sum_type] in candidates {
        let specs = [
            DirectPrimitiveColumnSpec {
                name: key_column,
                column_type: key_type,
            },
            DirectPrimitiveColumnSpec {
                name: group_column,
                column_type: group_type,
            },
            DirectPrimitiveColumnSpec {
                name: sum_column,
                column_type: sum_type,
            },
        ];
        let mut groups = JoinCoalesceGroupAccumulator::new();
        let mut selected_positions = Vec::<usize>::with_capacity(batch_size);
        let mut rows = 0usize;
        let scanned = engine.scan_parquet_primitive_columns_view(
            path,
            batch_size,
            &row_groups,
            &specs,
            |batch| {
                let Some(key) = DirectI64ishVector::from_batch(batch, 0) else {
                    return Err(DodamError::UnsupportedSql(
                        "direct join fusion key column is not i64-like".to_string(),
                    ));
                };
                let Some(group) = DirectI64ishVector::from_batch(batch, 1) else {
                    return Err(DodamError::UnsupportedSql(
                        "direct join fusion group column is not i64-like".to_string(),
                    ));
                };
                let Some(sum) = DirectI64ishVector::from_batch(batch, 2) else {
                    return Err(DodamError::UnsupportedSql(
                        "direct join fusion sum column is not i64-like".to_string(),
                    ));
                };
                let dense_lookup = right_lookup.dense_slices();
                match filter {
                    DirectI64Filter::All => {
                        rows = rows.saturating_add(key.len());
                        accumulate_direct_join_coalesce_rows(
                            &mut groups,
                            &key,
                            &group,
                            &sum,
                            right_lookup,
                            dense_lookup,
                            0..key.len(),
                        );
                    }
                    DirectI64Filter::None => {}
                    _ => {
                        filter.push_selected_positions(&group, &mut selected_positions);
                        rows = rows.saturating_add(selected_positions.len());
                        accumulate_direct_join_coalesce_rows(
                            &mut groups,
                            &key,
                            &group,
                            &sum,
                            right_lookup,
                            dense_lookup,
                            selected_positions.iter().copied(),
                        );
                    }
                }
                Ok(())
            },
        )?;
        if let Some(metrics) = scanned {
            trace_join_coalesce_fusion(
                "accept-direct-left",
                &format!(
                    "types={key_type:?}/{group_type:?}/{sum_type:?} rows={} batches={}",
                    rows, metrics.batches
                ),
            );
            return Ok(Some(DirectJoinCoalesceLeftAggregate {
                groups: groups.into_entries(),
                rows,
                batches: metrics.batches,
                aggregate_nanos: metrics.consume_nanos,
            }));
        }
    }
    Ok(None)
}

pub(super) fn accumulate_direct_join_coalesce_rows(
    groups: &mut JoinCoalesceGroupAccumulator,
    key: &DirectI64ishVector<'_>,
    group: &DirectI64ishVector<'_>,
    sum: &DirectI64ishVector<'_>,
    right_lookup: &AdaptiveI64Map<usize>,
    dense_lookup: Option<(&[usize], &[bool])>,
    rows: impl IntoIterator<Item = usize>,
) {
    if !key.has_nulls() && !group.has_nulls() && !sum.has_nulls() {
        for row in rows {
            if let Some(class_id) = right_lookup.get_cached(dense_lookup, key.value(row)) {
                groups.update_non_null(group.value(row), class_id, sum.value(row));
            }
        }
        return;
    }
    for row in rows {
        if key.is_null(row) {
            continue;
        }
        let Some(class_id) = right_lookup.get_cached(dense_lookup, key.value(row)) else {
            continue;
        };
        groups.update(
            (!group.is_null(row)).then(|| group.value(row)),
            class_id,
            (!sum.is_null(row)).then(|| sum.value(row)),
        );
    }
}

pub(super) fn direct_join_coalesce_left_aggregate_enabled() -> bool {
    std::env::var("DODAM_ENABLE_DIRECT_JOIN_COALESCE_LEFT_AGGREGATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn direct_join_left_i64_filter(
    filter: Option<&FilterExpr>,
    group_column: &str,
) -> Result<Option<DirectI64Filter>> {
    let Some(filter) = filter else {
        return Ok(Some(DirectI64Filter::All));
    };
    match filter.expr() {
        Expr::Boolean(Some(true)) => Ok(Some(DirectI64Filter::All)),
        Expr::Boolean(Some(false) | None) => Ok(Some(DirectI64Filter::None)),
        Expr::InList {
            column,
            values,
            negated: false,
            has_null: false,
        } if column == group_column => {
            let mut parsed_values = Vec::with_capacity(values.len());
            for value in values {
                parsed_values.push(value.as_i64(column)?);
            }
            parsed_values.sort_unstable();
            parsed_values.dedup();
            if parsed_values.len() <= 8 {
                return Ok(Some(DirectI64Filter::TinyIn(parsed_values)));
            }
            Ok(Some(DirectI64Filter::In(
                parsed_values.into_iter().collect(),
            )))
        }
        _ => Ok(None),
    }
}

pub(super) struct UniqueI64ToUtf8IdLookup {
    pub(super) lookup: AdaptiveI64Map<usize>,
    pub(super) values: Vec<Option<String>>,
}

pub(super) fn build_unique_i64_to_utf8_id_lookup(
    batches: &[RecordBatch],
    key_column: &str,
    value_column: &str,
) -> Result<Option<UniqueI64ToUtf8IdLookup>> {
    let mut lookup = FastHashMap::default();
    let mut value_ids = FastHashMap::<Option<String>, usize>::default();
    let mut values = Vec::<Option<String>>::new();
    for batch in batches {
        let key = i64_array_like(batch, key_column)?;
        let value = string_array(batch, value_column)?;
        for row in 0..batch.num_rows() {
            if key.is_null(row) {
                continue;
            }
            let payload = (!value.is_null(row)).then(|| value.value(row).to_string());
            let value_id = if let Some(value_id) = value_ids.get(&payload).copied() {
                value_id
            } else {
                let value_id = values.len();
                values.push(payload.clone());
                value_ids.insert(payload, value_id);
                value_id
            };
            let key_value = key.value(row);
            if lookup.contains_key(&key_value) {
                return Ok(None);
            }
            lookup.insert(key_value, value_id);
        }
    }
    let lookup = AdaptiveI64Map::from_hash(lookup);
    Ok(Some(UniqueI64ToUtf8IdLookup { lookup, values }))
}

pub(super) fn i64_array_like<'a>(
    batch: &'a RecordBatch,
    column: &str,
) -> Result<I64LikeColumn<'a>> {
    let index = batch_column_index(batch, column)?;
    let array = batch.column(index);
    match array.data_type() {
        DataType::Int32 => Ok(I64LikeColumn::Int32(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type"),
        )),
        DataType::Int64 => Ok(I64LikeColumn::Int64(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type"),
        )),
        other => Err(DodamError::UnsupportedSql(format!(
            "join aggregate fusion requires integer column {column}, got {other:?}"
        ))),
    }
}

pub(super) enum I64LikeColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
}

impl I64LikeColumn<'_> {
    pub(super) fn has_nulls(&self) -> bool {
        match self {
            Self::Int32(values) => values.null_count() > 0,
            Self::Int64(values) => values.null_count() > 0,
        }
    }

    pub(super) fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Int32(values) => values.is_null(row),
            Self::Int64(values) => values.is_null(row),
        }
    }

    pub(super) fn value(&self, row: usize) -> i64 {
        match self {
            Self::Int32(values) => i64::from(values.value(row)),
            Self::Int64(values) => values.value(row),
        }
    }

    pub(super) fn raw_values(&self) -> I64LikeValues<'_> {
        match self {
            Self::Int32(values) => I64LikeValues::I32(values.values().as_ref()),
            Self::Int64(values) => I64LikeValues::I64(values.values().as_ref()),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum I64LikeValues<'a> {
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl I64LikeValues<'_> {
    #[inline]
    pub(super) fn value(self, row: usize) -> i64 {
        match self {
            Self::I32(values) => i64::from(values[row]),
            Self::I64(values) => values[row],
        }
    }
}

pub(super) fn string_array<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a StringArray> {
    let index = batch_column_index(batch, column)?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DodamError::UnsupportedSql(format!("{column} must be Utf8")))
}

pub(super) fn unique_columns(columns: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut output = Vec::new();
    for column in columns {
        add_column_once(&mut output, column);
    }
    output
}

pub(super) fn join_column_belongs_to(column: &str, alias: &str) -> bool {
    column
        .strip_prefix(alias)
        .and_then(|rest| rest.strip_prefix('.'))
        .is_some()
}

pub(super) fn compare_join_fused_group_keys(
    left: &[GroupValue],
    right: &[GroupValue],
) -> std::cmp::Ordering {
    for (left, right) in left.iter().zip(right.iter()) {
        let ordering = match (left, right) {
            (GroupValue::Int64(left), GroupValue::Int64(right)) => left.cmp(right),
            (GroupValue::Utf8(left), GroupValue::Utf8(right)) => left.cmp(right),
            _ => std::cmp::Ordering::Equal,
        };
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

pub(super) fn same_join_column(left: &str, right: &str) -> bool {
    if let (Some((left_function, left_argument)), Some((right_function, right_argument))) =
        (aggregate_column_parts(left), aggregate_column_parts(right))
        && left_function.eq_ignore_ascii_case(right_function)
        && same_join_column(left_argument, right_argument)
    {
        return true;
    }
    left == right
        || left.rsplit('.').next() == Some(right)
        || right.rsplit('.').next() == Some(left)
}
