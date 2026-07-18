use super::*;

pub(super) async fn shipping_mode_counts_from_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Vec<ShippingPriorityRow>> {
    if order_direct_column_reader_enabled()
        && let Some(rows) = shipping_mode_counts_from_orders_direct(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            pending,
        )?
    {
        return Ok(rows);
    }
    if order_late_materialized_enabled()
        && let Some(rows) = shipping_mode_counts_from_orders_late(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            pending,
        )
        .await?
    {
        return Ok(rows);
    }
    if order_fused_scan_enabled()
        && !order_row_filter_enabled()
        && !order_bloom_filter_enabled()
        && let Some(rows) = shipping_mode_counts_from_orders_fused(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            pending,
        )
        .await?
    {
        return Ok(rows);
    }
    if order_sorted_pending_lookup_enabled() {
        return shipping_mode_counts_from_orders_sorted(
            engine, path, batch_size, shipmodes, pending,
        )
        .await;
    }
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let mut stream = if order_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "o_orderkey",
                pending.keys().copied().collect(),
            )
            .await?
    } else if order_bloom_filter_enabled() {
        engine
            .scan_parquet_batches_i64_bloom_filtered(
                path,
                batch_size,
                projection,
                "o_orderkey",
                pending.keys().copied().collect(),
            )
            .await?
    } else if order_dictionary_priority_enabled() {
        engine
            .scan_parquet_batches_dictionary_columns(
                path,
                batch_size,
                projection,
                vec!["o_orderpriority".to_string()],
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let collect_profile = tpch_profile_enabled();
    let partial = parallel_batch_fold_view_chunks(
        &mut stream,
        order_chunk_size(),
        ShippingOrdersPartial::default,
        move |view, partial| {
            merge_orders_partial(
                partial,
                shipping_mode_counts_projected_view_partial(view, &pending, collect_profile)?,
            );
            Ok(Some(()))
        },
        Ok,
        ShippingOrdersPartial::default(),
        merge_orders_partial,
        "shipping priority counts orders aggregate",
    )?;
    log_orders_profile(&partial.profile);
    let groups = partial.groups;
    let rows = groups
        .into_iter()
        .enumerate()
        .map(|(index, state)| ShippingPriorityRow {
            shipmode: shipmodes[index].clone(),
            high_line_count: state.high_line_count,
            low_line_count: state.low_line_count,
        })
        .collect::<Vec<_>>();
    Ok(rows)
}

pub(super) async fn shipping_mode_counts_from_orders_sorted(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Vec<ShippingPriorityRow>> {
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    let pending = Arc::new(SortedI64Lookup::from_hash_map(pending));
    let groups = parallel_batch_fold_view_chunks(
        &mut stream,
        order_chunk_size(),
        || [ShippingPriorityState::default(); 2],
        move |view, groups| {
            merge_shipping_mode_counts(
                groups,
                shipping_mode_counts_projected_view_sorted(view, &pending)?,
            );
            Ok(Some(()))
        },
        Ok,
        [ShippingPriorityState::default(); 2],
        merge_shipping_mode_counts,
        "shipping priority counts orders sorted aggregate",
    )?;
    Ok(groups
        .into_iter()
        .enumerate()
        .map(|(index, state)| ShippingPriorityRow {
            shipmode: shipmodes[index].clone(),
            high_line_count: state.high_line_count,
            low_line_count: state.low_line_count,
        })
        .collect())
}

pub(super) async fn shipping_mode_counts_from_orders_fused(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Option<Vec<ShippingPriorityRow>>> {
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let build_state = || [ShippingPriorityState::default(); 2];
    let finish = |groups| Ok(Some(groups));
    let partials = if order_fused_dictionary_priority_enabled() {
        engine
            .parquet_row_group_map_dictionary_columns_pruned_view(
                path,
                batch_size,
                projection,
                vec!["o_orderpriority".to_string()],
                Vec::new(),
                order_fused_row_group_chunk(),
                build_state,
                {
                    let pending = pending.clone();
                    move |view, groups: &mut [ShippingPriorityState; 2]| {
                        merge_shipping_mode_counts(
                            groups,
                            shipping_mode_counts_projected_view(view, &pending)?,
                        );
                        Ok(Some(()))
                    }
                },
                finish,
            )
            .await?
    } else {
        engine
            .parquet_row_group_map_view(
                path,
                batch_size,
                projection,
                order_fused_row_group_chunk(),
                build_state,
                {
                    let pending = pending.clone();
                    move |view, groups: &mut [ShippingPriorityState; 2]| {
                        merge_shipping_mode_counts(
                            groups,
                            shipping_mode_counts_projected_view(view, &pending)?,
                        );
                        Ok(Some(()))
                    }
                },
                finish,
            )
            .await?
    };
    let Some(partials) = partials else {
        return Ok(None);
    };
    let mut groups = [ShippingPriorityState::default(); 2];
    for partial in partials {
        merge_shipping_mode_counts(&mut groups, partial);
    }
    Ok(Some(
        groups
            .into_iter()
            .enumerate()
            .map(|(index, state)| ShippingPriorityRow {
                shipmode: shipmodes[index].clone(),
                high_line_count: state.high_line_count,
                low_line_count: state.low_line_count,
            })
            .collect(),
    ))
}

pub(super) fn shipping_mode_counts_from_orders_direct(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Option<Vec<ShippingPriorityRow>>> {
    let started = tpch_profile_start();
    let row_groups = (0..engine.parquet_row_group_count(&path)?).collect::<Vec<_>>();
    let pending = Arc::new(AdaptiveI64Map::from_hash(pending.clone()));
    let chunks = row_groups
        .chunks(order_direct_row_group_chunk())
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    let profile = tpch_profile_enabled();
    let partials = chunks
        .into_par_iter()
        .map(|row_groups| {
            order_direct_row_group_chunk_scan(
                engine,
                path.clone(),
                batch_size,
                row_groups,
                pending.clone(),
                profile,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut metrics = OrderDirectMetrics::default();
    for partial in partials {
        let Some(partial) = partial else {
            return Ok(None);
        };
        merge_shipping_mode_counts(&mut groups, partial.groups);
        metrics.add(partial.metrics);
    }
    if let Some(started) = started {
        eprintln!(
            "[dodam:tpch-profile] shipping priority counts orders direct_column_reader: total={:.3} ms row_groups={} batches={} rows={} hits={} misses={} payload_runs={} selective_batches={} full_batches={} read={:.3} ms consume={:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0,
            metrics.row_groups,
            metrics.batches,
            metrics.rows,
            metrics.hits,
            metrics.misses,
            metrics.payload_runs,
            metrics.selective_batches,
            metrics.full_batches,
            sql_nanos_to_millis(metrics.read_nanos),
            sql_nanos_to_millis(metrics.consume_nanos),
        );
    }
    Ok(Some(
        groups
            .into_iter()
            .enumerate()
            .map(|(index, state)| ShippingPriorityRow {
                shipmode: shipmodes[index].clone(),
                high_line_count: state.high_line_count,
                low_line_count: state.low_line_count,
            })
            .collect(),
    ))
}

pub(super) fn order_direct_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_ORDER_DIRECT_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

#[derive(Default)]
pub(super) struct OrderDirectPartial {
    groups: [ShippingPriorityState; 2],
    metrics: OrderDirectMetrics,
}

#[derive(Default)]
pub(super) struct OrderDirectMetrics {
    row_groups: usize,
    batches: usize,
    rows: usize,
    hits: usize,
    misses: usize,
    payload_runs: usize,
    selective_batches: usize,
    full_batches: usize,
    read_nanos: u64,
    consume_nanos: u64,
}

impl OrderDirectMetrics {
    fn add_scan_metrics(&mut self, metrics: DirectColumnScanMetrics) {
        self.row_groups = self.row_groups.saturating_add(metrics.row_groups);
        self.batches = self.batches.saturating_add(metrics.batches);
        self.rows = self.rows.saturating_add(metrics.rows);
        self.read_nanos = self.read_nanos.saturating_add(metrics.read_nanos);
        self.consume_nanos = self.consume_nanos.saturating_add(metrics.consume_nanos);
    }

    fn add(&mut self, other: Self) {
        self.row_groups = self.row_groups.saturating_add(other.row_groups);
        self.batches = self.batches.saturating_add(other.batches);
        self.rows = self.rows.saturating_add(other.rows);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.payload_runs = self.payload_runs.saturating_add(other.payload_runs);
        self.selective_batches = self
            .selective_batches
            .saturating_add(other.selective_batches);
        self.full_batches = self.full_batches.saturating_add(other.full_batches);
        self.read_nanos = self.read_nanos.saturating_add(other.read_nanos);
        self.consume_nanos = self.consume_nanos.saturating_add(other.consume_nanos);
    }
}

pub(super) fn order_direct_row_group_chunk_scan(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    pending: Arc<AdaptiveI64Map<PendingShippingOrder>>,
    profile: bool,
) -> Result<Option<OrderDirectPartial>> {
    let started = profile.then(Instant::now);
    let mut partial = OrderDirectPartial::default();
    let mut priorities = Vec::<parquet::data_type::ByteArray>::with_capacity(batch_size);
    let mut priority_def_levels = Vec::<i16>::with_capacity(batch_size);
    let mut hits = Vec::<(usize, PendingShippingOrder)>::with_capacity(batch_size.min(1024));
    let Some(scan_metrics) = engine.scan_parquet_i64_byte_array_payload_columns(
        &path,
        batch_size,
        &row_groups,
        ["o_orderkey", "o_orderpriority"],
        |orderkeys, priority_reader| {
            priorities.clear();
            priority_def_levels.clear();
            hits.clear();
            let order_records = orderkeys.len();
            for row in 0..orderkeys.len() {
                let Some(order) = pending.get(orderkeys[row]) else {
                    partial.metrics.misses += 1;
                    continue;
                };
                partial.metrics.hits += 1;
                hits.push((row, order));
            }
            if hits.is_empty() {
                let skipped = priority_reader.skip_records(order_records)?;
                if skipped != order_records {
                    return Ok(None);
                }
                return Ok(Some(()));
            }
            let payload_runs = order_direct_payload_runs(&hits);
            partial.metrics.payload_runs =
                partial.metrics.payload_runs.saturating_add(payload_runs);
            if payload_runs > order_direct_max_selective_runs_per_batch() {
                partial.metrics.full_batches = partial.metrics.full_batches.saturating_add(1);
                priorities.clear();
                priority_def_levels.clear();
                let (priority_records, priority_values, priority_levels) = priority_reader
                    .read_records(order_records, &mut priority_def_levels, &mut priorities)?;
                if priority_records != order_records || priority_levels != order_records {
                    return Ok(None);
                }
                if priority_values == order_records {
                    for &(row, order) in &hits {
                        let is_high_priority = is_high_priority_bytes(priorities[row].data());
                        apply_pending_order(&mut partial.groups, order, is_high_priority);
                    }
                } else {
                    let mut value_row = vec![None; order_records];
                    let mut value_index = 0usize;
                    for row in 0..order_records {
                        let Some(level) = priority_def_levels.get(row) else {
                            return Ok(None);
                        };
                        if *level == 0 {
                            continue;
                        }
                        value_row[row] = Some(value_index);
                        value_index += 1;
                    }
                    if value_index != priority_values {
                        return Ok(None);
                    }
                    for &(row, order) in &hits {
                        let Some(value_index) = value_row[row] else {
                            continue;
                        };
                        let Some(priority) = priorities.get(value_index) else {
                            return Ok(None);
                        };
                        let is_high_priority = is_high_priority_bytes(priority.data());
                        apply_pending_order(&mut partial.groups, order, is_high_priority);
                    }
                }
                return Ok(Some(()));
            }
            partial.metrics.selective_batches = partial.metrics.selective_batches.saturating_add(1);
            let mut cursor = 0usize;
            let mut hit_index = 0usize;
            while hit_index < hits.len() {
                let start = hits[hit_index].0;
                let mut end_index = hit_index + 1;
                while end_index < hits.len() && hits[end_index].0 == hits[end_index - 1].0 + 1 {
                    end_index += 1;
                }
                let run_len = end_index - hit_index;
                if start > cursor {
                    let skipped = priority_reader.skip_records(start - cursor)?;
                    if skipped != start - cursor {
                        return Ok(None);
                    }
                }
                priorities.clear();
                priority_def_levels.clear();
                let (priority_records, priority_values, priority_levels) = priority_reader
                    .read_records(run_len, &mut priority_def_levels, &mut priorities)?;
                if priority_records != run_len || priority_levels != run_len {
                    return Ok(None);
                }
                if priority_values == run_len {
                    for (offset, hit) in hits[hit_index..end_index].iter().enumerate() {
                        let is_high_priority = is_high_priority_bytes(priorities[offset].data());
                        apply_pending_order(&mut partial.groups, hit.1, is_high_priority);
                    }
                } else {
                    let mut value_index = 0usize;
                    for hit in &hits[hit_index..end_index] {
                        let Some(level) = priority_def_levels.get(hit.0 - start) else {
                            return Ok(None);
                        };
                        if *level == 0 {
                            continue;
                        }
                        let Some(priority) = priorities.get(value_index) else {
                            return Ok(None);
                        };
                        value_index += 1;
                        let is_high_priority = is_high_priority_bytes(priority.data());
                        apply_pending_order(&mut partial.groups, hit.1, is_high_priority);
                    }
                    if value_index != priority_values {
                        return Ok(None);
                    }
                }
                cursor = start + run_len;
                hit_index = end_index;
            }
            if cursor < order_records {
                let skipped = priority_reader.skip_records(order_records - cursor)?;
                if skipped != order_records - cursor {
                    return Ok(None);
                }
            }
            Ok(Some(()))
        },
    )?
    else {
        return Ok(None);
    };
    partial.metrics.add_scan_metrics(scan_metrics);
    if let Some(started) = started {
        eprintln!(
            "[dodam:tpch-profile] shipping priority counts orders direct_column_chunk: row_groups={} rows={} hits={} misses={} payload_runs={} selective_batches={} full_batches={} elapsed={:.3} ms read={:.3} ms consume={:.3} ms",
            partial.metrics.row_groups,
            partial.metrics.rows,
            partial.metrics.hits,
            partial.metrics.misses,
            partial.metrics.payload_runs,
            partial.metrics.selective_batches,
            partial.metrics.full_batches,
            started.elapsed().as_secs_f64() * 1000.0,
            sql_nanos_to_millis(partial.metrics.read_nanos),
            sql_nanos_to_millis(partial.metrics.consume_nanos),
        );
    }
    Ok(Some(partial))
}

pub(super) fn order_direct_payload_runs(hits: &[(usize, PendingShippingOrder)]) -> usize {
    if hits.is_empty() {
        return 0;
    }
    let mut runs = 1usize;
    for index in 1..hits.len() {
        if hits[index].0 != hits[index - 1].0 + 1 {
            runs += 1;
        }
    }
    runs
}

pub(super) fn order_direct_max_selective_runs_per_batch() -> usize {
    std::env::var("DODAM_Q12_ORDER_DIRECT_MAX_SELECTIVE_RUNS_PER_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64)
}

pub(super) async fn shipping_mode_counts_from_orders_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Option<Vec<ShippingPriorityRow>>> {
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["o_orderkey".to_string()]),
            Projection::Columns(vec!["o_orderpriority".to_string()]),
            Vec::new(),
            order_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::always(),
            {
                let pending = pending.clone();
                move || OrderLateState {
                    pending: pending.clone(),
                    selected_orders: Vec::new(),
                    selected_offset: 0,
                    groups: [ShippingPriorityState::default(); 2],
                }
            },
            order_late_build_selection_view,
            order_late_consume_priority_view,
            |state, _metrics| {
                if state.selected_offset != state.selected_orders.len() {
                    return Err(DodamError::UnsupportedSql(
                        "shipping priority counts order priority payload mismatch".to_string(),
                    ));
                }
                Ok(Some(state.groups))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        merge_shipping_mode_counts(&mut groups, chunk.output);
        metrics.add(chunk.metrics);
    }
    log_order_late_materialized_profile(metrics, order_late_materialized_row_group_chunk());
    Ok(Some(
        groups
            .into_iter()
            .enumerate()
            .map(|(index, state)| ShippingPriorityRow {
                shipmode: shipmodes[index].clone(),
                high_line_count: state.high_line_count,
                low_line_count: state.low_line_count,
            })
            .collect(),
    ))
}

pub(super) fn order_bloom_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_BLOOM_FILTER").is_some()
}

pub(super) fn order_fused_scan_enabled() -> bool {
    if std::env::var_os("DODAM_Q12_ENABLE_ORDER_FUSED_SCAN").is_some() {
        return true;
    }
    std::env::var_os("DODAM_Q12_DISABLE_ORDER_FUSED_SCAN").is_none()
}

pub(super) fn order_dictionary_priority_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_DICTIONARY_PRIORITY").is_some()
}

pub(super) fn order_fused_dictionary_priority_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_ORDER_FUSED_DICTIONARY_PRIORITY").is_none()
}

pub(super) fn order_sorted_pending_lookup_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_SORTED_PENDING_LOOKUP").is_none()
}

pub(super) fn order_direct_column_reader_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_DIRECT_COLUMN_READER").is_some()
}

pub(super) fn order_chunk_size() -> usize {
    std::env::var("DODAM_Q12_ORDER_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

pub(super) fn order_fused_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_ORDER_FUSED_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

pub(super) fn order_late_materialized_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_LATE_MATERIALIZE").is_some()
}

pub(super) fn order_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_ORDER_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn order_row_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_ROW_FILTER").is_some()
}

pub(super) struct OrderLateState {
    pending: Arc<AdaptiveI64Map<PendingShippingOrder>>,
    selected_orders: Vec<PendingShippingOrder>,
    selected_offset: usize,
    groups: [ShippingPriorityState; 2],
}

pub(super) struct OrderPriorityView<'a> {
    priorities: Utf8VectorView<'a>,
}

impl<'a> OrderPriorityView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        (view.num_columns() == 1).then_some(Self {
            priorities: view.utf8_vector(0)?,
        })
    }
}

pub(super) fn order_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    let Some(orderkeys) = batch_column(&batch, "o_orderkey")?
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 {
        let orderkey_values = orderkeys.values();
        let pending_dense = state.pending.dense_slices();
        selection.push_selected_offsets(
            orderkey_values.len(),
            (0..orderkey_values.len()).filter_map(|row| {
                let orderkey = orderkey_values[row];
                if let Some(order) = state.pending.get_cached(pending_dense, orderkey) {
                    state.selected_orders.push(order);
                    Some(row)
                } else {
                    None
                }
            }),
        );
        return Ok(Some(()));
    }
    selection.push_selected_offsets(
        orderkeys.len(),
        (0..orderkeys.len()).filter_map(|row| {
            if orderkeys.is_null(row) {
                return None;
            }
            state.pending.get(orderkeys.value(row)).map(|order| {
                state.selected_orders.push(order);
                row
            })
        }),
    );
    Ok(Some(()))
}

pub(super) fn order_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(orderkeys) = view.i64_vector(0) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return order_late_build_selection_batch(batch.clone(), selection, state);
        };
        if let Some(orderkey_values) = orderkeys.values_if_null_free() {
            let pending_dense = state.pending.dense_slices();
            selection.push_selected_offsets(
                orderkey_values.len(),
                (0..orderkey_values.len()).filter_map(|row| {
                    let orderkey = orderkey_values[row];
                    if let Some(order) = state.pending.get_cached(pending_dense, orderkey) {
                        state.selected_orders.push(order);
                        Some(row)
                    } else {
                        None
                    }
                }),
            );
            return Ok(Some(()));
        }
        selection.push_selected_offsets(
            orderkeys.len(),
            (0..orderkeys.len()).filter_map(|row| {
                if orderkeys.is_null(row) {
                    return None;
                }
                state.pending.get(orderkeys.value(row)).map(|order| {
                    state.selected_orders.push(order);
                    row
                })
            }),
        );
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    order_late_build_selection_batch(batch.clone(), selection, state)
}

pub(super) fn order_late_consume_priority_batch(
    batch: RecordBatch,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    let priorities = batch_string_column(&batch, "o_orderpriority")?;
    let priority_offsets = priorities.value_offsets();
    let priority_data = priorities.value_data();
    for row in 0..batch.num_rows() {
        let Some(&order) = state.selected_orders.get(state.selected_offset) else {
            return Err(DodamError::UnsupportedSql(
                "shipping priority counts order priority payload overflow".to_string(),
            ));
        };
        state.selected_offset += 1;
        if priorities.is_null(row) {
            continue;
        }
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = is_high_priority_bytes(priority);
        apply_pending_order(&mut state.groups, order, is_high_priority);
    }
    Ok(Some(()))
}

pub(super) fn order_late_consume_priority_view(
    view: BatchView<'_>,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    if let Some(layout) = OrderPriorityView::try_new(view) {
        let priorities = layout.priorities;
        for row in 0..view.num_rows() {
            let Some(&order) = state.selected_orders.get(state.selected_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "shipping priority counts order priority payload overflow".to_string(),
                ));
            };
            state.selected_offset += 1;
            if priorities.is_null(row) {
                continue;
            }
            let is_high_priority = is_high_priority_bytes(priorities.value_bytes(row));
            apply_pending_order(&mut state.groups, order, is_high_priority);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    order_late_consume_priority_batch(batch.clone(), state)
}

pub(super) fn is_high_priority_str(priority: &str) -> bool {
    is_high_priority_bytes(priority.as_bytes())
}

pub(super) fn is_high_priority_bytes(priority: &[u8]) -> bool {
    matches!(priority.first(), Some(b'1' | b'2')) && matches!(priority, b"1-URGENT" | b"2-HIGH")
}

pub(super) fn log_order_late_materialized_profile(
    metrics: LateMaterializedMetrics,
    row_group_chunk: usize,
) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] shipping priority counts orders: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

pub(super) fn shipping_mode_counts_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
    if shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_batch_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let mut groups = [ShippingPriorityState::default(); 2];
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
        let is_high_priority = is_high_priority_str(orderpriorities.value(row));
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

#[allow(dead_code)]
pub(super) fn shipping_mode_counts_projected_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_batch_typed(batch.column(0), orderpriorities, pending)
    {
        return Ok(groups);
    }
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch
            .column(1)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
        && shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_batch_dictionary_typed(
            batch.column(0),
            DictionaryI32View::Arrow(orderpriorities),
            pending,
        )
    {
        return Ok(groups);
    }
    shipping_mode_counts_batch(batch, pending)
}

pub(super) fn shipping_mode_counts_projected_view(
    view: BatchView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) =
            (view.i64_vector(0), view.dictionary_i32_view(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_dictionary_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "shipping priority counts shipping mode raw vector columns have unsupported types"
                .to_string(),
        ));
    };
    shipping_mode_counts_batch(batch.clone(), pending)
}

pub(super) fn shipping_mode_counts_projected_batch_sorted(
    batch: RecordBatch,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_batch_typed_sorted(batch.column(0), orderpriorities, pending)
    {
        return Ok(groups);
    }
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
    let mut groups = [ShippingPriorityState::default(); 2];
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
        let is_high_priority = is_high_priority_str(orderpriorities.value(row));
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Ok(groups)
}

pub(super) fn shipping_mode_counts_projected_view_sorted(
    view: BatchView<'_>,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_vector_typed_sorted(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "shipping priority counts sorted shipping mode raw vector columns have unsupported types".to_string(),
        ));
    };
    shipping_mode_counts_projected_batch_sorted(batch.clone(), pending)
}

#[allow(dead_code)]
pub(super) fn shipping_mode_counts_projected_batch_partial(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
    collect_profile: bool,
) -> Result<ShippingOrdersPartial> {
    if !collect_profile {
        return Ok(ShippingOrdersPartial {
            groups: shipping_mode_counts_projected_batch(batch, pending)?,
            profile: ShippingOrdersProfile::default(),
        });
    }
    let started = Instant::now();
    let rows = batch.num_rows();
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && shipping_typed_loop_enabled()
        && let Some(mut partial) =
            shipping_mode_counts_batch_typed_profile(batch.column(0), orderpriorities, pending)
    {
        partial.profile.total_nanos = sql_elapsed_nanos(started);
        return Ok(partial);
    }
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch
            .column(1)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
        && shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_batch_dictionary_typed(
            batch.column(0),
            DictionaryI32View::Arrow(orderpriorities),
            pending,
        )
    {
        return Ok(ShippingOrdersPartial {
            groups,
            profile: ShippingOrdersProfile {
                batches: 1,
                typed_batches: 1,
                rows,
                total_nanos: sql_elapsed_nanos(started),
                ..Default::default()
            },
        });
    }
    let groups = shipping_mode_counts_batch(batch, pending)?;
    let profile = ShippingOrdersProfile {
        batches: 1,
        fallback_batches: 1,
        rows,
        total_nanos: sql_elapsed_nanos(started),
        ..Default::default()
    };
    Ok(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_mode_counts_projected_view_partial(
    view: BatchView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
    collect_profile: bool,
) -> Result<ShippingOrdersPartial> {
    if !collect_profile {
        return Ok(ShippingOrdersPartial {
            groups: shipping_mode_counts_projected_view(view, pending)?,
            profile: ShippingOrdersProfile::default(),
        });
    }
    let started = Instant::now();
    let rows = view.num_rows();
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && shipping_typed_loop_enabled()
        && let Some(mut partial) =
            shipping_mode_counts_vector_typed_profile(orderkeys, orderpriorities, pending)
    {
        partial.profile.total_nanos = sql_elapsed_nanos(started);
        return Ok(partial);
    }
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) =
            (view.i64_vector(0), view.dictionary_i32_view(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_dictionary_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(ShippingOrdersPartial {
            groups,
            profile: ShippingOrdersProfile {
                batches: 1,
                typed_batches: 1,
                rows,
                total_nanos: sql_elapsed_nanos(started),
                ..Default::default()
            },
        });
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "shipping priority counts profiled shipping mode raw vector columns have unsupported types".to_string(),
        ));
    };
    let groups = shipping_mode_counts_batch(batch.clone(), pending)?;
    let profile = ShippingOrdersProfile {
        batches: 1,
        fallback_batches: 1,
        rows,
        total_nanos: sql_elapsed_nanos(started),
        ..Default::default()
    };
    Ok(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_profile_sample(row: usize) -> bool {
    row & 255 == 0
}

pub(super) fn shipping_profile_dense_lookup(
    orderkey: i64,
    pending_values: &[PendingShippingOrder],
    pending_present: &[bool],
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> Option<PendingShippingOrder> {
    if shipping_profile_sample(row) {
        profile.lookup_samples += 1;
        let started = Instant::now();
        let order = usize::try_from(orderkey)
            .ok()
            .filter(|index| *index < pending_present.len() && pending_present[*index])
            .map(|index| pending_values[index]);
        profile.lookup_nanos = profile
            .lookup_nanos
            .saturating_add(sql_elapsed_nanos(started));
        return order;
    }
    usize::try_from(orderkey)
        .ok()
        .filter(|index| *index < pending_present.len() && pending_present[*index])
        .map(|index| pending_values[index])
}

pub(super) fn shipping_profile_map_lookup(
    pending: &AdaptiveI64Map<PendingShippingOrder>,
    orderkey: i64,
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> Option<PendingShippingOrder> {
    if shipping_profile_sample(row) {
        profile.lookup_samples += 1;
        let started = Instant::now();
        let order = pending.get(orderkey);
        profile.lookup_nanos = profile
            .lookup_nanos
            .saturating_add(sql_elapsed_nanos(started));
        return order;
    }
    pending.get(orderkey)
}

pub(super) fn shipping_profile_priority(
    priority_offsets: &[i32],
    priority_data: &[u8],
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> bool {
    if shipping_profile_sample(profile.priority_rows) {
        profile.priority_samples += 1;
        let started = Instant::now();
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = is_high_priority_bytes(priority);
        profile.priority_nanos = profile
            .priority_nanos
            .saturating_add(sql_elapsed_nanos(started));
        profile.priority_rows += 1;
        return is_high_priority;
    }
    let priority = bytes_string_parts(priority_offsets, priority_data, row);
    let is_high_priority = is_high_priority_bytes(priority);
    profile.priority_rows += 1;
    is_high_priority
}

pub(super) fn shipping_profile_priority_vector(
    priorities: Utf8VectorView<'_>,
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> bool {
    if shipping_profile_sample(profile.priority_rows) {
        profile.priority_samples += 1;
        let started = Instant::now();
        let is_high_priority = is_high_priority_bytes(priorities.value_bytes(row));
        profile.priority_nanos = profile
            .priority_nanos
            .saturating_add(sql_elapsed_nanos(started));
        profile.priority_rows += 1;
        return is_high_priority;
    }
    let is_high_priority = is_high_priority_bytes(priorities.value_bytes(row));
    profile.priority_rows += 1;
    is_high_priority
}

pub(super) fn shipping_profile_apply(
    groups: &mut [ShippingPriorityState; 2],
    order: PendingShippingOrder,
    is_high_priority: bool,
    profile: &mut ShippingOrdersProfile,
) {
    if shipping_profile_sample(profile.apply_rows) {
        profile.apply_samples += 1;
        let started = Instant::now();
        apply_pending_order(groups, order, is_high_priority);
        profile.apply_nanos = profile
            .apply_nanos
            .saturating_add(sql_elapsed_nanos(started));
    } else {
        apply_pending_order(groups, order, is_high_priority);
    }
    profile.apply_rows += 1;
}

pub(super) fn shipping_mode_counts_batch_typed_profile(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<ShippingOrdersPartial> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut profile = ShippingOrdersProfile {
        batches: 1,
        typed_batches: 1,
        rows: orderkeys.len(),
        ..Default::default()
    };
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let order = shipping_profile_dense_lookup(
                    orderkey_values[row],
                    pending_values,
                    pending_present,
                    row,
                    &mut profile,
                );
                let Some(order) = order else {
                    profile.lookup_misses += 1;
                    continue;
                };
                profile.lookup_hits += 1;
                let is_high_priority =
                    shipping_profile_priority(priority_offsets, priority_data, row, &mut profile);
                shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
            }
            return Some(ShippingOrdersPartial { groups, profile });
        }
        for row in 0..orderkey_values.len() {
            let order =
                shipping_profile_map_lookup(pending, orderkey_values[row], row, &mut profile);
            let Some(order) = order else {
                profile.lookup_misses += 1;
                continue;
            };
            profile.lookup_hits += 1;
            let is_high_priority =
                shipping_profile_priority(priority_offsets, priority_data, row, &mut profile);
            shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
        }
        return Some(ShippingOrdersPartial { groups, profile });
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            profile.null_rows += 1;
            continue;
        }
        let order = shipping_profile_map_lookup(pending, orderkeys.value(row), row, &mut profile);
        let Some(order) = order else {
            profile.lookup_misses += 1;
            continue;
        };
        profile.lookup_hits += 1;
        let is_high_priority =
            shipping_profile_priority(priority_offsets, priority_data, row, &mut profile);
        shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
    }
    Some(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_mode_counts_batch_dictionary_typed(
    orderkeys: &ArrayRef,
    orderpriorities: DictionaryI32View<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_flags =
        dictionary_i32_view_match_flags(orderpriorities, &[b"1-URGENT", b"2-HIGH"])?;
    let priority_keys = orderpriorities.keys();
    let mut groups = [ShippingPriorityState::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let is_high_priority =
                    dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority =
                dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
            apply_pending_order(&mut groups, order, is_high_priority);
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
        let is_high_priority =
            dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_dictionary_vector_typed(
    orderkeys: I64VectorView<'_>,
    orderpriorities: DictionaryI32View<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let priority_flags =
        dictionary_i32_view_match_flags(orderpriorities, &[b"1-URGENT", b"2-HIGH"])?;
    let priority_keys = orderpriorities.keys();
    let mut groups = [ShippingPriorityState::default(); 2];
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let is_high_priority =
                    dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority =
                dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
            apply_pending_order(&mut groups, order, is_high_priority);
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
        let is_high_priority =
            dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_batch_typed(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [ShippingPriorityState::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let priority = bytes_string_parts(priority_offsets, priority_data, row);
                let is_high_priority = is_high_priority_bytes(priority);
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let priority = bytes_string_parts(priority_offsets, priority_data, row);
            let is_high_priority = is_high_priority_bytes(priority);
            apply_pending_order(&mut groups, order, is_high_priority);
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
        let is_high_priority = is_high_priority_bytes(priority);
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_vector_typed(
    orderkeys: I64VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let mut groups = [ShippingPriorityState::default(); 2];
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
            apply_pending_order(&mut groups, order, is_high_priority);
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
        let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_vector_typed_sorted(
    orderkeys: I64VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let mut groups = [ShippingPriorityState::default(); 2];
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
            apply_pending_order(&mut groups, order, is_high_priority);
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
        let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_vector_typed_profile(
    orderkeys: I64VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<ShippingOrdersPartial> {
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut profile = ShippingOrdersProfile {
        batches: 1,
        typed_batches: 1,
        rows: orderkeys.len(),
        ..Default::default()
    };
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let order = shipping_profile_dense_lookup(
                    orderkey_values[row],
                    pending_values,
                    pending_present,
                    row,
                    &mut profile,
                );
                let Some(order) = order else {
                    profile.lookup_misses += 1;
                    continue;
                };
                profile.lookup_hits += 1;
                let is_high_priority =
                    shipping_profile_priority_vector(orderpriorities, row, &mut profile);
                shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
            }
            return Some(ShippingOrdersPartial { groups, profile });
        }
        for row in 0..orderkey_values.len() {
            let order =
                shipping_profile_map_lookup(pending, orderkey_values[row], row, &mut profile);
            let Some(order) = order else {
                profile.lookup_misses += 1;
                continue;
            };
            profile.lookup_hits += 1;
            let is_high_priority =
                shipping_profile_priority_vector(orderpriorities, row, &mut profile);
            shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
        }
        return Some(ShippingOrdersPartial { groups, profile });
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            profile.null_rows += 1;
            continue;
        }
        let order = shipping_profile_map_lookup(pending, orderkeys.value(row), row, &mut profile);
        let Some(order) = order else {
            profile.lookup_misses += 1;
            continue;
        };
        profile.lookup_hits += 1;
        let is_high_priority = shipping_profile_priority_vector(orderpriorities, row, &mut profile);
        shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
    }
    Some(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_mode_counts_batch_typed_sorted(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [ShippingPriorityState::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let priority = bytes_string_parts(priority_offsets, priority_data, row);
            let is_high_priority = is_high_priority_bytes(priority);
            apply_pending_order(&mut groups, order, is_high_priority);
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
        let is_high_priority = is_high_priority_bytes(priority);
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn apply_pending_order(
    groups: &mut [ShippingPriorityState; 2],
    order: PendingShippingOrder,
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

pub(super) fn merge_shipping_mode_counts(
    groups: &mut [ShippingPriorityState; 2],
    batch_groups: [ShippingPriorityState; 2],
) {
    for index in 0..groups.len() {
        groups[index].high_line_count += batch_groups[index].high_line_count;
        groups[index].low_line_count += batch_groups[index].low_line_count;
    }
}

pub(super) fn merge_orders_partial(
    output: &mut ShippingOrdersPartial,
    partial: ShippingOrdersPartial,
) {
    merge_shipping_mode_counts(&mut output.groups, partial.groups);
    output.profile.add(partial.profile);
}

pub(super) fn log_orders_profile(profile: &ShippingOrdersProfile) {
    if !tpch_profile_enabled() || profile.batches == 0 {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] shipping priority counts orders detail: batches={} typed_batches={} fallback_batches={} rows={} null_rows={} lookup_hits={} lookup_misses={} priority_rows={} apply_rows={} total={:.3} ms lookup_sample={:.3} ms/{} priority_sample={:.3} ms/{} apply_sample={:.3} ms/{}",
        profile.batches,
        profile.typed_batches,
        profile.fallback_batches,
        profile.rows,
        profile.null_rows,
        profile.lookup_hits,
        profile.lookup_misses,
        profile.priority_rows,
        profile.apply_rows,
        sql_nanos_to_millis(profile.total_nanos),
        sql_nanos_to_millis(profile.lookup_nanos),
        profile.lookup_samples,
        sql_nanos_to_millis(profile.priority_nanos),
        profile.priority_samples,
        sql_nanos_to_millis(profile.apply_nanos),
        profile.apply_samples,
    );
}
