use super::*;

pub(super) async fn shipping_mode_counts_from_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Vec<ShippingPriorityRow>> {
    if let Some(rows) =
        shipping_mode_counts_from_orders_fused(engine, path.clone(), batch_size, shipmodes, pending)
            .await?
    {
        return Ok(rows);
    }
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
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
    let partials = engine
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
        .await?;
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

pub(super) fn order_chunk_size() -> usize {
    2
}

pub(super) fn order_fused_row_group_chunk() -> usize {
    generic_row_group_map_chunk_size(1)
}

pub(super) fn is_high_priority_str(priority: &str) -> bool {
    is_high_priority_bytes(priority.as_bytes())
}

pub(super) fn is_high_priority_bytes(priority: &[u8]) -> bool {
    matches!(priority.first(), Some(b'1' | b'2')) && matches!(priority, b"1-URGENT" | b"2-HIGH")
}

pub(super) fn shipping_mode_counts_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
    if let Some(groups) = shipping_mode_counts_batch_typed(orderkeys, orderpriorities, pending) {
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

pub(super) fn shipping_mode_counts_projected_view(
    view: BatchView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && let Some(groups) = shipping_mode_counts_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) =
            (view.i64_vector(0), view.dictionary_i32_view(1))
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
        && let Some(mut partial) =
            shipping_mode_counts_vector_typed_profile(orderkeys, orderpriorities, pending)
    {
        partial.profile.total_nanos = sql_elapsed_nanos(started);
        return Ok(partial);
    }
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) =
            (view.i64_vector(0), view.dictionary_i32_view(1))
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
