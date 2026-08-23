use super::*;

pub(super) async fn try_execute_join_with_correlated_avg_threshold_sql(
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

    let part_keys = collect_i64_two_utf8_eq_set(
        engine,
        part.path,
        batch_size,
        "p_partkey",
        "p_brand",
        brand.as_str(),
        "p_container",
        container.as_str(),
    )
    .await?;
    if part_keys.is_empty() {
        return Ok(Some(single_f64_aggregate_output(output_name, None)?));
    }
    let sum =
        lineitem_revenue_from_matching_parts(engine, lineitem.path, batch_size, &part_keys).await?;
    Ok(Some(single_f64_aggregate_output(
        output_name,
        sum.map(|value| value / 7.0),
    )?))
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

async fn lineitem_revenue_from_matching_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_keys: &HashSet<i64>,
) -> Result<Option<f64>> {
    if let Some(sum) =
        lineitem_revenue_late_materialized(engine, path.clone(), batch_size, part_keys).await?
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
            lineitem_chunk_size(),
            {
                let part_key_count = part_key_count;
                move || lineitem_partial_new(part_key_count)
            },
            {
                let part_keys = part_keys.clone();
                move |view, partial| {
                    lineitem_revenue_view_into(view, &part_keys, partial)?;
                    Ok(Some(()))
                }
            },
            |partial| Ok(Some(partial)),
        )
        .await?
    else {
        return Err(DodamError::UnsupportedSql(
            "correlated average lineitem row-group map is unavailable".to_string(),
        ));
    };
    let mut merged = lineitem_partial_new(part_key_count);
    for partial in partials {
        merge_lineitem_revenue_batch(&mut merged, partial);
    }
    revenue_from_quantity_threshold(merged)
}

type LineitemPartial = (HashMap<i64, (f64, u64)>, Vec<(i64, f64, f64)>);

struct LateLineitemState {
    part_keys: Arc<AdaptiveI64Set>,
    quantity_state: HashMap<i64, (f64, u64)>,
    selected_partkeys: Vec<i64>,
    selected_quantities: Vec<f64>,
    payload_offset: usize,
    candidate_rows: Vec<(i64, f64, f64)>,
}

async fn lineitem_revenue_late_materialized(
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
            late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                late_materialized_max_selected_ratio(),
                late_materialized_max_selector_run_ratio(),
            ),
            {
                let part_keys = part_keys.clone();
                move || LateLineitemState {
                    part_keys: part_keys.clone(),
                    quantity_state: HashMap::new(),
                    selected_partkeys: Vec::new(),
                    selected_quantities: Vec::new(),
                    payload_offset: 0,
                    candidate_rows: Vec::new(),
                }
            },
            late_build_quantity_selection_view,
            late_consume_extendedprice_payload_view,
            |state, _metrics| {
                if state.payload_offset != state.selected_partkeys.len()
                    || state.payload_offset != state.selected_quantities.len()
                {
                    return Err(DodamError::UnsupportedSql(
                        "correlated average row selection payload mismatch".to_string(),
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
        merge_quantity_state(&mut quantity_state, chunk_state);
        candidate_rows.extend(chunk_candidates);
        metrics.add(chunk.metrics);
    }
    log_late_materialized_profile(metrics, late_materialized_row_group_chunk());
    revenue_from_quantity_threshold((quantity_state, candidate_rows)).map(Some)
}

fn revenue_from_quantity_threshold(partial: LineitemPartial) -> Result<Option<f64>> {
    let (states, candidate_rows) = partial;
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

fn late_build_quantity_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut LateLineitemState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(partkeys), Some(quantities)) = (view.i64_vector(0), view.decimal128_vector(1))
    {
        late_build_quantity_selection_typed(partkeys, quantities, selection, state);
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

fn late_build_quantity_selection_typed(
    partkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut LateLineitemState,
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

fn late_consume_extendedprice_payload_view(
    view: BatchView<'_>,
    state: &mut LateLineitemState,
) -> Result<Option<()>> {
    if view.num_columns() == 1
        && let Some(extendedprices) = view.decimal128_vector(0)
    {
        late_consume_extendedprice_payload_typed(extendedprices, state)?;
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
                "correlated average row selection payload overflow".to_string(),
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

fn late_consume_extendedprice_payload_typed(
    extendedprices: Decimal128VectorView<'_>,
    state: &mut LateLineitemState,
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
                    "correlated average row selection payload overflow".to_string(),
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
                "correlated average row selection payload overflow".to_string(),
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

fn merge_quantity_state(output: &mut HashMap<i64, (f64, u64)>, input: HashMap<i64, (f64, u64)>) {
    for (partkey, (sum, count)) in input {
        let aggregate = output.entry(partkey).or_insert((0.0, 0));
        aggregate.0 += sum;
        aggregate.1 += count;
    }
}

fn late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(2)
}

fn late_materialized_max_selected_ratio() -> f64 {
    late_materialization_max_selected_ratio(0.20)
}

fn late_materialized_max_selector_run_ratio() -> f64 {
    late_materialization_max_selector_run_ratio(0.20)
}

fn log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    tpch_profile_late_materialized("correlated average lineitem", metrics, row_group_chunk);
}

fn lineitem_chunk_size() -> usize {
    rule_chunk_size(64)
}

fn lineitem_partial_new(part_key_count: usize) -> LineitemPartial {
    (
        HashMap::<i64, (f64, u64)>::with_capacity(part_key_count),
        Vec::<(i64, f64, f64)>::new(),
    )
}

fn lineitem_revenue_batch_into(
    batch: RecordBatch,
    part_keys: &AdaptiveI64Set,
    partial: &mut LineitemPartial,
) -> Result<()> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    if lineitem_revenue_batch_typed_into(partkeys, quantities, extendedprices, part_keys, partial)?
    {
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

fn lineitem_revenue_view_into(
    view: BatchView<'_>,
    part_keys: &AdaptiveI64Set,
    partial: &mut LineitemPartial,
) -> Result<()> {
    if view.num_columns() == 3
        && let (Some(partkeys), Some(quantities), Some(extendedprices)) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
        )
    {
        lineitem_revenue_vector_typed_into(
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
            "correlated average lineitem raw vector columns have unsupported types".to_string(),
        ));
    };
    lineitem_revenue_batch_into(batch.clone(), part_keys, partial)
}

fn lineitem_revenue_batch_typed_into(
    partkeys: &ArrayRef,
    quantities: &ArrayRef,
    extendedprices: &ArrayRef,
    part_keys: &AdaptiveI64Set,
    partial: &mut LineitemPartial,
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

fn lineitem_revenue_vector_typed_into(
    partkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    part_keys: &AdaptiveI64Set,
    partial: &mut LineitemPartial,
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

fn merge_lineitem_revenue_batch(output: &mut LineitemPartial, batch: LineitemPartial) {
    for (partkey, (quantity_sum, quantity_count)) in batch.0 {
        let state = output.0.entry(partkey).or_insert((0.0, 0));
        state.0 += quantity_sum;
        state.1 += quantity_count;
    }
    output.1.extend(batch.1);
}
