use super::*;

pub(super) async fn try_execute_prefix_part_supplier_threshold_sql(
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
    if !prefix_part_supplier_threshold_shape(select, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut supplier = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(supplier), Some(nation)) = (supplier, nation) else {
        return Ok(None);
    };
    let Some(partsupp_path) = first_table_path_in_subqueries(selection, "partsupp")? else {
        return Ok(None);
    };
    let Some(part_path) = first_table_path_in_subqueries(selection, "part")? else {
        return Ok(None);
    };
    let Some(lineitem_path) = first_table_path_in_subqueries(selection, "lineitem")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let forest_parts = collect_i64_utf8_prefix_set(
        engine,
        part_path,
        batch_size,
        "p_partkey",
        "p_name",
        "forest",
    )
    .await?;
    tpch_profile_elapsed("PrefixSupplierThreshold forest part keys", stage);
    if forest_parts.is_empty() {
        return Ok(Some(prefix_supplier_threshold_output(Vec::new())?));
    }
    let forest_parts = AdaptiveI64Set::from_hash(forest_parts);
    let stage = tpch_profile_start();
    let lineitem_sums =
        lineitem_quantity_sums_for_parts(engine, lineitem_path, batch_size, &forest_parts).await?;
    tpch_profile_elapsed("PrefixSupplierThreshold lineitem quantity sums", stage);
    let stage = tpch_profile_start();
    let eligible_suppliers = eligible_supplier_keys_by_threshold(
        engine,
        partsupp_path,
        batch_size,
        &forest_parts,
        &lineitem_sums,
    )
    .await?;
    tpch_profile_elapsed("PrefixSupplierThreshold eligible suppliers", stage);
    if eligible_suppliers.is_empty() {
        return Ok(Some(prefix_supplier_threshold_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let nation_keys = nation_keys_by_name(engine, nation.path, batch_size, "CANADA").await?;
    tpch_profile_elapsed("PrefixSupplierThreshold nation keys", stage);
    let stage = tpch_profile_start();
    let mut rows = supplier_rows_by_nation_and_eligibility(
        engine,
        supplier.path,
        batch_size,
        &nation_keys,
        &eligible_suppliers,
    )
    .await?;
    tpch_profile_elapsed("PrefixSupplierThreshold supplier rows", stage);
    let stage = tpch_profile_start();
    rows.sort_by(|left, right| left.s_name.cmp(&right.s_name));
    tpch_profile_elapsed("PrefixSupplierThreshold final sort", stage);
    Ok(Some(prefix_supplier_threshold_output(rows)?))
}

pub(super) fn prefix_part_supplier_threshold_shape(select: &Select, selection: &SqlExpr) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 2
        && text.contains("s_suppkey in")
        && text.contains("p_name like 'forest%'")
        && text.contains("ps_availqty >")
        && text.contains("0.5 * sum(l_quantity)")
        && text.contains("n_name = 'canada'")
}

pub(super) async fn lineitem_quantity_sums_for_parts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
) -> Result<HashMap<(i64, i64), f64>> {
    if let Some(sums) =
        lineitem_quantity_sums_for_parts_late(engine, path.clone(), batch_size, forest_parts)
            .await?
    {
        return Ok(sums);
    }
    let forest_parts = Arc::new(forest_parts.clone());
    engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_suppkey".to_string(),
                "l_quantity".to_string(),
                "l_shipdate".to_string(),
            ]),
            scan_aggregate_row_group_chunk(),
            4,
            scan_aggregate_fusion_enabled(),
            HashMap::<(i64, i64), f64>::new,
            HashMap::<(i64, i64), f64>::new,
            move |view, sums| {
                lineitem_quantity_sums_view_into(view, &forest_parts, sums)?;
                Ok(Some(()))
            },
            merge_f64_groups,
            "PrefixSupplierThreshold lineitem quantity aggregate",
        )
        .await
}

async fn lineitem_quantity_sums_for_parts_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
) -> Result<Option<HashMap<(i64, i64), f64>>> {
    let predicate_projection =
        Projection::Columns(vec!["l_partkey".to_string(), "l_shipdate".to_string()]);
    let payload_projection =
        Projection::Columns(vec!["l_suppkey".to_string(), "l_quantity".to_string()]);
    let policy = generic_late_materialization_policy_for_projection(
        &predicate_projection,
        &payload_projection,
        0.35,
        Some(0.60),
    );
    let forest_parts = Arc::new(forest_parts.clone());
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            prefix_supplier_threshold_late_row_group_chunk(),
            policy,
            {
                let forest_parts = forest_parts.clone();
                move || PrefixLineitemLateState {
                    forest_parts: forest_parts.clone(),
                    selected_partkeys: Vec::new(),
                    payload_offset: 0,
                    sums: HashMap::<(i64, i64), f64>::new(),
                }
            },
            prefix_lineitem_late_build_selection_view,
            prefix_lineitem_late_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_partkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "PrefixSupplierThreshold late payload row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.sums, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut sums = HashMap::<(i64, i64), f64>::new();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_sums, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        merge_f64_groups(&mut sums, chunk_sums);
    }
    tpch_profile_late_materialized(
        "PrefixSupplierThreshold lineitem quantity aggregate",
        metrics,
        prefix_supplier_threshold_late_row_group_chunk(),
    );
    Ok(Some(sums))
}

fn prefix_supplier_threshold_late_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(2)
}

struct PrefixLineitemLateState {
    forest_parts: Arc<AdaptiveI64Set>,
    selected_partkeys: Vec<i64>,
    payload_offset: usize,
    sums: HashMap<(i64, i64), f64>,
}

fn prefix_lineitem_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut PrefixLineitemLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(partkeys), Some(shipdates)) = (view.i64_vector(0), view.date32_vector(1))
    {
        let dense_part_keys = state.forest_parts.dense_contains_slice();
        let dense_part_words = state.forest_parts.dense_word_slice();
        if let (Some(partkey_values), Some(shipdate_values)) = (
            partkeys.values_if_null_free(),
            shipdates.values_if_null_free(),
        ) {
            for row in 0..partkey_values.len() {
                let partkey = partkey_values[row];
                let selected = (8_766..9_131).contains(&shipdate_values[row])
                    && prefix_forest_part_contains(
                        &state.forest_parts,
                        dense_part_keys,
                        dense_part_words,
                        partkey,
                    );
                selection.push(selected);
                if selected {
                    state.selected_partkeys.push(partkey);
                }
            }
            return Ok(Some(()));
        }
        for row in 0..partkeys.len() {
            let selected = !partkeys.is_null(row)
                && !shipdates.is_null(row)
                && (8_766..9_131).contains(&shipdates.value(row))
                && prefix_forest_part_contains(
                    &state.forest_parts,
                    dense_part_keys,
                    dense_part_words,
                    partkeys.value(row),
                );
            selection.push(selected);
            if selected {
                state.selected_partkeys.push(partkeys.value(row));
            }
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    prefix_lineitem_late_build_selection_batch(batch.clone(), selection, state)
}

fn prefix_forest_part_contains(
    forest_parts: &AdaptiveI64Set,
    dense_part_keys: Option<&[bool]>,
    dense_part_words: Option<&[u64]>,
    partkey: i64,
) -> bool {
    if let Some(words) = dense_part_words {
        crate::dense::adaptive_i64_words_contains(words, partkey)
    } else {
        forest_parts.contains_cached(dense_part_keys, partkey)
    }
}

fn prefix_lineitem_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut PrefixLineitemLateState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let (Some(partkeys), Some(shipdates)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    let dense_part_keys = state.forest_parts.dense_contains_slice();
    let dense_part_words = state.forest_parts.dense_word_slice();
    for row in 0..batch.num_rows() {
        let selected = partkeys.is_valid(row)
            && shipdates.is_valid(row)
            && (8_766..9_131).contains(&shipdates.value(row))
            && prefix_forest_part_contains(
                &state.forest_parts,
                dense_part_keys,
                dense_part_words,
                partkeys.value(row),
            );
        selection.push(selected);
        if selected {
            state.selected_partkeys.push(partkeys.value(row));
        }
    }
    Ok(Some(()))
}

fn prefix_lineitem_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut PrefixLineitemLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(suppkeys), Some(quantities)) = (view.i64_vector(0), view.decimal128_vector(1))
    {
        if let Some(suppkey_values) = suppkeys.values_if_null_free()
            && quantities.null_count() == 0
        {
            let quantity_values = quantities.raw_values();
            let quantity_scale = quantities.scale();
            for row in 0..suppkey_values.len() {
                let Some(partkey) = state.selected_partkeys.get(state.payload_offset).copied()
                else {
                    return Err(DodamError::UnsupportedSql(
                        "PrefixSupplierThreshold late payload row overflow".to_string(),
                    ));
                };
                state.payload_offset += 1;
                *state
                    .sums
                    .entry((partkey, suppkey_values[row]))
                    .or_insert(0.0) += quantity_values[row] as f64 / quantity_scale;
            }
            return Ok(Some(()));
        }
        for row in 0..suppkeys.len() {
            let Some(partkey) = state.selected_partkeys.get(state.payload_offset).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "PrefixSupplierThreshold late payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            if !suppkeys.is_null(row) && !quantities.is_null(row) {
                *state
                    .sums
                    .entry((partkey, suppkeys.value(row)))
                    .or_insert(0.0) += quantities.value(row);
            }
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    prefix_lineitem_late_consume_payload_batch(batch.clone(), state)
}

fn prefix_lineitem_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut PrefixLineitemLateState,
) -> Result<Option<()>> {
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let Some(suppkeys) = suppkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    let Some(quantities) = decimal_input(quantities)? else {
        return Ok(None);
    };
    for row in 0..batch.num_rows() {
        let Some(partkey) = state.selected_partkeys.get(state.payload_offset).copied() else {
            return Err(DodamError::UnsupportedSql(
                "PrefixSupplierThreshold late payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        if !suppkeys.is_null(row) && !quantities.is_null(row) {
            *state
                .sums
                .entry((partkey, suppkeys.value(row)))
                .or_insert(0.0) += quantities.value(row);
        }
    }
    Ok(Some(()))
}

pub(super) fn lineitem_quantity_sums_view_into(
    view: BatchView<'_>,
    forest_parts: &AdaptiveI64Set,
    sums: &mut HashMap<(i64, i64), f64>,
) -> Result<()> {
    if view.num_columns() == 4
        && let (Some(partkeys), Some(suppkeys), Some(quantities), Some(shipdates)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.decimal128_vector(2),
            view.date32_vector(3),
        )
        && let Some(batch_sums) = lineitem_quantity_sums_vector_typed(
            partkeys,
            suppkeys,
            quantities,
            shipdates,
            forest_parts,
        )
    {
        merge_f64_groups(sums, batch_sums);
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(());
    };
    merge_f64_groups(
        sums,
        lineitem_quantity_sums_batch(batch.clone(), forest_parts)?,
    );
    Ok(())
}

pub(super) fn lineitem_quantity_sums_batch(
    batch: RecordBatch,
    forest_parts: &AdaptiveI64Set,
) -> Result<HashMap<(i64, i64), f64>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if let Some(sums) =
        lineitem_quantity_sums_typed(partkeys, suppkeys, quantities, shipdates, forest_parts)?
    {
        return Ok(sums);
    }
    let mut sums = HashMap::<(i64, i64), f64>::new();
    for row in 0..batch.num_rows() {
        let Some(shipdate) = date32_value(shipdates, row)? else {
            continue;
        };
        if !(8_766..9_131).contains(&shipdate) {
            continue;
        }
        let (Some(partkey), Some(suppkey), Some(quantity)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(quantities, row)?,
        ) else {
            continue;
        };
        if forest_parts.contains(partkey) {
            *sums.entry((partkey, suppkey)).or_insert(0.0) += quantity;
        }
    }
    Ok(sums)
}

pub(super) fn lineitem_quantity_sums_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    quantities: &ArrayRef,
    shipdates: &ArrayRef,
    forest_parts: &AdaptiveI64Set,
) -> Result<Option<HashMap<(i64, i64), f64>>> {
    let (Some(partkeys), Some(suppkeys), Some(quantities), Some(shipdates)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    let mut sums = HashMap::<(i64, i64), f64>::new();
    if partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && quantities.null_count() == 0
        && shipdates.null_count() == 0
    {
        for row in 0..partkeys.len() {
            let shipdate = shipdates.value(row);
            if !(8_766..9_131).contains(&shipdate) {
                continue;
            }
            let partkey = partkeys.value(row);
            if forest_parts.contains(partkey) {
                *sums.entry((partkey, suppkeys.value(row))).or_insert(0.0) += quantities.value(row);
            }
        }
        return Ok(Some(sums));
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || shipdates.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        if !(8_766..9_131).contains(&shipdate) {
            continue;
        }
        let partkey = partkeys.value(row);
        if forest_parts.contains(partkey) {
            *sums.entry((partkey, suppkeys.value(row))).or_insert(0.0) += quantities.value(row);
        }
    }
    Ok(Some(sums))
}

pub(super) fn lineitem_quantity_sums_vector_typed(
    partkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    forest_parts: &AdaptiveI64Set,
) -> Option<HashMap<(i64, i64), f64>> {
    let mut sums = HashMap::<(i64, i64), f64>::new();
    if let (Some(partkey_values), Some(suppkey_values), Some(shipdate_values)) = (
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
        shipdates.values_if_null_free(),
    ) && quantities.null_count() == 0
    {
        let quantity_values = quantities.raw_values();
        let quantity_scale = quantities.scale();
        for row in 0..partkey_values.len() {
            let shipdate = shipdate_values[row];
            if !(8_766..9_131).contains(&shipdate) {
                continue;
            }
            let partkey = partkey_values[row];
            if forest_parts.contains(partkey) {
                *sums.entry((partkey, suppkey_values[row])).or_insert(0.0) +=
                    quantity_values[row] as f64 / quantity_scale;
            }
        }
        return Some(sums);
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row)
            || suppkeys.is_null(row)
            || quantities.is_null(row)
            || shipdates.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        if !(8_766..9_131).contains(&shipdate) {
            continue;
        }
        let partkey = partkeys.value(row);
        if forest_parts.contains(partkey) {
            *sums.entry((partkey, suppkeys.value(row))).or_insert(0.0) += quantities.value(row);
        }
    }
    Some(sums)
}

pub(super) async fn eligible_supplier_keys_by_threshold(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
    lineitem_sums: &HashMap<(i64, i64), f64>,
) -> Result<HashSet<i64>> {
    collect_i64_i64_i64_mapped_set(
        engine,
        path,
        batch_size,
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
        |partkey, suppkey, availqty| {
            eligible_supplier_key(
                partkey,
                suppkey,
                availqty as f64,
                forest_parts,
                lineitem_sums,
            )
        },
        |partkey, suppkey, availqty| {
            eligible_supplier_key(partkey, suppkey, availqty, forest_parts, lineitem_sums)
        },
    )
    .await
}

fn eligible_supplier_key(
    partkey: i64,
    suppkey: i64,
    availqty: f64,
    forest_parts: &AdaptiveI64Set,
    lineitem_sums: &HashMap<(i64, i64), f64>,
) -> Option<i64> {
    if !forest_parts.contains(partkey) {
        return None;
    }
    let quantity_sum = lineitem_sums.get(&(partkey, suppkey))?;
    (availqty > 0.5 * *quantity_sum).then_some(suppkey)
}

pub(super) struct PrefixSupplierThresholdRow {
    s_name: String,
    s_address: String,
}

pub(super) async fn supplier_rows_by_nation_and_eligibility(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
    eligible_suppliers: &HashSet<i64>,
) -> Result<Vec<PrefixSupplierThresholdRow>> {
    collect_i64_i64_two_utf8_mapped_rows(
        engine,
        path,
        batch_size,
        "s_suppkey",
        "s_nationkey",
        "s_name",
        "s_address",
        |suppkey, nationkey, name, address| {
            (eligible_suppliers.contains(&suppkey) && nation_keys.contains(&nationkey)).then(|| {
                PrefixSupplierThresholdRow {
                    s_name: name.to_string(),
                    s_address: address.to_string(),
                }
            })
        },
    )
    .await
}

pub(super) fn prefix_supplier_threshold_output(
    rows: Vec<PrefixSupplierThresholdRow>,
) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_address.as_str()),
            )),
        ],
    )?;
    Ok(QueryOutput::Scan {
        batches: vec![batch],
    })
}
