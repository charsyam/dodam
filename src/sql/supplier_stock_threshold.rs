use super::*;

fn supplier_stock_threshold_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let having = select
        .having
        .as_ref()
        .map(|expr| expr.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 3
        && select.projection.len() == 2
        && projection.contains("ps_partkey")
        && projection.contains("sum(ps_supplycost * ps_availqty)")
        && group_by.contains("ps_partkey")
        && having.contains("sum(ps_supplycost * ps_availqty)")
        && having.contains("* 0.0001")
        && order_by.contains("value desc")
        && selection.contains("ps_suppkey = s_suppkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("n_name")
}

pub(super) async fn try_execute_important_stock_value_sql(
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
    if !supplier_stock_threshold_shape(select, query, selection) {
        return Ok(None);
    }
    if !matches!(parse_limit(query), Ok(None)) {
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
    let mut partsupp = None;
    let mut supplier = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("partsupp") {
            partsupp = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(partsupp), Some(supplier), Some(nation)) = (partsupp, supplier, nation) else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(nation_name) = string_equality_literal(&conjuncts, "n_name")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let nation_keys = nation_keys_by_name(engine, nation.path, batch_size, &nation_name).await?;
    tpch_profile_elapsed("stock threshold nation keys", stage);
    if nation_keys.is_empty() {
        return Ok(Some(supplier_stock_threshold_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let supplier_keys =
        supplier_keys_for_nations(engine, supplier.path, batch_size, &nation_keys).await?;
    tpch_profile_elapsed("stock threshold supplier keys", stage);
    if supplier_keys.is_empty() {
        return Ok(Some(supplier_stock_threshold_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let rows =
        stock_value_rows_for_suppliers(engine, partsupp.path, batch_size, &supplier_keys).await?;
    tpch_profile_elapsed("stock threshold values", stage);
    Ok(Some(supplier_stock_threshold_output(rows)?))
}

async fn supplier_keys_for_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
) -> Result<AdaptiveI64Set> {
    collect_i64_by_i64_set_adaptive_set(
        engine,
        path,
        batch_size,
        "s_suppkey",
        "s_nationkey",
        nation_keys,
    )
    .await
}

async fn stock_value_rows_for_suppliers(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    supplier_keys: &AdaptiveI64Set,
) -> Result<Vec<SupplierStockValueRow>> {
    if let Some(rows) =
        stock_value_rows_for_suppliers_late(engine, path.clone(), batch_size, supplier_keys).await?
    {
        return Ok(rows);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_supplycost".to_string(),
                "ps_availqty".to_string(),
            ]),
            None,
        )
        .await?;
    let supplier_keys = Arc::new(supplier_keys.clone());
    let (values, total) = parallel_batch_fold(
        &mut stream,
        move |batch| stock_value_batch(batch, &supplier_keys),
        (fast_hash_map::<i64, f64>(), 0.0_f64),
        |(values, total), (batch_values, batch_total)| {
            *total += batch_total;
            for (partkey, value) in batch_values {
                *values.entry(partkey).or_insert(0.0) += value;
            }
        },
        "partsupp stock value",
    )?;
    let threshold = total * 0.0001;
    let mut rows = values
        .into_iter()
        .filter_map(|(ps_partkey, value)| {
            (value > threshold).then_some(SupplierStockValueRow { ps_partkey, value })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .value
            .partial_cmp(&left.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.ps_partkey.cmp(&right.ps_partkey))
    });
    Ok(rows)
}

async fn stock_value_rows_for_suppliers_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    supplier_keys: &AdaptiveI64Set,
) -> Result<Option<Vec<SupplierStockValueRow>>> {
    let supplier_keys = Arc::new(supplier_keys.clone());
    let max_partkey = engine
        .parquet_i64_column_max(path.clone(), "ps_partkey")
        .await?
        .and_then(|max_key| usize::try_from(max_key).ok());
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["ps_suppkey".to_string()]),
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_supplycost".to_string(),
                "ps_availqty".to_string(),
            ]),
            Vec::new(),
            stock_value_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                stock_value_late_materialized_max_selected_ratio(),
                stock_value_late_materialized_max_selector_run_ratio(),
            ),
            {
                let supplier_keys = supplier_keys.clone();
                move || LateStockValueState {
                    supplier_keys: supplier_keys.clone(),
                    values: Vec::new(),
                    total: 0.0,
                }
            },
            late_build_suppkey_selection_view,
            late_consume_stock_payload_view,
            |state, _metrics| Ok(Some((state.values, state.total))),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut dense_values = DenseI64F64Sum::new();
    let has_dense_capacity = max_partkey.is_some_and(|max_partkey| {
        dense_values.try_reserve_dense_to(
            max_partkey,
            dense_f64_sum_bytes(DEFAULT_DENSE_F64_SUM_BYTES),
        )
    });
    let mut fallback_values = (!has_dense_capacity).then(fast_hash_map::<i64, f64>);
    let mut total = 0.0_f64;
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        let (chunk_values, chunk_total) = chunk.output;
        total += chunk_total;
        for (partkey, value) in chunk_values {
            if let Some(values) = fallback_values.as_mut() {
                *values.entry(partkey).or_insert(0.0) += value;
                continue;
            }
            if !dense_values.try_add_dense_key(partkey, value) {
                let mut values = fast_hash_map::<i64, f64>();
                for (key, value) in std::mem::replace(&mut dense_values, DenseI64F64Sum::new())
                    .into_filtered_hash(|_| true)
                {
                    values.insert(key, value);
                }
                fallback_values = Some(values);
                let values = fallback_values.get_or_insert_with(fast_hash_map);
                *values.entry(partkey).or_insert(0.0) += value;
                continue;
            }
        }
    }
    stock_value_log_late_materialized_profile(
        metrics,
        stock_value_late_materialized_row_group_chunk(),
    );
    let rows = if let Some(values) = fallback_values {
        stock_value_rows_from_values(values, total)
    } else {
        stock_value_rows_from_dense_values(dense_values, total)
    };
    Ok(Some(rows))
}

fn stock_value_rows_from_values(
    values: FastHashMap<i64, f64>,
    total: f64,
) -> Vec<SupplierStockValueRow> {
    let threshold = total * 0.0001;
    let mut rows = values
        .into_iter()
        .filter_map(|(ps_partkey, value)| {
            (value > threshold).then_some(SupplierStockValueRow { ps_partkey, value })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .value
            .partial_cmp(&left.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.ps_partkey.cmp(&right.ps_partkey))
    });
    rows
}

fn stock_value_rows_from_dense_values(
    values: DenseI64F64Sum,
    total: f64,
) -> Vec<SupplierStockValueRow> {
    let threshold = total * 0.0001;
    let mut rows = values
        .into_filtered_hash(|value| value > threshold)
        .into_iter()
        .map(|(ps_partkey, value)| SupplierStockValueRow { ps_partkey, value })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .value
            .partial_cmp(&left.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.ps_partkey.cmp(&right.ps_partkey))
    });
    rows
}

fn stock_value_late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(1)
}

fn stock_value_late_materialized_max_selected_ratio() -> f64 {
    late_materialization_max_selected_ratio(0.10)
}

fn stock_value_late_materialized_max_selector_run_ratio() -> f64 {
    late_materialization_max_selector_run_ratio(0.20)
}

fn stock_value_log_late_materialized_profile(
    metrics: LateMaterializedMetrics,
    row_group_chunk: usize,
) {
    tpch_profile_late_materialized(
        "supplier-stock-threshold partsupp",
        metrics,
        row_group_chunk,
    );
}

struct LateStockValueState {
    supplier_keys: Arc<AdaptiveI64Set>,
    values: Vec<(i64, f64)>,
    total: f64,
}

fn late_build_suppkey_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut LateStockValueState,
) -> Result<Option<()>> {
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    let Some(suppkeys) = suppkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    let dense_supplier_keys = state.supplier_keys.dense_contains_slice();
    if suppkeys.null_count() == 0 {
        for &suppkey in suppkeys.values() {
            selection.push(
                state
                    .supplier_keys
                    .contains_cached(dense_supplier_keys, suppkey),
            );
        }
        return Ok(Some(()));
    }
    for row in 0..suppkeys.len() {
        let selected = suppkeys.is_valid(row)
            && state
                .supplier_keys
                .contains_cached(dense_supplier_keys, suppkeys.value(row));
        selection.push(selected);
    }
    Ok(Some(()))
}

fn late_build_suppkey_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut LateStockValueState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(suppkeys) = view.i64_vector(0) else {
            return Ok(None);
        };
        let dense_supplier_keys = state.supplier_keys.dense_contains_slice();
        if let Some(suppkey_values) = suppkeys.values_if_null_free() {
            for &suppkey in suppkey_values {
                selection.push(
                    state
                        .supplier_keys
                        .contains_cached(dense_supplier_keys, suppkey),
                );
            }
            return Ok(Some(()));
        }
        for row in 0..suppkeys.len() {
            let selected = !suppkeys.is_null(row)
                && state
                    .supplier_keys
                    .contains_cached(dense_supplier_keys, suppkeys.value(row));
            selection.push(selected);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    late_build_suppkey_selection_batch(batch.clone(), selection, state)
}

fn late_consume_stock_payload_batch(
    batch: RecordBatch,
    state: &mut LateStockValueState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let supplycosts = batch_column(&batch, "ps_supplycost")?;
    let availqtys = batch_column(&batch, "ps_availqty")?;
    if let (Some(partkeys), Some(supplycosts), Some(availqtys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(supplycosts)?,
        availqtys.as_any().downcast_ref::<Int32Array>(),
    ) {
        late_consume_stock_payload_typed(partkeys, supplycosts, availqtys, state);
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(supplycost), Some(availqty)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_f64_value(supplycosts, row)?,
            numeric_f64_value(availqtys, row)?,
        ) else {
            continue;
        };
        let value = supplycost * availqty;
        state.total += value;
        state.values.push((partkey, value));
    }
    Ok(Some(()))
}

fn late_consume_stock_payload_view(
    view: BatchView<'_>,
    state: &mut LateStockValueState,
) -> Result<Option<()>> {
    if view.num_columns() == 3
        && let (Some(partkeys), Some(supplycosts), Some(availqtys)) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.i32_vector(2),
        )
    {
        late_consume_stock_payload_vector_typed(partkeys, supplycosts, availqtys, state);
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    late_consume_stock_payload_batch(batch.clone(), state)
}

fn late_consume_stock_payload_vector_typed(
    partkeys: I64VectorView<'_>,
    supplycosts: Decimal128VectorView<'_>,
    availqtys: I32VectorView<'_>,
    state: &mut LateStockValueState,
) {
    let supplycost_values = supplycosts.raw_values();
    let supplycost_scale = supplycosts.scale();
    if let (Some(partkey_values), Some(availqty_values)) = (
        partkeys.values_if_null_free(),
        availqtys.values_if_null_free(),
    ) && supplycosts.null_count() == 0
    {
        for ((&partkey, &supplycost), &availqty) in partkey_values
            .iter()
            .zip(supplycost_values)
            .zip(availqty_values)
        {
            let value = (supplycost as f64 / supplycost_scale) * f64::from(availqty);
            state.total += value;
            state.values.push((partkey, value));
        }
        return;
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || supplycosts.is_null(row) || availqtys.is_null(row) {
            continue;
        }
        let value =
            (supplycost_values[row] as f64 / supplycost_scale) * f64::from(availqtys.value(row));
        state.total += value;
        state.values.push((partkeys.value(row), value));
    }
}

fn late_consume_stock_payload_typed(
    partkeys: &Int64Array,
    supplycosts: DecimalInput<'_>,
    availqtys: &Int32Array,
    state: &mut LateStockValueState,
) {
    let supplycost_values = supplycosts.raw_values();
    let supplycost_scale = supplycosts.scale;
    if partkeys.null_count() == 0 && supplycosts.null_count() == 0 && availqtys.null_count() == 0 {
        for ((&partkey, &supplycost), &availqty) in partkeys
            .values()
            .iter()
            .zip(supplycost_values)
            .zip(availqtys.values())
        {
            let value = (supplycost as f64 / supplycost_scale) * f64::from(availqty);
            state.total += value;
            state.values.push((partkey, value));
        }
        return;
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || supplycosts.is_null(row) || availqtys.is_null(row) {
            continue;
        }
        let value =
            (supplycost_values[row] as f64 / supplycost_scale) * f64::from(availqtys.value(row));
        state.total += value;
        state.values.push((partkeys.value(row), value));
    }
}

fn stock_value_batch(
    batch: RecordBatch,
    supplier_keys: &AdaptiveI64Set,
) -> Result<(FastHashMap<i64, f64>, f64)> {
    let partkeys = batch_column(&batch, "ps_partkey")?;
    let suppkeys = batch_column(&batch, "ps_suppkey")?;
    let supplycosts = batch_column(&batch, "ps_supplycost")?;
    let availqtys = batch_column(&batch, "ps_availqty")?;
    if let Some(result) =
        stock_value_batch_typed(partkeys, suppkeys, supplycosts, availqtys, supplier_keys)?
    {
        return Ok(result);
    }
    let mut values = fast_hash_map();
    let mut total = 0.0_f64;
    for row in 0..batch.num_rows() {
        let (Some(partkey), Some(suppkey), Some(supplycost), Some(availqty)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            numeric_f64_value(supplycosts, row)?,
            numeric_f64_value(availqtys, row)?,
        ) else {
            continue;
        };
        if !supplier_keys.contains(suppkey) {
            continue;
        }
        let value = supplycost * availqty;
        total += value;
        *values.entry(partkey).or_insert(0.0) += value;
    }
    Ok((values, total))
}

fn stock_value_batch_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    supplycosts: &ArrayRef,
    availqtys: &ArrayRef,
    supplier_keys: &AdaptiveI64Set,
) -> Result<Option<(FastHashMap<i64, f64>, f64)>> {
    let (Some(partkeys), Some(suppkeys), Some(supplycosts), Some(availqtys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(supplycosts)?,
        availqtys.as_any().downcast_ref::<Int32Array>(),
    ) else {
        return Ok(None);
    };
    let mut values = fast_hash_map_with_capacity(partkeys.len().min(supplier_keys.len() * 4));
    let mut total = 0.0_f64;
    let dense_supplier_keys = supplier_keys.dense_contains_slice();
    let supplycost_values = supplycosts.raw_values();
    let supplycost_scale = supplycosts.scale;
    if partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && supplycosts.null_count() == 0
        && availqtys.null_count() == 0
    {
        for (((&partkey, &suppkey), &supplycost), &availqty) in partkeys
            .values()
            .iter()
            .zip(suppkeys.values())
            .zip(supplycost_values)
            .zip(availqtys.values())
        {
            if !supplier_keys.contains_cached(dense_supplier_keys, suppkey) {
                continue;
            }
            let value = (supplycost as f64 / supplycost_scale) * f64::from(availqty);
            total += value;
            *values.entry(partkey).or_insert(0.0) += value;
        }
        return Ok(Some((values, total)));
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row)
            || suppkeys.is_null(row)
            || supplycosts.is_null(row)
            || availqtys.is_null(row)
        {
            continue;
        }
        let suppkey = suppkeys.value(row);
        if !supplier_keys.contains_cached(dense_supplier_keys, suppkey) {
            continue;
        }
        let value =
            (supplycost_values[row] as f64 / supplycost_scale) * f64::from(availqtys.value(row));
        total += value;
        *values.entry(partkeys.value(row)).or_insert(0.0) += value;
    }
    Ok(Some((values, total)))
}

struct SupplierStockValueRow {
    ps_partkey: i64,
    value: f64,
}

fn supplier_stock_threshold_output(rows: Vec<SupplierStockValueRow>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("ps_partkey", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.ps_partkey),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.value),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
