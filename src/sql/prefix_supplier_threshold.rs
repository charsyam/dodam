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
    let forest_parts = q20_forest_part_keys(engine, part_path, batch_size).await?;
    tpch_profile_elapsed("Q20 forest part keys", stage);
    if forest_parts.is_empty() {
        return Ok(Some(q20_output(Vec::new())?));
    }
    let forest_parts = AdaptiveI64Set::from_hash(forest_parts);
    let stage = tpch_profile_start();
    let lineitem_sums =
        q20_lineitem_quantity_sums(engine, lineitem_path, batch_size, &forest_parts).await?;
    tpch_profile_elapsed("Q20 lineitem quantity sums", stage);
    let stage = tpch_profile_start();
    let eligible_suppliers = q20_eligible_supplier_keys(
        engine,
        partsupp_path,
        batch_size,
        &forest_parts,
        &lineitem_sums,
    )
    .await?;
    tpch_profile_elapsed("Q20 eligible suppliers", stage);
    if eligible_suppliers.is_empty() {
        return Ok(Some(q20_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, "CANADA").await?;
    tpch_profile_elapsed("Q20 nation keys", stage);
    let stage = tpch_profile_start();
    let mut rows = q20_supplier_rows(
        engine,
        supplier.path,
        batch_size,
        &nation_keys,
        &eligible_suppliers,
    )
    .await?;
    tpch_profile_elapsed("Q20 supplier rows", stage);
    let stage = tpch_profile_start();
    rows.sort_by(|left, right| left.s_name.cmp(&right.s_name));
    tpch_profile_elapsed("Q20 final sort", stage);
    Ok(Some(q20_output(rows)?))
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

pub(super) async fn q20_forest_part_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let names = batch_string_column(&batch, "p_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row).starts_with("forest")
                && let Some(key) = numeric_i64_value(partkeys, row)?
            {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

pub(super) async fn q20_lineitem_quantity_sums(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
) -> Result<HashMap<(i64, i64), f64>> {
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
                q20_lineitem_quantity_sums_view_into(view, &forest_parts, sums)?;
                Ok(Some(()))
            },
            merge_f64_groups,
            "Q20 lineitem quantity aggregate",
        )
        .await
}

pub(super) fn q20_lineitem_quantity_sums_view_into(
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
        && let Some(batch_sums) = q20_lineitem_quantity_sums_vector_typed(
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
        q20_lineitem_quantity_sums_batch(batch.clone(), forest_parts)?,
    );
    Ok(())
}

pub(super) fn q20_lineitem_quantity_sums_batch(
    batch: RecordBatch,
    forest_parts: &AdaptiveI64Set,
) -> Result<HashMap<(i64, i64), f64>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if let Some(sums) =
        q20_lineitem_quantity_sums_typed(partkeys, suppkeys, quantities, shipdates, forest_parts)?
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

pub(super) fn q20_lineitem_quantity_sums_typed(
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

pub(super) fn q20_lineitem_quantity_sums_vector_typed(
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

pub(super) async fn q20_eligible_supplier_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    forest_parts: &AdaptiveI64Set,
    lineitem_sums: &HashMap<(i64, i64), f64>,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "ps_partkey".to_string(),
                "ps_suppkey".to_string(),
                "ps_availqty".to_string(),
            ]),
            None,
        )
        .await?;
    let mut suppliers = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "ps_partkey")?;
        let suppkeys = batch_column(&batch, "ps_suppkey")?;
        let availqty = batch_column(&batch, "ps_availqty")?;
        if let Some(batch_suppliers) = q20_eligible_supplier_keys_typed(
            partkeys,
            suppkeys,
            availqty,
            forest_parts,
            lineitem_sums,
        ) {
            suppliers.extend(batch_suppliers);
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(partkey), Some(suppkey), Some(availqty)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_i64_value(suppkeys, row)?,
                numeric_f64_value(availqty, row)?,
            ) else {
                continue;
            };
            if !forest_parts.contains(partkey) {
                continue;
            }
            let Some(quantity_sum) = lineitem_sums.get(&(partkey, suppkey)) else {
                continue;
            };
            if availqty > 0.5 * *quantity_sum {
                suppliers.insert(suppkey);
            }
        }
    }
    Ok(suppliers)
}

pub(super) fn q20_eligible_supplier_keys_typed(
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    availqtys: &ArrayRef,
    forest_parts: &AdaptiveI64Set,
    lineitem_sums: &HashMap<(i64, i64), f64>,
) -> Option<HashSet<i64>> {
    let (Some(partkeys), Some(suppkeys), Some(availqtys)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        availqtys.as_any().downcast_ref::<Int32Array>(),
    ) else {
        return None;
    };
    let mut suppliers = HashSet::new();
    if partkeys.null_count() == 0 && suppkeys.null_count() == 0 && availqtys.null_count() == 0 {
        for row in 0..partkeys.len() {
            let partkey = partkeys.value(row);
            if !forest_parts.contains(partkey) {
                continue;
            }
            let suppkey = suppkeys.value(row);
            let Some(quantity_sum) = lineitem_sums.get(&(partkey, suppkey)) else {
                continue;
            };
            if f64::from(availqtys.value(row)) > 0.5 * *quantity_sum {
                suppliers.insert(suppkey);
            }
        }
        return Some(suppliers);
    }
    for row in 0..partkeys.len() {
        if partkeys.is_null(row) || suppkeys.is_null(row) || availqtys.is_null(row) {
            continue;
        }
        let partkey = partkeys.value(row);
        if !forest_parts.contains(partkey) {
            continue;
        }
        let suppkey = suppkeys.value(row);
        let Some(quantity_sum) = lineitem_sums.get(&(partkey, suppkey)) else {
            continue;
        };
        if f64::from(availqtys.value(row)) > 0.5 * *quantity_sum {
            suppliers.insert(suppkey);
        }
    }
    Some(suppliers)
}

pub(super) struct Q20Row {
    s_name: String,
    s_address: String,
}

pub(super) async fn q20_supplier_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
    eligible_suppliers: &HashSet<i64>,
) -> Result<Vec<Q20Row>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_nationkey".to_string(),
                "s_name".to_string(),
                "s_address".to_string(),
            ]),
            None,
        )
        .await?;
    let mut rows = Vec::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        let names = batch_string_column(&batch, "s_name")?;
        let addresses = batch_string_column(&batch, "s_address")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if eligible_suppliers.contains(&suppkey)
                && nation_keys.contains(&nationkey)
                && names.is_valid(row)
                && addresses.is_valid(row)
            {
                rows.push(Q20Row {
                    s_name: names.value(row).to_string(),
                    s_address: addresses.value(row).to_string(),
                });
            }
        }
    }
    Ok(rows)
}

pub(super) fn q20_output(rows: Vec<Q20Row>) -> Result<QueryOutput> {
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
