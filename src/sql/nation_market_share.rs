use super::*;

fn q08_outer_shape(select: &Select, query: &Query) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let group_by = select.group_by.to_string().to_ascii_lowercase();
    let order_by = query
        .order_by
        .as_ref()
        .map(|order_by| order_by.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    select.from.len() == 1
        && projection.contains("o_year")
        && projection.contains("mkt_share")
        && projection.contains("nation = 'brazil'")
        && group_by.contains("o_year")
        && order_by.contains("o_year")
}

fn q08_inner_shape(select: &Select, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 8
        && projection.contains("extract(year from o_orderdate)")
        && projection.contains("l_extendedprice * (1 - l_discount)")
        && projection.contains("n2.n_name as nation")
        && selection.contains("p_partkey = l_partkey")
        && selection.contains("s_suppkey = l_suppkey")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("o_custkey = c_custkey")
        && selection.contains("c_nationkey = n1.n_nationkey")
        && selection.contains("n1.n_regionkey = r_regionkey")
        && selection.contains("s_nationkey = n2.n_nationkey")
        && selection.contains("o_orderdate between")
        && selection.contains("p_type")
}

pub(super) async fn try_execute_nation_market_share_sql(
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
    if !q08_outer_shape(select, query) {
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
    let Some(selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    if !q08_inner_shape(inner_select, selection) {
        return Ok(None);
    }
    reject_query_features(inner_query)?;
    reject_select_features(inner_select)?;
    let Some(tables) = parse_comma_join_table_refs(inner_select)? else {
        return Ok(None);
    };
    if tables.len() != 8 {
        return Ok(None);
    }
    let mut part = None;
    let mut supplier = None;
    let mut lineitem = None;
    let mut orders = None;
    let mut customer = None;
    let mut nation = None;
    let mut region = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("n1") || alias.eq_ignore_ascii_case("n2") {
            nation.get_or_insert(table);
        } else if alias.eq_ignore_ascii_case("region") {
            region = Some(table);
        }
    }
    let (
        Some(part),
        Some(supplier),
        Some(lineitem),
        Some(orders),
        Some(customer),
        Some(nation),
        Some(region),
    ) = (part, supplier, lineitem, orders, customer, nation, region)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };
    let Some(part_type) = string_equality_literal(&conjuncts, "p_type")? else {
        return Ok(None);
    };
    let Some((start_days, end_days)) = date_between_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };

    let region_keys = region_keys_by_name(engine, region.path, batch_size, &region_name).await?;
    if region_keys.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let customer_nation_names =
        nation_names_by_region_keys(engine, nation.path.clone(), batch_size, &region_keys).await?;
    if customer_nation_names.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let customer_nations =
        q07_customer_nations(engine, customer.path, batch_size, &customer_nation_names).await?;
    if customer_nations.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let order_years = q08_order_years(
        engine,
        orders.path,
        batch_size,
        &customer_nations,
        start_days,
        end_days,
    )
    .await?;
    if order_years.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let part_keys = q08_part_keys(engine, part.path, batch_size, &part_type).await?;
    if part_keys.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let all_nation_names = nation_names_by_keys(engine, nation.path, batch_size).await?;
    let supplier_is_brazil =
        q08_supplier_is_brazil(engine, supplier.path, batch_size, &all_nation_names).await?;
    if supplier_is_brazil.is_empty() {
        return Ok(Some(q08_output(Vec::new())?));
    }
    let mut rows = q08_market_share_rows(
        engine,
        lineitem.path,
        batch_size,
        &order_years,
        &part_keys,
        &supplier_is_brazil,
    )
    .await?;
    rows.sort_by(|left, right| left.o_year.cmp(&right.o_year));
    Ok(Some(q08_output(rows)?))
}

async fn q08_part_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    part_type: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["p_partkey".to_string(), "p_type".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q08_part_keys_view_into(BatchView::new(&batch), part_type, &mut keys)?;
    }
    Ok(keys)
}

fn q08_part_keys_view_into(
    view: BatchView<'_>,
    part_type: &str,
    keys: &mut HashSet<i64>,
) -> Result<()> {
    if view.num_columns() == 2
        && let (Ok(partkeys), Ok(types)) = (view.required_i64(0), view.required_utf8(1))
    {
        for row in 0..view.num_rows() {
            if partkeys.is_valid(row) && types.is_valid(row) && types.value(row) == part_type {
                keys.insert(partkeys.value(row));
            }
        }
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q08 part-key raw vector columns have unsupported types".to_string(),
        ));
    };
    let partkeys = batch_column(batch, "p_partkey")?;
    let types = batch_string_column(batch, "p_type")?;
    for row in 0..batch.num_rows() {
        if types.is_valid(row)
            && types.value(row) == part_type
            && let Some(partkey) = numeric_i64_value(partkeys, row)?
        {
            keys.insert(partkey);
        }
    }
    Ok(())
}

async fn q08_order_years(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i32>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_custkey".to_string(),
                "o_orderdate".to_string(),
            ]),
            None,
        )
        .await?;
    let customer_nations = Arc::new(AdaptiveI64Map::from_hash(customer_nations.clone()));
    parallel_batch_fold_view_chunks(
        &mut stream,
        4,
        HashMap::<i64, i32>::new,
        move |view, orders| {
            merge_maps(
                orders,
                q08_order_years_view(view, &customer_nations, start_days, end_days)?,
            );
            Ok(Some(()))
        },
        Ok,
        HashMap::<i64, i32>::new(),
        merge_maps,
        "Q08 order years",
    )
}

fn q08_order_years_batch(
    batch: RecordBatch,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i32>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let Some(orders) = q08_order_years_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        customer_nations,
        start_days,
        end_days,
    )? {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    let mut year_cache = Date32YearCache::default();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(custkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        if orderdate >= start_days
            && orderdate <= end_days
            && customer_nations.get(custkey).is_some()
        {
            orders.insert(orderkey, year_cache.year(orderdate)?);
        }
    }
    Ok(orders)
}

fn q08_order_years_view(
    view: BatchView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i32>> {
    if view.num_columns() == 3
        && let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
        )
    {
        return q08_order_years_vector(
            orderkeys,
            custkeys,
            orderdates,
            customer_nations,
            start_days,
            end_days,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q08 order year raw vector columns have unsupported types".to_string(),
        ));
    };
    q08_order_years_batch(batch.clone(), customer_nations, start_days, end_days)
}

fn q08_order_years_vector(
    orderkeys: I64VectorView<'_>,
    custkeys: I64VectorView<'_>,
    orderdates: Date32VectorView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<HashMap<i64, i32>> {
    let mut orders = HashMap::new();
    let mut year_cache = Date32YearCache::default();
    if let (Some(orderkey_values), Some(custkey_values), Some(orderdate_values)) = (
        orderkeys.values_if_null_free(),
        custkeys.values_if_null_free(),
        orderdates.values_if_null_free(),
    ) {
        if let Some((_nation_values, nation_present)) = customer_nations.dense_slices() {
            for row in 0..orderkey_values.len() {
                let orderdate = orderdate_values[row];
                if orderdate < start_days || orderdate > end_days {
                    continue;
                }
                let Ok(custkey) = usize::try_from(custkey_values[row]) else {
                    continue;
                };
                if nation_present.get(custkey).copied().unwrap_or(false) {
                    orders.insert(orderkey_values[row], year_cache.year(orderdate)?);
                }
            }
            return Ok(orders);
        }
        for row in 0..orderkey_values.len() {
            let orderdate = orderdate_values[row];
            if orderdate >= start_days
                && orderdate <= end_days
                && customer_nations.get(custkey_values[row]).is_some()
            {
                orders.insert(orderkey_values[row], year_cache.year(orderdate)?);
            }
        }
        return Ok(orders);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        if orderdate >= start_days
            && orderdate <= end_days
            && customer_nations.get(custkeys.value(row)).is_some()
        {
            orders.insert(orderkeys.value(row), year_cache.year(orderdate)?);
        }
    }
    Ok(orders)
}

fn q08_order_years_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<HashMap<i64, i32>>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    let mut orders = HashMap::new();
    let mut year_cache = Date32YearCache::default();
    if orderkeys.null_count() == 0 && custkeys.null_count() == 0 && orderdates.null_count() == 0 {
        for row in 0..orderkeys.len() {
            let orderdate = orderdates.value(row);
            if orderdate >= start_days
                && orderdate <= end_days
                && customer_nations.get(custkeys.value(row)).is_some()
            {
                orders.insert(orderkeys.value(row), year_cache.year(orderdate)?);
            }
        }
        return Ok(Some(orders));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        if orderdate >= start_days
            && orderdate <= end_days
            && customer_nations.get(custkeys.value(row)).is_some()
        {
            orders.insert(orderkeys.value(row), year_cache.year(orderdate)?);
        }
    }
    Ok(Some(orders))
}

async fn q08_supplier_is_brazil(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<HashMap<i64, bool>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q08_supplier_is_brazil_view_into(BatchView::new(&batch), nation_names, &mut suppliers)?;
    }
    Ok(suppliers)
}

fn q08_supplier_is_brazil_view_into(
    view: BatchView<'_>,
    nation_names: &HashMap<i64, String>,
    suppliers: &mut HashMap<i64, bool>,
) -> Result<()> {
    if view.num_columns() == 2
        && let (Ok(suppkeys), Ok(nationkeys)) = (view.required_i64(0), view.required_i64(1))
    {
        for row in 0..view.num_rows() {
            if suppkeys.is_null(row) || nationkeys.is_null(row) {
                continue;
            }
            if let Some(name) = nation_names.get(&nationkeys.value(row)) {
                suppliers.insert(suppkeys.value(row), name == "BRAZIL");
            }
        }
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q08 supplier raw vector columns have unsupported types".to_string(),
        ));
    };
    let suppkeys = batch_column(batch, "s_suppkey")?;
    let nationkeys = batch_column(batch, "s_nationkey")?;
    for row in 0..batch.num_rows() {
        let (Some(suppkey), Some(nationkey)) = (
            numeric_i64_value(suppkeys, row)?,
            numeric_i64_value(nationkeys, row)?,
        ) else {
            continue;
        };
        if let Some(name) = nation_names.get(&nationkey) {
            suppliers.insert(suppkey, name == "BRAZIL");
        }
    }
    Ok(())
}

struct Q08Row {
    o_year: i32,
    mkt_share: f64,
}

async fn q08_market_share_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_years: &HashMap<i64, i32>,
    part_keys: &HashSet<i64>,
    supplier_is_brazil: &HashMap<i64, bool>,
) -> Result<Vec<Q08Row>> {
    if q08_late_materialized_enabled()
        && let Some(rows) = q08_market_share_rows_late_materialized(
            engine,
            path.clone(),
            batch_size,
            order_years,
            part_keys,
            supplier_is_brazil,
        )
        .await?
    {
        return Ok(rows);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_partkey".to_string(),
        "l_suppkey".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let mut pruning_predicates = Vec::new();
    if let Some((min_key, max_key)) = selective_i64_key_range(order_years.keys().copied()) {
        pruning_predicates.extend(i64_range_pruning_predicates("l_orderkey", min_key, max_key));
    }
    if let Some((min_key, max_key)) = selective_i64_key_range(part_keys.iter().copied()) {
        pruning_predicates.extend(i64_range_pruning_predicates("l_partkey", min_key, max_key));
    }
    let mut stream = if should_use_i64_set_row_filter_for_keys(
        true,
        "DODAM_Q08_DISABLE_PARTKEY_ROW_FILTER",
        None,
        part_keys,
        projection_column_count(&projection),
    ) {
        engine
            .scan_parquet_batches_i64_set_filtered_with_row_group_chunk(
                path,
                batch_size,
                projection,
                "l_partkey",
                part_keys.clone(),
                q08_partkey_row_filter_row_group_chunk(),
            )
            .await?
    } else if !pruning_predicates.is_empty() {
        engine
            .scan_parquet_batches_pruned(path, batch_size, projection, pruning_predicates)
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let order_years = Arc::new(AdaptiveI64Map::from_hash(order_years.clone()));
    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let supplier_is_brazil = Arc::new(AdaptiveI64Map::from_hash(supplier_is_brazil.clone()));
    let groups = parallel_batch_fold_view_chunks(
        &mut stream,
        join_aggregate_chunk_size(),
        HashMap::<i32, (f64, f64)>::new,
        move |view, groups| {
            q08_merge_market_share_groups(
                groups,
                q08_market_share_view(view, &order_years, &part_keys, &supplier_is_brazil)?,
            );
            Ok(Some(()))
        },
        Ok,
        HashMap::<i32, (f64, f64)>::new(),
        q08_merge_market_share_groups,
        "Q08 market share aggregate",
    )?;
    Ok(groups
        .into_iter()
        .filter_map(|(o_year, (brazil_volume, total_volume))| {
            (total_volume > 0.0).then_some(Q08Row {
                o_year,
                mkt_share: brazil_volume / total_volume,
            })
        })
        .collect())
}

async fn q08_market_share_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_years: &HashMap<i64, i32>,
    part_keys: &HashSet<i64>,
    supplier_is_brazil: &HashMap<i64, bool>,
) -> Result<Option<Vec<Q08Row>>> {
    let order_years = Arc::new(AdaptiveI64Map::from_hash(order_years.clone()));
    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let supplier_is_brazil = Arc::new(AdaptiveI64Map::from_hash(supplier_is_brazil.clone()));
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["l_partkey".to_string()]),
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            Vec::new(),
            q08_late_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q08_late_max_selected_ratio(),
                q08_late_max_selector_run_ratio(),
            ),
            {
                let order_years = order_years.clone();
                let part_keys = part_keys.clone();
                let supplier_is_brazil = supplier_is_brazil.clone();
                move || Q08LateMarketState {
                    order_years: order_years.clone(),
                    part_keys: part_keys.clone(),
                    supplier_is_brazil: supplier_is_brazil.clone(),
                    groups: HashMap::new(),
                }
            },
            q08_late_build_partkey_selection_view,
            q08_late_consume_market_payload_view,
            |state, _metrics| Ok(Some(state.groups)),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut metrics = LateMaterializedMetrics::default();
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        q08_merge_market_share_groups(&mut groups, chunk.output);
    }
    q08_log_late_market_profile(metrics, q08_late_row_group_chunk());
    Ok(Some(
        groups
            .into_iter()
            .filter_map(|(o_year, (brazil_volume, total_volume))| {
                (total_volume > 0.0).then_some(Q08Row {
                    o_year,
                    mkt_share: brazil_volume / total_volume,
                })
            })
            .collect(),
    ))
}

fn q08_late_materialized_enabled() -> bool {
    std::env::var_os("DODAM_Q08_DISABLE_LATE").is_none()
}

fn q08_partkey_row_filter_row_group_chunk() -> usize {
    std::env::var("DODAM_Q08_PARTKEY_ROW_FILTER_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

fn q08_late_row_group_chunk() -> usize {
    std::env::var("DODAM_Q08_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn q08_late_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q08_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.05)
}

fn q08_late_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q08_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.05)
}

struct Q08LateMarketState {
    order_years: Arc<AdaptiveI64Map<i32>>,
    part_keys: Arc<AdaptiveI64Set>,
    supplier_is_brazil: Arc<AdaptiveI64Map<bool>>,
    groups: HashMap<i32, (f64, f64)>,
}

fn q08_late_build_partkey_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q08LateMarketState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    if partkeys.null_count() == 0 {
        for &partkey in partkeys.values().as_ref() {
            selection.push(state.part_keys.contains(partkey));
        }
        return Ok(Some(()));
    }
    for row in 0..partkeys.len() {
        selection.push(partkeys.is_valid(row) && state.part_keys.contains(partkeys.value(row)));
    }
    Ok(Some(()))
}

fn q08_late_build_partkey_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q08LateMarketState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(partkeys) = view.i64_vector(0) else {
            return Ok(None);
        };
        let dense_part_keys = state.part_keys.dense_contains_slice();
        if let Some(partkey_values) = partkeys.values_if_null_free() {
            for &partkey in partkey_values {
                selection.push(state.part_keys.contains_cached(dense_part_keys, partkey));
            }
            return Ok(Some(()));
        }
        for row in 0..partkeys.len() {
            selection.push(
                !partkeys.is_null(row)
                    && state
                        .part_keys
                        .contains_cached(dense_part_keys, partkeys.value(row)),
            );
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q08_late_build_partkey_selection_batch(batch.clone(), selection, state)
}

fn q08_late_consume_market_payload_batch(
    batch: RecordBatch,
    state: &mut Q08LateMarketState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let (Some(orderkeys), Some(suppkeys), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let (discount_scale, revenue_scale) =
            decimal_discounted_revenue_scales(extendedprices, discounts);
        for row in 0..orderkey_values.len() {
            let Some(o_year) = state.order_years.get(orderkey_values[row]) else {
                continue;
            };
            let Some(is_brazil) = state.supplier_is_brazil.get(suppkey_values[row]) else {
                continue;
            };
            let volume = decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
            let group = state.groups.entry(o_year).or_insert((0.0, 0.0));
            if is_brazil {
                group.0 += volume;
            }
            group.1 += volume;
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let Some(o_year) = state.order_years.get(orderkeys.value(row)) else {
            continue;
        };
        let Some(is_brazil) = state.supplier_is_brazil.get(suppkeys.value(row)) else {
            continue;
        };
        let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
        let group = state.groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
    Ok(Some(()))
}

fn q08_late_consume_market_payload_view(
    view: BatchView<'_>,
    state: &mut Q08LateMarketState,
) -> Result<Option<()>> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(suppkeys), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        )
    {
        q08_late_consume_market_payload_vector(
            orderkeys,
            suppkeys,
            extendedprices,
            discounts,
            state,
        );
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q08_late_consume_market_payload_batch(batch.clone(), state)
}

fn q08_late_consume_market_payload_vector(
    orderkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    state: &mut Q08LateMarketState,
) {
    if let (Some(orderkey_values), Some(suppkey_values)) = (
        orderkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
        for row in 0..orderkey_values.len() {
            let Some(o_year) = state.order_years.get(orderkey_values[row]) else {
                continue;
            };
            let Some(is_brazil) = state.supplier_is_brazil.get(suppkey_values[row]) else {
                continue;
            };
            let volume = decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
            let group = state.groups.entry(o_year).or_insert((0.0, 0.0));
            if is_brazil {
                group.0 += volume;
            }
            group.1 += volume;
        }
        return;
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let Some(o_year) = state.order_years.get(orderkeys.value(row)) else {
            continue;
        };
        let Some(is_brazil) = state.supplier_is_brazil.get(suppkeys.value(row)) else {
            continue;
        };
        let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
        let group = state.groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
}

fn q08_log_late_market_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q08 market share: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q08_market_share_batch(
    batch: RecordBatch,
    order_years: &AdaptiveI64Map<i32>,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> Result<HashMap<i32, (f64, f64)>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let partkeys = batch_column(&batch, "l_partkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q08_market_share_batch_typed(
        orderkeys,
        partkeys,
        suppkeys,
        extendedprices,
        discounts,
        order_years,
        part_keys,
        supplier_is_brazil,
    )? {
        return Ok(groups);
    }
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(partkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(partkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if !part_keys.contains(partkey) {
            continue;
        }
        let Some(o_year) = order_years.get(orderkey) else {
            continue;
        };
        let Some(is_brazil) = supplier_is_brazil.get(suppkey) else {
            continue;
        };
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        let volume = extendedprice * (1.0 - discount);
        let group = groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
    Ok(groups)
}

fn q08_market_share_view(
    view: BatchView<'_>,
    order_years: &AdaptiveI64Map<i32>,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> Result<HashMap<i32, (f64, f64)>> {
    if view.num_columns() == 5
        && let (
            Some(orderkeys),
            Some(partkeys),
            Some(suppkeys),
            Some(extendedprices),
            Some(discounts),
        ) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.i64_vector(2),
            view.decimal128_vector(3),
            view.decimal128_vector(4),
        )
    {
        return Ok(q08_market_share_vector(
            orderkeys,
            partkeys,
            suppkeys,
            extendedprices,
            discounts,
            order_years,
            part_keys,
            supplier_is_brazil,
        ));
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q08 market share raw vector columns have unsupported types".to_string(),
        ));
    };
    q08_market_share_batch(batch.clone(), order_years, part_keys, supplier_is_brazil)
}

fn q08_market_share_vector(
    orderkeys: I64VectorView<'_>,
    partkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    order_years: &AdaptiveI64Map<i32>,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> HashMap<i32, (f64, f64)> {
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    if let (Some(orderkey_values), Some(partkey_values), Some(suppkey_values)) = (
        orderkeys.values_if_null_free(),
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
        let part_contains = part_keys.dense_contains_slice();
        if let (
            Some((order_year_values, order_year_present)),
            Some((brazil_values, brazil_present)),
        ) = (
            order_years.dense_slices(),
            supplier_is_brazil.dense_slices(),
        ) {
            for row in 0..orderkey_values.len() {
                if !part_keys.contains_cached(part_contains, partkey_values[row]) {
                    continue;
                }
                let Ok(orderkey) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if !order_year_present.get(orderkey).copied().unwrap_or(false) {
                    continue;
                }
                let Ok(suppkey) = usize::try_from(suppkey_values[row]) else {
                    continue;
                };
                if !brazil_present.get(suppkey).copied().unwrap_or(false) {
                    continue;
                }
                let volume = decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
                let group = groups
                    .entry(order_year_values[orderkey])
                    .or_insert((0.0, 0.0));
                if brazil_values[suppkey] {
                    group.0 += volume;
                }
                group.1 += volume;
            }
            return groups;
        }
        for row in 0..orderkey_values.len() {
            if !part_keys.contains_cached(part_contains, partkey_values[row]) {
                continue;
            }
            let Some(o_year) = order_years.get(orderkey_values[row]) else {
                continue;
            };
            let Some(is_brazil) = supplier_is_brazil.get(suppkey_values[row]) else {
                continue;
            };
            let volume = decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
            let group = groups.entry(o_year).or_insert((0.0, 0.0));
            if is_brazil {
                group.0 += volume;
            }
            group.1 += volume;
        }
        return groups;
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || partkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        if !part_keys.contains(partkeys.value(row)) {
            continue;
        }
        let Some(o_year) = order_years.get(orderkeys.value(row)) else {
            continue;
        };
        let Some(is_brazil) = supplier_is_brazil.get(suppkeys.value(row)) else {
            continue;
        };
        let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
        let group = groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
    groups
}

fn q08_market_share_batch_typed(
    orderkeys: &ArrayRef,
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    order_years: &AdaptiveI64Map<i32>,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> Result<Option<HashMap<i32, (f64, f64)>>> {
    let (Some(orderkeys), Some(partkeys), Some(suppkeys), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        partkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    if orderkeys.null_count() == 0
        && partkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        for row in 0..orderkeys.len() {
            if !part_keys.contains(partkeys.value(row)) {
                continue;
            }
            let Some(o_year) = order_years.get(orderkeys.value(row)) else {
                continue;
            };
            let Some(is_brazil) = supplier_is_brazil.get(suppkeys.value(row)) else {
                continue;
            };
            let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
            let group = groups.entry(o_year).or_insert((0.0, 0.0));
            if is_brazil {
                group.0 += volume;
            }
            group.1 += volume;
        }
        return Ok(Some(groups));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || partkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        if !part_keys.contains(partkeys.value(row)) {
            continue;
        }
        let Some(o_year) = order_years.get(orderkeys.value(row)) else {
            continue;
        };
        let Some(is_brazil) = supplier_is_brazil.get(suppkeys.value(row)) else {
            continue;
        };
        let volume = extendedprices.value(row) * (1.0 - discounts.value(row));
        let group = groups.entry(o_year).or_insert((0.0, 0.0));
        if is_brazil {
            group.0 += volume;
        }
        group.1 += volume;
    }
    Ok(Some(groups))
}

fn q08_merge_market_share_groups<S>(
    groups: &mut HashMap<i32, (f64, f64), S>,
    batch_groups: HashMap<i32, (f64, f64), S>,
) where
    S: BuildHasher,
{
    for (year, (brazil, total)) in batch_groups {
        let group = groups.entry(year).or_insert((0.0, 0.0));
        group.0 += brazil;
        group.1 += total;
    }
}

fn q08_output(rows: Vec<Q08Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("o_year", DataType::Int64, false),
            Field::new("mkt_share", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| i64::from(row.o_year)),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.mkt_share),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
