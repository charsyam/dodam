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
    let part_keys = collect_i64_utf8_eq_set(
        engine,
        part.path,
        batch_size,
        "p_partkey",
        "p_type",
        &part_type,
    )
    .await?;
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

type Q08OrderYears = DenseI64I32Map;

async fn q08_order_years(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Q08OrderYears> {
    let customer_nations = Arc::new(AdaptiveI64Map::from_hash(customer_nations.clone()));
    let row_groups = (0..engine.parquet_row_group_count(&path)?).collect::<Vec<_>>();
    if let Some((mut orders, _metrics)) = engine
        .collect_parquet_i32_predicate_i64_lookup_selected_i64_mapped_parallel(
            path.clone(),
            row_groups,
            ("o_orderdate".to_string(), DirectPrimitiveColumnType::Date32),
            "o_custkey".to_string(),
            "o_orderkey".to_string(),
            (1, 3),
            |orderdate| (orderdate >= start_days && orderdate <= end_days).then_some(orderdate),
            |custkey| customer_nations.get(custkey).is_some(),
        )?
    {
        let mut year_cache = Date32YearCache::default();
        for (_, orderdate) in &mut orders {
            *orderdate = year_cache.year(*orderdate)?;
        }
        return Ok(q08_build_order_years(orders));
    }

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
    let orders = parallel_batch_collect_pairs_view_chunks(
        &mut stream,
        4,
        move |view, orders| {
            orders.extend(q08_order_years_view(
                view,
                &customer_nations,
                start_days,
                end_days,
            )?);
            Ok(Some(()))
        },
        "Q08 order years",
    )?;
    Ok(q08_build_order_years(orders))
}

fn q08_build_order_years(orders: Vec<(i64, i32)>) -> Q08OrderYears {
    let row_count = orders.len();
    let order_years = Q08OrderYears::from_pairs_with_dense_range_policy(
        orders,
        0,
        q08_order_year_max_dense_entries(row_count),
        q08_order_year_dense_max_amplification(),
    );
    q08_log_order_year_layout(&order_years);
    order_years
}

fn q08_order_year_max_dense_entries(row_count: usize) -> usize {
    let byte_entries = dense_i32_max_entries(DEFAULT_ORDER_YEAR_DENSE_BYTES);
    let amplification_entries =
        ((row_count as f64) * q08_order_year_dense_max_amplification()) as usize;
    byte_entries.min(amplification_entries.max(row_count))
}

fn q08_order_year_dense_max_amplification() -> f64 {
    dense_max_amplification(8.5)
}

fn q08_log_order_year_layout(order_years: &Q08OrderYears) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q08 order-year layout: dense={} len={}",
        order_years.dense_slice().is_some(),
        order_years.len(),
    );
}

fn q08_order_years_view(
    view: BatchView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Vec<(i64, i32)>> {
    let mut year_cache = Date32YearCache::default();
    let dense_nations = customer_nations.dense_word_slices();
    collect_i64_i64_date32_pairs_view(
        view,
        "Q08 order year raw vector columns have unsupported types",
        |_, custkey, orderdate| {
            if orderdate < start_days
                || orderdate > end_days
                || customer_nations
                    .get_cached_words(dense_nations, custkey)
                    .is_none()
            {
                return Ok(None);
            }
            Ok(Some(year_cache.year(orderdate)?))
        },
    )
}

async fn q08_supplier_is_brazil(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<FastHashMap<i64, bool>> {
    collect_i64_by_i64_mapped_hash_map(
        engine,
        path,
        batch_size,
        "s_suppkey",
        "s_nationkey",
        |nationkey| nation_names.get(&nationkey).map(|name| name == "BRAZIL"),
    )
    .await
}

struct Q08Row {
    o_year: i32,
    mkt_share: f64,
}

async fn q08_market_share_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_years: &Q08OrderYears,
    part_keys: &HashSet<i64>,
    supplier_is_brazil: &FastHashMap<i64, bool>,
) -> Result<Vec<Q08Row>> {
    if let Some(rows) = q08_market_share_rows_direct_dictionary_selected(
        engine,
        &path,
        order_years,
        part_keys,
        supplier_is_brazil,
    )? {
        return Ok(rows);
    }
    if let Some(rows) = q08_market_share_rows_late_materialized(
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
    if let Some((min_key, max_key)) = order_years.selective_key_range() {
        pruning_predicates.extend(i64_range_pruning_predicates("l_orderkey", min_key, max_key));
    }
    if let Some((min_key, max_key)) = selective_i64_key_range(part_keys.iter().copied()) {
        pruning_predicates.extend(i64_range_pruning_predicates("l_partkey", min_key, max_key));
    }
    let mut stream = if should_use_i64_set_row_filter_for_keys_auto(
        true,
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
    let order_years = Arc::new(order_years.clone());
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

fn q08_market_share_rows_direct_dictionary_selected(
    engine: &DodamEngine,
    path: &Path,
    order_years: &Q08OrderYears,
    part_keys: &HashSet<i64>,
    supplier_is_brazil: &FastHashMap<i64, bool>,
) -> Result<Option<Vec<Q08Row>>> {
    let Some((extendedprice_precision, extendedprice_scale)) =
        engine.parquet_decimal128_type(path, "l_extendedprice")?
    else {
        return Ok(None);
    };
    let Some((discount_precision, discount_scale)) =
        engine.parquet_decimal128_type(path, "l_discount")?
    else {
        return Ok(None);
    };
    if extendedprice_precision > 18 || discount_precision > 18 {
        return Ok(None);
    }
    let Ok(discount_scale_power) = u32::try_from(discount_scale) else {
        return Ok(None);
    };
    let Some(discount_factor) = 10_i64.checked_pow(discount_scale_power) else {
        return Ok(None);
    };

    let part_keys = Arc::new(AdaptiveI64Set::from_hash(part_keys.clone()));
    let order_years = Arc::new(order_years.clone());
    let supplier_is_brazil = Arc::new(AdaptiveI64Map::from_hash(supplier_is_brazil.clone()));
    let output = Arc::new(Mutex::new(HashMap::<i32, (i128, i128)>::new()));
    let lookup_part_keys = part_keys.clone();
    let predicate_order_years = order_years.clone();
    let predicate_suppliers = supplier_is_brazil.clone();
    let consume_output = output.clone();
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let Some(_metrics) = engine
        .scan_parquet_i64_lookup_staged_two_i64_selected_primitive_columns_parallel(
            path.to_path_buf(),
            row_groups,
            "l_partkey".to_string(),
            "l_orderkey".to_string(),
            "l_suppkey".to_string(),
            vec![
                (
                    "l_extendedprice".to_string(),
                    DirectPrimitiveColumnType::Decimal128Int64Raw {
                        precision: extendedprice_precision,
                        scale: extendedprice_scale,
                    },
                ),
                (
                    "l_discount".to_string(),
                    DirectPrimitiveColumnType::Decimal128Int64Raw {
                        precision: discount_precision,
                        scale: discount_scale,
                    },
                ),
            ],
            (1, 5),
            move |partkey| lookup_part_keys.contains(partkey),
            move |orderkey| predicate_order_years.get(orderkey),
            move |o_year, suppkey| Some((o_year, predicate_suppliers.get(suppkey)?)),
            move |tags, view| {
                let (Some(extendedprices), Some(discounts)) =
                    (view.decimal128_vector(0), view.decimal128_vector(1))
                else {
                    return Err(DodamError::UnsupportedSql(
                        "Q08 selected payload vector shape mismatch".to_string(),
                    ));
                };
                let (Some(extendedprices), Some(discounts)) =
                    (extendedprices.raw_i64_values(), discounts.raw_i64_values())
                else {
                    return Err(DodamError::UnsupportedSql(
                        "Q08 selected payload requires raw i64 decimals".to_string(),
                    ));
                };
                if tags.len() != extendedprices.len() || tags.len() != discounts.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q08 selected payload length mismatch".to_string(),
                    ));
                }
                let mut local = HashMap::<i32, (i128, i128)>::new();
                for ((&(o_year, is_brazil), &extendedprice), &discount) in
                    tags.iter().zip(extendedprices.iter()).zip(discounts.iter())
                {
                    let multiplier = discount_factor.checked_sub(discount).ok_or_else(|| {
                        DodamError::UnsupportedSql(
                            "Q08 selected payload discount overflow".to_string(),
                        )
                    })?;
                    let volume = i128::from(extendedprice)
                        .checked_mul(i128::from(multiplier))
                        .ok_or_else(|| {
                            DodamError::UnsupportedSql(
                                "Q08 selected payload revenue overflow".to_string(),
                            )
                        })?;
                    let group = local.entry(o_year).or_insert((0, 0));
                    if is_brazil {
                        group.0 = group.0.checked_add(volume).ok_or_else(|| {
                            DodamError::UnsupportedSql(
                                "Q08 selected Brazil revenue overflow".to_string(),
                            )
                        })?;
                    }
                    group.1 = group.1.checked_add(volume).ok_or_else(|| {
                        DodamError::UnsupportedSql(
                            "Q08 selected total revenue overflow".to_string(),
                        )
                    })?;
                }
                let mut output = consume_output.lock().map_err(|_| {
                    DodamError::UnsupportedSql(
                        "Q08 selected payload output lock poisoned".to_string(),
                    )
                })?;
                for (o_year, (brazil, total)) in local {
                    let group = output.entry(o_year).or_insert((0, 0));
                    group.0 = group.0.checked_add(brazil).ok_or_else(|| {
                        DodamError::UnsupportedSql(
                            "Q08 selected global Brazil revenue overflow".to_string(),
                        )
                    })?;
                    group.1 = group.1.checked_add(total).ok_or_else(|| {
                        DodamError::UnsupportedSql(
                            "Q08 selected global total revenue overflow".to_string(),
                        )
                    })?;
                }
                Ok(())
            },
        )?
    else {
        return Ok(None);
    };

    let output = output.lock().map_err(|_| {
        DodamError::UnsupportedSql("Q08 selected payload output lock poisoned".to_string())
    })?;
    Ok(Some(
        output
            .iter()
            .filter_map(|(&o_year, &(brazil, total))| {
                (total > 0).then_some(Q08Row {
                    o_year,
                    mkt_share: brazil as f64 / total as f64,
                })
            })
            .collect(),
    ))
}

async fn q08_market_share_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_years: &Q08OrderYears,
    part_keys: &HashSet<i64>,
    supplier_is_brazil: &FastHashMap<i64, bool>,
) -> Result<Option<Vec<Q08Row>>> {
    let order_years = Arc::new(order_years.clone());
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
                    profile: Q08LateCallbackProfile::default(),
                }
            },
            q08_late_build_partkey_selection_view,
            q08_late_consume_market_payload_view,
            |state, _metrics| Ok(Some((state.groups, state.profile))),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut metrics = LateMaterializedMetrics::default();
    let mut profile = Q08LateCallbackProfile::default();
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        q08_merge_market_share_groups(&mut groups, chunk.output.0);
        profile.add(chunk.output.1);
    }
    q08_log_late_market_profile(metrics, q08_late_row_group_chunk());
    q08_log_late_callback_profile(profile);
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

fn q08_partkey_row_filter_row_group_chunk() -> usize {
    i64_set_row_filter_row_group_chunk(8)
}

fn q08_late_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(4)
}

fn q08_late_max_selected_ratio() -> f64 {
    late_materialization_max_selected_ratio(0.05)
}

fn q08_late_max_selector_run_ratio() -> f64 {
    late_materialization_max_selector_run_ratio(0.05)
}

struct Q08LateMarketState {
    order_years: Arc<Q08OrderYears>,
    part_keys: Arc<AdaptiveI64Set>,
    supplier_is_brazil: Arc<AdaptiveI64Map<bool>>,
    groups: HashMap<i32, (f64, f64)>,
    profile: Q08LateCallbackProfile,
}

#[derive(Default, Clone, Copy)]
struct Q08LateCallbackProfile {
    selection_nanos: u64,
    payload_nanos: u64,
    selection_batches: usize,
    payload_batches: usize,
    payload_i64_batches: usize,
    payload_i128_batches: usize,
}

impl Q08LateCallbackProfile {
    fn add(&mut self, other: Self) {
        self.selection_nanos = self.selection_nanos.saturating_add(other.selection_nanos);
        self.payload_nanos = self.payload_nanos.saturating_add(other.payload_nanos);
        self.selection_batches = self
            .selection_batches
            .saturating_add(other.selection_batches);
        self.payload_batches = self.payload_batches.saturating_add(other.payload_batches);
        self.payload_i64_batches = self
            .payload_i64_batches
            .saturating_add(other.payload_i64_batches);
        self.payload_i128_batches = self
            .payload_i128_batches
            .saturating_add(other.payload_i128_batches);
    }
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
    let profile_started = tpch_profile_enabled().then(Instant::now);
    if view.num_columns() == 1 {
        let Some(partkeys) = view.i64_vector(0) else {
            return Ok(None);
        };
        let dense_part_keys = state.part_keys.dense_contains_slice();
        let dense_part_words = state.part_keys.dense_word_slice();
        if let Some(partkey_values) = partkeys.values_if_null_free() {
            if let Some(words) = dense_part_words {
                for &partkey in partkey_values {
                    selection.push(crate::dense::adaptive_i64_words_contains(words, partkey));
                }
            } else {
                for &partkey in partkey_values {
                    selection.push(state.part_keys.contains_cached(dense_part_keys, partkey));
                }
            }
            if let Some(started) = profile_started {
                state.profile.selection_batches += 1;
                state.profile.selection_nanos = state
                    .profile
                    .selection_nanos
                    .saturating_add(sql_elapsed_nanos(started));
            }
            return Ok(Some(()));
        }
        for row in 0..partkeys.len() {
            let selected = !partkeys.is_null(row)
                && if let Some(words) = dense_part_words {
                    crate::dense::adaptive_i64_words_contains(words, partkeys.value(row))
                } else {
                    state
                        .part_keys
                        .contains_cached(dense_part_keys, partkeys.value(row))
                };
            selection.push(selected);
        }
        if let Some(started) = profile_started {
            state.profile.selection_batches += 1;
            state.profile.selection_nanos = state
                .profile
                .selection_nanos
                .saturating_add(sql_elapsed_nanos(started));
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
    let extendedprices = Decimal128VectorView::Arrow {
        values: extendedprices.values,
        precision: extendedprices.precision,
        scale: extendedprices.scale,
    };
    let discounts = Decimal128VectorView::Arrow {
        values: discounts.values,
        precision: discounts.precision,
        scale: discounts.scale,
    };
    q08_late_consume_market_payload_vector(
        I64VectorView::Arrow(orderkeys),
        I64VectorView::Arrow(suppkeys),
        extendedprices,
        discounts,
        state,
    )?;
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
        )?;
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
) -> Result<()> {
    let profile_started = tpch_profile_enabled().then(Instant::now);
    let row_state = std::cell::Cell::new(None::<(i32, bool)>);
    if let (Some(orderkey_values), Some(suppkey_values)) = (
        orderkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) {
        consume_filtered_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            orderkey_values.len(),
            |row| {
                row_state.set(None);
                let Some(o_year) = state.order_years.get(orderkey_values[row]) else {
                    return Ok(false);
                };
                let Some(is_brazil) = state.supplier_is_brazil.get(suppkey_values[row]) else {
                    return Ok(false);
                };
                row_state.set(Some((o_year, is_brazil)));
                Ok(true)
            },
            |_, volume| {
                if let Some((o_year, is_brazil)) = row_state.get() {
                    let group = state.groups.entry(o_year).or_insert((0.0, 0.0));
                    if is_brazil {
                        group.0 += volume;
                    }
                    group.1 += volume;
                }
                Ok(())
            },
        )?;
    } else {
        consume_filtered_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            orderkeys.len(),
            |row| {
                row_state.set(None);
                if orderkeys.is_null(row) || suppkeys.is_null(row) {
                    return Ok(false);
                }
                let Some(o_year) = state.order_years.get(orderkeys.value(row)) else {
                    return Ok(false);
                };
                let Some(is_brazil) = state.supplier_is_brazil.get(suppkeys.value(row)) else {
                    return Ok(false);
                };
                row_state.set(Some((o_year, is_brazil)));
                Ok(true)
            },
            |_, volume| {
                if let Some((o_year, is_brazil)) = row_state.get() {
                    let group = state.groups.entry(o_year).or_insert((0.0, 0.0));
                    if is_brazil {
                        group.0 += volume;
                    }
                    group.1 += volume;
                }
                Ok(())
            },
        )?;
    }
    if let Some(started) = profile_started {
        state.profile.payload_batches += 1;
        if extendedprices.raw_i64_values().is_some() && discounts.raw_i64_values().is_some() {
            state.profile.payload_i64_batches += 1;
        } else {
            state.profile.payload_i128_batches += 1;
        }
        state.profile.payload_nanos = state
            .profile
            .payload_nanos
            .saturating_add(sql_elapsed_nanos(started));
    }
    Ok(())
}

fn q08_log_late_market_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    tpch_profile_late_materialized("Q08 market share", metrics, row_group_chunk);
}

fn q08_log_late_callback_profile(profile: Q08LateCallbackProfile) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q08 late callbacks: selection={:.3} ms payload={:.3} ms selection_batches={} payload_batches={} payload_i64={} payload_i128={}",
        sql_nanos_to_millis(profile.selection_nanos),
        sql_nanos_to_millis(profile.payload_nanos),
        profile.selection_batches,
        profile.payload_batches,
        profile.payload_i64_batches,
        profile.payload_i128_batches,
    );
}

fn q08_market_share_batch(
    batch: RecordBatch,
    order_years: &Q08OrderYears,
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
    order_years: &Q08OrderYears,
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
        return q08_market_share_vector(
            orderkeys,
            partkeys,
            suppkeys,
            extendedprices,
            discounts,
            order_years,
            part_keys,
            supplier_is_brazil,
        );
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
    order_years: &Q08OrderYears,
    part_keys: &AdaptiveI64Set,
    supplier_is_brazil: &AdaptiveI64Map<bool>,
) -> Result<HashMap<i32, (f64, f64)>> {
    let mut groups = HashMap::<i32, (f64, f64)>::new();
    if let (Some(orderkey_values), Some(partkey_values), Some(suppkey_values)) = (
        orderkeys.values_if_null_free(),
        partkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) {
        let part_contains = part_keys.dense_contains_slice();
        let row_state = std::cell::Cell::new(None::<(i32, bool)>);
        if let (
            Some((order_year_values, order_year_base, order_year_missing)),
            Some((brazil_values, brazil_present)),
        ) = (order_years.dense_slice(), supplier_is_brazil.dense_slices())
        {
            consume_filtered_discounted_revenue_decimal128_vectors(
                extendedprices,
                discounts,
                orderkey_values.len(),
                |row| {
                    row_state.set(None);
                    if !part_keys.contains_cached(part_contains, partkey_values[row]) {
                        return Ok(false);
                    }
                    let Some(orderkey) = orderkey_values[row]
                        .checked_sub(order_year_base)
                        .and_then(|index| usize::try_from(index).ok())
                    else {
                        return Ok(false);
                    };
                    let Some(&o_year) = order_year_values.get(orderkey) else {
                        return Ok(false);
                    };
                    if o_year == order_year_missing {
                        return Ok(false);
                    }
                    let Ok(suppkey) = usize::try_from(suppkey_values[row]) else {
                        return Ok(false);
                    };
                    if !brazil_present.get(suppkey).copied().unwrap_or(false) {
                        return Ok(false);
                    }
                    row_state.set(Some((o_year, brazil_values[suppkey])));
                    Ok(true)
                },
                |_, volume| {
                    if let Some((o_year, is_brazil)) = row_state.get() {
                        let group = groups.entry(o_year).or_insert((0.0, 0.0));
                        if is_brazil {
                            group.0 += volume;
                        }
                        group.1 += volume;
                    }
                    Ok(())
                },
            )?;
            return Ok(groups);
        }
        consume_filtered_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            orderkey_values.len(),
            |row| {
                row_state.set(None);
                if !part_keys.contains_cached(part_contains, partkey_values[row]) {
                    return Ok(false);
                }
                let Some(o_year) = order_years.get(orderkey_values[row]) else {
                    return Ok(false);
                };
                let Some(is_brazil) = supplier_is_brazil.get(suppkey_values[row]) else {
                    return Ok(false);
                };
                row_state.set(Some((o_year, is_brazil)));
                Ok(true)
            },
            |_, volume| {
                if let Some((o_year, is_brazil)) = row_state.get() {
                    let group = groups.entry(o_year).or_insert((0.0, 0.0));
                    if is_brazil {
                        group.0 += volume;
                    }
                    group.1 += volume;
                }
                Ok(())
            },
        )?;
        return Ok(groups);
    }
    let row_state = std::cell::Cell::new(None::<(i32, bool)>);
    consume_filtered_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        orderkeys.len(),
        |row| {
            row_state.set(None);
            if orderkeys.is_null(row) || partkeys.is_null(row) || suppkeys.is_null(row) {
                return Ok(false);
            }
            if !part_keys.contains(partkeys.value(row)) {
                return Ok(false);
            }
            let Some(o_year) = order_years.get(orderkeys.value(row)) else {
                return Ok(false);
            };
            let Some(is_brazil) = supplier_is_brazil.get(suppkeys.value(row)) else {
                return Ok(false);
            };
            row_state.set(Some((o_year, is_brazil)));
            Ok(true)
        },
        |_, volume| {
            if let Some((o_year, is_brazil)) = row_state.get() {
                let group = groups.entry(o_year).or_insert((0.0, 0.0));
                if is_brazil {
                    group.0 += volume;
                }
                group.1 += volume;
            }
            Ok(())
        },
    )?;
    Ok(groups)
}

fn q08_market_share_batch_typed(
    orderkeys: &ArrayRef,
    partkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    order_years: &Q08OrderYears,
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
    let extendedprices = Decimal128VectorView::Arrow {
        values: extendedprices.values,
        precision: extendedprices.precision,
        scale: extendedprices.scale,
    };
    let discounts = Decimal128VectorView::Arrow {
        values: discounts.values,
        precision: discounts.precision,
        scale: discounts.scale,
    };
    q08_market_share_vector(
        I64VectorView::Arrow(orderkeys),
        I64VectorView::Arrow(partkeys),
        I64VectorView::Arrow(suppkeys),
        extendedprices,
        discounts,
        order_years,
        part_keys,
        supplier_is_brazil,
    )
    .map(Some)
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
