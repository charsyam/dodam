use super::*;

fn regional_supplier_revenue_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 6
        && select.projection.len() == 2
        && projection.contains("n_name")
        && projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && group_by.contains("n_name")
        && order_by.contains("revenue desc")
        && selection.contains("c_custkey = o_custkey")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("l_suppkey = s_suppkey")
        && selection.contains("c_nationkey = s_nationkey")
        && selection.contains("s_nationkey = n_nationkey")
        && selection.contains("n_regionkey = r_regionkey")
        && selection.contains("r_name")
        && selection.contains("o_orderdate")
}

pub(super) async fn try_execute_regional_supplier_revenue_sql(
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
    if !regional_supplier_revenue_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 6 {
        return Ok(None);
    }
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    let mut supplier = None;
    let mut nation = None;
    let mut region = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        } else if alias.eq_ignore_ascii_case("region") {
            region = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem), Some(supplier), Some(nation), Some(region)) =
        (customer, orders, lineitem, supplier, nation, region)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(region_name) = string_equality_literal(&conjuncts, "r_name")? else {
        return Ok(None);
    };
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };

    let region_keys = region_keys_by_name(engine, region.path, batch_size, &region_name).await?;
    if region_keys.is_empty() {
        return Ok(Some(regional_supplier_revenue_output(Vec::new())?));
    }
    let nation_names =
        nation_names_by_region_keys(engine, nation.path, batch_size, &region_keys).await?;
    if nation_names.is_empty() {
        return Ok(Some(regional_supplier_revenue_output(Vec::new())?));
    }
    let customer_nations =
        customer_nations_in_region(engine, customer.path, batch_size, &nation_names).await?;
    if customer_nations.is_empty() {
        return Ok(Some(regional_supplier_revenue_output(Vec::new())?));
    }
    let supplier_nations =
        supplier_nations_in_region(engine, supplier.path, batch_size, &nation_names).await?;
    if supplier_nations.is_empty() {
        return Ok(Some(regional_supplier_revenue_output(Vec::new())?));
    }
    let order_customer_nations = order_customer_nations(
        engine,
        orders.path,
        batch_size,
        &customer_nations,
        start_days,
        end_days,
    )
    .await?;
    if order_customer_nations.is_empty() {
        return Ok(Some(regional_supplier_revenue_output(Vec::new())?));
    }
    let rows = revenue_by_nation(
        engine,
        lineitem.path,
        batch_size,
        &order_customer_nations,
        &supplier_nations,
        &nation_names,
    )
    .await?;
    Ok(Some(regional_supplier_revenue_output(rows)?))
}

pub(super) async fn region_keys_by_name(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    region_name: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["r_regionkey".to_string(), "r_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let regionkeys = batch_column(&batch, "r_regionkey")?;
        let names = batch_string_column(&batch, "r_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row) == region_name
                && let Some(regionkey) = numeric_i64_value(regionkeys, row)?
            {
                keys.insert(regionkey);
            }
        }
    }
    Ok(keys)
}

pub(super) async fn nation_names_by_region_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    region_keys: &HashSet<i64>,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "n_nationkey".to_string(),
                "n_regionkey".to_string(),
                "n_name".to_string(),
            ]),
            None,
        )
        .await?;
    let mut nations = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let regionkeys = batch_column(&batch, "n_regionkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        for row in 0..batch.num_rows() {
            let (Some(nationkey), Some(regionkey)) = (
                numeric_i64_value(nationkeys, row)?,
                numeric_i64_value(regionkeys, row)?,
            ) else {
                continue;
            };
            if region_keys.contains(&regionkey) && names.is_valid(row) {
                nations.insert(nationkey, names.value(row).to_string());
            }
        }
    }
    Ok(nations)
}

async fn customer_nations_in_region(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<FastHashMap<i64, i64>> {
    let nation_keys = AdaptiveI64Set::from_hash(nation_names.keys().copied().collect());
    collect_i64_by_i64_set_hash_map(
        engine,
        path,
        batch_size,
        "c_custkey",
        "c_nationkey",
        &nation_keys,
    )
    .await
}

async fn supplier_nations_in_region(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<FastHashMap<i64, i64>> {
    let nation_keys = AdaptiveI64Set::from_hash(nation_names.keys().copied().collect());
    collect_i64_by_i64_set_hash_map(
        engine,
        path,
        batch_size,
        "s_suppkey",
        "s_nationkey",
        &nation_keys,
    )
    .await
}

async fn order_customer_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<i64, i64>> {
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
    let orders = parallel_batch_collect_pairs_view_chunks(
        &mut stream,
        4,
        move |view, orders| {
            orders.extend(order_customer_nations_view(
                view,
                &customer_nations,
                start_days,
                end_days,
            )?);
            Ok(Some(()))
        },
        "regional order customer nations",
    )?;
    Ok(fast_hash_map_from_pairs_profiled(
        orders,
        "regional order customer nations build",
    ))
}

fn order_customer_nations_view(
    view: BatchView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Vec<(i64, i64)>> {
    let dense_nations = customer_nations.dense_word_slices();
    collect_i64_i64_date32_pairs_view(
        view,
        "regional order customer nation raw vector columns have unsupported types",
        |_, custkey, orderdate| {
            if orderdate < start_days || orderdate >= end_days {
                return Ok(None);
            }
            Ok(customer_nations.get_cached_words(dense_nations, custkey))
        },
    )
}

struct RegionalSupplierRevenueRow {
    n_name: String,
    revenue: f64,
}

async fn revenue_by_nation(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &FastHashMap<i64, i64>,
    nation_names: &HashMap<i64, String>,
) -> Result<Vec<RegionalSupplierRevenueRow>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let pruning_predicates = if let (Some(min_key), Some(max_key)) = (
        order_customer_nations.keys().copied().min(),
        order_customer_nations.keys().copied().max(),
    ) {
        i64_range_pruning_predicates("l_orderkey", min_key, max_key)
    } else {
        Vec::new()
    };
    let order_customer_nations = Arc::new(order_customer_nations.clone());
    let supplier_nations = Arc::new(AdaptiveI64Map::from_hash(supplier_nations.clone()));
    let groups = if let Some(groups) = revenue_by_nation_late_materialized(
        engine,
        path.clone(),
        batch_size,
        pruning_predicates.clone(),
        order_customer_nations.clone(),
        supplier_nations.clone(),
    )
    .await?
    {
        groups
    } else {
        let order_customer_nations_for_scan = order_customer_nations.clone();
        let supplier_nations_for_scan = supplier_nations.clone();
        if let Some(partials) = engine
            .parquet_row_group_map_pruned_view(
                path.clone(),
                batch_size,
                projection.clone(),
                pruning_predicates.clone(),
                regional_revenue_row_group_map_chunk(),
                fast_hash_map::<i64, f64>,
                move |view, groups| {
                    merge_f64_groups(
                        groups,
                        revenue_by_nation_projected_view(
                            view,
                            &order_customer_nations_for_scan,
                            &supplier_nations_for_scan,
                        )?,
                    );
                    Ok(Some(()))
                },
                |groups| Ok(Some(groups)),
            )
            .await?
        {
            let mut groups = fast_hash_map::<i64, f64>();
            for partial in partials {
                merge_f64_groups(&mut groups, partial);
            }
            groups
        } else {
            revenue_by_nation_stream(
                engine,
                path,
                batch_size,
                projection,
                pruning_predicates,
                order_customer_nations,
                supplier_nations,
            )
            .await?
        }
    };
    let mut rows = groups
        .into_iter()
        .filter_map(|(nationkey, revenue)| {
            nation_names
                .get(&nationkey)
                .map(|n_name| RegionalSupplierRevenueRow {
                    n_name: n_name.clone(),
                    revenue,
                })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .revenue
            .partial_cmp(&left.revenue)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

async fn revenue_by_nation_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    pruning_predicates: Vec<Expr>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
) -> Result<Option<FastHashMap<i64, f64>>> {
    let predicate_projection =
        Projection::Columns(vec!["l_orderkey".to_string(), "l_suppkey".to_string()]);
    let payload_projection = Projection::Columns(vec![
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let policy = generic_late_materialization_policy_for_projection(
        &predicate_projection,
        &payload_projection,
        0.35,
        Some(0.60),
    );
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            pruning_predicates,
            regional_revenue_late_materialized_row_group_chunk(),
            policy,
            {
                let order_customer_nations = order_customer_nations.clone();
                let supplier_nations = supplier_nations.clone();
                move || RegionalRevenueLateState {
                    order_customer_nations: order_customer_nations.clone(),
                    supplier_nations: supplier_nations.clone(),
                    selected_nations: Vec::new(),
                    selected_offsets: Vec::new(),
                    payload_offset: 0,
                    groups: fast_hash_map(),
                }
            },
            regional_revenue_late_build_selection_view,
            regional_revenue_late_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_nations.len() {
                    return Err(DodamError::UnsupportedSql(
                        "regional revenue late payload row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.groups, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut groups = fast_hash_map::<i64, f64>();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_groups, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        merge_f64_groups(&mut groups, chunk_groups);
    }
    regional_revenue_log_late_materialized_profile(
        metrics,
        regional_revenue_late_materialized_row_group_chunk(),
    );
    Ok(Some(groups))
}

fn regional_revenue_late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(16)
}

struct RegionalRevenueLateState {
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    selected_nations: Vec<Option<i64>>,
    selected_offsets: Vec<u32>,
    payload_offset: usize,
    groups: FastHashMap<i64, f64>,
}

fn regional_revenue_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut RegionalRevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(suppkeys)) = (view.i64_vector(0), view.i64_vector(1))
    {
        if let (Some(orderkey_values), Some(suppkey_values)) = (
            orderkeys.values_if_null_free(),
            suppkeys.values_if_null_free(),
        ) {
            regional_revenue_push_late_selection_slices(
                selection,
                orderkey_values,
                suppkey_values,
                state,
            );
            return Ok(Some(()));
        }
        for row in 0..orderkeys.len() {
            let selected = if orderkeys.is_null(row) || suppkeys.is_null(row) {
                None
            } else {
                regional_revenue_late_selected_nation(
                    orderkeys.value(row),
                    suppkeys.value(row),
                    state,
                )
            };
            if let Some(nation) = selected {
                selection.push(true);
                state.selected_nations.push(Some(nation));
            } else {
                selection.push(false);
            }
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    regional_revenue_late_build_selection_batch(batch.clone(), selection, state)
}

fn regional_revenue_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut RegionalRevenueLateState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    if let (Some(orderkeys), Some(suppkeys)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
    ) && orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        regional_revenue_push_late_selection_slices(
            selection,
            orderkey_values,
            suppkey_values,
            state,
        );
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let selected = match (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) {
            (Some(orderkey), Some(suppkey)) => {
                regional_revenue_late_selected_nation(orderkey, suppkey, state)
            }
            _ => None,
        };
        if let Some(nation) = selected {
            selection.push(true);
            state.selected_nations.push(Some(nation));
        } else {
            selection.push(false);
        }
    }
    Ok(Some(()))
}

fn regional_revenue_late_selected_nation(
    orderkey: i64,
    suppkey: i64,
    state: &RegionalRevenueLateState,
) -> Option<i64> {
    let supplier_nation = state.supplier_nations.get(suppkey)?;
    let nation = *state.order_customer_nations.get(&orderkey)?;
    (nation == supplier_nation).then_some(nation)
}

fn regional_revenue_late_selected_nation_cached(
    orderkey: i64,
    suppkey: i64,
    state: &RegionalRevenueLateState,
    supplier_words: Option<(&[i64], &[u64])>,
) -> Option<i64> {
    let supplier_nation = state
        .supplier_nations
        .get_cached_words(supplier_words, suppkey)?;
    let nation = *state.order_customer_nations.get(&orderkey)?;
    (nation == supplier_nation).then_some(nation)
}

fn regional_revenue_push_late_selection_slices(
    selection: &mut LateSelectionBuilder,
    orderkeys: &[i64],
    suppkeys: &[i64],
    state: &mut RegionalRevenueLateState,
) {
    state.selected_offsets.clear();
    state.selected_offsets.reserve(orderkeys.len().min(1024));
    let supplier_words = state.supplier_nations.dense_word_slices();
    let max_gap = late_materialization_coalesce_max_gap(8);
    for row in 0..orderkeys.len() {
        if regional_revenue_late_selected_nation_cached(
            orderkeys[row],
            suppkeys[row],
            state,
            supplier_words,
        )
        .is_some()
        {
            state.selected_offsets.push(row as u32);
        }
    }
    let order_customer_nations = state.order_customer_nations.clone();
    let supplier_nations = state.supplier_nations.clone();
    selection.push_selected_u32_offsets_coalesced(
        orderkeys.len(),
        &state.selected_offsets,
        max_gap,
        |row| {
            let nation = row.and_then(|row| {
                let supplier_nation = supplier_nations.get(suppkeys[row])?;
                let nation = *order_customer_nations.get(&orderkeys[row])?;
                (nation == supplier_nation).then_some(nation)
            });
            state.selected_nations.push(nation);
        },
    );
}

fn regional_revenue_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut RegionalRevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(extendedprices), Some(discounts)) =
            (view.decimal128_vector(0), view.decimal128_vector(1))
    {
        regional_revenue_late_consume_payload_vector(
            extendedprices,
            discounts,
            view.num_rows(),
            state,
        )?;
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    regional_revenue_late_consume_payload_batch(batch.clone(), state)
}

fn regional_revenue_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut RegionalRevenueLateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let (Some(extendedprices), Some(discounts)) =
        (decimal_input(extendedprices)?, decimal_input(discounts)?)
    {
        regional_revenue_late_consume_payload_vector(
            Decimal128VectorView::Arrow {
                values: extendedprices.values,
                precision: extendedprices.precision,
                scale: extendedprices.scale,
            },
            Decimal128VectorView::Arrow {
                values: discounts.values,
                precision: discounts.precision,
                scale: discounts.scale,
            },
            batch.num_rows(),
            state,
        )?;
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let Some(nation) = state.selected_nations.get(state.payload_offset).copied() else {
            return Err(DodamError::UnsupportedSql(
                "regional revenue late payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        let Some(nation) = nation else {
            continue;
        };
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *state.groups.entry(nation).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(Some(()))
}

fn regional_revenue_late_consume_payload_vector(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
    state: &mut RegionalRevenueLateState,
) -> Result<()> {
    consume_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        row_count,
        |_, revenue| {
            let Some(nation) = state.selected_nations.get(state.payload_offset).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "regional revenue late payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            let Some(nation) = nation else {
                return Ok(());
            };
            if let Some(revenue) = revenue {
                *state.groups.entry(nation).or_insert(0.0) += revenue;
            }
            Ok(())
        },
    )
}

fn regional_revenue_log_late_materialized_profile(
    metrics: LateMaterializedMetrics,
    row_group_chunk: usize,
) {
    tpch_profile_late_materialized("regional revenue", metrics, row_group_chunk);
}

async fn revenue_by_nation_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
) -> Result<FastHashMap<i64, f64>> {
    let stream = if pruning_predicates.is_empty() {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    } else {
        engine
            .scan_parquet_batches_pruned(path, batch_size, projection, pruning_predicates)
            .await?
    };
    revenue_by_nation_stream_from_batches(
        stream,
        order_customer_nations,
        supplier_nations,
        "regional revenue aggregate",
    )
}

fn revenue_by_nation_stream_from_batches(
    mut stream: SendableBatchStream,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    label: &str,
) -> Result<FastHashMap<i64, f64>> {
    parallel_batch_fold_view_chunks(
        &mut stream,
        join_aggregate_chunk_size(),
        || fast_hash_map::<i64, f64>(),
        move |view, groups| {
            merge_f64_groups(
                groups,
                revenue_by_nation_projected_view(view, &order_customer_nations, &supplier_nations)?,
            );
            Ok(Some(()))
        },
        Ok,
        fast_hash_map::<i64, f64>(),
        merge_f64_groups,
        label,
    )
}

fn regional_revenue_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(2)
}

fn revenue_by_nation_batch(
    batch: RecordBatch,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<FastHashMap<i64, f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = revenue_by_nation_typed(
        orderkeys,
        suppkeys,
        extendedprices,
        discounts,
        order_customer_nations,
        supplier_nations,
    )? {
        return Ok(groups);
    }
    let mut groups = fast_hash_map::<i64, f64>();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        let Some(supplier_nation) = supplier_nations.get(suppkey) else {
            continue;
        };
        let Some(customer_nation) = order_customer_nations.get(&orderkey).copied() else {
            continue;
        };
        if customer_nation != supplier_nation {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *groups.entry(customer_nation).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(groups)
}

fn revenue_by_nation_projected_view(
    view: BatchView<'_>,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<FastHashMap<i64, f64>> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(suppkeys), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        )
    {
        return revenue_by_nation_vector(
            orderkeys,
            suppkeys,
            extendedprices,
            discounts,
            order_customer_nations,
            supplier_nations,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "regional revenue raw vector columns have unsupported types".to_string(),
        ));
    };
    revenue_by_nation_batch(batch.clone(), order_customer_nations, supplier_nations)
}

fn revenue_by_nation_vector(
    orderkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<FastHashMap<i64, f64>> {
    let mut groups = fast_hash_map::<i64, f64>();
    revenue_by_nation_vector_into(
        orderkeys,
        suppkeys,
        extendedprices,
        discounts,
        order_customer_nations,
        supplier_nations,
        &mut groups,
    )?;
    Ok(groups)
}

fn revenue_by_nation_vector_into(
    orderkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
    groups: &mut FastHashMap<i64, f64>,
) -> Result<()> {
    if let (Some(orderkey_values), Some(suppkey_values)) = (
        orderkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
    ) {
        if let Some((supplier_nation_values, supplier_nation_present)) =
            supplier_nations.dense_slices()
        {
            consume_filtered_discounted_revenue_decimal128_vectors_with_payload(
                extendedprices,
                discounts,
                orderkey_values.len(),
                |row| {
                    let Ok(suppkey) = usize::try_from(suppkey_values[row]) else {
                        return Ok(None);
                    };
                    if supplier_nation_present
                        .get(suppkey)
                        .copied()
                        .unwrap_or(false)
                        && let Some(nation) =
                            order_customer_nations.get(&orderkey_values[row]).copied()
                        && supplier_nation_values[suppkey] == nation
                    {
                        return Ok(Some(nation));
                    }
                    Ok(None)
                },
                |_, nation, revenue| {
                    *groups.entry(nation).or_insert(0.0) += revenue;
                    Ok(())
                },
            )?;
            return Ok(());
        }
        consume_filtered_discounted_revenue_decimal128_vectors_with_payload(
            extendedprices,
            discounts,
            orderkey_values.len(),
            |row| {
                let Some(supplier_nation) = supplier_nations.get(suppkey_values[row]) else {
                    return Ok(None);
                };
                let Some(nation) = order_customer_nations.get(&orderkey_values[row]).copied()
                else {
                    return Ok(None);
                };
                if nation == supplier_nation {
                    return Ok(Some(nation));
                }
                Ok(None)
            },
            |_, nation, revenue| {
                *groups.entry(nation).or_insert(0.0) += revenue;
                Ok(())
            },
        )?;
        return Ok(());
    }
    consume_filtered_discounted_revenue_decimal128_vectors_with_payload(
        extendedprices,
        discounts,
        orderkeys.len(),
        |row| {
            if orderkeys.is_null(row) || suppkeys.is_null(row) {
                return Ok(None);
            }
            let Some(supplier_nation) = supplier_nations.get(suppkeys.value(row)) else {
                return Ok(None);
            };
            let Some(nation) = order_customer_nations.get(&orderkeys.value(row)).copied() else {
                return Ok(None);
            };
            if nation == supplier_nation {
                return Ok(Some(nation));
            }
            Ok(None)
        },
        |_, nation, revenue| {
            *groups.entry(nation).or_insert(0.0) += revenue;
            Ok(())
        },
    )?;
    Ok(())
}

fn revenue_by_nation_typed(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<Option<FastHashMap<i64, f64>>> {
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
    revenue_by_nation_vector(
        I64VectorView::Arrow(orderkeys),
        I64VectorView::Arrow(suppkeys),
        extendedprices,
        discounts,
        order_customer_nations,
        supplier_nations,
    )
    .map(Some)
}

fn regional_supplier_revenue_output(rows: Vec<RegionalSupplierRevenueRow>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("n_name", DataType::Utf8, false),
            Field::new("revenue", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.n_name.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
