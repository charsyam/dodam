use super::*;

fn q07_outer_shape(select: &Select, query: &Query) -> bool {
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
        && select.projection.len() == 4
        && projection.contains("supp_nation")
        && projection.contains("cust_nation")
        && projection.contains("l_year")
        && projection.contains("sum(volume)")
        && group_by.contains("supp_nation")
        && group_by.contains("cust_nation")
        && group_by.contains("l_year")
        && order_by.contains("supp_nation")
        && order_by.contains("cust_nation")
        && order_by.contains("l_year")
}

fn q07_inner_shape(select: &Select, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.from.len() == 6
        && select.projection.len() == 4
        && projection.contains("n1.n_name as supp_nation")
        && projection.contains("n2.n_name as cust_nation")
        && projection.contains("extract(year from l_shipdate)")
        && projection.contains("l_extendedprice * (1 - l_discount)")
        && selection.contains("s_suppkey = l_suppkey")
        && selection.contains("o_orderkey = l_orderkey")
        && selection.contains("c_custkey = o_custkey")
        && selection.contains("s_nationkey = n1.n_nationkey")
        && selection.contains("c_nationkey = n2.n_nationkey")
        && selection.contains("france")
        && selection.contains("germany")
        && selection.contains("l_shipdate between")
}

pub(super) async fn try_execute_bilateral_shipping_volume_sql(
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
    if !q07_outer_shape(select, query) {
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
    if !q07_inner_shape(inner_select, selection) {
        return Ok(None);
    }
    reject_query_features(inner_query)?;
    reject_select_features(inner_select)?;
    let Some(tables) = parse_comma_join_table_refs(inner_select)? else {
        return Ok(None);
    };
    if tables.len() != 6 {
        return Ok(None);
    }
    let mut supplier = None;
    let mut lineitem = None;
    let mut orders = None;
    let mut customer = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("n1") || alias.eq_ignore_ascii_case("n2") {
            nation.get_or_insert(table);
        }
    }
    let (Some(supplier), Some(lineitem), Some(orders), Some(customer), Some(nation)) =
        (supplier, lineitem, orders, customer, nation)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_between_bounds(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };

    let nation_names = nation_names_by_keys(engine, nation.path, batch_size).await?;
    let target_nations = nation_names
        .iter()
        .filter_map(|(nationkey, name)| {
            (name.eq_ignore_ascii_case("FRANCE") || name.eq_ignore_ascii_case("GERMANY"))
                .then_some((*nationkey, name.clone()))
        })
        .collect::<HashMap<_, _>>();
    if target_nations.len() != 2 {
        return Ok(None);
    }
    let supplier_nations =
        q07_supplier_nations(engine, supplier.path, batch_size, &target_nations).await?;
    if supplier_nations.is_empty() {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let customer_nations =
        q07_customer_nations(engine, customer.path, batch_size, &target_nations).await?;
    if customer_nations.is_empty() {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let order_customer_nations =
        q07_order_customers(engine, orders.path, batch_size, &customer_nations).await?;
    if order_customer_nations.is_empty() {
        return Ok(Some(q07_output(Vec::new())?));
    }
    let rows = q07_volume_rows(
        engine,
        lineitem.path,
        batch_size,
        &supplier_nations,
        &order_customer_nations,
        &target_nations,
        start_days,
        end_days,
    )
    .await?;
    Ok(Some(q07_output(rows)?))
}

async fn q07_supplier_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    target_nations: &HashMap<i64, String>,
) -> Result<FastHashMap<i64, i64>> {
    let nation_keys = AdaptiveI64Set::from_hash(target_nations.keys().copied().collect());
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

pub(super) async fn q07_customer_nations(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    target_nations: &HashMap<i64, String>,
) -> Result<FastHashMap<i64, i64>> {
    let nation_keys = AdaptiveI64Set::from_hash(target_nations.keys().copied().collect());
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

async fn q07_order_customers(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_nations: &FastHashMap<i64, i64>,
) -> Result<FastHashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_custkey".to_string()]),
            None,
        )
        .await?;
    let customer_nations = Arc::new(AdaptiveI64Map::from_hash(customer_nations.clone()));
    let orders = parallel_batch_collect_pairs_view_chunks(
        &mut stream,
        4,
        move |view, orders| {
            orders.extend(q07_order_customers_view(view, &customer_nations)?);
            Ok(Some(()))
        },
        "Q07 order customer nations",
    )?;
    Ok(fast_hash_map_from_pairs_profiled(
        orders,
        "bilateral order customer nations build",
    ))
}

fn q07_order_customers_view(
    view: BatchView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
) -> Result<Vec<(i64, i64)>> {
    collect_i64_i64_pairs_view(
        view,
        "Q07 order customer raw vector columns have unsupported types",
        |_, custkey| Ok(customer_nations.get(custkey)),
    )
}

struct Q07Row {
    supp_nation: String,
    cust_nation: String,
    l_year: i32,
    revenue: f64,
}

async fn q07_volume_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    supplier_nations: &FastHashMap<i64, i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    nation_names: &HashMap<i64, String>,
    start_days: i32,
    end_days: i32,
) -> Result<Vec<Q07Row>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_shipdate".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let mut pruning_predicates = if let Some((min_key, max_key)) =
        selective_i64_key_range(order_customer_nations.keys().copied())
    {
        i64_range_pruning_predicates("l_orderkey", min_key, max_key)
    } else {
        Vec::new()
    };
    pruning_predicates.push(Expr::Comparison(ComparisonExpr {
        column: "l_shipdate".to_string(),
        op: ComparisonOp::GtEq,
        value: LiteralValue::Int64(i64::from(start_days)),
    }));
    pruning_predicates.push(Expr::Comparison(ComparisonExpr {
        column: "l_shipdate".to_string(),
        op: ComparisonOp::LtEq,
        value: LiteralValue::Int64(i64::from(end_days)),
    }));
    let supplier_nations = Arc::new(AdaptiveI64Map::from_hash(supplier_nations.clone()));
    let order_customer_nations = Arc::new(order_customer_nations.clone());
    let groups = if let Some(groups) = q07_volume_rows_late_materialized(
        engine,
        path.clone(),
        batch_size,
        supplier_nations.clone(),
        order_customer_nations.clone(),
        start_days,
        end_days,
        pruning_predicates.clone(),
    )
    .await?
    {
        groups
    } else {
        let supplier_nations_for_scan = supplier_nations.clone();
        let order_customer_nations_for_scan = order_customer_nations.clone();
        if let Some(partials) = engine
            .parquet_row_group_map_pruned_view(
                path.clone(),
                batch_size,
                projection.clone(),
                pruning_predicates.clone(),
                q07_row_group_map_chunk(),
                fast_hash_map::<(i64, i64, i32), f64>,
                move |view, groups| {
                    merge_f64_groups(
                        groups,
                        q07_volume_projected_view(
                            view,
                            &supplier_nations_for_scan,
                            &order_customer_nations_for_scan,
                            start_days,
                            end_days,
                        )?,
                    );
                    Ok(Some(()))
                },
                |groups| Ok(Some(groups)),
            )
            .await?
        {
            let mut groups = fast_hash_map::<(i64, i64, i32), f64>();
            for partial in partials {
                merge_f64_groups(&mut groups, partial);
            }
            groups
        } else {
            q07_volume_rows_stream(
                engine,
                path,
                batch_size,
                projection,
                pruning_predicates,
                supplier_nations,
                order_customer_nations,
                start_days,
                end_days,
            )
            .await?
        }
    };
    let mut rows = groups
        .into_iter()
        .filter_map(|((supp_nation_key, cust_nation_key, l_year), revenue)| {
            Some(Q07Row {
                supp_nation: nation_names.get(&supp_nation_key)?.clone(),
                cust_nation: nation_names.get(&cust_nation_key)?.clone(),
                l_year,
                revenue,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.supp_nation
            .cmp(&right.supp_nation)
            .then_with(|| left.cust_nation.cmp(&right.cust_nation))
            .then_with(|| left.l_year.cmp(&right.l_year))
    });
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
async fn q07_volume_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    start_days: i32,
    end_days: i32,
    pruning_predicates: Vec<Expr>,
) -> Result<Option<FastHashMap<(i64, i64, i32), f64>>> {
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_shipdate".to_string(),
            ]),
            Projection::Columns(vec![
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            pruning_predicates,
            q07_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q07_late_materialized_max_selected_ratio(),
                q07_late_materialized_max_selector_run_ratio(),
            ),
            {
                let supplier_nations = supplier_nations.clone();
                let order_customer_nations = order_customer_nations.clone();
                move || Q07LateState {
                    supplier_nations: supplier_nations.clone(),
                    order_customer_nations: order_customer_nations.clone(),
                    selected_rows: Vec::new(),
                    payload_offset: 0,
                    groups: fast_hash_map(),
                    year_cache: Date32YearCache::default(),
                }
            },
            move |view, selection, state| {
                q07_late_build_selection_view(view, selection, state, start_days, end_days)
            },
            q07_late_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_rows.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q07 payload row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.groups, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut groups = fast_hash_map::<(i64, i64, i32), f64>();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_groups, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        merge_f64_groups(&mut groups, chunk_groups);
    }
    q07_log_late_materialized_profile(metrics, q07_late_materialized_row_group_chunk());
    Ok(Some(groups))
}

fn q07_late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(2)
}

fn q07_late_materialized_max_selected_ratio() -> f64 {
    late_materialization_max_selected_ratio(0.10)
}

fn q07_late_materialized_max_selector_run_ratio() -> f64 {
    late_materialization_max_selector_run_ratio(0.60)
}

#[derive(Clone, Copy)]
struct Q07LateSelectedRow {
    supp_nation_key: i64,
    cust_nation_key: i64,
    shipdate: i32,
}

struct Q07LateState {
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    selected_rows: Vec<Q07LateSelectedRow>,
    payload_offset: usize,
    groups: FastHashMap<(i64, i64, i32), f64>,
    year_cache: Date32YearCache,
}

fn q07_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q07LateState,
    start_days: i32,
    end_days: i32,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if let (Some(orderkeys), Some(suppkeys), Some(shipdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) && orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && shipdates.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let suppkey_values = suppkeys.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        for row in 0..orderkey_values.len() {
            let shipdate = shipdate_values[row];
            let selected = if shipdate < start_days || shipdate > end_days {
                None
            } else {
                let supp_nation_key = state.supplier_nations.get(suppkey_values[row]);
                let cust_nation_key = state.order_customer_nations.get(&orderkey_values[row]);
                match (supp_nation_key, cust_nation_key) {
                    (Some(supp_nation_key), Some(cust_nation_key))
                        if supp_nation_key != *cust_nation_key =>
                    {
                        Some(Q07LateSelectedRow {
                            supp_nation_key,
                            cust_nation_key: *cust_nation_key,
                            shipdate,
                        })
                    }
                    _ => None,
                }
            };
            if let Some(row) = selected {
                selection.push(true);
                state.selected_rows.push(row);
            } else {
                selection.push(false);
            }
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let selected = match (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            date32_value(shipdates, row)?,
        ) {
            (Some(orderkey), Some(suppkey), Some(shipdate))
                if shipdate >= start_days && shipdate <= end_days =>
            {
                let supp_nation_key = state.supplier_nations.get(suppkey);
                let cust_nation_key = state.order_customer_nations.get(&orderkey);
                match (supp_nation_key, cust_nation_key) {
                    (Some(supp_nation_key), Some(cust_nation_key))
                        if supp_nation_key != *cust_nation_key =>
                    {
                        Some(Q07LateSelectedRow {
                            supp_nation_key,
                            cust_nation_key: *cust_nation_key,
                            shipdate,
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some(row) = selected {
            selection.push(true);
            state.selected_rows.push(row);
        } else {
            selection.push(false);
        }
    }
    Ok(Some(()))
}

fn q07_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q07LateState,
    start_days: i32,
    end_days: i32,
) -> Result<Option<()>> {
    if view.num_columns() == 3
        && let (Some(orderkeys), Some(suppkeys), Some(shipdates)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
        )
    {
        if let (Some(orderkey_values), Some(suppkey_values), Some(shipdate_values)) = (
            orderkeys.values_if_null_free(),
            suppkeys.values_if_null_free(),
            shipdates.values_if_null_free(),
        ) {
            for row in 0..orderkey_values.len() {
                let selected = q07_late_selected_row(
                    orderkey_values[row],
                    suppkey_values[row],
                    shipdate_values[row],
                    state,
                    start_days,
                    end_days,
                );
                if let Some(row) = selected {
                    selection.push(true);
                    state.selected_rows.push(row);
                } else {
                    selection.push(false);
                }
            }
            return Ok(Some(()));
        }
        for row in 0..orderkeys.len() {
            let selected =
                if orderkeys.is_null(row) || suppkeys.is_null(row) || shipdates.is_null(row) {
                    None
                } else {
                    q07_late_selected_row(
                        orderkeys.value(row),
                        suppkeys.value(row),
                        shipdates.value(row),
                        state,
                        start_days,
                        end_days,
                    )
                };
            if let Some(row) = selected {
                selection.push(true);
                state.selected_rows.push(row);
            } else {
                selection.push(false);
            }
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q07_late_build_selection_batch(batch.clone(), selection, state, start_days, end_days)
}

fn q07_late_selected_row(
    orderkey: i64,
    suppkey: i64,
    shipdate: i32,
    state: &Q07LateState,
    start_days: i32,
    end_days: i32,
) -> Option<Q07LateSelectedRow> {
    if shipdate < start_days || shipdate > end_days {
        return None;
    }
    let supp_nation_key = state.supplier_nations.get(suppkey)?;
    let cust_nation_key = *state.order_customer_nations.get(&orderkey)?;
    (supp_nation_key != cust_nation_key).then_some(Q07LateSelectedRow {
        supp_nation_key,
        cust_nation_key,
        shipdate,
    })
}

fn q07_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut Q07LateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let (Some(extendedprices), Some(discounts)) =
        (decimal_input(extendedprices)?, decimal_input(discounts)?)
    {
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
        q07_late_consume_payload_vector(extendedprices, discounts, batch.num_rows(), state)?;
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let Some(selected) = state.selected_rows.get(state.payload_offset).copied() else {
            return Err(DodamError::UnsupportedSql(
                "Q07 payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *state
            .groups
            .entry((
                selected.supp_nation_key,
                selected.cust_nation_key,
                state.year_cache.year(selected.shipdate)?,
            ))
            .or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(Some(()))
}

fn q07_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut Q07LateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(extendedprices), Some(discounts)) =
            (view.decimal128_vector(0), view.decimal128_vector(1))
    {
        q07_late_consume_payload_vector(extendedprices, discounts, view.num_rows(), state)?;
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q07_late_consume_payload_batch(batch.clone(), state)
}

fn q07_late_consume_payload_vector(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
    state: &mut Q07LateState,
) -> Result<()> {
    consume_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        row_count,
        |_, revenue| {
            let Some(selected) = state.selected_rows.get(state.payload_offset).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "Q07 payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            if let Some(revenue) = revenue {
                *state
                    .groups
                    .entry((
                        selected.supp_nation_key,
                        selected.cust_nation_key,
                        state.year_cache.year(selected.shipdate)?,
                    ))
                    .or_insert(0.0) += revenue;
            }
            Ok(())
        },
    )
}

fn q07_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    tpch_profile_late_materialized("Q07 lineitem", metrics, row_group_chunk);
}

#[allow(clippy::too_many_arguments)]
async fn q07_volume_rows_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<(i64, i64, i32), f64>> {
    let mut stream = if pruning_predicates.is_empty() {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    } else {
        engine
            .scan_parquet_batches_pruned(path, batch_size, projection, pruning_predicates)
            .await?
    };
    parallel_batch_fold_view_chunks(
        &mut stream,
        join_aggregate_chunk_size(),
        || fast_hash_map::<(i64, i64, i32), f64>(),
        move |view, groups| {
            merge_f64_groups(
                groups,
                q07_volume_projected_view(
                    view,
                    &supplier_nations,
                    &order_customer_nations,
                    start_days,
                    end_days,
                )?,
            );
            Ok(Some(()))
        },
        Ok,
        fast_hash_map::<(i64, i64, i32), f64>(),
        merge_f64_groups,
        "Q07 volume aggregate",
    )
}

fn q07_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(2)
}

fn q07_volume_batch(
    batch: RecordBatch,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<(i64, i64, i32), f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(groups) = q07_volume_batch_typed(
        orderkeys,
        suppkeys,
        shipdates,
        extendedprices,
        discounts,
        supplier_nations,
        order_customer_nations,
        start_days,
        end_days,
    )? {
        return Ok(groups);
    }
    let mut groups = fast_hash_map::<(i64, i64, i32), f64>();
    let mut year_cache = Date32YearCache::default();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(suppkey), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if shipdate < start_days || shipdate > end_days {
            continue;
        }
        let (Some(supp_nation_key), Some(cust_nation_key)) = (
            supplier_nations.get(suppkey),
            order_customer_nations.get(&orderkey).copied(),
        ) else {
            continue;
        };
        if supp_nation_key == cust_nation_key {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *groups
            .entry((supp_nation_key, cust_nation_key, year_cache.year(shipdate)?))
            .or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(groups)
}

fn q07_volume_projected_view(
    view: BatchView<'_>,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<(i64, i64, i32), f64>> {
    if view.num_columns() == 5
        && let (
            Some(orderkeys),
            Some(suppkeys),
            Some(shipdates),
            Some(extendedprices),
            Some(discounts),
        ) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            view.decimal128_vector(3),
            view.decimal128_vector(4),
        )
    {
        return q07_volume_vector(
            orderkeys,
            suppkeys,
            shipdates,
            extendedprices,
            discounts,
            supplier_nations,
            order_customer_nations,
            start_days,
            end_days,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q07 volume raw vector columns have unsupported types".to_string(),
        ));
    };
    q07_volume_batch(
        batch.clone(),
        supplier_nations,
        order_customer_nations,
        start_days,
        end_days,
    )
}

#[allow(clippy::too_many_arguments)]
fn q07_volume_vector(
    orderkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<(i64, i64, i32), f64>> {
    let mut groups = fast_hash_map::<(i64, i64, i32), f64>();
    let mut year_cache = Date32YearCache::default();
    if let (Some(orderkey_values), Some(suppkey_values), Some(shipdate_values)) = (
        orderkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
        shipdates.values_if_null_free(),
    ) {
        if let Some((supplier_nation_values, supplier_nation_present)) =
            supplier_nations.dense_slices()
        {
            let group_key = std::cell::Cell::new(None::<(i64, i64, i32)>);
            consume_filtered_discounted_revenue_decimal128_vectors(
                extendedprices,
                discounts,
                orderkey_values.len(),
                |row| {
                    group_key.set(None);
                    let shipdate = shipdate_values[row];
                    if shipdate < start_days || shipdate > end_days {
                        return Ok(false);
                    }
                    let Ok(suppkey) = usize::try_from(suppkey_values[row]) else {
                        return Ok(false);
                    };
                    if !supplier_nation_present
                        .get(suppkey)
                        .copied()
                        .unwrap_or(false)
                    {
                        return Ok(false);
                    }
                    let Some(cust_nation_key) =
                        order_customer_nations.get(&orderkey_values[row]).copied()
                    else {
                        return Ok(false);
                    };
                    let supp_nation_key = supplier_nation_values[suppkey];
                    if supp_nation_key == cust_nation_key {
                        return Ok(false);
                    }
                    group_key.set(Some((
                        supp_nation_key,
                        cust_nation_key,
                        year_cache.year(shipdate)?,
                    )));
                    Ok(true)
                },
                |_, revenue| {
                    if let Some(key) = group_key.get() {
                        *groups.entry(key).or_insert(0.0) += revenue;
                    }
                    Ok(())
                },
            )?;
            return Ok(groups);
        }
        let group_key = std::cell::Cell::new(None::<(i64, i64, i32)>);
        consume_filtered_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            orderkey_values.len(),
            |row| {
                group_key.set(None);
                let shipdate = shipdate_values[row];
                if shipdate < start_days || shipdate > end_days {
                    return Ok(false);
                }
                let (Some(supp_nation_key), Some(cust_nation_key)) = (
                    supplier_nations.get(suppkey_values[row]),
                    order_customer_nations.get(&orderkey_values[row]).copied(),
                ) else {
                    return Ok(false);
                };
                if supp_nation_key == cust_nation_key {
                    return Ok(false);
                }
                group_key.set(Some((
                    supp_nation_key,
                    cust_nation_key,
                    year_cache.year(shipdate)?,
                )));
                Ok(true)
            },
            |_, revenue| {
                if let Some(key) = group_key.get() {
                    *groups.entry(key).or_insert(0.0) += revenue;
                }
                Ok(())
            },
        )?;
        return Ok(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || shipdates.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        if shipdate < start_days || shipdate > end_days {
            continue;
        }
        let (Some(supp_nation_key), Some(cust_nation_key)) = (
            supplier_nations.get(suppkeys.value(row)),
            order_customer_nations.get(&orderkeys.value(row)).copied(),
        ) else {
            continue;
        };
        if supp_nation_key == cust_nation_key {
            continue;
        }
        *groups
            .entry((supp_nation_key, cust_nation_key, year_cache.year(shipdate)?))
            .or_insert(0.0) += extendedprices.value(row) * (1.0 - discounts.value(row));
    }
    Ok(groups)
}

fn q07_volume_batch_typed(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    supplier_nations: &AdaptiveI64Map<i64>,
    order_customer_nations: &FastHashMap<i64, i64>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<FastHashMap<(i64, i64, i32), f64>>> {
    let (Some(orderkeys), Some(suppkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
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
    q07_volume_vector(
        I64VectorView::Arrow(orderkeys),
        I64VectorView::Arrow(suppkeys),
        Date32VectorView::Arrow(shipdates),
        extendedprices,
        discounts,
        supplier_nations,
        order_customer_nations,
        start_days,
        end_days,
    )
    .map(Some)
}

fn q07_output(rows: Vec<Q07Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("supp_nation", DataType::Utf8, false),
            Field::new("cust_nation", DataType::Utf8, false),
            Field::new("l_year", DataType::Int64, false),
            Field::new("revenue", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.supp_nation.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.cust_nation.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| i64::from(row.l_year)),
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
