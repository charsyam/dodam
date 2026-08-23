use super::*;

fn q03_shape(select: &Select, _query: &Query, selection: &SqlExpr) -> bool {
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 4
        && projection.contains("l_orderkey")
        && projection.contains("sum(")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && projection.contains("o_orderdate")
        && projection.contains("o_shippriority")
        && selection.contains("c_mktsegment")
        && selection.contains("c_custkey")
        && selection.contains("o_custkey")
        && selection.contains("l_orderkey")
        && selection.contains("o_orderkey")
        && selection.contains("o_orderdate")
        && selection.contains("l_shipdate")
}

pub(super) async fn try_execute_shipping_priority_revenue_sql(
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
    if !q03_shape(select, query, selection) {
        return Ok(None);
    }
    if !matches!(parse_limit(query), Ok(Some(10))) {
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
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem)) = (customer, orders, lineitem) else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(segment) = string_equality_literal(&conjuncts, "c_mktsegment")? else {
        return Ok(None);
    };
    let Some(order_cutoff) = upper_date_bound(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };
    let Some(ship_cutoff) = lower_date_bound(&conjuncts, "l_shipdate")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let customers = collect_i64_utf8_eq_adaptive_set(
        engine,
        customer.path,
        batch_size,
        "c_custkey",
        "c_mktsegment",
        &segment,
    )
    .await?;
    tpch_profile_elapsed("Q03 customer keys", stage);
    if customers.is_empty() {
        return Ok(Some(q03_output(Vec::new())?));
    }
    let customers = Arc::new(customers);

    let stage = tpch_profile_start();
    let orders = q03_order_rows(engine, orders.path, batch_size, customers, order_cutoff).await?;
    tpch_profile_elapsed("Q03 order rows", stage);
    if orders.is_empty() {
        return Ok(Some(q03_output(Vec::new())?));
    }
    let orders = Arc::new(orders);

    let stage = tpch_profile_start();
    let rows = q03_revenue_rows(engine, lineitem.path, batch_size, orders, ship_cutoff).await?;
    tpch_profile_elapsed("Q03 revenue rows", stage);
    Ok(Some(q03_output(rows)?))
}

fn lower_date_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<i32>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(left, column)
            && let Some(days) = maybe_literal_date_days(right)?
        {
            bound = Some(days);
        } else if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(right, column)
            && let Some(days) = maybe_literal_date_days(left)?
        {
            bound = Some(days);
        }
    }
    Ok(bound)
}

fn upper_date_bound(conjuncts: &[SqlExpr], column: &str) -> Result<Option<i32>> {
    let mut bound = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(left, column)
            && let Some(days) = maybe_literal_date_days(right)?
        {
            bound = Some(days);
        } else if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(right, column)
            && let Some(days) = maybe_literal_date_days(left)?
        {
            bound = Some(days);
        }
    }
    Ok(bound)
}

#[derive(Clone, Copy)]
struct Q03Order {
    o_orderdate: i32,
    o_shippriority: i64,
}

enum Q03OrderMap {
    DenseConstantPriority {
        orderdates: DenseI64RankMap<i32>,
        shippriority: i64,
    },
    ConstantPriority {
        orderdates: FastHashMap<i64, i32>,
        shippriority: i64,
    },
    Variable(FastHashMap<i64, Q03Order>),
}

impl Q03OrderMap {
    fn is_empty(&self) -> bool {
        match self {
            Self::DenseConstantPriority { orderdates, .. } => orderdates.is_empty(),
            Self::ConstantPriority { orderdates, .. } => orderdates.is_empty(),
            Self::Variable(orders) => orders.is_empty(),
        }
    }

    fn get(&self, orderkey: &i64) -> Option<Q03Order> {
        match self {
            Self::DenseConstantPriority {
                orderdates,
                shippriority,
            } => orderdates.get(*orderkey).map(|o_orderdate| Q03Order {
                o_orderdate,
                o_shippriority: *shippriority,
            }),
            Self::ConstantPriority {
                orderdates,
                shippriority,
            } => orderdates
                .get(orderkey)
                .copied()
                .map(|o_orderdate| Q03Order {
                    o_orderdate,
                    o_shippriority: *shippriority,
                }),
            Self::Variable(orders) => orders.get(orderkey).copied(),
        }
    }

    fn contains_key(&self, orderkey: &i64) -> bool {
        match self {
            Self::DenseConstantPriority { orderdates, .. } => orderdates.contains_key(*orderkey),
            Self::ConstantPriority { orderdates, .. } => orderdates.contains_key(orderkey),
            Self::Variable(orders) => orders.contains_key(orderkey),
        }
    }

    fn selective_key_range(&self) -> Option<(i64, i64)> {
        match self {
            Self::DenseConstantPriority { orderdates, .. } => orderdates.selective_key_range(),
            Self::ConstantPriority { orderdates, .. } => {
                selective_i64_key_range(orderdates.keys().copied())
            }
            Self::Variable(orders) => selective_i64_key_range(orders.keys().copied()),
        }
    }

    fn max_key(&self) -> Option<i64> {
        match self {
            Self::DenseConstantPriority { orderdates, .. } => Some(orderdates.max_key()),
            Self::ConstantPriority { orderdates, .. } => orderdates.keys().copied().max(),
            Self::Variable(orders) => orders.keys().copied().max(),
        }
    }

    fn dense_probe(&self, max_key: usize) -> Option<crate::dense::DenseI64Probe> {
        match self {
            Self::DenseConstantPriority { .. } => None,
            Self::ConstantPriority { orderdates, .. } => {
                crate::dense::DenseI64Probe::from_keys_with_max_key(
                    orderdates.keys().copied(),
                    max_key,
                )
            }
            Self::Variable(orders) => {
                crate::dense::DenseI64Probe::from_keys_with_max_key(orders.keys().copied(), max_key)
            }
        }
    }
}

type Q03RevenueMap = FastHashMap<i64, f64>;

async fn q03_order_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customers: Arc<AdaptiveI64Set>,
    order_cutoff: i32,
) -> Result<Q03OrderMap> {
    let constant_shippriority = engine
        .parquet_i64_column_constant(path.clone(), "o_shippriority")
        .await?;
    if let Some(constant_shippriority) = constant_shippriority
        && let Some(orders) = q03_order_rows_late_materialized(
            engine,
            path.clone(),
            batch_size,
            customers.clone(),
            order_cutoff,
            constant_shippriority,
        )
        .await?
    {
        return Ok(orders);
    }
    let mut projection_columns = vec![
        "o_orderkey".to_string(),
        "o_custkey".to_string(),
        "o_orderdate".to_string(),
    ];
    if constant_shippriority.is_none() {
        projection_columns.push("o_shippriority".to_string());
    }
    let projection = Projection::Columns(projection_columns);
    if let Some(partials) = engine
        .parquet_row_group_map_view(
            path.clone(),
            batch_size,
            projection.clone(),
            q03_order_row_group_map_chunk(),
            Vec::<(i64, Q03Order)>::new,
            {
                let customers = customers.clone();
                move |view, order_rows| {
                    q03_order_rows_projected_view_pairs_into(
                        view,
                        &customers,
                        order_cutoff,
                        constant_shippriority,
                        order_rows,
                    )?;
                    Ok(Some(()))
                }
            },
            |orders| Ok(Some(orders)),
        )
        .await?
    {
        let capacity = partials.iter().map(Vec::len).sum();
        let mut orders = fast_hash_map_with_capacity(capacity);
        for partial in partials {
            orders.extend(partial);
        }
        return Ok(Q03OrderMap::Variable(orders));
    }
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    let orders = parallel_batch_fold_view_chunks(
        &mut stream,
        build_map_chunk_size(),
        || fast_hash_map::<i64, Q03Order>(),
        move |view, orders| {
            q03_order_rows_projected_view_into(
                view,
                &customers,
                order_cutoff,
                constant_shippriority,
                orders,
            )?;
            Ok(Some(()))
        },
        Ok,
        fast_hash_map::<i64, Q03Order>(),
        merge_maps,
        "Q03 order rows",
    )?;
    Ok(Q03OrderMap::Variable(orders))
}

fn q03_order_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(14)
}

struct Q03OrderLateState {
    customers: Arc<AdaptiveI64Set>,
    order_cutoff: i32,
    selected_orderdates: Vec<i32>,
    selected_offsets: Vec<u32>,
    payload_offset: usize,
    orderdates: Vec<(i64, i32)>,
}

async fn q03_order_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customers: Arc<AdaptiveI64Set>,
    order_cutoff: i32,
    constant_shippriority: i64,
) -> Result<Option<Q03OrderMap>> {
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["o_custkey".to_string(), "o_orderdate".to_string()]),
            Projection::Columns(vec!["o_orderkey".to_string()]),
            Vec::new(),
            late_materialization_row_group_chunk(14),
            LateMaterializationPolicy::selective(0.20),
            {
                let customers = customers.clone();
                move || Q03OrderLateState {
                    customers: customers.clone(),
                    order_cutoff,
                    selected_orderdates: Vec::new(),
                    selected_offsets: Vec::new(),
                    payload_offset: 0,
                    orderdates: Vec::new(),
                }
            },
            q03_order_late_build_selection_view,
            q03_order_late_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_orderdates.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 order payload row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.orderdates, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };

    let capacity = chunks.iter().map(|chunk| chunk.output.0.len()).sum();
    let mut metrics = LateMaterializedMetrics::default();
    let mut orderdate_chunks = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let (chunk_orderdates, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        orderdate_chunks.push(chunk_orderdates);
    }
    q03_log_order_late_materialized_profile(metrics);
    let rank_map_bytes = dense_i64_rank_map_bytes(512 * 1024 * 1024);
    let orderdates = if env_flag_enabled("DODAM_DISABLE_DENSE_I64_RANK_MAP_PARALLEL_BUILD") {
        DenseI64RankMap::from_pairs(
            orderdate_chunks
                .iter()
                .flat_map(|chunk| chunk.iter().copied()),
            rank_map_bytes,
        )
    } else {
        DenseI64RankMap::from_chunks_parallel(&orderdate_chunks, rank_map_bytes)
    };
    if let Some(orderdates) = orderdates {
        return Ok(Some(Q03OrderMap::DenseConstantPriority {
            orderdates,
            shippriority: constant_shippriority,
        }));
    }
    let mut orderdates = fast_hash_map_with_capacity(capacity);
    for chunk in orderdate_chunks {
        orderdates.extend(chunk);
    }
    Ok(Some(Q03OrderMap::ConstantPriority {
        orderdates,
        shippriority: constant_shippriority,
    }))
}

fn q03_order_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q03OrderLateState,
) -> Result<Option<()>> {
    let (Some(custkeys), Some(orderdates)) = (view.i64_vector(0), view.date32_vector(1)) else {
        return Ok(None);
    };
    state.selected_offsets.clear();
    state.selected_offsets.reserve(view.num_rows().min(1024));
    state.selected_orderdates.reserve(view.num_rows() / 8);
    if let (Some(custkey_values), Some(orderdate_values)) = (
        custkeys.values_if_null_free(),
        orderdates.values_if_null_free(),
    ) {
        if let Some(customer_words) = state.customers.dense_word_slice() {
            for row in 0..view.num_rows() {
                let custkey = custkey_values[row];
                let orderdate = orderdate_values[row];
                if orderdate < state.order_cutoff
                    && crate::dense::adaptive_i64_words_contains(customer_words, custkey)
                {
                    state.selected_offsets.push(row as u32);
                    state.selected_orderdates.push(orderdate);
                }
            }
        } else {
            for row in 0..view.num_rows() {
                let custkey = custkey_values[row];
                let orderdate = orderdate_values[row];
                if orderdate < state.order_cutoff && state.customers.contains(custkey) {
                    state.selected_offsets.push(row as u32);
                    state.selected_orderdates.push(orderdate);
                }
            }
        }
    } else {
        for row in 0..view.num_rows() {
            if custkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            let custkey = custkeys.value(row);
            let orderdate = orderdates.value(row);
            if orderdate < state.order_cutoff && state.customers.contains(custkey) {
                state.selected_offsets.push(row as u32);
                state.selected_orderdates.push(orderdate);
            }
        }
    }
    selection.push_selected_u32_offsets(view.num_rows(), &state.selected_offsets);
    Ok(Some(()))
}

fn q03_order_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut Q03OrderLateState,
) -> Result<Option<()>> {
    let Some(orderkeys) = view.i64_vector(0) else {
        return Ok(None);
    };
    let payload_end = state.payload_offset.saturating_add(view.num_rows());
    let Some(selected_orderdates) = state
        .selected_orderdates
        .get(state.payload_offset..payload_end)
    else {
        return Err(DodamError::UnsupportedSql(
            "Q03 order payload exceeded selected rows".to_string(),
        ));
    };
    state.orderdates.reserve(view.num_rows());
    if let Some(orderkey_values) = orderkeys.values_if_null_free() {
        state.orderdates.extend(
            orderkey_values
                .iter()
                .copied()
                .zip(selected_orderdates.iter().copied()),
        );
    } else {
        for (row, orderdate) in selected_orderdates.iter().copied().enumerate() {
            if orderkeys.is_null(row) {
                continue;
            }
            state.orderdates.push((orderkeys.value(row), orderdate));
        }
    }
    state.payload_offset = payload_end;
    Ok(Some(()))
}

fn q03_log_order_late_materialized_profile(metrics: LateMaterializedMetrics) {
    if !std::env::var("DODAM_TPCH_PROFILE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q03 orders late_materialized rows={} selected={} ratio={:.6} selector_runs={} predicate_read={:.3} ms payload_read={:.3} ms predicate_batches={} payload_batches={} payload_rows={}",
        metrics.total_rows,
        metrics.selected_rows,
        metrics.selected_ratio(),
        metrics.selector_runs,
        metrics.predicate_read_nanos as f64 / 1_000_000.0,
        metrics.payload_read_nanos as f64 / 1_000_000.0,
        metrics.predicate_batches,
        metrics.payload_batches,
        metrics.payload_rows,
    );
}

fn q03_order_rows_projected_view_into(
    view: BatchView<'_>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut FastHashMap<i64, Q03Order>,
) -> Result<()> {
    orders.extend(q03_order_rows_view_pairs(
        view,
        customers,
        order_cutoff,
        constant_shippriority,
    )?);
    Ok(())
}

fn q03_order_rows_projected_view_pairs_into(
    view: BatchView<'_>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut Vec<(i64, Q03Order)>,
) -> Result<()> {
    if orders.is_empty() {
        orders.reserve(view.num_rows() / 8);
    }
    if let Some(customer_words) = customers.dense_word_slice() {
        collect_i64_i64_date32_optional_i64_pairs_view_into(
            view,
            "Q03 order raw vector columns have unsupported types",
            constant_shippriority,
            orders,
            |_, custkey, orderdate, priority| {
                if orderdate >= order_cutoff
                    || !crate::dense::adaptive_i64_words_contains(customer_words, custkey)
                {
                    return Ok(None);
                }
                Ok(Some(Q03Order {
                    o_orderdate: orderdate,
                    o_shippriority: priority,
                }))
            },
        )?;
        return Ok(());
    }
    collect_i64_i64_date32_optional_i64_pairs_view_into(
        view,
        "Q03 order raw vector columns have unsupported types",
        constant_shippriority,
        orders,
        |_, custkey, orderdate, priority| {
            if orderdate >= order_cutoff || !customers.contains(custkey) {
                return Ok(None);
            }
            Ok(Some(Q03Order {
                o_orderdate: orderdate,
                o_shippriority: priority,
            }))
        },
    )?;
    Ok(())
}

fn q03_order_rows_view_pairs(
    view: BatchView<'_>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
) -> Result<Vec<(i64, Q03Order)>> {
    if let Some(customer_words) = customers.dense_word_slice() {
        return collect_i64_i64_date32_optional_i64_pairs_view(
            view,
            "Q03 order raw vector columns have unsupported types",
            constant_shippriority,
            |_, custkey, orderdate, priority| {
                if orderdate >= order_cutoff
                    || !crate::dense::adaptive_i64_words_contains(customer_words, custkey)
                {
                    return Ok(None);
                }
                Ok(Some(Q03Order {
                    o_orderdate: orderdate,
                    o_shippriority: priority,
                }))
            },
        );
    }
    collect_i64_i64_date32_optional_i64_pairs_view(
        view,
        "Q03 order raw vector columns have unsupported types",
        constant_shippriority,
        |_, custkey, orderdate, priority| {
            if orderdate >= order_cutoff || !customers.contains(custkey) {
                return Ok(None);
            }
            Ok(Some(Q03Order {
                o_orderdate: orderdate,
                o_shippriority: priority,
            }))
        },
    )
}

struct Q03Row {
    l_orderkey: i64,
    revenue: f64,
    o_orderdate: i32,
    o_shippriority: i64,
}

async fn q03_revenue_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    orders: Arc<Q03OrderMap>,
    ship_cutoff: i32,
) -> Result<Vec<Q03Row>> {
    if let Some(mut rows) = q03_revenue_rows_late_materialized_carry_order(
        engine,
        path.clone(),
        batch_size,
        orders.clone(),
        ship_cutoff,
        {
            let mut predicates = if let Some((min_key, max_key)) = orders.selective_key_range() {
                i64_range_pruning_predicates("l_orderkey", min_key, max_key)
            } else {
                Vec::new()
            };
            predicates.push(Expr::Comparison(ComparisonExpr {
                column: "l_shipdate".to_string(),
                op: ComparisonOp::Gt,
                value: LiteralValue::Int64(i64::from(ship_cutoff)),
            }));
            predicates
        },
    )
    .await?
    {
        q03_sort_limit_rows(&mut rows);
        return Ok(rows);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_shipdate".to_string(),
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let mut pruning_predicates = if let Some((min_key, max_key)) = orders.selective_key_range() {
        i64_range_pruning_predicates("l_orderkey", min_key, max_key)
    } else {
        Vec::new()
    };
    pruning_predicates.push(Expr::Comparison(ComparisonExpr {
        column: "l_shipdate".to_string(),
        op: ComparisonOp::Gt,
        value: LiteralValue::Int64(i64::from(ship_cutoff)),
    }));
    let revenues = if let Some(revenues) = q03_revenue_rows_late_materialized(
        engine,
        path.clone(),
        batch_size,
        orders.clone(),
        ship_cutoff,
        pruning_predicates.clone(),
    )
    .await?
    {
        revenues
    } else {
        let orders_for_scan = orders.clone();
        let order_probe = Arc::new(q03_dense_order_probe(orders.as_ref()));
        q03_revenue_rows_row_group_map(
            engine,
            path,
            batch_size,
            projection,
            pruning_predicates,
            move |view| {
                q03_revenue_projected_view(
                    view,
                    &orders_for_scan,
                    order_probe.as_ref().as_ref(),
                    ship_cutoff,
                )
            },
        )
        .await?
    };
    let mut rows = revenues
        .into_iter()
        .filter_map(|(orderkey, revenue)| {
            orders.get(&orderkey).map(|order| Q03Row {
                l_orderkey: orderkey,
                revenue,
                o_orderdate: order.o_orderdate,
                o_shippriority: order.o_shippriority,
            })
        })
        .collect::<Vec<_>>();
    q03_sort_limit_rows(&mut rows);
    Ok(rows)
}

fn q03_sort_limit_rows(rows: &mut Vec<Q03Row>) {
    if rows.len() > 10 {
        rows.select_nth_unstable_by(10, q03_row_ordering);
        rows.truncate(10);
    }
    rows.sort_by(q03_row_ordering);
}

async fn q03_revenue_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
) -> Result<SendableBatchStream> {
    if pruning_predicates.is_empty() {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await
    } else {
        engine
            .scan_parquet_batches_pruned(path, batch_size, projection, pruning_predicates)
            .await
    }
}

async fn q03_revenue_rows_row_group_map<Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    map: Map,
) -> Result<Q03RevenueMap>
where
    Map: for<'a> Fn(BatchView<'a>) -> Result<Q03RevenueMap> + Clone + Send + Sync + 'static,
{
    let map_for_row_group = map.clone();
    if let Some(partials) = engine
        .parquet_row_group_map_pruned_view(
            path.clone(),
            batch_size,
            projection.clone(),
            pruning_predicates.clone(),
            q03_row_group_map_chunk(),
            fast_hash_map::<i64, f64>,
            move |view, revenues| {
                merge_f64_groups(revenues, map_for_row_group(view)?);
                Ok(Some(()))
            },
            |revenues| Ok(Some(revenues)),
        )
        .await?
    {
        let mut revenues = fast_hash_map::<i64, f64>();
        for partial in partials {
            merge_f64_groups(&mut revenues, partial);
        }
        Ok(revenues)
    } else {
        let mut stream =
            q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
        parallel_batch_fold_view_chunks(
            &mut stream,
            4,
            fast_hash_map::<i64, f64>,
            move |view, revenues| {
                merge_f64_groups(revenues, map(view)?);
                Ok(Some(()))
            },
            Ok,
            fast_hash_map::<i64, f64>(),
            merge_f64_groups,
            "Q03 revenue aggregate",
        )
    }
}

fn q03_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(2)
}

async fn q03_revenue_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    orders: Arc<Q03OrderMap>,
    ship_cutoff: i32,
    pruning_predicates: Vec<Expr>,
) -> Result<Option<Q03RevenueMap>> {
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_shipdate".to_string()]),
            Projection::Columns(vec![
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            pruning_predicates,
            q03_revenue_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q03_revenue_late_materialized_max_selected_ratio(),
                q03_revenue_late_materialized_max_selector_run_ratio(),
            ),
            {
                let orders = orders.clone();
                move || Q03RevenueLateState {
                    orders: orders.clone(),
                    ship_cutoff,
                    selected_orderkeys: Vec::new(),
                    selected_offsets: Vec::new(),
                    selected_key_values: Vec::new(),
                    payload_offset: 0,
                    revenues: fast_hash_map::<i64, f64>(),
                    revenues_reserved: false,
                }
            },
            q03_revenue_late_build_selection_view,
            q03_revenue_late_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_orderkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue payload row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.revenues, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut revenues = fast_hash_map::<i64, f64>();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_revenues, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        merge_f64_groups(&mut revenues, chunk_revenues);
    }
    q03_log_revenue_late_materialized_profile(
        metrics,
        q03_revenue_late_materialized_row_group_chunk(),
    );
    Ok(Some(revenues))
}

fn q03_revenue_late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(16)
}

fn q03_revenue_late_materialized_max_selected_ratio() -> f64 {
    late_materialization_max_selected_ratio(0.20)
}

fn q03_revenue_late_materialized_max_selector_run_ratio() -> f64 {
    late_materialization_max_selector_run_ratio(0.50)
}

fn q03_revenue_late_coalesce_max_gap() -> usize {
    late_materialization_coalesce_max_gap(8)
}

async fn q03_revenue_rows_late_materialized_carry_order(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    orders: Arc<Q03OrderMap>,
    ship_cutoff: i32,
    pruning_predicates: Vec<Expr>,
) -> Result<Option<Vec<Q03Row>>> {
    let order_probe = q03_dense_order_probe(orders.as_ref()).map(Arc::new);
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_shipdate".to_string()]),
            Projection::Columns(vec![
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            pruning_predicates,
            q03_revenue_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                q03_revenue_late_materialized_max_selected_ratio(),
                q03_revenue_late_materialized_max_selector_run_ratio(),
            ),
            {
                let orders = orders.clone();
                let order_probe = order_probe.clone();
                move || Q03RevenueLateCarryState {
                    orders: orders.clone(),
                    order_probe: order_probe.clone(),
                    ship_cutoff,
                    selected_orders: Vec::new(),
                    selected_payload_offsets: Vec::new(),
                    selected_offsets: Vec::new(),
                    selected_order_values: Vec::new(),
                    payload_offset: 0,
                    selected_payload_offset: 0,
                    selected_payload_rows: 0,
                    rows: fast_hash_map::<i64, Q03Row>(),
                    rows_reserved: false,
                    cache_adjacent_orderkeys: q03_adjacent_orderkey_cache_enabled(),
                }
            },
            q03_revenue_late_carry_build_selection_view,
            q03_revenue_late_carry_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_payload_rows {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue carry payload row mismatch".to_string(),
                    ));
                }
                if state.selected_payload_offset != state.selected_orders.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue carry selected row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.rows, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut rows = fast_hash_map::<i64, Q03Row>();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_rows, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        q03_merge_rows(&mut rows, chunk_rows);
    }
    q03_log_revenue_late_materialized_profile(
        metrics,
        q03_revenue_late_materialized_row_group_chunk(),
    );
    Ok(Some(rows.into_values().collect()))
}

struct Q03RevenueLateCarryState {
    orders: Arc<Q03OrderMap>,
    order_probe: Option<Arc<crate::dense::DenseI64Probe>>,
    ship_cutoff: i32,
    selected_orders: Vec<(i64, Q03Order)>,
    selected_payload_offsets: Vec<u32>,
    selected_offsets: Vec<u32>,
    selected_order_values: Vec<(u32, i64, Q03Order)>,
    payload_offset: usize,
    selected_payload_offset: usize,
    selected_payload_rows: usize,
    rows: FastHashMap<i64, Q03Row>,
    rows_reserved: bool,
    cache_adjacent_orderkeys: bool,
}

fn q03_revenue_late_carry_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q03RevenueLateCarryState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let (Some(orderkeys), Some(shipdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 && shipdates.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        q03_push_late_carry_selection_slices(
            selection,
            orderkey_values,
            shipdate_values,
            state.orders.as_ref(),
            state.order_probe.as_deref(),
            state.ship_cutoff,
            &mut state.selected_orders,
            &mut state.selected_payload_offsets,
            &mut state.selected_payload_rows,
            &mut state.selected_offsets,
            &mut state.selected_order_values,
            state.cache_adjacent_orderkeys,
        );
        return Ok(Some(()));
    }
    selection.push_selected_offsets(
        orderkeys.len(),
        (0..orderkeys.len()).filter_map(|row| {
            let orderkey = orderkeys.is_valid(row).then(|| orderkeys.value(row));
            let selected_order = if let Some(orderkey) = orderkey
                && shipdates.is_valid(row)
                && shipdates.value(row) > state.ship_cutoff
            {
                q03_order_probe_get(
                    state.orders.as_ref(),
                    state.order_probe.as_deref(),
                    orderkey,
                )
            } else {
                None
            };
            if let Some(order) = selected_order {
                state.selected_payload_offsets.push(
                    u32::try_from(state.selected_payload_rows).expect("payload rows fit in u32"),
                );
                state.selected_payload_rows += 1;
                state
                    .selected_orders
                    .push((orderkey.expect("validated orderkey"), order));
                Some(row)
            } else {
                None
            }
        }),
    );
    Ok(Some(()))
}

fn q03_revenue_late_carry_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q03RevenueLateCarryState,
) -> Result<Option<()>> {
    if view.num_columns() == 2 {
        let (Some(orderkeys), Some(shipdates)) = (view.i64_vector(0), view.date32_vector(1)) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return q03_revenue_late_carry_build_selection_batch(batch.clone(), selection, state);
        };
        if let (Some(orderkey_values), Some(shipdate_values)) = (
            orderkeys.values_if_null_free(),
            shipdates.values_if_null_free(),
        ) {
            q03_push_late_carry_selection_slices(
                selection,
                orderkey_values,
                shipdate_values,
                state.orders.as_ref(),
                state.order_probe.as_deref(),
                state.ship_cutoff,
                &mut state.selected_orders,
                &mut state.selected_payload_offsets,
                &mut state.selected_payload_rows,
                &mut state.selected_offsets,
                &mut state.selected_order_values,
                state.cache_adjacent_orderkeys,
            );
            return Ok(Some(()));
        }
        selection.push_selected_offsets(
            orderkeys.len(),
            (0..orderkeys.len()).filter_map(|row| {
                let orderkey = (!orderkeys.is_null(row)).then(|| orderkeys.value(row));
                let selected_order = if let Some(orderkey) = orderkey
                    && !shipdates.is_null(row)
                    && shipdates.value(row) > state.ship_cutoff
                {
                    q03_order_probe_get(
                        state.orders.as_ref(),
                        state.order_probe.as_deref(),
                        orderkey,
                    )
                } else {
                    None
                };
                if let Some(order) = selected_order {
                    state.selected_payload_offsets.push(
                        u32::try_from(state.selected_payload_rows)
                            .expect("payload rows fit in u32"),
                    );
                    state.selected_payload_rows += 1;
                    state
                        .selected_orders
                        .push((orderkey.expect("validated orderkey"), order));
                    Some(row)
                } else {
                    None
                }
            }),
        );
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q03_revenue_late_carry_build_selection_batch(batch.clone(), selection, state)
}

fn q03_push_late_carry_selection_slices(
    selection: &mut LateSelectionBuilder,
    orderkeys: &[i64],
    shipdates: &[i32],
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    ship_cutoff: i32,
    selected_orders: &mut Vec<(i64, Q03Order)>,
    selected_payload_offsets: &mut Vec<u32>,
    selected_payload_rows: &mut usize,
    selected_offsets: &mut Vec<u32>,
    selected_order_values: &mut Vec<(u32, i64, Q03Order)>,
    cache_adjacent_orderkeys: bool,
) {
    selected_offsets.clear();
    selected_order_values.clear();
    selected_offsets.reserve(orderkeys.len().min(1024));
    selected_order_values.reserve(orderkeys.len().min(1024));
    let max_gap = q03_revenue_late_coalesce_max_gap();
    let mut cached_orderkey = None;
    let mut cached_order = None;
    for row in 0..orderkeys.len() {
        if shipdates[row] <= ship_cutoff {
            continue;
        }
        let orderkey = orderkeys[row];
        let order = if cache_adjacent_orderkeys && cached_orderkey == Some(orderkey) {
            cached_order
        } else {
            let order = q03_order_probe_get(orders, order_probe, orderkey);
            if cache_adjacent_orderkeys {
                cached_orderkey = Some(orderkey);
                cached_order = order;
            }
            order
        };
        if let Some(order) = order {
            selected_offsets.push(row as u32);
            selected_order_values.push((row as u32, orderkey, order));
        }
    }
    selected_orders.reserve(selected_offsets.len());
    selected_payload_offsets.reserve(selected_offsets.len());
    let mut selected_index = 0usize;
    selection.push_selected_u32_offsets_coalesced(
        orderkeys.len(),
        selected_offsets,
        max_gap,
        |row| {
            if let Some(row) = row {
                if let Some((offset, orderkey, order)) =
                    selected_order_values.get(selected_index).copied()
                {
                    selected_index += 1;
                    debug_assert_eq!(offset as usize, row);
                    selected_payload_offsets.push(
                        u32::try_from(*selected_payload_rows).expect("payload rows fit in u32"),
                    );
                    selected_orders.push((orderkey, order));
                }
            }
            *selected_payload_rows += 1;
        },
    );
}

fn q03_adjacent_orderkey_cache_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_Q03_ADJACENT_ORDERKEY_CACHE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q03_reserve_late_carry_rows(state: &mut Q03RevenueLateCarryState) {
    if !state.rows_reserved {
        state.rows.reserve(state.selected_orders.len());
        state.rows_reserved = true;
    }
}

fn q03_revenue_late_carry_consume_payload_batch(
    batch: RecordBatch,
    state: &mut Q03RevenueLateCarryState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let (Some(extendedprices), Some(discounts)) =
        (decimal_input(extendedprices)?, decimal_input(discounts)?)
    else {
        return Ok(None);
    };
    q03_reserve_late_carry_rows(state);
    q03_consume_late_carry_payload_vectors(
        state,
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
    )?;
    Ok(Some(()))
}

fn q03_revenue_late_carry_consume_payload_view(
    view: BatchView<'_>,
    state: &mut Q03RevenueLateCarryState,
) -> Result<Option<()>> {
    if view.num_columns() == 2 {
        let (Some(extendedprices), Some(discounts)) =
            (view.decimal128_vector(0), view.decimal128_vector(1))
        else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return q03_revenue_late_carry_consume_payload_batch(batch.clone(), state);
        };
        q03_reserve_late_carry_rows(state);
        q03_consume_late_carry_payload_vectors(state, extendedprices, discounts, view.num_rows())?;
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q03_revenue_late_carry_consume_payload_batch(batch.clone(), state)
}

fn q03_consume_late_carry_payload_vectors(
    state: &mut Q03RevenueLateCarryState,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    row_count: usize,
) -> Result<()> {
    let row_base = state.payload_offset;
    let selected_range = q03_late_carry_selected_range_for_payload_batch(state, row_count)?;
    let selected_offsets = &state.selected_payload_offsets[selected_range.clone()];
    let selected_orders = &state.selected_orders;
    let rows = &mut state.rows;
    let mut selected_index = selected_range.start;
    consume_discounted_revenue_decimal128_vectors_at_offsets(
        extendedprices,
        discounts,
        row_count,
        row_base,
        selected_offsets,
        |_, revenue| {
            let Some((orderkey, order)) = selected_orders.get(selected_index).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "Q03 revenue carry selected row overflow".to_string(),
                ));
            };
            selected_index += 1;
            if let Some(revenue) = revenue {
                q03_accumulate_row(rows, orderkey, order, revenue);
            }
            Ok(())
        },
    )
}

fn q03_late_carry_selected_range_for_payload_batch(
    state: &mut Q03RevenueLateCarryState,
    row_count: usize,
) -> Result<std::ops::Range<usize>> {
    let row_base = state.payload_offset;
    let row_end = row_base.checked_add(row_count).ok_or_else(|| {
        DodamError::UnsupportedSql("Q03 revenue carry payload row overflow".to_string())
    })?;
    if row_end > u32::MAX as usize {
        return Err(DodamError::UnsupportedSql(
            "Q03 revenue carry payload row offset out of range".to_string(),
        ));
    }
    let start = state.selected_payload_offset;
    if let Some(&offset) = state.selected_payload_offsets.get(start)
        && (offset as usize) < row_base
    {
        return Err(DodamError::UnsupportedSql(
            "Q03 revenue carry selected payload row out of order".to_string(),
        ));
    }
    let end = start
        + state.selected_payload_offsets[start..]
            .partition_point(|offset| (*offset as usize) < row_end);
    state.payload_offset = row_end;
    state.selected_payload_offset = end;
    Ok(start..end)
}

fn q03_accumulate_row(
    rows: &mut FastHashMap<i64, Q03Row>,
    orderkey: i64,
    order: Q03Order,
    revenue: f64,
) {
    rows.entry(orderkey)
        .and_modify(|row| row.revenue += revenue)
        .or_insert(Q03Row {
            l_orderkey: orderkey,
            revenue,
            o_orderdate: order.o_orderdate,
            o_shippriority: order.o_shippriority,
        });
}

fn q03_merge_rows(output: &mut FastHashMap<i64, Q03Row>, rows: FastHashMap<i64, Q03Row>) {
    for (orderkey, row) in rows {
        output
            .entry(orderkey)
            .and_modify(|existing| existing.revenue += row.revenue)
            .or_insert(row);
    }
}

struct Q03RevenueLateState {
    orders: Arc<Q03OrderMap>,
    ship_cutoff: i32,
    selected_orderkeys: Vec<Option<i64>>,
    selected_offsets: Vec<u32>,
    selected_key_values: Vec<(u32, i64)>,
    payload_offset: usize,
    revenues: Q03RevenueMap,
    revenues_reserved: bool,
}

fn q03_revenue_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q03RevenueLateState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let (Some(orderkeys), Some(shipdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 && shipdates.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        q03_push_late_key_selection_slices(
            selection,
            orderkey_values,
            shipdate_values,
            state.orders.as_ref(),
            state.ship_cutoff,
            &mut state.selected_orderkeys,
            &mut state.selected_offsets,
            &mut state.selected_key_values,
        );
        return Ok(Some(()));
    }
    selection.push_selected_offsets(
        orderkeys.len(),
        (0..orderkeys.len()).filter_map(|row| {
            let selected = orderkeys.is_valid(row)
                && shipdates.is_valid(row)
                && shipdates.value(row) > state.ship_cutoff
                && state.orders.contains_key(&orderkeys.value(row));
            if selected {
                state.selected_orderkeys.push(Some(orderkeys.value(row)));
                Some(row)
            } else {
                None
            }
        }),
    );
    Ok(Some(()))
}

fn q03_revenue_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q03RevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2 {
        let (Some(orderkeys), Some(shipdates)) = (view.i64_vector(0), view.date32_vector(1)) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return q03_revenue_late_build_selection_batch(batch.clone(), selection, state);
        };
        if let (Some(orderkey_values), Some(shipdate_values)) = (
            orderkeys.values_if_null_free(),
            shipdates.values_if_null_free(),
        ) {
            q03_push_late_key_selection_slices(
                selection,
                orderkey_values,
                shipdate_values,
                state.orders.as_ref(),
                state.ship_cutoff,
                &mut state.selected_orderkeys,
                &mut state.selected_offsets,
                &mut state.selected_key_values,
            );
            return Ok(Some(()));
        }
        selection.push_selected_offsets(
            orderkeys.len(),
            (0..orderkeys.len()).filter_map(|row| {
                let selected = !orderkeys.is_null(row)
                    && !shipdates.is_null(row)
                    && shipdates.value(row) > state.ship_cutoff
                    && state.orders.contains_key(&orderkeys.value(row));
                if selected {
                    state.selected_orderkeys.push(Some(orderkeys.value(row)));
                    Some(row)
                } else {
                    None
                }
            }),
        );
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q03_revenue_late_build_selection_batch(batch.clone(), selection, state)
}

fn q03_push_late_key_selection_slices(
    selection: &mut LateSelectionBuilder,
    orderkeys: &[i64],
    shipdates: &[i32],
    orders: &Q03OrderMap,
    ship_cutoff: i32,
    selected_orderkeys: &mut Vec<Option<i64>>,
    selected_offsets: &mut Vec<u32>,
    selected_key_values: &mut Vec<(u32, i64)>,
) {
    selected_offsets.clear();
    selected_key_values.clear();
    selected_offsets.reserve(orderkeys.len().min(1024));
    selected_key_values.reserve(orderkeys.len().min(1024));
    let max_gap = q03_revenue_late_coalesce_max_gap();
    for row in 0..orderkeys.len() {
        if shipdates[row] <= ship_cutoff {
            continue;
        }
        let orderkey = orderkeys[row];
        if orders.contains_key(&orderkey) {
            selected_offsets.push(row as u32);
            selected_key_values.push((row as u32, orderkey));
        }
    }
    let mut selected_index = 0usize;
    selection.push_selected_u32_offsets_coalesced(
        orderkeys.len(),
        selected_offsets,
        max_gap,
        |row| {
            let orderkey = row.and_then(|row| {
                let (offset, orderkey) = selected_key_values.get(selected_index).copied()?;
                selected_index += 1;
                debug_assert_eq!(offset as usize, row);
                Some(orderkey)
            });
            selected_orderkeys.push(orderkey);
        },
    );
}

fn q03_reserve_late_revenues(state: &mut Q03RevenueLateState) {
    if !state.revenues_reserved {
        state.revenues.reserve(state.selected_orderkeys.len());
        state.revenues_reserved = true;
    }
}

fn q03_revenue_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut Q03RevenueLateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let (Some(extendedprices), Some(discounts)) =
        (decimal_input(extendedprices)?, decimal_input(discounts)?)
    else {
        return Ok(None);
    };
    q03_reserve_late_revenues(state);
    consume_discounted_revenue_decimal128_vectors(
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
        |_, revenue| {
            let Some(orderkey) = state.selected_orderkeys.get(state.payload_offset).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "Q03 revenue payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            let Some(orderkey) = orderkey else {
                return Ok(());
            };
            if let Some(revenue) = revenue {
                *state.revenues.entry(orderkey).or_insert(0.0) += revenue;
            }
            Ok(())
        },
    )?;
    Ok(Some(()))
}

fn q03_revenue_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut Q03RevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2 {
        let (Some(extendedprices), Some(discounts)) =
            (view.decimal128_vector(0), view.decimal128_vector(1))
        else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return q03_revenue_late_consume_payload_batch(batch.clone(), state);
        };
        q03_reserve_late_revenues(state);
        consume_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            view.num_rows(),
            |_, revenue| {
                let Some(orderkey) = state.selected_orderkeys.get(state.payload_offset).copied()
                else {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue payload row overflow".to_string(),
                    ));
                };
                state.payload_offset += 1;
                let Some(orderkey) = orderkey else {
                    return Ok(());
                };
                if let Some(revenue) = revenue {
                    *state.revenues.entry(orderkey).or_insert(0.0) += revenue;
                }
                Ok(())
            },
        )?;
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q03_revenue_late_consume_payload_batch(batch.clone(), state)
}

fn q03_log_revenue_late_materialized_profile(
    metrics: LateMaterializedMetrics,
    row_group_chunk: usize,
) {
    tpch_profile_late_materialized("Q03 revenue", metrics, row_group_chunk);
}

fn q03_dense_order_probe(orders: &Q03OrderMap) -> Option<crate::dense::DenseI64Probe> {
    let max_key = orders.max_key()?;
    let max_key = usize::try_from(max_key).ok()?;
    if max_key > q03_dense_order_probe_max_key() {
        return None;
    }
    orders.dense_probe(max_key)
}

fn q03_dense_order_probe_max_key() -> usize {
    dense_i64_probe_max_key(256 * 1024 * 1024)
}

fn q03_order_probe_contains(
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    orderkey: i64,
) -> bool {
    if let Some(order_probe) = order_probe {
        return order_probe.contains(orderkey);
    }
    orders.contains_key(&orderkey)
}

fn q03_order_probe_get(
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    orderkey: i64,
) -> Option<Q03Order> {
    if let Some(order_probe) = order_probe
        && !order_probe.contains(orderkey)
    {
        return None;
    }
    orders.get(&orderkey)
}

fn q03_row_ordering(left: &Q03Row, right: &Q03Row) -> std::cmp::Ordering {
    right
        .revenue
        .partial_cmp(&left.revenue)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.o_orderdate.cmp(&right.o_orderdate))
}

fn q03_revenue_batch(
    batch: RecordBatch,
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(revenues) = q03_revenue_batch_typed(
        orderkeys,
        shipdates,
        extendedprices,
        discounts,
        orders,
        order_probe,
        ship_cutoff,
    )? {
        return Ok(revenues);
    }
    let mut revenues = fast_hash_map::<i64, f64>();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if shipdate <= ship_cutoff || !q03_order_probe_contains(orders, order_probe, orderkey) {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(orderkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(revenues)
}

fn q03_revenue_projected_view(
    view: BatchView<'_>,
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    if view.num_columns() == 4 {
        if let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        ) && let Some(revenues) = q03_revenue_vector_typed(
            orderkeys,
            shipdates,
            extendedprices,
            discounts,
            orders,
            order_probe,
            ship_cutoff,
        )? {
            return Ok(revenues);
        }
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(fast_hash_map::<i64, f64>());
    };
    q03_revenue_batch(batch.clone(), orders, order_probe, ship_cutoff)
}

fn q03_revenue_batch_typed(
    orderkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    ship_cutoff: i32,
) -> Result<Option<Q03RevenueMap>> {
    let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    q03_revenue_vector_typed(
        I64VectorView::Arrow(orderkeys),
        Date32VectorView::Arrow(shipdates),
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
        orders,
        order_probe,
        ship_cutoff,
    )
}

fn q03_revenue_vector_typed(
    orderkeys: I64VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    orders: &Q03OrderMap,
    order_probe: Option<&crate::dense::DenseI64Probe>,
    ship_cutoff: i32,
) -> Result<Option<Q03RevenueMap>> {
    let mut revenues = fast_hash_map::<i64, f64>();
    if let (Some(orderkey_values), Some(shipdate_values)) = (
        orderkeys.values_if_null_free(),
        shipdates.values_if_null_free(),
    ) {
        consume_filtered_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            orderkey_values.len(),
            |row| {
                let shipdate = shipdate_values[row];
                let orderkey = orderkey_values[row];
                Ok(shipdate > ship_cutoff
                    && q03_order_probe_contains(orders, order_probe, orderkey))
            },
            |row, revenue| {
                *revenues.entry(orderkey_values[row]).or_insert(0.0) += revenue;
                Ok(())
            },
        )?;
        return Ok(Some(revenues));
    }
    consume_filtered_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        orderkeys.len(),
        |row| {
            if orderkeys.is_null(row) || shipdates.is_null(row) {
                return Ok(false);
            }
            let shipdate = shipdates.value(row);
            let orderkey = orderkeys.value(row);
            Ok(shipdate > ship_cutoff && q03_order_probe_contains(orders, order_probe, orderkey))
        },
        |row, revenue| {
            *revenues.entry(orderkeys.value(row)).or_insert(0.0) += revenue;
            Ok(())
        },
    )?;
    Ok(Some(revenues))
}

fn q03_output(rows: Vec<Q03Row>) -> Result<QueryOutput> {
    let orderdates = rows
        .iter()
        .map(|row| date32_to_ymd_string(row.o_orderdate))
        .collect::<Result<Vec<_>>>()?;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("revenue", DataType::Float64, false),
            Field::new("o_orderdate", DataType::Utf8, false),
            Field::new("o_shippriority", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.l_orderkey),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
            Arc::new(StringArray::from_iter_values(
                orderdates.iter().map(String::as_str),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.o_shippriority),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
