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
    let customers = q03_customer_keys(engine, customer.path, batch_size, &segment).await?;
    tpch_profile_elapsed("Q03 customer keys", stage);
    if customers.is_empty() {
        return Ok(Some(q03_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let orders = q03_order_rows(engine, orders.path, batch_size, &customers, order_cutoff).await?;
    tpch_profile_elapsed("Q03 order rows", stage);
    if orders.is_empty() {
        return Ok(Some(q03_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let rows = q03_revenue_rows(engine, lineitem.path, batch_size, &orders, ship_cutoff).await?;
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

async fn q03_customer_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    segment: &str,
) -> Result<AdaptiveI64Set> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_mktsegment".to_string()]),
            None,
        )
        .await?;
    let mut keys = AdaptiveI64Set::new_dense();
    let segment_bytes = segment.as_bytes();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let segments = batch_string_column(&batch, "c_mktsegment")?;
        if q03_customer_keys_typed(custkeys, segments, segment_bytes, &mut keys)? {
            continue;
        }
        for row in 0..batch.num_rows() {
            if segments.is_valid(row)
                && segments.value(row) == segment
                && let Some(custkey) = numeric_i64_value(custkeys, row)?
            {
                keys.insert(custkey);
            }
        }
    }
    Ok(keys)
}

fn q03_customer_keys_typed(
    custkeys: &ArrayRef,
    segments: &StringArray,
    segment: &[u8],
    keys: &mut AdaptiveI64Set,
) -> Result<bool> {
    let Some(custkeys) = custkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(false);
    };
    if custkeys.null_count() == 0 && segments.null_count() == 0 {
        let custkey_values = custkeys.values().as_ref();
        let offsets = segments.value_offsets();
        let values = segments.value_data();
        for row in 0..custkey_values.len() {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            if &values[start..end] == segment {
                keys.insert(custkey_values[row]);
            }
        }
        return Ok(true);
    }
    for row in 0..custkeys.len() {
        if custkeys.is_valid(row)
            && segments.is_valid(row)
            && segments.value(row).as_bytes() == segment
        {
            keys.insert(custkeys.value(row));
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
struct Q03Order {
    o_orderdate: i32,
    o_shippriority: i64,
}

type Q03OrderMap = FastHashMap<i64, Q03Order>;
type Q03RevenueMap = FastHashMap<i64, f64>;

async fn q03_order_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
) -> Result<Q03OrderMap> {
    let constant_shippriority = if q03_constant_shippriority_enabled() {
        engine
            .parquet_i64_column_constant(path.clone(), "o_shippriority")
            .await?
    } else {
        None
    };
    let mut projection_columns = vec![
        "o_orderkey".to_string(),
        "o_custkey".to_string(),
        "o_orderdate".to_string(),
    ];
    if constant_shippriority.is_none() {
        projection_columns.push("o_shippriority".to_string());
    }
    let projection = Projection::Columns(projection_columns);
    let customer_filter = customers.clone();
    let customers = Arc::new(customer_filter.clone());
    if q03_order_row_group_map_enabled()
        && let Some(partials) = engine
            .parquet_row_group_map_view(
                path.clone(),
                batch_size,
                projection.clone(),
                q03_order_row_group_map_chunk(),
                || fast_hash_map_with_capacity(q03_order_row_group_map_initial_capacity()),
                {
                    let customers = customers.clone();
                    move |view, orders| {
                        q03_order_rows_projected_view_into(
                            view,
                            &customers,
                            order_cutoff,
                            constant_shippriority,
                            orders,
                        )?;
                        Ok(Some(()))
                    }
                },
                |orders| Ok(Some(orders)),
            )
            .await?
    {
        let capacity = partials.iter().map(|partial| partial.len()).sum();
        let mut orders = fast_hash_map_with_capacity(capacity);
        for partial in partials {
            merge_maps(&mut orders, partial);
        }
        return Ok(orders);
    }
    let mut stream = if q03_order_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "o_custkey",
                customer_filter.to_hash_set(),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    if q03_order_stream_accumulate_enabled() {
        let mut orders = fast_hash_map::<i64, Q03Order>();
        while let Some(batch) = stream.next() {
            merge_maps(
                &mut orders,
                q03_order_rows_batch(batch?, &customers, order_cutoff, constant_shippriority)?,
            );
        }
        return Ok(orders);
    }
    parallel_batch_fold_view_chunks(
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
    )
}

fn q03_order_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q03_DISABLE_ORDER_ROW_GROUP_MAP").is_none()
}

fn q03_order_row_filter_enabled() -> bool {
    std::env::var("DODAM_Q03_ENABLE_ORDER_ROW_FILTER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q03_order_stream_accumulate_enabled() -> bool {
    std::env::var("DODAM_Q03_ENABLE_ORDER_STREAM_ACCUMULATE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q03_constant_shippriority_enabled() -> bool {
    std::env::var("DODAM_Q03_DISABLE_CONSTANT_SHIPPRIORITY")
        .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true)
}

fn q03_order_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q03_ORDER_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

fn q03_order_row_group_map_initial_capacity() -> usize {
    std::env::var("DODAM_Q03_ORDER_ROW_GROUP_MAP_INITIAL_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(131_072)
}

fn q03_order_rows_batch(
    batch: RecordBatch,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
) -> Result<Q03OrderMap> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let priorities = if constant_shippriority.is_none() {
        Some(batch_column(&batch, "o_shippriority")?)
    } else {
        None
    };
    if let Some(orders) = q03_order_rows_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        priorities,
        customers,
        order_cutoff,
        constant_shippriority,
    )? {
        return Ok(orders);
    }
    let mut orders = fast_hash_map();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(custkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        let Some(priority) = constant_shippriority.or_else(|| {
            priorities.and_then(|priorities| numeric_i64_value(priorities, row).ok()?)
        }) else {
            continue;
        };
        if customers.contains(custkey) && orderdate < order_cutoff {
            orders.insert(
                orderkey,
                Q03Order {
                    o_orderdate: orderdate,
                    o_shippriority: priority,
                },
            );
        }
    }
    Ok(orders)
}

#[allow(dead_code)]
fn q03_order_rows_projected_batch_into(
    batch: RecordBatch,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut Q03OrderMap,
) -> Result<()> {
    if batch.num_columns() == 3
        && constant_shippriority.is_some()
        && q03_order_rows_batch_typed_into(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            None,
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        )?
    {
        return Ok(());
    }
    if batch.num_columns() == 4
        && q03_order_rows_batch_typed_into(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            Some(batch.column(3)),
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        )?
    {
        return Ok(());
    }
    merge_maps(
        orders,
        q03_order_rows_batch(batch, customers, order_cutoff, constant_shippriority)?,
    );
    Ok(())
}

fn q03_order_rows_projected_view_into(
    view: BatchView<'_>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut Q03OrderMap,
) -> Result<()> {
    if view.num_columns() == 3
        && constant_shippriority.is_some()
        && q03_order_rows_vectors_into(
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            None,
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        )
    {
        return Ok(());
    }
    if view.num_columns() == 4
        && q03_order_rows_vectors_into(
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            view.i64_vector(3),
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        )
    {
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q03 order raw vector columns have unsupported types".to_string(),
        ));
    };
    merge_maps(
        orders,
        q03_order_rows_batch(
            batch.clone(),
            customers,
            order_cutoff,
            constant_shippriority,
        )?,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn q03_order_rows_vectors_into(
    orderkeys: Option<I64VectorView<'_>>,
    custkeys: Option<I64VectorView<'_>>,
    orderdates: Option<Date32VectorView<'_>>,
    priorities: Option<I64VectorView<'_>>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut Q03OrderMap,
) -> bool {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (orderkeys, custkeys, orderdates)
    else {
        return false;
    };
    if constant_shippriority.is_none() && priorities.is_none() {
        return false;
    }
    let Some(orderkey_values) = orderkeys.values_if_null_free() else {
        return q03_order_rows_vectors_nullable_into(
            orderkeys,
            custkeys,
            orderdates,
            priorities,
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        );
    };
    let Some(custkey_values) = custkeys.values_if_null_free() else {
        return q03_order_rows_vectors_nullable_into(
            orderkeys,
            custkeys,
            orderdates,
            priorities,
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        );
    };
    let Some(orderdate_values) = orderdates.values_if_null_free() else {
        return q03_order_rows_vectors_nullable_into(
            orderkeys,
            custkeys,
            orderdates,
            priorities,
            customers,
            order_cutoff,
            constant_shippriority,
            orders,
        );
    };
    let priority_values = match priorities {
        Some(priorities) => match priorities.values_if_null_free() {
            Some(values) => Some(values),
            None => {
                return q03_order_rows_vectors_nullable_into(
                    orderkeys,
                    custkeys,
                    orderdates,
                    Some(priorities),
                    customers,
                    order_cutoff,
                    constant_shippriority,
                    orders,
                );
            }
        },
        None => None,
    };
    if let Some(customer_contains) = customers.dense_contains_slice() {
        for row in 0..orderkey_values.len() {
            if orderdate_values[row] >= order_cutoff {
                continue;
            }
            let custkey = custkey_values[row];
            let customer_hit = usize::try_from(custkey)
                .ok()
                .and_then(|index| customer_contains.get(index))
                .copied()
                .unwrap_or(false);
            if customer_hit {
                orders.insert(
                    orderkey_values[row],
                    Q03Order {
                        o_orderdate: orderdate_values[row],
                        o_shippriority: constant_shippriority
                            .unwrap_or_else(|| priority_values.expect("priority values")[row]),
                    },
                );
            }
        }
        return true;
    }
    for row in 0..orderkey_values.len() {
        if orderdate_values[row] < order_cutoff && customers.contains(custkey_values[row]) {
            orders.insert(
                orderkey_values[row],
                Q03Order {
                    o_orderdate: orderdate_values[row],
                    o_shippriority: constant_shippriority
                        .unwrap_or_else(|| priority_values.expect("priority values")[row]),
                },
            );
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn q03_order_rows_vectors_nullable_into(
    orderkeys: I64VectorView<'_>,
    custkeys: I64VectorView<'_>,
    orderdates: Date32VectorView<'_>,
    priorities: Option<I64VectorView<'_>>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut Q03OrderMap,
) -> bool {
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || custkeys.is_null(row)
            || orderdates.is_null(row)
            || priorities.is_some_and(|priorities| priorities.is_null(row))
        {
            continue;
        }
        if orderdates.value(row) < order_cutoff && customers.contains(custkeys.value(row)) {
            orders.insert(
                orderkeys.value(row),
                Q03Order {
                    o_orderdate: orderdates.value(row),
                    o_shippriority: constant_shippriority
                        .unwrap_or_else(|| priorities.expect("priority vector").value(row)),
                },
            );
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn q03_order_rows_batch_typed_into(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    priorities: Option<&ArrayRef>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
    orders: &mut Q03OrderMap,
) -> Result<bool> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    let priorities =
        priorities.and_then(|priorities| priorities.as_any().downcast_ref::<Int64Array>());
    if constant_shippriority.is_none() && priorities.is_none() {
        return Ok(false);
    }
    if orderkeys.null_count() == 0
        && custkeys.null_count() == 0
        && orderdates.null_count() == 0
        && priorities.is_none_or(|priorities| priorities.null_count() == 0)
    {
        let orderkey_values = orderkeys.values().as_ref();
        let custkey_values = custkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        let priority_values = priorities.map(|priorities| priorities.values().as_ref());
        if let Some(customer_contains) = customers.dense_contains_slice() {
            for row in 0..orderkey_values.len() {
                if orderdate_values[row] >= order_cutoff {
                    continue;
                }
                let custkey = custkey_values[row];
                let customer_hit = usize::try_from(custkey)
                    .ok()
                    .and_then(|index| customer_contains.get(index))
                    .copied()
                    .unwrap_or(false);
                if customer_hit {
                    orders.insert(
                        orderkey_values[row],
                        Q03Order {
                            o_orderdate: orderdate_values[row],
                            o_shippriority: constant_shippriority
                                .unwrap_or_else(|| priority_values.expect("priority values")[row]),
                        },
                    );
                }
            }
            return Ok(true);
        }
        for row in 0..orderkey_values.len() {
            if orderdate_values[row] < order_cutoff && customers.contains(custkey_values[row]) {
                orders.insert(
                    orderkey_values[row],
                    Q03Order {
                        o_orderdate: orderdate_values[row],
                        o_shippriority: constant_shippriority
                            .unwrap_or_else(|| priority_values.expect("priority values")[row]),
                    },
                );
            }
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || custkeys.is_null(row)
            || orderdates.is_null(row)
            || priorities.is_some_and(|priorities| priorities.is_null(row))
        {
            continue;
        }
        if orderdates.value(row) < order_cutoff && customers.contains(custkeys.value(row)) {
            orders.insert(
                orderkeys.value(row),
                Q03Order {
                    o_orderdate: orderdates.value(row),
                    o_shippriority: constant_shippriority
                        .unwrap_or_else(|| priorities.expect("priority array").value(row)),
                },
            );
        }
    }
    Ok(true)
}

fn q03_order_rows_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    priorities: Option<&ArrayRef>,
    customers: &AdaptiveI64Set,
    order_cutoff: i32,
    constant_shippriority: Option<i64>,
) -> Result<Option<Q03OrderMap>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    let priorities =
        priorities.and_then(|priorities| priorities.as_any().downcast_ref::<Int64Array>());
    if constant_shippriority.is_none() && priorities.is_none() {
        return Ok(None);
    }
    let mut orders = fast_hash_map();
    if orderkeys.null_count() == 0
        && custkeys.null_count() == 0
        && orderdates.null_count() == 0
        && priorities.is_none_or(|priorities| priorities.null_count() == 0)
    {
        let orderkey_values = orderkeys.values().as_ref();
        let custkey_values = custkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        let priority_values = priorities.map(|priorities| priorities.values().as_ref());
        if let Some(customer_contains) = customers.dense_contains_slice() {
            for row in 0..orderkey_values.len() {
                let custkey = custkey_values[row];
                let customer_hit = usize::try_from(custkey)
                    .ok()
                    .and_then(|index| customer_contains.get(index))
                    .copied()
                    .unwrap_or(false);
                if orderdate_values[row] < order_cutoff && customer_hit {
                    orders.insert(
                        orderkey_values[row],
                        Q03Order {
                            o_orderdate: orderdate_values[row],
                            o_shippriority: constant_shippriority
                                .unwrap_or_else(|| priority_values.expect("priority values")[row]),
                        },
                    );
                }
            }
            return Ok(Some(orders));
        }
        for row in 0..orderkey_values.len() {
            if orderdate_values[row] < order_cutoff && customers.contains(custkey_values[row]) {
                orders.insert(
                    orderkey_values[row],
                    Q03Order {
                        o_orderdate: orderdate_values[row],
                        o_shippriority: constant_shippriority
                            .unwrap_or_else(|| priority_values.expect("priority values")[row]),
                    },
                );
            }
        }
        return Ok(Some(orders));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || custkeys.is_null(row)
            || orderdates.is_null(row)
            || priorities.is_some_and(|priorities| priorities.is_null(row))
        {
            continue;
        }
        if orderdates.value(row) < order_cutoff && customers.contains(custkeys.value(row)) {
            orders.insert(
                orderkeys.value(row),
                Q03Order {
                    o_orderdate: orderdates.value(row),
                    o_shippriority: constant_shippriority
                        .unwrap_or_else(|| priorities.expect("priority array").value(row)),
                },
            );
        }
    }
    Ok(Some(orders))
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
    orders: &Q03OrderMap,
    ship_cutoff: i32,
) -> Result<Vec<Q03Row>> {
    if q03_revenue_late_carry_order_enabled()
        && let Some(mut rows) = q03_revenue_rows_late_materialized_carry_order(
            engine,
            path.clone(),
            batch_size,
            orders,
            ship_cutoff,
            {
                let mut predicates = if let Some((min_key, max_key)) =
                    selective_i64_key_range(orders.keys().copied())
                {
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
    let mut pruning_predicates =
        if let Some((min_key, max_key)) = selective_i64_key_range(orders.keys().copied()) {
            i64_range_pruning_predicates("l_orderkey", min_key, max_key)
        } else {
            Vec::new()
        };
    pruning_predicates.push(Expr::Comparison(ComparisonExpr {
        column: "l_shipdate".to_string(),
        op: ComparisonOp::Gt,
        value: LiteralValue::Int64(i64::from(ship_cutoff)),
    }));
    let revenues = if q03_revenue_late_materialized_enabled()
        && let Some(revenues) = q03_revenue_rows_late_materialized(
            engine,
            path.clone(),
            batch_size,
            orders,
            ship_cutoff,
            pruning_predicates.clone(),
        )
        .await?
    {
        revenues
    } else if q03_revenue_vector_enabled()
        && q03_row_group_map_enabled()
        && q03_sorted_order_lookup_enabled()
    {
        let orders_for_scan = Arc::new(SortedI64Lookup::from_hash_map(orders));
        q03_revenue_rows_vector_row_group_map(
            engine,
            path,
            batch_size,
            projection,
            pruning_predicates,
            move |view, revenues| {
                q03_revenue_projected_view_sorted_into(
                    view,
                    &orders_for_scan,
                    ship_cutoff,
                    revenues,
                )
            },
        )
        .await?
    } else if q03_row_group_map_enabled() {
        if q03_sorted_order_lookup_enabled() {
            let orders_for_scan = Arc::new(SortedI64Lookup::from_hash_map(orders));
            q03_revenue_rows_row_group_map(
                engine,
                path,
                batch_size,
                projection,
                pruning_predicates,
                move |view| q03_revenue_projected_view_sorted(view, &orders_for_scan, ship_cutoff),
            )
            .await?
        } else {
            let orders_for_scan = Arc::new(orders.clone());
            let order_probe = Arc::new(q03_dense_order_probe(orders));
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
                        order_probe.as_deref(),
                        ship_cutoff,
                    )
                },
            )
            .await?
        }
    } else if q03_sorted_order_lookup_enabled() {
        let mut stream =
            q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
        let orders_for_scan = Arc::new(SortedI64Lookup::from_hash_map(orders));
        parallel_batch_fold_view_chunks(
            &mut stream,
            4,
            fast_hash_map::<i64, f64>,
            move |view, revenues| {
                let Some(batch) = view.try_record_batch() else {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 sorted revenue raw vector columns have unsupported fallback layout"
                            .to_string(),
                    ));
                };
                merge_f64_groups(
                    revenues,
                    q03_revenue_batch_sorted(batch.clone(), &orders_for_scan, ship_cutoff)?,
                );
                Ok(Some(()))
            },
            Ok,
            fast_hash_map::<i64, f64>(),
            merge_f64_groups,
            "Q03 revenue aggregate",
        )?
    } else {
        let mut stream =
            q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
        let orders_for_scan = Arc::new(orders.clone());
        let order_probe = Arc::new(q03_dense_order_probe(orders));
        parallel_batch_fold_view_chunks(
            &mut stream,
            4,
            fast_hash_map::<i64, f64>,
            move |view, revenues| {
                let Some(batch) = view.try_record_batch() else {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue raw vector columns have unsupported fallback layout"
                            .to_string(),
                    ));
                };
                merge_f64_groups(
                    revenues,
                    q03_revenue_batch(
                        batch.clone(),
                        &orders_for_scan,
                        order_probe.as_deref(),
                        ship_cutoff,
                    )?,
                );
                Ok(Some(()))
            },
            Ok,
            fast_hash_map::<i64, f64>(),
            merge_f64_groups,
            "Q03 revenue aggregate",
        )?
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

fn q03_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q03_DISABLE_ROW_GROUP_MAP").is_none()
}

fn q03_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q03_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

fn q03_revenue_vector_enabled() -> bool {
    std::env::var_os("DODAM_Q03_ENABLE_REVENUE_VECTOR").is_some()
}

async fn q03_revenue_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    orders: &Q03OrderMap,
    ship_cutoff: i32,
    pruning_predicates: Vec<Expr>,
) -> Result<Option<Q03RevenueMap>> {
    let orders = Arc::new(orders.clone());
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
                    payload_offset: 0,
                    revenues: fast_hash_map::<i64, f64>(),
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

fn q03_revenue_late_materialized_enabled() -> bool {
    if std::env::var_os("DODAM_Q03_ENABLE_REVENUE_LATE_MATERIALIZE").is_some() {
        return true;
    }
    std::env::var_os("DODAM_Q03_DISABLE_REVENUE_LATE_MATERIALIZE").is_none()
}

fn q03_revenue_late_carry_order_enabled() -> bool {
    if std::env::var_os("DODAM_Q03_ENABLE_REVENUE_LATE_CARRY_ORDER").is_some() {
        return true;
    }
    std::env::var_os("DODAM_Q03_DISABLE_REVENUE_LATE_CARRY_ORDER").is_none()
}

fn q03_revenue_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q03_REVENUE_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

fn q03_revenue_late_materialized_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q03_REVENUE_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

fn q03_revenue_late_materialized_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q03_REVENUE_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.50)
}

async fn q03_revenue_rows_late_materialized_carry_order(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    orders: &Q03OrderMap,
    ship_cutoff: i32,
    pruning_predicates: Vec<Expr>,
) -> Result<Option<Vec<Q03Row>>> {
    let orders = Arc::new(orders.clone());
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
                move || Q03RevenueLateCarryState {
                    orders: orders.clone(),
                    ship_cutoff,
                    selected_orders: Vec::new(),
                    selected_offsets: Vec::new(),
                    payload_offset: 0,
                    rows: fast_hash_map::<i64, Q03Row>(),
                }
            },
            q03_revenue_late_carry_build_selection_view,
            q03_revenue_late_carry_consume_payload_view,
            |state, metrics| {
                if state.payload_offset != state.selected_orders.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue carry payload row mismatch".to_string(),
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
    ship_cutoff: i32,
    selected_orders: Vec<(i64, Q03Order)>,
    selected_offsets: Vec<u32>,
    payload_offset: usize,
    rows: FastHashMap<i64, Q03Row>,
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
            state.ship_cutoff,
            &mut state.selected_orders,
            &mut state.selected_offsets,
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
                state.orders.get(&orderkey).copied()
            } else {
                None
            };
            if let Some(order) = selected_order {
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
                state.ship_cutoff,
                &mut state.selected_orders,
                &mut state.selected_offsets,
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
                    state.orders.get(&orderkey).copied()
                } else {
                    None
                };
                if let Some(order) = selected_order {
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
    ship_cutoff: i32,
    selected_orders: &mut Vec<(i64, Q03Order)>,
    selected_offsets: &mut Vec<u32>,
) {
    selected_offsets.clear();
    selected_offsets.reserve(orderkeys.len().min(1024));
    for row in 0..orderkeys.len() {
        if shipdates[row] <= ship_cutoff {
            continue;
        }
        let orderkey = orderkeys[row];
        let Some(order) = orders.get(&orderkey).copied() else {
            continue;
        };
        selected_orders.push((orderkey, order));
        selected_offsets.push(row as u32);
    }
    selection.push_selected_u32_offsets(orderkeys.len(), selected_offsets);
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
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let (discount_scale, revenue_scale) =
        decimal_discounted_revenue_scales(extendedprices, discounts);
    if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
        for row in 0..batch.num_rows() {
            let Some(&(orderkey, order)) = state.selected_orders.get(state.payload_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "Q03 revenue carry payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            let revenue = decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
            q03_accumulate_row(&mut state.rows, orderkey, order, revenue);
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let Some(&(orderkey, order)) = state.selected_orders.get(state.payload_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q03 revenue carry payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        if extendedprices.is_null(row) || discounts.is_null(row) {
            continue;
        }
        q03_accumulate_row(
            &mut state.rows,
            orderkey,
            order,
            decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            ),
        );
    }
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
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
        if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
            for row in 0..view.num_rows() {
                let Some(&(orderkey, order)) = state.selected_orders.get(state.payload_offset)
                else {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue carry payload row overflow".to_string(),
                    ));
                };
                state.payload_offset += 1;
                let revenue = decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
                q03_accumulate_row(&mut state.rows, orderkey, order, revenue);
            }
            return Ok(Some(()));
        }
        for row in 0..view.num_rows() {
            let Some(&(orderkey, order)) = state.selected_orders.get(state.payload_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "Q03 revenue carry payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            if extendedprices.is_null(row) || discounts.is_null(row) {
                continue;
            }
            q03_accumulate_row(
                &mut state.rows,
                orderkey,
                order,
                decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                ),
            );
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q03_revenue_late_carry_consume_payload_batch(batch.clone(), state)
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
        q03_accumulate_row(
            output,
            orderkey,
            Q03Order {
                o_orderdate: row.o_orderdate,
                o_shippriority: row.o_shippriority,
            },
            row.revenue,
        );
    }
}

struct Q03RevenueLateState {
    orders: Arc<Q03OrderMap>,
    ship_cutoff: i32,
    selected_orderkeys: Vec<i64>,
    selected_offsets: Vec<u32>,
    payload_offset: usize,
    revenues: Q03RevenueMap,
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
                state.selected_orderkeys.push(orderkeys.value(row));
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
                    state.selected_orderkeys.push(orderkeys.value(row));
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
    selected_orderkeys: &mut Vec<i64>,
    selected_offsets: &mut Vec<u32>,
) {
    selected_offsets.clear();
    selected_offsets.reserve(orderkeys.len().min(1024));
    for row in 0..orderkeys.len() {
        if shipdates[row] <= ship_cutoff {
            continue;
        }
        let orderkey = orderkeys[row];
        if !orders.contains_key(&orderkey) {
            continue;
        }
        selected_orderkeys.push(orderkey);
        selected_offsets.push(row as u32);
    }
    selection.push_selected_u32_offsets(orderkeys.len(), selected_offsets);
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
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let (discount_scale, revenue_scale) =
        decimal_discounted_revenue_scales(extendedprices, discounts);
    if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
        for row in 0..batch.num_rows() {
            let Some(&orderkey) = state.selected_orderkeys.get(state.payload_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "Q03 revenue payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            *state.revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let Some(&orderkey) = state.selected_orderkeys.get(state.payload_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q03 revenue payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        if extendedprices.is_null(row) || discounts.is_null(row) {
            continue;
        }
        *state.revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
            extendedprice_values[row],
            discount_values[row],
            discount_scale,
            revenue_scale,
        );
    }
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
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
        if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
            for row in 0..view.num_rows() {
                let Some(&orderkey) = state.selected_orderkeys.get(state.payload_offset) else {
                    return Err(DodamError::UnsupportedSql(
                        "Q03 revenue payload row overflow".to_string(),
                    ));
                };
                state.payload_offset += 1;
                *state.revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
            return Ok(Some(()));
        }
        for row in 0..view.num_rows() {
            let Some(&orderkey) = state.selected_orderkeys.get(state.payload_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "Q03 revenue payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            if extendedprices.is_null(row) || discounts.is_null(row) {
                continue;
            }
            *state.revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
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
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q03 revenue: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q03_dense_order_probe(orders: &Q03OrderMap) -> Option<Vec<bool>> {
    if !q03_dense_order_probe_enabled() {
        return None;
    }
    let max_key = orders.keys().copied().max()?;
    let max_key = usize::try_from(max_key).ok()?;
    if max_key > q03_dense_order_probe_max_key() {
        return None;
    }
    let mut contains = vec![false; max_key + 1];
    for &key in orders.keys() {
        let Ok(index) = usize::try_from(key) else {
            return None;
        };
        contains[index] = true;
    }
    Some(contains)
}

fn q03_dense_order_probe_enabled() -> bool {
    std::env::var("DODAM_Q03_ENABLE_DENSE_ORDER_PROBE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn q03_dense_order_probe_max_key() -> usize {
    std::env::var("DODAM_Q03_DENSE_ORDER_PROBE_MAX_KEY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100_000_000)
}

fn q03_order_probe_contains(
    orders: &Q03OrderMap,
    order_probe: Option<&[bool]>,
    orderkey: i64,
) -> bool {
    if let Some(order_probe) = order_probe
        && let Ok(index) = usize::try_from(orderkey)
    {
        return order_probe.get(index).copied().unwrap_or(false);
    }
    orders.contains_key(&orderkey)
}

fn q03_sorted_order_lookup_enabled() -> bool {
    if std::env::var("DODAM_Q03_ENABLE_SORTED_ORDER_LOOKUP")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return true;
    }
    false
}

fn q03_row_ordering(left: &Q03Row, right: &Q03Row) -> std::cmp::Ordering {
    right
        .revenue
        .partial_cmp(&left.revenue)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.o_orderdate.cmp(&right.o_orderdate))
}

async fn q03_revenue_rows_vector_row_group_map<Map>(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    map: Map,
) -> Result<Q03RevenueMap>
where
    Map: for<'a> Fn(BatchView<'a>, &mut Vec<(i64, f64)>) -> Result<()>
        + Clone
        + Send
        + Sync
        + 'static,
{
    let map_for_row_group = map.clone();
    if let Some(partials) = engine
        .parquet_row_group_map_pruned_view(
            path.clone(),
            batch_size,
            projection.clone(),
            pruning_predicates.clone(),
            q03_row_group_map_chunk(),
            Vec::<(i64, f64)>::new,
            move |view, revenues| {
                map_for_row_group(view, revenues)?;
                Ok(Some(()))
            },
            |revenues| Ok(Some(revenues)),
        )
        .await?
    {
        let mut revenues = Vec::<(i64, f64)>::new();
        for mut partial in partials {
            revenues.append(&mut partial);
        }
        return Ok(q03_reduce_revenue_pairs(revenues));
    }
    let mut stream =
        q03_revenue_stream(engine, path, batch_size, projection, pruning_predicates).await?;
    let revenues = parallel_batch_fold_view_chunks(
        &mut stream,
        4,
        Vec::<(i64, f64)>::new,
        move |view, revenues| {
            map(view, revenues)?;
            Ok(Some(()))
        },
        Ok,
        Vec::<(i64, f64)>::new(),
        q03_merge_revenue_pairs,
        "Q03 revenue aggregate",
    )?;
    Ok(q03_reduce_revenue_pairs(revenues))
}

fn q03_merge_revenue_pairs(output: &mut Vec<(i64, f64)>, mut batch: Vec<(i64, f64)>) {
    output.append(&mut batch);
}

fn q03_reduce_revenue_pairs(mut pairs: Vec<(i64, f64)>) -> Q03RevenueMap {
    if pairs.is_empty() {
        return fast_hash_map();
    }
    pairs.sort_unstable_by_key(|(key, _)| *key);
    let mut revenues = fast_hash_map_with_capacity(pairs.len());
    let mut iter = pairs.into_iter();
    let Some((mut current_key, mut current_value)) = iter.next() else {
        return revenues;
    };
    for (key, value) in iter {
        if key == current_key {
            current_value += value;
        } else {
            revenues.insert(current_key, current_value);
            current_key = key;
            current_value = value;
        }
    }
    revenues.insert(current_key, current_value);
    revenues
}

fn q03_revenue_batch(
    batch: RecordBatch,
    orders: &Q03OrderMap,
    order_probe: Option<&[bool]>,
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

#[allow(dead_code)]
fn q03_revenue_projected_batch(
    batch: RecordBatch,
    orders: &Q03OrderMap,
    order_probe: Option<&[bool]>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    if batch.num_columns() == 4
        && let Some(revenues) = q03_revenue_batch_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            orders,
            order_probe,
            ship_cutoff,
        )?
    {
        return Ok(revenues);
    }
    q03_revenue_batch(batch, orders, order_probe, ship_cutoff)
}

fn q03_revenue_projected_view(
    view: BatchView<'_>,
    orders: &Q03OrderMap,
    order_probe: Option<&[bool]>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        )
        && let Some(revenues) = q03_revenue_vector_typed(
            orderkeys,
            shipdates,
            extendedprices,
            discounts,
            orders,
            order_probe,
            ship_cutoff,
        )
    {
        return Ok(revenues);
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
    order_probe: Option<&[bool]>,
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
    let mut revenues = fast_hash_map::<i64, f64>();
    if orderkeys.null_count() == 0
        && shipdates.null_count() == 0
        && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let (discount_scale, revenue_scale) =
            decimal_discounted_revenue_scales(extendedprices, discounts);
        for row in 0..orderkeys.len() {
            let shipdate = shipdate_values[row];
            let orderkey = orderkey_values[row];
            if shipdate > ship_cutoff && q03_order_probe_contains(orders, order_probe, orderkey) {
                *revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
        }
        return Ok(Some(revenues));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || shipdates.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        let orderkey = orderkeys.value(row);
        if shipdate > ship_cutoff && q03_order_probe_contains(orders, order_probe, orderkey) {
            *revenues.entry(orderkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
    }
    Ok(Some(revenues))
}

fn q03_revenue_vector_typed(
    orderkeys: I64VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    orders: &Q03OrderMap,
    order_probe: Option<&[bool]>,
    ship_cutoff: i32,
) -> Option<Q03RevenueMap> {
    let mut revenues = fast_hash_map::<i64, f64>();
    if let (Some(orderkey_values), Some(shipdate_values)) = (
        orderkeys.values_if_null_free(),
        shipdates.values_if_null_free(),
    ) && extendedprices.null_count() == 0
        && discounts.null_count() == 0
    {
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let discount_scale = discounts.scale();
        let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
        for row in 0..orderkey_values.len() {
            let shipdate = shipdate_values[row];
            let orderkey = orderkey_values[row];
            if shipdate > ship_cutoff && q03_order_probe_contains(orders, order_probe, orderkey) {
                *revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
        }
        return Some(revenues);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || shipdates.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let shipdate = shipdates.value(row);
        let orderkey = orderkeys.value(row);
        if shipdate > ship_cutoff && q03_order_probe_contains(orders, order_probe, orderkey) {
            *revenues.entry(orderkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
    }
    Some(revenues)
}

fn q03_revenue_projected_batch_sorted_into(
    batch: RecordBatch,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
    revenues: &mut Vec<(i64, f64)>,
) -> Result<()> {
    if batch.num_columns() == 4
        && q03_revenue_batch_sorted_typed_into(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            orders,
            ship_cutoff,
            revenues,
        )?
    {
        return Ok(());
    }
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if shipdate <= ship_cutoff || orders.get(orderkey).is_none() {
            continue;
        }
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        revenues.push((orderkey, extendedprice * (1.0 - discount)));
    }
    Ok(())
}

fn q03_revenue_projected_view_sorted_into(
    view: BatchView<'_>,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
    revenues: &mut Vec<(i64, f64)>,
) -> Result<()> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        )
        && q03_revenue_vector_sorted_typed_into(
            orderkeys,
            shipdates,
            extendedprices,
            discounts,
            orders,
            ship_cutoff,
            revenues,
        )
    {
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(());
    };
    q03_revenue_projected_batch_sorted_into(batch.clone(), orders, ship_cutoff, revenues)
}

fn q03_revenue_batch_sorted_typed_into(
    orderkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
    revenues: &mut Vec<(i64, f64)>,
) -> Result<bool> {
    let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        shipdates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() != 0
        || shipdates.null_count() != 0
        || extendedprices.null_count() != 0
        || discounts.null_count() != 0
    {
        return Ok(false);
    }
    let orderkey_values = orderkeys.values().as_ref();
    let shipdate_values = shipdates.values().as_ref();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let (discount_scale, revenue_scale) =
        decimal_discounted_revenue_scales(extendedprices, discounts);
    for row in 0..orderkeys.len() {
        let shipdate = shipdate_values[row];
        let orderkey = orderkey_values[row];
        if shipdate > ship_cutoff && orders.get(orderkey).is_some() {
            revenues.push((
                orderkey,
                decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                ),
            ));
        }
    }
    Ok(true)
}

fn q03_revenue_vector_sorted_typed_into(
    orderkeys: I64VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
    revenues: &mut Vec<(i64, f64)>,
) -> bool {
    let (Some(orderkey_values), Some(shipdate_values)) = (
        orderkeys.values_if_null_free(),
        shipdates.values_if_null_free(),
    ) else {
        return false;
    };
    if extendedprices.null_count() != 0 || discounts.null_count() != 0 {
        return false;
    }
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let discount_scale = discounts.scale();
    let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
    for row in 0..orderkey_values.len() {
        let shipdate = shipdate_values[row];
        let orderkey = orderkey_values[row];
        if shipdate > ship_cutoff && orders.get(orderkey).is_some() {
            revenues.push((
                orderkey,
                decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                ),
            ));
        }
    }
    true
}

fn q03_revenue_batch_sorted(
    batch: RecordBatch,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    if let Some(revenues) = q03_revenue_batch_sorted_typed(
        orderkeys,
        shipdates,
        extendedprices,
        discounts,
        orders,
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
        if shipdate <= ship_cutoff || orders.get(orderkey).is_none() {
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

#[allow(dead_code)]
fn q03_revenue_projected_batch_sorted(
    batch: RecordBatch,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    if batch.num_columns() == 4
        && let Some(revenues) = q03_revenue_batch_sorted_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            orders,
            ship_cutoff,
        )?
    {
        return Ok(revenues);
    }
    q03_revenue_batch_sorted(batch, orders, ship_cutoff)
}

fn q03_revenue_projected_view_sorted(
    view: BatchView<'_>,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Result<Q03RevenueMap> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(shipdates), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        )
        && let Some(revenues) = q03_revenue_vector_sorted_typed(
            orderkeys,
            shipdates,
            extendedprices,
            discounts,
            orders,
            ship_cutoff,
        )
    {
        return Ok(revenues);
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(fast_hash_map::<i64, f64>());
    };
    q03_revenue_batch_sorted(batch.clone(), orders, ship_cutoff)
}

fn q03_revenue_batch_sorted_typed(
    orderkeys: &ArrayRef,
    shipdates: &ArrayRef,
    extendedprices: &ArrayRef,
    discounts: &ArrayRef,
    orders: &SortedI64Lookup<Q03Order>,
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
    if orderkeys.null_count() != 0
        || shipdates.null_count() != 0
        || extendedprices.null_count() != 0
        || discounts.null_count() != 0
    {
        return Ok(None);
    }
    let mut revenues = fast_hash_map::<i64, f64>();
    let orderkey_values = orderkeys.values().as_ref();
    let shipdate_values = shipdates.values().as_ref();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let (discount_scale, revenue_scale) =
        decimal_discounted_revenue_scales(extendedprices, discounts);
    for row in 0..orderkeys.len() {
        let shipdate = shipdate_values[row];
        let orderkey = orderkey_values[row];
        if shipdate > ship_cutoff && orders.get(orderkey).is_some() {
            *revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Ok(Some(revenues))
}

fn q03_revenue_vector_sorted_typed(
    orderkeys: I64VectorView<'_>,
    shipdates: Date32VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    orders: &SortedI64Lookup<Q03Order>,
    ship_cutoff: i32,
) -> Option<Q03RevenueMap> {
    let (Some(orderkey_values), Some(shipdate_values)) = (
        orderkeys.values_if_null_free(),
        shipdates.values_if_null_free(),
    ) else {
        return None;
    };
    if extendedprices.null_count() != 0 || discounts.null_count() != 0 {
        return None;
    }
    let mut revenues = fast_hash_map::<i64, f64>();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    let discount_scale = discounts.scale();
    let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
    for row in 0..orderkey_values.len() {
        let shipdate = shipdate_values[row];
        let orderkey = orderkey_values[row];
        if shipdate > ship_cutoff && orders.get(orderkey).is_some() {
            *revenues.entry(orderkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
    }
    Some(revenues)
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
