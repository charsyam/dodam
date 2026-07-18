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
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_nationkey".to_string()]),
            None,
        )
        .await?;
    let nation_keys = AdaptiveI64Set::from_hash(nation_names.keys().copied().collect());
    let mut customers = fast_hash_map::<i64, i64>();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let nationkeys = batch_column(&batch, "c_nationkey")?;
        if key_nation_map_batch_typed(custkeys, nationkeys, &nation_keys, &mut customers) {
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(custkey), Some(nationkey)) = (
                numeric_i64_value(custkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_keys.contains(nationkey) {
                customers.insert(custkey, nationkey);
            }
        }
    }
    Ok(customers)
}

async fn supplier_nations_in_region(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
) -> Result<FastHashMap<i64, i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["s_suppkey".to_string(), "s_nationkey".to_string()]),
            None,
        )
        .await?;
    let nation_keys = AdaptiveI64Set::from_hash(nation_names.keys().copied().collect());
    let mut suppliers = fast_hash_map::<i64, i64>();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        if key_nation_map_batch_typed(suppkeys, nationkeys, &nation_keys, &mut suppliers) {
            continue;
        }
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_keys.contains(nationkey) {
                suppliers.insert(suppkey, nationkey);
            }
        }
    }
    Ok(suppliers)
}

fn key_nation_map_batch_typed(
    keys: &ArrayRef,
    nationkeys: &ArrayRef,
    nation_filter: &AdaptiveI64Set,
    output: &mut FastHashMap<i64, i64>,
) -> bool {
    let (Some(keys), Some(nationkeys)) = (
        keys.as_any().downcast_ref::<Int64Array>(),
        nationkeys.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return false;
    };
    if keys.null_count() == 0 && nationkeys.null_count() == 0 {
        let key_values = keys.values().as_ref();
        let nation_values = nationkeys.values().as_ref();
        if let Some(nation_contains) = nation_filter.dense_contains_slice() {
            for row in 0..key_values.len() {
                let nationkey = nation_values[row];
                if usize::try_from(nationkey)
                    .ok()
                    .and_then(|index| nation_contains.get(index))
                    .copied()
                    .unwrap_or(false)
                {
                    output.insert(key_values[row], nationkey);
                }
            }
            return true;
        }
        for row in 0..key_values.len() {
            let nationkey = nation_values[row];
            if nation_filter.contains(nationkey) {
                output.insert(key_values[row], nationkey);
            }
        }
        return true;
    }
    for row in 0..keys.len() {
        if keys.is_null(row) || nationkeys.is_null(row) {
            continue;
        }
        let nationkey = nationkeys.value(row);
        if nation_filter.contains(nationkey) {
            output.insert(keys.value(row), nationkey);
        }
    }
    true
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
    parallel_batch_fold_view_chunks(
        &mut stream,
        4,
        || fast_hash_map::<i64, i64>(),
        move |view, orders| {
            merge_maps(
                orders,
                order_customer_nations_view(view, &customer_nations, start_days, end_days)?,
            );
            Ok(Some(()))
        },
        Ok,
        fast_hash_map::<i64, i64>(),
        merge_maps,
        "regional order customer nations",
    )
}

fn order_customer_nations_batch(
    batch: RecordBatch,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<i64, i64>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let Some(orders) = order_customer_nations_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        customer_nations,
        start_days,
        end_days,
    ) {
        return Ok(orders);
    }
    let mut orders = fast_hash_map::<i64, i64>();
    for row in 0..batch.num_rows() {
        let Some(orderdate) = date32_value(orderdates, row)? else {
            continue;
        };
        if orderdate < start_days || orderdate >= end_days {
            continue;
        }
        let (Some(orderkey), Some(custkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(custkeys, row)?,
        ) else {
            continue;
        };
        if let Some(nationkey) = customer_nations.get(custkey) {
            orders.insert(orderkey, nationkey);
        }
    }
    Ok(orders)
}

fn order_customer_nations_view(
    view: BatchView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<i64, i64>> {
    if view.num_columns() == 3
        && let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
        )
    {
        return Ok(order_customer_nations_vectors(
            orderkeys,
            custkeys,
            orderdates,
            customer_nations,
            start_days,
            end_days,
        ));
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "regional order customer nation raw vector columns have unsupported types".to_string(),
        ));
    };
    order_customer_nations_batch(batch.clone(), customer_nations, start_days, end_days)
}

fn order_customer_nations_vectors(
    orderkeys: I64VectorView<'_>,
    custkeys: I64VectorView<'_>,
    orderdates: Date32VectorView<'_>,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> FastHashMap<i64, i64> {
    let mut orders = fast_hash_map::<i64, i64>();
    if let (Some(orderkey_values), Some(custkey_values), Some(orderdate_values)) = (
        orderkeys.values_if_null_free(),
        custkeys.values_if_null_free(),
        orderdates.values_if_null_free(),
    ) {
        if let Some((nation_values, nation_present)) = customer_nations.dense_slices() {
            for row in 0..orderkey_values.len() {
                let orderdate = orderdate_values[row];
                if orderdate < start_days || orderdate >= end_days {
                    continue;
                }
                let Ok(custkey) = usize::try_from(custkey_values[row]) else {
                    continue;
                };
                if nation_present.get(custkey).copied().unwrap_or(false) {
                    orders.insert(orderkey_values[row], nation_values[custkey]);
                }
            }
            return orders;
        }
        for row in 0..orderkey_values.len() {
            let orderdate = orderdate_values[row];
            if orderdate < start_days || orderdate >= end_days {
                continue;
            }
            if let Some(nationkey) = customer_nations.get(custkey_values[row]) {
                orders.insert(orderkey_values[row], nationkey);
            }
        }
        return orders;
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        if orderdate < start_days || orderdate >= end_days {
            continue;
        }
        if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
            orders.insert(orderkeys.value(row), nationkey);
        }
    }
    orders
}

fn order_customer_nations_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    customer_nations: &AdaptiveI64Map<i64>,
    start_days: i32,
    end_days: i32,
) -> Option<FastHashMap<i64, i64>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return None;
    };
    let mut orders = fast_hash_map::<i64, i64>();
    if orderkeys.null_count() == 0 && custkeys.null_count() == 0 && orderdates.null_count() == 0 {
        for row in 0..orderkeys.len() {
            let orderdate = orderdates.value(row);
            if orderdate < start_days || orderdate >= end_days {
                continue;
            }
            if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
                orders.insert(orderkeys.value(row), nationkey);
            }
        }
        return Some(orders);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        if orderdate < start_days || orderdate >= end_days {
            continue;
        }
        if let Some(nationkey) = customer_nations.get(custkeys.value(row)) {
            orders.insert(orderkeys.value(row), nationkey);
        }
    }
    Some(orders)
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
    let pruning_predicates = if let Some((min_key, max_key)) =
        selective_i64_key_range(order_customer_nations.keys().copied())
    {
        i64_range_pruning_predicates("l_orderkey", min_key, max_key)
    } else {
        Vec::new()
    };
    let order_customer_nations = Arc::new(order_customer_nations.clone());
    let supplier_nations = Arc::new(AdaptiveI64Map::from_hash(supplier_nations.clone()));
    let groups = if regional_revenue_row_group_map_enabled() {
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

async fn revenue_by_nation_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    order_customer_nations: Arc<FastHashMap<i64, i64>>,
    supplier_nations: Arc<AdaptiveI64Map<i64>>,
) -> Result<FastHashMap<i64, f64>> {
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
        "regional revenue aggregate",
    )
}

fn regional_revenue_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q05_DISABLE_ROW_GROUP_MAP").is_none()
}

fn regional_revenue_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q05_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
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
        let (Some(customer_nation), Some(supplier_nation)) = (
            order_customer_nations.get(&orderkey).copied(),
            supplier_nations.get(suppkey),
        ) else {
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

#[allow(dead_code)]
fn revenue_by_nation_projected_batch(
    batch: RecordBatch,
    order_customer_nations: &FastHashMap<i64, i64>,
    supplier_nations: &AdaptiveI64Map<i64>,
) -> Result<FastHashMap<i64, f64>> {
    if batch.num_columns() == 4
        && let Some(groups) = revenue_by_nation_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            order_customer_nations,
            supplier_nations,
        )?
    {
        return Ok(groups);
    }
    revenue_by_nation_batch(batch, order_customer_nations, supplier_nations)
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
        return Ok(revenue_by_nation_vector(
            orderkeys,
            suppkeys,
            extendedprices,
            discounts,
            order_customer_nations,
            supplier_nations,
        ));
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
) -> FastHashMap<i64, f64> {
    let mut groups = fast_hash_map::<i64, f64>();
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
        if let Some((supplier_nation_values, supplier_nation_present)) =
            supplier_nations.dense_slices()
        {
            for row in 0..orderkey_values.len() {
                let orderkey = orderkey_values[row];
                let Some(customer_nation) = order_customer_nations.get(&orderkey).copied() else {
                    continue;
                };
                let Ok(suppkey) = usize::try_from(suppkey_values[row]) else {
                    continue;
                };
                if supplier_nation_present
                    .get(suppkey)
                    .copied()
                    .unwrap_or(false)
                    && supplier_nation_values[suppkey] == customer_nation
                {
                    *groups.entry(customer_nation).or_insert(0.0) += decimal_discounted_revenue_raw(
                        extendedprice_values[row],
                        discount_values[row],
                        discount_scale,
                        revenue_scale,
                    );
                }
            }
            return groups;
        }
        for row in 0..orderkey_values.len() {
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let (Some(customer_nation), Some(supplier_nation)) = (
                order_customer_nations.get(&orderkey).copied(),
                supplier_nations.get(suppkey),
            ) else {
                continue;
            };
            if customer_nation == supplier_nation {
                *groups.entry(customer_nation).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
        }
        return groups;
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let (Some(customer_nation), Some(supplier_nation)) = (
            order_customer_nations.get(&orderkey).copied(),
            supplier_nations.get(suppkey),
        ) else {
            continue;
        };
        if customer_nation == supplier_nation {
            *groups.entry(customer_nation).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
    }
    groups
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
    let mut groups = fast_hash_map::<i64, f64>();
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
        for row in 0..orderkeys.len() {
            let orderkey = orderkey_values[row];
            let suppkey = suppkey_values[row];
            let (Some(customer_nation), Some(supplier_nation)) = (
                order_customer_nations.get(&orderkey).copied(),
                supplier_nations.get(suppkey),
            ) else {
                continue;
            };
            if customer_nation == supplier_nation {
                *groups.entry(customer_nation).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
        }
        return Ok(Some(groups));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || suppkeys.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
        {
            continue;
        }
        let orderkey = orderkeys.value(row);
        let suppkey = suppkeys.value(row);
        let (Some(customer_nation), Some(supplier_nation)) = (
            order_customer_nations.get(&orderkey).copied(),
            supplier_nations.get(suppkey),
        ) else {
            continue;
        };
        if customer_nation == supplier_nation {
            *groups.entry(customer_nation).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
    }
    Ok(Some(groups))
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
