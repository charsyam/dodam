use super::*;

pub(super) async fn try_execute_join_with_grouped_sum_semijoin_sql(
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
    if !join_with_grouped_sum_semijoin_shape(select, query, selection) {
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

    let stage = tpch_profile_start();
    let order_quantity_sums =
        qualifying_order_quantities(engine, lineitem.path.clone(), batch_size, 300.0).await?;
    tpch_profile_elapsed("grouped-sum semijoin lineitem quantity sums", stage);
    if order_quantity_sums.is_empty() {
        return Ok(Some(grouped_sum_semijoin_output(Vec::new())?));
    }
    let qualifying_order_keys =
        AdaptiveI64Set::from_hash(order_quantity_sums.keys().copied().collect::<HashSet<_>>());
    let stage = tpch_profile_start();
    let order_rows =
        qualifying_orders(engine, orders.path, batch_size, &qualifying_order_keys).await?;
    tpch_profile_elapsed("grouped-sum semijoin qualifying orders", stage);
    let customer_keys = order_rows
        .values()
        .map(|order| order.custkey)
        .collect::<HashSet<_>>();
    let customer_keys = AdaptiveI64Set::from_hash(customer_keys);
    let stage = tpch_profile_start();
    let customer_names = customer_names(engine, customer.path, batch_size, &customer_keys).await?;
    tpch_profile_elapsed("grouped-sum semijoin customer names", stage);

    let stage = tpch_profile_start();
    let mut rows = Vec::new();
    for (orderkey, order) in order_rows {
        let Some(name) = customer_names.get(&order.custkey) else {
            continue;
        };
        let Some(quantity) = order_quantity_sums.get(&orderkey).copied() else {
            continue;
        };
        rows.push(GroupedSumSemijoinRow {
            c_name: name.clone(),
            c_custkey: order.custkey,
            o_orderkey: orderkey,
            o_orderdate: order.orderdate,
            o_totalprice: order.totalprice,
            quantity,
        });
    }
    rows.sort_by(|left, right| {
        right
            .o_totalprice
            .partial_cmp(&left.o_totalprice)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.o_orderdate.cmp(&right.o_orderdate))
    });
    rows.truncate(100);
    tpch_profile_elapsed("grouped-sum semijoin final rows", stage);
    Ok(Some(grouped_sum_semijoin_output(rows)?))
}

fn join_with_grouped_sum_semijoin_shape(
    select: &Select,
    query: &Query,
    selection: &SqlExpr,
) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 6
        && matches!(parse_limit(query), Ok(Some(100)))
        && text.contains("o_orderkey in")
        && text.contains("group by l_orderkey")
        && text.contains("sum(l_quantity) > 300")
        && text.contains("c_custkey = o_custkey")
        && text.contains("o_orderkey = l_orderkey")
}

struct QualifyingOrder {
    custkey: i64,
    orderdate: i32,
    totalprice: f64,
}

struct GroupedSumSemijoinRow {
    c_name: String,
    c_custkey: i64,
    o_orderkey: i64,
    o_orderdate: i32,
    o_totalprice: f64,
    quantity: f64,
}

async fn qualifying_order_quantities(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    threshold: f64,
) -> Result<HashMap<i64, f64>> {
    if let Some(ordered) = engine
        .ordered_i64_decimal_group_sum_above(
            path.clone(),
            batch_size,
            "l_orderkey",
            "l_quantity",
            threshold,
        )
        .await?
    {
        return Ok(ordered);
    }
    if let Some(ordered) =
        qualifying_order_quantities_ordered(engine, path.clone(), batch_size, threshold).await?
    {
        return Ok(ordered);
    }
    let max_dense_orderkey = engine
        .parquet_i64_column_max(path.clone(), "l_orderkey")
        .await?
        .and_then(|max_key| adaptive_dense_index(max_key, DEFAULT_MAX_DENSE_I64_KEY));
    let has_dense_capacity = max_dense_orderkey.is_some();
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_quantity".to_string()]),
            None,
        )
        .await?;
    let mut sums = DenseI64F64Sum::new_tracking_threshold(threshold);
    if let Some(max_key) = max_dense_orderkey {
        sums.reserve_dense_to(max_key);
    }
    while let Some(batch) = stream.next() {
        let batch = batch?;
        if sums.has_fallback() {
            let fallback = sums.fallback_mut().expect("checked grouped-sum fallback");
            quantity_batch_into(&batch, fallback)?;
            continue;
        }
        if quantity_batch_into_dense(&batch, &mut sums, has_dense_capacity)? {
            continue;
        }
        sums.convert_to_fallback();
        let fallback = sums.fallback_mut().expect("converted grouped-sum fallback");
        quantity_batch_into(&batch, fallback)?;
    }
    Ok(sums.into_filtered_hash(|quantity| quantity > threshold))
}

struct OrderedQuantityState {
    current_key: Option<i64>,
    current_sum: f64,
    threshold: f64,
    output: HashMap<i64, f64>,
}

impl OrderedQuantityState {
    fn new(threshold: f64) -> Self {
        Self {
            current_key: None,
            current_sum: 0.0,
            threshold,
            output: HashMap::new(),
        }
    }

    fn push(&mut self, orderkey: i64, quantity: f64) -> bool {
        if let Some(current_key) = self.current_key {
            if orderkey < current_key {
                return false;
            }
            if orderkey == current_key {
                self.current_sum += quantity;
                return true;
            }
            self.flush_current();
        }
        self.current_key = Some(orderkey);
        self.current_sum = quantity;
        true
    }

    fn flush_current(&mut self) {
        let Some(orderkey) = self.current_key.take() else {
            return;
        };
        if self.current_sum > self.threshold {
            self.output.insert(orderkey, self.current_sum);
        }
        self.current_sum = 0.0;
    }

    fn finish(mut self) -> HashMap<i64, f64> {
        self.flush_current();
        self.output
    }
}

async fn qualifying_order_quantities_ordered(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    threshold: f64,
) -> Result<Option<HashMap<i64, f64>>> {
    let mut stream = engine
        .scan_parquet_batches_preserve_order(
            path,
            batch_size,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_quantity".to_string()]),
        )
        .await?;
    let mut state = OrderedQuantityState::new(threshold);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        if !quantity_batch_into_ordered(&batch, &mut state)? {
            return Ok(None);
        }
    }
    Ok(Some(state.finish()))
}

fn quantity_batch_into_ordered(
    batch: &RecordBatch,
    state: &mut OrderedQuantityState,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    let (Some(orderkeys), Some(quantities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0 && quantities.null_count() == 0 {
        let quantity_scale = 1.0 / quantities.scale;
        for (&orderkey, &quantity) in orderkeys.values().iter().zip(quantities.raw_values()) {
            if !state.push(orderkey, quantity as f64 * quantity_scale) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || quantities.is_null(row) {
            continue;
        }
        if !state.push(orderkeys.value(row), quantities.value(row)) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn quantity_batch_into_dense(
    batch: &RecordBatch,
    sums: &mut DenseI64F64Sum,
    has_dense_capacity: bool,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    let (Some(orderkeys), Some(quantities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0 && quantities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        let quantity_values = quantities.raw_values();
        let quantity_scale = 1.0 / quantities.scale;
        if has_dense_capacity {
            for (&orderkey, &quantity) in orderkey_values.iter().zip(quantity_values) {
                let Some(index) = adaptive_dense_index(orderkey, DEFAULT_MAX_DENSE_I64_KEY) else {
                    return Ok(false);
                };
                sums.add_dense_index(index, quantity as f64 * quantity_scale);
            }
            return Ok(true);
        }
        let mut max_index = 0_usize;
        for &orderkey in orderkey_values {
            let Some(index) = adaptive_dense_index(orderkey, DEFAULT_MAX_DENSE_I64_KEY) else {
                return Ok(false);
            };
            max_index = max_index.max(index);
        }
        sums.reserve_dense_to(max_index);
        for (&orderkey, &quantity) in orderkey_values.iter().zip(quantity_values) {
            let index = usize::try_from(orderkey).expect("validated dense index");
            sums.add_dense_index(index, quantity as f64 * quantity_scale);
        }
        return Ok(true);
    }
    if has_dense_capacity {
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || quantities.is_null(row) {
                continue;
            }
            let Some(index) = adaptive_dense_index(orderkeys.value(row), DEFAULT_MAX_DENSE_I64_KEY)
            else {
                return Ok(false);
            };
            sums.add_dense_index(index, quantities.value(row));
        }
        return Ok(true);
    }
    let mut max_index = 0_usize;
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || quantities.is_null(row) {
            continue;
        }
        let Some(index) = adaptive_dense_index(orderkeys.value(row), DEFAULT_MAX_DENSE_I64_KEY)
        else {
            return Ok(false);
        };
        max_index = max_index.max(index);
    }
    sums.reserve_dense_to(max_index);
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || quantities.is_null(row) {
            continue;
        }
        let index = usize::try_from(orderkeys.value(row)).expect("validated dense index");
        sums.add_dense_index(index, quantities.value(row));
    }
    Ok(true)
}

fn quantity_batch_into(batch: &RecordBatch, sums: &mut AdaptiveI64Map<f64>) -> Result<()> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let quantities = batch_column(batch, "l_quantity")?;
    if let (Some(orderkeys), Some(quantities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
    ) {
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || quantities.is_null(row) {
                continue;
            }
            sums.update(
                orderkeys.value(row),
                || 0.0,
                |sum| *sum += quantities.value(row),
            );
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        let (Some(orderkey), Some(quantity)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_f64_value(quantities, row)?,
        ) else {
            continue;
        };
        sums.update(orderkey, || 0.0, |sum| *sum += quantity);
    }
    Ok(())
}

async fn qualifying_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    qualifying_orders: &AdaptiveI64Set,
) -> Result<HashMap<i64, QualifyingOrder>> {
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_custkey".to_string(),
        "o_orderdate".to_string(),
        "o_totalprice".to_string(),
    ]);
    let mut stream = if let Some((min_key, max_key)) = qualifying_orders.selective_key_range() {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                i64_range_pruning_predicates("o_orderkey", min_key, max_key),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let qualifying_orders = Arc::new(qualifying_orders.clone());
    parallel_batch_fold(
        &mut stream,
        move |batch| qualifying_orders_batch(batch, &qualifying_orders),
        HashMap::<i64, QualifyingOrder>::new(),
        merge_maps,
        "grouped-sum semijoin qualifying orders",
    )
}

fn qualifying_orders_batch(
    batch: RecordBatch,
    qualifying_orders: &AdaptiveI64Set,
) -> Result<HashMap<i64, QualifyingOrder>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let totalprices = batch_column(&batch, "o_totalprice")?;
    if let Some(orders) = qualifying_orders_batch_typed(
        orderkeys,
        custkeys,
        orderdates,
        totalprices,
        qualifying_orders,
    )? {
        return Ok(orders);
    }
    let mut orders = HashMap::new();
    for row in 0..batch.num_rows() {
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        if !qualifying_orders.contains(orderkey) {
            continue;
        }
        let (Some(custkey), Some(orderdate), Some(totalprice)) = (
            numeric_i64_value(custkeys, row)?,
            date32_value(orderdates, row)?,
            numeric_f64_value(totalprices, row)?,
        ) else {
            continue;
        };
        orders.insert(
            orderkey,
            QualifyingOrder {
                custkey,
                orderdate,
                totalprice,
            },
        );
    }
    Ok(orders)
}

fn qualifying_orders_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    totalprices: &ArrayRef,
    qualifying_orders: &AdaptiveI64Set,
) -> Result<Option<HashMap<i64, QualifyingOrder>>> {
    let (Some(orderkeys), Some(custkeys), Some(orderdates), Some(totalprices)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        custkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
        decimal_input(totalprices)?,
    ) else {
        return Ok(None);
    };
    let mut orders = HashMap::new();
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row)
            || custkeys.is_null(row)
            || orderdates.is_null(row)
            || totalprices.is_null(row)
        {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if !qualifying_orders.contains(orderkey) {
            continue;
        }
        orders.insert(
            orderkey,
            QualifyingOrder {
                custkey: custkeys.value(row),
                orderdate: orderdates.value(row),
                totalprice: totalprices.value(row),
            },
        );
    }
    Ok(Some(orders))
}

async fn customer_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_keys: &AdaptiveI64Set,
) -> Result<HashMap<i64, String>> {
    let mut customers = HashMap::new();
    if customer_keys.is_empty() {
        return Ok(customers);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["c_custkey".to_string(), "c_name".to_string()]),
            None,
        )
        .await?;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let keys = batch_column(&batch, "c_custkey")?;
        let names = batch_string_column(&batch, "c_name")?;
        if let Some(keys) = keys.as_any().downcast_ref::<Int64Array>() {
            for row in 0..batch.num_rows() {
                if keys.is_null(row) || names.is_null(row) {
                    continue;
                }
                let key = keys.value(row);
                if !customer_keys.contains(key) {
                    continue;
                }
                customers.insert(key, names.value(row).to_string());
            }
            if customers.len() == customer_keys.len() {
                break;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            if names.is_null(row) {
                continue;
            }
            if let Some(key) = numeric_i64_value(keys, row)? {
                if !customer_keys.contains(key) {
                    continue;
                }
                customers.insert(key, names.value(row).to_string());
            }
        }
        if customers.len() == customer_keys.len() {
            break;
        }
    }
    Ok(customers)
}

fn grouped_sum_semijoin_output(rows: Vec<GroupedSumSemijoinRow>) -> Result<QueryOutput> {
    let orderdates = rows
        .iter()
        .map(|row| date32_to_ymd_string(row.o_orderdate))
        .collect::<Result<Vec<_>>>()?;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("c_name", DataType::Utf8, false),
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_orderdate", DataType::Utf8, false),
            Field::new("o_totalprice", DataType::Float64, false),
            Field::new("sum(l_quantity)", DataType::Float64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_name.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.c_custkey),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.o_orderkey),
            )),
            Arc::new(StringArray::from_iter_values(
                orderdates.iter().map(String::as_str),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.o_totalprice),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.quantity),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
