use super::*;

fn returned_customer_revenue_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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
    select.from.len() == 4
        && select.projection.len() == 8
        && matches!(parse_limit(query), Ok(Some(20)))
        && projection.contains("c_custkey")
        && projection.contains("c_name")
        && projection.contains("sum(l_extendedprice * (1 - l_discount))")
        && projection.contains("c_acctbal")
        && projection.contains("n_name")
        && projection.contains("c_address")
        && projection.contains("c_phone")
        && projection.contains("c_comment")
        && group_by.contains("c_custkey")
        && group_by.contains("c_name")
        && group_by.contains("c_acctbal")
        && group_by.contains("n_name")
        && order_by.contains("revenue desc")
        && selection.contains("c_custkey = o_custkey")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("l_returnflag = 'r'")
        && selection.contains("c_nationkey = n_nationkey")
}

pub(super) async fn try_execute_returned_customer_revenue_sql(
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
    if !returned_customer_revenue_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 4 {
        return Ok(None);
    }
    let mut customer = None;
    let mut orders = None;
    let mut lineitem = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("customer") {
            customer = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(customer), Some(orders), Some(lineitem), Some(nation)) =
        (customer, orders, lineitem, nation)
    else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };

    let nation_names = nation_names_by_keys(engine, nation.path, batch_size).await?;
    let order_customers =
        order_customer_keys(engine, orders.path, batch_size, start_days, end_days).await?;
    if order_customers.is_empty() {
        return Ok(Some(returned_customer_revenue_output(Vec::new())?));
    }
    let revenues =
        returned_revenue_by_customer(engine, lineitem.path, batch_size, &order_customers).await?;
    if revenues.is_empty() {
        return Ok(Some(returned_customer_revenue_output(Vec::new())?));
    }
    if let Some(rows) = returned_customer_topk_rows_with_late_payload(
        engine,
        customer.path.clone(),
        batch_size,
        &nation_names,
        &revenues,
        20,
    )
    .await?
    {
        return Ok(Some(returned_customer_revenue_output(rows)?));
    }
    let customer_key_filter = revenues.keys().copied().collect::<HashSet<_>>();
    let customer_keys = AdaptiveI64Set::from_hash(customer_key_filter.clone());
    let customers = returned_customer_rows(
        engine,
        customer.path,
        batch_size,
        &customer_keys,
        &customer_key_filter,
    )
    .await?;
    let rows = returned_customer_rows_from_payload(revenues, &customers, &nation_names, 20);
    Ok(Some(returned_customer_revenue_output(rows)?))
}

async fn returned_customer_topk_rows_with_late_payload<S: BuildHasher>(
    engine: &DodamEngine,
    customer_path: PathBuf,
    batch_size: usize,
    nation_names: &HashMap<i64, String>,
    revenues: &HashMap<i64, f64, S>,
    limit: usize,
) -> Result<Option<Vec<ReturnedCustomerRevenueRow>>> {
    let candidate_count = revenues.len().min(limit.saturating_mul(8).max(limit));
    if candidate_count == revenues.len() {
        return Ok(None);
    }
    let mut candidates = revenues
        .iter()
        .map(|(&custkey, &revenue)| (custkey, revenue))
        .collect::<Vec<_>>();
    candidates.select_nth_unstable_by(candidate_count, |left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    candidates.truncate(candidate_count);
    let candidate_revenues = candidates.iter().copied().collect::<HashMap<i64, f64>>();
    let customer_key_filter = candidate_revenues.keys().copied().collect::<HashSet<_>>();
    let customer_keys = AdaptiveI64Set::from_hash(customer_key_filter.clone());
    let customers = returned_customer_rows(
        engine,
        customer_path,
        batch_size,
        &customer_keys,
        &customer_key_filter,
    )
    .await?;
    let rows =
        returned_customer_rows_from_payload(candidate_revenues, &customers, nation_names, limit);
    Ok((rows.len() == limit).then_some(rows))
}

fn returned_customer_rows_from_payload<S: BuildHasher>(
    revenues: HashMap<i64, f64, S>,
    customers: &HashMap<i64, ReturnedCustomer>,
    nation_names: &HashMap<i64, String>,
    limit: usize,
) -> Vec<ReturnedCustomerRevenueRow> {
    let mut rows = revenues
        .into_iter()
        .filter_map(|(c_custkey, revenue)| {
            let customer = customers.get(&c_custkey)?;
            let n_name = nation_names.get(&customer.c_nationkey)?;
            Some(ReturnedCustomerRevenueRow {
                c_custkey,
                c_name: customer.c_name.clone(),
                revenue,
                c_acctbal: customer.c_acctbal,
                n_name: n_name.clone(),
                c_address: customer.c_address.clone(),
                c_phone: customer.c_phone.clone(),
                c_comment: customer.c_comment.clone(),
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .revenue
            .partial_cmp(&left.revenue)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.c_custkey.cmp(&right.c_custkey))
    });
    rows.truncate(limit);
    rows
}

pub(super) fn date_range_bounds(conjuncts: &[SqlExpr], column: &str) -> Result<Option<(i32, i32)>> {
    let mut start = None;
    let mut end = None;
    for conjunct in conjuncts {
        let SqlExpr::BinaryOp { left, op, right } = conjunct else {
            continue;
        };
        if matches!(op, BinaryOperator::GtEq | BinaryOperator::Gt)
            && sql_expr_column_matches(left, column)
        {
            if let Some(days) = maybe_literal_date_days(right)? {
                start = Some(days);
            }
        } else if matches!(op, BinaryOperator::Lt | BinaryOperator::LtEq)
            && sql_expr_column_matches(left, column)
        {
            if let Some(days) = maybe_literal_date_days(right)? {
                end = Some(days);
            }
        } else if matches!(op, BinaryOperator::LtEq | BinaryOperator::Lt)
            && sql_expr_column_matches(right, column)
        {
            if let Some(days) = maybe_literal_date_days(left)? {
                start = Some(days);
            }
        } else if matches!(op, BinaryOperator::Gt | BinaryOperator::GtEq)
            && sql_expr_column_matches(right, column)
        {
            if let Some(days) = maybe_literal_date_days(left)? {
                end = Some(days);
            }
        }
    }
    Ok(start.zip(end))
}

pub(super) fn maybe_literal_date_days(expr: &SqlExpr) -> Result<Option<i32>> {
    match sql_literal_value(expr) {
        Ok(LiteralValue::Utf8(value)) => {
            let (year, month, day) = parse_ymd(&value)?;
            let days = days_from_civil(year, month, day)?;
            Ok(Some(i32::try_from(days).map_err(|_| {
                DodamError::UnsupportedSql("DATE overflow".to_string())
            })?))
        }
        Ok(_) => Ok(None),
        Err(DodamError::UnsupportedSql(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

struct ReturnedCustomer {
    c_name: String,
    c_acctbal: f64,
    c_nationkey: i64,
    c_address: String,
    c_phone: String,
    c_comment: String,
}

struct ReturnedCustomerRevenueRow {
    c_custkey: i64,
    c_name: String,
    revenue: f64,
    c_acctbal: f64,
    n_name: String,
    c_address: String,
    c_phone: String,
    c_comment: String,
}

pub(super) async fn nation_names_by_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["n_nationkey".to_string(), "n_name".to_string()]),
            None,
        )
        .await?;
    let mut nations = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && let Some(nationkey) = numeric_i64_value(nationkeys, row)?
            {
                nations.insert(nationkey, names.value(row).to_string());
            }
        }
    }
    Ok(nations)
}

async fn returned_customer_rows(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_keys: &AdaptiveI64Set,
    customer_key_filter: &HashSet<i64>,
) -> Result<HashMap<i64, ReturnedCustomer>> {
    if customer_late_materialized_enabled()
        && let Some(customers) = returned_customer_rows_late_materialized(
            engine,
            path.clone(),
            batch_size,
            customer_keys.clone(),
            customer_key_filter.clone(),
        )
        .await?
    {
        return Ok(customers);
    }
    let projection = Projection::Columns(vec![
        "c_custkey".to_string(),
        "c_name".to_string(),
        "c_acctbal".to_string(),
        "c_nationkey".to_string(),
        "c_address".to_string(),
        "c_phone".to_string(),
        "c_comment".to_string(),
    ]);
    let mut stream = if should_use_i64_set_row_filter_for_keys(
        true,
        "DODAM_Q10_DISABLE_CUSTOMER_ROW_FILTER",
        None,
        customer_key_filter,
        projection_column_count(&projection),
    ) {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "c_custkey",
                customer_key_filter.clone(),
            )
            .await?
    } else if let Some((min_key, max_key)) = customer_keys.selective_key_range() {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                i64_range_pruning_predicates("c_custkey", min_key, max_key),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let mut customers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let custkeys = batch_column(&batch, "c_custkey")?;
        let names = batch_string_column(&batch, "c_name")?;
        let acctbals = batch_column(&batch, "c_acctbal")?;
        let nationkeys = batch_column(&batch, "c_nationkey")?;
        let addresses = batch_string_column(&batch, "c_address")?;
        let phones = batch_string_column(&batch, "c_phone")?;
        let comments = batch_string_column(&batch, "c_comment")?;
        if let (Some(custkeys), Some(acctbals), Some(nationkeys)) = (
            custkeys.as_any().downcast_ref::<Int64Array>(),
            decimal_input(acctbals)?,
            nationkeys.as_any().downcast_ref::<Int64Array>(),
        ) {
            for row in 0..batch.num_rows() {
                if custkeys.is_null(row)
                    || acctbals.is_null(row)
                    || nationkeys.is_null(row)
                    || names.is_null(row)
                    || addresses.is_null(row)
                    || phones.is_null(row)
                    || comments.is_null(row)
                {
                    continue;
                }
                let custkey = custkeys.value(row);
                if !customer_keys.contains(custkey) {
                    continue;
                }
                customers.insert(
                    custkey,
                    ReturnedCustomer {
                        c_name: names.value(row).to_string(),
                        c_acctbal: acctbals.value(row),
                        c_nationkey: nationkeys.value(row),
                        c_address: addresses.value(row).to_string(),
                        c_phone: phones.value(row).to_string(),
                        c_comment: comments.value(row).to_string(),
                    },
                );
            }
            if customers.len() == customer_keys.len() {
                break;
            }
            continue;
        }
        for row in 0..batch.num_rows() {
            let Some(custkey) = numeric_i64_value(custkeys, row)? else {
                continue;
            };
            if !customer_keys.contains(custkey) {
                continue;
            }
            let (Some(acctbal), Some(nationkey)) = (
                numeric_f64_value(acctbals, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if names.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
                || comments.is_null(row)
            {
                continue;
            }
            customers.insert(
                custkey,
                ReturnedCustomer {
                    c_name: names.value(row).to_string(),
                    c_acctbal: acctbal,
                    c_nationkey: nationkey,
                    c_address: addresses.value(row).to_string(),
                    c_phone: phones.value(row).to_string(),
                    c_comment: comments.value(row).to_string(),
                },
            );
        }
        if customers.len() == customer_keys.len() {
            break;
        }
    }
    Ok(customers)
}

fn customer_late_materialized_enabled() -> bool {
    std::env::var_os("DODAM_Q10_DISABLE_CUSTOMER_LATE").is_none()
}

async fn returned_customer_rows_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    customer_keys: AdaptiveI64Set,
    customer_key_filter: HashSet<i64>,
) -> Result<Option<HashMap<i64, ReturnedCustomer>>> {
    let customer_keys = Arc::new(customer_keys);
    let customer_key_filter = Arc::new(AdaptiveI64Set::from_hash(customer_key_filter));
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["c_custkey".to_string()]),
            Projection::Columns(vec![
                "c_name".to_string(),
                "c_acctbal".to_string(),
                "c_nationkey".to_string(),
                "c_address".to_string(),
                "c_phone".to_string(),
                "c_comment".to_string(),
            ]),
            Vec::new(),
            customer_late_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                customer_late_max_selected_ratio(),
                customer_late_max_selector_run_ratio(),
            ),
            {
                let customer_keys = customer_keys.clone();
                let customer_key_filter = customer_key_filter.clone();
                move || {
                    ReturnedCustomerLateState::new(
                        customer_key_filter.clone(),
                        CustomerLatePayload {
                            customer_keys: customer_keys.clone(),
                            customers: HashMap::new(),
                        },
                    )
                }
            },
            customer_late_build_selection_view,
            customer_late_consume_payload_view,
            |state, _metrics| {
                if !state.payload_consumed() {
                    return Err(DodamError::UnsupportedSql(
                        "Q10 customer payload row mismatch".to_string(),
                    ));
                }
                Ok(Some(state.payload.customers))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut customers = HashMap::new();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        customers.extend(chunk.output);
    }
    log_customer_late_profile(metrics, customer_late_row_group_chunk());
    Ok(Some(customers))
}

fn customer_late_row_group_chunk() -> usize {
    std::env::var("DODAM_Q10_CUSTOMER_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn customer_late_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q10_CUSTOMER_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.01)
}

fn customer_late_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q10_CUSTOMER_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.01)
}

struct I64SetLateSelectionState<T> {
    key_filter: Arc<AdaptiveI64Set>,
    selected_keys: Vec<i64>,
    payload_offset: usize,
    payload: T,
}

impl<T> I64SetLateSelectionState<T> {
    fn new(key_filter: Arc<AdaptiveI64Set>, payload: T) -> Self {
        Self {
            key_filter,
            selected_keys: Vec::new(),
            payload_offset: 0,
            payload,
        }
    }

    fn next_payload_key(&mut self, overflow_message: &str) -> Result<i64> {
        let Some(&key) = self.selected_keys.get(self.payload_offset) else {
            return Err(DodamError::UnsupportedSql(overflow_message.to_string()));
        };
        self.payload_offset += 1;
        Ok(key)
    }

    fn payload_consumed(&self) -> bool {
        self.payload_offset == self.selected_keys.len()
    }
}

#[derive(Default)]
pub(super) struct I64SetLateSelectionBatchMetrics {
    pub(super) total_rows: usize,
    pub(super) selected_rows: usize,
}

fn i64_set_late_build_selection_batch<T>(
    batch: RecordBatch,
    key_column: &str,
    selection: &mut LateSelectionBuilder,
    state: &mut I64SetLateSelectionState<T>,
) -> Result<Option<()>> {
    i64_set_late_build_selection_batch_into(
        batch,
        key_column,
        &state.key_filter,
        &mut state.selected_keys,
        selection,
    )
    .map(|metrics| metrics.map(|_| ()))
}

fn i64_set_late_build_selection_view<T>(
    view: BatchView<'_>,
    key_index: usize,
    selection: &mut LateSelectionBuilder,
    state: &mut I64SetLateSelectionState<T>,
) -> Result<Option<()>> {
    let Some(keys) = view.i64(key_index) else {
        return Ok(None);
    };
    let dense_key_filter = state.key_filter.dense_contains_slice();
    if keys.null_count() == 0 {
        for &key in keys.values().as_ref() {
            let selected = state.key_filter.contains_cached(dense_key_filter, key);
            selection.push(selected);
            if selected {
                state.selected_keys.push(key);
            }
        }
        return Ok(Some(()));
    }
    for row in 0..keys.len() {
        let selected = keys.is_valid(row)
            && state
                .key_filter
                .contains_cached(dense_key_filter, keys.value(row));
        selection.push(selected);
        if selected {
            state.selected_keys.push(keys.value(row));
        }
    }
    Ok(Some(()))
}

pub(super) fn i64_set_late_build_selection_batch_into(
    batch: RecordBatch,
    key_column: &str,
    key_filter: &AdaptiveI64Set,
    selected_keys: &mut Vec<i64>,
    selection: &mut LateSelectionBuilder,
) -> Result<Option<I64SetLateSelectionBatchMetrics>> {
    let keys = batch_column(&batch, key_column)?;
    let Some(keys) = keys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    let dense_key_filter = key_filter.dense_contains_slice();
    let mut metrics = I64SetLateSelectionBatchMetrics {
        total_rows: keys.len(),
        selected_rows: 0,
    };
    if keys.null_count() == 0 {
        for &key in keys.values().as_ref() {
            let selected = key_filter.contains_cached(dense_key_filter, key);
            selection.push(selected);
            if selected {
                selected_keys.push(key);
                metrics.selected_rows += 1;
            }
        }
        return Ok(Some(metrics));
    }
    for row in 0..keys.len() {
        let selected =
            keys.is_valid(row) && key_filter.contains_cached(dense_key_filter, keys.value(row));
        selection.push(selected);
        if selected {
            selected_keys.push(keys.value(row));
            metrics.selected_rows += 1;
        }
    }
    Ok(Some(metrics))
}

struct CustomerLatePayload {
    customer_keys: Arc<AdaptiveI64Set>,
    customers: HashMap<i64, ReturnedCustomer>,
}

type ReturnedCustomerLateState = I64SetLateSelectionState<CustomerLatePayload>;

fn customer_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut ReturnedCustomerLateState,
) -> Result<Option<()>> {
    i64_set_late_build_selection_batch(batch, "c_custkey", selection, state)
}

fn customer_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut ReturnedCustomerLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        return i64_set_late_build_selection_view(view, 0, selection, state);
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    customer_late_build_selection_batch(batch.clone(), selection, state)
}

fn customer_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut ReturnedCustomerLateState,
) -> Result<Option<()>> {
    let names = batch_string_column(&batch, "c_name")?;
    let acctbals = batch_column(&batch, "c_acctbal")?;
    let nationkeys = batch_column(&batch, "c_nationkey")?;
    let addresses = batch_string_column(&batch, "c_address")?;
    let phones = batch_string_column(&batch, "c_phone")?;
    let comments = batch_string_column(&batch, "c_comment")?;
    let (Some(acctbals), Some(nationkeys)) = (
        decimal_input(acctbals)?,
        nationkeys.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return Ok(None);
    };
    for row in 0..batch.num_rows() {
        let custkey = state.next_payload_key("Q10 customer payload row overflow")?;
        if !state.payload.customer_keys.contains(custkey)
            || names.is_null(row)
            || acctbals.is_null(row)
            || nationkeys.is_null(row)
            || addresses.is_null(row)
            || phones.is_null(row)
            || comments.is_null(row)
        {
            continue;
        }
        state.payload.customers.insert(
            custkey,
            ReturnedCustomer {
                c_name: names.value(row).to_string(),
                c_acctbal: acctbals.value(row),
                c_nationkey: nationkeys.value(row),
                c_address: addresses.value(row).to_string(),
                c_phone: phones.value(row).to_string(),
                c_comment: comments.value(row).to_string(),
            },
        );
    }
    Ok(Some(()))
}

fn customer_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut ReturnedCustomerLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 6 {
        let (
            Some(names),
            Some(acctbals),
            Some(nationkeys),
            Some(addresses),
            Some(phones),
            Some(comments),
        ) = (
            view.utf8_vector(0),
            view.decimal128_vector(1),
            view.i64_vector(2),
            view.utf8_vector(3),
            view.utf8_vector(4),
            view.utf8_vector(5),
        )
        else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return customer_late_consume_payload_batch(batch.clone(), state);
        };
        for row in 0..view.num_rows() {
            let custkey = state.next_payload_key("Q10 customer payload row overflow")?;
            if !state.payload.customer_keys.contains(custkey)
                || names.is_null(row)
                || acctbals.is_null(row)
                || nationkeys.is_null(row)
                || addresses.is_null(row)
                || phones.is_null(row)
                || comments.is_null(row)
            {
                continue;
            }
            let c_name = std::str::from_utf8(names.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            let c_address = std::str::from_utf8(addresses.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            let c_phone = std::str::from_utf8(phones.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            let c_comment = std::str::from_utf8(comments.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            state.payload.customers.insert(
                custkey,
                ReturnedCustomer {
                    c_name: c_name.to_string(),
                    c_acctbal: acctbals.value(row),
                    c_nationkey: nationkeys.value(row),
                    c_address: c_address.to_string(),
                    c_phone: c_phone.to_string(),
                    c_comment: c_comment.to_string(),
                },
            );
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    customer_late_consume_payload_batch(batch.clone(), state)
}

fn log_customer_late_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] returned customers: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

async fn order_customer_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
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
    parallel_batch_fold_view_chunks(
        &mut stream,
        4,
        fast_hash_map::<i64, i64>,
        move |view, orders| {
            merge_maps(
                orders,
                order_customer_keys_view(view, start_days, end_days)?,
            );
            Ok(Some(()))
        },
        Ok,
        fast_hash_map::<i64, i64>(),
        merge_maps,
        "returned order customers",
    )
}

fn order_customer_keys_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
) -> Result<FastHashMap<i64, i64>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let custkeys = batch_column(&batch, "o_custkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    if let Some(orders) =
        order_customer_keys_batch_typed(orderkeys, custkeys, orderdates, start_days, end_days)?
    {
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
        orders.insert(orderkey, custkey);
    }
    Ok(orders)
}

fn order_customer_keys_view(
    view: BatchView<'_>,
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
        let mut orders = fast_hash_map::<i64, i64>();
        if let (Some(orderkey_values), Some(custkey_values), Some(orderdate_values)) = (
            orderkeys.values_if_null_free(),
            custkeys.values_if_null_free(),
            orderdates.values_if_null_free(),
        ) {
            for row in 0..view.num_rows() {
                let orderdate = orderdate_values[row];
                if orderdate >= start_days && orderdate < end_days {
                    orders.insert(orderkey_values[row], custkey_values[row]);
                }
            }
            return Ok(orders);
        }
        for row in 0..view.num_rows() {
            if orderkeys.is_null(row) || custkeys.is_null(row) || orderdates.is_null(row) {
                continue;
            }
            let orderdate = orderdates.value(row);
            if orderdate >= start_days && orderdate < end_days {
                orders.insert(orderkeys.value(row), custkeys.value(row));
            }
        }
        return Ok(orders);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q10 order customer raw vector columns have unsupported types".to_string(),
        ));
    };
    order_customer_keys_batch(batch.clone(), start_days, end_days)
}

fn order_customer_keys_batch_typed(
    orderkeys: &ArrayRef,
    custkeys: &ArrayRef,
    orderdates: &ArrayRef,
    start_days: i32,
    end_days: i32,
) -> Result<Option<FastHashMap<i64, i64>>> {
    let mut orders = fast_hash_map::<i64, i64>();
    if !try_for_each_i64_i64_date32(
        orderkeys,
        custkeys,
        orderdates,
        |orderkey, custkey, orderdate| {
            if orderdate >= start_days && orderdate < end_days {
                orders.insert(orderkey, custkey);
            }
            Ok(())
        },
    )? {
        return Ok(None);
    };
    Ok(Some(orders))
}

async fn returned_revenue_by_customer(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_customers: &FastHashMap<i64, i64>,
) -> Result<FastHashMap<i64, f64>> {
    if returned_revenue_late_enabled()
        && let Some(revenues) =
            returned_revenue_late(engine, path.clone(), batch_size, order_customers).await?
    {
        return Ok(revenues);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_returnflag".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            None,
        )
        .await?;
    let order_customers = Arc::new(order_customers.clone());
    parallel_batch_fold_view_chunks(
        &mut stream,
        4,
        fast_hash_map::<i64, f64>,
        move |view, revenues| {
            merge_f64_groups(revenues, returned_revenue_view(view, &order_customers)?);
            Ok(Some(()))
        },
        Ok,
        fast_hash_map::<i64, f64>(),
        merge_f64_groups,
        "returned revenue aggregate",
    )
}

async fn returned_revenue_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    order_customers: &FastHashMap<i64, i64>,
) -> Result<Option<FastHashMap<i64, f64>>> {
    let order_customers = Arc::new(order_customers.clone());
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["l_orderkey".to_string(), "l_returnflag".to_string()]),
            Projection::Columns(vec![
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
            ]),
            Vec::new(),
            returned_revenue_late_row_group_chunk(),
            late_materialization_policy_from_env(
                "DODAM_Q10_RETURNED_LATE_MAX_SELECTED_RATIO",
                0.50,
            ),
            {
                let order_customers = order_customers.clone();
                move || ReturnedRevenueLateState {
                    order_customers: order_customers.clone(),
                    selected_custkeys: Vec::new(),
                    payload_offset: 0,
                    revenues: fast_hash_map::<i64, f64>(),
                }
            },
            returned_revenue_late_build_selection_view,
            returned_revenue_late_consume_payload_view,
            |state, _metrics| {
                if state.payload_offset != state.selected_custkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "returned revenue payload row mismatch".to_string(),
                    ));
                }
                Ok(Some(state.revenues))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut revenues = fast_hash_map::<i64, f64>();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        merge_f64_groups(&mut revenues, chunk.output);
    }
    log_returned_revenue_late_profile(metrics, returned_revenue_late_row_group_chunk());
    Ok(Some(revenues))
}

fn returned_revenue_late_enabled() -> bool {
    std::env::var_os("DODAM_Q10_DISABLE_RETURNED_LATE").is_none()
}

fn returned_revenue_late_row_group_chunk() -> usize {
    std::env::var("DODAM_Q10_RETURNED_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

struct ReturnedRevenueLateState {
    order_customers: Arc<FastHashMap<i64, i64>>,
    selected_custkeys: Vec<i64>,
    payload_offset: usize,
    revenues: FastHashMap<i64, f64>,
}

fn returned_revenue_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut ReturnedRevenueLateState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let returnflags = batch_string_column(&batch, "l_returnflag")?;
    let Some(orderkeys) = orderkeys.as_any().downcast_ref::<Int64Array>() else {
        return Ok(None);
    };
    let returnflag_offsets = returnflags.value_offsets();
    let returnflag_data = returnflags.value_data();
    if orderkeys.null_count() == 0 && returnflags.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        for row in 0..orderkey_values.len() {
            let selected = utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
                && state
                    .order_customers
                    .get(&orderkey_values[row])
                    .copied()
                    .inspect(|custkey| state.selected_custkeys.push(*custkey))
                    .is_some();
            selection.push(selected);
        }
        return Ok(Some(()));
    }
    for row in 0..orderkeys.len() {
        let selected = orderkeys.is_valid(row)
            && returnflags.is_valid(row)
            && utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
            && state
                .order_customers
                .get(&orderkeys.value(row))
                .copied()
                .inspect(|custkey| state.selected_custkeys.push(*custkey))
                .is_some();
        selection.push(selected);
    }
    Ok(Some(()))
}

fn returned_revenue_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut ReturnedRevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2 {
        let (Some(orderkeys), Some(returnflags)) = (view.i64_vector(0), view.utf8_vector(1)) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return returned_revenue_late_build_selection_batch(batch.clone(), selection, state);
        };
        if let Some(orderkey_values) = orderkeys.values_if_null_free()
            && returnflags.null_count() == 0
        {
            for row in 0..orderkey_values.len() {
                let selected = returnflags.value_bytes(row) == b"R"
                    && state
                        .order_customers
                        .get(&orderkey_values[row])
                        .copied()
                        .inspect(|custkey| state.selected_custkeys.push(*custkey))
                        .is_some();
                selection.push(selected);
            }
            return Ok(Some(()));
        }
        for row in 0..orderkeys.len() {
            let selected = !orderkeys.is_null(row)
                && returnflags.is_valid(row)
                && returnflags.value_bytes(row) == b"R"
                && state
                    .order_customers
                    .get(&orderkeys.value(row))
                    .copied()
                    .inspect(|custkey| state.selected_custkeys.push(*custkey))
                    .is_some();
            selection.push(selected);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    returned_revenue_late_build_selection_batch(batch.clone(), selection, state)
}

fn returned_revenue_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut ReturnedRevenueLateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let (Some(extendedprices), Some(discounts)) =
        (decimal_input(extendedprices)?, decimal_input(discounts)?)
    else {
        return Ok(None);
    };
    if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
        let extendedprice_values = extendedprices.raw_values();
        let discount_values = discounts.raw_values();
        let (discount_scale, revenue_scale) =
            decimal_discounted_revenue_scales(extendedprices, discounts);
        for row in 0..batch.num_rows() {
            let Some(&custkey) = state.selected_custkeys.get(state.payload_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "returned revenue payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            *state.revenues.entry(custkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                extendedprice_values[row],
                discount_values[row],
                discount_scale,
                revenue_scale,
            );
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let Some(&custkey) = state.selected_custkeys.get(state.payload_offset) else {
            return Err(DodamError::UnsupportedSql(
                "returned revenue payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        if extendedprices.is_null(row) || discounts.is_null(row) {
            continue;
        }
        *state.revenues.entry(custkey).or_insert(0.0) +=
            extendedprices.value(row) * (1.0 - discounts.value(row));
    }
    Ok(Some(()))
}

fn returned_revenue_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut ReturnedRevenueLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2 {
        let (Some(extendedprices), Some(discounts)) =
            (view.decimal128_vector(0), view.decimal128_vector(1))
        else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return returned_revenue_late_consume_payload_batch(batch.clone(), state);
        };
        if extendedprices.null_count() == 0 && discounts.null_count() == 0 {
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            let discount_scale = discounts.scale();
            let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
            for row in 0..view.num_rows() {
                let Some(&custkey) = state.selected_custkeys.get(state.payload_offset) else {
                    return Err(DodamError::UnsupportedSql(
                        "returned revenue payload row overflow".to_string(),
                    ));
                };
                state.payload_offset += 1;
                *state.revenues.entry(custkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
            return Ok(Some(()));
        }
        for row in 0..view.num_rows() {
            let Some(&custkey) = state.selected_custkeys.get(state.payload_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "returned revenue payload row overflow".to_string(),
                ));
            };
            state.payload_offset += 1;
            if extendedprices.is_null(row) || discounts.is_null(row) {
                continue;
            }
            *state.revenues.entry(custkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    returned_revenue_late_consume_payload_batch(batch.clone(), state)
}

fn log_returned_revenue_late_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] returned revenue: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn returned_revenue_batch(
    batch: RecordBatch,
    order_customers: &FastHashMap<i64, i64>,
) -> Result<FastHashMap<i64, f64>> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let returnflags = batch_string_column(&batch, "l_returnflag")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let mut revenues = fast_hash_map::<i64, f64>();
    if let (Some(orderkeys), Some(extendedprices), Some(discounts)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) {
        let returnflag_offsets = returnflags.value_offsets();
        let returnflag_data = returnflags.value_data();
        if orderkeys.null_count() == 0
            && extendedprices.null_count() == 0
            && discounts.null_count() == 0
        {
            let orderkey_values = orderkeys.values().as_ref();
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            let (discount_scale, revenue_scale) =
                decimal_discounted_revenue_scales(extendedprices, discounts);
            for row in 0..batch.num_rows() {
                if returnflags.is_null(row)
                    || !utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
                {
                    continue;
                }
                let Some(custkey) = order_customers.get(&orderkey_values[row]).copied() else {
                    continue;
                };
                *revenues.entry(custkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
            return Ok(revenues);
        }
        for row in 0..batch.num_rows() {
            if returnflags.is_null(row)
                || !utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
                || orderkeys.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let Some(custkey) = order_customers.get(&orderkeys.value(row)).copied() else {
                continue;
            };
            *revenues.entry(custkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(revenues);
    }
    let returnflag_offsets = returnflags.value_offsets();
    let returnflag_data = returnflags.value_data();
    for row in 0..batch.num_rows() {
        if returnflags.is_null(row)
            || !utf8_value_is_one_byte(returnflag_offsets, returnflag_data, row, b'R')
        {
            continue;
        }
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        let Some(custkey) = order_customers.get(&orderkey).copied() else {
            continue;
        };
        let (Some(extendedprice), Some(discount)) = (
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            continue;
        };
        *revenues.entry(custkey).or_insert(0.0) += extendedprice * (1.0 - discount);
    }
    Ok(revenues)
}

fn returned_revenue_view(
    view: BatchView<'_>,
    order_customers: &FastHashMap<i64, i64>,
) -> Result<FastHashMap<i64, f64>> {
    if view.num_columns() == 4 {
        let (Some(orderkeys), Some(returnflags), Some(extendedprices), Some(discounts)) = (
            view.i64_vector(0),
            view.utf8_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
        ) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(fast_hash_map::<i64, f64>());
            };
            return returned_revenue_batch(batch.clone(), order_customers);
        };
        let mut revenues = fast_hash_map::<i64, f64>();
        if let Some(orderkey_values) = orderkeys.values_if_null_free()
            && returnflags.null_count() == 0
            && extendedprices.null_count() == 0
            && discounts.null_count() == 0
        {
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            let discount_scale = discounts.scale();
            let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
            for row in 0..view.num_rows() {
                if returnflags.value_bytes(row) != b"R" {
                    continue;
                }
                let Some(custkey) = order_customers.get(&orderkey_values[row]).copied() else {
                    continue;
                };
                *revenues.entry(custkey).or_insert(0.0) += decimal_discounted_revenue_raw(
                    extendedprice_values[row],
                    discount_values[row],
                    discount_scale,
                    revenue_scale,
                );
            }
            return Ok(revenues);
        }
        for row in 0..view.num_rows() {
            if returnflags.is_null(row)
                || returnflags.value_bytes(row) != b"R"
                || orderkeys.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
            {
                continue;
            }
            let Some(custkey) = order_customers.get(&orderkeys.value(row)).copied() else {
                continue;
            };
            *revenues.entry(custkey).or_insert(0.0) +=
                extendedprices.value(row) * (1.0 - discounts.value(row));
        }
        return Ok(revenues);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "returned revenue raw vector columns have unsupported types".to_string(),
        ));
    };
    returned_revenue_batch(batch.clone(), order_customers)
}

fn returned_customer_revenue_output(rows: Vec<ReturnedCustomerRevenueRow>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("c_name", DataType::Utf8, false),
            Field::new("revenue", DataType::Float64, false),
            Field::new("c_acctbal", DataType::Float64, false),
            Field::new("n_name", DataType::Utf8, false),
            Field::new("c_address", DataType::Utf8, false),
            Field::new("c_phone", DataType::Utf8, false),
            Field::new("c_comment", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|row| row.c_custkey),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_name.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.revenue),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|row| row.c_acctbal),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.n_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_address.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_phone.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.c_comment.as_str()),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
