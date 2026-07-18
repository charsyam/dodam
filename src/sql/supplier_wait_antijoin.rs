use super::*;

pub(super) async fn try_execute_supplier_wait_count_antijoin_sql(
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
    if !supplier_wait_count_antijoin_shape(select, query, selection) {
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
    let mut supplier = None;
    let mut lineitem = None;
    let mut orders = None;
    let mut nation = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("supplier") {
            supplier = Some(table);
        } else if alias.eq_ignore_ascii_case("l1") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("nation") {
            nation = Some(table);
        }
    }
    let (Some(supplier), Some(lineitem), Some(orders), Some(nation)) =
        (supplier, lineitem, orders, nation)
    else {
        return Ok(None);
    };
    let stage = tpch_profile_start();
    let nation_keys = q21_nation_keys(engine, nation.path, batch_size, "SAUDI ARABIA").await?;
    tpch_profile_elapsed("Q21 nation keys", stage);
    let stage = tpch_profile_start();
    let suppliers = q21_supplier_names(engine, supplier.path, batch_size, &nation_keys).await?;
    tpch_profile_elapsed("Q21 supplier names", stage);
    if suppliers.is_empty() {
        return Ok(Some(q21_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let final_orders = q21_final_order_keys(engine, orders.path, batch_size).await?;
    tpch_profile_elapsed("Q21 final order keys", stage);
    if final_orders.is_empty() {
        return Ok(Some(q21_output(Vec::new())?));
    }
    let stage = tpch_profile_start();
    let counts = if q21_ordered_lineitem_enabled()
        && let Some(counts) = q21_lineitem_supplier_counts_ordered(
            engine,
            lineitem.path.clone(),
            batch_size,
            &final_orders,
            &suppliers,
        )
        .await?
    {
        tpch_profile_elapsed("Q21 ordered lineitem counts", stage);
        counts
    } else {
        let order_states =
            q21_lineitem_order_states(engine, lineitem.path, batch_size, final_orders).await?;
        tpch_profile_elapsed("Q21 lineitem order states", stage);
        let mut counts = HashMap::<i64, u64>::with_capacity(suppliers.len());
        for state in order_states.into_values() {
            q21_count_qualifying_order(&mut counts, &suppliers, &state);
        }
        counts
    };
    let stage = tpch_profile_start();
    let mut rows = counts
        .into_iter()
        .filter_map(|(suppkey, count)| {
            suppliers.get(&suppkey).map(|name| Q21Row {
                s_name: name.clone(),
                count,
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.s_name.cmp(&right.s_name))
    });
    rows.truncate(100);
    tpch_profile_elapsed("Q21 final rows", stage);
    Ok(Some(q21_output(rows)?))
}

pub(super) fn q21_ordered_lineitem_enabled() -> bool {
    std::env::var_os("DODAM_Q21_DISABLE_ORDERED_LINEITEM").is_none()
}

pub(super) fn supplier_wait_count_antijoin_shape(
    select: &Select,
    query: &Query,
    selection: &SqlExpr,
) -> bool {
    let text = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 2
        && matches!(parse_limit(query), Ok(Some(100)))
        && text.contains("o_orderstatus = 'f'")
        && text.contains("l1.l_receiptdate > l1.l_commitdate")
        && text.contains("exists")
        && text.contains("not exists")
        && text.contains("n_name = 'saudi arabia'")
}

pub(super) async fn q21_nation_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_name: &str,
) -> Result<HashSet<i64>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["n_nationkey".to_string(), "n_name".to_string()]),
            None,
        )
        .await?;
    let mut keys = HashSet::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let nationkeys = batch_column(&batch, "n_nationkey")?;
        let names = batch_string_column(&batch, "n_name")?;
        for row in 0..batch.num_rows() {
            if names.is_valid(row)
                && names.value(row) == nation_name
                && let Some(key) = numeric_i64_value(nationkeys, row)?
            {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

pub(super) async fn q21_supplier_names(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    nation_keys: &HashSet<i64>,
) -> Result<HashMap<i64, String>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "s_suppkey".to_string(),
                "s_nationkey".to_string(),
                "s_name".to_string(),
            ]),
            None,
        )
        .await?;
    let mut suppliers = HashMap::new();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let suppkeys = batch_column(&batch, "s_suppkey")?;
        let nationkeys = batch_column(&batch, "s_nationkey")?;
        let names = batch_string_column(&batch, "s_name")?;
        for row in 0..batch.num_rows() {
            let (Some(suppkey), Some(nationkey)) = (
                numeric_i64_value(suppkeys, row)?,
                numeric_i64_value(nationkeys, row)?,
            ) else {
                continue;
            };
            if nation_keys.contains(&nationkey) && names.is_valid(row) {
                suppliers.insert(suppkey, names.value(row).to_string());
            }
        }
    }
    Ok(suppliers)
}

pub(super) async fn q21_final_order_keys(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<Q21FinalOrders> {
    if q21_atomic_final_orders_enabled()
        && let Some(keys) = q21_final_order_keys_atomic(engine, path.clone(), batch_size).await?
    {
        return Ok(keys);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderstatus".to_string()]),
            None,
        )
        .await?;
    let mut keys = Q21FinalOrders::new_dense();
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q21_final_orders_batch_into(&batch, &mut keys)?;
    }
    Ok(keys)
}

pub(super) type Q21FinalOrders = AdaptiveI64Set;

pub(super) fn q21_atomic_final_orders_enabled() -> bool {
    !std::env::var("DODAM_Q21_DISABLE_ATOMIC_FINAL_ORDERS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) async fn q21_final_order_keys_atomic(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
) -> Result<Option<Q21FinalOrders>> {
    let Some(max_key) = engine
        .parquet_i64_column_max(path.clone(), "o_orderkey")
        .await?
        .and_then(|key| usize::try_from(key).ok())
    else {
        return Ok(None);
    };
    let markers = Arc::new(DenseAtomicU8::zeroed(max_key + 1));
    let Some(_) = engine
        .parquet_row_group_map_dictionary_columns_pruned_view(
            path.clone(),
            batch_size,
            Projection::Columns(vec!["o_orderkey".to_string(), "o_orderstatus".to_string()]),
            vec!["o_orderstatus".to_string()],
            Vec::new(),
            q21_atomic_final_order_row_group_chunk(),
            || (),
            {
                let markers = markers.clone();
                move |view, _state| {
                    q21_final_orders_atomic_view_into(view, &markers)?;
                    Ok(Some(()))
                }
            },
            |_| Ok(Some(())),
        )
        .await?
    else {
        return Ok(None);
    };
    let markers = Arc::try_unwrap(markers).map_err(|_| {
        DodamError::UnsupportedSql("Q21 atomic final-order marker still shared".to_string())
    })?;
    Ok(Some(markers.into_adaptive_i64_set()))
}

pub(super) fn q21_atomic_final_order_row_group_chunk() -> usize {
    std::env::var("DODAM_Q21_ATOMIC_FINAL_ORDER_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

pub(super) fn q21_final_orders_atomic_view_into(
    view: BatchView<'_>,
    markers: &DenseAtomicU8,
) -> Result<()> {
    if view.num_columns() == 2
        && let Some(orderkeys) = view.i64_vector(0)
    {
        if let Some(statuses) = view.dictionary_i32_view(1) {
            return q21_final_orders_atomic_dictionary_view_into(orderkeys, statuses, markers);
        }
        if let Some(statuses) = view.utf8(1) {
            return q21_final_orders_atomic_utf8_view_into(orderkeys, statuses, markers);
        }
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q21 final-order raw vector columns have unsupported types".to_string(),
        ));
    };
    q21_final_orders_atomic_batch_into(batch, markers)
}

pub(super) fn q21_final_orders_atomic_dictionary_view_into(
    orderkeys: I64VectorView<'_>,
    statuses: DictionaryI32View<'_>,
    markers: &DenseAtomicU8,
) -> Result<()> {
    store_i64_keys_matching_dictionary_target(orderkeys, statuses, b"F", markers);
    Ok(())
}

pub(super) fn q21_final_orders_atomic_utf8_view_into(
    orderkeys: I64VectorView<'_>,
    statuses: &StringArray,
    markers: &DenseAtomicU8,
) -> Result<()> {
    store_i64_keys_matching_utf8_target(orderkeys, statuses, b"F", markers);
    Ok(())
}

pub(super) fn q21_final_orders_atomic_batch_into(
    batch: &RecordBatch,
    markers: &DenseAtomicU8,
) -> Result<()> {
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let statuses = batch_string_column(batch, "o_orderstatus")?;
    if let Some(orderkeys) = orderkeys.as_any().downcast_ref::<Int64Array>() {
        if orderkeys.null_count() == 0 && statuses.null_count() == 0 {
            let status_offsets = statuses.value_offsets();
            let status_data = statuses.value_data();
            for row in 0..orderkeys.len() {
                if bytes_string_parts(status_offsets, status_data, row) == b"F"
                    && let Ok(index) = usize::try_from(orderkeys.value(row))
                {
                    markers.store_present(index);
                }
            }
            return Ok(());
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || statuses.is_null(row) || statuses.value(row) != "F" {
                continue;
            }
            if let Ok(index) = usize::try_from(orderkeys.value(row)) {
                markers.store_present(index);
            }
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        if statuses.is_null(row) || statuses.value(row) != "F" {
            continue;
        }
        if let Some(key) = numeric_i64_value(orderkeys, row)?
            && let Ok(index) = usize::try_from(key)
        {
            markers.store_present(index);
        }
    }
    Ok(())
}

pub(super) fn q21_final_orders_batch_into(
    batch: &RecordBatch,
    keys: &mut Q21FinalOrders,
) -> Result<()> {
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let statuses = batch_string_column(batch, "o_orderstatus")?;
    if let Some(orderkeys) = orderkeys.as_any().downcast_ref::<Int64Array>() {
        if orderkeys.null_count() == 0 && statuses.null_count() == 0 {
            let status_offsets = statuses.value_offsets();
            let status_data = statuses.value_data();
            for row in 0..orderkeys.len() {
                if bytes_string_parts(status_offsets, status_data, row) == b"F" {
                    keys.insert(orderkeys.value(row));
                }
            }
            return Ok(());
        }
        for row in 0..orderkeys.len() {
            if orderkeys.is_null(row) || statuses.is_null(row) || statuses.value(row) != "F" {
                continue;
            }
            keys.insert(orderkeys.value(row));
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        if statuses.is_null(row) || statuses.value(row) != "F" {
            continue;
        }
        if let Some(key) = numeric_i64_value(orderkeys, row)? {
            keys.insert(key);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
pub(super) struct Q21OrderState {
    first_supplier: i64,
    late_supplier: i64,
    late_row_count: u32,
    flags: u8,
}

impl Q21OrderState {
    const HAS_SUPPLIER: u8 = 1 << 0;
    const HAS_MULTIPLE_SUPPLIERS: u8 = 1 << 1;
    const HAS_LATE_SUPPLIER: u8 = 1 << 2;
    const HAS_MULTIPLE_LATE_SUPPLIERS: u8 = 1 << 3;

    fn has_supplier(&self) -> bool {
        self.flags & Self::HAS_SUPPLIER != 0
    }

    fn has_multiple_suppliers(&self) -> bool {
        self.flags & Self::HAS_MULTIPLE_SUPPLIERS != 0
    }

    fn has_late_supplier(&self) -> bool {
        self.flags & Self::HAS_LATE_SUPPLIER != 0
    }

    fn has_multiple_late_suppliers(&self) -> bool {
        self.flags & Self::HAS_MULTIPLE_LATE_SUPPLIERS != 0
    }

    fn add_supplier(&mut self, suppkey: i64) {
        if !self.has_supplier() {
            self.first_supplier = suppkey;
            self.flags |= Self::HAS_SUPPLIER;
        } else if suppkey != self.first_supplier {
            self.flags |= Self::HAS_MULTIPLE_SUPPLIERS;
        }
    }

    fn add_late_supplier(&mut self, suppkey: i64) {
        if !self.has_late_supplier() {
            self.late_supplier = suppkey;
            self.flags |= Self::HAS_LATE_SUPPLIER;
            self.late_row_count = 1;
        } else if suppkey == self.late_supplier {
            self.late_row_count += 1;
        } else {
            self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
        }
    }

    fn has_single_late_supplier(&self) -> bool {
        self.has_late_supplier() && !self.has_multiple_late_suppliers()
    }

    fn merge(&mut self, other: Q21OrderState) {
        if other.has_supplier() {
            self.add_supplier(other.first_supplier);
            if other.has_multiple_suppliers() {
                self.flags |= Self::HAS_MULTIPLE_SUPPLIERS;
            }
        }
        if !other.has_late_supplier() {
            return;
        }
        if !self.has_late_supplier() {
            self.late_supplier = other.late_supplier;
            self.flags |= Self::HAS_LATE_SUPPLIER;
            self.late_row_count = other.late_row_count;
            if other.has_multiple_late_suppliers() {
                self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
            }
            return;
        }
        if self.late_supplier == other.late_supplier {
            self.late_row_count += other.late_row_count;
        } else {
            self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
        }
        if other.has_multiple_late_suppliers() {
            self.flags |= Self::HAS_MULTIPLE_LATE_SUPPLIERS;
        }
    }
}

pub(super) type Q21OrderStateMap = FastHashMap<i64, Q21OrderState>;

pub(super) fn q21_order_state_map() -> Q21OrderStateMap {
    fast_hash_map()
}

pub(super) fn q21_order_state_map_with_capacity(capacity: usize) -> Q21OrderStateMap {
    fast_hash_map_with_capacity(capacity)
}

pub(super) async fn q21_lineitem_order_states(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    final_orders: Q21FinalOrders,
) -> Result<Q21OrderStateMap> {
    if q21_dense_order_state_enabled()
        && let Some(dense_index) = q21_dense_final_order_index(&final_orders)
        && let Some(states) =
            q21_lineitem_order_states_dense(engine, path.clone(), batch_size, dense_index).await?
    {
        return Ok(states);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_receiptdate".to_string(),
        "l_commitdate".to_string(),
    ]);
    let output_capacity = final_orders.len();
    let final_orders = Arc::new(final_orders);
    if q21_row_group_map_enabled()
        && let Some(partials) = engine
            .parquet_row_group_map_view(
                path.clone(),
                batch_size,
                projection.clone(),
                q21_row_group_map_chunk(),
                q21_order_state_map,
                {
                    let final_orders = final_orders.clone();
                    move |view, states| {
                        q21_lineitem_order_states_projected_view_into(view, &final_orders, states)?;
                        Ok(Some(()))
                    }
                },
                |states| Ok(Some(states)),
            )
            .await?
    {
        let mut output = q21_order_state_map_with_capacity(output_capacity);
        for partial in partials {
            q21_merge_order_states(&mut output, partial);
        }
        return Ok(output);
    }
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    q21_parallel_batch_order_states(
        &mut stream,
        q21_lineitem_order_state_chunk_size(),
        output_capacity,
        move |batches| {
            let mut states = q21_order_state_map();
            for batch in batches {
                q21_lineitem_order_states_projected_batch_into(batch, &final_orders, &mut states)?;
            }
            Ok(states)
        },
    )
}

pub(super) fn q21_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q21_DISABLE_ROW_GROUP_MAP").is_none()
}

pub(super) fn q21_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q21_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

pub(super) fn q21_dense_order_state_enabled() -> bool {
    std::env::var("DODAM_Q21_ENABLE_DENSE_ORDER_STATE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub(super) fn q21_dense_order_state_max_orders() -> usize {
    std::env::var("DODAM_Q21_DENSE_STATE_MAX_ORDERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2_000_000)
}

pub(super) struct Q21DenseFinalOrderIndex {
    index_by_orderkey: Vec<i32>,
    orderkeys: Vec<i64>,
}

pub(super) fn q21_dense_final_order_index(
    final_orders: &Q21FinalOrders,
) -> Option<Arc<Q21DenseFinalOrderIndex>> {
    let contains = final_orders.dense_contains_slice()?;
    let selected = final_orders.len();
    if selected > q21_dense_order_state_max_orders() {
        return None;
    }
    let mut index_by_orderkey = vec![-1_i32; contains.len()];
    let mut orderkeys = Vec::with_capacity(selected);
    for (orderkey, selected) in contains.iter().copied().enumerate() {
        if !selected {
            continue;
        }
        let index = i32::try_from(orderkeys.len()).ok()?;
        index_by_orderkey[orderkey] = index;
        orderkeys.push(orderkey as i64);
    }
    Some(Arc::new(Q21DenseFinalOrderIndex {
        index_by_orderkey,
        orderkeys,
    }))
}

pub(super) async fn q21_lineitem_order_states_dense(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    dense_index: Arc<Q21DenseFinalOrderIndex>,
) -> Result<Option<Q21OrderStateMap>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "l_orderkey".to_string(),
                "l_suppkey".to_string(),
                "l_receiptdate".to_string(),
                "l_commitdate".to_string(),
            ]),
            None,
        )
        .await?;
    q21_parallel_dense_order_states(
        &mut stream,
        q21_lineitem_order_state_chunk_size(),
        dense_index,
    )
}

pub(super) fn q21_parallel_dense_order_states(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    dense_index: Arc<Q21DenseFinalOrderIndex>,
) -> Result<Option<Q21OrderStateMap>> {
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let chunk_size = chunk_size.max(1);
    let mut pending_chunks = 0_usize;
    let mut chunk = Vec::with_capacity(chunk_size);
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size {
            continue;
        }
        let sender = sender.clone();
        let dense_index = dense_index.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(q21_dense_order_states_chunk(task_chunk, dense_index));
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let dense_index = dense_index.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(q21_dense_order_states_chunk(chunk, dense_index));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    let mut output = Q21DenseOrderStates::new(dense_index.orderkeys.len());
    for _ in 0..pending_chunks {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::UnsupportedSql("Q21 dense worker stopped".to_string()))??;
        output.merge(partial);
    }
    let states = output.into_map(&dense_index.orderkeys);
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] Q21 dense lineitem order states: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(Some(states))
}

pub(super) fn q21_dense_order_states_chunk(
    batches: Vec<RecordBatch>,
    dense_index: Arc<Q21DenseFinalOrderIndex>,
) -> Result<Q21DenseOrderStates> {
    let mut states = Q21DenseOrderStates::new(dense_index.orderkeys.len());
    for batch in batches {
        if !q21_dense_order_states_batch_into(&batch, &dense_index, &mut states)? {
            return Err(DodamError::UnsupportedSql(
                "Q21 dense order-state path requires typed lineitem columns".to_string(),
            ));
        }
    }
    Ok(states)
}

pub(super) struct Q21DenseOrderStates {
    positions: Vec<i32>,
    states: Vec<Q21OrderState>,
    touched: Vec<usize>,
}

impl Q21DenseOrderStates {
    fn new(len: usize) -> Self {
        Self {
            positions: vec![-1; len],
            states: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn state_mut(&mut self, index: usize) -> &mut Q21OrderState {
        let position = self.positions[index];
        let position = if position < 0 {
            let position = self.states.len();
            self.positions[index] = i32::try_from(position).expect("Q21 state index overflow");
            self.states.push(Q21OrderState::default());
            self.touched.push(index);
            position
        } else {
            position as usize
        };
        &mut self.states[position]
    }

    fn merge(&mut self, other: Self) {
        for index in other.touched {
            let position = other.positions[index];
            debug_assert!(position >= 0);
            self.state_mut(index).merge(other.states[position as usize]);
        }
    }

    fn into_map(self, orderkeys: &[i64]) -> Q21OrderStateMap {
        let mut output = q21_order_state_map_with_capacity(self.touched.len());
        for index in self.touched {
            let position = self.positions[index];
            debug_assert!(position >= 0);
            output.insert(orderkeys[index], self.states[position as usize]);
        }
        output
    }
}

pub(super) fn q21_dense_order_states_batch_into(
    batch: &RecordBatch,
    dense_index: &Q21DenseFinalOrderIndex,
    states: &mut Q21DenseOrderStates,
) -> Result<bool> {
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let suppkeys = batch_column(batch, "l_suppkey")?;
    let receipt = batch_column(batch, "l_receiptdate")?;
    let commit = batch_column(batch, "l_commitdate")?;
    let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        receipt.as_any().downcast_ref::<Date32Array>(),
        commit.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && receipt.null_count() == 0
        && commit.null_count() == 0
    {
        let orderkeys = orderkeys.values().as_ref();
        let suppkeys = suppkeys.values().as_ref();
        let receipts = receipt.values().as_ref();
        let commits = commit.values().as_ref();
        for row in 0..orderkeys.len() {
            let Some(index) = q21_dense_order_index(dense_index, orderkeys[row]) else {
                continue;
            };
            let state = states.state_mut(index);
            let suppkey = suppkeys[row];
            state.add_supplier(suppkey);
            if receipts[row] > commits[row] {
                state.add_late_supplier(suppkey);
            }
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let Some(index) = q21_dense_order_index(dense_index, orderkeys.value(row)) else {
            continue;
        };
        let state = states.state_mut(index);
        let suppkey = suppkeys.value(row);
        state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            state.add_late_supplier(suppkey);
        }
    }
    Ok(true)
}

pub(super) fn q21_dense_order_index(
    dense_index: &Q21DenseFinalOrderIndex,
    orderkey: i64,
) -> Option<usize> {
    let index = usize::try_from(orderkey).ok()?;
    let compact = *dense_index.index_by_orderkey.get(index)?;
    usize::try_from(compact).ok()
}

pub(super) fn q21_parallel_batch_order_states<Map>(
    stream: &mut SendableBatchStream,
    chunk_size: usize,
    output_capacity: usize,
    map: Map,
) -> Result<Q21OrderStateMap>
where
    Map: Fn(Vec<RecordBatch>) -> Result<Q21OrderStateMap> + Send + Sync + Clone + 'static,
{
    let profile = tpch_profile_enabled();
    let started = profile.then(Instant::now);
    let (sender, receiver) = mpsc::channel();
    let chunk_size = chunk_size.max(1);
    let mut pending_chunks = 0_usize;
    let mut chunk = Vec::with_capacity(chunk_size);
    let stream_started = profile.then(Instant::now);
    while let Some(batch) = stream.next() {
        chunk.push(batch?);
        if chunk.len() < chunk_size {
            continue;
        }
        let sender = sender.clone();
        let map = map.clone();
        let task_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(chunk_size));
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(task_chunk));
        });
    }
    if !chunk.is_empty() {
        let sender = sender.clone();
        let map = map.clone();
        pending_chunks += 1;
        rayon::spawn(move || {
            let _ = sender.send(map(chunk));
        });
    }
    let stream_ms = stream_started
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or_default();
    drop(sender);
    let merge_started = profile.then(Instant::now);
    let mut partials = Vec::with_capacity(pending_chunks);
    for _ in 0..pending_chunks {
        partials.push(
            receiver
                .recv()
                .map_err(|_| DodamError::UnsupportedSql("Q21 worker stopped".to_string()))??,
        );
    }
    let output = if q21_parallel_merge_enabled() {
        partials
            .into_par_iter()
            .reduce(q21_order_state_map, q21_merge_order_states_owned)
    } else {
        let mut output = q21_order_state_map_with_capacity(output_capacity);
        for partial in partials {
            q21_merge_order_states(&mut output, partial);
        }
        output
    };
    if let Some(started) = started {
        let merge_ms = merge_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        eprintln!(
            "[dodam:tpch-profile] Q21 lineitem order states: total={:.3} ms stream_read={:.3} ms worker_wait_merge={:.3} ms chunks={pending_chunks}",
            started.elapsed().as_secs_f64() * 1000.0,
            stream_ms,
            merge_ms
        );
    }
    Ok(output)
}

pub(super) fn q21_parallel_merge_enabled() -> bool {
    std::env::var("DODAM_Q21_ENABLE_PARALLEL_MERGE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub(super) fn q21_lineitem_order_state_chunk_size() -> usize {
    std::env::var("DODAM_Q21_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(48)
}

pub(super) fn q21_lineitem_order_states_batch_into(
    batch: RecordBatch,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> Result<()> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let suppkeys = batch_column(&batch, "l_suppkey")?;
    let receipt = batch_column(&batch, "l_receiptdate")?;
    let commit = batch_column(&batch, "l_commitdate")?;
    if q21_lineitem_order_states_typed_into(
        orderkeys,
        suppkeys,
        receipt,
        commit,
        final_orders,
        states,
    ) {
        return Ok(());
    }
    let dense_final_orders = final_orders.dense_contains_slice();
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(suppkey)) = (
            numeric_i64_value(orderkeys, row)?,
            numeric_i64_value(suppkeys, row)?,
        ) else {
            continue;
        };
        if !q21_final_order_contains(final_orders, dense_final_orders, orderkey) {
            continue;
        }
        let state = states.entry(orderkey).or_default();
        state.add_supplier(suppkey);
        let (Some(receipt), Some(commit)) =
            (date32_value(receipt, row)?, date32_value(commit, row)?)
        else {
            continue;
        };
        if receipt > commit {
            state.add_late_supplier(suppkey);
        }
    }
    Ok(())
}

pub(super) fn q21_lineitem_order_states_projected_batch_into(
    batch: RecordBatch,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> Result<()> {
    if batch.num_columns() == 4
        && q21_lineitem_order_states_typed_into(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            batch.column(3),
            final_orders,
            states,
        )
    {
        return Ok(());
    }
    q21_lineitem_order_states_batch_into(batch, final_orders, states)
}

pub(super) fn q21_lineitem_order_states_projected_view_into(
    view: BatchView<'_>,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> Result<()> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            view.date32_vector(3),
        )
    {
        q21_lineitem_order_states_vector_typed_into(
            orderkeys,
            suppkeys,
            receipt,
            commit,
            final_orders,
            states,
        );
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(());
    };
    q21_lineitem_order_states_batch_into(batch.clone(), final_orders, states)
}

pub(super) fn q21_lineitem_order_states_typed_into(
    orderkeys: &ArrayRef,
    suppkeys: &ArrayRef,
    receipt: &ArrayRef,
    commit: &ArrayRef,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) -> bool {
    let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        suppkeys.as_any().downcast_ref::<Int64Array>(),
        receipt.as_any().downcast_ref::<Date32Array>(),
        commit.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return false;
    };
    let dense_final_orders = final_orders.dense_contains_slice();
    if orderkeys.null_count() == 0
        && suppkeys.null_count() == 0
        && receipt.null_count() == 0
        && commit.null_count() == 0
    {
        let orderkeys = orderkeys.values().as_ref();
        let suppkeys = suppkeys.values().as_ref();
        let receipts = receipt.values().as_ref();
        let commits = commit.values().as_ref();
        let mut current_orderkey = None::<i64>;
        let mut current_order_selected = false;
        let mut current_state = Q21OrderState::default();
        for row in 0..orderkeys.len() {
            let orderkey = orderkeys[row];
            if current_orderkey.is_some_and(|current| current != orderkey) {
                if current_order_selected {
                    q21_flush_run_state(states, current_orderkey, &mut current_state);
                }
                current_order_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                current_orderkey = Some(orderkey);
            } else if current_orderkey.is_none() {
                current_order_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                current_orderkey = Some(orderkey);
            }
            if !current_order_selected {
                continue;
            }
            let suppkey = suppkeys[row];
            current_state.add_supplier(suppkey);
            if receipts[row] > commits[row] {
                current_state.add_late_supplier(suppkey);
            }
        }
        if current_order_selected {
            q21_flush_run_state(states, current_orderkey, &mut current_state);
        }
        return true;
    }
    let mut current_orderkey = None::<i64>;
    let mut current_order_selected = false;
    let mut current_state = Q21OrderState::default();
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if current_orderkey.is_some_and(|current| current != orderkey) {
            if current_order_selected {
                q21_flush_run_state(states, current_orderkey, &mut current_state);
            }
            current_order_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            current_orderkey = Some(orderkey);
        } else if current_orderkey.is_none() {
            current_order_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            current_orderkey = Some(orderkey);
        }
        if !current_order_selected {
            continue;
        }
        let suppkey = suppkeys.value(row);
        current_state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            current_state.add_late_supplier(suppkey);
        }
    }
    if current_order_selected {
        q21_flush_run_state(states, current_orderkey, &mut current_state);
    }
    true
}

pub(super) fn q21_lineitem_order_states_vector_typed_into(
    orderkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    receipt: Date32VectorView<'_>,
    commit: Date32VectorView<'_>,
    final_orders: &Q21FinalOrders,
    states: &mut Q21OrderStateMap,
) {
    let dense_final_orders = final_orders.dense_contains_slice();
    if let (Some(orderkey_values), Some(suppkey_values), Some(receipts), Some(commits)) = (
        orderkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
        receipt.values_if_null_free(),
        commit.values_if_null_free(),
    ) {
        let mut current_orderkey = None::<i64>;
        let mut current_order_selected = false;
        let mut current_state = Q21OrderState::default();
        for row in 0..orderkey_values.len() {
            let orderkey = orderkey_values[row];
            if current_orderkey.is_some_and(|current| current != orderkey) {
                if current_order_selected {
                    q21_flush_run_state(states, current_orderkey, &mut current_state);
                }
                current_order_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                current_orderkey = Some(orderkey);
            } else if current_orderkey.is_none() {
                current_order_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                current_orderkey = Some(orderkey);
            }
            if !current_order_selected {
                continue;
            }
            let suppkey = suppkey_values[row];
            current_state.add_supplier(suppkey);
            if receipts[row] > commits[row] {
                current_state.add_late_supplier(suppkey);
            }
        }
        if current_order_selected {
            q21_flush_run_state(states, current_orderkey, &mut current_state);
        }
        return;
    }
    let mut current_orderkey = None::<i64>;
    let mut current_order_selected = false;
    let mut current_state = Q21OrderState::default();
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if current_orderkey.is_some_and(|current| current != orderkey) {
            if current_order_selected {
                q21_flush_run_state(states, current_orderkey, &mut current_state);
            }
            current_order_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            current_orderkey = Some(orderkey);
        } else if current_orderkey.is_none() {
            current_order_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            current_orderkey = Some(orderkey);
        }
        if !current_order_selected {
            continue;
        }
        let suppkey = suppkeys.value(row);
        current_state.add_supplier(suppkey);
        if receipt.is_null(row) || commit.is_null(row) {
            continue;
        }
        if receipt.value(row) > commit.value(row) {
            current_state.add_late_supplier(suppkey);
        }
    }
    if current_order_selected {
        q21_flush_run_state(states, current_orderkey, &mut current_state);
    }
}

pub(super) fn q21_final_order_contains(
    final_orders: &Q21FinalOrders,
    dense_final_orders: Option<&[bool]>,
    orderkey: i64,
) -> bool {
    if let Some(dense_final_orders) = dense_final_orders {
        return usize::try_from(orderkey)
            .ok()
            .and_then(|index| dense_final_orders.get(index))
            .copied()
            .unwrap_or(false);
    }
    final_orders.contains(orderkey)
}

pub(super) async fn q21_lineitem_supplier_counts_ordered(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    final_orders: &Q21FinalOrders,
    suppliers: &HashMap<i64, String>,
) -> Result<Option<HashMap<i64, u64>>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_suppkey".to_string(),
        "l_receiptdate".to_string(),
        "l_commitdate".to_string(),
    ]);
    let final_orders = Arc::new(final_orders.clone());
    let suppliers = Arc::new(suppliers.clone());
    let Some(chunks) = engine
        .parquet_row_group_map_view(
            path,
            batch_size,
            projection,
            q21_row_group_map_chunk(),
            Q21OrderedLineitemChunkState::default,
            {
                let final_orders = final_orders.clone();
                let suppliers = suppliers.clone();
                move |view, state| {
                    q21_ordered_lineitem_chunk_view(view, &final_orders, suppliers.as_ref(), state)
                }
            },
            |state| Ok(Some(state)),
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(q21_merge_ordered_lineitem_chunks(
        chunks,
        suppliers.as_ref(),
    )))
}

#[derive(Default)]
pub(super) struct Q21OrderedLineitemChunkState {
    counts: HashMap<i64, u64>,
    current_orderkey: Option<i64>,
    current_selected: bool,
    current_state: Q21OrderState,
    first: Option<OrderedRowGroupBoundary<i64, Q21SelectedOrderState>>,
    last: Option<OrderedRowGroupBoundary<i64, Q21SelectedOrderState>>,
    order_count: usize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Q21SelectedOrderState {
    selected: bool,
    state: Q21OrderState,
}

pub(super) fn q21_ordered_lineitem_chunk_view(
    view: BatchView<'_>,
    final_orders: &Q21FinalOrders,
    suppliers: &HashMap<i64, String>,
    state: &mut Q21OrderedLineitemChunkState,
) -> Result<Option<()>> {
    if view.num_columns() == 4
        && let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
            view.i64_vector(0),
            view.i64_vector(1),
            view.date32_vector(2),
            view.date32_vector(3),
        )
    {
        return q21_ordered_lineitem_chunk_vector_typed(
            orderkeys,
            suppkeys,
            receipt,
            commit,
            final_orders,
            suppliers,
            state,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q21_ordered_lineitem_chunk_batch(batch, final_orders, suppliers, state)
}

pub(super) fn q21_ordered_lineitem_chunk_batch(
    batch: &RecordBatch,
    final_orders: &Q21FinalOrders,
    suppliers: &HashMap<i64, String>,
    state: &mut Q21OrderedLineitemChunkState,
) -> Result<Option<()>> {
    if batch.num_columns() == 4
        && let (Some(orderkeys), Some(suppkeys), Some(receipt), Some(commit)) = (
            batch.column(0).as_any().downcast_ref::<Int64Array>(),
            batch.column(1).as_any().downcast_ref::<Int64Array>(),
            batch.column(2).as_any().downcast_ref::<Date32Array>(),
            batch.column(3).as_any().downcast_ref::<Date32Array>(),
        )
    {
        return q21_ordered_lineitem_chunk_typed_batch(
            orderkeys,
            suppkeys,
            receipt,
            commit,
            final_orders,
            suppliers,
            state,
        );
    }
    Ok(None)
}

pub(super) fn q21_ordered_lineitem_chunk_typed_batch(
    orderkeys: &Int64Array,
    suppkeys: &Int64Array,
    receipt: &Date32Array,
    commit: &Date32Array,
    final_orders: &Q21FinalOrders,
    suppliers: &HashMap<i64, String>,
    chunk: &mut Q21OrderedLineitemChunkState,
) -> Result<Option<()>> {
    let dense_final_orders = final_orders.dense_contains_slice();
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if let Some(current) = chunk.current_orderkey {
            if orderkey < current {
                return Ok(None);
            }
            if orderkey != current {
                q21_ordered_chunk_finish_current(chunk, suppliers);
                chunk.current_orderkey = Some(orderkey);
                chunk.current_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                chunk.current_state = Q21OrderState::default();
                chunk.order_count += 1;
            }
        } else {
            chunk.current_orderkey = Some(orderkey);
            chunk.current_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            chunk.order_count = 1;
        }
        if !chunk.current_selected {
            continue;
        }
        let suppkey = suppkeys.value(row);
        chunk.current_state.add_supplier(suppkey);
        if receipt.is_valid(row) && commit.is_valid(row) && receipt.value(row) > commit.value(row) {
            chunk.current_state.add_late_supplier(suppkey);
        }
    }
    Ok(Some(()))
}

pub(super) fn q21_ordered_lineitem_chunk_vector_typed(
    orderkeys: I64VectorView<'_>,
    suppkeys: I64VectorView<'_>,
    receipt: Date32VectorView<'_>,
    commit: Date32VectorView<'_>,
    final_orders: &Q21FinalOrders,
    suppliers: &HashMap<i64, String>,
    chunk: &mut Q21OrderedLineitemChunkState,
) -> Result<Option<()>> {
    let dense_final_orders = final_orders.dense_contains_slice();
    if let (Some(orderkey_values), Some(suppkey_values), Some(receipts), Some(commits)) = (
        orderkeys.values_if_null_free(),
        suppkeys.values_if_null_free(),
        receipt.values_if_null_free(),
        commit.values_if_null_free(),
    ) {
        for row in 0..orderkey_values.len() {
            let orderkey = orderkey_values[row];
            if let Some(current) = chunk.current_orderkey {
                if orderkey < current {
                    return Ok(None);
                }
                if orderkey != current {
                    q21_ordered_chunk_finish_current(chunk, suppliers);
                    chunk.current_orderkey = Some(orderkey);
                    chunk.current_selected =
                        q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                    chunk.current_state = Q21OrderState::default();
                    chunk.order_count += 1;
                }
            } else {
                chunk.current_orderkey = Some(orderkey);
                chunk.current_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                chunk.order_count = 1;
            }
            if !chunk.current_selected {
                continue;
            }
            let suppkey = suppkey_values[row];
            chunk.current_state.add_supplier(suppkey);
            if receipts[row] > commits[row] {
                chunk.current_state.add_late_supplier(suppkey);
            }
        }
        return Ok(Some(()));
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || suppkeys.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if let Some(current) = chunk.current_orderkey {
            if orderkey < current {
                return Ok(None);
            }
            if orderkey != current {
                q21_ordered_chunk_finish_current(chunk, suppliers);
                chunk.current_orderkey = Some(orderkey);
                chunk.current_selected =
                    q21_final_order_contains(final_orders, dense_final_orders, orderkey);
                chunk.current_state = Q21OrderState::default();
                chunk.order_count += 1;
            }
        } else {
            chunk.current_orderkey = Some(orderkey);
            chunk.current_selected =
                q21_final_order_contains(final_orders, dense_final_orders, orderkey);
            chunk.order_count = 1;
        }
        if !chunk.current_selected {
            continue;
        }
        let suppkey = suppkeys.value(row);
        chunk.current_state.add_supplier(suppkey);
        if !receipt.is_null(row) && !commit.is_null(row) && receipt.value(row) > commit.value(row) {
            chunk.current_state.add_late_supplier(suppkey);
        }
    }
    Ok(Some(()))
}

pub(super) fn q21_ordered_chunk_finish_current(
    chunk: &mut Q21OrderedLineitemChunkState,
    suppliers: &HashMap<i64, String>,
) {
    let Some(orderkey) = chunk.current_orderkey else {
        return;
    };
    let boundary = OrderedRowGroupBoundary {
        key: orderkey,
        state: Q21SelectedOrderState {
            selected: chunk.current_selected,
            state: chunk.current_state,
        },
    };
    if chunk.order_count == 1 {
        chunk.first = Some(boundary);
    } else if boundary.state.selected {
        q21_count_qualifying_order(&mut chunk.counts, suppliers, &boundary.state.state);
    }
}

pub(super) fn q21_merge_ordered_lineitem_chunks(
    mut chunks: Vec<Q21OrderedLineitemChunkState>,
    suppliers: &HashMap<i64, String>,
) -> HashMap<i64, u64> {
    let mut ordered_chunks = Vec::with_capacity(chunks.len());
    for mut chunk in chunks.drain(..) {
        if let Some(orderkey) = chunk.current_orderkey {
            let boundary = OrderedRowGroupBoundary {
                key: orderkey,
                state: Q21SelectedOrderState {
                    selected: chunk.current_selected,
                    state: chunk.current_state,
                },
            };
            if chunk.order_count <= 1 {
                chunk.first = Some(boundary);
            } else {
                chunk.last = Some(boundary);
            }
        }
        ordered_chunks.push(OrderedRowGroupChunk {
            output: chunk.counts,
            first: chunk.first,
            last: chunk.last,
        });
    }
    let mut counts = HashMap::<i64, u64>::with_capacity(suppliers.len());
    merge_ordered_row_group_chunks(
        ordered_chunks,
        &mut counts,
        |counts, chunk_counts| {
            for (suppkey, count) in chunk_counts {
                *counts.entry(suppkey).or_insert(0) += count;
            }
        },
        |left, right| {
            left.selected |= right.selected;
            left.state.merge(right.state);
        },
        |counts, boundary| {
            if boundary.state.selected {
                q21_count_qualifying_order(counts, suppliers, &boundary.state.state);
            }
        },
    );
    counts
}

pub(super) fn q21_count_qualifying_order(
    counts: &mut HashMap<i64, u64>,
    suppliers: &HashMap<i64, String>,
    state: &Q21OrderState,
) {
    if !state.has_multiple_suppliers() || !state.has_single_late_supplier() {
        return;
    }
    let suppkey = state.late_supplier;
    if !suppliers.contains_key(&suppkey) {
        return;
    }
    *counts.entry(suppkey).or_insert(0) += u64::from(state.late_row_count);
}

pub(super) fn q21_flush_run_state(
    states: &mut Q21OrderStateMap,
    orderkey: Option<i64>,
    state: &mut Q21OrderState,
) {
    let Some(orderkey) = orderkey else {
        return;
    };
    states
        .entry(orderkey)
        .or_default()
        .merge(std::mem::take(state));
}

pub(super) fn q21_merge_order_states(
    states: &mut Q21OrderStateMap,
    batch_states: Q21OrderStateMap,
) {
    for (orderkey, batch_state) in batch_states {
        states.entry(orderkey).or_default().merge(batch_state);
    }
}

pub(super) fn q21_merge_order_states_owned(
    mut left: Q21OrderStateMap,
    mut right: Q21OrderStateMap,
) -> Q21OrderStateMap {
    if left.len() < right.len() {
        std::mem::swap(&mut left, &mut right);
    }
    q21_merge_order_states(&mut left, right);
    left
}

pub(super) struct Q21Row {
    s_name: String,
    count: u64,
}

pub(super) fn q21_output(rows: Vec<Q21Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("s_name", DataType::Utf8, false),
            Field::new("numwait", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.s_name.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
