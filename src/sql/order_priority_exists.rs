use super::*;

fn selective_candidate_priority_range(candidate_priorities: &[u8]) -> Option<(i64, i64)> {
    let mut min_key = usize::MAX;
    let mut max_key = 0_usize;
    let mut len = 0_usize;
    for (key, priority) in candidate_priorities.iter().copied().enumerate() {
        if priority == 0 {
            continue;
        }
        min_key = min_key.min(key);
        max_key = max_key.max(key);
        len += 1;
    }
    if min_key == usize::MAX {
        return None;
    }
    selective_i64_range_from_parts(min_key as i64, max_key as i64, len)
}

fn q04_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
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
    select.projection.len() == 2
        && projection.contains("o_orderpriority")
        && projection.contains("count(*)")
        && group_by.contains("o_orderpriority")
        && order_by.contains("o_orderpriority")
        && selection.contains("o_orderdate")
        && selection.contains("exists")
        && selection.contains("l_orderkey = o_orderkey")
        && selection.contains("l_commitdate < l_receiptdate")
}

pub(super) async fn try_execute_order_priority_exists_count_sql(
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
    if !q04_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let orders = parse_from(select)?;
    if !table_ref_alias_or_name(&orders).eq_ignore_ascii_case("orders") {
        return Ok(None);
    }
    let Some(lineitem_path) = q04_lineitem_path(selection)? else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "o_orderdate")? else {
        return Ok(None);
    };

    let stage = tpch_profile_start();
    let (mut candidate_priorities, priority_labels, candidate_count) =
        q04_candidate_order_priorities(engine, orders.path, batch_size, start_days, end_days)
            .await?;
    tpch_profile_elapsed("Q04 candidate orders", stage);
    if candidate_count == 0 || priority_labels.is_empty() {
        return Ok(Some(q04_output(Vec::new())?));
    }

    let stage = tpch_profile_start();
    let counts = q04_count_late_candidate_priorities(
        engine,
        lineitem_path,
        batch_size,
        &mut candidate_priorities,
        candidate_count,
        priority_labels.len(),
    )
    .await?;
    tpch_profile_elapsed("Q04 late lineitem priorities", stage);
    let rows = q04_priority_count_rows(priority_labels, counts);
    Ok(Some(q04_output(rows)?))
}

fn q04_lineitem_path(selection: &SqlExpr) -> Result<Option<PathBuf>> {
    let mut stack = vec![selection];
    while let Some(expr) = stack.pop() {
        match expr {
            SqlExpr::Exists { subquery, .. } => {
                let SetExpr::Select(select) = subquery.body.as_ref() else {
                    continue;
                };
                for table in parse_select_table_refs(select)? {
                    if table_ref_alias_or_name(&table).eq_ignore_ascii_case("lineitem") {
                        return Ok(Some(table.path));
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => stack.push(expr),
            _ => {}
        }
    }
    Ok(None)
}

struct Q04Row {
    priority: String,
    count: u64,
}

async fn q04_candidate_order_priorities(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<(Vec<u8>, Vec<String>, usize)> {
    if let Some(candidates) = q04_candidate_order_priorities_row_group_map(
        engine,
        path.clone(),
        batch_size,
        start_days,
        end_days,
    )
    .await?
    {
        return Ok(candidates);
    }
    if let Some(candidates) =
        q04_candidate_order_priorities_late(engine, path.clone(), batch_size, start_days, end_days)
            .await?
    {
        return Ok(candidates);
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_orderdate".to_string(),
                "o_orderpriority".to_string(),
            ]),
            None,
        )
        .await?;
    let mut priorities = Vec::<u8>::new();
    let mut labels = Vec::<String>::new();
    let mut label_indices = HashMap::<String, u8>::new();
    let mut candidate_count = 0usize;
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q04_candidate_order_priorities_view_into(
            BatchView::new(&batch),
            start_days,
            end_days,
            &mut priorities,
            &mut labels,
            &mut label_indices,
            &mut candidate_count,
        )?;
    }
    Ok((priorities, labels, candidate_count))
}

async fn q04_candidate_order_priorities_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<Option<(Vec<u8>, Vec<String>, usize)>> {
    let dictionary_columns = vec!["o_orderpriority".to_string()];
    let Some(partials) = engine
        .parquet_row_group_map_scan_view(
            path,
            batch_size,
            Projection::Columns(vec![
                "o_orderkey".to_string(),
                "o_orderdate".to_string(),
                "o_orderpriority".to_string(),
            ]),
            dictionary_columns,
            date_range_pruning_predicates("o_orderdate", start_days, end_days),
            q04_candidate_row_group_map_chunk(),
            || Q04CandidatePartial {
                labels: Vec::new(),
                label_indices: HashMap::new(),
                rows: Vec::new(),
            },
            move |batch, partial| {
                q04_candidate_order_priorities_partial_view(batch, start_days, end_days, partial)
            },
            |partial| Ok(Some(partial)),
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(q04_candidate_priorities_from_partials(partials)?))
}

fn q04_candidate_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(2)
}

async fn q04_candidate_order_priorities_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    start_days: i32,
    end_days: i32,
) -> Result<Option<(Vec<u8>, Vec<String>, usize)>> {
    let predicate_projection =
        Projection::Columns(vec!["o_orderkey".to_string(), "o_orderdate".to_string()]);
    let payload_projection = Projection::Columns(vec!["o_orderpriority".to_string()]);
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection.clone(),
            payload_projection.clone(),
            date_range_pruning_predicates("o_orderdate", start_days, end_days),
            q04_late_candidate_row_group_chunk(),
            generic_late_materialization_policy_for_projection(
                &predicate_projection,
                &payload_projection,
                0.60,
                None,
            ),
            move || Q04LateCandidateState {
                start_days,
                end_days,
                selected_orderkeys: Vec::new(),
                priority_offset: 0,
                labels: Vec::new(),
                label_indices: HashMap::new(),
                rows: Vec::new(),
            },
            q04_late_candidate_build_selection_view,
            q04_late_candidate_consume_priority_view,
            |state, _metrics| {
                if state.priority_offset != state.selected_orderkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q04 candidate row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some(Q04CandidatePartial {
                    labels: state.labels,
                    label_indices: state.label_indices,
                    rows: state.rows,
                }))
            },
        )
        .await?
    else {
        return Ok(None);
    };

    let mut metrics = LateMaterializedMetrics::default();
    let mut partials = Vec::new();
    for chunk in chunks {
        metrics.add(chunk.metrics);
        partials.push(chunk.output);
    }
    q04_log_late_candidate_profile(metrics, q04_late_candidate_row_group_chunk());
    Ok(Some(q04_candidate_priorities_from_partials(partials)?))
}

fn q04_late_candidate_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(4)
}

struct Q04LateCandidateState {
    start_days: i32,
    end_days: i32,
    selected_orderkeys: Vec<i64>,
    priority_offset: usize,
    labels: Vec<String>,
    label_indices: HashMap<String, u8>,
    rows: Vec<(i64, u8)>,
}

struct Q04CandidatePredicateView<'a> {
    orderkeys: I64VectorView<'a>,
    orderdates: Date32VectorView<'a>,
}

impl<'a> Q04CandidatePredicateView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        (view.num_columns() == 2).then_some(Self {
            orderkeys: view.i64_vector(0)?,
            orderdates: view.date32_vector(1)?,
        })
    }
}

struct Q04CandidatePayloadView<'a> {
    priorities: Utf8VectorView<'a>,
}

impl<'a> Q04CandidatePayloadView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        (view.num_columns() == 1).then_some(Self {
            priorities: view.utf8_vector(0)?,
        })
    }
}

struct Q04CandidatePartial {
    labels: Vec<String>,
    label_indices: HashMap<String, u8>,
    rows: Vec<(i64, u8)>,
}

fn q04_candidate_order_priorities_partial_batch(
    batch: RecordBatch,
    start_days: i32,
    end_days: i32,
    partial: &mut Q04CandidatePartial,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let orderpriorities = batch_column(&batch, "o_orderpriority")?;
    if let (Some(orderkeys), Some(orderdates), Some(orderpriorities)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
        orderpriorities
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>(),
    ) && q04_candidate_order_priorities_dictionary_typed(
        orderkeys,
        orderdates,
        DictionaryI32View::Arrow(orderpriorities),
        start_days,
        end_days,
        partial,
    )? {
        return Ok(Some(()));
    }
    let Some(orderpriorities) = orderpriorities.as_any().downcast_ref::<StringArray>() else {
        return Ok(None);
    };
    if let (Some(orderkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) && orderkeys.null_count() == 0
        && orderdates.null_count() == 0
        && orderpriorities.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        for row in 0..orderkey_values.len() {
            let orderdate = orderdate_values[row];
            let orderkey = orderkey_values[row];
            if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
                continue;
            }
            let priority = orderpriorities.value(row);
            let priority_index = if let Some(index) = partial.label_indices.get(priority) {
                *index
            } else {
                let next_index = u8::try_from(partial.labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                let priority = priority.to_string();
                partial.labels.push(priority.clone());
                partial.label_indices.insert(priority, next_index);
                next_index
            };
            partial.rows.push((orderkey, priority_index));
        }
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        if orderpriorities.is_null(row) {
            continue;
        }
        let (Some(orderkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
            continue;
        }
        let priority = orderpriorities.value(row);
        let priority_index = if let Some(index) = partial.label_indices.get(priority) {
            *index
        } else {
            let next_index = u8::try_from(partial.labels.len()).map_err(|_| {
                DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
            })?;
            let priority = priority.to_string();
            partial.labels.push(priority.clone());
            partial.label_indices.insert(priority, next_index);
            next_index
        };
        partial.rows.push((orderkey, priority_index));
    }
    Ok(Some(()))
}

fn q04_candidate_order_priorities_partial_view(
    view: BatchView<'_>,
    start_days: i32,
    end_days: i32,
    partial: &mut Q04CandidatePartial,
) -> Result<Option<()>> {
    if let (Some(orderkeys), Some(orderdates), Some(orderpriorities)) =
        (view.i64(0), view.date32(1), view.dictionary_i32_view(2))
        && q04_candidate_order_priorities_dictionary_typed(
            orderkeys,
            orderdates,
            orderpriorities,
            start_days,
            end_days,
            partial,
        )?
    {
        return Ok(Some(()));
    }
    if let (Some(orderkeys), Some(orderdates), Some(orderpriorities)) = (
        view.i64_vector(0),
        view.date32_vector(1),
        view.utf8_vector(2),
    ) && let (Some(orderkey_values), Some(orderdate_values)) = (
        orderkeys.values_if_null_free(),
        orderdates.values_if_null_free(),
    ) && orderpriorities.null_count() == 0
    {
        for row in 0..orderkey_values.len() {
            let orderdate = orderdate_values[row];
            let orderkey = orderkey_values[row];
            if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
                continue;
            }
            let priority = std::str::from_utf8(orderpriorities.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            let priority_index = if let Some(index) = partial.label_indices.get(priority) {
                *index
            } else {
                let next_index = u8::try_from(partial.labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                let priority = priority.to_string();
                partial.labels.push(priority.clone());
                partial.label_indices.insert(priority, next_index);
                next_index
            };
            partial.rows.push((orderkey, priority_index));
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q04_candidate_order_priorities_partial_batch(batch.clone(), start_days, end_days, partial)
}

fn q04_candidate_order_priorities_dictionary_typed(
    orderkeys: &Int64Array,
    orderdates: &Date32Array,
    orderpriorities: DictionaryI32View<'_>,
    start_days: i32,
    end_days: i32,
    partial: &mut Q04CandidatePartial,
) -> Result<bool> {
    let Some(priority_values) = orderpriorities.string_values() else {
        return Ok(false);
    };
    let priority_keys = orderpriorities.keys();
    let mut priority_label_indices = Vec::<Option<u8>>::with_capacity(priority_values.len());
    for index in 0..priority_values.len() {
        let priority = priority_values.value_bytes(index);
        let priority = std::str::from_utf8(priority)
            .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
        priority_label_indices.push(Some(q04_priority_label_index(priority, partial)?));
    }
    if orderkeys.null_count() == 0
        && orderdates.null_count() == 0
        && orderpriorities.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        for row in 0..orderkey_values.len() {
            let orderdate = orderdate_values[row];
            let orderkey = orderkey_values[row];
            if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
                continue;
            }
            let Ok(priority_key) = usize::try_from(priority_keys[row]) else {
                continue;
            };
            let Some(Some(priority_index)) = priority_label_indices.get(priority_key) else {
                continue;
            };
            partial.rows.push((orderkey, *priority_index));
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderdates.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let orderdate = orderdates.value(row);
        let orderkey = orderkeys.value(row);
        if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
            continue;
        }
        let Ok(priority_key) = usize::try_from(priority_keys[row]) else {
            continue;
        };
        let Some(Some(priority_index)) = priority_label_indices.get(priority_key) else {
            continue;
        };
        partial.rows.push((orderkey, *priority_index));
    }
    Ok(true)
}

fn q04_priority_label_index(priority: &str, partial: &mut Q04CandidatePartial) -> Result<u8> {
    if let Some(index) = partial.label_indices.get(priority) {
        return Ok(*index);
    }
    let next_index = u8::try_from(partial.labels.len())
        .map_err(|_| DodamError::UnsupportedSql("too many Q04 order priorities".to_string()))?;
    let priority = priority.to_string();
    partial.labels.push(priority.clone());
    partial.label_indices.insert(priority, next_index);
    Ok(next_index)
}

fn q04_late_candidate_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q04LateCandidateState,
) -> Result<Option<()>> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderdates = batch_column(&batch, "o_orderdate")?;
    let (Some(orderkeys), Some(orderdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        orderdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 && orderdates.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        let orderdate_values = orderdates.values().as_ref();
        selection.push_selected_offsets(
            orderkey_values.len(),
            (0..orderkey_values.len()).filter_map(|row| {
                let orderkey = orderkey_values[row];
                let orderdate = orderdate_values[row];
                let selected =
                    orderkey >= 0 && orderdate >= state.start_days && orderdate < state.end_days;
                if selected {
                    state.selected_orderkeys.push(orderkey);
                    Some(row)
                } else {
                    None
                }
            }),
        );
        return Ok(Some(()));
    }
    selection.push_selected_offsets(
        batch.num_rows(),
        (0..batch.num_rows()).filter_map(|row| {
            let selected = if orderkeys.is_null(row) || orderdates.is_null(row) {
                false
            } else {
                let orderkey = orderkeys.value(row);
                let orderdate = orderdates.value(row);
                orderkey >= 0 && orderdate >= state.start_days && orderdate < state.end_days
            };
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

fn q04_late_candidate_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q04LateCandidateState,
) -> Result<Option<()>> {
    if let Some(layout) = Q04CandidatePredicateView::try_new(view) {
        let orderkeys = layout.orderkeys;
        let orderdates = layout.orderdates;
        if let (Some(orderkey_values), Some(orderdate_values)) = (
            orderkeys.values_if_null_free(),
            orderdates.values_if_null_free(),
        ) {
            selection.push_selected_offsets(
                orderkey_values.len(),
                (0..orderkey_values.len()).filter_map(|row| {
                    let orderkey = orderkey_values[row];
                    let orderdate = orderdate_values[row];
                    let selected = orderkey >= 0
                        && orderdate >= state.start_days
                        && orderdate < state.end_days;
                    if selected {
                        state.selected_orderkeys.push(orderkey);
                        Some(row)
                    } else {
                        None
                    }
                }),
            );
            return Ok(Some(()));
        }
        selection.push_selected_offsets(
            view.num_rows(),
            (0..view.num_rows()).filter_map(|row| {
                let selected = if orderkeys.is_null(row) || orderdates.is_null(row) {
                    false
                } else {
                    let orderkey = orderkeys.value(row);
                    let orderdate = orderdates.value(row);
                    orderkey >= 0 && orderdate >= state.start_days && orderdate < state.end_days
                };
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
    q04_late_candidate_build_selection_batch(batch.clone(), selection, state)
}

fn q04_late_candidate_consume_priority_batch(
    batch: RecordBatch,
    state: &mut Q04LateCandidateState,
) -> Result<Option<()>> {
    let priorities = batch_string_column(&batch, "o_orderpriority")?;
    for row in 0..batch.num_rows() {
        let Some(&orderkey) = state.selected_orderkeys.get(state.priority_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q04 candidate payload row overflow".to_string(),
            ));
        };
        state.priority_offset += 1;
        if priorities.is_valid(row) {
            let priority = priorities.value(row);
            let priority_index = if let Some(index) = state.label_indices.get(priority) {
                *index
            } else {
                let next_index = u8::try_from(state.labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                state.labels.push(priority.to_string());
                state.label_indices.insert(priority.to_string(), next_index);
                next_index
            };
            state.rows.push((orderkey, priority_index));
        }
    }
    Ok(Some(()))
}

fn q04_late_candidate_consume_priority_view(
    view: BatchView<'_>,
    state: &mut Q04LateCandidateState,
) -> Result<Option<()>> {
    if let Some(layout) = Q04CandidatePayloadView::try_new(view) {
        let priorities = layout.priorities;
        for row in 0..view.num_rows() {
            let Some(&orderkey) = state.selected_orderkeys.get(state.priority_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "Q04 candidate payload row overflow".to_string(),
                ));
            };
            state.priority_offset += 1;
            if priorities.is_valid(row) {
                let priority = std::str::from_utf8(priorities.value_bytes(row))
                    .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
                let priority_index = if let Some(index) = state.label_indices.get(priority) {
                    *index
                } else {
                    let next_index = u8::try_from(state.labels.len()).map_err(|_| {
                        DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                    })?;
                    state.labels.push(priority.to_string());
                    state.label_indices.insert(priority.to_string(), next_index);
                    next_index
                };
                state.rows.push((orderkey, priority_index));
            }
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q04_late_candidate_consume_priority_batch(batch.clone(), state)
}

fn q04_candidate_priorities_from_partials(
    partials: Vec<Q04CandidatePartial>,
) -> Result<(Vec<u8>, Vec<String>, usize)> {
    let mut priorities = Vec::<u8>::new();
    let mut labels = Vec::<String>::new();
    let mut label_indices = HashMap::<String, u8>::new();
    let mut candidate_count = 0usize;
    for partial in partials {
        let mut label_remap = Vec::with_capacity(partial.labels.len());
        for priority in partial.labels {
            let priority_index = if let Some(index) = label_indices.get(priority.as_str()) {
                *index
            } else {
                let next_index = u8::try_from(labels.len()).map_err(|_| {
                    DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
                })?;
                labels.push(priority.clone());
                label_indices.insert(priority, next_index);
                next_index
            };
            label_remap.push(priority_index);
        }
        for (orderkey, local_priority_index) in partial.rows {
            let priority_index = *label_remap
                .get(usize::from(local_priority_index))
                .ok_or_else(|| {
                    DodamError::UnsupportedSql("Q04 priority label mismatch".to_string())
                })?;
            let orderkey = usize::try_from(orderkey)
                .map_err(|_| DodamError::UnsupportedSql("order key overflow".to_string()))?;
            if orderkey >= priorities.len() {
                priorities.resize(orderkey + 1, 0);
            }
            priorities[orderkey] = priority_index + 1;
            candidate_count += 1;
        }
    }
    Ok((priorities, labels, candidate_count))
}

fn q04_log_late_candidate_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    tpch_profile_late_materialized("Q04 candidates", metrics, row_group_chunk);
}

#[allow(clippy::too_many_arguments)]
fn q04_candidate_order_priorities_view_into(
    view: BatchView<'_>,
    start_days: i32,
    end_days: i32,
    priorities: &mut Vec<u8>,
    labels: &mut Vec<String>,
    label_indices: &mut HashMap<String, u8>,
    candidate_count: &mut usize,
) -> Result<()> {
    if view.num_columns() == 3
        && let (Some(orderkeys), Some(orderdates), Some(orderpriorities)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.utf8_vector(2),
        )
    {
        q04_candidate_order_priorities_vector_into(
            orderkeys,
            orderdates,
            orderpriorities,
            start_days,
            end_days,
            priorities,
            labels,
            label_indices,
            candidate_count,
        )?;
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q04 candidate priority raw vector columns have unsupported types".to_string(),
        ));
    };
    let orderkeys = batch_column(batch, "o_orderkey")?;
    let orderdates = batch_column(batch, "o_orderdate")?;
    let orderpriorities = batch_string_column(batch, "o_orderpriority")?;
    for row in 0..batch.num_rows() {
        if orderpriorities.is_null(row) {
            continue;
        }
        let (Some(orderkey), Some(orderdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(orderdates, row)?,
        ) else {
            continue;
        };
        if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
            continue;
        }
        let priority_index = if let Some(index) = label_indices.get(orderpriorities.value(row)) {
            *index
        } else {
            let next_index = u8::try_from(labels.len()).map_err(|_| {
                DodamError::UnsupportedSql("too many Q04 order priorities".to_string())
            })?;
            labels.push(orderpriorities.value(row).to_string());
            label_indices.insert(orderpriorities.value(row).to_string(), next_index);
            next_index
        };
        let orderkey = usize::try_from(orderkey)
            .map_err(|_| DodamError::UnsupportedSql("order key overflow".to_string()))?;
        if orderkey >= priorities.len() {
            priorities.resize(orderkey + 1, 0);
        }
        priorities[orderkey] = priority_index + 1;
        *candidate_count += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn q04_candidate_order_priorities_vector_into(
    orderkeys: I64VectorView<'_>,
    orderdates: Date32VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    start_days: i32,
    end_days: i32,
    priorities: &mut Vec<u8>,
    labels: &mut Vec<String>,
    label_indices: &mut HashMap<String, u8>,
    candidate_count: &mut usize,
) -> Result<()> {
    if let (Some(orderkey_values), Some(orderdate_values)) = (
        orderkeys.values_if_null_free(),
        orderdates.values_if_null_free(),
    ) && orderpriorities.null_count() == 0
    {
        for row in 0..orderkey_values.len() {
            let orderkey = orderkey_values[row];
            let orderdate = orderdate_values[row];
            if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
                continue;
            }
            let priority = std::str::from_utf8(orderpriorities.value_bytes(row))
                .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
            q04_store_candidate_priority(
                orderkey,
                priority,
                priorities,
                labels,
                label_indices,
                candidate_count,
            )?;
        }
        return Ok(());
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderdates.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        let orderdate = orderdates.value(row);
        if orderdate < start_days || orderdate >= end_days || orderkey < 0 {
            continue;
        }
        let priority = std::str::from_utf8(orderpriorities.value_bytes(row))
            .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
        q04_store_candidate_priority(
            orderkey,
            priority,
            priorities,
            labels,
            label_indices,
            candidate_count,
        )?;
    }
    Ok(())
}

fn q04_store_candidate_priority(
    orderkey: i64,
    priority: &str,
    priorities: &mut Vec<u8>,
    labels: &mut Vec<String>,
    label_indices: &mut HashMap<String, u8>,
    candidate_count: &mut usize,
) -> Result<()> {
    let priority_index = if let Some(index) = label_indices.get(priority) {
        *index
    } else {
        let next_index = u8::try_from(labels.len())
            .map_err(|_| DodamError::UnsupportedSql("too many Q04 order priorities".to_string()))?;
        labels.push(priority.to_string());
        label_indices.insert(priority.to_string(), next_index);
        next_index
    };
    let orderkey = usize::try_from(orderkey)
        .map_err(|_| DodamError::UnsupportedSql("order key overflow".to_string()))?;
    if orderkey >= priorities.len() {
        priorities.resize(orderkey + 1, 0);
    }
    priorities[orderkey] = priority_index + 1;
    *candidate_count += 1;
    Ok(())
}

async fn q04_count_late_candidate_priorities(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    candidate_priorities: &mut [u8],
    candidate_count: usize,
    priority_count: usize,
) -> Result<Vec<u64>> {
    if let Some(counts) = q04_count_late_candidate_priorities_late_materialized(
        engine,
        path.clone(),
        batch_size,
        candidate_priorities,
        priority_count,
    )
    .await?
    {
        return Ok(counts);
    }
    if let Some(counts) = q04_count_late_candidate_priorities_row_group_map(
        engine,
        path.clone(),
        batch_size,
        candidate_priorities,
        priority_count,
    )
    .await?
    {
        return Ok(counts);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
    ]);
    let mut stream = if q04_lineitem_row_filter_enabled(candidate_count) {
        let candidate_keys = q04_candidate_key_set(candidate_priorities);
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "l_orderkey",
                candidate_keys,
            )
            .await?
    } else if let Some((min_key, max_key)) =
        selective_candidate_priority_range(candidate_priorities)
    {
        engine
            .scan_parquet_batches_pruned(
                path,
                batch_size,
                projection,
                i64_range_pruning_predicates("l_orderkey", min_key, max_key),
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let mut counts = vec![0_u64; priority_count];
    while let Some(batch) = stream.next() {
        let batch = batch?;
        q04_count_late_candidate_priorities_view_into(
            BatchView::new(&batch),
            candidate_priorities,
            &mut counts,
        )?;
    }
    Ok(counts)
}

async fn q04_count_late_candidate_priorities_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    candidate_priorities: &[u8],
    priority_count: usize,
) -> Result<Option<Vec<u64>>> {
    let pruning_predicates = selective_candidate_priority_range(candidate_priorities)
        .map(|(min_key, max_key)| i64_range_pruning_predicates("l_orderkey", min_key, max_key))
        .unwrap_or_default();
    let candidate_priorities = q04_atomic_candidate_priorities(candidate_priorities);
    let predicate_projection = Projection::Columns(vec!["l_orderkey".to_string()]);
    let payload_projection = Projection::Columns(vec![
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
    ]);
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection.clone(),
            payload_projection.clone(),
            pruning_predicates,
            q04_lineitem_late_materialized_row_group_chunk(),
            generic_late_materialization_policy_for_projection(
                &predicate_projection,
                &payload_projection,
                0.10,
                Some(0.50),
            ),
            {
                let candidate_priorities = candidate_priorities.clone();
                move || Q04LineitemLateState {
                    candidate_priorities: candidate_priorities.clone(),
                    selected_orderkeys: Vec::new(),
                    payload_offset: 0,
                    counts: vec![0_u64; priority_count],
                }
            },
            q04_lineitem_late_build_selection_view,
            q04_lineitem_late_consume_dates_view,
            |state, metrics| {
                if state.payload_offset != state.selected_orderkeys.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q04 lineitem payload row mismatch".to_string(),
                    ));
                }
                Ok(Some((state.counts, metrics)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut counts = vec![0_u64; priority_count];
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        let (chunk_counts, chunk_metrics) = chunk.output;
        metrics.add(chunk_metrics);
        for (index, count) in chunk_counts.into_iter().enumerate() {
            counts[index] += count;
        }
    }
    q04_log_lineitem_late_materialized_profile(
        metrics,
        q04_lineitem_late_materialized_row_group_chunk(),
    );
    Ok(Some(counts))
}

fn q04_lineitem_late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(2)
}

struct Q04LineitemLateState {
    candidate_priorities: Arc<Vec<AtomicU8>>,
    selected_orderkeys: Vec<usize>,
    payload_offset: usize,
    counts: Vec<u64>,
}

fn q04_lineitem_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q04LineitemLateState,
) -> Result<Option<()>> {
    let Some(orderkeys) = batch_column(&batch, "l_orderkey")?
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 {
        let orderkey_values = orderkeys.values();
        selection.push_selected_offsets(
            orderkeys.len(),
            (0..orderkeys.len()).filter_map(|row| {
                let orderkey = orderkey_values[row];
                dense_atomic_marker_present_index(orderkey, &state.candidate_priorities).map(
                    |index| {
                        state.selected_orderkeys.push(index);
                        row
                    },
                )
            }),
        );
        return Ok(Some(()));
    }
    selection.push_selected_offsets(
        orderkeys.len(),
        (0..orderkeys.len()).filter_map(|row| {
            if orderkeys.is_null(row) {
                return None;
            }
            dense_atomic_marker_present_index(orderkeys.value(row), &state.candidate_priorities)
                .map(|index| {
                    state.selected_orderkeys.push(index);
                    row
                })
        }),
    );
    Ok(Some(()))
}

fn q04_lineitem_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q04LineitemLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(orderkeys) = view.i64_vector(0) else {
            return Ok(None);
        };
        if let Some(orderkey_values) = orderkeys.values_if_null_free() {
            selection.push_selected_offsets(
                orderkey_values.len(),
                (0..orderkey_values.len()).filter_map(|row| {
                    let orderkey = orderkey_values[row];
                    dense_atomic_marker_present_index(orderkey, &state.candidate_priorities).map(
                        |index| {
                            state.selected_orderkeys.push(index);
                            row
                        },
                    )
                }),
            );
            return Ok(Some(()));
        }
        selection.push_selected_offsets(
            orderkeys.len(),
            (0..orderkeys.len()).filter_map(|row| {
                if orderkeys.is_null(row) {
                    return None;
                }
                dense_atomic_marker_present_index(orderkeys.value(row), &state.candidate_priorities)
                    .map(|index| {
                        state.selected_orderkeys.push(index);
                        row
                    })
            }),
        );
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q04_lineitem_late_build_selection_batch(batch.clone(), selection, state)
}

fn q04_lineitem_late_consume_dates_batch(
    batch: RecordBatch,
    state: &mut Q04LineitemLateState,
) -> Result<Option<()>> {
    let commitdates = batch_column(&batch, "l_commitdate")?;
    let receiptdates = batch_column(&batch, "l_receiptdate")?;
    if let (Some(commitdates), Some(receiptdates)) = (
        commitdates.as_any().downcast_ref::<Date32Array>(),
        receiptdates.as_any().downcast_ref::<Date32Array>(),
    ) {
        if commitdates.null_count() == 0 && receiptdates.null_count() == 0 {
            let end = state.payload_offset.saturating_add(commitdates.len());
            if end > state.selected_orderkeys.len() {
                return Err(DodamError::UnsupportedSql(
                    "Q04 lineitem payload row overflow".to_string(),
                ));
            }
            let orderkeys = &state.selected_orderkeys[state.payload_offset..end];
            state.payload_offset = end;
            for (row, &orderkey) in orderkeys.iter().enumerate() {
                if commitdates.value(row) >= receiptdates.value(row) {
                    continue;
                }
                let Some(marker) = state.candidate_priorities.get(orderkey) else {
                    continue;
                };
                let priority_marker = marker.swap(0, Ordering::Relaxed);
                if priority_marker != 0 {
                    state.counts[usize::from(priority_marker - 1)] += 1;
                }
            }
            return Ok(Some(()));
        }
    }
    for row in 0..batch.num_rows() {
        let Some(&orderkey) = state.selected_orderkeys.get(state.payload_offset) else {
            return Err(DodamError::UnsupportedSql(
                "Q04 lineitem payload row overflow".to_string(),
            ));
        };
        state.payload_offset += 1;
        let (Some(commitdate), Some(receiptdate)) = (
            date32_value(commitdates, row)?,
            date32_value(receiptdates, row)?,
        ) else {
            continue;
        };
        if commitdate >= receiptdate {
            continue;
        }
        let Some(marker) = state.candidate_priorities.get(orderkey) else {
            continue;
        };
        let priority_marker = marker.swap(0, Ordering::Relaxed);
        if priority_marker != 0 {
            state.counts[usize::from(priority_marker - 1)] += 1;
        }
    }
    Ok(Some(()))
}

fn q04_lineitem_late_consume_dates_view(
    view: BatchView<'_>,
    state: &mut Q04LineitemLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(commitdates), Some(receiptdates)) =
            (view.date32_vector(0), view.date32_vector(1))
    {
        if q04_lineitem_late_consume_date_vectors(commitdates, receiptdates, state)? {
            return Ok(Some(()));
        }
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q04_lineitem_late_consume_dates_batch(batch.clone(), state)
}

fn q04_lineitem_late_consume_date_vectors(
    commitdates: Date32VectorView<'_>,
    receiptdates: Date32VectorView<'_>,
    state: &mut Q04LineitemLateState,
) -> Result<bool> {
    let (Some(commit_values), Some(receipt_values)) = (
        commitdates.values_if_null_free(),
        receiptdates.values_if_null_free(),
    ) else {
        return Ok(false);
    };
    let end = state.payload_offset.saturating_add(commit_values.len());
    if end > state.selected_orderkeys.len() {
        return Err(DodamError::UnsupportedSql(
            "Q04 lineitem payload row overflow".to_string(),
        ));
    }
    let orderkeys = &state.selected_orderkeys[state.payload_offset..end];
    state.payload_offset = end;
    for ((&commit_value, &receipt_value), &orderkey) in commit_values
        .iter()
        .zip(receipt_values.iter())
        .zip(orderkeys.iter())
    {
        if commit_value >= receipt_value {
            continue;
        }
        let Some(marker) = state.candidate_priorities.get(orderkey) else {
            continue;
        };
        let priority_marker = marker.swap(0, Ordering::Relaxed);
        if priority_marker != 0 {
            state.counts[usize::from(priority_marker - 1)] += 1;
        }
    }
    Ok(true)
}

fn q04_log_lineitem_late_materialized_profile(
    metrics: LateMaterializedMetrics,
    row_group_chunk: usize,
) {
    tpch_profile_late_materialized("Q04 lineitem", metrics, row_group_chunk);
}

fn q04_candidate_key_set(candidate_priorities: &[u8]) -> HashSet<i64> {
    let mut keys = HashSet::with_capacity(candidate_priorities.iter().filter(|&&p| p != 0).count());
    for (orderkey, priority) in candidate_priorities.iter().copied().enumerate() {
        if priority != 0 {
            keys.insert(orderkey as i64);
        }
    }
    keys
}

async fn q04_count_late_candidate_priorities_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    candidate_priorities: &[u8],
    priority_count: usize,
) -> Result<Option<Vec<u64>>> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
    ]);
    let pruning_predicates = selective_candidate_priority_range(candidate_priorities)
        .map(|(min_key, max_key)| i64_range_pruning_predicates("l_orderkey", min_key, max_key))
        .unwrap_or_default();
    q04_count_late_candidate_priorities_atomic_row_group_map(
        engine,
        path,
        batch_size,
        projection,
        pruning_predicates,
        candidate_priorities,
        priority_count,
    )
    .await
}

async fn q04_count_late_candidate_priorities_atomic_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    pruning_predicates: Vec<Expr>,
    candidate_priorities: &[u8],
    priority_count: usize,
) -> Result<Option<Vec<u64>>> {
    let candidate_priorities = q04_atomic_candidate_priorities(candidate_priorities);
    let Some(partials) = engine
        .parquet_row_group_map_pruned_view(
            path,
            batch_size,
            projection,
            pruning_predicates,
            q04_lineitem_row_group_map_chunk(),
            move || vec![0_u64; priority_count],
            {
                let candidate_priorities = candidate_priorities.clone();
                move |view, counts: &mut Vec<u64>| {
                    q04_count_late_candidate_priorities_atomic_view(
                        view,
                        &candidate_priorities,
                        counts,
                    )
                }
            },
            |counts| Ok(Some(counts)),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut counts = vec![0_u64; priority_count];
    for partial in partials {
        for (index, count) in partial.into_iter().enumerate() {
            counts[index] += count;
        }
    }
    Ok(Some(counts))
}

fn q04_count_late_candidate_priorities_atomic_view(
    view: BatchView<'_>,
    candidate_priorities: &[AtomicU8],
    counts: &mut [u64],
) -> Result<Option<()>> {
    q04_count_late_candidate_priorities_atomic_view_hits(view, candidate_priorities, counts)?;
    Ok(Some(()))
}

fn q04_count_late_candidate_priorities_atomic_view_hits(
    view: BatchView<'_>,
    candidate_priorities: &[AtomicU8],
    counts: &mut [u64],
) -> Result<usize> {
    if let Some((orderkeys, commitdates, receiptdates)) = view.raw_i64_i32_i32() {
        return Ok(q04_count_late_candidate_priorities_atomic_raw(
            orderkeys,
            commitdates,
            receiptdates,
            candidate_priorities,
            counts,
        ));
    }
    if view.num_columns() == 3
        && let (Some(orderkeys), Some(commitdates), Some(receiptdates)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.date32_vector(2),
        )
    {
        q04_count_late_candidate_priorities_atomic_vector(
            orderkeys,
            commitdates,
            receiptdates,
            candidate_priorities,
            counts,
        );
        return Ok(0);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q04 atomic lineitem raw vector columns have unsupported types".to_string(),
        ));
    };
    let _ = q04_count_late_candidate_priorities_atomic_batch(
        batch.clone(),
        candidate_priorities,
        counts,
    )?;
    Ok(0)
}

fn q04_lineitem_row_filter_enabled(candidate_key_count: usize) -> bool {
    candidate_key_count <= q04_lineitem_row_filter_max_keys()
}

fn q04_lineitem_row_filter_max_keys() -> usize {
    i64_set_row_filter_max_keys_with_default(100_000)
}

fn q04_atomic_candidate_priorities(candidate_priorities: &[u8]) -> Arc<Vec<AtomicU8>> {
    Arc::new(DenseAtomicU8::from_values_parallel(candidate_priorities).into_markers())
}

fn q04_lineitem_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(2)
}

fn q04_count_late_candidate_priorities_atomic_batch(
    batch: RecordBatch,
    candidate_priorities: &[AtomicU8],
    counts: &mut [u64],
) -> Result<Option<()>> {
    if batch.num_columns() == 3
        && q04_count_late_candidate_priorities_atomic_typed(
            batch.column(0),
            batch.column(1),
            batch.column(2),
            candidate_priorities,
            counts,
        )?
    {
        return Ok(Some(()));
    }
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let commitdates = batch_column(&batch, "l_commitdate")?;
    let receiptdates = batch_column(&batch, "l_receiptdate")?;
    if q04_count_late_candidate_priorities_atomic_typed(
        orderkeys,
        commitdates,
        receiptdates,
        candidate_priorities,
        counts,
    )? {
        return Ok(Some(()));
    }
    for row in 0..batch.num_rows() {
        let (Some(commitdate), Some(receiptdate)) = (
            date32_value(commitdates, row)?,
            date32_value(receiptdates, row)?,
        ) else {
            continue;
        };
        if commitdate >= receiptdate {
            continue;
        }
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        if orderkey < 0 {
            continue;
        }
        let Ok(orderkey) = usize::try_from(orderkey) else {
            continue;
        };
        let Some(marker) = candidate_priorities.get(orderkey) else {
            continue;
        };
        let priority_marker = marker.load(Ordering::Relaxed);
        if priority_marker == 0 {
            continue;
        }
        let priority_marker = marker.swap(0, Ordering::Relaxed);
        if priority_marker == 0 {
            continue;
        }
        counts[usize::from(priority_marker - 1)] += 1;
    }
    Ok(Some(()))
}

fn q04_count_late_candidate_priorities_atomic_typed(
    orderkeys: &ArrayRef,
    commitdates: &ArrayRef,
    receiptdates: &ArrayRef,
    candidate_priorities: &[AtomicU8],
    counts: &mut [u64],
) -> Result<bool> {
    let (Some(orderkeys), Some(commitdates), Some(receiptdates)) = (
        orderkeys.as_any().downcast_ref::<Int64Array>(),
        commitdates.as_any().downcast_ref::<Date32Array>(),
        receiptdates.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    if orderkeys.null_count() == 0
        && commitdates.null_count() == 0
        && receiptdates.null_count() == 0
    {
        let orderkey_values = orderkeys.values().as_ref();
        let commitdate_values = commitdates.values().as_ref();
        let receiptdate_values = receiptdates.values().as_ref();
        for row in 0..orderkey_values.len() {
            if commitdate_values[row] >= receiptdate_values[row] {
                continue;
            }
            let Some(orderkey) =
                dense_marker_index_i64(orderkey_values[row], candidate_priorities.len())
            else {
                continue;
            };
            count_dense_atomic_marker(orderkey, candidate_priorities, counts);
        }
        return Ok(true);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || commitdates.is_null(row) || receiptdates.is_null(row) {
            continue;
        }
        if commitdates.value(row) >= receiptdates.value(row) {
            continue;
        }
        let Some(orderkey) =
            dense_marker_index_i64(orderkeys.value(row), candidate_priorities.len())
        else {
            continue;
        };
        count_dense_atomic_marker(orderkey, candidate_priorities, counts);
    }
    Ok(true)
}

fn q04_count_late_candidate_priorities_atomic_vector(
    orderkeys: I64VectorView<'_>,
    commitdates: Date32VectorView<'_>,
    receiptdates: Date32VectorView<'_>,
    candidate_priorities: &[AtomicU8],
    counts: &mut [u64],
) {
    if let (Some(orderkey_values), Some(commitdate_values), Some(receiptdate_values)) = (
        orderkeys.values_if_null_free(),
        commitdates.values_if_null_free(),
        receiptdates.values_if_null_free(),
    ) {
        for row in 0..orderkey_values.len() {
            if commitdate_values[row] >= receiptdate_values[row] {
                continue;
            }
            let Some(orderkey) =
                dense_marker_index_i64(orderkey_values[row], candidate_priorities.len())
            else {
                continue;
            };
            count_dense_atomic_marker(orderkey, candidate_priorities, counts);
        }
        return;
    }

    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || commitdates.is_null(row) || receiptdates.is_null(row) {
            continue;
        }
        if commitdates.value(row) >= receiptdates.value(row) {
            continue;
        }
        let Some(orderkey) =
            dense_marker_index_i64(orderkeys.value(row), candidate_priorities.len())
        else {
            continue;
        };
        count_dense_atomic_marker(orderkey, candidate_priorities, counts);
    }
}

fn q04_count_late_candidate_priorities_atomic_raw(
    orderkeys: &[i64],
    commitdates: &[i32],
    receiptdates: &[i32],
    candidate_priorities: &[AtomicU8],
    counts: &mut [u64],
) -> usize {
    let mut hits = 0usize;
    for row in 0..orderkeys.len() {
        if commitdates[row] >= receiptdates[row] {
            continue;
        }
        let Some(orderkey) = dense_marker_index_i64(orderkeys[row], candidate_priorities.len())
        else {
            continue;
        };
        if count_dense_atomic_marker(orderkey, candidate_priorities, counts) {
            hits += 1;
        }
    }
    hits
}

#[inline(always)]
fn dense_atomic_marker_present_index(key: i64, markers: &[AtomicU8]) -> Option<usize> {
    let index = dense_marker_index_i64(key, markers.len())?;
    let marker = unsafe { markers.get_unchecked(index) };
    (marker.load(Ordering::Relaxed) != 0).then_some(index)
}

#[inline(always)]
fn dense_marker_index_i64(key: i64, len: usize) -> Option<usize> {
    if key < 0 {
        return None;
    }
    let key = key as u64;
    if key >= len as u64 {
        return None;
    }
    Some(key as usize)
}

#[inline(always)]
fn count_dense_atomic_marker(index: usize, markers: &[AtomicU8], counts: &mut [u64]) -> bool {
    debug_assert!(index < markers.len());
    // The explicit range check is done by dense_marker_index_i64 at the call site.
    let marker = unsafe { markers.get_unchecked(index) };
    let priority_marker = marker.load(Ordering::Relaxed);
    if priority_marker == 0 {
        return false;
    }
    let priority_marker = marker.swap(0, Ordering::Relaxed);
    if priority_marker == 0 {
        return false;
    }
    counts[usize::from(priority_marker - 1)] += 1;
    true
}

fn q04_count_late_candidate_priorities_view_into(
    view: BatchView<'_>,
    candidate_priorities: &mut [u8],
    counts: &mut [u64],
) -> Result<()> {
    if view.num_columns() == 3
        && let (Some(orderkeys), Some(commitdates), Some(receiptdates)) = (
            view.i64_vector(0),
            view.date32_vector(1),
            view.date32_vector(2),
        )
    {
        q04_count_late_candidate_priorities_vector(
            orderkeys,
            commitdates,
            receiptdates,
            candidate_priorities,
            counts,
        );
        return Ok(());
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q04 late candidate priority raw vector columns have unsupported types".to_string(),
        ));
    };
    let orderkeys = batch_column(batch, "l_orderkey")?;
    let commitdates = batch_column(batch, "l_commitdate")?;
    let receiptdates = batch_column(batch, "l_receiptdate")?;
    for row in 0..batch.num_rows() {
        let (Some(orderkey), Some(commitdate), Some(receiptdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(commitdates, row)?,
            date32_value(receiptdates, row)?,
        ) else {
            continue;
        };
        if commitdate >= receiptdate || orderkey < 0 {
            continue;
        }
        let Ok(orderkey) = usize::try_from(orderkey) else {
            continue;
        };
        let Some(priority_marker) = candidate_priorities.get_mut(orderkey) else {
            continue;
        };
        if *priority_marker == 0 {
            continue;
        }
        let priority_index = usize::from(*priority_marker - 1);
        counts[priority_index] += 1;
        *priority_marker = 0;
    }
    Ok(())
}

fn q04_count_late_candidate_priorities_vector(
    orderkeys: I64VectorView<'_>,
    commitdates: Date32VectorView<'_>,
    receiptdates: Date32VectorView<'_>,
    candidate_priorities: &mut [u8],
    counts: &mut [u64],
) {
    if let (Some(orderkey_values), Some(commitdate_values), Some(receiptdate_values)) = (
        orderkeys.values_if_null_free(),
        commitdates.values_if_null_free(),
        receiptdates.values_if_null_free(),
    ) {
        for row in 0..orderkey_values.len() {
            if commitdate_values[row] >= receiptdate_values[row] {
                continue;
            }
            let orderkey = orderkey_values[row];
            if orderkey < 0 {
                continue;
            }
            let Ok(orderkey) = usize::try_from(orderkey) else {
                continue;
            };
            let Some(priority_marker) = candidate_priorities.get_mut(orderkey) else {
                continue;
            };
            if *priority_marker == 0 {
                continue;
            }
            let priority_index = usize::from(*priority_marker - 1);
            counts[priority_index] += 1;
            *priority_marker = 0;
        }
        return;
    }

    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || commitdates.is_null(row) || receiptdates.is_null(row) {
            continue;
        }
        if commitdates.value(row) >= receiptdates.value(row) {
            continue;
        }
        let orderkey = orderkeys.value(row);
        if orderkey < 0 {
            continue;
        }
        let Ok(orderkey) = usize::try_from(orderkey) else {
            continue;
        };
        let Some(priority_marker) = candidate_priorities.get_mut(orderkey) else {
            continue;
        };
        if *priority_marker == 0 {
            continue;
        }
        let priority_index = usize::from(*priority_marker - 1);
        counts[priority_index] += 1;
        *priority_marker = 0;
    }
}

fn q04_priority_count_rows(priority_labels: Vec<String>, counts: Vec<u64>) -> Vec<Q04Row> {
    let mut rows = priority_labels
        .into_iter()
        .zip(counts)
        .filter_map(|(priority, count)| (count > 0).then_some(Q04Row { priority, count }))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.priority.cmp(&right.priority));
    rows
}

fn q04_output(rows: Vec<Q04Row>) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("o_orderpriority", DataType::Utf8, false),
            Field::new("order_count", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.priority.as_str()),
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
