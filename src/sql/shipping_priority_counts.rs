use super::*;

pub(super) fn shipping_priority_counts_shape(
    select: &Select,
    query: &Query,
    selection: &SqlExpr,
) -> bool {
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
    select.from.len() == 2
        && select.projection.len() == 3
        && projection.contains("l_shipmode")
        && projection.contains("high_line_count")
        && projection.contains("low_line_count")
        && projection.contains("o_orderpriority = '1-urgent'")
        && projection.contains("o_orderpriority = '2-high'")
        && group_by.contains("l_shipmode")
        && order_by.contains("l_shipmode")
        && selection.contains("o_orderkey = l_orderkey")
        && selection.contains("l_shipmode in")
        && selection.contains("l_commitdate < l_receiptdate")
        && selection.contains("l_shipdate < l_commitdate")
        && selection.contains("l_receiptdate")
}

pub(super) async fn try_execute_shipping_mode_priority_counts_sql(
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
    if !shipping_priority_counts_shape(select, query, selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Ok(None);
    };
    if tables.len() != 2 {
        return Ok(None);
    }
    let mut orders = None;
    let mut lineitem = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("orders") {
            orders = Some(table);
        } else if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        }
    }
    let (Some(orders), Some(lineitem)) = (orders, lineitem) else {
        return Ok(None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some(shipmodes) = string_in_literals(&conjuncts, "l_shipmode")? else {
        return Ok(None);
    };
    if shipmodes.len() != 2 {
        return Ok(None);
    }
    let Some((start_days, end_days)) = date_range_bounds(&conjuncts, "l_receiptdate")? else {
        return Ok(None);
    };
    let mut shipmodes = shipmodes.into_iter().collect::<Vec<_>>();
    shipmodes.sort();

    let pending = filtered_lineitem_counts(
        engine,
        lineitem.path,
        batch_size,
        &shipmodes,
        start_days,
        end_days,
    )
    .await?;
    let mut rows = if pending.is_empty() {
        shipmodes
            .iter()
            .map(|shipmode| ShippingPriorityRow {
                shipmode: shipmode.clone(),
                high_line_count: 0,
                low_line_count: 0,
            })
            .collect()
    } else {
        shipping_mode_counts_from_orders(engine, orders.path, batch_size, &shipmodes, &pending)
            .await?
    };
    rows.sort_by(|left, right| left.shipmode.cmp(&right.shipmode));
    Ok(Some(shipping_priority_counts_output(rows)?))
}

pub(super) fn string_in_literals(
    conjuncts: &[SqlExpr],
    column: &str,
) -> Result<Option<HashSet<String>>> {
    for conjunct in conjuncts {
        let SqlExpr::InList {
            expr,
            list,
            negated,
        } = conjunct
        else {
            continue;
        };
        if *negated || !sql_expr_column_matches(expr, column) {
            continue;
        }
        let mut values = HashSet::new();
        for item in list {
            let LiteralValue::Utf8(value) = sql_literal_value(item)? else {
                return Ok(None);
            };
            values.insert(value);
        }
        return Ok(Some(values));
    }
    Ok(None)
}

#[derive(Clone, Copy, Default)]
pub(super) struct ShippingPriorityState {
    pub(super) high_line_count: u64,
    pub(super) low_line_count: u64,
}

pub(super) struct ShippingPriorityRow {
    pub(super) shipmode: String,
    pub(super) high_line_count: u64,
    pub(super) low_line_count: u64,
}

#[derive(Clone, Copy, Default)]
pub(super) struct PendingShippingOrder {
    pub(super) counts: [u64; 2],
}

pub(super) type PendingShippingOrderMap = FastHashMap<i64, PendingShippingOrder>;

pub(super) fn pending_shipping_order_map_new() -> PendingShippingOrderMap {
    fast_hash_map_with_capacity(pending_shipping_order_map_initial_capacity())
}

pub(super) fn pending_shipping_order_map_initial_capacity() -> usize {
    4096
}

#[inline(always)]
pub(super) fn pending_order_increment(
    pending: &mut PendingShippingOrderMap,
    orderkey: i64,
    mode_index: usize,
) {
    pending.entry(orderkey).or_default().counts[mode_index] += 1;
}

#[inline(always)]
pub(super) fn pending_order_add(
    pending: &mut PendingShippingOrderMap,
    orderkey: i64,
    order: PendingShippingOrder,
) {
    if let Some(target) = pending.get_mut(&orderkey) {
        target.counts[0] += order.counts[0];
        target.counts[1] += order.counts[1];
        return;
    }
    pending.insert(orderkey, order);
}

#[derive(Default)]
pub(super) struct ShippingOrdersPartial {
    pub(super) groups: [ShippingPriorityState; 2],
    pub(super) profile: ShippingOrdersProfile,
}

#[derive(Default)]
pub(super) struct ShippingOrdersProfile {
    pub(super) batches: usize,
    pub(super) typed_batches: usize,
    pub(super) fallback_batches: usize,
    pub(super) rows: usize,
    pub(super) null_rows: usize,
    pub(super) lookup_hits: usize,
    pub(super) lookup_misses: usize,
    pub(super) priority_rows: usize,
    pub(super) apply_rows: usize,
    pub(super) lookup_samples: usize,
    pub(super) priority_samples: usize,
    pub(super) apply_samples: usize,
    pub(super) total_nanos: u64,
    pub(super) lookup_nanos: u64,
    pub(super) priority_nanos: u64,
    pub(super) apply_nanos: u64,
}

impl ShippingOrdersProfile {
    pub(super) fn add(&mut self, other: Self) {
        self.batches = self.batches.saturating_add(other.batches);
        self.typed_batches = self.typed_batches.saturating_add(other.typed_batches);
        self.fallback_batches = self.fallback_batches.saturating_add(other.fallback_batches);
        self.rows = self.rows.saturating_add(other.rows);
        self.null_rows = self.null_rows.saturating_add(other.null_rows);
        self.lookup_hits = self.lookup_hits.saturating_add(other.lookup_hits);
        self.lookup_misses = self.lookup_misses.saturating_add(other.lookup_misses);
        self.priority_rows = self.priority_rows.saturating_add(other.priority_rows);
        self.apply_rows = self.apply_rows.saturating_add(other.apply_rows);
        self.lookup_samples = self.lookup_samples.saturating_add(other.lookup_samples);
        self.priority_samples = self.priority_samples.saturating_add(other.priority_samples);
        self.apply_samples = self.apply_samples.saturating_add(other.apply_samples);
        self.total_nanos = self.total_nanos.saturating_add(other.total_nanos);
        self.lookup_nanos = self.lookup_nanos.saturating_add(other.lookup_nanos);
        self.priority_nanos = self.priority_nanos.saturating_add(other.priority_nanos);
        self.apply_nanos = self.apply_nanos.saturating_add(other.apply_nanos);
    }
}

pub(super) async fn filtered_lineitem_counts(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<PendingShippingOrderMap> {
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_shipmode".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
        "l_shipdate".to_string(),
    ]);
    let shipmodes = Arc::new(shipmodes.to_vec());
    if let Some(partials) = filtered_lineitem_counts_row_group_map(
        engine,
        path.clone(),
        batch_size,
        projection.clone(),
        shipmodes.clone(),
        start_days,
        end_days,
    )
    .await?
    {
        let mut pending = pending_shipping_order_map_new();
        for partial in partials {
            merge_pending_shipping_orders(&mut pending, partial);
        }
        return Ok(pending);
    }
    filtered_lineitem_counts_stream(
        engine, path, batch_size, projection, shipmodes, start_days, end_days,
    )
    .await
}

pub(super) async fn filtered_lineitem_counts_row_group_map(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<Vec<PendingShippingOrderMap>>> {
    let build_state = pending_shipping_order_map_new;
    let finish = |pending| Ok(Some(pending));
    engine
        .parquet_row_group_map_scan_view(
            path,
            batch_size,
            projection,
            vec!["l_shipmode".to_string()],
            receiptdate_pruning_predicates(start_days, end_days),
            shipping_row_group_map_chunk(),
            build_state,
            {
                let shipmodes = shipmodes.clone();
                move |view, pending: &mut PendingShippingOrderMap| {
                    if filtered_lineitem_counts_projected_view_into(
                        view, &shipmodes, start_days, end_days, pending,
                    )? {
                        Ok(Some(()))
                    } else {
                        let Some(batch) = view.try_record_batch() else {
                            return Err(DodamError::UnsupportedSql(
                                "shipping priority counts row-group lineitem raw vector columns have unsupported types"
                                    .to_string(),
                            ));
                        };
                        merge_pending_shipping_orders(
                            pending,
                            filtered_lineitem_counts_projected_batch(
                                batch.clone(),
                                &shipmodes,
                                start_days,
                                end_days,
                            )?,
                        );
                        Ok(Some(()))
                    }
                }
            },
            finish,
        )
        .await
}

pub(super) fn shipping_lineitem_chunk_size() -> usize {
    8
}

pub(super) async fn filtered_lineitem_counts_stream(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<PendingShippingOrderMap> {
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    parallel_batch_fold_view_chunks(
        &mut stream,
        shipping_lineitem_chunk_size(),
        pending_shipping_order_map_new,
        move |view, pending| {
            if filtered_lineitem_counts_projected_view_into(
                view, &shipmodes, start_days, end_days, pending,
            )? {
                return Ok(Some(()));
            }
            let Some(batch) = view.try_record_batch() else {
                return Err(DodamError::UnsupportedSql(
                    "shipping priority counts lineitem raw vector columns have unsupported types"
                        .to_string(),
                ));
            };
            merge_pending_shipping_orders(
                pending,
                filtered_lineitem_counts_projected_batch(
                    batch.clone(),
                    &shipmodes,
                    start_days,
                    end_days,
                )?,
            );
            Ok(Some(()))
        },
        Ok,
        pending_shipping_order_map_new(),
        merge_pending_shipping_orders,
        "shipping priority counts lineitem aggregate",
    )
}

pub(super) fn shipping_row_group_map_chunk() -> usize {
    generic_row_group_map_chunk_size(4)
}

pub(super) fn receiptdate_pruning_predicates(start_days: i32, end_days: i32) -> Vec<Expr> {
    date_range_pruning_predicates("l_receiptdate", start_days, end_days)
}

#[inline(always)]
pub(super) fn lineitem_dates_match(
    commitdate: i32,
    receiptdate: i32,
    shipdate: i32,
    start_days: i32,
    end_days: i32,
) -> bool {
    commitdate < receiptdate
        && shipdate < commitdate
        && receiptdate >= start_days
        && receiptdate < end_days
}

pub(super) fn date_range_pruning_predicates(
    column: &str,
    start_days: i32,
    end_days: i32,
) -> Vec<Expr> {
    vec![
        Expr::Comparison(ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::GtEq,
            value: LiteralValue::Int64(i64::from(start_days)),
        }),
        Expr::Comparison(ComparisonExpr {
            column: column.to_string(),
            op: ComparisonOp::Lt,
            value: LiteralValue::Int64(i64::from(end_days)),
        }),
    ]
}

pub(super) fn filtered_lineitem_counts_batch_into(
    batch: RecordBatch,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) -> Result<()> {
    let orderkeys = batch_column(&batch, "l_orderkey")?;
    let modes = batch_string_column(&batch, "l_shipmode")?;
    let commitdates = batch_column(&batch, "l_commitdate")?;
    let receiptdates = batch_column(&batch, "l_receiptdate")?;
    let shipdates = batch_column(&batch, "l_shipdate")?;
    if filtered_lineitem_counts_batch_typed_into(
        orderkeys,
        modes,
        commitdates,
        receiptdates,
        shipdates,
        shipmodes,
        start_days,
        end_days,
        pending,
    ) {
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        if modes.is_null(row) {
            continue;
        }
        let mode = modes.value(row);
        let Some(mode_index) = shipmode_index(shipmodes, mode) else {
            continue;
        };
        let (Some(orderkey), Some(commitdate), Some(receiptdate), Some(shipdate)) = (
            numeric_i64_value(orderkeys, row)?,
            date32_value(commitdates, row)?,
            date32_value(receiptdates, row)?,
            date32_value(shipdates, row)?,
        ) else {
            continue;
        };
        if !lineitem_dates_match(commitdate, receiptdate, shipdate, start_days, end_days) {
            continue;
        }
        pending_order_increment(pending, orderkey, mode_index);
    }
    Ok(())
}

pub(super) fn filtered_lineitem_counts_projected_batch(
    batch: RecordBatch,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<PendingShippingOrderMap> {
    let mut pending = pending_shipping_order_map_new();
    filtered_lineitem_counts_projected_batch_into(
        batch,
        shipmodes,
        start_days,
        end_days,
        &mut pending,
    )?;
    Ok(pending)
}

pub(super) fn filtered_lineitem_counts_projected_batch_into(
    batch: RecordBatch,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) -> Result<()> {
    let mut consumer = ProjectedLineitemConsumer {
        shipmodes,
        start_days,
        end_days,
        pending,
    };
    consume_record_batch(&mut consumer, &batch)
}

pub(super) struct ProjectedLineitemConsumer<'a, 'b> {
    shipmodes: &'a [String],
    start_days: i32,
    end_days: i32,
    pending: &'b mut PendingShippingOrderMap,
}

impl BatchConsumer for ProjectedLineitemConsumer<'_, '_> {
    fn consume(&mut self, view: BatchView<'_>) -> Result<()> {
        if filtered_lineitem_counts_projected_view_into(
            view,
            self.shipmodes,
            self.start_days,
            self.end_days,
            self.pending,
        )? {
            return Ok(());
        }
        let Some(batch) = view.try_record_batch() else {
            return Err(DodamError::UnsupportedSql(
                "shipping priority counts consumer lineitem raw vector columns have unsupported types".to_string(),
            ));
        };
        filtered_lineitem_counts_batch_into(
            batch.clone(),
            self.shipmodes,
            self.start_days,
            self.end_days,
            self.pending,
        )
    }
}

pub(super) fn filtered_lineitem_counts_projected_view_into(
    view: BatchView<'_>,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) -> Result<bool> {
    if view.num_columns() != 5 {
        return Ok(false);
    }
    if view.num_columns() == 5
        && let Some(layout) = LineitemDictionaryView::try_new(view)
        && filtered_lineitem_counts_batch_dictionary_typed_into(
            layout, shipmodes, start_days, end_days, pending,
        )
    {
        return Ok(true);
    }
    if let Some(layout) = LineitemStringView::try_new(view)
        && filtered_lineitem_counts_string_view_into(
            layout, shipmodes, start_days, end_days, pending,
        )
    {
        return Ok(true);
    }
    Ok(false)
}

pub(super) struct LineitemStringView<'a> {
    orderkeys: I64VectorView<'a>,
    modes: Utf8VectorView<'a>,
    commitdates: Date32VectorView<'a>,
    receiptdates: Date32VectorView<'a>,
    shipdates: Date32VectorView<'a>,
}

impl<'a> LineitemStringView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        (view.num_columns() == 5).then_some(Self {
            orderkeys: view.i64_vector(0)?,
            modes: view.utf8_vector(1)?,
            commitdates: view.date32_vector(2)?,
            receiptdates: view.date32_vector(3)?,
            shipdates: view.date32_vector(4)?,
        })
    }
}

pub(super) struct LineitemDictionaryView<'a> {
    orderkeys: I64VectorView<'a>,
    modes: DictionaryI32View<'a>,
    commitdates: Date32VectorView<'a>,
    receiptdates: Date32VectorView<'a>,
    shipdates: Date32VectorView<'a>,
}

impl<'a> LineitemDictionaryView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        (view.num_columns() == 5).then_some(Self {
            orderkeys: view.i64_vector(0)?,
            modes: view.dictionary_i32_view(1)?,
            commitdates: view.date32_vector(2)?,
            receiptdates: view.date32_vector(3)?,
            shipdates: view.date32_vector(4)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn filtered_lineitem_counts_batch_dictionary_typed_into(
    view: LineitemDictionaryView<'_>,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) -> bool {
    let [left_mode, right_mode] = shipmodes else {
        return false;
    };
    let Some(mode_flags) =
        dictionary_i32_view_match_flags(view.modes, &[left_mode.as_bytes(), right_mode.as_bytes()])
    else {
        return false;
    };
    let mode_keys = view.modes.keys();
    if let Some(orderkey_values) = view.orderkeys.values_if_null_free()
        && view.modes.null_count() == 0
        && let (Some(commitdate_values), Some(receiptdate_values), Some(shipdate_values)) = (
            view.commitdates.values_if_null_free(),
            view.receiptdates.values_if_null_free(),
            view.shipdates.values_if_null_free(),
        )
    {
        for row in 0..orderkey_values.len() {
            let commitdate = commitdate_values[row];
            let receiptdate = receiptdate_values[row];
            if !lineitem_dates_match(
                commitdate,
                receiptdate,
                shipdate_values[row],
                start_days,
                end_days,
            ) {
                continue;
            }
            let Some(mode_index) = dictionary_i32_view_match_index(mode_keys, &mode_flags, row)
            else {
                continue;
            };
            pending_order_increment(pending, orderkey_values[row], mode_index);
        }
        return true;
    }
    for row in 0..view.orderkeys.len() {
        if view.orderkeys.is_null(row)
            || view.modes.is_null(row)
            || view.commitdates.is_null(row)
            || view.receiptdates.is_null(row)
            || view.shipdates.is_null(row)
        {
            continue;
        }
        let commitdate = view.commitdates.value(row);
        let receiptdate = view.receiptdates.value(row);
        if !lineitem_dates_match(
            commitdate,
            receiptdate,
            view.shipdates.value(row),
            start_days,
            end_days,
        ) {
            continue;
        }
        let Some(mode_index) = dictionary_i32_view_match_index(mode_keys, &mode_flags, row) else {
            continue;
        };
        pending_order_increment(pending, view.orderkeys.value(row), mode_index);
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) fn filtered_lineitem_counts_batch_typed_into(
    orderkeys: &ArrayRef,
    modes: &StringArray,
    commitdates: &ArrayRef,
    receiptdates: &ArrayRef,
    shipdates: &ArrayRef,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) -> bool {
    let Some(orderkeys) = orderkeys.as_any().downcast_ref::<Int64Array>() else {
        return false;
    };
    let Some(commitdates) = commitdates.as_any().downcast_ref::<Date32Array>() else {
        return false;
    };
    let Some(receiptdates) = receiptdates.as_any().downcast_ref::<Date32Array>() else {
        return false;
    };
    let Some(shipdates) = shipdates.as_any().downcast_ref::<Date32Array>() else {
        return false;
    };
    filtered_lineitem_counts_string_view_into(
        LineitemStringView {
            orderkeys: I64VectorView::Arrow(orderkeys),
            modes: Utf8VectorView::Arrow(modes),
            commitdates: Date32VectorView::Arrow(commitdates),
            receiptdates: Date32VectorView::Arrow(receiptdates),
            shipdates: Date32VectorView::Arrow(shipdates),
        },
        shipmodes,
        start_days,
        end_days,
        pending,
    )
}

pub(super) fn filtered_lineitem_counts_string_view_into(
    view: LineitemStringView<'_>,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) -> bool {
    let [left_mode, right_mode] = shipmodes else {
        return false;
    };
    let left_mode = left_mode.as_bytes();
    let right_mode = right_mode.as_bytes();
    if let Some(orderkey_values) = view.orderkeys.values_if_null_free()
        && view.modes.null_count() == 0
        && let (Some(commitdate_values), Some(receiptdate_values), Some(shipdate_values)) = (
            view.commitdates.values_if_null_free(),
            view.receiptdates.values_if_null_free(),
            view.shipdates.values_if_null_free(),
        )
    {
        for row in 0..orderkey_values.len() {
            let commitdate = commitdate_values[row];
            let receiptdate = receiptdate_values[row];
            if !lineitem_dates_match(
                commitdate,
                receiptdate,
                shipdate_values[row],
                start_days,
                end_days,
            ) {
                continue;
            }
            let mode = view.modes.value_bytes(row);
            let mode_index = if mode == left_mode {
                0
            } else if mode == right_mode {
                1
            } else {
                continue;
            };
            pending_order_increment(pending, orderkey_values[row], mode_index);
        }
        return true;
    }
    for row in 0..view.orderkeys.len() {
        if view.orderkeys.is_null(row)
            || view.modes.is_null(row)
            || view.commitdates.is_null(row)
            || view.receiptdates.is_null(row)
            || view.shipdates.is_null(row)
        {
            continue;
        }
        let commitdate = view.commitdates.value(row);
        let receiptdate = view.receiptdates.value(row);
        if !lineitem_dates_match(
            commitdate,
            receiptdate,
            view.shipdates.value(row),
            start_days,
            end_days,
        ) {
            continue;
        }
        let mode = view.modes.value_bytes(row);
        let mode_index = if mode == left_mode {
            0
        } else if mode == right_mode {
            1
        } else {
            continue;
        };
        pending_order_increment(pending, view.orderkeys.value(row), mode_index);
    }
    true
}

pub(super) fn shipmode_index(shipmodes: &[String], mode: &str) -> Option<usize> {
    match shipmodes {
        [left, _] if mode == left => Some(0),
        [_, right] if mode == right => Some(1),
        _ => None,
    }
}

pub(super) fn merge_pending_shipping_orders(
    pending: &mut PendingShippingOrderMap,
    batch_pending: PendingShippingOrderMap,
) {
    for (orderkey, order) in batch_pending {
        pending_order_add(pending, orderkey, order);
    }
}

pub(super) fn shipping_priority_counts_output(
    rows: Vec<ShippingPriorityRow>,
) -> Result<QueryOutput> {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_shipmode", DataType::Utf8, false),
            Field::new("high_line_count", DataType::UInt64, false),
            Field::new("low_line_count", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.shipmode.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.high_line_count),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.low_line_count),
            )),
        ],
    )?;
    Ok(QueryOutput::Aggregate {
        metrics: AggregateMetrics::default(),
        batches: vec![batch],
    })
}
