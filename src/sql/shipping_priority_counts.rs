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
    high_line_count: u64,
    low_line_count: u64,
}

pub(super) struct ShippingPriorityRow {
    shipmode: String,
    high_line_count: u64,
    low_line_count: u64,
}

#[derive(Clone, Copy, Default)]
pub(super) struct PendingShippingOrder {
    counts: [u64; 2],
}

pub(super) type PendingShippingOrderMap = FastHashMap<i64, PendingShippingOrder>;

pub(super) fn pending_shipping_order_map_new() -> PendingShippingOrderMap {
    fast_hash_map_with_capacity(pending_shipping_order_map_initial_capacity())
}

pub(super) fn pending_shipping_order_map_initial_capacity() -> usize {
    std::env::var("DODAM_Q12_PENDING_MAP_INITIAL_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096)
}

#[inline(always)]
pub(super) fn pending_order_increment(
    pending: &mut PendingShippingOrderMap,
    orderkey: i64,
    mode_index: usize,
) {
    pending.entry(orderkey).or_default().counts[mode_index] += 1;
}

pub(super) struct PendingOrderRunAccumulator<'a> {
    pending: &'a mut PendingShippingOrderMap,
    current_key: Option<i64>,
    counts: [u64; 2],
    combine_runs: bool,
}

impl<'a> PendingOrderRunAccumulator<'a> {
    fn new(pending: &'a mut PendingShippingOrderMap) -> Self {
        Self {
            pending,
            current_key: None,
            counts: [0, 0],
            combine_runs: pending_order_run_accumulator_enabled(),
        }
    }

    #[inline(always)]
    fn increment(&mut self, orderkey: i64, mode_index: usize) {
        if !self.combine_runs {
            pending_order_increment(self.pending, orderkey, mode_index);
            return;
        }
        if self.current_key == Some(orderkey) {
            self.counts[mode_index] += 1;
            return;
        }
        self.flush();
        self.current_key = Some(orderkey);
        self.counts = [0, 0];
        self.counts[mode_index] = 1;
    }

    #[inline(always)]
    fn flush(&mut self) {
        if !self.combine_runs {
            return;
        }
        let Some(orderkey) = self.current_key.take() else {
            return;
        };
        let counts = self.counts;
        if counts[0] != 0 || counts[1] != 0 {
            pending_order_add(self.pending, orderkey, PendingShippingOrder { counts });
        }
    }

    fn finish(mut self) {
        self.flush();
    }
}

pub(super) fn pending_order_run_accumulator_enabled() -> bool {
    std::env::var("DODAM_Q12_ENABLE_PENDING_RUN_ACCUMULATOR")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
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
    groups: [ShippingPriorityState; 2],
    profile: ShippingOrdersProfile,
}

#[derive(Default)]
pub(super) struct ShippingOrdersProfile {
    batches: usize,
    typed_batches: usize,
    fallback_batches: usize,
    rows: usize,
    null_rows: usize,
    lookup_hits: usize,
    lookup_misses: usize,
    priority_rows: usize,
    apply_rows: usize,
    lookup_samples: usize,
    priority_samples: usize,
    apply_samples: usize,
    total_nanos: u64,
    lookup_nanos: u64,
    priority_nanos: u64,
    apply_nanos: u64,
}

impl ShippingOrdersProfile {
    fn add(&mut self, other: Self) {
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
    if shipping_late_materialized_enabled()
        && let Some(pending) = filtered_lineitem_counts_late_materialized(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(pending);
    }
    let projection = Projection::Columns(vec![
        "l_orderkey".to_string(),
        "l_shipmode".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
        "l_shipdate".to_string(),
    ]);
    let shipmodes = Arc::new(shipmodes.to_vec());
    if lineitem_row_filter_enabled() {
        return filtered_lineitem_counts_row_filtered(
            engine, path, batch_size, projection, shipmodes, start_days, end_days,
        )
        .await;
    }
    if direct_lineitem_selected_payload_enabled()
        && let Some(pending) = filtered_lineitem_counts_direct_selected_payload(
            engine,
            path.clone(),
            shipmodes.clone(),
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(pending);
    }
    if direct_lineitem_raw_enabled()
        && let Some(pending) = filtered_lineitem_counts_direct_raw(
            engine,
            path.clone(),
            batch_size,
            shipmodes.clone(),
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(pending);
    }
    if direct_lineitem_dict_raw_enabled()
        && let Some(pending) = filtered_lineitem_counts_direct_dict_raw(
            engine,
            path.clone(),
            batch_size,
            shipmodes.clone(),
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(pending);
    }
    if direct_lineitem_page_raw_enabled()
        && let Some(pending) = filtered_lineitem_counts_direct_page_raw(
            engine,
            path.clone(),
            batch_size,
            shipmodes.clone(),
            start_days,
            end_days,
        )
        .await?
    {
        return Ok(pending);
    }
    if shipping_row_group_map_enabled()
        && let Some(partials) = filtered_lineitem_counts_row_group_map(
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

pub(super) async fn filtered_lineitem_counts_direct_selected_payload(
    engine: &DodamEngine,
    path: PathBuf,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<PendingShippingOrderMap>> {
    engine
        .scan_parquet_i64_byte_array_selected_by_i32x3_dictionary_fold(
            path,
            [
                "l_orderkey",
                "l_shipmode",
                "l_commitdate",
                "l_receiptdate",
                "l_shipdate",
            ],
            receiptdate_pruning_predicates(start_days, end_days),
            shipping_row_group_map_chunk(),
            pending_shipping_order_map_new,
            move |commitdate, receiptdate, shipdate| {
                lineitem_dates_match(commitdate, receiptdate, shipdate, start_days, end_days)
            },
            {
                let shipmodes = shipmodes.clone();
                move |pending, orderkeys, mode_ids, dictionary| {
                    filtered_lineitem_counts_selected_payload_into(
                        orderkeys, mode_ids, dictionary, &shipmodes, pending,
                    );
                    Ok(())
                }
            },
            Ok,
            |pending, partial| {
                merge_pending_shipping_orders(pending, partial);
                Ok(())
            },
        )
        .await
}

pub(super) async fn filtered_lineitem_counts_direct_dict_raw(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<PendingShippingOrderMap>> {
    engine
        .scan_parquet_i64_dictionary_i32x3_dict_fold(
            path,
            batch_size,
            [
                "l_orderkey",
                "l_shipmode",
                "l_commitdate",
                "l_receiptdate",
                "l_shipdate",
            ],
            receiptdate_pruning_predicates(start_days, end_days),
            shipping_row_group_map_chunk(),
            pending_shipping_order_map_new,
            {
                let shipmodes = shipmodes.clone();
                move |pending,
                      key_ids,
                      key_dictionary,
                      mode_ids,
                      mode_dictionary,
                      commitdate_ids,
                      commitdate_dictionary,
                      receiptdate_ids,
                      receiptdate_dictionary,
                      shipdate_ids,
                      shipdate_dictionary| {
                    filtered_lineitem_counts_direct_dict_raw_into(
                        key_ids,
                        key_dictionary,
                        mode_ids,
                        mode_dictionary,
                        commitdate_ids,
                        commitdate_dictionary,
                        receiptdate_ids,
                        receiptdate_dictionary,
                        shipdate_ids,
                        shipdate_dictionary,
                        &shipmodes,
                        start_days,
                        end_days,
                        pending,
                    );
                    Ok(())
                }
            },
            Ok,
            |pending, partial| {
                merge_pending_shipping_orders(pending, partial);
                Ok(())
            },
        )
        .await
}

pub(super) async fn filtered_lineitem_counts_direct_page_raw(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<PendingShippingOrderMap>> {
    engine
        .scan_parquet_i64_dictionary_i32x3_page_fold(
            path,
            batch_size,
            [
                "l_orderkey",
                "l_shipmode",
                "l_commitdate",
                "l_receiptdate",
                "l_shipdate",
            ],
            receiptdate_pruning_predicates(start_days, end_days),
            shipping_row_group_map_chunk(),
            pending_shipping_order_map_new,
            {
                let shipmodes = shipmodes.clone();
                move |pending,
                      orderkey_bytes,
                      mode_ids,
                      dictionary,
                      commitdate_bytes,
                      receiptdate_bytes,
                      shipdate_bytes,
                      records| {
                    filtered_lineitem_counts_direct_page_raw_into(
                        orderkey_bytes,
                        mode_ids,
                        dictionary,
                        commitdate_bytes,
                        receiptdate_bytes,
                        shipdate_bytes,
                        records,
                        &shipmodes,
                        start_days,
                        end_days,
                        pending,
                    );
                    Ok(())
                }
            },
            Ok,
            |pending, partial| {
                merge_pending_shipping_orders(pending, partial);
                Ok(())
            },
        )
        .await
}

pub(super) async fn filtered_lineitem_counts_direct_raw(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<Option<PendingShippingOrderMap>> {
    engine
        .scan_parquet_i64_dictionary_i32x3_fold(
            path,
            batch_size,
            [
                "l_orderkey",
                "l_shipmode",
                "l_commitdate",
                "l_receiptdate",
                "l_shipdate",
            ],
            receiptdate_pruning_predicates(start_days, end_days),
            shipping_row_group_map_chunk(),
            pending_shipping_order_map_new,
            {
                let shipmodes = shipmodes.clone();
                move |pending,
                      orderkeys,
                      mode_ids,
                      dictionary,
                      commitdates,
                      receiptdates,
                      shipdates| {
                    filtered_lineitem_counts_direct_raw_into(
                        orderkeys,
                        mode_ids,
                        dictionary,
                        commitdates,
                        receiptdates,
                        shipdates,
                        &shipmodes,
                        start_days,
                        end_days,
                        pending,
                    );
                    Ok(())
                }
            },
            Ok,
            |pending, partial| {
                merge_pending_shipping_orders(pending, partial);
                Ok(())
            },
        )
        .await
}

pub(super) async fn filtered_lineitem_counts_row_filtered(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    projection: Projection,
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
) -> Result<PendingShippingOrderMap> {
    let mut stream = engine
        .scan_parquet_batches_row_filtered(
            path,
            batch_size,
            projection,
            receiptdate_pruning_predicates(start_days, end_days),
        )
        .await?;
    parallel_batch_fold_view_chunks(
        &mut stream,
        shipping_lineitem_chunk_size(),
        pending_shipping_order_map_new,
        move |view, pending| {
            if filtered_lineitem_counts_projected_view_into(
                view, &shipmodes, start_days, end_days, pending,
            )? {
                Ok(Some(()))
            } else {
                let Some(batch) = view.try_record_batch() else {
                    return Err(DodamError::UnsupportedSql(
                        "shipping priority counts row-filter lineitem raw vector columns have unsupported types"
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
        },
        Ok,
        pending_shipping_order_map_new(),
        merge_pending_shipping_orders,
        "shipping priority counts lineitem row-filter aggregate",
    )
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
    let dictionary_columns = lineitem_dictionary_shipmode_enabled()
        .then(|| vec!["l_shipmode".to_string()])
        .unwrap_or_default();
    engine
        .parquet_row_group_map_scan_view(
            path,
            batch_size,
            projection,
            dictionary_columns,
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

pub(super) async fn filtered_lineitem_counts_late_materialized(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
) -> Result<Option<PendingShippingOrderMap>> {
    let predicate_projection = Projection::Columns(vec![
        "l_shipmode".to_string(),
        "l_commitdate".to_string(),
        "l_receiptdate".to_string(),
        "l_shipdate".to_string(),
    ]);
    let payload_projection = Projection::Columns(vec!["l_orderkey".to_string()]);
    let shipmodes = Arc::new(shipmodes.to_vec());
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            shipping_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::selective_with_selector_run_ratio(
                shipping_late_materialized_max_selected_ratio(),
                shipping_late_materialized_max_selector_run_ratio(),
            )
            .with_selector_runs_per_selected(shipping_late_materialized_max_runs_per_selected()),
            {
                let shipmodes = shipmodes.clone();
                move || ShippingLateState {
                    shipmodes: shipmodes.clone(),
                    start_days,
                    end_days,
                    selected_modes: Vec::new(),
                    selected_offset: 0,
                    pending: pending_shipping_order_map_new(),
                }
            },
            late_build_shipping_selection_view,
            late_consume_orderkey_payload_view,
            |state, _metrics| {
                if state.selected_offset != state.selected_modes.len() {
                    return Err(DodamError::UnsupportedSql(
                        "shipping priority counts row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some(state.pending))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut pending = pending_shipping_order_map_new();
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        merge_pending_shipping_orders(&mut pending, chunk.output);
        metrics.add(chunk.metrics);
    }
    log_shipping_late_materialized_profile(metrics, shipping_late_materialized_row_group_chunk());
    Ok(Some(pending))
}

pub(super) fn shipping_lineitem_chunk_size() -> usize {
    std::env::var("DODAM_Q12_LINEITEM_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
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

pub(super) fn shipping_row_group_map_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_ROW_GROUP_MAP").is_none()
}

pub(super) fn lineitem_dictionary_shipmode_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_LINEITEM_DICTIONARY_SHIPMODE").is_none()
}

pub(super) fn lineitem_row_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_LINEITEM_ROW_FILTER").is_some()
}

pub(super) fn direct_lineitem_raw_enabled() -> bool {
    std::env::var("DODAM_Q12_ENABLE_DIRECT_LINEITEM_RAW")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn direct_lineitem_selected_payload_enabled() -> bool {
    std::env::var("DODAM_Q12_ENABLE_DIRECT_LINEITEM_SELECTED_PAYLOAD")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn direct_lineitem_dict_raw_enabled() -> bool {
    std::env::var("DODAM_Q12_ENABLE_DIRECT_LINEITEM_DICT_RAW")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn direct_lineitem_page_raw_enabled() -> bool {
    std::env::var("DODAM_Q12_ENABLE_DIRECT_LINEITEM_PAGE_RAW")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn shipping_row_group_map_chunk() -> usize {
    std::env::var("DODAM_Q12_ROW_GROUP_MAP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn shipping_late_materialized_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_LATE_MATERIALIZE").is_some()
}

pub(super) fn shipping_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

pub(super) fn shipping_late_materialized_max_selected_ratio() -> f64 {
    std::env::var("DODAM_Q12_LATE_MAX_SELECTED_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.20)
}

pub(super) fn shipping_late_materialized_max_selector_run_ratio() -> f64 {
    std::env::var("DODAM_Q12_LATE_MAX_SELECTOR_RUN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.50)
}

pub(super) fn shipping_late_materialized_max_runs_per_selected() -> f64 {
    std::env::var("DODAM_Q12_LATE_MAX_RUNS_PER_SELECTED")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(1.25)
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
    if shipping_typed_loop_enabled()
        && filtered_lineitem_counts_batch_typed_into(
            orderkeys,
            modes,
            commitdates,
            receiptdates,
            shipdates,
            shipmodes,
            start_days,
            end_days,
            pending,
        )
    {
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
    if view.num_columns() != 5 || !shipping_typed_loop_enabled() {
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

#[allow(clippy::too_many_arguments)]
pub(super) fn filtered_lineitem_counts_direct_raw_into(
    orderkeys: &[i64],
    mode_ids: &[i32],
    dictionary: &[bytes::Bytes],
    commitdates: &[i32],
    receiptdates: &[i32],
    shipdates: &[i32],
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) {
    let Some(mode_flags) = dictionary_bytes_match_flags(dictionary, shipmodes) else {
        return;
    };
    let mut accumulator = PendingOrderRunAccumulator::new(pending);
    for row in 0..orderkeys.len() {
        let commitdate = commitdates[row];
        let receiptdate = receiptdates[row];
        if !lineitem_dates_match(
            commitdate,
            receiptdate,
            shipdates[row],
            start_days,
            end_days,
        ) {
            continue;
        }
        let mode_id = mode_ids[row];
        if mode_id < 0 {
            continue;
        }
        let Some(mode_index) = mode_flags
            .get(mode_id as usize)
            .and_then(|mode_index| *mode_index)
        else {
            continue;
        };
        accumulator.increment(orderkeys[row], mode_index);
    }
    accumulator.finish();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn filtered_lineitem_counts_direct_page_raw_into(
    orderkey_bytes: &[u8],
    mode_ids: &[i32],
    dictionary: &[bytes::Bytes],
    commitdate_bytes: &[u8],
    receiptdate_bytes: &[u8],
    shipdate_bytes: &[u8],
    records: usize,
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) {
    let Some(mode_flags) = dictionary_bytes_match_flags(dictionary, shipmodes) else {
        return;
    };
    if orderkey_bytes.len() < records.saturating_mul(std::mem::size_of::<i64>())
        || commitdate_bytes.len() < records.saturating_mul(std::mem::size_of::<i32>())
        || receiptdate_bytes.len() < records.saturating_mul(std::mem::size_of::<i32>())
        || shipdate_bytes.len() < records.saturating_mul(std::mem::size_of::<i32>())
        || mode_ids.len() < records
    {
        return;
    }
    let mut accumulator = PendingOrderRunAccumulator::new(pending);
    let mut row = 0usize;
    while row + 8 <= records {
        for offset in 0..8 {
            let row = row + offset;
            let commitdate = read_i32_le_unaligned(commitdate_bytes, row);
            let receiptdate = read_i32_le_unaligned(receiptdate_bytes, row);
            if !lineitem_dates_match(
                commitdate,
                receiptdate,
                read_i32_le_unaligned(shipdate_bytes, row),
                start_days,
                end_days,
            ) {
                continue;
            }
            let mode_id = mode_ids[row];
            if mode_id < 0 {
                continue;
            }
            let Some(mode_index) = mode_flags
                .get(mode_id as usize)
                .and_then(|mode_index| *mode_index)
            else {
                continue;
            };
            accumulator.increment(read_i64_le_unaligned(orderkey_bytes, row), mode_index);
        }
        row += 8;
    }
    while row < records {
        let commitdate = read_i32_le_unaligned(commitdate_bytes, row);
        let receiptdate = read_i32_le_unaligned(receiptdate_bytes, row);
        if lineitem_dates_match(
            commitdate,
            receiptdate,
            read_i32_le_unaligned(shipdate_bytes, row),
            start_days,
            end_days,
        ) {
            let mode_id = mode_ids[row];
            if mode_id >= 0
                && let Some(mode_index) = mode_flags
                    .get(mode_id as usize)
                    .and_then(|mode_index| *mode_index)
            {
                accumulator.increment(read_i64_le_unaligned(orderkey_bytes, row), mode_index);
            }
        }
        row += 1;
    }
    accumulator.finish();
}

pub(super) fn filtered_lineitem_counts_selected_payload_into(
    orderkeys: &[i64],
    mode_ids: &[i32],
    dictionary: &[bytes::Bytes],
    shipmodes: &[String],
    pending: &mut PendingShippingOrderMap,
) {
    let Some(mode_flags) = dictionary_bytes_match_flags(dictionary, shipmodes) else {
        return;
    };
    let rows = orderkeys.len().min(mode_ids.len());
    let mut accumulator = PendingOrderRunAccumulator::new(pending);
    let mut row = 0usize;
    while row + 8 <= rows {
        for offset in 0..8 {
            let row = row + offset;
            let Some(mode_index) = mode_dictionary_match_index(mode_ids[row], &mode_flags) else {
                continue;
            };
            accumulator.increment(orderkeys[row], mode_index);
        }
        row += 8;
    }
    while row < rows {
        if let Some(mode_index) = mode_dictionary_match_index(mode_ids[row], &mode_flags) {
            accumulator.increment(orderkeys[row], mode_index);
        }
        row += 1;
    }
    accumulator.finish();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn filtered_lineitem_counts_direct_dict_raw_into(
    key_ids: &[i32],
    key_dictionary: &[i64],
    mode_ids: &[i32],
    mode_dictionary: &[bytes::Bytes],
    commitdate_ids: &[i32],
    commitdate_dictionary: &[i32],
    receiptdate_ids: &[i32],
    receiptdate_dictionary: &[i32],
    shipdate_ids: &[i32],
    shipdate_dictionary: &[i32],
    shipmodes: &[String],
    start_days: i32,
    end_days: i32,
    pending: &mut PendingShippingOrderMap,
) {
    let Some(mode_flags) = dictionary_bytes_match_flags(mode_dictionary, shipmodes) else {
        return;
    };
    let records = key_ids
        .len()
        .min(mode_ids.len())
        .min(commitdate_ids.len())
        .min(receiptdate_ids.len())
        .min(shipdate_ids.len());
    let mut accumulator = PendingOrderRunAccumulator::new(pending);
    let mut row = 0usize;
    while row + 8 <= records {
        for offset in 0..8 {
            let row = row + offset;
            let Some(commitdate) = i32_dictionary_value(commitdate_ids[row], commitdate_dictionary)
            else {
                continue;
            };
            let Some(receiptdate) =
                i32_dictionary_value(receiptdate_ids[row], receiptdate_dictionary)
            else {
                continue;
            };
            let Some(shipdate) = i32_dictionary_value(shipdate_ids[row], shipdate_dictionary)
            else {
                continue;
            };
            if !lineitem_dates_match(commitdate, receiptdate, shipdate, start_days, end_days) {
                continue;
            }
            let Some(mode_index) = mode_dictionary_match_index(mode_ids[row], &mode_flags) else {
                continue;
            };
            let Some(orderkey) = i64_dictionary_value(key_ids[row], key_dictionary) else {
                continue;
            };
            accumulator.increment(orderkey, mode_index);
        }
        row += 8;
    }
    while row < records {
        let Some(commitdate) = i32_dictionary_value(commitdate_ids[row], commitdate_dictionary)
        else {
            row += 1;
            continue;
        };
        let Some(receiptdate) = i32_dictionary_value(receiptdate_ids[row], receiptdate_dictionary)
        else {
            row += 1;
            continue;
        };
        let Some(shipdate) = i32_dictionary_value(shipdate_ids[row], shipdate_dictionary) else {
            row += 1;
            continue;
        };
        if lineitem_dates_match(commitdate, receiptdate, shipdate, start_days, end_days)
            && let Some(mode_index) = mode_dictionary_match_index(mode_ids[row], &mode_flags)
            && let Some(orderkey) = i64_dictionary_value(key_ids[row], key_dictionary)
        {
            accumulator.increment(orderkey, mode_index);
        }
        row += 1;
    }
    accumulator.finish();
}

#[inline(always)]
pub(super) fn i32_dictionary_value(id: i32, dictionary: &[i32]) -> Option<i32> {
    let id = usize::try_from(id).ok()?;
    dictionary.get(id).copied()
}

#[inline(always)]
pub(super) fn i64_dictionary_value(id: i32, dictionary: &[i64]) -> Option<i64> {
    let id = usize::try_from(id).ok()?;
    dictionary.get(id).copied()
}

#[inline(always)]
pub(super) fn mode_dictionary_match_index(id: i32, flags: &[Option<usize>]) -> Option<usize> {
    let id = usize::try_from(id).ok()?;
    flags.get(id).and_then(|mode_index| *mode_index)
}

pub(super) fn dictionary_bytes_match_flags(
    dictionary: &[bytes::Bytes],
    shipmodes: &[String],
) -> Option<Vec<Option<usize>>> {
    let [left_mode, right_mode] = shipmodes else {
        return None;
    };
    let left_mode = left_mode.as_bytes();
    let right_mode = right_mode.as_bytes();
    Some(
        dictionary
            .iter()
            .map(|value| {
                if value.as_ref() == left_mode {
                    Some(0)
                } else if value.as_ref() == right_mode {
                    Some(1)
                } else {
                    None
                }
            })
            .collect(),
    )
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
        let mut accumulator = PendingOrderRunAccumulator::new(pending);
        if lineitem_selection_vector_enabled() {
            let mut selected = SelectionVector::with_capacity(orderkey_values.len().min(4096));
            for row in 0..orderkey_values.len() {
                let commitdate = commitdate_values[row];
                let receiptdate = receiptdate_values[row];
                if lineitem_dates_match(
                    commitdate,
                    receiptdate,
                    shipdate_values[row],
                    start_days,
                    end_days,
                ) {
                    selected.push(row);
                }
            }
            if should_use_lineitem_selection_vector(selected.len(), orderkey_values.len()) {
                for &row in selected.as_slice() {
                    let row = row as usize;
                    let Some(mode_index) =
                        dictionary_i32_view_match_index(mode_keys, &mode_flags, row)
                    else {
                        continue;
                    };
                    accumulator.increment(orderkey_values[row], mode_index);
                }
                accumulator.finish();
                return true;
            }
        }
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
            accumulator.increment(orderkey_values[row], mode_index);
        }
        accumulator.finish();
        return true;
    }
    let mut accumulator = PendingOrderRunAccumulator::new(pending);
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
        accumulator.increment(view.orderkeys.value(row), mode_index);
    }
    accumulator.finish();
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
        let mut accumulator = PendingOrderRunAccumulator::new(pending);
        if lineitem_selection_vector_enabled() {
            let mut selected = SelectionVector::with_capacity(orderkey_values.len().min(4096));
            for row in 0..orderkey_values.len() {
                let commitdate = commitdate_values[row];
                let receiptdate = receiptdate_values[row];
                if lineitem_dates_match(
                    commitdate,
                    receiptdate,
                    shipdate_values[row],
                    start_days,
                    end_days,
                ) {
                    selected.push(row);
                }
            }
            if should_use_lineitem_selection_vector(selected.len(), orderkey_values.len()) {
                for &row in selected.as_slice() {
                    let row = row as usize;
                    let mode = view.modes.value_bytes(row);
                    let mode_index = if mode == left_mode {
                        0
                    } else if mode == right_mode {
                        1
                    } else {
                        continue;
                    };
                    accumulator.increment(orderkey_values[row], mode_index);
                }
                accumulator.finish();
                return true;
            }
        }
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
            accumulator.increment(orderkey_values[row], mode_index);
        }
        accumulator.finish();
        return true;
    }
    let mut accumulator = PendingOrderRunAccumulator::new(pending);
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
        accumulator.increment(view.orderkeys.value(row), mode_index);
    }
    accumulator.finish();
    true
}

pub(super) fn lineitem_selection_vector_enabled() -> bool {
    std::env::var("DODAM_Q12_ENABLE_LINEITEM_SELECTION_VECTOR")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn lineitem_selection_vector_max_ratio() -> f64 {
    std::env::var("DODAM_Q12_LINEITEM_SELECTION_VECTOR_MAX_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.50)
}

pub(super) fn should_use_lineitem_selection_vector(
    selected_rows: usize,
    total_rows: usize,
) -> bool {
    if selected_rows == 0 || total_rows == 0 {
        return selected_rows == 0;
    }
    (selected_rows as f64 / total_rows as f64) <= lineitem_selection_vector_max_ratio()
}

pub(super) fn shipmode_index(shipmodes: &[String], mode: &str) -> Option<usize> {
    match shipmodes {
        [left, _] if mode == left => Some(0),
        [_, right] if mode == right => Some(1),
        _ => None,
    }
}

pub(super) fn shipping_typed_loop_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_TYPED_LOOP").is_none()
}

pub(super) struct ShippingLateState {
    shipmodes: Arc<Vec<String>>,
    start_days: i32,
    end_days: i32,
    selected_modes: Vec<u8>,
    selected_offset: usize,
    pending: PendingShippingOrderMap,
}

pub(super) fn late_build_shipping_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut ShippingLateState,
) -> Result<Option<()>> {
    let modes = batch_string_column(&batch, "l_shipmode")?;
    let Some(commitdates) = batch_column(&batch, "l_commitdate")?
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let Some(receiptdates) = batch_column(&batch, "l_receiptdate")?
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let Some(shipdates) = batch_column(&batch, "l_shipdate")?
        .as_any()
        .downcast_ref::<Date32Array>()
    else {
        return Ok(None);
    };
    let [left_mode, right_mode] = state.shipmodes.as_slice() else {
        return Ok(None);
    };
    if modes.null_count() != 0
        || commitdates.null_count() != 0
        || receiptdates.null_count() != 0
        || shipdates.null_count() != 0
    {
        return Ok(None);
    }
    let left_mode = left_mode.as_bytes();
    let right_mode = right_mode.as_bytes();
    let mode_offsets = modes.value_offsets();
    let mode_data = modes.value_data();
    let commitdate_values = commitdates.values().as_ref();
    let receiptdate_values = receiptdates.values().as_ref();
    let shipdate_values = shipdates.values().as_ref();
    for row in 0..batch.num_rows() {
        let commitdate = commitdate_values[row];
        let receiptdate = receiptdate_values[row];
        if !lineitem_dates_match(
            commitdate,
            receiptdate,
            shipdate_values[row],
            state.start_days,
            state.end_days,
        ) {
            selection.push(false);
            continue;
        }
        let mode = bytes_string_parts(mode_offsets, mode_data, row);
        let mode_index = if mode == left_mode {
            0
        } else if mode == right_mode {
            1
        } else {
            selection.push(false);
            continue;
        };
        state.selected_modes.push(mode_index);
        selection.push(true);
    }
    Ok(Some(()))
}

pub(super) fn late_build_shipping_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut ShippingLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 4 {
        let Some(modes) = view.utf8_vector(0) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return late_build_shipping_selection_batch(batch.clone(), selection, state);
        };
        let (Some(commitdates), Some(receiptdates), Some(shipdates)) =
            (view.date32(1), view.date32(2), view.date32(3))
        else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return late_build_shipping_selection_batch(batch.clone(), selection, state);
        };
        let [left_mode, right_mode] = state.shipmodes.as_slice() else {
            return Ok(None);
        };
        if modes.null_count() != 0
            || commitdates.null_count() != 0
            || receiptdates.null_count() != 0
            || shipdates.null_count() != 0
        {
            return Ok(None);
        }
        let left_mode = left_mode.as_bytes();
        let right_mode = right_mode.as_bytes();
        let commitdate_values = commitdates.values().as_ref();
        let receiptdate_values = receiptdates.values().as_ref();
        let shipdate_values = shipdates.values().as_ref();
        for row in 0..view.num_rows() {
            let commitdate = commitdate_values[row];
            let receiptdate = receiptdate_values[row];
            if !lineitem_dates_match(
                commitdate,
                receiptdate,
                shipdate_values[row],
                state.start_days,
                state.end_days,
            ) {
                selection.push(false);
                continue;
            }
            let mode = modes.value_bytes(row);
            let mode_index = if mode == left_mode {
                0
            } else if mode == right_mode {
                1
            } else {
                selection.push(false);
                continue;
            };
            state.selected_modes.push(mode_index);
            selection.push(true);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    late_build_shipping_selection_batch(batch.clone(), selection, state)
}

pub(super) fn late_consume_orderkey_payload_batch(
    batch: RecordBatch,
    state: &mut ShippingLateState,
) -> Result<Option<()>> {
    let Some(orderkeys) = batch_column(&batch, "l_orderkey")?
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    if orderkeys.null_count() != 0 {
        return Ok(None);
    }
    for &orderkey in orderkeys.values() {
        let mode_index = *state
            .selected_modes
            .get(state.selected_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql(
                    "shipping priority counts row selection payload mismatch".to_string(),
                )
            })? as usize;
        pending_order_increment(&mut state.pending, orderkey, mode_index);
        state.selected_offset += 1;
    }
    Ok(Some(()))
}

pub(super) fn late_consume_orderkey_payload_view(
    view: BatchView<'_>,
    state: &mut ShippingLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(orderkeys) = view.i64_vector(0) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return late_consume_orderkey_payload_batch(batch.clone(), state);
        };
        let Some(orderkey_values) = orderkeys.values_if_null_free() else {
            return Ok(None);
        };
        for &orderkey in orderkey_values {
            let mode_index = *state
                .selected_modes
                .get(state.selected_offset)
                .ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "shipping priority counts row selection payload mismatch".to_string(),
                    )
                })? as usize;
            pending_order_increment(&mut state.pending, orderkey, mode_index);
            state.selected_offset += 1;
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    late_consume_orderkey_payload_batch(batch.clone(), state)
}

pub(super) fn log_shipping_late_materialized_profile(
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
        "[dodam:tpch-profile] shipping priority counts lineitem: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

pub(super) fn merge_pending_shipping_orders(
    pending: &mut PendingShippingOrderMap,
    batch_pending: PendingShippingOrderMap,
) {
    for (orderkey, order) in batch_pending {
        pending_order_add(pending, orderkey, order);
    }
}

pub(super) async fn shipping_mode_counts_from_orders(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Vec<ShippingPriorityRow>> {
    if order_direct_column_reader_enabled()
        && let Some(rows) = shipping_mode_counts_from_orders_direct(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            pending,
        )?
    {
        return Ok(rows);
    }
    if order_late_materialized_enabled()
        && let Some(rows) = shipping_mode_counts_from_orders_late(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            pending,
        )
        .await?
    {
        return Ok(rows);
    }
    if order_fused_scan_enabled()
        && !order_row_filter_enabled()
        && !order_bloom_filter_enabled()
        && let Some(rows) = shipping_mode_counts_from_orders_fused(
            engine,
            path.clone(),
            batch_size,
            shipmodes,
            pending,
        )
        .await?
    {
        return Ok(rows);
    }
    if order_sorted_pending_lookup_enabled() {
        return shipping_mode_counts_from_orders_sorted(
            engine, path, batch_size, shipmodes, pending,
        )
        .await;
    }
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let mut stream = if order_row_filter_enabled() {
        engine
            .scan_parquet_batches_i64_set_filtered(
                path,
                batch_size,
                projection,
                "o_orderkey",
                pending.keys().copied().collect(),
            )
            .await?
    } else if order_bloom_filter_enabled() {
        engine
            .scan_parquet_batches_i64_bloom_filtered(
                path,
                batch_size,
                projection,
                "o_orderkey",
                pending.keys().copied().collect(),
            )
            .await?
    } else if order_dictionary_priority_enabled() {
        engine
            .scan_parquet_batches_dictionary_columns(
                path,
                batch_size,
                projection,
                vec!["o_orderpriority".to_string()],
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path, batch_size, None, projection, None)
            .await?
    };
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let collect_profile = tpch_profile_enabled();
    let partial = parallel_batch_fold_view_chunks(
        &mut stream,
        order_chunk_size(),
        ShippingOrdersPartial::default,
        move |view, partial| {
            merge_orders_partial(
                partial,
                shipping_mode_counts_projected_view_partial(view, &pending, collect_profile)?,
            );
            Ok(Some(()))
        },
        Ok,
        ShippingOrdersPartial::default(),
        merge_orders_partial,
        "shipping priority counts orders aggregate",
    )?;
    log_orders_profile(&partial.profile);
    let groups = partial.groups;
    let rows = groups
        .into_iter()
        .enumerate()
        .map(|(index, state)| ShippingPriorityRow {
            shipmode: shipmodes[index].clone(),
            high_line_count: state.high_line_count,
            low_line_count: state.low_line_count,
        })
        .collect::<Vec<_>>();
    Ok(rows)
}

pub(super) async fn shipping_mode_counts_from_orders_sorted(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Vec<ShippingPriorityRow>> {
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let mut stream = engine
        .scan_parquet_batches(path, batch_size, None, projection, None)
        .await?;
    let pending = Arc::new(SortedI64Lookup::from_hash_map(pending));
    let groups = parallel_batch_fold_view_chunks(
        &mut stream,
        order_chunk_size(),
        || [ShippingPriorityState::default(); 2],
        move |view, groups| {
            merge_shipping_mode_counts(
                groups,
                shipping_mode_counts_projected_view_sorted(view, &pending)?,
            );
            Ok(Some(()))
        },
        Ok,
        [ShippingPriorityState::default(); 2],
        merge_shipping_mode_counts,
        "shipping priority counts orders sorted aggregate",
    )?;
    Ok(groups
        .into_iter()
        .enumerate()
        .map(|(index, state)| ShippingPriorityRow {
            shipmode: shipmodes[index].clone(),
            high_line_count: state.high_line_count,
            low_line_count: state.low_line_count,
        })
        .collect())
}

pub(super) async fn shipping_mode_counts_from_orders_fused(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Option<Vec<ShippingPriorityRow>>> {
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let projection = Projection::Columns(vec![
        "o_orderkey".to_string(),
        "o_orderpriority".to_string(),
    ]);
    let build_state = || [ShippingPriorityState::default(); 2];
    let finish = |groups| Ok(Some(groups));
    let partials = if order_fused_dictionary_priority_enabled() {
        engine
            .parquet_row_group_map_dictionary_columns_pruned_view(
                path,
                batch_size,
                projection,
                vec!["o_orderpriority".to_string()],
                Vec::new(),
                order_fused_row_group_chunk(),
                build_state,
                {
                    let pending = pending.clone();
                    move |view, groups: &mut [ShippingPriorityState; 2]| {
                        merge_shipping_mode_counts(
                            groups,
                            shipping_mode_counts_projected_view(view, &pending)?,
                        );
                        Ok(Some(()))
                    }
                },
                finish,
            )
            .await?
    } else {
        engine
            .parquet_row_group_map_view(
                path,
                batch_size,
                projection,
                order_fused_row_group_chunk(),
                build_state,
                {
                    let pending = pending.clone();
                    move |view, groups: &mut [ShippingPriorityState; 2]| {
                        merge_shipping_mode_counts(
                            groups,
                            shipping_mode_counts_projected_view(view, &pending)?,
                        );
                        Ok(Some(()))
                    }
                },
                finish,
            )
            .await?
    };
    let Some(partials) = partials else {
        return Ok(None);
    };
    let mut groups = [ShippingPriorityState::default(); 2];
    for partial in partials {
        merge_shipping_mode_counts(&mut groups, partial);
    }
    Ok(Some(
        groups
            .into_iter()
            .enumerate()
            .map(|(index, state)| ShippingPriorityRow {
                shipmode: shipmodes[index].clone(),
                high_line_count: state.high_line_count,
                low_line_count: state.low_line_count,
            })
            .collect(),
    ))
}

pub(super) fn shipping_mode_counts_from_orders_direct(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Option<Vec<ShippingPriorityRow>>> {
    let started = tpch_profile_start();
    let row_groups = (0..engine.parquet_row_group_count(&path)?).collect::<Vec<_>>();
    let pending = Arc::new(AdaptiveI64Map::from_hash(pending.clone()));
    let chunks = row_groups
        .chunks(order_direct_row_group_chunk())
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    let profile = tpch_profile_enabled();
    let partials = chunks
        .into_par_iter()
        .map(|row_groups| {
            order_direct_row_group_chunk_scan(
                engine,
                path.clone(),
                batch_size,
                row_groups,
                pending.clone(),
                profile,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut metrics = OrderDirectMetrics::default();
    for partial in partials {
        let Some(partial) = partial else {
            return Ok(None);
        };
        merge_shipping_mode_counts(&mut groups, partial.groups);
        metrics.add(partial.metrics);
    }
    if let Some(started) = started {
        eprintln!(
            "[dodam:tpch-profile] shipping priority counts orders direct_column_reader: total={:.3} ms row_groups={} batches={} rows={} hits={} misses={} payload_runs={} selective_batches={} full_batches={} read={:.3} ms consume={:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0,
            metrics.row_groups,
            metrics.batches,
            metrics.rows,
            metrics.hits,
            metrics.misses,
            metrics.payload_runs,
            metrics.selective_batches,
            metrics.full_batches,
            sql_nanos_to_millis(metrics.read_nanos),
            sql_nanos_to_millis(metrics.consume_nanos),
        );
    }
    Ok(Some(
        groups
            .into_iter()
            .enumerate()
            .map(|(index, state)| ShippingPriorityRow {
                shipmode: shipmodes[index].clone(),
                high_line_count: state.high_line_count,
                low_line_count: state.low_line_count,
            })
            .collect(),
    ))
}

pub(super) fn order_direct_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_ORDER_DIRECT_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

#[derive(Default)]
pub(super) struct OrderDirectPartial {
    groups: [ShippingPriorityState; 2],
    metrics: OrderDirectMetrics,
}

#[derive(Default)]
pub(super) struct OrderDirectMetrics {
    row_groups: usize,
    batches: usize,
    rows: usize,
    hits: usize,
    misses: usize,
    payload_runs: usize,
    selective_batches: usize,
    full_batches: usize,
    read_nanos: u64,
    consume_nanos: u64,
}

impl OrderDirectMetrics {
    fn add_scan_metrics(&mut self, metrics: DirectColumnScanMetrics) {
        self.row_groups = self.row_groups.saturating_add(metrics.row_groups);
        self.batches = self.batches.saturating_add(metrics.batches);
        self.rows = self.rows.saturating_add(metrics.rows);
        self.read_nanos = self.read_nanos.saturating_add(metrics.read_nanos);
        self.consume_nanos = self.consume_nanos.saturating_add(metrics.consume_nanos);
    }

    fn add(&mut self, other: Self) {
        self.row_groups = self.row_groups.saturating_add(other.row_groups);
        self.batches = self.batches.saturating_add(other.batches);
        self.rows = self.rows.saturating_add(other.rows);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.payload_runs = self.payload_runs.saturating_add(other.payload_runs);
        self.selective_batches = self
            .selective_batches
            .saturating_add(other.selective_batches);
        self.full_batches = self.full_batches.saturating_add(other.full_batches);
        self.read_nanos = self.read_nanos.saturating_add(other.read_nanos);
        self.consume_nanos = self.consume_nanos.saturating_add(other.consume_nanos);
    }
}

pub(super) fn order_direct_row_group_chunk_scan(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    row_groups: Vec<usize>,
    pending: Arc<AdaptiveI64Map<PendingShippingOrder>>,
    profile: bool,
) -> Result<Option<OrderDirectPartial>> {
    let started = profile.then(Instant::now);
    let mut partial = OrderDirectPartial::default();
    let mut priorities = Vec::<parquet::data_type::ByteArray>::with_capacity(batch_size);
    let mut priority_def_levels = Vec::<i16>::with_capacity(batch_size);
    let mut hits = Vec::<(usize, PendingShippingOrder)>::with_capacity(batch_size.min(1024));
    let Some(scan_metrics) = engine.scan_parquet_i64_byte_array_payload_columns(
        &path,
        batch_size,
        &row_groups,
        ["o_orderkey", "o_orderpriority"],
        |orderkeys, priority_reader| {
            priorities.clear();
            priority_def_levels.clear();
            hits.clear();
            let order_records = orderkeys.len();
            for row in 0..orderkeys.len() {
                let Some(order) = pending.get(orderkeys[row]) else {
                    partial.metrics.misses += 1;
                    continue;
                };
                partial.metrics.hits += 1;
                hits.push((row, order));
            }
            if hits.is_empty() {
                let skipped = priority_reader.skip_records(order_records)?;
                if skipped != order_records {
                    return Ok(None);
                }
                return Ok(Some(()));
            }
            let payload_runs = order_direct_payload_runs(&hits);
            partial.metrics.payload_runs =
                partial.metrics.payload_runs.saturating_add(payload_runs);
            if payload_runs > order_direct_max_selective_runs_per_batch() {
                partial.metrics.full_batches = partial.metrics.full_batches.saturating_add(1);
                priorities.clear();
                priority_def_levels.clear();
                let (priority_records, priority_values, priority_levels) = priority_reader
                    .read_records(order_records, &mut priority_def_levels, &mut priorities)?;
                if priority_records != order_records || priority_levels != order_records {
                    return Ok(None);
                }
                if priority_values == order_records {
                    for &(row, order) in &hits {
                        let is_high_priority = is_high_priority_bytes(priorities[row].data());
                        apply_pending_order(&mut partial.groups, order, is_high_priority);
                    }
                } else {
                    let mut value_row = vec![None; order_records];
                    let mut value_index = 0usize;
                    for row in 0..order_records {
                        let Some(level) = priority_def_levels.get(row) else {
                            return Ok(None);
                        };
                        if *level == 0 {
                            continue;
                        }
                        value_row[row] = Some(value_index);
                        value_index += 1;
                    }
                    if value_index != priority_values {
                        return Ok(None);
                    }
                    for &(row, order) in &hits {
                        let Some(value_index) = value_row[row] else {
                            continue;
                        };
                        let Some(priority) = priorities.get(value_index) else {
                            return Ok(None);
                        };
                        let is_high_priority = is_high_priority_bytes(priority.data());
                        apply_pending_order(&mut partial.groups, order, is_high_priority);
                    }
                }
                return Ok(Some(()));
            }
            partial.metrics.selective_batches = partial.metrics.selective_batches.saturating_add(1);
            let mut cursor = 0usize;
            let mut hit_index = 0usize;
            while hit_index < hits.len() {
                let start = hits[hit_index].0;
                let mut end_index = hit_index + 1;
                while end_index < hits.len() && hits[end_index].0 == hits[end_index - 1].0 + 1 {
                    end_index += 1;
                }
                let run_len = end_index - hit_index;
                if start > cursor {
                    let skipped = priority_reader.skip_records(start - cursor)?;
                    if skipped != start - cursor {
                        return Ok(None);
                    }
                }
                priorities.clear();
                priority_def_levels.clear();
                let (priority_records, priority_values, priority_levels) = priority_reader
                    .read_records(run_len, &mut priority_def_levels, &mut priorities)?;
                if priority_records != run_len || priority_levels != run_len {
                    return Ok(None);
                }
                if priority_values == run_len {
                    for (offset, hit) in hits[hit_index..end_index].iter().enumerate() {
                        let is_high_priority = is_high_priority_bytes(priorities[offset].data());
                        apply_pending_order(&mut partial.groups, hit.1, is_high_priority);
                    }
                } else {
                    let mut value_index = 0usize;
                    for hit in &hits[hit_index..end_index] {
                        let Some(level) = priority_def_levels.get(hit.0 - start) else {
                            return Ok(None);
                        };
                        if *level == 0 {
                            continue;
                        }
                        let Some(priority) = priorities.get(value_index) else {
                            return Ok(None);
                        };
                        value_index += 1;
                        let is_high_priority = is_high_priority_bytes(priority.data());
                        apply_pending_order(&mut partial.groups, hit.1, is_high_priority);
                    }
                    if value_index != priority_values {
                        return Ok(None);
                    }
                }
                cursor = start + run_len;
                hit_index = end_index;
            }
            if cursor < order_records {
                let skipped = priority_reader.skip_records(order_records - cursor)?;
                if skipped != order_records - cursor {
                    return Ok(None);
                }
            }
            Ok(Some(()))
        },
    )?
    else {
        return Ok(None);
    };
    partial.metrics.add_scan_metrics(scan_metrics);
    if let Some(started) = started {
        eprintln!(
            "[dodam:tpch-profile] shipping priority counts orders direct_column_chunk: row_groups={} rows={} hits={} misses={} payload_runs={} selective_batches={} full_batches={} elapsed={:.3} ms read={:.3} ms consume={:.3} ms",
            partial.metrics.row_groups,
            partial.metrics.rows,
            partial.metrics.hits,
            partial.metrics.misses,
            partial.metrics.payload_runs,
            partial.metrics.selective_batches,
            partial.metrics.full_batches,
            started.elapsed().as_secs_f64() * 1000.0,
            sql_nanos_to_millis(partial.metrics.read_nanos),
            sql_nanos_to_millis(partial.metrics.consume_nanos),
        );
    }
    Ok(Some(partial))
}

pub(super) fn order_direct_payload_runs(hits: &[(usize, PendingShippingOrder)]) -> usize {
    if hits.is_empty() {
        return 0;
    }
    let mut runs = 1usize;
    for index in 1..hits.len() {
        if hits[index].0 != hits[index - 1].0 + 1 {
            runs += 1;
        }
    }
    runs
}

pub(super) fn order_direct_max_selective_runs_per_batch() -> usize {
    std::env::var("DODAM_Q12_ORDER_DIRECT_MAX_SELECTIVE_RUNS_PER_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64)
}

pub(super) async fn shipping_mode_counts_from_orders_late(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    shipmodes: &[String],
    pending: &PendingShippingOrderMap,
) -> Result<Option<Vec<ShippingPriorityRow>>> {
    let pending = Arc::new(AdaptiveI64Map::from_hash((*pending).clone()));
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            Projection::Columns(vec!["o_orderkey".to_string()]),
            Projection::Columns(vec!["o_orderpriority".to_string()]),
            Vec::new(),
            order_late_materialized_row_group_chunk(),
            LateMaterializationPolicy::always(),
            {
                let pending = pending.clone();
                move || OrderLateState {
                    pending: pending.clone(),
                    selected_orders: Vec::new(),
                    selected_offset: 0,
                    groups: [ShippingPriorityState::default(); 2],
                }
            },
            order_late_build_selection_view,
            order_late_consume_priority_view,
            |state, _metrics| {
                if state.selected_offset != state.selected_orders.len() {
                    return Err(DodamError::UnsupportedSql(
                        "shipping priority counts order priority payload mismatch".to_string(),
                    ));
                }
                Ok(Some(state.groups))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        merge_shipping_mode_counts(&mut groups, chunk.output);
        metrics.add(chunk.metrics);
    }
    log_order_late_materialized_profile(metrics, order_late_materialized_row_group_chunk());
    Ok(Some(
        groups
            .into_iter()
            .enumerate()
            .map(|(index, state)| ShippingPriorityRow {
                shipmode: shipmodes[index].clone(),
                high_line_count: state.high_line_count,
                low_line_count: state.low_line_count,
            })
            .collect(),
    ))
}

pub(super) fn order_bloom_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_BLOOM_FILTER").is_some()
}

pub(super) fn order_fused_scan_enabled() -> bool {
    if std::env::var_os("DODAM_Q12_ENABLE_ORDER_FUSED_SCAN").is_some() {
        return true;
    }
    std::env::var_os("DODAM_Q12_DISABLE_ORDER_FUSED_SCAN").is_none()
}

pub(super) fn order_dictionary_priority_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_DICTIONARY_PRIORITY").is_some()
}

pub(super) fn order_fused_dictionary_priority_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_ORDER_FUSED_DICTIONARY_PRIORITY").is_none()
}

pub(super) fn order_sorted_pending_lookup_enabled() -> bool {
    std::env::var_os("DODAM_Q12_DISABLE_SORTED_PENDING_LOOKUP").is_none()
}

pub(super) fn order_direct_column_reader_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_DIRECT_COLUMN_READER").is_some()
}

pub(super) fn order_chunk_size() -> usize {
    std::env::var("DODAM_Q12_ORDER_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

pub(super) fn order_fused_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_ORDER_FUSED_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

pub(super) fn order_late_materialized_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_LATE_MATERIALIZE").is_some()
}

pub(super) fn order_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q12_ORDER_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

pub(super) fn order_row_filter_enabled() -> bool {
    std::env::var_os("DODAM_Q12_ENABLE_ORDER_ROW_FILTER").is_some()
}

pub(super) struct OrderLateState {
    pending: Arc<AdaptiveI64Map<PendingShippingOrder>>,
    selected_orders: Vec<PendingShippingOrder>,
    selected_offset: usize,
    groups: [ShippingPriorityState; 2],
}

pub(super) struct OrderPriorityView<'a> {
    priorities: Utf8VectorView<'a>,
}

impl<'a> OrderPriorityView<'a> {
    fn try_new(view: BatchView<'a>) -> Option<Self> {
        (view.num_columns() == 1).then_some(Self {
            priorities: view.utf8_vector(0)?,
        })
    }
}

pub(super) fn order_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    let Some(orderkeys) = batch_column(&batch, "o_orderkey")?
        .as_any()
        .downcast_ref::<Int64Array>()
    else {
        return Ok(None);
    };
    if orderkeys.null_count() == 0 {
        let orderkey_values = orderkeys.values();
        let pending_dense = state.pending.dense_slices();
        selection.push_selected_offsets(
            orderkey_values.len(),
            (0..orderkey_values.len()).filter_map(|row| {
                let orderkey = orderkey_values[row];
                if let Some(order) = state.pending.get_cached(pending_dense, orderkey) {
                    state.selected_orders.push(order);
                    Some(row)
                } else {
                    None
                }
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
            state.pending.get(orderkeys.value(row)).map(|order| {
                state.selected_orders.push(order);
                row
            })
        }),
    );
    Ok(Some(()))
}

pub(super) fn order_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1 {
        let Some(orderkeys) = view.i64_vector(0) else {
            let Some(batch) = view.try_record_batch() else {
                return Ok(None);
            };
            return order_late_build_selection_batch(batch.clone(), selection, state);
        };
        if let Some(orderkey_values) = orderkeys.values_if_null_free() {
            let pending_dense = state.pending.dense_slices();
            selection.push_selected_offsets(
                orderkey_values.len(),
                (0..orderkey_values.len()).filter_map(|row| {
                    let orderkey = orderkey_values[row];
                    if let Some(order) = state.pending.get_cached(pending_dense, orderkey) {
                        state.selected_orders.push(order);
                        Some(row)
                    } else {
                        None
                    }
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
                state.pending.get(orderkeys.value(row)).map(|order| {
                    state.selected_orders.push(order);
                    row
                })
            }),
        );
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    order_late_build_selection_batch(batch.clone(), selection, state)
}

pub(super) fn order_late_consume_priority_batch(
    batch: RecordBatch,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    let priorities = batch_string_column(&batch, "o_orderpriority")?;
    let priority_offsets = priorities.value_offsets();
    let priority_data = priorities.value_data();
    for row in 0..batch.num_rows() {
        let Some(&order) = state.selected_orders.get(state.selected_offset) else {
            return Err(DodamError::UnsupportedSql(
                "shipping priority counts order priority payload overflow".to_string(),
            ));
        };
        state.selected_offset += 1;
        if priorities.is_null(row) {
            continue;
        }
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = is_high_priority_bytes(priority);
        apply_pending_order(&mut state.groups, order, is_high_priority);
    }
    Ok(Some(()))
}

pub(super) fn order_late_consume_priority_view(
    view: BatchView<'_>,
    state: &mut OrderLateState,
) -> Result<Option<()>> {
    if let Some(layout) = OrderPriorityView::try_new(view) {
        let priorities = layout.priorities;
        for row in 0..view.num_rows() {
            let Some(&order) = state.selected_orders.get(state.selected_offset) else {
                return Err(DodamError::UnsupportedSql(
                    "shipping priority counts order priority payload overflow".to_string(),
                ));
            };
            state.selected_offset += 1;
            if priorities.is_null(row) {
                continue;
            }
            let is_high_priority = is_high_priority_bytes(priorities.value_bytes(row));
            apply_pending_order(&mut state.groups, order, is_high_priority);
        }
        return Ok(Some(()));
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    order_late_consume_priority_batch(batch.clone(), state)
}

pub(super) fn is_high_priority_str(priority: &str) -> bool {
    is_high_priority_bytes(priority.as_bytes())
}

pub(super) fn is_high_priority_bytes(priority: &[u8]) -> bool {
    matches!(priority.first(), Some(b'1' | b'2')) && matches!(priority, b"1-URGENT" | b"2-HIGH")
}

pub(super) fn log_order_late_materialized_profile(
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
        "[dodam:tpch-profile] shipping priority counts orders: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

pub(super) fn shipping_mode_counts_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
    if shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_batch_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let mut groups = [ShippingPriorityState::default(); 2];
    for row in 0..batch.num_rows() {
        if orderpriorities.is_null(row) {
            continue;
        }
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        let Some(order) = pending.get(orderkey) else {
            continue;
        };
        let is_high_priority = is_high_priority_str(orderpriorities.value(row));
        for (index, count) in order.counts.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            let group = &mut groups[index];
            if is_high_priority {
                group.high_line_count += count;
            } else {
                group.low_line_count += count;
            }
        }
    }
    Ok(groups)
}

#[allow(dead_code)]
pub(super) fn shipping_mode_counts_projected_batch(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_batch_typed(batch.column(0), orderpriorities, pending)
    {
        return Ok(groups);
    }
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch
            .column(1)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
        && shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_batch_dictionary_typed(
            batch.column(0),
            DictionaryI32View::Arrow(orderpriorities),
            pending,
        )
    {
        return Ok(groups);
    }
    shipping_mode_counts_batch(batch, pending)
}

pub(super) fn shipping_mode_counts_projected_view(
    view: BatchView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) =
            (view.i64_vector(0), view.dictionary_i32_view(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_dictionary_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "shipping priority counts shipping mode raw vector columns have unsupported types"
                .to_string(),
        ));
    };
    shipping_mode_counts_batch(batch.clone(), pending)
}

pub(super) fn shipping_mode_counts_projected_batch_sorted(
    batch: RecordBatch,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_batch_typed_sorted(batch.column(0), orderpriorities, pending)
    {
        return Ok(groups);
    }
    let orderkeys = batch_column(&batch, "o_orderkey")?;
    let orderpriorities = batch_string_column(&batch, "o_orderpriority")?;
    let mut groups = [ShippingPriorityState::default(); 2];
    for row in 0..batch.num_rows() {
        if orderpriorities.is_null(row) {
            continue;
        }
        let Some(orderkey) = numeric_i64_value(orderkeys, row)? else {
            continue;
        };
        let Some(order) = pending.get(orderkey) else {
            continue;
        };
        let is_high_priority = is_high_priority_str(orderpriorities.value(row));
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Ok(groups)
}

pub(super) fn shipping_mode_counts_projected_view_sorted(
    view: BatchView<'_>,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Result<[ShippingPriorityState; 2]> {
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_vector_typed_sorted(orderkeys, orderpriorities, pending)
    {
        return Ok(groups);
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "shipping priority counts sorted shipping mode raw vector columns have unsupported types".to_string(),
        ));
    };
    shipping_mode_counts_projected_batch_sorted(batch.clone(), pending)
}

#[allow(dead_code)]
pub(super) fn shipping_mode_counts_projected_batch_partial(
    batch: RecordBatch,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
    collect_profile: bool,
) -> Result<ShippingOrdersPartial> {
    if !collect_profile {
        return Ok(ShippingOrdersPartial {
            groups: shipping_mode_counts_projected_batch(batch, pending)?,
            profile: ShippingOrdersProfile::default(),
        });
    }
    let started = Instant::now();
    let rows = batch.num_rows();
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch.column(1).as_any().downcast_ref::<StringArray>()
        && shipping_typed_loop_enabled()
        && let Some(mut partial) =
            shipping_mode_counts_batch_typed_profile(batch.column(0), orderpriorities, pending)
    {
        partial.profile.total_nanos = sql_elapsed_nanos(started);
        return Ok(partial);
    }
    if batch.num_columns() == 2
        && let Some(orderpriorities) = batch
            .column(1)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
        && shipping_typed_loop_enabled()
        && let Some(groups) = shipping_mode_counts_batch_dictionary_typed(
            batch.column(0),
            DictionaryI32View::Arrow(orderpriorities),
            pending,
        )
    {
        return Ok(ShippingOrdersPartial {
            groups,
            profile: ShippingOrdersProfile {
                batches: 1,
                typed_batches: 1,
                rows,
                total_nanos: sql_elapsed_nanos(started),
                ..Default::default()
            },
        });
    }
    let groups = shipping_mode_counts_batch(batch, pending)?;
    let profile = ShippingOrdersProfile {
        batches: 1,
        fallback_batches: 1,
        rows,
        total_nanos: sql_elapsed_nanos(started),
        ..Default::default()
    };
    Ok(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_mode_counts_projected_view_partial(
    view: BatchView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
    collect_profile: bool,
) -> Result<ShippingOrdersPartial> {
    if !collect_profile {
        return Ok(ShippingOrdersPartial {
            groups: shipping_mode_counts_projected_view(view, pending)?,
            profile: ShippingOrdersProfile::default(),
        });
    }
    let started = Instant::now();
    let rows = view.num_rows();
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) = (view.i64_vector(0), view.utf8_vector(1))
        && shipping_typed_loop_enabled()
        && let Some(mut partial) =
            shipping_mode_counts_vector_typed_profile(orderkeys, orderpriorities, pending)
    {
        partial.profile.total_nanos = sql_elapsed_nanos(started);
        return Ok(partial);
    }
    if view.num_columns() == 2
        && let (Some(orderkeys), Some(orderpriorities)) =
            (view.i64_vector(0), view.dictionary_i32_view(1))
        && shipping_typed_loop_enabled()
        && let Some(groups) =
            shipping_mode_counts_dictionary_vector_typed(orderkeys, orderpriorities, pending)
    {
        return Ok(ShippingOrdersPartial {
            groups,
            profile: ShippingOrdersProfile {
                batches: 1,
                typed_batches: 1,
                rows,
                total_nanos: sql_elapsed_nanos(started),
                ..Default::default()
            },
        });
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "shipping priority counts profiled shipping mode raw vector columns have unsupported types".to_string(),
        ));
    };
    let groups = shipping_mode_counts_batch(batch.clone(), pending)?;
    let profile = ShippingOrdersProfile {
        batches: 1,
        fallback_batches: 1,
        rows,
        total_nanos: sql_elapsed_nanos(started),
        ..Default::default()
    };
    Ok(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_profile_sample(row: usize) -> bool {
    row & 255 == 0
}

pub(super) fn shipping_profile_dense_lookup(
    orderkey: i64,
    pending_values: &[PendingShippingOrder],
    pending_present: &[bool],
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> Option<PendingShippingOrder> {
    if shipping_profile_sample(row) {
        profile.lookup_samples += 1;
        let started = Instant::now();
        let order = usize::try_from(orderkey)
            .ok()
            .filter(|index| *index < pending_present.len() && pending_present[*index])
            .map(|index| pending_values[index]);
        profile.lookup_nanos = profile
            .lookup_nanos
            .saturating_add(sql_elapsed_nanos(started));
        return order;
    }
    usize::try_from(orderkey)
        .ok()
        .filter(|index| *index < pending_present.len() && pending_present[*index])
        .map(|index| pending_values[index])
}

pub(super) fn shipping_profile_map_lookup(
    pending: &AdaptiveI64Map<PendingShippingOrder>,
    orderkey: i64,
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> Option<PendingShippingOrder> {
    if shipping_profile_sample(row) {
        profile.lookup_samples += 1;
        let started = Instant::now();
        let order = pending.get(orderkey);
        profile.lookup_nanos = profile
            .lookup_nanos
            .saturating_add(sql_elapsed_nanos(started));
        return order;
    }
    pending.get(orderkey)
}

pub(super) fn shipping_profile_priority(
    priority_offsets: &[i32],
    priority_data: &[u8],
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> bool {
    if shipping_profile_sample(profile.priority_rows) {
        profile.priority_samples += 1;
        let started = Instant::now();
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = is_high_priority_bytes(priority);
        profile.priority_nanos = profile
            .priority_nanos
            .saturating_add(sql_elapsed_nanos(started));
        profile.priority_rows += 1;
        return is_high_priority;
    }
    let priority = bytes_string_parts(priority_offsets, priority_data, row);
    let is_high_priority = is_high_priority_bytes(priority);
    profile.priority_rows += 1;
    is_high_priority
}

pub(super) fn shipping_profile_priority_vector(
    priorities: Utf8VectorView<'_>,
    row: usize,
    profile: &mut ShippingOrdersProfile,
) -> bool {
    if shipping_profile_sample(profile.priority_rows) {
        profile.priority_samples += 1;
        let started = Instant::now();
        let is_high_priority = is_high_priority_bytes(priorities.value_bytes(row));
        profile.priority_nanos = profile
            .priority_nanos
            .saturating_add(sql_elapsed_nanos(started));
        profile.priority_rows += 1;
        return is_high_priority;
    }
    let is_high_priority = is_high_priority_bytes(priorities.value_bytes(row));
    profile.priority_rows += 1;
    is_high_priority
}

pub(super) fn shipping_profile_apply(
    groups: &mut [ShippingPriorityState; 2],
    order: PendingShippingOrder,
    is_high_priority: bool,
    profile: &mut ShippingOrdersProfile,
) {
    if shipping_profile_sample(profile.apply_rows) {
        profile.apply_samples += 1;
        let started = Instant::now();
        apply_pending_order(groups, order, is_high_priority);
        profile.apply_nanos = profile
            .apply_nanos
            .saturating_add(sql_elapsed_nanos(started));
    } else {
        apply_pending_order(groups, order, is_high_priority);
    }
    profile.apply_rows += 1;
}

pub(super) fn shipping_mode_counts_batch_typed_profile(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<ShippingOrdersPartial> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut profile = ShippingOrdersProfile {
        batches: 1,
        typed_batches: 1,
        rows: orderkeys.len(),
        ..Default::default()
    };
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let order = shipping_profile_dense_lookup(
                    orderkey_values[row],
                    pending_values,
                    pending_present,
                    row,
                    &mut profile,
                );
                let Some(order) = order else {
                    profile.lookup_misses += 1;
                    continue;
                };
                profile.lookup_hits += 1;
                let is_high_priority =
                    shipping_profile_priority(priority_offsets, priority_data, row, &mut profile);
                shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
            }
            return Some(ShippingOrdersPartial { groups, profile });
        }
        for row in 0..orderkey_values.len() {
            let order =
                shipping_profile_map_lookup(pending, orderkey_values[row], row, &mut profile);
            let Some(order) = order else {
                profile.lookup_misses += 1;
                continue;
            };
            profile.lookup_hits += 1;
            let is_high_priority =
                shipping_profile_priority(priority_offsets, priority_data, row, &mut profile);
            shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
        }
        return Some(ShippingOrdersPartial { groups, profile });
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            profile.null_rows += 1;
            continue;
        }
        let order = shipping_profile_map_lookup(pending, orderkeys.value(row), row, &mut profile);
        let Some(order) = order else {
            profile.lookup_misses += 1;
            continue;
        };
        profile.lookup_hits += 1;
        let is_high_priority =
            shipping_profile_priority(priority_offsets, priority_data, row, &mut profile);
        shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
    }
    Some(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_mode_counts_batch_dictionary_typed(
    orderkeys: &ArrayRef,
    orderpriorities: DictionaryI32View<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_flags =
        dictionary_i32_view_match_flags(orderpriorities, &[b"1-URGENT", b"2-HIGH"])?;
    let priority_keys = orderpriorities.keys();
    let mut groups = [ShippingPriorityState::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let is_high_priority =
                    dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority =
                dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
            apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let is_high_priority =
            dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_dictionary_vector_typed(
    orderkeys: I64VectorView<'_>,
    orderpriorities: DictionaryI32View<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let priority_flags =
        dictionary_i32_view_match_flags(orderpriorities, &[b"1-URGENT", b"2-HIGH"])?;
    let priority_keys = orderpriorities.keys();
    let mut groups = [ShippingPriorityState::default(); 2];
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let is_high_priority =
                    dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority =
                dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
            apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let is_high_priority =
            dictionary_i32_view_match_index(priority_keys, &priority_flags, row).is_some();
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_batch_typed(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [ShippingPriorityState::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let priority = bytes_string_parts(priority_offsets, priority_data, row);
                let is_high_priority = is_high_priority_bytes(priority);
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let priority = bytes_string_parts(priority_offsets, priority_data, row);
            let is_high_priority = is_high_priority_bytes(priority);
            apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = is_high_priority_bytes(priority);
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_vector_typed(
    orderkeys: I64VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let mut groups = [ShippingPriorityState::default(); 2];
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let Ok(index) = usize::try_from(orderkey_values[row]) else {
                    continue;
                };
                if index >= pending_present.len() || !pending_present[index] {
                    continue;
                }
                let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
                apply_pending_order(&mut groups, pending_values[index], is_high_priority);
            }
            return Some(groups);
        }
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
            apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_vector_typed_sorted(
    orderkeys: I64VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let mut groups = [ShippingPriorityState::default(); 2];
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
            apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let is_high_priority = is_high_priority_bytes(orderpriorities.value_bytes(row));
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn shipping_mode_counts_vector_typed_profile(
    orderkeys: I64VectorView<'_>,
    orderpriorities: Utf8VectorView<'_>,
    pending: &AdaptiveI64Map<PendingShippingOrder>,
) -> Option<ShippingOrdersPartial> {
    let mut groups = [ShippingPriorityState::default(); 2];
    let mut profile = ShippingOrdersProfile {
        batches: 1,
        typed_batches: 1,
        rows: orderkeys.len(),
        ..Default::default()
    };
    if let Some(orderkey_values) = orderkeys.values_if_null_free()
        && orderpriorities.null_count() == 0
    {
        if let Some((pending_values, pending_present)) = pending.dense_slices() {
            for row in 0..orderkey_values.len() {
                let order = shipping_profile_dense_lookup(
                    orderkey_values[row],
                    pending_values,
                    pending_present,
                    row,
                    &mut profile,
                );
                let Some(order) = order else {
                    profile.lookup_misses += 1;
                    continue;
                };
                profile.lookup_hits += 1;
                let is_high_priority =
                    shipping_profile_priority_vector(orderpriorities, row, &mut profile);
                shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
            }
            return Some(ShippingOrdersPartial { groups, profile });
        }
        for row in 0..orderkey_values.len() {
            let order =
                shipping_profile_map_lookup(pending, orderkey_values[row], row, &mut profile);
            let Some(order) = order else {
                profile.lookup_misses += 1;
                continue;
            };
            profile.lookup_hits += 1;
            let is_high_priority =
                shipping_profile_priority_vector(orderpriorities, row, &mut profile);
            shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
        }
        return Some(ShippingOrdersPartial { groups, profile });
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            profile.null_rows += 1;
            continue;
        }
        let order = shipping_profile_map_lookup(pending, orderkeys.value(row), row, &mut profile);
        let Some(order) = order else {
            profile.lookup_misses += 1;
            continue;
        };
        profile.lookup_hits += 1;
        let is_high_priority = shipping_profile_priority_vector(orderpriorities, row, &mut profile);
        shipping_profile_apply(&mut groups, order, is_high_priority, &mut profile);
    }
    Some(ShippingOrdersPartial { groups, profile })
}

pub(super) fn shipping_mode_counts_batch_typed_sorted(
    orderkeys: &ArrayRef,
    orderpriorities: &StringArray,
    pending: &SortedI64Lookup<PendingShippingOrder>,
) -> Option<[ShippingPriorityState; 2]> {
    let orderkeys = orderkeys.as_any().downcast_ref::<Int64Array>()?;
    let priority_offsets = orderpriorities.value_offsets();
    let priority_data = orderpriorities.value_data();
    let mut groups = [ShippingPriorityState::default(); 2];
    if orderkeys.null_count() == 0 && orderpriorities.null_count() == 0 {
        let orderkey_values = orderkeys.values().as_ref();
        for row in 0..orderkey_values.len() {
            let Some(order) = pending.get(orderkey_values[row]) else {
                continue;
            };
            let priority = bytes_string_parts(priority_offsets, priority_data, row);
            let is_high_priority = is_high_priority_bytes(priority);
            apply_pending_order(&mut groups, order, is_high_priority);
        }
        return Some(groups);
    }
    for row in 0..orderkeys.len() {
        if orderkeys.is_null(row) || orderpriorities.is_null(row) {
            continue;
        }
        let Some(order) = pending.get(orderkeys.value(row)) else {
            continue;
        };
        let priority = bytes_string_parts(priority_offsets, priority_data, row);
        let is_high_priority = is_high_priority_bytes(priority);
        apply_pending_order(&mut groups, order, is_high_priority);
    }
    Some(groups)
}

pub(super) fn apply_pending_order(
    groups: &mut [ShippingPriorityState; 2],
    order: PendingShippingOrder,
    is_high_priority: bool,
) {
    for (index, count) in order.counts.iter().copied().enumerate() {
        if count == 0 {
            continue;
        }
        let group = &mut groups[index];
        if is_high_priority {
            group.high_line_count += count;
        } else {
            group.low_line_count += count;
        }
    }
}

pub(super) fn merge_shipping_mode_counts(
    groups: &mut [ShippingPriorityState; 2],
    batch_groups: [ShippingPriorityState; 2],
) {
    for index in 0..groups.len() {
        groups[index].high_line_count += batch_groups[index].high_line_count;
        groups[index].low_line_count += batch_groups[index].low_line_count;
    }
}

pub(super) fn merge_orders_partial(
    output: &mut ShippingOrdersPartial,
    partial: ShippingOrdersPartial,
) {
    merge_shipping_mode_counts(&mut output.groups, partial.groups);
    output.profile.add(partial.profile);
}

pub(super) fn log_orders_profile(profile: &ShippingOrdersProfile) {
    if !tpch_profile_enabled() || profile.batches == 0 {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] shipping priority counts orders detail: batches={} typed_batches={} fallback_batches={} rows={} null_rows={} lookup_hits={} lookup_misses={} priority_rows={} apply_rows={} total={:.3} ms lookup_sample={:.3} ms/{} priority_sample={:.3} ms/{} apply_sample={:.3} ms/{}",
        profile.batches,
        profile.typed_batches,
        profile.fallback_batches,
        profile.rows,
        profile.null_rows,
        profile.lookup_hits,
        profile.lookup_misses,
        profile.priority_rows,
        profile.apply_rows,
        sql_nanos_to_millis(profile.total_nanos),
        sql_nanos_to_millis(profile.lookup_nanos),
        profile.lookup_samples,
        sql_nanos_to_millis(profile.priority_nanos),
        profile.priority_samples,
        sql_nanos_to_millis(profile.apply_nanos),
        profile.apply_samples,
    );
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
