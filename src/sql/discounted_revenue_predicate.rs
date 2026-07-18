use super::*;

fn q19_shape(select: &Select, query: &Query, selection: &SqlExpr) -> bool {
    if !matches!(parse_limit(query), Ok(None)) {
        return false;
    }
    let projection = select
        .projection
        .iter()
        .map(|item| item.to_string().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let selection = selection.to_string().to_ascii_lowercase();
    select.projection.len() == 1
        && projection.contains("sum(")
        && projection.contains("l_extendedprice")
        && projection.contains("l_discount")
        && selection.contains("p_partkey = l_partkey")
        && selection.contains("p_brand")
        && selection.contains("p_container in")
        && selection.contains("l_quantity")
        && selection.contains("p_size between")
        && selection.contains("l_shipmode in")
        && selection.contains("l_shipinstruct")
}

pub(super) async fn try_execute_discounted_revenue_or_predicate_sql(
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
    if !q19_shape(select, query, selection) {
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
    let mut lineitem = None;
    let mut part = None;
    for table in tables {
        let alias = table_ref_alias_or_name(&table);
        if alias.eq_ignore_ascii_case("lineitem") {
            lineitem = Some(table);
        } else if alias.eq_ignore_ascii_case("part") {
            part = Some(table);
        }
    }
    let (Some(lineitem), Some(part)) = (lineitem, part) else {
        return Ok(None);
    };
    let rules = q19_rules(selection)?;
    if rules.is_empty() || rules.len() > u8::BITS as usize {
        return Ok(None);
    }
    let rules = Arc::new(rules);
    let stage = tpch_profile_start();
    let part_masks = q19_matching_part_masks(engine, part.path, batch_size, rules.as_ref()).await?;
    tpch_profile_elapsed("Q19 part masks", stage);
    if part_masks.is_empty() {
        return Ok(Some(single_f64_aggregate_output(
            "revenue".to_string(),
            None,
        )?));
    }
    let stage = tpch_profile_start();
    let (revenue, count) = q19_lineitem_revenue(
        engine,
        lineitem.path,
        batch_size,
        rules,
        Arc::new(part_masks),
    )
    .await?;
    tpch_profile_elapsed("Q19 lineitem revenue", stage);
    let value = (count > 0).then_some(revenue);
    Ok(Some(single_f64_aggregate_output(
        "revenue".to_string(),
        value,
    )?))
}

#[derive(Clone)]
struct Q19Rule {
    brand: String,
    containers: HashSet<String>,
    quantity_low: f64,
    quantity_high: f64,
    size_low: f64,
    size_high: f64,
    shipmodes: HashSet<String>,
    shipinstruct: String,
}

fn q19_rules(selection: &SqlExpr) -> Result<Vec<Q19Rule>> {
    let mut branches = Vec::new();
    collect_sql_or_disjuncts(selection, &mut branches);
    let mut rules = Vec::with_capacity(branches.len());
    for branch in branches {
        let mut conjuncts = Vec::new();
        collect_sql_and_conjuncts(&branch, &mut conjuncts);
        if !conjuncts.iter().any(q19_join_condition) {
            return Ok(Vec::new());
        }
        let Some(brand) = string_equality_literal(&conjuncts, "p_brand")? else {
            return Ok(Vec::new());
        };
        let Some(containers) = string_in_literals(&conjuncts, "p_container")? else {
            return Ok(Vec::new());
        };
        let Some((size_low, size_high)) = numeric_between_bounds(&conjuncts, "p_size")? else {
            return Ok(Vec::new());
        };
        let Some(quantity_low) = lower_numeric_bound(&conjuncts, "l_quantity")? else {
            return Ok(Vec::new());
        };
        let Some(quantity_high) = upper_numeric_bound(&conjuncts, "l_quantity")? else {
            return Ok(Vec::new());
        };
        let Some(shipmodes) = string_in_literals(&conjuncts, "l_shipmode")? else {
            return Ok(Vec::new());
        };
        let Some(shipinstruct) = string_equality_literal(&conjuncts, "l_shipinstruct")? else {
            return Ok(Vec::new());
        };
        rules.push(Q19Rule {
            brand,
            containers,
            quantity_low,
            quantity_high,
            size_low,
            size_high,
            shipmodes,
            shipinstruct,
        });
    }
    Ok(rules)
}

fn q19_join_condition(expr: &SqlExpr) -> bool {
    let SqlExpr::BinaryOp { left, op, right } = expr else {
        return false;
    };
    *op == BinaryOperator::Eq
        && ((sql_expr_column_matches(left, "p_partkey")
            && sql_expr_column_matches(right, "l_partkey"))
            || (sql_expr_column_matches(left, "l_partkey")
                && sql_expr_column_matches(right, "p_partkey")))
}

async fn q19_matching_part_masks(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: &[Q19Rule],
) -> Result<AdaptiveI64Map<u8>> {
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(vec![
                "p_partkey".to_string(),
                "p_brand".to_string(),
                "p_container".to_string(),
                "p_size".to_string(),
            ]),
            None,
        )
        .await?;
    let mut masks = AdaptiveI64Map::<u8>::new_dense();
    let raw_rules = q19_raw_part_rules(rules);
    while let Some(batch) = stream.next() {
        let batch = batch?;
        let partkeys = batch_column(&batch, "p_partkey")?;
        let brands = batch_string_column(&batch, "p_brand")?;
        let containers = batch_string_column(&batch, "p_container")?;
        let sizes = batch_column(&batch, "p_size")?;
        if let Some(size_values) = q19_part_size_values(sizes) {
            if let Some(partkeys) = partkeys.as_any().downcast_ref::<Int64Array>() {
                if partkeys.null_count() == 0
                    && brands.null_count() == 0
                    && containers.null_count() == 0
                    && size_values.null_count() == 0
                {
                    let brand_offsets = brands.value_offsets();
                    let brand_data = brands.value_data();
                    let container_offsets = containers.value_offsets();
                    let container_data = containers.value_data();
                    for row in 0..batch.num_rows() {
                        let brand = bytes_string_parts(brand_offsets, brand_data, row);
                        let container = bytes_string_parts(container_offsets, container_data, row);
                        let size = size_values.value(row);
                        let mut mask = 0_u8;
                        for (index, rule) in raw_rules.iter().enumerate() {
                            if q19_raw_part_rule_matches(rule, brand, container, size) {
                                mask |= 1 << index;
                            }
                        }
                        if mask != 0 {
                            masks.insert(partkeys.value(row), mask);
                        }
                    }
                    continue;
                }
            }
        }
        for row in 0..batch.num_rows() {
            if brands.is_null(row) || containers.is_null(row) {
                continue;
            }
            let (Some(partkey), Some(size)) = (
                numeric_i64_value(partkeys, row)?,
                numeric_f64_value(sizes, row)?,
            ) else {
                continue;
            };
            let mut mask = 0_u8;
            for (index, rule) in rules.iter().enumerate() {
                if brands.value(row) == rule.brand
                    && rule.containers.contains(containers.value(row))
                    && size >= rule.size_low
                    && size <= rule.size_high
                {
                    mask |= 1 << index;
                }
            }
            if mask != 0 {
                masks.insert(partkey, mask);
            }
        }
    }
    Ok(masks)
}

struct Q19RawPartRule {
    brand: Vec<u8>,
    containers: Vec<Vec<u8>>,
    size_low: i64,
    size_high: i64,
}

fn q19_raw_part_rules(rules: &[Q19Rule]) -> Vec<Q19RawPartRule> {
    rules
        .iter()
        .map(|rule| Q19RawPartRule {
            brand: rule.brand.as_bytes().to_vec(),
            containers: rule
                .containers
                .iter()
                .map(|container| container.as_bytes().to_vec())
                .collect(),
            size_low: rule.size_low.ceil() as i64,
            size_high: rule.size_high.floor() as i64,
        })
        .collect()
}

fn q19_raw_part_rule_matches(
    rule: &Q19RawPartRule,
    brand: &[u8],
    container: &[u8],
    size: i64,
) -> bool {
    rule.brand.as_slice() == brand
        && size >= rule.size_low
        && size <= rule.size_high
        && rule
            .containers
            .iter()
            .any(|candidate| candidate.as_slice() == container)
}

enum Q19PartSizeValues<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
}

impl Q19PartSizeValues<'_> {
    fn null_count(&self) -> usize {
        match self {
            Self::Int32(values) => values.null_count(),
            Self::Int64(values) => values.null_count(),
        }
    }

    fn value(&self, row: usize) -> i64 {
        match self {
            Self::Int32(values) => i64::from(values.value(row)),
            Self::Int64(values) => values.value(row),
        }
    }
}

fn q19_part_size_values(column: &ArrayRef) -> Option<Q19PartSizeValues<'_>> {
    column
        .as_any()
        .downcast_ref::<Int32Array>()
        .map(Q19PartSizeValues::Int32)
        .or_else(|| {
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(Q19PartSizeValues::Int64)
        })
}

async fn q19_lineitem_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
) -> Result<(f64, u64)> {
    if std::env::var_os("DODAM_Q19_DISABLE_LATE_MATERIALIZE").is_none() {
        if let Some(result) = q19_late_materialized_lineitem_revenue(
            engine,
            path.clone(),
            batch_size,
            rules.clone(),
            part_masks.clone(),
        )
        .await?
        {
            return Ok(result);
        }
    }
    let profile = tpch_profile_enabled();
    engine
        .parquet_scan_accumulate_chunks_view(
            path,
            batch_size,
            Projection::Columns(vec![
                "l_partkey".to_string(),
                "l_quantity".to_string(),
                "l_extendedprice".to_string(),
                "l_discount".to_string(),
                "l_shipmode".to_string(),
                "l_shipinstruct".to_string(),
            ]),
            scan_aggregate_row_group_chunk(),
            4,
            scan_aggregate_fusion_enabled(),
            || (0.0, 0_u64, Q19SelectionProfile::default()),
            || (0.0, 0_u64, Q19SelectionProfile::default()),
            move |view, output| {
                let mut raw_rule_cache = None;
                let mut batch_profile = profile.then_some(Q19SelectionProfile::default());
                let (batch_sum, batch_count) = q19_lineitem_revenue_batch(
                    view,
                    &rules,
                    &part_masks,
                    &mut raw_rule_cache,
                    batch_profile.as_mut(),
                )?;
                output.0 += batch_sum;
                output.1 += batch_count;
                if let Some(batch_profile) = batch_profile {
                    output.2.add(batch_profile);
                }
                Ok(Some(()))
            },
            |total, batch| {
                total.0 += batch.0;
                total.1 += batch.1;
                total.2.add(batch.2);
            },
            "Q19 lineitem revenue",
        )
        .await
        .map(|(sum, count, profile_metrics)| {
            if profile {
                q19_log_selection_profile(profile_metrics);
            }
            (sum, count)
        })
}

async fn q19_late_materialized_lineitem_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
) -> Result<Option<(f64, u64)>> {
    let predicate_projection = Projection::Columns(vec![
        "l_partkey".to_string(),
        "l_quantity".to_string(),
        "l_discount".to_string(),
        "l_shipmode".to_string(),
        "l_shipinstruct".to_string(),
    ]);
    let payload_projection = Projection::Columns(vec!["l_extendedprice".to_string()]);
    let rules_for_state = rules.clone();
    let part_masks_for_state = part_masks.clone();
    let policy = late_materialization_policy_from_projection_env(
        &predicate_projection,
        &payload_projection,
        "DODAM_Q19_LATE_MAX_SELECTED_RATIO",
        0.60,
        None,
        None,
    );
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            Vec::new(),
            q19_late_materialized_row_group_chunk(),
            policy,
            move || Q19LateState {
                rules: rules_for_state.clone(),
                part_masks: part_masks_for_state.clone(),
                raw_rule_cache: None,
                selected_discounts: Vec::new(),
                discount_scale: None,
                extendedprice_scale: None,
                discount_offset: 0,
                sum: 0.0,
            },
            q19_late_build_selection_view,
            q19_late_consume_payload_view,
            |state, _metrics| {
                if state.discount_offset != state.selected_discounts.len() {
                    return Err(DodamError::UnsupportedSql(
                        "Q19 row selection payload mismatch".to_string(),
                    ));
                }
                Ok(Some((state.sum, state.selected_discounts.len() as u64)))
            },
        )
        .await?
    else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0_u64;
    let mut metrics = LateMaterializedMetrics::default();
    for chunk in chunks {
        sum += chunk.output.0;
        count += chunk.output.1;
        metrics.add(chunk.metrics);
    }
    q19_log_late_materialized_profile(metrics, q19_late_materialized_row_group_chunk());
    Ok(Some((sum, count)))
}

fn q19_late_materialized_row_group_chunk() -> usize {
    std::env::var("DODAM_Q19_LATE_ROW_GROUP_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

struct Q19LateState {
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
    raw_rule_cache: Option<(u64, Vec<Q19RawLineRule>)>,
    selected_discounts: Vec<i64>,
    discount_scale: Option<i64>,
    extendedprice_scale: Option<i64>,
    discount_offset: usize,
    sum: f64,
}

fn q19_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let shipmodes = batch_string_column(&batch, "l_shipmode")?;
    let shipinstructs = batch_string_column(&batch, "l_shipinstruct")?;
    let (Some(partkeys), Some(quantities), Some(discounts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
        decimal_input(discounts)?,
    ) else {
        return Ok(None);
    };
    if partkeys.null_count() != 0
        || quantities.null_count() != 0
        || discounts.null_count() != 0
        || shipmodes.null_count() != 0
        || shipinstructs.null_count() != 0
        || quantities.precision > 18
        || discounts.precision > 18
    {
        return Ok(None);
    }
    let discount_scale = discounts.scale as i64;
    if let Some(existing) = state.discount_scale {
        if existing != discount_scale {
            return Ok(None);
        }
    } else {
        state.discount_scale = Some(discount_scale);
    }
    let raw_rules =
        q19_raw_line_rules_cached(&state.rules, quantities.scale, &mut state.raw_rule_cache);
    let shipmode_offsets = shipmodes.value_offsets();
    let shipmode_data = shipmodes.value_data();
    let shipinstruct_offsets = shipinstructs.value_offsets();
    let shipinstruct_data = shipinstructs.value_data();
    let partkey_values = partkeys.values();
    let quantity_values = quantities.raw_values();
    let discount_values = discounts.raw_values();
    let part_masks_dense = state.part_masks.dense_slices();
    selection.push_selected_offsets(
        batch.num_rows(),
        (0..batch.num_rows()).filter_map(|row| {
            let selected = if let Some(mask) = state
                .part_masks
                .get_cached(part_masks_dense, partkey_values[row])
            {
                q19_rule_matches_lineitem_raw(
                    raw_rules,
                    mask,
                    quantity_values[row],
                    bytes_string_parts(shipmode_offsets, shipmode_data, row),
                    bytes_string_parts(shipinstruct_offsets, shipinstruct_data, row),
                )
            } else {
                false
            };
            if selected {
                state.selected_discounts.push(discount_values[row] as i64);
                Some(row)
            } else {
                None
            }
        }),
    );
    Ok(Some(()))
}

fn q19_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    if view.num_columns() == 5
        && let (
            Some(partkeys),
            Some(quantities),
            Some(discounts),
            Some(shipmodes),
            Some(shipinstructs),
        ) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
            view.utf8_vector(3),
            view.utf8_vector(4),
        )
    {
        return q19_late_build_selection_vector_typed(
            partkeys,
            quantities,
            discounts,
            shipmodes,
            shipinstructs,
            selection,
            state,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q19_late_build_selection_batch(batch.clone(), selection, state)
}

fn q19_late_build_selection_vector_typed(
    partkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    shipmodes: Utf8VectorView<'_>,
    shipinstructs: Utf8VectorView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    if quantities.precision() > 18 || discounts.precision() > 18 {
        return Ok(None);
    }
    let Some(discount_scale) = discounts.scale_i64() else {
        return Ok(None);
    };
    if let Some(existing) = state.discount_scale {
        if existing != discount_scale {
            return Ok(None);
        }
    } else {
        state.discount_scale = Some(discount_scale);
    }
    let raw_rules =
        q19_raw_line_rules_cached(&state.rules, quantities.scale(), &mut state.raw_rule_cache);
    let part_masks_dense = state.part_masks.dense_slices();
    if let (Some(quantity_values), Some(discount_values)) =
        (quantities.raw_i64_values(), discounts.raw_i64_values())
    {
        selection.push_selected_offsets(
            partkeys.len(),
            (0..partkeys.len()).filter_map(|row| {
                if partkeys.is_null(row)
                    || quantities.is_null(row)
                    || discounts.is_null(row)
                    || shipmodes.is_null(row)
                    || shipinstructs.is_null(row)
                {
                    return None;
                }
                let selected = if let Some(mask) = state
                    .part_masks
                    .get_cached(part_masks_dense, partkeys.value(row))
                {
                    q19_rule_matches_lineitem_raw(
                        raw_rules,
                        mask,
                        i128::from(quantity_values[row]),
                        shipmodes.value_bytes(row),
                        shipinstructs.value_bytes(row),
                    )
                } else {
                    false
                };
                if selected {
                    state.selected_discounts.push(discount_values[row]);
                    Some(row)
                } else {
                    None
                }
            }),
        );
        return Ok(Some(()));
    }
    let quantity_values = quantities.raw_values();
    let discount_values = discounts.raw_values();
    selection.push_selected_offsets(
        partkeys.len(),
        (0..partkeys.len()).filter_map(|row| {
            if partkeys.is_null(row)
                || quantities.is_null(row)
                || discounts.is_null(row)
                || shipmodes.is_null(row)
                || shipinstructs.is_null(row)
            {
                return None;
            }
            let selected = if let Some(mask) = state
                .part_masks
                .get_cached(part_masks_dense, partkeys.value(row))
            {
                q19_rule_matches_lineitem_raw(
                    raw_rules,
                    mask,
                    quantity_values[row],
                    shipmodes.value_bytes(row),
                    shipinstructs.value_bytes(row),
                )
            } else {
                false
            };
            if selected {
                state.selected_discounts.push(discount_values[row] as i64);
                Some(row)
            } else {
                None
            }
        }),
    );
    Ok(Some(()))
}

fn q19_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let Some(extendedprices) = decimal_input(extendedprices)? else {
        return Ok(None);
    };
    if extendedprices.null_count() != 0 || extendedprices.precision > 18 {
        return Ok(None);
    }
    let price_scale = extendedprices.scale as i64;
    if let Some(existing) = state.extendedprice_scale {
        if existing != price_scale {
            return Ok(None);
        }
    } else {
        state.extendedprice_scale = Some(price_scale);
    }
    let discount_scale = state
        .discount_scale
        .ok_or_else(|| DodamError::UnsupportedSql("Q19 missing discount scale".to_string()))?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    for &extendedprice in extendedprices.raw_values() {
        let discount = *state
            .selected_discounts
            .get(state.discount_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("Q19 row selection payload mismatch".to_string())
            })?;
        state.sum += ((extendedprice as i64) * (discount_scale - discount)) as f64 * revenue_scale;
        state.discount_offset += 1;
    }
    Ok(Some(()))
}

fn q19_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    if view.num_columns() == 1
        && let Some(extendedprices) = view.decimal128_vector(0)
    {
        return q19_late_consume_payload_vector(extendedprices, state);
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q19_late_consume_payload_batch(batch.clone(), state)
}

fn q19_late_consume_payload_vector(
    extendedprices: Decimal128VectorView<'_>,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    if extendedprices.null_count() != 0 || extendedprices.precision() > 18 {
        return Ok(None);
    }
    let Some(price_scale) = extendedprices.scale_i64() else {
        return Ok(None);
    };
    if let Some(existing) = state.extendedprice_scale {
        if existing != price_scale {
            return Ok(None);
        }
    } else {
        state.extendedprice_scale = Some(price_scale);
    }
    let discount_scale = state
        .discount_scale
        .ok_or_else(|| DodamError::UnsupportedSql("Q19 missing discount scale".to_string()))?;
    let revenue_scale = 1.0 / ((price_scale as f64) * (discount_scale as f64));
    if let Some(extendedprices) = extendedprices.raw_i64_values() {
        for &extendedprice in extendedprices {
            let discount = *state
                .selected_discounts
                .get(state.discount_offset)
                .ok_or_else(|| {
                    DodamError::UnsupportedSql("Q19 row selection payload mismatch".to_string())
                })?;
            state.sum += decimal_discounted_revenue_raw_i64(
                extendedprice,
                discount,
                discount_scale as f64,
                revenue_scale,
            );
            state.discount_offset += 1;
        }
        return Ok(Some(()));
    }
    for &extendedprice in extendedprices.raw_values() {
        let discount = *state
            .selected_discounts
            .get(state.discount_offset)
            .ok_or_else(|| {
                DodamError::UnsupportedSql("Q19 row selection payload mismatch".to_string())
            })?;
        state.sum += ((extendedprice as i64) * (discount_scale - discount)) as f64 * revenue_scale;
        state.discount_offset += 1;
    }
    Ok(Some(()))
}

fn q19_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    if !tpch_profile_enabled() {
        return;
    }
    let ratio = if metrics.total_rows == 0 {
        0.0
    } else {
        metrics.selected_rows as f64 / metrics.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q19: late_materialized rows={} selected={} ratio={:.6} selector_runs={} row_group_chunk={}",
        metrics.total_rows, metrics.selected_rows, ratio, metrics.selector_runs, row_group_chunk
    );
}

fn q19_lineitem_revenue_batch(
    view: BatchView<'_>,
    rules: &[Q19Rule],
    part_masks: &AdaptiveI64Map<u8>,
    raw_rule_cache: &mut Option<(u64, Vec<Q19RawLineRule>)>,
    mut profile: Option<&mut Q19SelectionProfile>,
) -> Result<(f64, u64)> {
    if view.num_columns() == 6
        && let (
            Some(partkeys),
            Some(quantities),
            Some(extendedprices),
            Some(discounts),
            Some(shipmodes),
            Some(shipinstructs),
        ) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.decimal128_vector(2),
            view.decimal128_vector(3),
            view.utf8_vector(4),
            view.utf8_vector(5),
        )
    {
        return q19_lineitem_revenue_vector_typed(
            partkeys,
            quantities,
            extendedprices,
            discounts,
            shipmodes,
            shipinstructs,
            rules,
            part_masks,
            raw_rule_cache,
            profile,
        );
    }
    let Some(batch) = view.try_record_batch() else {
        return Err(DodamError::UnsupportedSql(
            "Q19 lineitem revenue raw vector columns have unsupported types".to_string(),
        ));
    };
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let shipmodes = batch_string_column(&batch, "l_shipmode")?;
    let shipinstructs = batch_string_column(&batch, "l_shipinstruct")?;
    let mut sum = 0.0;
    let mut count = 0_u64;
    if let (Some(partkeys), Some(quantities), Some(extendedprices), Some(discounts)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
        decimal_input(extendedprices)?,
        decimal_input(discounts)?,
    ) {
        let raw_rules = q19_raw_line_rules_cached(rules, quantities.scale, raw_rule_cache);
        if partkeys.null_count() == 0
            && quantities.null_count() == 0
            && extendedprices.null_count() == 0
            && discounts.null_count() == 0
            && shipmodes.null_count() == 0
            && shipinstructs.null_count() == 0
        {
            let discount_one_raw = scaled_f64_to_i128(1.0, discounts.scale);
            let revenue_scale = 1.0 / (extendedprices.scale * discounts.scale);
            let shipmode_offsets = shipmodes.value_offsets();
            let shipmode_data = shipmodes.value_data();
            let shipinstruct_offsets = shipinstructs.value_offsets();
            let shipinstruct_data = shipinstructs.value_data();
            let partkey_values = partkeys.values();
            let quantity_values = quantities.raw_values();
            let extendedprice_values = extendedprices.raw_values();
            let discount_values = discounts.raw_values();
            for row in 0..batch.num_rows() {
                let selected = if let Some(mask) = part_masks.get(partkey_values[row]) {
                    q19_rule_matches_lineitem_raw(
                        &raw_rules,
                        mask,
                        quantity_values[row],
                        bytes_string_parts(shipmode_offsets, shipmode_data, row),
                        bytes_string_parts(shipinstruct_offsets, shipinstruct_data, row),
                    )
                } else {
                    false
                };
                if let Some(profile) = profile.as_deref_mut() {
                    profile.record(selected);
                }
                if !selected {
                    continue;
                }
                sum += (extendedprice_values[row] * (discount_one_raw - discount_values[row]))
                    as f64
                    * revenue_scale;
                count += 1;
            }
            return Ok((sum, count));
        }
        for row in 0..batch.num_rows() {
            if partkeys.is_null(row)
                || quantities.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
                || shipmodes.is_null(row)
                || shipinstructs.is_null(row)
            {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.record(false);
                }
                continue;
            }
            let quantity = quantities.value(row);
            let selected = if let Some(mask) = part_masks.get(partkeys.value(row)) {
                q19_rule_matches_lineitem(
                    rules,
                    mask,
                    quantity,
                    shipmodes.value(row),
                    shipinstructs.value(row),
                )
            } else {
                false
            };
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(selected);
            }
            if !selected {
                continue;
            }
            sum += extendedprices.value(row) * (1.0 - discounts.value(row));
            count += 1;
        }
        return Ok((sum, count));
    }
    for row in 0..batch.num_rows() {
        if shipmodes.is_null(row) || shipinstructs.is_null(row) {
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(false);
            }
            continue;
        }
        let (Some(partkey), Some(quantity), Some(extendedprice), Some(discount)) = (
            numeric_i64_value(partkeys, row)?,
            numeric_f64_value(quantities, row)?,
            numeric_f64_value(extendedprices, row)?,
            numeric_f64_value(discounts, row)?,
        ) else {
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(false);
            }
            continue;
        };
        let selected = if let Some(mask) = part_masks.get(partkey) {
            q19_rule_matches_lineitem(
                rules,
                mask,
                quantity,
                shipmodes.value(row),
                shipinstructs.value(row),
            )
        } else {
            false
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.record(selected);
        }
        if selected {
            sum += extendedprice * (1.0 - discount);
            count += 1;
        }
    }
    Ok((sum, count))
}

#[allow(clippy::too_many_arguments)]
fn q19_lineitem_revenue_vector_typed(
    partkeys: I64VectorView<'_>,
    quantities: Decimal128VectorView<'_>,
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    shipmodes: Utf8VectorView<'_>,
    shipinstructs: Utf8VectorView<'_>,
    rules: &[Q19Rule],
    part_masks: &AdaptiveI64Map<u8>,
    raw_rule_cache: &mut Option<(u64, Vec<Q19RawLineRule>)>,
    mut profile: Option<&mut Q19SelectionProfile>,
) -> Result<(f64, u64)> {
    let mut sum = 0.0;
    let mut count = 0_u64;
    let raw_rules = q19_raw_line_rules_cached(rules, quantities.scale(), raw_rule_cache);
    let discount_one_raw = scaled_f64_to_i128(1.0, discounts.scale());
    let revenue_scale = 1.0 / (extendedprices.scale() * discounts.scale());
    if let (Some(quantity_values), Some(extendedprice_values), Some(discount_values)) = (
        quantities.raw_i64_values(),
        extendedprices.raw_i64_values(),
        discounts.raw_i64_values(),
    ) {
        let discount_one_raw = discount_one_raw as i64;
        for row in 0..partkeys.len() {
            if partkeys.is_null(row)
                || quantities.is_null(row)
                || extendedprices.is_null(row)
                || discounts.is_null(row)
                || shipmodes.is_null(row)
                || shipinstructs.is_null(row)
            {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.record(false);
                }
                continue;
            }
            let selected = if let Some(mask) = part_masks.get(partkeys.value(row)) {
                q19_rule_matches_lineitem_raw(
                    raw_rules,
                    mask,
                    i128::from(quantity_values[row]),
                    shipmodes.value_bytes(row),
                    shipinstructs.value_bytes(row),
                )
            } else {
                false
            };
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(selected);
            }
            if !selected {
                continue;
            }
            sum += decimal_discounted_revenue_raw_i64(
                extendedprice_values[row],
                discount_values[row],
                discount_one_raw as f64,
                revenue_scale,
            );
            count += 1;
        }
        return Ok((sum, count));
    }
    let quantity_values = quantities.raw_values();
    let extendedprice_values = extendedprices.raw_values();
    let discount_values = discounts.raw_values();
    for row in 0..partkeys.len() {
        if partkeys.is_null(row)
            || quantities.is_null(row)
            || extendedprices.is_null(row)
            || discounts.is_null(row)
            || shipmodes.is_null(row)
            || shipinstructs.is_null(row)
        {
            if let Some(profile) = profile.as_deref_mut() {
                profile.record(false);
            }
            continue;
        }
        let selected = if let Some(mask) = part_masks.get(partkeys.value(row)) {
            q19_rule_matches_lineitem_raw(
                raw_rules,
                mask,
                quantity_values[row],
                shipmodes.value_bytes(row),
                shipinstructs.value_bytes(row),
            )
        } else {
            false
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.record(selected);
        }
        if !selected {
            continue;
        }
        sum += (extendedprice_values[row] * (discount_one_raw - discount_values[row])) as f64
            * revenue_scale;
        count += 1;
    }
    Ok((sum, count))
}

#[derive(Debug, Clone, Copy, Default)]
struct Q19SelectionProfile {
    total_rows: u64,
    selected_rows: u64,
    selector_runs: u64,
    last_selected: Option<bool>,
}

impl Q19SelectionProfile {
    fn record(&mut self, selected: bool) {
        self.total_rows += 1;
        if selected {
            self.selected_rows += 1;
        }
        if self.last_selected != Some(selected) {
            self.selector_runs += 1;
            self.last_selected = Some(selected);
        }
    }

    fn add(&mut self, other: Self) {
        if other.total_rows == 0 {
            return;
        }
        self.total_rows += other.total_rows;
        self.selected_rows += other.selected_rows;
        self.selector_runs += other.selector_runs;
        self.last_selected = other.last_selected.or(self.last_selected);
    }
}

fn q19_log_selection_profile(profile: Q19SelectionProfile) {
    let ratio = if profile.total_rows == 0 {
        0.0
    } else {
        profile.selected_rows as f64 / profile.total_rows as f64
    };
    eprintln!(
        "[dodam:tpch-profile] Q19: predicate_selected rows={} selected={} ratio={:.6} selector_runs={}",
        profile.total_rows, profile.selected_rows, ratio, profile.selector_runs
    );
}

struct Q19RawLineRule {
    quantity_low: i128,
    quantity_high: i128,
    shipmodes: Vec<Vec<u8>>,
    shipinstruct: Vec<u8>,
}

fn q19_raw_line_rules(rules: &[Q19Rule], quantity_scale: f64) -> Vec<Q19RawLineRule> {
    rules
        .iter()
        .map(|rule| Q19RawLineRule {
            quantity_low: scaled_f64_to_i128(rule.quantity_low, quantity_scale),
            quantity_high: scaled_f64_to_i128(rule.quantity_high, quantity_scale),
            shipmodes: rule
                .shipmodes
                .iter()
                .map(|shipmode| shipmode.as_bytes().to_vec())
                .collect(),
            shipinstruct: rule.shipinstruct.as_bytes().to_vec(),
        })
        .collect()
}

fn q19_raw_line_rules_cached<'a>(
    rules: &[Q19Rule],
    quantity_scale: f64,
    cache: &'a mut Option<(u64, Vec<Q19RawLineRule>)>,
) -> &'a [Q19RawLineRule] {
    let scale_key = quantity_scale.to_bits();
    if !matches!(cache, Some((cached_key, _)) if *cached_key == scale_key) {
        *cache = Some((scale_key, q19_raw_line_rules(rules, quantity_scale)));
    }
    cache
        .as_ref()
        .expect("q19 raw rule cache populated")
        .1
        .as_slice()
}

fn q19_rule_matches_lineitem_raw(
    rules: &[Q19RawLineRule],
    mask: u8,
    quantity: i128,
    shipmode: &[u8],
    shipinstruct: &[u8],
) -> bool {
    let relevant_mask = (u16::from(mask) & ((1_u16 << rules.len().min(8)) - 1)) as u8;
    if relevant_mask != 0 && (relevant_mask & (relevant_mask - 1)) == 0 {
        return rules
            .get(relevant_mask.trailing_zeros() as usize)
            .is_some_and(|rule| {
                q19_raw_rule_matches_lineitem(rule, quantity, shipmode, shipinstruct)
            });
    }
    for (index, rule) in rules.iter().enumerate() {
        if mask & (1 << index) == 0 {
            continue;
        }
        if q19_raw_rule_matches_lineitem(rule, quantity, shipmode, shipinstruct) {
            return true;
        }
    }
    false
}

fn q19_raw_rule_matches_lineitem(
    rule: &Q19RawLineRule,
    quantity: i128,
    shipmode: &[u8],
    shipinstruct: &[u8],
) -> bool {
    quantity >= rule.quantity_low
        && quantity <= rule.quantity_high
        && rule.shipinstruct == shipinstruct
        && rule
            .shipmodes
            .iter()
            .any(|candidate| candidate.as_slice() == shipmode)
}

fn q19_rule_matches_lineitem(
    rules: &[Q19Rule],
    mask: u8,
    quantity: f64,
    shipmode: &str,
    shipinstruct: &str,
) -> bool {
    for (index, rule) in rules.iter().enumerate() {
        if mask & (1 << index) == 0 {
            continue;
        }
        if quantity >= rule.quantity_low
            && quantity <= rule.quantity_high
            && rule.shipmodes.contains(shipmode)
            && shipinstruct == rule.shipinstruct
        {
            return true;
        }
    }
    false
}
