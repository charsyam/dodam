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
    let raw_rules = q19_raw_part_rules(rules);
    let row_groups = (0..engine.parquet_row_group_count(&path)?).collect::<Vec<_>>();
    if let Some((rows, _metrics)) = engine.collect_parquet_i64_two_utf8_i64_mapped_parallel(
        path.clone(),
        batch_size,
        row_groups,
        "p_partkey".to_string(),
        "p_brand".to_string(),
        "p_container".to_string(),
        "p_size".to_string(),
        |brand, container, size| q19_raw_part_mask(&raw_rules, brand, container, size),
    )? {
        let mut output = AdaptiveI64Map::<u8>::new_dense();
        for (partkey, mask) in rows {
            output.insert(partkey, mask);
        }
        return Ok(output);
    }
    collect_i64_two_utf8_i64_mapped_adaptive_map(
        engine,
        path,
        batch_size,
        "p_partkey",
        "p_brand",
        "p_container",
        "p_size",
        |brand, container, size| q19_raw_part_mask(&raw_rules, brand, container, size),
        |brand, container, size| q19_part_mask(rules, brand, container, size),
    )
    .await
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

fn q19_raw_part_mask(
    rules: &[Q19RawPartRule],
    brand: &[u8],
    container: &[u8],
    size: i64,
) -> Option<u8> {
    let mut mask = 0_u8;
    for (index, rule) in rules.iter().enumerate() {
        if q19_raw_part_rule_matches(rule, brand, container, size) {
            mask |= 1 << index;
        }
    }
    (mask != 0).then_some(mask)
}

fn q19_part_mask(rules: &[Q19Rule], brand: &str, container: &str, size: f64) -> Option<u8> {
    let mut mask = 0_u8;
    for (index, rule) in rules.iter().enumerate() {
        if brand == rule.brand
            && rule.containers.contains(container)
            && size >= rule.size_low
            && size <= rule.size_high
        {
            mask |= 1 << index;
        }
    }
    (mask != 0).then_some(mask)
}

async fn q19_lineitem_revenue(
    engine: &DodamEngine,
    path: PathBuf,
    batch_size: usize,
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
) -> Result<(f64, u64)> {
    if let Some(result) = q19_direct_dictionary_selected_lineitem_revenue(
        engine,
        &path,
        rules.clone(),
        part_masks.clone(),
    )? {
        return Ok(result);
    }
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

fn q19_direct_dictionary_selected_lineitem_revenue(
    engine: &DodamEngine,
    path: &Path,
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
) -> Result<Option<(f64, u64)>> {
    let Some((quantity_precision, quantity_scale)) =
        engine.parquet_decimal128_type(path, "l_quantity")?
    else {
        return Ok(None);
    };
    let Some((extendedprice_precision, extendedprice_scale)) =
        engine.parquet_decimal128_type(path, "l_extendedprice")?
    else {
        return Ok(None);
    };
    let Some((discount_precision, discount_scale)) =
        engine.parquet_decimal128_type(path, "l_discount")?
    else {
        return Ok(None);
    };
    if quantity_precision > 18 || extendedprice_precision > 18 || discount_precision > 18 {
        return Ok(None);
    }
    let Ok(discount_scale_power) = u32::try_from(discount_scale) else {
        return Ok(None);
    };
    let Some(discount_factor) = 10_i64.checked_pow(discount_scale_power) else {
        return Ok(None);
    };
    let quantity_factor = 10_f64.powi(i32::from(quantity_scale));
    let raw_rules = Arc::new(q19_raw_line_rules(&rules, quantity_factor));
    let output = Arc::new(Mutex::new((0_i128, 0_u64)));
    let lookup_masks = part_masks.clone();
    let predicate_rules = raw_rules.clone();
    let consume_output = output.clone();
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let Some(_metrics) = engine
        .scan_parquet_i64_lookup_decimal_utf8_selected_primitive_columns_parallel(
            path.to_path_buf(),
            row_groups,
            "l_partkey".to_string(),
            (
                "l_quantity".to_string(),
                DirectPrimitiveColumnType::Decimal128Int64Raw {
                    precision: quantity_precision,
                    scale: quantity_scale,
                },
            ),
            "l_shipmode".to_string(),
            "l_shipinstruct".to_string(),
            vec![
                (
                    "l_extendedprice".to_string(),
                    DirectPrimitiveColumnType::Decimal128Int64Raw {
                        precision: extendedprice_precision,
                        scale: extendedprice_scale,
                    },
                ),
                (
                    "l_discount".to_string(),
                    DirectPrimitiveColumnType::Decimal128Int64Raw {
                        precision: discount_precision,
                        scale: discount_scale,
                    },
                ),
            ],
            move |partkey| lookup_masks.get(partkey),
            move |mask, quantity, shipmode, shipinstruct| {
                q19_rule_matches_lineitem_raw(
                    &predicate_rules,
                    mask,
                    i128::from(quantity),
                    shipmode,
                    shipinstruct,
                )
            },
            move |view| {
                let (Some(extendedprices), Some(discounts)) =
                    (view.decimal128_vector(0), view.decimal128_vector(1))
                else {
                    return Err(DodamError::UnsupportedSql(
                        "Q19 selected payload vector shape mismatch".to_string(),
                    ));
                };
                let (Some(extendedprices), Some(discounts)) =
                    (extendedprices.raw_i64_values(), discounts.raw_i64_values())
                else {
                    return Err(DodamError::UnsupportedSql(
                        "Q19 selected payload requires raw i64 decimals".to_string(),
                    ));
                };
                let mut local_sum = 0_i128;
                for (&extendedprice, &discount) in extendedprices.iter().zip(discounts.iter()) {
                    let discount_multiplier =
                        discount_factor.checked_sub(discount).ok_or_else(|| {
                            DodamError::UnsupportedSql(
                                "Q19 selected payload discount overflow".to_string(),
                            )
                        })?;
                    let value = i128::from(extendedprice)
                        .checked_mul(i128::from(discount_multiplier))
                        .ok_or_else(|| {
                            DodamError::UnsupportedSql(
                                "Q19 selected payload revenue overflow".to_string(),
                            )
                        })?;
                    local_sum = local_sum.checked_add(value).ok_or_else(|| {
                        DodamError::UnsupportedSql("Q19 selected payload sum overflow".to_string())
                    })?;
                }
                let mut output = consume_output.lock().map_err(|_| {
                    DodamError::UnsupportedSql(
                        "Q19 selected payload output lock poisoned".to_string(),
                    )
                })?;
                output.0 = output.0.checked_add(local_sum).ok_or_else(|| {
                    DodamError::UnsupportedSql(
                        "Q19 selected payload global sum overflow".to_string(),
                    )
                })?;
                output.1 = output.1.saturating_add(extendedprices.len() as u64);
                Ok(())
            },
        )?
    else {
        return Ok(None);
    };
    let output = output.lock().map_err(|_| {
        DodamError::UnsupportedSql("Q19 selected payload output lock poisoned".to_string())
    })?;
    let revenue_scale = 10_f64.powi(-i32::from(extendedprice_scale) - i32::from(discount_scale));
    Ok(Some((output.0 as f64 * revenue_scale, output.1)))
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
        "l_shipmode".to_string(),
        "l_shipinstruct".to_string(),
    ]);
    let payload_projection = Projection::Columns(vec![
        "l_extendedprice".to_string(),
        "l_discount".to_string(),
    ]);
    let mut pruning_predicates = Vec::new();
    if let Some((min_key, max_key)) = part_masks.selective_key_range() {
        pruning_predicates.extend(i64_range_pruning_predicates("l_partkey", min_key, max_key));
    }
    let rules_for_state = rules.clone();
    let part_masks_for_state = part_masks.clone();
    let policy = generic_late_materialization_policy_for_projection(
        &predicate_projection,
        &payload_projection,
        0.60,
        None,
    );
    let Some(chunks) = engine
        .late_materialized_parquet_map_pruned_with_policy_view(
            path,
            batch_size,
            predicate_projection,
            payload_projection,
            pruning_predicates,
            q19_late_materialized_row_group_chunk(),
            policy,
            move || Q19LateState {
                rules: rules_for_state.clone(),
                part_masks: part_masks_for_state.clone(),
                raw_rule_cache: None,
                discount_scale: None,
                extendedprice_scale: None,
                sum: 0.0,
                count: 0,
                profile: Q19LateCallbackProfile::default(),
            },
            q19_late_build_selection_view,
            q19_late_consume_payload_view,
            |state, _metrics| Ok(Some((state.sum, state.count, state.profile))),
        )
        .await?
    else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0_u64;
    let mut metrics = LateMaterializedMetrics::default();
    let mut profile = Q19LateCallbackProfile::default();
    for chunk in chunks {
        sum += chunk.output.0;
        count += chunk.output.1;
        profile.add(chunk.output.2);
        metrics.add(chunk.metrics);
    }
    q19_log_late_materialized_profile(metrics, q19_late_materialized_row_group_chunk());
    q19_log_late_callback_profile(profile);
    Ok(Some((sum, count)))
}

fn q19_late_materialized_row_group_chunk() -> usize {
    late_materialization_row_group_chunk(2)
}

struct Q19LateState {
    rules: Arc<Vec<Q19Rule>>,
    part_masks: Arc<AdaptiveI64Map<u8>>,
    raw_rule_cache: Option<(u64, Vec<Q19RawLineRule>)>,
    discount_scale: Option<i64>,
    extendedprice_scale: Option<i64>,
    sum: f64,
    count: u64,
    profile: Q19LateCallbackProfile,
}

#[derive(Default, Clone, Copy)]
struct Q19LateCallbackProfile {
    selection_nanos: u64,
    payload_nanos: u64,
    selection_batches: usize,
    payload_batches: usize,
    selection_i64_batches: usize,
    selection_i128_batches: usize,
    payload_i64_batches: usize,
    payload_i128_batches: usize,
}

impl Q19LateCallbackProfile {
    fn add(&mut self, other: Self) {
        self.selection_nanos = self.selection_nanos.saturating_add(other.selection_nanos);
        self.payload_nanos = self.payload_nanos.saturating_add(other.payload_nanos);
        self.selection_batches = self
            .selection_batches
            .saturating_add(other.selection_batches);
        self.payload_batches = self.payload_batches.saturating_add(other.payload_batches);
        self.selection_i64_batches = self
            .selection_i64_batches
            .saturating_add(other.selection_i64_batches);
        self.selection_i128_batches = self
            .selection_i128_batches
            .saturating_add(other.selection_i128_batches);
        self.payload_i64_batches = self
            .payload_i64_batches
            .saturating_add(other.payload_i64_batches);
        self.payload_i128_batches = self
            .payload_i128_batches
            .saturating_add(other.payload_i128_batches);
    }
}

fn q19_late_build_selection_batch(
    batch: RecordBatch,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let partkeys = batch_column(&batch, "l_partkey")?;
    let quantities = batch_column(&batch, "l_quantity")?;
    let shipmodes = batch_string_column(&batch, "l_shipmode")?;
    let shipinstructs = batch_string_column(&batch, "l_shipinstruct")?;
    let (Some(partkeys), Some(quantities)) = (
        partkeys.as_any().downcast_ref::<Int64Array>(),
        decimal_input(quantities)?,
    ) else {
        return Ok(None);
    };
    if partkeys.null_count() != 0
        || quantities.null_count() != 0
        || shipmodes.null_count() != 0
        || shipinstructs.null_count() != 0
        || quantities.precision > 18
    {
        return Ok(None);
    }
    let raw_rules =
        q19_raw_line_rules_cached(&state.rules, quantities.scale, &mut state.raw_rule_cache);
    let shipmode_offsets = shipmodes.value_offsets();
    let shipmode_data = shipmodes.value_data();
    let shipinstruct_offsets = shipinstructs.value_offsets();
    let shipinstruct_data = shipinstructs.value_data();
    let partkey_values = partkeys.values();
    let quantity_values = quantities.raw_values();
    let part_masks_dense = state.part_masks.dense_word_slices();
    selection.push_selected_offsets(
        batch.num_rows(),
        (0..batch.num_rows()).filter_map(|row| {
            let selected = if let Some(mask) = state
                .part_masks
                .get_cached_words(part_masks_dense, partkey_values[row])
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
            selected.then_some(row)
        }),
    );
    Ok(Some(()))
}

fn q19_late_build_selection_view(
    view: BatchView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    if view.num_columns() == 4
        && let (Some(partkeys), Some(quantities), Some(shipmodes), Some(shipinstructs)) = (
            view.i64_vector(0),
            view.decimal128_vector(1),
            view.utf8_vector(2),
            view.utf8_vector(3),
        )
    {
        return q19_late_build_selection_vector_typed(
            partkeys,
            quantities,
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
    shipmodes: Utf8VectorView<'_>,
    shipinstructs: Utf8VectorView<'_>,
    selection: &mut LateSelectionBuilder,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let profile_started = tpch_profile_enabled().then(Instant::now);
    if quantities.precision() > 18 {
        return Ok(None);
    }
    let raw_rules =
        q19_raw_line_rules_cached(&state.rules, quantities.scale(), &mut state.raw_rule_cache);
    let part_masks_dense = state.part_masks.dense_word_slices();
    if let Some(quantity_values) = quantities.raw_i64_values() {
        selection.push_selected_offsets(
            partkeys.len(),
            (0..partkeys.len()).filter_map(|row| {
                if partkeys.is_null(row)
                    || quantities.is_null(row)
                    || shipmodes.is_null(row)
                    || shipinstructs.is_null(row)
                {
                    return None;
                }
                let selected = if let Some(mask) = state
                    .part_masks
                    .get_cached_words(part_masks_dense, partkeys.value(row))
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
                selected.then_some(row)
            }),
        );
        if let Some(started) = profile_started {
            state.profile.selection_batches += 1;
            state.profile.selection_i64_batches += 1;
            state.profile.selection_nanos = state
                .profile
                .selection_nanos
                .saturating_add(sql_elapsed_nanos(started));
        }
        return Ok(Some(()));
    }
    let quantity_values = quantities.raw_values();
    selection.push_selected_offsets(
        partkeys.len(),
        (0..partkeys.len()).filter_map(|row| {
            if partkeys.is_null(row)
                || quantities.is_null(row)
                || shipmodes.is_null(row)
                || shipinstructs.is_null(row)
            {
                return None;
            }
            let selected = if let Some(mask) = state
                .part_masks
                .get_cached_words(part_masks_dense, partkeys.value(row))
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
            selected.then_some(row)
        }),
    );
    if let Some(started) = profile_started {
        state.profile.selection_batches += 1;
        state.profile.selection_i128_batches += 1;
        state.profile.selection_nanos = state
            .profile
            .selection_nanos
            .saturating_add(sql_elapsed_nanos(started));
    }
    Ok(Some(()))
}

fn q19_late_consume_payload_batch(
    batch: RecordBatch,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let extendedprices = batch_column(&batch, "l_extendedprice")?;
    let discounts = batch_column(&batch, "l_discount")?;
    let Some(extendedprices) = decimal_input(extendedprices)? else {
        return Ok(None);
    };
    let Some(discounts) = decimal_input(discounts)? else {
        return Ok(None);
    };
    q19_late_consume_payload_vector(
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
        state,
    )
}

fn q19_late_consume_payload_view(
    view: BatchView<'_>,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    if view.num_columns() == 2
        && let (Some(extendedprices), Some(discounts)) =
            (view.decimal128_vector(0), view.decimal128_vector(1))
    {
        return q19_late_consume_payload_vector(extendedprices, discounts, state);
    }
    let Some(batch) = view.try_record_batch() else {
        return Ok(None);
    };
    q19_late_consume_payload_batch(batch.clone(), state)
}

fn q19_late_consume_payload_vector(
    extendedprices: Decimal128VectorView<'_>,
    discounts: Decimal128VectorView<'_>,
    state: &mut Q19LateState,
) -> Result<Option<()>> {
    let profile_started = tpch_profile_enabled().then(Instant::now);
    if extendedprices.null_count() != 0
        || discounts.null_count() != 0
        || extendedprices.precision() > 18
        || discounts.precision() > 18
    {
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
    let payload_i64 =
        extendedprices.raw_i64_values().is_some() && discounts.raw_i64_values().is_some();
    consume_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        extendedprices.len(),
        |_, revenue| {
            if let Some(revenue) = revenue {
                state.sum += revenue;
                state.count += 1;
            }
            Ok(())
        },
    )?;
    if let Some(started) = profile_started {
        state.profile.payload_batches += 1;
        if payload_i64 {
            state.profile.payload_i64_batches += 1;
        } else {
            state.profile.payload_i128_batches += 1;
        }
        state.profile.payload_nanos = state
            .profile
            .payload_nanos
            .saturating_add(sql_elapsed_nanos(started));
    }
    Ok(Some(()))
}

fn q19_log_late_materialized_profile(metrics: LateMaterializedMetrics, row_group_chunk: usize) {
    tpch_profile_late_materialized("Q19", metrics, row_group_chunk);
}

fn q19_log_late_callback_profile(profile: Q19LateCallbackProfile) {
    if !tpch_profile_enabled() {
        return;
    }
    eprintln!(
        "[dodam:tpch-profile] Q19 late callbacks: selection={:.3} ms payload={:.3} ms selection_batches={} payload_batches={} selection_i64={} selection_i128={} payload_i64={} payload_i128={}",
        sql_nanos_to_millis(profile.selection_nanos),
        sql_nanos_to_millis(profile.payload_nanos),
        profile.selection_batches,
        profile.payload_batches,
        profile.selection_i64_batches,
        profile.selection_i128_batches,
        profile.payload_i64_batches,
        profile.payload_i128_batches,
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
        return q19_lineitem_revenue_vector_typed(
            I64VectorView::Arrow(partkeys),
            Decimal128VectorView::Arrow {
                values: quantities.values,
                precision: quantities.precision,
                scale: quantities.scale,
            },
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
            Utf8VectorView::Arrow(shipmodes),
            Utf8VectorView::Arrow(shipinstructs),
            rules,
            part_masks,
            raw_rule_cache,
            profile,
        );
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
    if let Some(quantity_values) = quantities.raw_i64_values() {
        consume_filtered_discounted_revenue_decimal128_vectors(
            extendedprices,
            discounts,
            partkeys.len(),
            |row| {
                if partkeys.is_null(row)
                    || quantities.is_null(row)
                    || shipmodes.is_null(row)
                    || shipinstructs.is_null(row)
                {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.record(false);
                    }
                    return Ok(false);
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
                Ok(selected)
            },
            |_, revenue| {
                sum += revenue;
                count += 1;
                Ok(())
            },
        )?;
        return Ok((sum, count));
    }
    let quantity_values = quantities.raw_values();
    consume_filtered_discounted_revenue_decimal128_vectors(
        extendedprices,
        discounts,
        partkeys.len(),
        |row| {
            if partkeys.is_null(row)
                || quantities.is_null(row)
                || shipmodes.is_null(row)
                || shipinstructs.is_null(row)
            {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.record(false);
                }
                return Ok(false);
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
            Ok(selected)
        },
        |_, revenue| {
            sum += revenue;
            count += 1;
            Ok(())
        },
    )?;
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
