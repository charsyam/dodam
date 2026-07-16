use super::*;

pub(super) struct JoinAggregateLookupFusionPlan {
    fact_index: usize,
    fact_alias: String,
    sum_column: String,
    dimensions: Vec<JoinAggregateLookupDimensionPlan>,
}

pub(super) fn choose_join_aggregate_lookup_fusion(
    plan: &JoinAggregateLookupFusionPlan,
    join_graph: &LogicalJoinGraph,
    left_deep_plan: Option<&crate::optimizer::LogicalJoinPlan>,
) -> bool {
    let fusion_cost = estimate_join_aggregate_lookup_fusion_cost(plan, join_graph);
    let join_graph_cost = best_multi_input_join_graph_cost(join_graph, left_deep_plan);
    let threshold_pct = join_aggregate_lookup_fusion_cost_threshold_pct();
    let forced = env_flag_enabled("DODAM_FORCE_JOIN_AGGREGATE_LOOKUP_FUSION");
    let selected = forced
        || join_graph_cost.is_none_or(|cost| {
            fusion_cost.saturating_mul(100) <= cost.saturating_mul(threshold_pct)
        });
    if env_flag_enabled("DODAM_OPTIMIZER_TRACE") {
        eprintln!(
            "[dodam:optimizer] rule=join_aggregate_lookup_fusion candidate_cost={} join_graph_cost={} threshold_pct={} selected={} forced={} fact_table={} dimensions={}",
            fusion_cost,
            join_graph_cost
                .map(|cost| cost.to_string())
                .unwrap_or_else(|| "none".to_string()),
            threshold_pct,
            selected,
            forced,
            plan.fact_index,
            plan.dimensions
                .iter()
                .map(|dimension| dimension.table_index.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    selected
}

fn best_multi_input_join_graph_cost(
    join_graph: &LogicalJoinGraph,
    left_deep_plan: Option<&crate::optimizer::LogicalJoinPlan>,
) -> Option<u128> {
    left_deep_plan
        .map(|plan| plan.estimated_cost)
        .into_iter()
        .chain(join_graph.choose_exhaustive_bushy_plan_cost())
        .min()
}

fn estimate_join_aggregate_lookup_fusion_cost(
    plan: &JoinAggregateLookupFusionPlan,
    join_graph: &LogicalJoinGraph,
) -> u128 {
    let mut dimension_table_indices = [usize::MAX; 4];
    for (index, dimension) in plan.dimensions.iter().take(4).enumerate() {
        dimension_table_indices[index] = dimension.table_index;
    }
    estimate_optimizer_join_aggregate_lookup_fusion_cost(
        join_graph,
        JoinAggregateLookupFusionCostInput {
            fact_index: plan.fact_index,
            dimension_count: plan.dimensions.len().min(4),
            dimension_table_indices,
            small_group_cardinality_cap: join_aggregate_lookup_small_group_limit() as u128,
        },
    )
}

fn join_aggregate_lookup_fusion_cost_threshold_pct() -> u128 {
    std::env::var("DODAM_JOIN_AGGREGATE_LOOKUP_FUSION_COST_THRESHOLD_PCT")
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(200)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn plan_join_aggregate_lookup_fusion(
    table_count: usize,
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    group_by: &[String],
    projection: &ParsedProjection,
    distinct: bool,
    having: Option<&FilterExpr>,
) -> Result<Option<JoinAggregateLookupFusionPlan>> {
    if distinct
        || having.is_some()
        || !projection.aggregate_expressions.is_empty()
        || !projection.filtered_aggregates.is_empty()
        || group_by.is_empty()
        || group_by.len() > 4
    {
        return Ok(None);
    }
    let [AggregateExpr::CountStar, AggregateExpr::Sum(sum_column)] =
        projection.aggregates.as_slice()
    else {
        return Ok(None);
    };
    let Some(fact_alias) = join_column_owner(sum_column, alias_refs) else {
        return Ok(None);
    };
    let Some(fact_index) = aliases
        .iter()
        .position(|alias| alias.eq_ignore_ascii_case(fact_alias))
    else {
        return Ok(None);
    };

    let mut dimensions = Vec::new();
    let mut seen_dimension_aliases = HashSet::new();
    for group_column in group_by {
        let Some(group_alias) = join_column_owner(group_column, alias_refs) else {
            return Ok(None);
        };
        if group_alias.eq_ignore_ascii_case(fact_alias) {
            return Ok(None);
        }
        if !seen_dimension_aliases.insert(group_alias.to_string()) {
            return Ok(None);
        }
        let Some(dimension_index) = aliases
            .iter()
            .position(|alias| alias.eq_ignore_ascii_case(group_alias))
        else {
            return Ok(None);
        };
        let Some((fact_key, dimension_key)) = comma_join_fact_dimension_key(
            conjuncts,
            used_conjuncts,
            fact_alias,
            group_alias,
            alias_refs,
        )?
        else {
            return Ok(None);
        };
        dimensions.push(JoinAggregateLookupDimensionPlan {
            table_index: dimension_index,
            fact_key,
            dimension_key,
            payload_column: unqualified_join_column(group_column, group_alias),
        });
    }
    if dimensions.is_empty() || dimensions.len() + 1 != table_count {
        return Ok(None);
    }
    if !join_aggregate_lookup_consumes_all_residual_join_edges(
        conjuncts,
        used_conjuncts,
        fact_alias,
        &dimensions,
        aliases,
        alias_refs,
    )? {
        return Ok(None);
    }

    Ok(Some(JoinAggregateLookupFusionPlan {
        fact_index,
        fact_alias: fact_alias.to_string(),
        sum_column: sum_column.to_string(),
        dimensions,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_join_aggregate_lookup_fusion(
    engine: &DodamEngine,
    tables: &[SqlTableRef],
    scan_filters: &[Option<FilterExpr>],
    group_by: &[String],
    projection: &ParsedProjection,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    batch_size: usize,
    plan: &JoinAggregateLookupFusionPlan,
) -> Result<Option<QueryOutput>> {
    let profile = join_profile_enabled_sql();
    let total_started = profile.then(Instant::now);
    let mut lookups = Vec::with_capacity(plan.dimensions.len());
    let lookup_started = profile.then(Instant::now);
    for dimension in &plan.dimensions {
        let table = &tables[dimension.table_index];
        let batches = scan_join_side_batches(
            engine,
            table,
            batch_size,
            scan_filters[dimension.table_index].as_ref(),
            Projection::Columns(unique_columns([
                dimension.dimension_key.clone(),
                dimension.payload_column.clone(),
            ])),
        )
        .await?;
        let Some(lookup) = build_unique_i64_to_utf8_id_lookup(
            &batches,
            &dimension.dimension_key,
            &dimension.payload_column,
        )?
        else {
            return Ok(None);
        };
        lookups.push(JoinAggregateLookupDimension {
            fact_key: dimension.fact_key.clone(),
            lookup,
        });
    }
    let lookup_nanos = elapsed_optional_nanos(lookup_started);
    let lookup_dense_slices = lookups
        .iter()
        .map(|lookup| lookup.lookup.lookup.dense_slices())
        .collect::<Vec<_>>();

    let fact_sum = strip_column_prefix(&plan.sum_column, &plan.fact_alias);
    let fact_projection = Projection::Columns(unique_columns(
        lookups
            .iter()
            .map(|lookup| lookup.fact_key.clone())
            .chain(std::iter::once(fact_sum.clone())),
    ));
    let fact_scan_started = profile.then(Instant::now);
    let mut fact_stream = engine
        .scan_parquet_batches(
            tables[plan.fact_index].path.clone(),
            batch_size,
            None,
            fact_projection,
            scan_filters[plan.fact_index].clone(),
        )
        .await?;
    let mut groups = JoinAggregateLookupCountSumGroups::new(&lookups);
    let mut rows = 0usize;
    let mut batches = 0usize;
    let mut fact_array_view_nanos = 0u64;
    let mut fact_update_nanos = 0u64;
    while let Some(batch) = fact_stream.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        batches += 1;
        rows = rows.saturating_add(batch.num_rows());
        let array_view_started = profile.then(Instant::now);
        let fact_keys = lookups
            .iter()
            .map(|lookup| i64_array_like(&batch, &lookup.fact_key))
            .collect::<Result<Vec<_>>>()?;
        let sum_values = i64_array_like(&batch, &fact_sum)?;
        fact_array_view_nanos =
            fact_array_view_nanos.saturating_add(elapsed_optional_nanos(array_view_started));

        let update_started = profile.then(Instant::now);
        let keys_null_free = fact_keys.iter().all(|key| !key.has_nulls());
        let sum_null_free = !sum_values.has_nulls();
        if keys_null_free && sum_null_free && lookup_dense_slices.iter().all(Option::is_some) {
            update_join_aggregate_lookup_dense_null_free(
                &fact_keys,
                &sum_values,
                &lookup_dense_slices,
                &lookups,
                &mut groups,
                batch.num_rows(),
            );
        } else if keys_null_free && sum_null_free {
            for row in 0..batch.num_rows() {
                let mut key = [0usize; 4];
                let mut key_len = 0usize;
                let mut matched = true;
                for ((fact_key, lookup), dense_lookup) in
                    fact_keys.iter().zip(&lookups).zip(&lookup_dense_slices)
                {
                    let Some(value_id) = lookup
                        .lookup
                        .lookup
                        .get_cached(*dense_lookup, fact_key.value(row))
                    else {
                        matched = false;
                        break;
                    };
                    key[key_len] = value_id;
                    key_len += 1;
                }
                if matched {
                    groups.update(&key[..key_len], Some(sum_values.value(row)));
                }
            }
        } else {
            for row in 0..batch.num_rows() {
                let mut key = [0usize; 4];
                let mut key_len = 0usize;
                let mut matched = true;
                for ((fact_key, lookup), dense_lookup) in
                    fact_keys.iter().zip(&lookups).zip(&lookup_dense_slices)
                {
                    if fact_key.is_null(row) {
                        matched = false;
                        break;
                    }
                    let Some(value_id) = lookup
                        .lookup
                        .lookup
                        .get_cached(*dense_lookup, fact_key.value(row))
                    else {
                        matched = false;
                        break;
                    };
                    key[key_len] = value_id;
                    key_len += 1;
                }
                if !matched {
                    continue;
                }
                groups.update(
                    &key[..key_len],
                    (!sum_values.is_null(row)).then(|| sum_values.value(row)),
                );
            }
        }
        fact_update_nanos =
            fact_update_nanos.saturating_add(elapsed_optional_nanos(update_started));
    }
    let fact_scan_nanos = elapsed_optional_nanos(fact_scan_started);

    let finish_started = profile.then(Instant::now);
    let mut group_results = groups.finish(&lookups, &plan.sum_column);
    group_results.sort_by(|left, right| compare_join_fused_group_keys(&left.keys, &right.keys));
    let finish_nanos = elapsed_optional_nanos(finish_started);
    let output_started = profile.then(Instant::now);
    let metrics = AggregateMetrics {
        fragments: tables.len(),
        batches,
        rows,
        groups: group_results,
        ..AggregateMetrics::default()
    };
    let mut output = aggregate_metrics_to_batches(&metrics, group_by, &projection.aggregates)?;
    output = apply_output_order_limit(output, order_by, limit, 0)?;
    output = rename_output_batches(output, &projection.aliases)?;
    let output_nanos = elapsed_optional_nanos(output_started);
    if profile {
        eprintln!(
            "[dodam:join-fusion-profile] rule=join_aggregate_lookup_fusion total={:.3}ms lookup={:.3}ms fact_scan_aggregate={:.3}ms fact_array_view={:.3}ms fact_update={:.3}ms finish={:.3}ms output={:.3}ms fact_batches={} fact_rows={} groups={}",
            sql_nanos_to_millis(elapsed_optional_nanos(total_started)),
            sql_nanos_to_millis(lookup_nanos),
            sql_nanos_to_millis(fact_scan_nanos),
            sql_nanos_to_millis(fact_array_view_nanos),
            sql_nanos_to_millis(fact_update_nanos),
            sql_nanos_to_millis(finish_nanos),
            sql_nanos_to_millis(output_nanos),
            batches,
            rows,
            metrics.groups.len(),
        );
    }
    Ok(Some(QueryOutput::Aggregate {
        metrics,
        batches: output,
    }))
}

struct JoinAggregateLookupDimensionPlan {
    table_index: usize,
    fact_key: String,
    dimension_key: String,
    payload_column: String,
}

struct JoinAggregateLookupDimension {
    fact_key: String,
    lookup: UniqueI64ToUtf8IdLookup,
}

fn update_join_aggregate_lookup_dense_null_free(
    fact_keys: &[I64LikeColumn<'_>],
    sum_values: &I64LikeColumn<'_>,
    lookup_dense_slices: &[Option<(&[usize], &[bool])>],
    lookups: &[JoinAggregateLookupDimension],
    groups: &mut JoinAggregateLookupCountSumGroups,
    rows: usize,
) {
    if update_join_aggregate_lookup_dense_groups_direct(
        fact_keys,
        sum_values,
        lookup_dense_slices,
        lookups,
        groups,
        rows,
    ) {
        return;
    }
    match lookups.len() {
        2 => {
            let (values0, present0) = lookup_dense_slices[0].expect("validated dense lookup");
            let (values1, present1) = lookup_dense_slices[1].expect("validated dense lookup");
            for row in 0..rows {
                let Some(first_id) = dense_usize_lookup(values0, present0, fact_keys[0].value(row))
                else {
                    continue;
                };
                let Some(second_id) =
                    dense_usize_lookup(values1, present1, fact_keys[1].value(row))
                else {
                    continue;
                };
                groups.update(&[first_id, second_id], Some(sum_values.value(row)));
            }
        }
        3 if lookups[1].fact_key == lookups[2].fact_key => {
            let (values0, present0) = lookup_dense_slices[0].expect("validated dense lookup");
            let (values1, present1) = lookup_dense_slices[1].expect("validated dense lookup");
            let (values2, present2) = lookup_dense_slices[2].expect("validated dense lookup");
            for row in 0..rows {
                let Some(first_id) = dense_usize_lookup(values0, present0, fact_keys[0].value(row))
                else {
                    continue;
                };
                let shared_key = fact_keys[1].value(row);
                let Some(second_id) = dense_usize_lookup(values1, present1, shared_key) else {
                    continue;
                };
                let Some(third_id) = dense_usize_lookup(values2, present2, shared_key) else {
                    continue;
                };
                groups.update(
                    &[first_id, second_id, third_id],
                    Some(sum_values.value(row)),
                );
            }
        }
        3 => {
            let (values0, present0) = lookup_dense_slices[0].expect("validated dense lookup");
            let (values1, present1) = lookup_dense_slices[1].expect("validated dense lookup");
            let (values2, present2) = lookup_dense_slices[2].expect("validated dense lookup");
            for row in 0..rows {
                let Some(first_id) = dense_usize_lookup(values0, present0, fact_keys[0].value(row))
                else {
                    continue;
                };
                let Some(second_id) =
                    dense_usize_lookup(values1, present1, fact_keys[1].value(row))
                else {
                    continue;
                };
                let Some(third_id) = dense_usize_lookup(values2, present2, fact_keys[2].value(row))
                else {
                    continue;
                };
                groups.update(
                    &[first_id, second_id, third_id],
                    Some(sum_values.value(row)),
                );
            }
        }
        _ => {
            for row in 0..rows {
                let mut key = [0usize; 4];
                let mut key_len = 0usize;
                let mut matched = true;
                for (fact_key, dense_lookup) in fact_keys.iter().zip(lookup_dense_slices) {
                    let (values, present) = dense_lookup.expect("validated dense lookup");
                    let Some(value_id) = dense_usize_lookup(values, present, fact_key.value(row))
                    else {
                        matched = false;
                        break;
                    };
                    key[key_len] = value_id;
                    key_len += 1;
                }
                if matched {
                    groups.update(&key[..key_len], Some(sum_values.value(row)));
                }
            }
        }
    }
}

fn update_join_aggregate_lookup_dense_groups_direct(
    fact_keys: &[I64LikeColumn<'_>],
    sum_values: &I64LikeColumn<'_>,
    lookup_dense_slices: &[Option<(&[usize], &[bool])>],
    lookups: &[JoinAggregateLookupDimension],
    groups: &mut JoinAggregateLookupCountSumGroups,
    rows: usize,
) -> bool {
    match groups {
        JoinAggregateLookupCountSumGroups::TwoDense { second_len, groups }
            if lookups.len() == 2 =>
        {
            let (values0, present0) = lookup_dense_slices[0].expect("validated dense lookup");
            let (values1, present1) = lookup_dense_slices[1].expect("validated dense lookup");
            let second_len = *second_len;
            for row in 0..rows {
                let Some(first_id) = dense_usize_lookup(values0, present0, fact_keys[0].value(row))
                else {
                    continue;
                };
                let Some(second_id) =
                    dense_usize_lookup(values1, present1, fact_keys[1].value(row))
                else {
                    continue;
                };
                let slot = first_id * second_len + second_id;
                join_aggregate_lookup_update_group_slot(
                    &mut groups[slot],
                    Some(sum_values.value(row)),
                );
            }
            true
        }
        JoinAggregateLookupCountSumGroups::ThreeDense {
            second_len,
            third_len,
            groups,
        } if lookups.len() == 3 && lookups[1].fact_key == lookups[2].fact_key => {
            let (values0, present0) = lookup_dense_slices[0].expect("validated dense lookup");
            let (values1, present1) = lookup_dense_slices[1].expect("validated dense lookup");
            let (values2, present2) = lookup_dense_slices[2].expect("validated dense lookup");
            let second_len = *second_len;
            let third_len = *third_len;
            for row in 0..rows {
                let Some(first_id) = dense_usize_lookup(values0, present0, fact_keys[0].value(row))
                else {
                    continue;
                };
                let shared_key = fact_keys[1].value(row);
                let Some(second_id) = dense_usize_lookup(values1, present1, shared_key) else {
                    continue;
                };
                let Some(third_id) = dense_usize_lookup(values2, present2, shared_key) else {
                    continue;
                };
                let slot = (first_id * second_len + second_id) * third_len + third_id;
                join_aggregate_lookup_update_group_slot(
                    &mut groups[slot],
                    Some(sum_values.value(row)),
                );
            }
            true
        }
        JoinAggregateLookupCountSumGroups::ThreeDense {
            second_len,
            third_len,
            groups,
        } if lookups.len() == 3 => {
            let (values0, present0) = lookup_dense_slices[0].expect("validated dense lookup");
            let (values1, present1) = lookup_dense_slices[1].expect("validated dense lookup");
            let (values2, present2) = lookup_dense_slices[2].expect("validated dense lookup");
            let second_len = *second_len;
            let third_len = *third_len;
            for row in 0..rows {
                let Some(first_id) = dense_usize_lookup(values0, present0, fact_keys[0].value(row))
                else {
                    continue;
                };
                let Some(second_id) =
                    dense_usize_lookup(values1, present1, fact_keys[1].value(row))
                else {
                    continue;
                };
                let Some(third_id) = dense_usize_lookup(values2, present2, fact_keys[2].value(row))
                else {
                    continue;
                };
                let slot = (first_id * second_len + second_id) * third_len + third_id;
                join_aggregate_lookup_update_group_slot(
                    &mut groups[slot],
                    Some(sum_values.value(row)),
                );
            }
            true
        }
        _ => false,
    }
}

#[inline]
fn join_aggregate_lookup_update_group_slot(
    slot: &mut Option<JoinAggregateLookupCountSumGroup>,
    sum: Option<i64>,
) {
    let group = slot.get_or_insert_with(JoinAggregateLookupCountSumGroup::default);
    group.count = group.count.saturating_add(1);
    if let Some(sum) = sum {
        group.sum = group.sum.saturating_add(sum);
        group.sum_count = group.sum_count.saturating_add(1);
    }
}

pub(super) fn join_aggregate_lookup_fusion_disabled() -> bool {
    env_flag_enabled("DODAM_DISABLE_JOIN_AGGREGATE_LOOKUP_FUSION")
        || env_flag_enabled("DODAM_DISABLE_MULTI_COMMA_LOOKUP_COUNT_SUM_FUSION")
}

#[inline]
fn dense_usize_lookup(values: &[usize], present: &[bool], key: i64) -> Option<usize> {
    let index = usize::try_from(key).ok()?;
    present
        .get(index)
        .copied()
        .filter(|present| *present)
        .map(|_| values[index])
}

fn join_aggregate_lookup_small_group_limit() -> usize {
    env_usize_with_legacy_alias(
        "DODAM_JOIN_AGGREGATE_LOOKUP_SMALL_GROUP_LIMIT",
        "DODAM_MULTI_COMMA_LOOKUP_SMALL_GROUP_LIMIT",
        64,
    )
}

fn join_aggregate_lookup_dense_group_slots() -> usize {
    env_usize_with_legacy_alias(
        "DODAM_JOIN_AGGREGATE_LOOKUP_DENSE_GROUP_SLOTS",
        "DODAM_MULTI_COMMA_LOOKUP_DENSE_GROUP_SLOTS",
        4096,
    )
}

fn env_usize_with_legacy_alias(primary: &str, legacy: &str, default: usize) -> usize {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(legacy).ok())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

#[derive(Clone, Default)]
struct JoinAggregateLookupCountSumGroup {
    count: u64,
    sum: i64,
    sum_count: u64,
}

enum JoinAggregateLookupCountSumGroups {
    TwoDense {
        second_len: usize,
        groups: Vec<Option<JoinAggregateLookupCountSumGroup>>,
    },
    TwoSmall(Vec<((usize, usize), JoinAggregateLookupCountSumGroup)>),
    Two(FastHashMap<(usize, usize), JoinAggregateLookupCountSumGroup>),
    ThreeDense {
        second_len: usize,
        third_len: usize,
        groups: Vec<Option<JoinAggregateLookupCountSumGroup>>,
    },
    ThreeSmall(Vec<((usize, usize, usize), JoinAggregateLookupCountSumGroup)>),
    Three(FastHashMap<(usize, usize, usize), JoinAggregateLookupCountSumGroup>),
    Generic(FastHashMap<Vec<usize>, JoinAggregateLookupCountSumGroup>),
}

impl JoinAggregateLookupCountSumGroups {
    fn new(lookups: &[JoinAggregateLookupDimension]) -> Self {
        match lookups.len() {
            2 => {
                let first_len = lookups[0].lookup.values.len();
                let second_len = lookups[1].lookup.values.len();
                let slots = first_len.saturating_mul(second_len);
                if slots > 0 && slots <= join_aggregate_lookup_dense_group_slots() {
                    let mut groups = Vec::with_capacity(slots);
                    groups.resize_with(slots, || None);
                    Self::TwoDense { second_len, groups }
                } else {
                    Self::TwoSmall(Vec::new())
                }
            }
            3 => {
                let first_len = lookups[0].lookup.values.len();
                let second_len = lookups[1].lookup.values.len();
                let third_len = lookups[2].lookup.values.len();
                let slots = first_len
                    .saturating_mul(second_len)
                    .saturating_mul(third_len);
                if slots > 0 && slots <= join_aggregate_lookup_dense_group_slots() {
                    let mut groups = Vec::with_capacity(slots);
                    groups.resize_with(slots, || None);
                    Self::ThreeDense {
                        second_len,
                        third_len,
                        groups,
                    }
                } else {
                    Self::ThreeSmall(Vec::new())
                }
            }
            _ => Self::Generic(FastHashMap::default()),
        }
    }

    fn update(&mut self, key: &[usize], sum: Option<i64>) {
        let group = match self {
            Self::TwoDense { second_len, groups } if key.len() == 2 => {
                let Some(slot) = key[0]
                    .checked_mul(*second_len)
                    .and_then(|slot| slot.checked_add(key[1]))
                else {
                    return;
                };
                let Some(group) = groups.get_mut(slot) else {
                    return;
                };
                group.get_or_insert_with(JoinAggregateLookupCountSumGroup::default)
            }
            Self::TwoSmall(groups) if key.len() == 2 => {
                let tuple = (key[0], key[1]);
                if let Some((_, group)) =
                    groups.iter_mut().find(|(candidate, _)| *candidate == tuple)
                {
                    group
                } else if groups.len() < join_aggregate_lookup_small_group_limit() {
                    groups.push((tuple, JoinAggregateLookupCountSumGroup::default()));
                    &mut groups.last_mut().expect("pushed small group").1
                } else {
                    let mut hash = FastHashMap::default();
                    for (key, group) in std::mem::take(groups) {
                        hash.insert(key, group);
                    }
                    *self = Self::Two(hash);
                    let Self::Two(groups) = self else {
                        unreachable!("converted to two-key hash groups");
                    };
                    groups.entry(tuple).or_default()
                }
            }
            Self::Two(groups) if key.len() == 2 => groups.entry((key[0], key[1])).or_default(),
            Self::ThreeDense {
                second_len,
                third_len,
                groups,
            } if key.len() == 3 => {
                let Some(slot) = key[0]
                    .checked_mul(*second_len)
                    .and_then(|slot| slot.checked_add(key[1]))
                    .and_then(|slot| slot.checked_mul(*third_len))
                    .and_then(|slot| slot.checked_add(key[2]))
                else {
                    return;
                };
                let Some(group) = groups.get_mut(slot) else {
                    return;
                };
                group.get_or_insert_with(JoinAggregateLookupCountSumGroup::default)
            }
            Self::ThreeSmall(groups) if key.len() == 3 => {
                let tuple = (key[0], key[1], key[2]);
                if let Some((_, group)) =
                    groups.iter_mut().find(|(candidate, _)| *candidate == tuple)
                {
                    group
                } else if groups.len() < join_aggregate_lookup_small_group_limit() {
                    groups.push((tuple, JoinAggregateLookupCountSumGroup::default()));
                    &mut groups.last_mut().expect("pushed small group").1
                } else {
                    let mut hash = FastHashMap::default();
                    for (key, group) in std::mem::take(groups) {
                        hash.insert(key, group);
                    }
                    *self = Self::Three(hash);
                    let Self::Three(groups) = self else {
                        unreachable!("converted to three-key hash groups");
                    };
                    groups.entry(tuple).or_default()
                }
            }
            Self::Three(groups) if key.len() == 3 => {
                groups.entry((key[0], key[1], key[2])).or_default()
            }
            Self::Generic(groups) => groups.entry(key.to_vec()).or_default(),
            _ => return,
        };
        group.count = group.count.saturating_add(1);
        if let Some(sum) = sum {
            group.sum = group.sum.saturating_add(sum);
            group.sum_count = group.sum_count.saturating_add(1);
        }
    }

    fn finish(
        self,
        lookups: &[JoinAggregateLookupDimension],
        sum_column: &str,
    ) -> Vec<GroupAggregateResult> {
        match self {
            Self::TwoDense { second_len, groups } => groups
                .into_iter()
                .enumerate()
                .filter_map(|(slot, state)| {
                    let state = state?;
                    let first = slot / second_len;
                    let second = slot % second_len;
                    Some(join_aggregate_lookup_group_result(
                        &[first, second],
                        state,
                        lookups,
                        sum_column,
                    ))
                })
                .collect(),
            Self::TwoSmall(groups) => groups
                .into_iter()
                .map(|((first, second), state)| {
                    join_aggregate_lookup_group_result(&[first, second], state, lookups, sum_column)
                })
                .collect(),
            Self::Two(groups) => groups
                .into_iter()
                .map(|((first, second), state)| {
                    join_aggregate_lookup_group_result(&[first, second], state, lookups, sum_column)
                })
                .collect(),
            Self::ThreeDense {
                second_len,
                third_len,
                groups,
            } => groups
                .into_iter()
                .enumerate()
                .filter_map(|(slot, state)| {
                    let state = state?;
                    let first = slot / (second_len * third_len);
                    let remainder = slot % (second_len * third_len);
                    let second = remainder / third_len;
                    let third = remainder % third_len;
                    Some(join_aggregate_lookup_group_result(
                        &[first, second, third],
                        state,
                        lookups,
                        sum_column,
                    ))
                })
                .collect(),
            Self::ThreeSmall(groups) => groups
                .into_iter()
                .map(|((first, second, third), state)| {
                    join_aggregate_lookup_group_result(
                        &[first, second, third],
                        state,
                        lookups,
                        sum_column,
                    )
                })
                .collect(),
            Self::Three(groups) => groups
                .into_iter()
                .map(|((first, second, third), state)| {
                    join_aggregate_lookup_group_result(
                        &[first, second, third],
                        state,
                        lookups,
                        sum_column,
                    )
                })
                .collect(),
            Self::Generic(groups) => groups
                .into_iter()
                .map(|(key, state)| {
                    join_aggregate_lookup_group_result(&key, state, lookups, sum_column)
                })
                .collect(),
        }
    }
}

fn join_aggregate_lookup_group_result(
    key: &[usize],
    state: JoinAggregateLookupCountSumGroup,
    lookups: &[JoinAggregateLookupDimension],
    sum_column: &str,
) -> GroupAggregateResult {
    let keys = key
        .iter()
        .enumerate()
        .map(|(index, value_id)| GroupValue::Utf8(lookups[index].lookup.values[*value_id].clone()))
        .collect::<Vec<_>>();
    GroupAggregateResult {
        keys,
        values: vec![
            AggregateResult {
                expr: AggregateExpr::CountStar,
                value: AggregateValue::Count(state.count),
            },
            AggregateResult {
                expr: AggregateExpr::Sum(sum_column.to_string()),
                value: if state.sum_count == 0 {
                    AggregateValue::Int64(None)
                } else {
                    AggregateValue::Int64(Some(state.sum))
                },
            },
        ],
    }
}

fn comma_join_fact_dimension_key(
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    fact_alias: &str,
    dimension_alias: &str,
    alias_refs: &[&str],
) -> Result<Option<(String, String)>> {
    let mut output = None;
    for (conjunct, used) in conjuncts.iter().zip(used_conjuncts) {
        if *used {
            continue;
        }
        let Some((left_alias, left_key, right_alias, right_key)) =
            comma_join_base_edge(conjunct, alias_refs)?
        else {
            continue;
        };
        let candidate = if left_alias.eq_ignore_ascii_case(fact_alias)
            && right_alias.eq_ignore_ascii_case(dimension_alias)
        {
            Some((left_key, right_key))
        } else if right_alias.eq_ignore_ascii_case(fact_alias)
            && left_alias.eq_ignore_ascii_case(dimension_alias)
        {
            Some((right_key, left_key))
        } else {
            None
        };
        if let Some(candidate) = candidate {
            if output.is_some() {
                return Ok(None);
            }
            output = Some(candidate);
        }
    }
    Ok(output)
}

fn join_aggregate_lookup_consumes_all_residual_join_edges(
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    fact_alias: &str,
    dimensions: &[JoinAggregateLookupDimensionPlan],
    aliases: &[String],
    alias_refs: &[&str],
) -> Result<bool> {
    let dimension_aliases = dimensions
        .iter()
        .map(|dimension| aliases[dimension.table_index].as_str())
        .collect::<HashSet<_>>();
    for (conjunct, used) in conjuncts.iter().zip(used_conjuncts) {
        if *used {
            continue;
        }
        let Some((left_alias, _, right_alias, _)) = comma_join_base_edge(conjunct, alias_refs)?
        else {
            return Ok(false);
        };
        let fact_left = left_alias.eq_ignore_ascii_case(fact_alias);
        let fact_right = right_alias.eq_ignore_ascii_case(fact_alias);
        if fact_left == fact_right {
            return Ok(false);
        }
        let dimension_alias = if fact_left { right_alias } else { left_alias };
        if !dimension_aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(dimension_alias))
        {
            return Ok(false);
        }
    }
    Ok(true)
}
