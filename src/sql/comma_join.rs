use super::*;

pub(super) fn record_batch_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

pub(super) fn estimated_batches_row_width(batches: &[RecordBatch]) -> u128 {
    batches
        .first()
        .map(|batch| {
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| estimated_type_width(field.data_type()))
                .sum::<u128>()
                .max(1)
        })
        .unwrap_or(1)
}

pub(super) fn estimated_type_width(data_type: &DataType) -> u128 {
    match data_type {
        DataType::Boolean => 1,
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => 8,
        DataType::Decimal128(_, _) => 16,
        DataType::Decimal256(_, _) => 32,
        DataType::Utf8 | DataType::LargeUtf8 => 24,
        DataType::Binary | DataType::LargeBinary => 24,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => 32,
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| estimated_type_width(field.data_type()))
            .sum::<u128>()
            .max(1),
        _ => 16,
    }
}

pub(super) fn build_logical_comma_join_graph(
    scanned: &[Option<Vec<RecordBatch>>],
    row_counts: &[usize],
    base_row_counts: &[usize],
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
) -> Result<LogicalJoinGraph> {
    let (edges, key_columns) = comma_join_graph_edges_and_keys(aliases, alias_refs, conjuncts)?;
    build_logical_multi_join_graph(scanned, row_counts, base_row_counts, &key_columns, edges)
}

pub(super) fn build_logical_multi_join_graph_from_metadata(
    engine: &DodamEngine,
    tables: &[SqlTableRef],
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    scan_projections: &[Projection],
) -> Result<LogicalJoinGraph> {
    let (edges, key_columns) = comma_join_graph_edges_and_keys(aliases, alias_refs, conjuncts)?;
    let mut table_stats = Vec::with_capacity(tables.len());
    for (index, table) in tables.iter().enumerate() {
        let rows = engine
            .parquet_total_row_count(&table.path)
            .unwrap_or_else(|_| 0)
            .max(1) as u128;
        table_stats.push(LogicalJoinTableStats {
            base_rows: rows,
            rows,
            row_width: scan_projections
                .get(index)
                .map(estimated_projection_width)
                .unwrap_or(128)
                .max(1),
            key_ndv: HashMap::new(),
            column_ranges: HashMap::new(),
        });
    }
    for (index, keys) in key_columns.iter().enumerate() {
        let rows = table_stats[index].rows;
        for key in keys {
            table_stats[index].key_ndv.insert(key.clone(), rows);
        }
    }
    for edge in &edges {
        let left_rows = table_stats[edge.left].rows;
        let right_rows = table_stats[edge.right].rows;
        let edge_ndv = left_rows.min(right_rows).max(1);
        table_stats[edge.left]
            .key_ndv
            .entry(edge.left_key.clone())
            .and_modify(|value| *value = (*value).min(edge_ndv))
            .or_insert(edge_ndv);
        table_stats[edge.right]
            .key_ndv
            .entry(edge.right_key.clone())
            .and_modify(|value| *value = (*value).min(edge_ndv))
            .or_insert(edge_ndv);
    }
    for (table_index, keys) in key_columns.iter().enumerate() {
        for key in keys {
            if let Some(stats) =
                parquet_column_range_stats_from_metadata(engine, &tables[table_index].path, key)?
            {
                let range_width = stats
                    .max_i128
                    .checked_sub(stats.min_i128)
                    .and_then(|value| value.checked_add(1))
                    .and_then(|value| u128::try_from(value).ok())
                    .filter(|value| *value > 0);
                table_stats[table_index]
                    .column_ranges
                    .insert(key.clone(), stats);
                if let Some(range_width) = range_width {
                    let table_rows = table_stats[table_index].rows;
                    let ndv_upper = range_width.min(table_rows).max(1);
                    table_stats[table_index]
                        .key_ndv
                        .entry(key.clone())
                        .and_modify(|value| *value = (*value).min(ndv_upper))
                        .or_insert(ndv_upper);
                }
            }
        }
    }
    Ok(LogicalJoinGraph {
        tables: table_stats,
        edges,
    })
}

pub(super) fn parquet_column_range_stats_from_metadata(
    engine: &DodamEngine,
    path: &Path,
    column: &str,
) -> Result<Option<ColumnRangeStats>> {
    let Some(row_groups) = engine.parquet_primitive_column_min_max_by_row_group(path, column)?
    else {
        return Ok(None);
    };
    let mut rows = 0u128;
    let mut min_i128 = i128::MAX;
    let mut max_i128 = i128::MIN;
    for row_group in row_groups {
        rows = rows.saturating_add(row_group.rows as u128);
        min_i128 = min_i128.min(row_group.min);
        max_i128 = max_i128.max(row_group.max);
    }
    Ok((rows > 0).then_some(ColumnRangeStats {
        min_i128,
        max_i128,
        null_count: 0,
        rows,
    }))
}

pub(super) fn comma_join_graph_edges_and_keys(
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
) -> Result<(Vec<LogicalJoinEdge>, Vec<Vec<String>>)> {
    let mut edges = Vec::new();
    let mut key_columns = vec![Vec::<String>::new(); aliases.len()];
    for conjunct in conjuncts {
        let Some((left_alias, left_key, right_alias, right_key)) =
            comma_join_base_edge(conjunct, alias_refs)?
        else {
            continue;
        };
        let Some(left) = aliases
            .iter()
            .position(|alias| alias.eq_ignore_ascii_case(left_alias))
        else {
            continue;
        };
        let Some(right) = aliases
            .iter()
            .position(|alias| alias.eq_ignore_ascii_case(right_alias))
        else {
            continue;
        };
        add_column_once(&mut key_columns[left], left_key.clone());
        add_column_once(&mut key_columns[right], right_key.clone());
        edges.push(LogicalJoinEdge {
            left,
            left_key,
            right,
            right_key,
        });
    }
    Ok((edges, key_columns))
}

pub(super) fn build_logical_multi_join_graph(
    scanned: &[Option<Vec<RecordBatch>>],
    row_counts: &[usize],
    base_row_counts: &[usize],
    key_columns: &[Vec<String>],
    edges: Vec<LogicalJoinEdge>,
) -> Result<LogicalJoinGraph> {
    let mut tables = Vec::with_capacity(scanned.len());
    for (index, batches) in scanned.iter().enumerate() {
        let batches = batches.as_ref().expect("comma join input scanned");
        let mut key_ndv = HashMap::new();
        let mut column_ranges = HashMap::new();
        for key in key_columns.get(index).into_iter().flatten() {
            key_ndv.insert(
                key.clone(),
                sampled_key_ndv(batches, std::slice::from_ref(key), 100_000)? as u128,
            );
            if let Some(range) = primitive_column_range_stats(batches, key)? {
                column_ranges.insert(key.clone(), range);
            }
        }
        tables.push(LogicalJoinTableStats {
            base_rows: base_row_counts
                .get(index)
                .copied()
                .unwrap_or(row_counts[index])
                .max(row_counts[index])
                .max(1) as u128,
            rows: row_counts[index].max(1) as u128,
            row_width: estimated_batches_row_width(batches),
            key_ndv,
            column_ranges,
        });
    }

    Ok(LogicalJoinGraph { tables, edges })
}

pub(super) fn log_multi_input_join_optimizer_plan(
    rule_name: &str,
    join_graph: &LogicalJoinGraph,
    left_deep_plan: Option<&crate::optimizer::LogicalJoinPlan>,
) {
    if !std::env::var("DODAM_OPTIMIZER_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    let left_deep_cost = left_deep_plan
        .map(|plan| plan.estimated_cost.to_string())
        .unwrap_or_else(|| "none".to_string());
    let left_deep_order = left_deep_plan
        .map(|plan| {
            let mut order = vec![plan.start.to_string()];
            order.extend(plan.steps.iter().map(|step| step.table_index.to_string()));
            order.join(",")
        })
        .unwrap_or_else(|| "none".to_string());
    let bushy = join_graph.choose_exhaustive_bushy_plan();
    let bushy_cost = bushy
        .as_ref()
        .map(|plan| plan.estimated_cost().to_string())
        .unwrap_or_else(|| "none".to_string());
    let mut bushy_tables = Vec::new();
    if let Some(plan) = bushy.as_ref() {
        plan.collect_tables(&mut bushy_tables);
    }
    eprintln!(
        "[dodam:optimizer] rule={} left_deep_order={} left_deep_cost={} bushy_cost={} bushy_tables={}",
        rule_name,
        left_deep_order,
        left_deep_cost,
        bushy_cost,
        bushy_tables
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
}

pub(super) fn choose_bushy_multi_input_join_execution_tree(
    join_graph: &LogicalJoinGraph,
    left_deep_plan: Option<&crate::optimizer::LogicalJoinPlan>,
) -> Option<LogicalJoinPlanTree> {
    let tree = join_graph.choose_exhaustive_bushy_plan()?;
    if tree.table_count() <= 2 {
        return None;
    }
    if std::env::var("DODAM_FORCE_BUSHY_COMMA_JOIN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return Some(tree);
    }
    let left_deep_cost = left_deep_plan?.estimated_cost;
    (tree.estimated_cost().saturating_mul(100) < left_deep_cost.saturating_mul(95)).then_some(tree)
}

pub(super) struct CommaJoinSubtreeResult {
    pub(super) batches: Vec<RecordBatch>,
    aliases: Vec<String>,
    rows: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bushy_comma_join_tree(
    tree: &LogicalJoinPlanTree,
    scanned: &mut [Option<Vec<RecordBatch>>],
    row_counts: &[usize],
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &mut [bool],
    final_columns: &HashSet<String>,
    memory_limit_bytes: u64,
) -> Result<CommaJoinSubtreeResult> {
    match tree {
        LogicalJoinPlanTree::Leaf { table_index, .. } => {
            let batches = scanned[*table_index]
                .take()
                .expect("bushy leaf input scanned");
            Ok(CommaJoinSubtreeResult {
                batches,
                aliases: vec![aliases[*table_index].clone()],
                rows: row_counts[*table_index],
            })
        }
        LogicalJoinPlanTree::Join { left, right, .. } => {
            let left = execute_bushy_comma_join_tree(
                left,
                scanned,
                row_counts,
                aliases,
                alias_refs,
                conjuncts,
                used_conjuncts,
                final_columns,
                memory_limit_bytes,
            )?;
            let right = execute_bushy_comma_join_tree(
                right,
                scanned,
                row_counts,
                aliases,
                alias_refs,
                conjuncts,
                used_conjuncts,
                final_columns,
                memory_limit_bytes,
            )?;
            let (left_keys, right_keys, conjunct_indexes) = comma_join_keys_between_subtrees(
                conjuncts,
                used_conjuncts,
                &left.aliases,
                &right.aliases,
                alias_refs,
            )?;
            if left_keys.is_empty() {
                return Err(DodamError::UnsupportedSql(format!(
                    "bushy comma join could not find equality predicate between [{}] and [{}]",
                    left.aliases.join(", "),
                    right.aliases.join(", ")
                )));
            }
            for index in conjunct_indexes {
                used_conjuncts[index] = true;
            }
            let mut joined_aliases = left.aliases.clone();
            joined_aliases.extend(right.aliases.clone());
            let left_prefix = if left.aliases.len() == 1 {
                left.aliases[0].clone()
            } else {
                "__dodam_bushy_left".to_string()
            };
            let right_prefix = if right.aliases.len() == 1 {
                right.aliases[0].clone()
            } else {
                "__dodam_bushy_right".to_string()
            };
            let build_side = if left.rows <= right.rows {
                JoinBuildSide::Left
            } else {
                JoinBuildSide::Right
            };
            let output_projection = comma_join_hash_output_projection(
                &left.batches,
                &right.batches,
                &left_prefix,
                &right_prefix,
                &joined_aliases,
                alias_refs,
                conjuncts,
                used_conjuncts,
                final_columns,
                &left_keys,
                &right_keys,
            )?;
            let mut batches = execute_comma_hash_join(
                left.batches,
                right.batches,
                left_keys,
                right_keys,
                left_prefix.clone(),
                right_prefix.clone(),
                build_side,
                output_projection,
                memory_limit_bytes,
            )?;
            if left.aliases.len() > 1 {
                batches = strip_batch_field_prefix(batches, "__dodam_bushy_left.")?;
            }
            if right.aliases.len() > 1 {
                batches = strip_batch_field_prefix(batches, "__dodam_bushy_right.")?;
            }
            let rows = record_batch_rows(&batches);
            batches = prune_comma_join_current_columns(
                batches,
                &joined_aliases,
                alias_refs,
                conjuncts,
                used_conjuncts,
                final_columns,
            )?;
            Ok(CommaJoinSubtreeResult {
                batches,
                aliases: joined_aliases,
                rows,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_left_deep_comma_join(
    mut scanned: Vec<Option<Vec<RecordBatch>>>,
    row_counts: &[usize],
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &mut [bool],
    final_columns: &HashSet<String>,
    join_graph: &LogicalJoinGraph,
    join_plan: Option<&crate::optimizer::LogicalJoinPlan>,
    memory_limit_bytes: u64,
) -> Result<Vec<RecordBatch>> {
    if let Some(plan) = join_plan
        && choose_streaming_left_deep_comma_join_for_plan(plan, final_columns)
    {
        let mut trial_used_conjuncts = used_conjuncts.to_vec();
        if let Some(output) = try_execute_streaming_left_deep_comma_join(
            &mut scanned,
            row_counts,
            aliases,
            alias_refs,
            conjuncts,
            &mut trial_used_conjuncts,
            plan,
            final_columns,
        )? {
            used_conjuncts.copy_from_slice(&trial_used_conjuncts);
            return Ok(output);
        }
    }
    let start_index = join_plan.map(|plan| plan.start).unwrap_or_else(|| {
        row_counts
            .iter()
            .enumerate()
            .min_by_key(|(_, rows)| *rows)
            .map(|(index, _)| index)
            .expect("at least one comma join table")
    });
    let mut current = scanned[start_index].take().expect("start input scanned");
    let mut current_rows = row_counts[start_index];
    let mut joined_aliases = vec![aliases[start_index].clone()];
    let mut remaining = (0..aliases.len())
        .filter(|index| *index != start_index)
        .collect::<Vec<_>>();
    while !remaining.is_empty() {
        let mut candidates = Vec::new();
        for (remaining_index, table_index) in remaining.iter().copied().enumerate() {
            let alias = &aliases[table_index];
            let mut left_keys = Vec::new();
            let mut right_keys = Vec::new();
            let mut conjunct_indexes = Vec::new();
            for (index, conjunct) in conjuncts.iter().enumerate() {
                if used_conjuncts[index] {
                    continue;
                }
                if let Some((left_key, right_key)) =
                    comma_join_keys_for_next(conjunct, &joined_aliases, alias, alias_refs)?
                {
                    left_keys.push(left_key);
                    right_keys.push(right_key);
                    conjunct_indexes.push(index);
                }
            }
            if !left_keys.is_empty() {
                candidates.push((
                    remaining_index,
                    table_index,
                    left_keys,
                    right_keys,
                    conjunct_indexes,
                ));
            }
        }
        let selected = if candidates.len() <= 1 {
            candidates.into_iter().next()
        } else {
            let joined = aliases
                .iter()
                .map(|alias| joined_aliases.iter().any(|joined| joined == alias))
                .collect::<Vec<_>>();
            let candidate_table_indexes = candidates
                .iter()
                .map(|(_, table_index, _, _, _)| *table_index)
                .collect::<Vec<_>>();
            let selected_step = join_graph.choose_next_join(
                &joined,
                current_rows as u128,
                estimated_batches_row_width(&current),
                &candidate_table_indexes,
            );
            selected_step
                .and_then(|step| {
                    candidates
                        .iter()
                        .position(|(_, table_index, _, _, _)| *table_index == step.table_index)
                })
                .map(|index| candidates.swap_remove(index))
        };
        let Some((remaining_index, table_index, left_keys, right_keys, conjunct_indexes)) =
            selected
        else {
            return Err(DodamError::UnsupportedSql(format!(
                "comma join could not find equality predicate for remaining tables: {}",
                remaining
                    .iter()
                    .map(|index| aliases[*index].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        for index in conjunct_indexes {
            used_conjuncts[index] = true;
        }
        remaining.remove(remaining_index);
        let alias = &aliases[table_index];
        let right = scanned[table_index]
            .take()
            .expect("remaining input scanned");
        let right_rows = row_counts[table_index];
        let left_prefix = if joined_aliases.len() == 1 {
            joined_aliases[0].as_str()
        } else {
            "__dodam_join"
        };
        let build_side = if current_rows <= right_rows {
            JoinBuildSide::Left
        } else {
            JoinBuildSide::Right
        };
        let mut next_joined_aliases = joined_aliases.clone();
        next_joined_aliases.push(alias.clone());
        let output_projection = comma_join_hash_output_projection(
            &current,
            &right,
            left_prefix,
            alias,
            &next_joined_aliases,
            alias_refs,
            conjuncts,
            used_conjuncts,
            final_columns,
            &left_keys,
            &right_keys,
        )?;
        current = execute_comma_hash_join(
            current,
            right,
            left_keys,
            right_keys,
            left_prefix.to_string(),
            alias.clone(),
            build_side,
            output_projection,
            memory_limit_bytes,
        )?;
        current_rows = record_batch_rows(&current);
        if left_prefix == "__dodam_join" {
            current = strip_batch_field_prefix(current, "__dodam_join.")?;
        }
        joined_aliases = next_joined_aliases;
        current = prune_comma_join_current_columns(
            current,
            &joined_aliases,
            alias_refs,
            conjuncts,
            used_conjuncts,
            final_columns,
        )?;
    }
    Ok(current)
}

pub(super) fn choose_streaming_left_deep_comma_join_for_plan(
    plan: &crate::optimizer::LogicalJoinPlan,
    final_columns: &HashSet<String>,
) -> bool {
    if std::env::var("DODAM_DISABLE_STREAM_LEFT_DEEP_COMMA_JOIN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return false;
    }
    if std::env::var("DODAM_STREAM_LEFT_DEEP_COMMA_JOIN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return true;
    }
    std::env::var("DODAM_AUTO_STREAM_LEFT_DEEP_COMMA_JOIN")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        && choose_streaming_left_deep_join(StreamingLeftDeepJoinCostInput {
            table_count: plan.steps.len() + 1,
            projected_output_columns: final_columns.len(),
            estimated_final_rows: plan
                .steps
                .last()
                .map(|step| step.estimated_rows)
                .unwrap_or(1),
        })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute_streaming_left_deep_comma_join(
    scanned: &mut [Option<Vec<RecordBatch>>],
    row_counts: &[usize],
    aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &mut [bool],
    plan: &crate::optimizer::LogicalJoinPlan,
    final_columns: &HashSet<String>,
) -> Result<Option<Vec<RecordBatch>>> {
    if plan.steps.len() + 1 != aliases.len() {
        return Ok(None);
    }
    let mut joined_aliases = vec![aliases[plan.start].clone()];
    let start_batches = scanned[plan.start].take().expect("start input scanned");
    let mut current_schema_batches = start_batches
        .first()
        .cloned()
        .map(|batch| vec![batch])
        .unwrap_or_default();
    let mut current_rows = row_counts[plan.start];
    let mut current: Box<dyn PhysicalPlan> = Box::new(MemoryExec::new(start_batches));
    for step in &plan.steps {
        let table_index = step.table_index;
        let alias = &aliases[table_index];
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();
        let mut conjunct_indexes = Vec::new();
        for (index, conjunct) in conjuncts.iter().enumerate() {
            if used_conjuncts[index] {
                continue;
            }
            if let Some((left_key, right_key)) =
                comma_join_keys_for_next(conjunct, &joined_aliases, alias, alias_refs)?
            {
                left_keys.push(left_key);
                right_keys.push(right_key);
                conjunct_indexes.push(index);
            }
        }
        if left_keys.is_empty() {
            return Ok(None);
        }
        for index in conjunct_indexes {
            used_conjuncts[index] = true;
        }
        let right = scanned[table_index]
            .take()
            .expect("remaining input scanned");
        let right_schema_batches = right
            .first()
            .cloned()
            .map(|batch| vec![batch])
            .unwrap_or_default();
        let left_prefix = if joined_aliases.len() == 1 {
            joined_aliases[0].clone()
        } else {
            "__dodam_join".to_string()
        };
        let build_side = if row_counts[table_index] <= current_rows {
            JoinBuildSide::Right
        } else {
            JoinBuildSide::Left
        };
        let mut next_joined_aliases = joined_aliases.clone();
        next_joined_aliases.push(alias.clone());
        let output_projection = comma_join_hash_output_projection(
            &current_schema_batches,
            &right_schema_batches,
            &left_prefix,
            alias,
            &next_joined_aliases,
            alias_refs,
            conjuncts,
            used_conjuncts,
            final_columns,
            &left_keys,
            &right_keys,
        )?;
        let output_schema = comma_join_output_schema_from_projection(
            current_schema_batches.first(),
            right_schema_batches.first(),
            &left_prefix,
            alias,
            build_side,
            &output_projection,
        )?;
        if sql_join_profile_enabled() {
            eprintln!(
                "[dodam:sql-join-profile] mode=stream-plan step={} left_prefix={} right_prefix={} build_side={:?} estimated_rows={} left_schema_cols={} right_schema_cols={} output_cols={}",
                joined_aliases.len(),
                left_prefix,
                alias,
                build_side,
                step.estimated_rows,
                current_schema_batches
                    .first()
                    .map(RecordBatch::num_columns)
                    .unwrap_or(0),
                right_schema_batches
                    .first()
                    .map(RecordBatch::num_columns)
                    .unwrap_or(0),
                projection_column_count(&output_projection),
            );
        }
        current = Box::new(HashJoinExec::new(
            current,
            Box::new(MemoryExec::new(right)),
            left_keys,
            right_keys,
            left_prefix.clone(),
            alias.clone(),
            build_side,
            JoinType::Inner,
            output_projection,
        ));
        if left_prefix == "__dodam_join" {
            current = Box::new(StripPrefixExec::new(current, "__dodam_join.".to_string()));
            current_schema_batches = vec![strip_record_batch_prefix_for_schema(
                output_schema,
                "__dodam_join.",
            )];
        } else {
            current_schema_batches = vec![schema_record_batch(output_schema)];
        }
        current_rows = step.estimated_rows.max(1) as usize;
        joined_aliases = next_joined_aliases;
    }
    let started = Instant::now();
    let output = collect_batches(current.execute()?)?;
    if sql_join_profile_enabled() {
        eprintln!(
            "[dodam:sql-join-profile] mode=stream-total tables={} output_rows={} output_cols={} elapsed={:.3}ms",
            aliases.len(),
            record_batch_rows(&output),
            output.first().map(RecordBatch::num_columns).unwrap_or(0),
            elapsed_millis(started),
        );
    }
    Ok(Some(output))
}

pub(super) fn comma_join_output_schema_from_projection(
    left: Option<&RecordBatch>,
    right: Option<&RecordBatch>,
    left_prefix: &str,
    right_prefix: &str,
    build_side: JoinBuildSide,
    projection: &Projection,
) -> Result<Arc<Schema>> {
    let left_fields =
        comma_join_projected_side_fields(left, left_prefix, projection, ProjectionSide::Left)?;
    let right_fields =
        comma_join_projected_side_fields(right, right_prefix, projection, ProjectionSide::Right)?;
    let fields = match build_side {
        JoinBuildSide::Left | JoinBuildSide::Right => left_fields
            .into_iter()
            .chain(right_fields)
            .collect::<Vec<_>>(),
    };
    Ok(Arc::new(Schema::new(fields)))
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ProjectionSide {
    Left,
    Right,
}

pub(super) fn comma_join_projected_side_fields(
    batch: Option<&RecordBatch>,
    prefix: &str,
    projection: &Projection,
    _side: ProjectionSide,
) -> Result<Vec<Field>> {
    let Some(batch) = batch else {
        return Ok(Vec::new());
    };
    let field_indexes = match projection {
        Projection::All => (0..batch.num_columns()).collect::<Vec<_>>(),
        Projection::Columns(columns) => columns
            .iter()
            .filter_map(|column| {
                column
                    .strip_prefix(prefix)
                    .and_then(|name| name.strip_prefix('.'))
            })
            .map(|name| batch_column_index(batch, name))
            .collect::<Result<Vec<_>>>()?,
    };
    Ok(field_indexes
        .into_iter()
        .map(|index| {
            let field = batch.schema().field(index).clone();
            Field::new(
                format!("{prefix}.{}", field.name()),
                field.data_type().clone(),
                true,
            )
        })
        .collect())
}

pub(super) fn schema_record_batch(schema: Arc<Schema>) -> RecordBatch {
    RecordBatch::new_empty(schema)
}

pub(super) fn strip_record_batch_prefix_for_schema(
    schema: Arc<Schema>,
    prefix: &str,
) -> RecordBatch {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let name = field
                .name()
                .as_str()
                .strip_prefix(prefix)
                .unwrap_or(field.name().as_str())
                .to_string();
            Arc::new(Field::new(
                name,
                field.data_type().clone(),
                field.is_nullable(),
            ))
        })
        .collect::<Vec<_>>();
    RecordBatch::new_empty(Arc::new(Schema::new(fields)))
}

pub(super) fn comma_join_keys_between_subtrees(
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    left_aliases: &[String],
    right_aliases: &[String],
    table_aliases: &[&str],
) -> Result<(Vec<String>, Vec<String>, Vec<usize>)> {
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut conjunct_indexes = Vec::new();
    for (index, conjunct) in conjuncts.iter().enumerate() {
        if used_conjuncts[index] {
            continue;
        }
        if let Some((left_key, right_key)) =
            comma_join_key_for_subtrees(conjunct, left_aliases, right_aliases, table_aliases)?
        {
            left_keys.push(left_key);
            right_keys.push(right_key);
            conjunct_indexes.push(index);
        }
    }
    Ok((left_keys, right_keys, conjunct_indexes))
}

pub(super) fn comma_join_key_for_subtrees(
    expr: &SqlExpr,
    left_aliases: &[String],
    right_aliases: &[String],
    table_aliases: &[&str],
) -> Result<Option<(String, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let Some(left_owner) = join_column_owner(&left_column, table_aliases) else {
        return Ok(None);
    };
    let Some(right_owner) = join_column_owner(&right_column, table_aliases) else {
        return Ok(None);
    };
    let left_has_left = left_aliases.iter().any(|alias| alias == left_owner);
    let left_has_right = left_aliases.iter().any(|alias| alias == right_owner);
    let right_has_left = right_aliases.iter().any(|alias| alias == left_owner);
    let right_has_right = right_aliases.iter().any(|alias| alias == right_owner);
    if left_has_left && right_has_right {
        return Ok(Some((
            joined_comma_join_key(&left_column, left_owner, left_aliases),
            joined_comma_join_key(&right_column, right_owner, right_aliases),
        )));
    }
    if left_has_right && right_has_left {
        return Ok(Some((
            joined_comma_join_key(&right_column, right_owner, left_aliases),
            joined_comma_join_key(&left_column, left_owner, right_aliases),
        )));
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_comma_hash_join(
    left: Vec<RecordBatch>,
    right: Vec<RecordBatch>,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_prefix: String,
    right_prefix: String,
    build_side: JoinBuildSide,
    output_projection: Projection,
    memory_limit_bytes: u64,
) -> Result<Vec<RecordBatch>> {
    let started = Instant::now();
    let left_rows = record_batch_rows(&left);
    let right_rows = record_batch_rows(&right);
    let left_cols = left.first().map(RecordBatch::num_columns).unwrap_or(0);
    let right_cols = right.first().map(RecordBatch::num_columns).unwrap_or(0);
    let output_projection_cols = projection_column_count(&output_projection);
    let profile_left_prefix = left_prefix.clone();
    let profile_right_prefix = right_prefix.clone();
    let estimated_rows = record_batch_rows(&left)
        .max(record_batch_rows(&right))
        .max(1) as u128;
    let estimated_row_width = estimated_batches_row_width(&left)
        .saturating_add(estimated_batches_row_width(&right))
        .max(1);
    let memory_strategy = choose_pipeline_memory_strategy(PipelineMemoryCostInput {
        estimated_rows,
        estimated_row_width,
        memory_limit_bytes: memory_limit_bytes as u128,
    });
    let stream = match memory_strategy {
        PipelineMemoryStrategy::InMemory => Box::new(HashJoinExec::new(
            Box::new(MemoryExec::new(left)),
            Box::new(MemoryExec::new(right)),
            left_keys,
            right_keys,
            left_prefix,
            right_prefix,
            build_side,
            JoinType::Inner,
            output_projection,
        )) as Box<dyn PhysicalPlan>,
        PipelineMemoryStrategy::Partitioned { partitions } => {
            Box::new(PartitionedHashJoinExec::new(
                Box::new(MemoryExec::new(left)),
                Box::new(MemoryExec::new(right)),
                left_keys,
                right_keys,
                left_prefix,
                right_prefix,
                PartitionedHashJoinOptions {
                    partitions,
                    memory_limit_bytes,
                    join_type: JoinType::Inner,
                    output_projection,
                },
            )) as Box<dyn PhysicalPlan>
        }
        PipelineMemoryStrategy::External => Box::new(PartitionedHashJoinExec::new(
            Box::new(MemoryExec::new(left)),
            Box::new(MemoryExec::new(right)),
            left_keys,
            right_keys,
            left_prefix,
            right_prefix,
            PartitionedHashJoinOptions {
                partitions: MAX_SQL_EXTERNAL_JOIN_PARTITIONS,
                memory_limit_bytes,
                join_type: JoinType::Inner,
                output_projection,
            },
        )) as Box<dyn PhysicalPlan>,
    }
    .execute()?;
    let output = collect_batches(stream)?;
    if sql_join_profile_enabled() {
        eprintln!(
            "[dodam:sql-join-profile] mode=materialized left_prefix={} right_prefix={} build_side={:?} strategy={:?} left_rows={} right_rows={} output_rows={} left_cols={} right_cols={} output_cols={} elapsed={:.3}ms",
            profile_left_prefix,
            profile_right_prefix,
            build_side,
            memory_strategy,
            left_rows,
            right_rows,
            record_batch_rows(&output),
            left_cols,
            right_cols,
            output_projection_cols,
            elapsed_millis(started),
        );
    }
    Ok(output)
}

pub(super) fn sql_join_profile_enabled() -> bool {
    std::env::var("DODAM_JOIN_PROFILE").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(super) fn elapsed_millis(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

#[allow(clippy::too_many_arguments)]
pub(super) fn comma_join_hash_output_projection(
    left: &[RecordBatch],
    right: &[RecordBatch],
    left_prefix: &str,
    right_prefix: &str,
    joined_aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    final_columns: &HashSet<String>,
    left_keys: &[String],
    right_keys: &[String],
) -> Result<Projection> {
    let needed = comma_join_needed_columns(alias_refs, conjuncts, used_conjuncts, final_columns)?;
    if needed.is_empty() {
        return Ok(Projection::All);
    }
    let mut columns = Vec::new();
    add_comma_join_output_projection_side(
        &mut columns,
        left.first().map(RecordBatch::schema).as_deref(),
        left_prefix,
        joined_aliases,
        &needed,
    );
    add_comma_join_output_projection_side(
        &mut columns,
        right.first().map(RecordBatch::schema).as_deref(),
        right_prefix,
        joined_aliases,
        &needed,
    );
    if !zero_column_join_output_enabled() {
        ensure_comma_join_projection_side_key(&mut columns, left_prefix, left_keys);
        ensure_comma_join_projection_side_key(&mut columns, right_prefix, right_keys);
    }
    log_comma_join_hash_output_projection(left_prefix, right_prefix, &columns, &needed);
    if columns.is_empty() {
        Ok(Projection::All)
    } else {
        Ok(Projection::Columns(columns))
    }
}

pub(super) fn zero_column_join_output_enabled() -> bool {
    !std::env::var("DODAM_DISABLE_ZERO_COLUMN_JOIN_OUTPUT")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn ensure_comma_join_projection_side_key(
    columns: &mut Vec<String>,
    prefix: &str,
    keys: &[String],
) {
    let side_prefix = format!("{prefix}.");
    if columns
        .iter()
        .any(|column| column.starts_with(&side_prefix))
    {
        return;
    }
    if let Some(key) = keys.first() {
        add_column_once(columns, format!("{prefix}.{key}"));
    }
}

pub(super) fn log_comma_join_hash_output_projection(
    left_prefix: &str,
    right_prefix: &str,
    columns: &[String],
    needed: &HashSet<String>,
) {
    if !std::env::var("DODAM_OPTIMIZER_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    {
        return;
    }
    eprintln!(
        "[dodam:optimizer] rule=multi_input_join_output_projection left_prefix={} right_prefix={} columns=[{}] needed=[{}]",
        left_prefix,
        right_prefix,
        columns.join(","),
        {
            let mut needed = needed.iter().cloned().collect::<Vec<_>>();
            needed.sort();
            needed.join(",")
        }
    );
}

pub(super) fn add_comma_join_output_projection_side(
    columns: &mut Vec<String>,
    schema: Option<&Schema>,
    prefix: &str,
    joined_aliases: &[String],
    needed: &HashSet<String>,
) {
    let Some(schema) = schema else {
        return;
    };
    for field in schema.fields() {
        let projected_name = format!("{prefix}.{}", field.name());
        let output_name = if prefix.starts_with("__dodam_") {
            field.name().to_string()
        } else {
            projected_name.clone()
        };
        if comma_join_field_needed(&output_name, joined_aliases, needed) {
            add_column_once(columns, projected_name);
        }
    }
}

pub(super) fn comma_join_needed_columns(
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    final_columns: &HashSet<String>,
) -> Result<HashSet<String>> {
    let mut needed = final_columns.clone();
    for (conjunct, used) in conjuncts.iter().zip(used_conjuncts) {
        if *used {
            continue;
        }
        if expr_contains_materializable_subquery(conjunct) {
            return Ok(HashSet::new());
        }
        let mut columns = Vec::new();
        collect_join_column_candidates(conjunct, alias_refs, &mut columns)?;
        needed.extend(columns);
    }
    Ok(needed)
}

pub(super) fn sampled_key_ndv(
    batches: &[RecordBatch],
    keys: &[String],
    sample_rows: usize,
) -> Result<usize> {
    let mut values = HashSet::new();
    let mut sampled = 0usize;
    for batch in batches {
        if sampled >= sample_rows {
            break;
        }
        let key_indices = keys
            .iter()
            .map(|key| batch_column_index(batch, key))
            .collect::<Result<Vec<_>>>()?;
        for row in 0..batch.num_rows() {
            if sampled >= sample_rows {
                break;
            }
            sampled += 1;
            let mut parts = Vec::with_capacity(key_indices.len());
            let mut has_null = false;
            for index in &key_indices {
                match semijoin_key_at(batch.column(*index), row)? {
                    Some(value) => parts.push(value),
                    None => {
                        has_null = true;
                        break;
                    }
                }
            }
            if !has_null {
                values.insert(parts.join("\x1f"));
            }
        }
    }
    Ok(values.len())
}

pub(super) fn primitive_column_range_stats(
    batches: &[RecordBatch],
    column: &str,
) -> Result<Option<ColumnRangeStats>> {
    let mut min_value = None::<i128>;
    let mut max_value = None::<i128>;
    let mut null_count = 0u128;
    let mut rows = 0u128;
    for batch in batches {
        let index = batch_column_index(batch, column)?;
        let array = batch.column(index);
        rows = rows.saturating_add(array.len() as u128);
        if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
            update_primitive_i32_range(values, &mut min_value, &mut max_value, &mut null_count);
        } else if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
            update_primitive_i64_range(values, &mut min_value, &mut max_value, &mut null_count);
        } else {
            return Ok(None);
        }
    }
    let Some(min_i128) = min_value else {
        return Ok(None);
    };
    Ok(Some(ColumnRangeStats {
        min_i128,
        max_i128: max_value.expect("max set with min"),
        null_count,
        rows,
    }))
}

pub(super) fn update_primitive_i32_range(
    values: &Int32Array,
    min_value: &mut Option<i128>,
    max_value: &mut Option<i128>,
    null_count: &mut u128,
) {
    for row in 0..values.len() {
        if values.is_null(row) {
            *null_count = null_count.saturating_add(1);
            continue;
        }
        update_i128_range(values.value(row) as i128, min_value, max_value);
    }
}

pub(super) fn update_primitive_i64_range(
    values: &Int64Array,
    min_value: &mut Option<i128>,
    max_value: &mut Option<i128>,
    null_count: &mut u128,
) {
    for row in 0..values.len() {
        if values.is_null(row) {
            *null_count = null_count.saturating_add(1);
            continue;
        }
        update_i128_range(values.value(row) as i128, min_value, max_value);
    }
}

pub(super) fn update_i128_range(
    value: i128,
    min_value: &mut Option<i128>,
    max_value: &mut Option<i128>,
) {
    *min_value = Some(min_value.map_or(value, |current| current.min(value)));
    *max_value = Some(max_value.map_or(value, |current| current.max(value)));
}

pub(super) fn comma_join_single_table_filters(
    conjuncts: &[SqlExpr],
    aliases: &[String],
    alias_refs: &[&str],
    used_conjuncts: &mut [bool],
) -> Result<Vec<Option<FilterExpr>>> {
    let mut filters = vec![Vec::<SqlExpr>::new(); aliases.len()];
    for (index, conjunct) in conjuncts.iter().enumerate() {
        if expr_contains_materializable_subquery(conjunct) {
            continue;
        }
        let Some(alias) = single_table_conjunct_alias(conjunct, alias_refs)? else {
            continue;
        };
        let Some(table_index) = aliases
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(alias))
        else {
            continue;
        };
        if parse_filter(conjunct, &[], Some(alias), false).is_err() {
            continue;
        }
        filters[table_index].push(conjunct.clone());
        used_conjuncts[index] = true;
    }
    filters
        .into_iter()
        .zip(aliases)
        .map(|(filters, alias)| {
            let Some(expr) = combine_sql_and_conjuncts(filters) else {
                return Ok(None);
            };
            parse_filter(&expr, &[], Some(alias), false).map(Some)
        })
        .collect()
}

pub(super) fn comma_join_scan_projections(
    conjuncts: &[SqlExpr],
    aliases: &[String],
    alias_refs: &[&str],
    group_by: &[String],
    projection: &ParsedProjection,
    having: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
) -> Result<Vec<Projection>> {
    if conjuncts.iter().any(expr_contains_materializable_subquery) {
        return Ok(vec![Projection::All; aliases.len()]);
    }

    let mut columns = vec![Vec::<String>::new(); aliases.len()];
    for conjunct in conjuncts {
        add_comma_join_expr_columns(&mut columns, conjunct, aliases, alias_refs)?;
    }
    for column in group_by {
        add_comma_join_column(&mut columns, column, aliases)?;
    }
    if let Projection::Columns(projected) = &projection.projection {
        for column in projected {
            add_comma_join_column(&mut columns, column, aliases)?;
        }
    } else {
        return Ok(vec![Projection::All; aliases.len()]);
    }
    for aggregate in &projection.aggregates {
        if let Some(column) = aggregate.referenced_column() {
            add_comma_join_column(&mut columns, column, aliases)?;
        }
    }
    for expression in &projection.aggregate_expressions {
        for column in join_scalar_expression_columns(&expression.expr, alias_refs)? {
            add_comma_join_column(&mut columns, &column, aliases)?;
        }
    }
    for expression in &projection.expressions {
        for column in join_scalar_expression_columns(&expression.expr, alias_refs)? {
            add_comma_join_column(&mut columns, &column, aliases)?;
        }
    }
    if let Some(having) = having {
        for column in having.referenced_columns() {
            add_comma_join_column(&mut columns, &column, aliases)?;
        }
    }
    if let Some(order_by) = order_by {
        for sort in &order_by.expressions {
            add_comma_join_column(&mut columns, &sort.column, aliases)?;
        }
    }

    Ok(columns
        .into_iter()
        .map(|columns| {
            if columns.is_empty() {
                Projection::All
            } else {
                Projection::Columns(columns)
            }
        })
        .collect())
}

pub(super) fn comma_join_final_columns(
    alias_refs: &[&str],
    group_by: &[String],
    projection: &ParsedProjection,
    having: Option<&FilterExpr>,
    order_by: Option<&SortKey>,
) -> Result<HashSet<String>> {
    let mut columns = HashSet::new();
    for column in group_by {
        columns.insert(column.clone());
    }
    if let Projection::Columns(projected) = &projection.projection {
        columns.extend(projected.iter().cloned());
    }
    for aggregate in &projection.aggregates {
        if let Some(column) = aggregate.referenced_column() {
            columns.insert(column.to_string());
        }
    }
    for expression in &projection.aggregate_expressions {
        columns.extend(join_scalar_expression_columns(
            &expression.expr,
            alias_refs,
        )?);
    }
    for expression in &projection.expressions {
        columns.extend(join_scalar_expression_columns(
            &expression.expr,
            alias_refs,
        )?);
    }
    if let Some(having) = having {
        columns.extend(having.referenced_columns());
    }
    if let Some(order_by) = order_by {
        columns.extend(order_by.expressions.iter().map(|sort| sort.column.clone()));
    }
    Ok(columns)
}

pub(super) fn prune_comma_join_current_columns(
    batches: Vec<RecordBatch>,
    joined_aliases: &[String],
    alias_refs: &[&str],
    conjuncts: &[SqlExpr],
    used_conjuncts: &[bool],
    final_columns: &HashSet<String>,
) -> Result<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(batches);
    }
    let mut needed = final_columns.clone();
    for (conjunct, used) in conjuncts.iter().zip(used_conjuncts) {
        if *used {
            continue;
        }
        if expr_contains_materializable_subquery(conjunct) {
            return Ok(batches);
        }
        let mut columns = Vec::new();
        collect_join_column_candidates(conjunct, alias_refs, &mut columns)?;
        needed.extend(columns);
    }
    let schema = batches[0].schema();
    let keep = schema
        .fields()
        .iter()
        .filter_map(|field| {
            let name = field.name();
            if comma_join_field_needed(name, joined_aliases, &needed) {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if keep.len() == schema.fields().len() {
        return Ok(batches);
    }
    if keep.is_empty() {
        return Ok(batches);
    }
    apply_output_projection(batches, &Projection::Columns(keep))
}

pub(super) fn comma_join_field_needed(
    field_name: &str,
    joined_aliases: &[String],
    needed: &HashSet<String>,
) -> bool {
    if needed.contains(field_name) {
        return true;
    }
    let Some((alias, column)) = field_name.split_once('.') else {
        return true;
    };
    if !joined_aliases
        .iter()
        .any(|joined| joined.eq_ignore_ascii_case(alias))
    {
        return true;
    }
    needed.contains(&format!("{alias}.{column}")) || needed.contains(column)
}

pub(super) fn add_comma_join_expr_columns(
    output: &mut [Vec<String>],
    expr: &SqlExpr,
    aliases: &[String],
    alias_refs: &[&str],
) -> Result<()> {
    let mut columns = Vec::new();
    collect_join_column_candidates(expr, alias_refs, &mut columns)?;
    for column in columns {
        add_comma_join_column(output, &column, aliases)?;
    }
    Ok(())
}

pub(super) fn add_comma_join_column(
    output: &mut [Vec<String>],
    qualified_column: &str,
    aliases: &[String],
) -> Result<()> {
    let Some((alias, column)) = qualified_column.split_once('.') else {
        return Ok(());
    };
    let Some(index) = aliases
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(alias))
    else {
        return Ok(());
    };
    add_column_once(&mut output[index], column.to_string());
    Ok(())
}

pub(super) fn single_table_conjunct_alias<'a>(
    expr: &SqlExpr,
    table_aliases: &'a [&str],
) -> Result<Option<&'a str>> {
    let mut columns = Vec::new();
    collect_join_column_candidates(expr, table_aliases, &mut columns)?;
    let mut owner: Option<&'a str> = None;
    for column in columns {
        let Some((alias, _)) = column.split_once('.') else {
            return Ok(None);
        };
        let Some(alias) = table_aliases
            .iter()
            .copied()
            .find(|candidate| candidate.eq_ignore_ascii_case(alias))
        else {
            return Ok(None);
        };
        if let Some(existing) = owner {
            if !existing.eq_ignore_ascii_case(alias) {
                return Ok(None);
            }
        } else {
            owner = Some(alias);
        }
    }
    Ok(owner)
}

pub(super) fn collect_join_column_candidates(
    expr: &SqlExpr,
    table_aliases: &[&str],
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_join_column_candidates(left, table_aliases, columns)?;
            collect_join_column_candidates(right, table_aliases, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
        }
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            for column in join_scalar_expression_columns(
                &parse_join_scalar_sql_expression(expr, table_aliases)?,
                table_aliases,
            )? {
                add_column_once(columns, column);
            }
        }
        SqlExpr::CompoundFieldAccess { .. } => {
            for column in join_scalar_expression_columns(
                &parse_join_scalar_sql_expression(expr, table_aliases)?,
                table_aliases,
            )? {
                add_column_once(columns, column);
            }
        }
        SqlExpr::Function(function) => {
            for arg in function_arg_exprs(function) {
                collect_join_column_candidates(arg, table_aliases, columns)?;
            }
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            if let Some(expr) = substring_from {
                collect_join_column_candidates(expr, table_aliases, columns)?;
            }
            if let Some(expr) = substring_for {
                collect_join_column_candidates(expr, table_aliases, columns)?;
            }
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            for item in list {
                collect_join_column_candidates(item, table_aliases, columns)?;
            }
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            collect_join_column_candidates(low, table_aliases, columns)?;
            collect_join_column_candidates(high, table_aliases, columns)?;
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            collect_join_column_candidates(expr, table_aliases, columns)?;
            collect_join_column_candidates(pattern, table_aliases, columns)?;
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(operand) = operand {
                collect_join_column_candidates(operand, table_aliases, columns)?;
            }
            for when in conditions {
                collect_join_column_candidates(&when.condition, table_aliases, columns)?;
                collect_join_column_candidates(&when.result, table_aliases, columns)?;
            }
            if let Some(else_result) = else_result {
                collect_join_column_candidates(else_result, table_aliases, columns)?;
            }
        }
        SqlExpr::Value(_) => {}
        SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } | SqlExpr::Subquery(_) => {}
        _ => {}
    }
    Ok(())
}

pub(super) async fn scan_table_for_comma_join(
    engine: &DodamEngine,
    table: &SqlTableRef,
    batch_size: usize,
    filter: Option<&FilterExpr>,
    projection: &Projection,
) -> Result<Vec<RecordBatch>> {
    let stream = engine
        .scan_parquet_batches(
            table.path.clone(),
            batch_size,
            None,
            projection.clone(),
            filter.cloned(),
        )
        .await?;
    collect_batches(stream)
}
