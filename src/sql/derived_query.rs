use super::*;

pub(super) async fn try_execute_derived_join_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
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
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    let [join] = table.joins.as_slice() else {
        return Ok(None);
    };
    if !is_materialized_join_relation(&table.relation)
        && !is_materialized_join_relation(&join.relation)
    {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let distinct = parse_distinct(select)?;

    let left = materialize_join_relation(engine, &table.relation, batch_size, options).await?;
    let right = materialize_join_relation(engine, &join.relation, batch_size, options).await?;
    let left_alias = left.alias.clone();
    let right_alias = right.alias.clone();
    let (join_type, left_keys, right_keys, right_filter) =
        parse_join_condition(join, &left_alias, &right_alias)?;
    let derived_join_graph =
        build_logical_materialized_join_graph(&left, &right, &left_keys, &right_keys)?;
    let derived_join_plan = derived_join_graph.choose_best_plan();
    log_multi_input_join_optimizer_plan(
        "derived_join_order",
        &derived_join_graph,
        derived_join_plan.as_ref(),
    );
    let exec_left_alias = left_alias.clone();
    let exec_right_alias = right_alias.clone();
    let output_aliases = if join_type == JoinType::Semi {
        vec![left_alias.as_str()]
    } else {
        vec![left_alias.as_str(), right_alias.as_str()]
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let (filter, expression_filter) = parse_join_filter_plan(
        select.selection.as_ref(),
        &projection.aliases,
        &output_aliases,
        false,
    )?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
    let limit = parse_limit(query)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        order_by.as_ref(),
    )?;

    let stream = Box::new(HashJoinExec::new(
        Box::new(MemoryExec::new(left.batches)),
        Box::new(MemoryExec::new(apply_output_filter(
            right.batches,
            right_filter.as_ref(),
        )?)),
        left_keys,
        right_keys,
        exec_left_alias,
        exec_right_alias,
        choose_materialized_join_build_side(&derived_join_graph).unwrap_or(JoinBuildSide::Right),
        join_type,
        Projection::All,
    ))
    .execute()?;
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, filter.as_ref())?;
    if let Some(expression_filter) = expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(batches, expression_filter, &output_aliases)?;
    }
    if !projection.aggregates.is_empty() {
        let batches =
            append_aggregate_expression_columns(batches, &projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &projection.aggregates)?;
        let having = select
            .having
            .as_ref()
            .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
            .transpose()?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection_order_limit(
            batches,
            &projection.expressions,
            order_by.as_ref(),
            limit,
            0,
        )?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    if !projection_requires_expression {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

pub(super) struct MaterializedJoinRelation {
    pub(super) alias: String,
    pub(super) batches: Vec<RecordBatch>,
}

fn is_materialized_join_relation(relation: &TableFactor) -> bool {
    matches!(relation, TableFactor::Derived { .. })
}

pub(super) async fn materialize_join_relation(
    engine: &DodamEngine,
    relation: &TableFactor,
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<MaterializedJoinRelation> {
    match relation {
        TableFactor::Derived {
            lateral,
            subquery,
            alias,
            sample,
        } => {
            if *lateral || sample.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "LATERAL and TABLESAMPLE derived tables are not supported".to_string(),
                ));
            }
            let alias = alias.as_ref().ok_or_else(|| {
                DodamError::UnsupportedSql("derived tables in joins must have aliases".to_string())
            })?;
            if !alias.columns.is_empty() || alias.at.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "derived table column aliases and AT aliases are not supported".to_string(),
                ));
            }
            let output = Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await?;
            Ok(MaterializedJoinRelation {
                alias: alias.name.value.clone(),
                batches: query_output_batches(output)?,
            })
        }
        TableFactor::Table { .. } => {
            let table = parse_table_factor(relation)?;
            let alias = table_ref_alias_or_name(&table);
            let stream = engine
                .scan_parquet_batches(table.path, batch_size, None, Projection::All, None)
                .await?;
            Ok(MaterializedJoinRelation {
                alias,
                batches: collect_batches(stream)?,
            })
        }
        _ => Err(DodamError::UnsupportedSql(
            "derived table joins only support table and derived table inputs".to_string(),
        )),
    }
}

pub(super) fn build_logical_materialized_join_graph(
    left: &MaterializedJoinRelation,
    right: &MaterializedJoinRelation,
    left_keys: &[String],
    right_keys: &[String],
) -> Result<LogicalJoinGraph> {
    let left_rows = record_batch_rows(&left.batches).max(1);
    let right_rows = record_batch_rows(&right.batches).max(1);
    let tables = vec![
        logical_materialized_table_stats(&left.batches, left_rows, left_keys)?,
        logical_materialized_table_stats(&right.batches, right_rows, right_keys)?,
    ];
    let edges = left_keys
        .iter()
        .zip(right_keys)
        .map(|(left_key, right_key)| LogicalJoinEdge {
            left: 0,
            left_key: left_key.clone(),
            right: 1,
            right_key: right_key.clone(),
        })
        .collect();
    Ok(LogicalJoinGraph { tables, edges })
}

fn logical_materialized_table_stats(
    batches: &[RecordBatch],
    rows: usize,
    keys: &[String],
) -> Result<LogicalJoinTableStats> {
    let mut key_ndv = HashMap::new();
    let mut column_ranges = HashMap::new();
    for key in keys {
        key_ndv.insert(
            key.clone(),
            sampled_key_ndv(batches, std::slice::from_ref(key), 100_000)? as u128,
        );
        if let Some(range) = primitive_column_range_stats(batches, key)? {
            column_ranges.insert(key.clone(), range);
        }
    }
    Ok(LogicalJoinTableStats {
        base_rows: rows.max(1) as u128,
        rows: rows.max(1) as u128,
        row_width: estimated_batches_row_width(batches).max(1),
        key_ndv,
        column_ranges,
    })
}

fn choose_materialized_join_build_side(join_graph: &LogicalJoinGraph) -> Option<JoinBuildSide> {
    let left = join_graph.tables.first()?;
    let right = join_graph.tables.get(1)?;
    let left_cost = left.rows.saturating_mul(left.row_width.max(1));
    let right_cost = right.rows.saturating_mul(right.row_width.max(1));
    Some(if left_cost <= right_cost {
        JoinBuildSide::Left
    } else {
        JoinBuildSide::Right
    })
}

pub(super) async fn execute_parsed_join_query(
    engine: &DodamEngine,
    query: SqlQuery,
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<Option<QueryOutput>> {
    let Some(join) = query.join.clone() else {
        return Ok(None);
    };
    let is_aggregate = query.is_aggregate();
    let aggregates = query.aggregates.clone();
    let group_by = query.group_by.clone();
    let join_input_projection = if query.qualified_wildcards.is_empty() {
        join_input_projection_with_expression_filter(&query)?
    } else {
        Projection::All
    };
    let join_plan = plan_join_inputs(
        &join_input_projection,
        query.filter.as_ref(),
        query.order_by.as_ref(),
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    if let Some(join_graph) = build_logical_explicit_join_graph(engine, &query, &join, &join_plan)?
    {
        log_multi_input_join_optimizer_plan(
            "explicit_join_order",
            &join_graph,
            join_graph.choose_best_plan().as_ref(),
        );
    }
    if is_aggregate
        && let Some(output) = try_execute_join_coalesce_count_sum_aggregate(
            engine, &query, &join, &join_plan, batch_size,
        )
        .await?
    {
        return Ok(Some(output));
    }
    let output_projection = pushed_join_output_projection(&query)?;
    let output_projection_is_final = output_projection == query.projection;
    let stream = engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: query.path.clone(),
            right_path: join.right.path,
            batch_size,
            left_keys: join.left_keys,
            right_keys: join.right_keys,
            left_prefix: join.left_alias.clone(),
            right_prefix: join.right_alias.clone(),
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
            output_projection,
            join_memory_limit_bytes: join_memory_limit_bytes(options),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: join.join_type,
        })
        .await?;
    if is_aggregate {
        let stream = apply_output_filter_stream(stream, query.filter.clone());
        let stream: SendableBatchStream =
            if let Some(expression_filter) = query.expression_filter.as_ref() {
                Box::new(MemoryExec::new(apply_output_join_expression_filter(
                    collect_batches(stream)?,
                    expression_filter,
                    &[join.left_alias.as_str(), join.right_alias.as_str()],
                )?))
                .execute()?
            } else {
                stream
            };
        let metrics = collect_aggregates_with_optional_expression_views(
            stream,
            2,
            &group_by,
            &aggregates,
            &query.filtered_aggregates,
            &query.aggregate_expressions,
        )?;
        let mut batches = aggregate_metrics_to_batches(&metrics, &group_by, &aggregates)?;
        batches = apply_output_filter(batches, query.having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&query.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &query.expressions)?;
        }
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &query.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let mut batches = collect_batches(stream)?;
    batches = apply_output_filter(batches, query.filter.as_ref())?;
    if let Some(expression_filter) = query.expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(
            batches,
            expression_filter,
            &[join.left_alias.as_str(), join.right_alias.as_str()],
        )?;
    }
    let projection_requires_expression = projection_requires_expression_path(&query.expressions);
    if projection_requires_expression {
        batches = apply_output_expression_projection_order_limit(
            batches,
            &query.expressions,
            query.order_by.as_ref(),
            query.limit,
            query.offset,
        )?;
    } else {
        batches =
            apply_output_order_limit(batches, query.order_by.as_ref(), query.limit, query.offset)?;
        if !output_projection_is_final && query.qualified_wildcards.is_empty() {
            batches = apply_output_projection(batches, &query.projection)?;
        }
    }
    if query.distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    if !query.qualified_wildcards.is_empty() {
        batches = apply_qualified_wildcard_projection(
            batches,
            &query.qualified_wildcards,
            &query.projection,
        )?;
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &query.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

pub(super) fn build_logical_explicit_join_graph(
    engine: &DodamEngine,
    query: &SqlQuery,
    join: &SqlJoin,
    join_plan: &JoinInputPlan,
) -> Result<Option<LogicalJoinGraph>> {
    if join.left_keys.is_empty() || join.left_keys.len() != join.right_keys.len() {
        return Ok(None);
    }
    let left_rows = engine
        .parquet_total_row_count(&query.path)
        .unwrap_or(0)
        .max(1) as u128;
    let right_rows = engine
        .parquet_total_row_count(&join.right.path)
        .unwrap_or(0)
        .max(1) as u128;
    let left_key_ndv = join
        .left_keys
        .iter()
        .map(|key| (key.clone(), left_rows))
        .collect::<HashMap<_, _>>();
    let right_key_ndv = join
        .right_keys
        .iter()
        .map(|key| (key.clone(), right_rows))
        .collect::<HashMap<_, _>>();
    let edges = join
        .left_keys
        .iter()
        .zip(&join.right_keys)
        .map(|(left_key, right_key)| LogicalJoinEdge {
            left: 0,
            left_key: left_key.clone(),
            right: 1,
            right_key: right_key.clone(),
        })
        .collect();
    Ok(Some(LogicalJoinGraph {
        tables: vec![
            LogicalJoinTableStats {
                base_rows: left_rows,
                rows: left_rows,
                row_width: estimated_projection_width(&join_plan.left_projection),
                key_ndv: left_key_ndv,
                column_ranges: HashMap::new(),
            },
            LogicalJoinTableStats {
                base_rows: right_rows,
                rows: right_rows,
                row_width: estimated_projection_width(&join_plan.right_projection),
                key_ndv: right_key_ndv,
                column_ranges: HashMap::new(),
            },
        ],
        edges,
    }))
}

pub(super) fn estimated_projection_width(projection: &Projection) -> u128 {
    match projection {
        Projection::Columns(columns) => (columns.len() as u128).saturating_mul(16).max(1),
        Projection::All => 128,
    }
}

pub(super) async fn try_execute_derived_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
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
    let Some((subquery, alias)) = parse_derived_from(select)? else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;

    let distinct = parse_distinct(select)?;
    let inner_has_materializable_subquery = match subquery.body.as_ref() {
        SetExpr::Select(inner_select) => inner_select
            .selection
            .as_ref()
            .is_some_and(expr_contains_materializable_subquery),
        _ => false,
    };
    let parsed_inner = match parse_query(subquery) {
        Ok(query) => Some(query),
        Err(DodamError::UnsupportedSql(_))
        | Err(DodamError::UnknownColumn(_))
        | Err(DodamError::UnknownTableQualifier(_)) => None,
        Err(error) => return Err(error),
    };
    let inner_output = if let Some(parsed_inner) = parsed_inner {
        if !inner_has_materializable_subquery
            && let Some(output) =
                execute_parsed_join_query(engine, parsed_inner, batch_size, options).await?
        {
            output
        } else {
            Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await?
        }
    } else {
        Box::pin(execute_sql_with_options(
            engine,
            &subquery.to_string(),
            batch_size,
            options,
        ))
        .await?
    };
    let group_by = parse_group_by(select, Some(&alias))?;
    let parsed_projection = parse_projection(select, &group_by, Some(&alias))?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_filter(expr, &[], Some(&alias), false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        Some(&alias),
    )?;
    let limit = parse_limit(query)?;

    if let QueryOutput::Aggregate {
        metrics: inner_metrics,
        batches: inner_batches,
    } = &inner_output
    {
        if let Some(output) = try_count_derived_aggregate_groups(
            inner_metrics,
            inner_batches,
            &group_by,
            &parsed_projection,
            filter.as_ref(),
            having.as_ref(),
            order_by.as_ref(),
            limit,
        )? {
            return Ok(Some(output));
        }
    }

    let inner_batches = query_output_batches(inner_output)?;
    if !parsed_projection.aggregates.is_empty() {
        let mut filtered_batches = apply_output_filter(inner_batches, filter.as_ref())?;
        filtered_batches = append_aggregate_expression_columns(
            filtered_batches,
            &parsed_projection.aggregate_expressions,
        )?;
        let stream = Box::new(MemoryExec::new(filtered_batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &parsed_projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &parsed_projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions =
            projection_requires_expression_path(&parsed_projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &parsed_projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;
    let mut batches = apply_output_filter(inner_batches, filter.as_ref())?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        apply_output_projection(batches, &parsed_projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

pub(super) async fn try_execute_multi_comma_join_sql(
    engine: &DodamEngine,
    sql: &str,
    batch_size: usize,
    options: SqlExecutionOptions,
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
    let Some((tables, mut conjuncts)) = parse_multi_input_join_table_refs_and_conjuncts(select)?
    else {
        return Ok(None);
    };
    if tables.len() <= 2 {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    let aliases = tables
        .iter()
        .map(table_ref_alias_or_name)
        .collect::<Vec<_>>();
    let alias_refs = aliases.iter().map(String::as_str).collect::<Vec<_>>();
    if let Some(selection) = select.selection.as_ref() {
        collect_sql_and_conjuncts(selection, &mut conjuncts);
    }
    if conjuncts.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "multi-input join requires an equality predicate".to_string(),
        ));
    }

    let mut used_conjuncts = vec![false; conjuncts.len()];
    let scan_filters =
        comma_join_single_table_filters(&conjuncts, &aliases, &alias_refs, &mut used_conjuncts)?;
    let group_by = parse_join_group_by(select, &alias_refs)?;
    let projection = parse_join_projection(select, &alias_refs, &group_by)?;
    let distinct = parse_distinct(select)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        None,
    )?;
    let rewritten_having = if let Some(expr) = select.having.as_ref() {
        Some(
            Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                engine,
                expr.clone(),
                batch_size,
            ))
            .await?,
        )
    } else {
        None
    };
    let having = rewritten_having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &alias_refs, true))
        .transpose()?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &alias_refs,
    )?;
    let limit = parse_limit(query)?;
    let memory_limit_bytes = join_memory_limit_bytes(options);
    let lookup_fusion_plan = if join_aggregate_lookup_fusion_disabled() {
        None
    } else {
        plan_join_aggregate_lookup_fusion(
            tables.len(),
            &aliases,
            &alias_refs,
            &conjuncts,
            &used_conjuncts,
            &group_by,
            &projection,
            distinct,
            having.as_ref(),
        )?
    };
    let scan_projections = comma_join_scan_projections(
        &conjuncts,
        &aliases,
        &alias_refs,
        &group_by,
        &projection,
        having.as_ref(),
        order_by.as_ref(),
    )?;
    let final_columns = comma_join_final_columns(
        &alias_refs,
        &group_by,
        &projection,
        having.as_ref(),
        order_by.as_ref(),
    )?;
    if let Some(lookup_fusion_plan) = lookup_fusion_plan.as_ref() {
        let metadata_graph = build_logical_multi_join_graph_from_metadata(
            engine,
            &tables,
            &aliases,
            &alias_refs,
            &conjuncts,
            &scan_projections,
        )?;
        let graph_plan = metadata_graph.choose_best_plan();
        log_multi_input_join_optimizer_plan(
            "join_aggregate_lookup_fusion_graph",
            &metadata_graph,
            graph_plan.as_ref(),
        );
        if choose_join_aggregate_lookup_fusion(
            lookup_fusion_plan,
            &metadata_graph,
            graph_plan.as_ref(),
        ) {
            if let Some(output) = execute_join_aggregate_lookup_fusion(
                engine,
                &tables,
                &scan_filters,
                &group_by,
                &projection,
                order_by.as_ref(),
                limit,
                batch_size,
                lookup_fusion_plan,
            )
            .await?
            {
                return Ok(Some(output));
            }
        }
    }
    let mut scanned = Vec::with_capacity(tables.len());
    let mut row_counts = Vec::with_capacity(tables.len());
    let mut base_row_counts = Vec::with_capacity(tables.len());
    for index in 0..tables.len() {
        base_row_counts.push(
            engine
                .parquet_total_row_count(&tables[index].path)
                .unwrap_or_else(|_| 0),
        );
        let batches = scan_table_for_comma_join(
            engine,
            &tables[index],
            batch_size,
            scan_filters[index].as_ref(),
            &scan_projections[index],
        )
        .await?;
        row_counts.push(record_batch_rows(&batches));
        scanned.push(Some(batches));
    }
    let join_graph = build_logical_comma_join_graph(
        &scanned,
        &row_counts,
        &base_row_counts,
        &aliases,
        &alias_refs,
        &conjuncts,
    )?;
    let join_plan = join_graph.choose_best_plan();
    log_multi_input_join_optimizer_plan("multi_input_join_order", &join_graph, join_plan.as_ref());
    let current = if let Some(tree) =
        choose_bushy_multi_input_join_execution_tree(&join_graph, join_plan.as_ref())
    {
        execute_bushy_comma_join_tree(
            &tree,
            &mut scanned,
            &row_counts,
            &aliases,
            &alias_refs,
            &conjuncts,
            &mut used_conjuncts,
            &final_columns,
            memory_limit_bytes,
        )?
        .batches
    } else {
        execute_left_deep_comma_join(
            scanned,
            &row_counts,
            &aliases,
            &alias_refs,
            &conjuncts,
            &mut used_conjuncts,
            &final_columns,
            &join_graph,
            join_plan.as_ref(),
            memory_limit_bytes,
        )?
    };

    let residual = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (!used_conjuncts[index]).then_some(conjunct))
        .collect::<Vec<_>>();
    let residual = combine_sql_and_conjuncts(residual);
    let (filter_residual, subquery_residual) = split_subquery_residual(residual);
    let (filter, expression_filter) = parse_join_filter_plan(
        filter_residual.as_ref(),
        &projection.aliases,
        &alias_refs,
        false,
    )?;

    let mut batches = apply_output_filter(current, filter.as_ref())?;
    if let Some(expression_filter) = expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(batches, expression_filter, &alias_refs)?;
    }
    if let Some(residual) = subquery_residual.as_ref() {
        if let Some(optimized) =
            try_apply_correlated_min_equality_filter(engine, batches.clone(), residual, batch_size)
                .await?
        {
            batches = optimized;
        } else {
            batches = apply_correlated_subquery_filter_batches(
                engine,
                batches,
                &residual.to_string(),
                batch_size,
            )
            .await?;
        }
    }
    if !projection.aggregates.is_empty() {
        batches = append_aggregate_expression_columns(batches, &projection.aggregate_expressions)?;
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 1, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 1, &group_by, &projection.aggregates)?
        };
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions = projection_requires_expression_path(&projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }
    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection_order_limit(
            batches,
            &projection.expressions,
            order_by.as_ref(),
            limit,
            0,
        )?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    if !projection_requires_expression {
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
    }
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}
