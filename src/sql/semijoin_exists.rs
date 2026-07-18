use super::*;
use crate::sql::semijoin::*;

pub(super) async fn try_execute_exists_subquery_sql(
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
    let Some((subquery, negated)) = top_level_exists_subquery(select.selection.as_ref()) else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return Err(DodamError::UnsupportedSql(
            "EXISTS subquery filters over JOIN inputs are not supported yet".to_string(),
        ));
    }
    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        return Err(DodamError::UnsupportedSql(
            "EXISTS subquery filters with aggregates are not supported yet".to_string(),
        ));
    }
    let distinct = parse_distinct(select)?;
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;
    let _offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let exists = !query_output_batches(
        Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await?,
    )?
    .is_empty();
    if exists == negated {
        return Ok(Some(QueryOutput::Scan {
            batches: Vec::new(),
        }));
    }

    let stream = if distinct {
        engine
            .scan_parquet_distinct_batches(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
                None,
                order_by,
            )
            .await?
    } else if let Some(order_by) = order_by {
        engine
            .scan_parquet_ordered_batches_by(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
                None,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(
                path.path,
                batch_size,
                limit,
                parsed_projection.projection,
                None,
            )
            .await?
    };
    let batches = rename_output_batches(collect_batches(stream)?, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

pub(super) fn top_level_exists_subquery(expr: Option<&SqlExpr>) -> Option<(&Query, bool)> {
    match expr? {
        SqlExpr::Exists { subquery, negated } => Some((subquery.as_ref(), *negated)),
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            top_level_exists_subquery(Some(expr)).map(|(query, negated)| (query, !negated))
        }
        SqlExpr::Nested(expr) => top_level_exists_subquery(Some(expr)),
        _ => None,
    }
}

fn exists_conjunct_subquery(expr: &SqlExpr) -> Option<(&Query, bool)> {
    top_level_exists_subquery(Some(expr))
}

pub(super) async fn try_execute_correlated_exists_semijoin_sql(
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
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1
        || select
            .from
            .first()
            .is_some_and(|table| !table.joins.is_empty())
    {
        return Ok(None);
    }

    let outer_path = parse_from(select)?;
    let outer_alias = table_ref_alias_or_name(&outer_path);
    let mut outer_conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut outer_conjuncts);
    let Some((exists_index, exists_subquery, exists_negated)) = outer_conjuncts
        .iter()
        .enumerate()
        .find_map(|(index, expr)| {
            exists_conjunct_subquery(expr).map(|(query, negated)| (index, query, negated))
        })
    else {
        return Ok(None);
    };

    let SetExpr::Select(inner_select) = exists_subquery.body.as_ref() else {
        return Ok(None);
    };
    reject_query_features(exists_subquery)?;
    reject_select_features(inner_select)?;
    if inner_select.from.len() != 1
        || inner_select
            .from
            .first()
            .is_some_and(|table| !table.joins.is_empty())
        || parse_distinct(inner_select)?
        || inner_select.having.is_some()
        || !parse_group_by(inner_select, None)?.is_empty()
    {
        return Ok(None);
    }

    let inner_path = parse_from(inner_select)?;
    let inner_alias = table_ref_alias_or_name(&inner_path);
    let mut inner_conjuncts = Vec::new();
    let Some(inner_selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    collect_sql_and_conjuncts(inner_selection, &mut inner_conjuncts);
    let Some((join_index, inner_key, outer_key)) =
        semijoin_exists_key_pair(&inner_conjuncts, &inner_alias, &outer_alias)?
    else {
        return Ok(None);
    };

    let inner_residual = inner_conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != join_index).then_some(conjunct))
        .collect::<Vec<_>>();
    if inner_residual
        .iter()
        .any(predicate_requires_expression_path)
    {
        return Ok(None);
    }
    let inner_filter = combine_sql_and_conjuncts(inner_residual)
        .as_ref()
        .map(|expr| parse_filter(expr, &[], inner_path.alias.as_deref(), false))
        .transpose()?;
    let profile = semijoin_profile_enabled();
    let total_started = profile.then(Instant::now);
    let inner_started = profile.then(Instant::now);
    let inner_keys = collect_semijoin_key_set(
        engine,
        inner_path.path,
        &inner_key,
        inner_filter,
        batch_size,
    )
    .await?;
    let inner_elapsed = inner_started.map(|started| started.elapsed());

    let outer_residual = outer_conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != exists_index).then_some(conjunct))
        .collect::<Vec<_>>();
    let outer_residual = combine_sql_and_conjuncts(outer_residual);
    let (outer_filter, outer_expression_filters) =
        if let Some(outer_residual) = outer_residual.as_ref() {
            if predicate_requires_expression_path(outer_residual)
                || expr_contains_materializable_subquery(outer_residual)
            {
                split_subquery_and_expression_filters(
                    engine,
                    outer_residual,
                    outer_path.alias.as_deref(),
                    batch_size,
                )
                .await?
            } else {
                (
                    Some(parse_filter(
                        outer_residual,
                        &[],
                        outer_path.alias.as_deref(),
                        false,
                    )?),
                    Vec::new(),
                )
            }
        } else {
            (None, Vec::new())
        };
    let plan_elapsed = total_started.map(|started| started.elapsed());

    let group_by = parse_group_by(select, outer_path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, outer_path.alias.as_deref())?;
    let distinct = parse_distinct(select)?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        outer_path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;
    let _offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let mut outer_projection = semijoin_outer_projection(
        &parsed_projection,
        &group_by,
        order_by.as_ref(),
        &outer_key,
        outer_filter.as_ref(),
    );
    for expression_filter in &outer_expression_filters {
        add_projection_columns(
            &mut outer_projection,
            predicate_expression_columns(expression_filter, outer_path.alias.as_deref())?,
        );
    }
    let scan_started = profile.then(Instant::now);
    let stream = engine
        .scan_parquet_batches(
            outer_path.path,
            batch_size,
            None,
            outer_projection,
            outer_filter,
        )
        .await?;
    let mut outer_batches = collect_batches(stream)?;
    let scan_elapsed = scan_started.map(|started| started.elapsed());
    let expression_started = profile.then(Instant::now);
    for expression_filter in outer_expression_filters {
        outer_batches = apply_output_expression_filter(
            outer_batches,
            &expression_filter,
            outer_path.alias.as_deref(),
        )?;
    }
    let expression_elapsed = expression_started.map(|started| started.elapsed());
    let membership_started = profile.then(Instant::now);
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mut mask = semijoin_membership_mask(&batch, &outer_key, &inner_keys)?;
        if exists_negated {
            mask = invert_boolean_array(&mask);
        }
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    let membership_elapsed = membership_started.map(|started| started.elapsed());
    if let (true, Some(total_started)) = (profile, total_started) {
        let rows = filtered.iter().map(RecordBatch::num_rows).sum::<usize>();
        eprintln!(
            "[dodam:semijoin-profile] total={}ms plan={}ms inner={}ms outer_scan={}ms expr={}ms membership={}ms output_rows={} negated={}",
            total_started.elapsed().as_millis(),
            plan_elapsed.map(|elapsed| elapsed.as_millis()).unwrap_or(0),
            inner_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            scan_elapsed.map(|elapsed| elapsed.as_millis()).unwrap_or(0),
            expression_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            membership_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            rows,
            exists_negated,
        );
    }

    if !parsed_projection.aggregates.is_empty() {
        let stream = Box::new(MemoryExec::new(filtered)).execute()?;
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
        batches = apply_aggregate_output_order_limit(
            batches,
            order_by.as_ref(),
            limit,
            0,
            &metrics,
            &group_by,
        )?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit, 0)?;
    let projection_requires_expression =
        projection_requires_expression_path(&parsed_projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &parsed_projection.expressions)?
    } else {
        apply_output_projection(batches, &parsed_projection.projection)?
    };
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}
