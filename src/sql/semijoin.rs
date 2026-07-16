use super::*;

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

pub(super) async fn try_execute_correlated_in_pair_semijoin_sql(
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
    let Some((in_index, outer_value, in_subquery)) =
        outer_conjuncts
            .iter()
            .enumerate()
            .find_map(|(index, expr)| match expr {
                SqlExpr::InSubquery {
                    expr,
                    subquery,
                    negated: false,
                } => Some((index, expr.as_ref(), subquery.as_ref())),
                SqlExpr::Nested(expr) => match expr.as_ref() {
                    SqlExpr::InSubquery {
                        expr,
                        subquery,
                        negated: false,
                    } => Some((index, expr.as_ref(), subquery.as_ref())),
                    _ => None,
                },
                _ => None,
            })
    else {
        return Ok(None);
    };
    let Some(outer_value_column) = semijoin_column_name(outer_value)? else {
        return Ok(None);
    };
    if semijoin_column_owner(&outer_value_column, "", &outer_alias)
        != Some(SemijoinColumnOwner::Outer)
    {
        return Ok(None);
    }
    let outer_value_column = unqualified_semijoin_column(&outer_value_column);

    let SetExpr::Select(inner_select) = in_subquery.body.as_ref() else {
        return Ok(None);
    };
    reject_query_features(in_subquery)?;
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
    let [SelectItem::UnnamedExpr(inner_value_expr)] = inner_select.projection.as_slice() else {
        return Ok(None);
    };
    let Some(inner_value_column) = semijoin_column_name(inner_value_expr)? else {
        return Ok(None);
    };
    let inner_path = parse_from(inner_select)?;
    let inner_alias = table_ref_alias_or_name(&inner_path);
    if semijoin_column_owner(&inner_value_column, &inner_alias, &outer_alias)
        != Some(SemijoinColumnOwner::Inner)
    {
        return Ok(None);
    }
    let inner_value_column = unqualified_semijoin_column(&inner_value_column);

    let Some(inner_selection) = inner_select.selection.as_ref() else {
        return Ok(None);
    };
    let mut inner_conjuncts = Vec::new();
    collect_sql_and_conjuncts(inner_selection, &mut inner_conjuncts);
    let Some((join_index, inner_corr_key, outer_corr_key)) =
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
    let inner_pairs = match collect_semijoin_i64_pair_set(
        engine,
        inner_path.path,
        &inner_corr_key,
        &inner_value_column,
        inner_filter,
        batch_size,
    )
    .await
    {
        Ok(values) => values,
        Err(DodamError::UnsupportedSql(message)) if message.contains("integer semijoin key") => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };

    let outer_residual = outer_conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != in_index).then_some(conjunct))
        .collect::<Vec<_>>();
    if outer_residual
        .iter()
        .any(predicate_requires_expression_path)
    {
        return Ok(None);
    }
    let outer_filter = combine_sql_and_conjuncts(outer_residual)
        .as_ref()
        .map(|expr| parse_filter(expr, &[], outer_path.alias.as_deref(), false))
        .transpose()?;
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

    let mut outer_projection = match semijoin_outer_projection(
        &parsed_projection,
        &group_by,
        order_by.as_ref(),
        &outer_corr_key,
        outer_filter.as_ref(),
    ) {
        Projection::All => Projection::All,
        Projection::Columns(mut columns) => {
            add_column_once(&mut columns, outer_value_column.clone());
            Projection::Columns(columns)
        }
    };
    if matches!(parsed_projection.projection, Projection::All) {
        outer_projection = Projection::All;
    }
    let stream = engine
        .scan_parquet_batches(
            outer_path.path,
            batch_size,
            None,
            outer_projection,
            outer_filter,
        )
        .await?;
    let outer_batches = collect_batches(stream)?;
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mask = match semijoin_i64_pair_mask(
            &batch,
            &outer_corr_key,
            &outer_value_column,
            &inner_pairs,
            false,
        ) {
            Ok(mask) => mask,
            Err(DodamError::UnsupportedSql(message))
                if message.contains("integer semijoin key") =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
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

fn semijoin_exists_key_pair(
    conjuncts: &[SqlExpr],
    inner_alias: &str,
    outer_alias: &str,
) -> Result<Option<(usize, String, String)>> {
    for (index, conjunct) in conjuncts.iter().enumerate() {
        let SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = conjunct
        else {
            continue;
        };
        let Some(left_column) = semijoin_column_name(left)? else {
            continue;
        };
        let Some(right_column) = semijoin_column_name(right)? else {
            continue;
        };
        let left_owner = semijoin_column_owner(&left_column, inner_alias, outer_alias);
        let right_owner = semijoin_column_owner(&right_column, inner_alias, outer_alias);
        match (left_owner, right_owner) {
            (Some(SemijoinColumnOwner::Inner), Some(SemijoinColumnOwner::Outer)) => {
                return Ok(Some((
                    index,
                    unqualified_semijoin_column(&left_column),
                    unqualified_semijoin_column(&right_column),
                )));
            }
            (Some(SemijoinColumnOwner::Outer), Some(SemijoinColumnOwner::Inner)) => {
                return Ok(Some((
                    index,
                    unqualified_semijoin_column(&right_column),
                    unqualified_semijoin_column(&left_column),
                )));
            }
            _ => {}
        }
    }
    Ok(None)
}

fn tuple_semijoin_columns(
    expr: &SqlExpr,
    inner_alias: &str,
    outer_alias: &str,
    expected_owner: SemijoinColumnOwner,
) -> Result<Option<(String, String)>> {
    if let SqlExpr::Nested(expr) = expr {
        return tuple_semijoin_columns(expr, inner_alias, outer_alias, expected_owner);
    }
    let SqlExpr::Tuple(values) = expr else {
        return Ok(None);
    };
    let [left, right] = values.as_slice() else {
        return Ok(None);
    };
    let Some(left) = semijoin_column_name(left)? else {
        return Ok(None);
    };
    let Some(right) = semijoin_column_name(right)? else {
        return Ok(None);
    };
    if semijoin_column_owner_in_scope(&left, inner_alias, outer_alias, expected_owner)
        != Some(expected_owner)
        || semijoin_column_owner_in_scope(&right, inner_alias, outer_alias, expected_owner)
            != Some(expected_owner)
    {
        return Ok(None);
    }
    Ok(Some((
        unqualified_semijoin_column(&left),
        unqualified_semijoin_column(&right),
    )))
}

fn projection_pair_semijoin_columns(
    select: &Select,
    inner_alias: &str,
    outer_alias: &str,
    expected_owner: SemijoinColumnOwner,
) -> Result<Option<(String, String)>> {
    let [
        SelectItem::UnnamedExpr(left),
        SelectItem::UnnamedExpr(right),
    ] = select.projection.as_slice()
    else {
        return Ok(None);
    };
    let Some(left) = semijoin_column_name(left)? else {
        return Ok(None);
    };
    let Some(right) = semijoin_column_name(right)? else {
        return Ok(None);
    };
    if semijoin_column_owner_in_scope(&left, inner_alias, outer_alias, expected_owner)
        != Some(expected_owner)
        || semijoin_column_owner_in_scope(&right, inner_alias, outer_alias, expected_owner)
            != Some(expected_owner)
    {
        return Ok(None);
    }
    Ok(Some((
        unqualified_semijoin_column(&left),
        unqualified_semijoin_column(&right),
    )))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SemijoinColumnOwner {
    Inner,
    Outer,
}

fn semijoin_column_owner_in_scope(
    column: &str,
    inner_alias: &str,
    outer_alias: &str,
    default_owner: SemijoinColumnOwner,
) -> Option<SemijoinColumnOwner> {
    semijoin_column_owner(column, inner_alias, outer_alias)
        .or_else(|| (!column.contains('.')).then_some(default_owner))
}

pub(super) fn semijoin_column_name(expr: &SqlExpr) -> Result<Option<String>> {
    ColumnResolver::raw_column(expr)
}

fn semijoin_column_owner(
    column: &str,
    inner_alias: &str,
    outer_alias: &str,
) -> Option<SemijoinColumnOwner> {
    if column.starts_with(&format!("{inner_alias}.")) {
        return Some(SemijoinColumnOwner::Inner);
    }
    if column.starts_with(&format!("{outer_alias}.")) {
        return Some(SemijoinColumnOwner::Outer);
    }
    let unqualified = unqualified_semijoin_column(column);
    if unqualified.starts_with(&semijoin_unqualified_prefix(inner_alias)?) {
        return Some(SemijoinColumnOwner::Inner);
    }
    if unqualified.starts_with(&semijoin_unqualified_prefix(outer_alias)?) {
        return Some(SemijoinColumnOwner::Outer);
    }
    None
}

fn semijoin_unqualified_prefix(alias: &str) -> Option<String> {
    Some(format!("{}_", alias.chars().next()?))
}

pub(super) fn unqualified_semijoin_column(column: &str) -> String {
    column
        .rsplit_once('.')
        .map(|(_, column)| column)
        .unwrap_or(column)
        .to_string()
}

async fn collect_semijoin_key_set(
    engine: &DodamEngine,
    path: PathBuf,
    key_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<SemijoinKeySet> {
    let mut projection = vec![key_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let batches = collect_batches(stream)?;
    let Some(first_batch) = batches.first() else {
        return Ok(SemijoinKeySet::rhs_empty());
    };
    let column_index = batch_column_index(first_batch, key_column)?;
    let mut keys = SemijoinKeySet::for_data_type(first_batch.column(column_index).data_type());
    for batch in &batches {
        let column_index = batch_column_index(batch, key_column)?;
        let column = batch.column(column_index);
        keys.insert_from_array(column)?;
    }
    Ok(keys)
}

fn semijoin_membership_mask(
    batch: &RecordBatch,
    key_column: &str,
    keys: &SemijoinKeySet,
) -> Result<BooleanArray> {
    let column_index = batch_column_index(batch, key_column)?;
    let column = batch.column(column_index);
    keys.membership_mask(column)
}

fn semijoin_anti_membership_mask(
    batch: &RecordBatch,
    key_column: &str,
    keys: &SemijoinKeySet,
) -> Result<BooleanArray> {
    let column_index = batch_column_index(batch, key_column)?;
    let column = batch.column(column_index);
    keys.anti_membership_mask(column)
}

fn collect_semijoin_filtered_limit_batches(
    mut stream: SendableBatchStream,
    key_column: &str,
    keys: &SemijoinKeySet,
    negated: bool,
    limit: usize,
) -> Result<Vec<RecordBatch>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::new();
    let mut remaining = limit;
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let mask = if negated {
            semijoin_anti_membership_mask(&batch, key_column, keys)?
        } else {
            semijoin_membership_mask(&batch, key_column, keys)?
        };
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() == 0 {
            continue;
        }
        let rows = remaining.min(batch.num_rows());
        output.push(batch.slice(0, rows));
        remaining -= rows;
        if remaining == 0 {
            break;
        }
    }
    Ok(output)
}

fn collect_semijoin_i64_pair_filtered_limit_batches(
    mut stream: SendableBatchStream,
    left_column: &str,
    right_column: &str,
    keys: &TupleSemijoinPairSet,
    pre_filter: Option<&FilterExpr>,
    negated: bool,
    limit: usize,
) -> Result<Vec<RecordBatch>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::new();
    let mut remaining = limit;
    for batch in stream.by_ref() {
        let mut batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        if let Some(pre_filter) = pre_filter {
            let mask = evaluate_filter_mask(&batch, pre_filter)?;
            batch = filter_record_batch(&batch, &mask)?;
            if batch.num_rows() == 0 {
                continue;
            }
        }
        let mask = tuple_semijoin_pair_mask(&batch, left_column, right_column, keys, negated)?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() == 0 {
            continue;
        }
        let rows = remaining.min(batch.num_rows());
        output.push(batch.slice(0, rows));
        remaining -= rows;
        if remaining == 0 {
            break;
        }
    }
    Ok(output)
}

fn same_source_tuple_semijoin_transferred_filter(
    engine: &DodamEngine,
    outer_path: &SqlTableRef,
    inner_path: &SqlTableRef,
    outer_left: &str,
    outer_right: &str,
    inner_left: &str,
    inner_right: &str,
    outer_filter: Option<FilterExpr>,
    inner_filter: Option<FilterExpr>,
) -> Result<Option<FilterExpr>> {
    if outer_path.path != inner_path.path || outer_left != inner_left || outer_right != inner_right
    {
        return Ok(None);
    }
    let Some(inner_filter) = inner_filter else {
        return Ok(None);
    };
    let key_columns = [outer_left, outer_right];
    let inner_filter_columns = inner_filter.referenced_columns();
    let key_only_predicate = inner_filter_columns
        .iter()
        .all(|column| key_columns.iter().any(|key| key == column));
    if key_only_predicate {
        if expr_contains_like(inner_filter.expr()) {
            return Ok(None);
        }
    } else if !same_source_tuple_semijoin_has_unique_column(engine, &outer_path.path, key_columns)?
    {
        return Ok(None);
    }
    Ok(Some(match outer_filter {
        Some(outer_filter) => FilterExpr::new(Expr::And(
            Box::new(outer_filter.expr().clone()),
            Box::new(inner_filter.expr().clone()),
        )),
        None => inner_filter,
    }))
}

fn same_source_tuple_semijoin_has_unique_column(
    engine: &DodamEngine,
    path: &Path,
    columns: [&str; 2],
) -> Result<bool> {
    for column in columns {
        if same_source_tuple_semijoin_unique_column(engine, path, column)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn same_source_tuple_semijoin_unique_column(
    engine: &DodamEngine,
    path: &Path,
    column: &str,
) -> Result<bool> {
    let Some(mut ranges) = engine.parquet_primitive_column_min_max_by_row_group(path, column)?
    else {
        return Ok(false);
    };
    if ranges.is_empty() {
        return Ok(false);
    }
    ranges.sort_unstable_by(|left, right| {
        left.min
            .cmp(&right.min)
            .then_with(|| left.max.cmp(&right.max))
            .then_with(|| left.row_group.cmp(&right.row_group))
    });
    let mut previous_max = None::<i128>;
    let mut total_rows = 0u128;
    for range in &ranges {
        if range.null_count != Some(0) {
            return Ok(false);
        }
        let span = range
            .max
            .checked_sub(range.min)
            .and_then(|value| value.checked_add(1));
        if span != Some(range.rows as i128) {
            return Ok(false);
        }
        if previous_max.is_some_and(|max| range.min <= max) {
            return Ok(false);
        }
        previous_max = Some(range.max);
        total_rows = total_rows.saturating_add(range.rows as u128);
    }
    Ok(total_rows > 0)
}

fn expr_contains_like(expr: &Expr) -> bool {
    match expr {
        Expr::Like { .. } => true,
        Expr::Not(expr) => expr_contains_like(expr),
        Expr::And(left, right) | Expr::Or(left, right) => {
            expr_contains_like(left) || expr_contains_like(right)
        }
        Expr::Boolean(_)
        | Expr::Comparison(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::IsNull { .. } => false,
    }
}

enum TupleSemijoinPairSet {
    I64(SemijoinI64PairKeys),
    I64Utf8(SemijoinI64Utf8PairKeys),
    Literal(SemijoinLiteralPairKeys),
}

impl TupleSemijoinPairSet {
    fn has_null(&self) -> bool {
        match self {
            Self::I64(keys) => keys.has_null,
            Self::I64Utf8(keys) => keys.has_null,
            Self::Literal(keys) => keys.has_null,
        }
    }
}

struct SemijoinI64Utf8PairKeys {
    values: SemijoinI64Utf8PairValues,
    numeric_values: SemijoinNumericValues,
    string_ids: FastHashMap<Vec<u8>, u32>,
    numeric_left: bool,
    has_null: bool,
}

struct SemijoinMixedPrefixCandidateKeys {
    numeric_values: SemijoinNumericValues,
    string_ids: FastHashMap<Vec<u8>, u32>,
    numeric_left: bool,
}

enum SemijoinNumericValues {
    I64(FastHashSet<i64>),
    I32(FastHashSet<i32>),
    I32Dense {
        min: i32,
        contains: Vec<u8>,
        len: usize,
    },
}

impl SemijoinNumericValues {
    fn empty_i64() -> Self {
        Self::I64(FastHashSet::default())
    }

    fn empty_i32() -> Self {
        Self::I32(FastHashSet::default())
    }

    fn reserve(&mut self, additional: usize) {
        match self {
            Self::I64(values) => values.reserve(additional),
            Self::I32(values) => values.reserve(additional),
            Self::I32Dense { .. } => {}
        }
    }

    fn insert_i32(&mut self, value: i32) {
        match self {
            Self::I32(values) => {
                values.insert(value);
            }
            Self::I64(values) => {
                values.insert(i64::from(value));
            }
            Self::I32Dense { .. } => {}
        }
    }

    fn insert_i64(&mut self, value: i64) {
        match self {
            Self::I64(values) => {
                values.insert(value);
            }
            Self::I32(values) => {
                if let Ok(value) = i32::try_from(value) {
                    values.insert(value);
                } else {
                    let mut wide = FastHashSet::<i64>::with_capacity_and_hasher(
                        values.len().saturating_add(1),
                        Default::default(),
                    );
                    wide.extend(values.drain().map(i64::from));
                    wide.insert(value);
                    *self = Self::I64(wide);
                }
            }
            Self::I32Dense { .. } => {}
        }
    }

    fn contains_i32(&self, value: i32) -> bool {
        match self {
            Self::I64(values) => values.contains(&i64::from(value)),
            Self::I32(values) => values.contains(&value),
            Self::I32Dense { min, contains, .. } => value
                .checked_sub(*min)
                .and_then(|offset| usize::try_from(offset).ok())
                .is_some_and(|index| contains.get(index).is_some_and(|byte| *byte != 0)),
        }
    }

    fn contains_i64(&self, value: i64) -> bool {
        match self {
            Self::I64(values) => values.contains(&value),
            Self::I32(values) => i32::try_from(value).is_ok_and(|value| values.contains(&value)),
            Self::I32Dense { .. } => {
                i32::try_from(value).is_ok_and(|value| self.contains_i32(value))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I32Dense { len, .. } => *len,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::I64(_) => "i64-hash",
            Self::I32(_) => "i32-hash",
            Self::I32Dense { .. } => "i32-dense",
        }
    }

    fn optimize_dense_i32(self) -> Self {
        let Self::I32(values) = self else {
            return self;
        };
        let len = values.len();
        if len == 0 {
            return Self::I32(values);
        }
        let Some(min) = values.iter().copied().min() else {
            return Self::I32(values);
        };
        let Some(max) = values.iter().copied().max() else {
            return Self::I32(values);
        };
        let span = i64::from(max) - i64::from(min) + 1;
        if span <= 0 {
            return Self::I32(values);
        }
        let Ok(span) = usize::try_from(span) else {
            return Self::I32(values);
        };
        let max_span = len.saturating_mul(mixed_tuple_dense_numeric_max_span_factor());
        if span > max_span || span > mixed_tuple_dense_numeric_max_bytes() {
            return Self::I32(values);
        }
        let mut contains = vec![0u8; span];
        for value in values {
            let offset = i64::from(value) - i64::from(min);
            if let Ok(index) = usize::try_from(offset)
                && let Some(slot) = contains.get_mut(index)
            {
                *slot = 1;
            }
        }
        Self::I32Dense { min, contains, len }
    }
}

fn mixed_tuple_dense_numeric_max_span_factor() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_NUMERIC_MAX_SPAN_FACTOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

fn mixed_tuple_dense_numeric_max_bytes() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_NUMERIC_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024)
}

enum SemijoinI64Utf8PairValues {
    Pair(FastHashSet<(i64, u32)>),
    PackedI32U32(FastHashSet<u64>),
    DenseI32ToU32 {
        min: i32,
        values: Vec<u32>,
        present: Vec<u8>,
        len: usize,
    },
}

impl SemijoinI64Utf8PairValues {
    fn empty_packed() -> Self {
        Self::PackedI32U32(FastHashSet::default())
    }

    fn reserve(&mut self, additional: usize) {
        match self {
            Self::Pair(values) => values.reserve(additional),
            Self::PackedI32U32(values) => values.reserve(additional),
            Self::DenseI32ToU32 { .. } => {}
        }
    }

    fn insert(&mut self, number: i64, string_id: u32) {
        match self {
            Self::PackedI32U32(values) => {
                if let Some(key) = pack_i32_u32_pair(number, string_id) {
                    values.insert(key);
                } else {
                    let mut pair_values = FastHashSet::with_capacity_and_hasher(
                        values.len().saturating_add(1),
                        Default::default(),
                    );
                    for key in values.drain() {
                        let (number, string_id) = unpack_i32_u32_pair(key);
                        pair_values.insert((number, string_id));
                    }
                    pair_values.insert((number, string_id));
                    *self = Self::Pair(pair_values);
                }
            }
            Self::Pair(values) => {
                values.insert((number, string_id));
            }
            Self::DenseI32ToU32 { .. } => {
                let mut values = FastHashSet::with_capacity_and_hasher(1, Default::default());
                values.insert((number, string_id));
                *self = Self::Pair(values);
            }
        }
    }

    fn contains(&self, number: i64, string_id: u32) -> bool {
        match self {
            Self::PackedI32U32(values) => {
                pack_i32_u32_pair(number, string_id).is_some_and(|key| values.contains(&key))
            }
            Self::Pair(values) => values.contains(&(number, string_id)),
            Self::DenseI32ToU32 {
                min,
                values,
                present,
                ..
            } => {
                let Ok(number) = i32::try_from(number) else {
                    return false;
                };
                number
                    .checked_sub(*min)
                    .and_then(|offset| usize::try_from(offset).ok())
                    .is_some_and(|index| {
                        present.get(index).is_some_and(|byte| *byte != 0)
                            && values.get(index).is_some_and(|value| *value == string_id)
                    })
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Pair(values) => values.len(),
            Self::PackedI32U32(values) => values.len(),
            Self::DenseI32ToU32 { len, .. } => *len,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Pair(_) => "i64-u32-hash",
            Self::PackedI32U32(_) => "i32-u32-hash",
            Self::DenseI32ToU32 { .. } => "i32-u32-dense",
        }
    }
}

fn mixed_tuple_dense_pair_max_span_factor() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_PAIR_MAX_SPAN_FACTOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

fn mixed_tuple_dense_pair_max_bytes() -> usize {
    std::env::var("DODAM_MIXED_TUPLE_DENSE_PAIR_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024)
}

fn semijoin_i32_u32_pairs_to_values(pairs: Vec<(i32, u32)>) -> SemijoinI64Utf8PairValues {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_DENSE_PAIR_MAP").is_none()
        && let Some(values) = semijoin_i32_u32_pairs_to_dense_values(&pairs)
    {
        return values;
    }
    let mut values = FastHashSet::with_capacity_and_hasher(pairs.len(), Default::default());
    for (number, string_id) in pairs {
        if let Some(key) = pack_i32_u32_pair(i64::from(number), string_id) {
            values.insert(key);
        }
    }
    SemijoinI64Utf8PairValues::PackedI32U32(values)
}

fn semijoin_i32_u32_pairs_to_dense_values(
    pairs: &[(i32, u32)],
) -> Option<SemijoinI64Utf8PairValues> {
    if pairs.is_empty() {
        return Some(SemijoinI64Utf8PairValues::DenseI32ToU32 {
            min: 0,
            values: Vec::new(),
            present: Vec::new(),
            len: 0,
        });
    }
    let min = pairs.iter().map(|(number, _)| *number).min()?;
    let max = pairs.iter().map(|(number, _)| *number).max()?;
    let span = i64::from(max) - i64::from(min) + 1;
    if span <= 0 {
        return None;
    }
    let span = usize::try_from(span).ok()?;
    let max_span = pairs
        .len()
        .saturating_mul(mixed_tuple_dense_pair_max_span_factor());
    if span > max_span || span > mixed_tuple_dense_pair_max_bytes() {
        return None;
    }
    let mut values = vec![0u32; span];
    let mut present = vec![0u8; span];
    let mut len = 0usize;
    for &(number, string_id) in pairs {
        let index = usize::try_from(i64::from(number) - i64::from(min)).ok()?;
        if present[index] != 0 {
            if values[index] != string_id {
                return None;
            }
        } else {
            present[index] = 1;
            values[index] = string_id;
            len += 1;
        }
    }
    Some(SemijoinI64Utf8PairValues::DenseI32ToU32 {
        min,
        values,
        present,
        len,
    })
}

fn pack_i32_u32_pair(number: i64, string_id: u32) -> Option<u64> {
    let number = u32::try_from(i32::try_from(number).ok()?).ok()?;
    Some((u64::from(number) << 32) | u64::from(string_id))
}

fn unpack_i32_u32_pair(key: u64) -> (i64, u32) {
    let number = (key >> 32) as u32;
    (
        i64::from(i32::from_ne_bytes(number.to_ne_bytes())),
        key as u32,
    )
}

struct SemijoinLiteralPairKeys {
    values: FastHashSet<(String, String)>,
    has_null: bool,
}

fn tuple_semijoin_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &TupleSemijoinPairSet,
    negated: bool,
) -> Result<BooleanArray> {
    match keys {
        TupleSemijoinPairSet::I64(keys) => {
            semijoin_i64_pair_mask(batch, left_column, right_column, keys, negated)
        }
        TupleSemijoinPairSet::I64Utf8(keys) => {
            semijoin_i64_utf8_pair_mask(batch, left_column, right_column, keys, negated)
        }
        TupleSemijoinPairSet::Literal(keys) => {
            semijoin_literal_pair_mask(batch, left_column, right_column, keys, negated)
        }
    }
}

async fn collect_semijoin_i64_pair_set(
    engine: &DodamEngine,
    path: PathBuf,
    left_column: &str,
    right_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<SemijoinI64PairKeys> {
    if std::env::var_os("DODAM_DISABLE_I32_PAIR_DIRECT_BUILD").is_none()
        && let Some(keys) = collect_semijoin_i32_pair_set_direct(
            engine,
            &path,
            left_column,
            right_column,
            filter.as_ref(),
            batch_size,
        )?
    {
        return Ok(keys);
    }
    let mut projection = vec![left_column.to_string(), right_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let mut values = None::<SemijoinI64PairSet>;
    let mut has_null = false;
    for batch in stream.by_ref() {
        let batch = batch?;
        let left_index = batch_column_index(&batch, left_column)?;
        let right_index = batch_column_index(&batch, right_column)?;
        let left = batch.column(left_index);
        let right = batch.column(right_index);
        has_null |= left.null_count() > 0 || right.null_count() > 0;
        let values = values.get_or_insert_with(|| SemijoinI64PairSet::for_arrays(left, right));
        values.insert_from_arrays(left, right)?;
    }
    Ok(SemijoinI64PairKeys {
        values: values.unwrap_or_else(SemijoinI64PairSet::empty_pair),
        has_null,
    })
}

fn collect_semijoin_i32_pair_set_direct(
    engine: &DodamEngine,
    path: &Path,
    left_column: &str,
    right_column: &str,
    filter: Option<&FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinI64PairKeys>> {
    let Some(filter_expr) = filter.map(FilterExpr::expr) else {
        return Ok(None);
    };
    if !semijoin_i32_pair_filter_supported(filter_expr, left_column, right_column) {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    let mut pair_values = Vec::<(i32, i32)>::new();
    let mut has_null = false;
    let metrics = engine.scan_parquet_i32_i32_columns(
        path,
        batch_size,
        &row_groups,
        [left_column, right_column],
        |left_values, left_def_levels, right_values, right_def_levels| {
            pair_values.reserve(left_values.len());
            for row in 0..left_values.len() {
                let left = (!left_def_levels.is_some_and(|levels| levels[row] == 0))
                    .then_some(left_values[row]);
                let right = (!right_def_levels.is_some_and(|levels| levels[row] == 0))
                    .then_some(right_values[row]);
                if !semijoin_i32_pair_nullable_filter_matches(
                    filter_expr,
                    left_column,
                    right_column,
                    left,
                    right,
                ) {
                    continue;
                }
                match (left, right) {
                    (Some(left), Some(right)) => {
                        pair_values.push((left, right));
                    }
                    _ => {
                        has_null = true;
                    }
                }
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    let values = if std::env::var_os("DODAM_DISABLE_I32_PAIR_DENSE_MAP").is_none() {
        semijoin_i32_pair_values_to_dense_map(&pair_values).unwrap_or_else(|| {
            SemijoinI64PairSet::PackedI32(
                pair_values
                    .iter()
                    .map(|(left, right)| pack_i32_pair(*left, *right))
                    .collect(),
            )
        })
    } else {
        SemijoinI64PairSet::PackedI32(
            pair_values
                .iter()
                .map(|(left, right)| pack_i32_pair(*left, *right))
                .collect(),
        )
    };
    Ok(Some(SemijoinI64PairKeys { values, has_null }))
}

fn semijoin_i32_pair_values_to_dense_map(pairs: &[(i32, i32)]) -> Option<SemijoinI64PairSet> {
    if pairs.is_empty() {
        return Some(SemijoinI64PairSet::DenseI32ToI32 {
            min: 0,
            values: Vec::new(),
            present: Vec::new(),
        });
    }
    let min = pairs.iter().map(|(left, _)| *left).min()?;
    let max = pairs.iter().map(|(left, _)| *left).max()?;
    let span = i64::from(max) - i64::from(min) + 1;
    if span <= 0 {
        return None;
    }
    let span = usize::try_from(span).ok()?;
    let max_span = pairs
        .len()
        .saturating_mul(i32_pair_dense_map_max_span_factor());
    if span > max_span || span > i32_pair_dense_map_max_bytes() {
        return None;
    }
    let mut values = vec![0i32; span];
    let mut present = vec![0u8; span];
    for &(left, right) in pairs {
        let index = usize::try_from(i64::from(left) - i64::from(min)).ok()?;
        if present[index] != 0 {
            if values[index] != right {
                return None;
            }
        } else {
            present[index] = 1;
            values[index] = right;
        }
    }
    Some(SemijoinI64PairSet::DenseI32ToI32 {
        min,
        values,
        present,
    })
}

fn i32_pair_dense_map_max_span_factor() -> usize {
    std::env::var("DODAM_I32_PAIR_DENSE_MAP_MAX_SPAN_FACTOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

fn i32_pair_dense_map_max_bytes() -> usize {
    std::env::var("DODAM_I32_PAIR_DENSE_MAP_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024)
}

fn semijoin_i32_pair_filter_supported(expr: &Expr, left_column: &str, right_column: &str) -> bool {
    match expr {
        Expr::Boolean(_) => true,
        Expr::And(left, right) => {
            semijoin_i32_pair_filter_supported(left, left_column, right_column)
                && semijoin_i32_pair_filter_supported(right, left_column, right_column)
        }
        Expr::Comparison(ComparisonExpr { column, value, .. }) => {
            (column == left_column || column == right_column)
                && matches!(value, LiteralValue::Int64(value) if i32::try_from(*value).is_ok())
        }
        Expr::IsNull { column, .. } => column == left_column || column == right_column,
        _ => false,
    }
}

fn semijoin_i32_pair_nullable_filter_matches(
    expr: &Expr,
    left_column: &str,
    right_column: &str,
    left: Option<i32>,
    right: Option<i32>,
) -> bool {
    match expr {
        Expr::Boolean(value) => value.unwrap_or(false),
        Expr::And(lhs, rhs) => {
            semijoin_i32_pair_nullable_filter_matches(lhs, left_column, right_column, left, right)
                && semijoin_i32_pair_nullable_filter_matches(
                    rhs,
                    left_column,
                    right_column,
                    left,
                    right,
                )
        }
        Expr::Comparison(ComparisonExpr { column, op, value }) => {
            let LiteralValue::Int64(value) = value else {
                return false;
            };
            let Ok(value) = i32::try_from(*value) else {
                return false;
            };
            let input = if column == left_column {
                left
            } else if column == right_column {
                right
            } else {
                return false;
            };
            input.is_some_and(|input| compare_i32(input, *op, value))
        }
        Expr::IsNull { column, negated } => {
            let input = if column == left_column {
                left
            } else if column == right_column {
                right
            } else {
                return false;
            };
            let is_null = input.is_none();
            if *negated { !is_null } else { is_null }
        }
        _ => false,
    }
}

async fn collect_semijoin_literal_pair_set(
    engine: &DodamEngine,
    path: PathBuf,
    left_column: &str,
    right_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<SemijoinLiteralPairKeys> {
    let mut projection = vec![left_column.to_string(), right_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let mut values = FastHashSet::<(String, String)>::default();
    let mut has_null = false;
    for batch in stream.by_ref() {
        let batch = batch?;
        let left_index = batch_column_index(&batch, left_column)?;
        let right_index = batch_column_index(&batch, right_column)?;
        let left = batch.column(left_index);
        let right = batch.column(right_index);
        has_null |= left.null_count() > 0 || right.null_count() > 0;
        for row in 0..batch.num_rows() {
            let Some(left) = semijoin_key_at(left, row)? else {
                continue;
            };
            let Some(right) = semijoin_key_at(right, row)? else {
                continue;
            };
            values.insert((left, right));
        }
    }
    Ok(SemijoinLiteralPairKeys { values, has_null })
}

async fn collect_semijoin_i64_utf8_pair_set(
    engine: &DodamEngine,
    path: PathBuf,
    left_column: &str,
    right_column: &str,
    filter: Option<FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_DIRECT_BUILD").is_none()
        && let Some(keys) = collect_semijoin_i64_utf8_pair_set_direct(
            engine,
            &path,
            left_column,
            right_column,
            filter.as_ref(),
            batch_size,
        )?
    {
        return Ok(Some(keys));
    }
    let mut projection = vec![left_column.to_string(), right_column.to_string()];
    if let Some(filter) = &filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut projection, column);
        }
    }
    let mut stream = engine
        .scan_parquet_batches(
            path,
            batch_size,
            None,
            Projection::Columns(projection),
            filter.clone(),
        )
        .await?;
    let mut values = SemijoinI64Utf8PairValues::empty_packed();
    let mut numeric_values = SemijoinNumericValues::empty_i64();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut numeric_left = None::<bool>;
    let mut has_null = false;
    for batch in stream.by_ref() {
        let batch = batch?;
        let left_index = batch_column_index(&batch, left_column)?;
        let right_index = batch_column_index(&batch, right_column)?;
        let left = batch.column(left_index);
        let right = batch.column(right_index);
        has_null |= left.null_count() > 0 || right.null_count() > 0;
        let batch_numeric_left = if semijoin_i64_array(left).is_some()
            && (right.as_any().downcast_ref::<StringArray>().is_some()
                || semijoin_dictionary_i32_view(right).is_some())
        {
            true
        } else if (left.as_any().downcast_ref::<StringArray>().is_some()
            || semijoin_dictionary_i32_view(left).is_some())
            && semijoin_i64_array(right).is_some()
        {
            false
        } else {
            return Ok(None);
        };
        if let Some(existing) = numeric_left {
            if existing != batch_numeric_left {
                return Ok(None);
            }
        } else {
            numeric_left = Some(batch_numeric_left);
        }
        insert_semijoin_i64_utf8_pairs(
            &mut values,
            &mut numeric_values,
            &mut string_ids,
            left,
            right,
            batch_numeric_left,
        )?;
    }
    Ok(numeric_left.map(|numeric_left| SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null,
    }))
}

fn collect_semijoin_i64_utf8_pair_set_direct(
    engine: &DodamEngine,
    path: &Path,
    left_column: &str,
    right_column: &str,
    filter: Option<&FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let Some((filter_column, prefix)) = semijoin_like_prefix_filter(filter) else {
        if semijoin_profile_enabled() {
            eprintln!("[dodam:semijoin-profile] mixed-direct-build skip=unsupported-filter");
        }
        return Ok(None);
    };
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(SemijoinI64Utf8PairKeys {
            values: SemijoinI64Utf8PairValues::empty_packed(),
            numeric_values: SemijoinNumericValues::empty_i32(),
            string_ids: FastHashMap::default(),
            numeric_left: true,
            has_null: false,
        }));
    }
    if filter_column == right_column {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build attempt numeric={left_column} string={right_column}"
            );
        }
        return collect_semijoin_i32_utf8_pair_set_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            left_column,
            right_column,
            prefix,
            true,
        );
    }
    if filter_column == left_column {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build attempt numeric={right_column} string={left_column}"
            );
        }
        return collect_semijoin_i32_utf8_pair_set_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            right_column,
            left_column,
            prefix,
            false,
        );
    }
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build skip=filter-column-not-tuple-key filter={filter_column}"
        );
    }
    Ok(None)
}

fn collect_semijoin_mixed_prefix_candidate_keys_direct(
    engine: &DodamEngine,
    path: &Path,
    left_column: &str,
    right_column: &str,
    filter: Option<&FilterExpr>,
    batch_size: usize,
) -> Result<Option<SemijoinMixedPrefixCandidateKeys>> {
    let Some((filter_column, prefix)) = semijoin_like_prefix_filter(filter) else {
        return Ok(None);
    };
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(SemijoinMixedPrefixCandidateKeys {
            numeric_values: SemijoinNumericValues::empty_i32(),
            string_ids: FastHashMap::default(),
            numeric_left: true,
        }));
    }
    if filter_column == right_column {
        collect_semijoin_i32_utf8_prefix_candidate_keys_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            left_column,
            right_column,
            prefix,
            true,
        )
    } else if filter_column == left_column {
        collect_semijoin_i32_utf8_prefix_candidate_keys_direct(
            engine,
            path,
            batch_size,
            &row_groups,
            right_column,
            left_column,
            prefix,
            false,
        )
    } else {
        Ok(None)
    }
}

fn collect_semijoin_i32_utf8_prefix_candidate_keys_direct(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinMixedPrefixCandidateKeys>> {
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    let metrics = engine.scan_parquet_i32_dictionary_id_columns(
        path,
        batch_size,
        row_groups,
        [numeric_column, string_column],
        |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_prefix(dictionary, &mut string_ids, prefix.as_bytes())?;
            numeric_values.reserve(numbers.len());
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        if usize::try_from(*dictionary_id)
                            .ok()
                            .and_then(|index| selected_string_ids.get(index))
                            .and_then(|id| *id)
                            .is_some()
                        {
                            numeric_values.insert_i32(numbers[row]);
                        }
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        if usize::try_from(dictionary_id)
                            .ok()
                            .and_then(|index| selected_string_ids.get(index))
                            .and_then(|id| *id)
                            .is_some()
                        {
                            numeric_values.insert_i32(numbers[row]);
                        }
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    numeric_values = numeric_values.optimize_dense_i32();
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-prefix-precheck rhs numeric_values={} strings={} numeric={}",
            numeric_values.len(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinMixedPrefixCandidateKeys {
        numeric_values,
        string_ids,
        numeric_left,
    }))
}

fn collect_semijoin_i32_utf8_pair_set_direct(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let mut pair_values = Vec::<(i32, u32)>::new();
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut has_null = false;
    if std::env::var_os("DODAM_MIXED_TUPLE_SELECTED_INNER_BUILD").is_some()
        && let Some(keys) = collect_semijoin_i32_utf8_pair_set_direct_selected_inner(
            engine,
            path,
            row_groups,
            numeric_column,
            string_column,
            prefix,
            numeric_left,
        )?
    {
        return Ok(Some(keys));
    }
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    let metrics = engine.scan_parquet_i32_dictionary_id_columns(
        path,
        batch_size,
        row_groups,
        [numeric_column, string_column],
        |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_prefix(dictionary, &mut string_ids, prefix.as_bytes())?;
            pair_values.reserve(numbers.len());
            numeric_values.reserve(numbers.len());
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    has_null |= levels.iter().any(|level| *level == 0);
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        insert_direct_semijoin_i32_utf8_pair_value(
                            numbers[row],
                            *dictionary_id,
                            &selected_string_ids,
                            &mut pair_values,
                            &mut numeric_values,
                        );
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        insert_direct_semijoin_i32_utf8_pair_value(
                            numbers[row],
                            dictionary_id,
                            &selected_string_ids,
                            &mut pair_values,
                            &mut numeric_values,
                        );
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-direct-build dictionary-id unsupported; trying bytearray"
            );
        }
        return collect_semijoin_i32_utf8_pair_set_direct_byte_array(
            engine,
            path,
            batch_size,
            row_groups,
            numeric_column,
            string_column,
            prefix,
            numeric_left,
        );
    }
    numeric_values = numeric_values.optimize_dense_i32();
    let values = semijoin_i32_u32_pairs_to_values(pair_values);
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build dictionary-id ok values={} pair={} strings={} numeric={}",
            values.len(),
            values.kind(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null,
    }))
}

fn collect_semijoin_i32_utf8_pair_set_direct_selected_inner(
    engine: &DodamEngine,
    path: &Path,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let mut pair_values = Vec::<(i32, u32)>::new();
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    let metrics = engine.scan_parquet_i32_selected_by_byte_array_prefix(
        path,
        row_groups,
        [numeric_column, string_column],
        prefix.as_bytes(),
        |numbers, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_prefix(dictionary, &mut string_ids, prefix.as_bytes())?;
            if numbers.len() != dictionary_ids.len() {
                return Ok(None);
            }
            pair_values.reserve(numbers.len());
            numeric_values.reserve(numbers.len());
            for (number, dictionary_id) in
                numbers.iter().copied().zip(dictionary_ids.iter().copied())
            {
                insert_direct_semijoin_i32_utf8_pair_value(
                    number,
                    dictionary_id,
                    selected_string_ids,
                    &mut pair_values,
                    &mut numeric_values,
                );
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    numeric_values = numeric_values.optimize_dense_i32();
    let values = semijoin_i32_u32_pairs_to_values(pair_values);
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build selected-inner ok values={} pair={} strings={} numeric={}",
            values.len(),
            values.kind(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null: false,
    }))
}

fn collect_semijoin_i32_utf8_pair_set_direct_byte_array(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    row_groups: &[usize],
    numeric_column: &str,
    string_column: &str,
    prefix: &str,
    numeric_left: bool,
) -> Result<Option<SemijoinI64Utf8PairKeys>> {
    let mut pair_values = Vec::<(i32, u32)>::new();
    let mut numeric_values = SemijoinNumericValues::empty_i32();
    let mut string_ids = FastHashMap::<Vec<u8>, u32>::default();
    let mut has_null = false;
    let metrics = engine.scan_parquet_i32_byte_array_columns(
        path,
        batch_size,
        row_groups,
        [numeric_column, string_column],
        |numbers, string_def_levels, strings| {
            pair_values.reserve(numbers.len());
            numeric_values.reserve(numbers.len());
            let mut string_value_offset = 0usize;
            for (row, level) in string_def_levels.iter().copied().enumerate() {
                if level == 0 {
                    has_null = true;
                    continue;
                }
                let Some(value) = strings.get(string_value_offset) else {
                    return Ok(None);
                };
                string_value_offset += 1;
                if !value.as_ref().starts_with(prefix.as_bytes()) {
                    continue;
                }
                let string_id = semijoin_intern_string_id(&mut string_ids, value.as_ref())?;
                numeric_values.insert_i32(numbers[row]);
                pair_values.push((numbers[row], string_id));
            }
            if string_value_offset != strings.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        if semijoin_profile_enabled() {
            eprintln!("[dodam:semijoin-profile] mixed-direct-build bytearray unsupported");
        }
        return Ok(None);
    }
    numeric_values = numeric_values.optimize_dense_i32();
    let values = semijoin_i32_u32_pairs_to_values(pair_values);
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-direct-build bytearray ok values={} pair={} strings={} numeric={}",
            values.len(),
            values.kind(),
            string_ids.len(),
            numeric_values.kind()
        );
    }
    Ok(Some(SemijoinI64Utf8PairKeys {
        values,
        numeric_values,
        string_ids,
        numeric_left,
        has_null,
    }))
}

#[derive(Default)]
struct SemijoinDictionaryStringIdCache {
    ptr: *const bytes::Bytes,
    len: usize,
    fingerprint: u64,
    ids: Vec<Option<u32>>,
}

impl SemijoinDictionaryStringIdCache {
    fn refresh_prefix<'a>(
        &'a mut self,
        dictionary: &[bytes::Bytes],
        string_ids: &mut FastHashMap<Vec<u8>, u32>,
        prefix: &[u8],
    ) -> Result<&'a [Option<u32>]> {
        let fingerprint = semijoin_dictionary_fingerprint(dictionary);
        if self.ptr == dictionary.as_ptr()
            && self.len == dictionary.len()
            && self.fingerprint == fingerprint
        {
            return Ok(&self.ids);
        }
        self.ptr = dictionary.as_ptr();
        self.len = dictionary.len();
        self.fingerprint = fingerprint;
        self.ids.clear();
        self.ids.reserve(dictionary.len());
        for value in dictionary {
            self.ids.push(if value.as_ref().starts_with(prefix) {
                Some(semijoin_intern_string_id(string_ids, value.as_ref())?)
            } else {
                None
            });
        }
        Ok(&self.ids)
    }

    fn refresh_existing<'a>(
        &'a mut self,
        dictionary: &[bytes::Bytes],
        string_ids: &FastHashMap<Vec<u8>, u32>,
    ) -> &'a [Option<u32>] {
        let fingerprint = semijoin_dictionary_fingerprint(dictionary);
        if self.ptr == dictionary.as_ptr()
            && self.len == dictionary.len()
            && self.fingerprint == fingerprint
        {
            return &self.ids;
        }
        self.ptr = dictionary.as_ptr();
        self.len = dictionary.len();
        self.fingerprint = fingerprint;
        self.ids.clear();
        self.ids.reserve(dictionary.len());
        self.ids.extend(
            dictionary
                .iter()
                .map(|value| string_ids.get(value.as_ref()).copied()),
        );
        &self.ids
    }
}

fn semijoin_dictionary_fingerprint(dictionary: &[bytes::Bytes]) -> u64 {
    let mut hash = dictionary.len() as u64;
    for value in dictionary {
        hash = hash.wrapping_mul(0x9E37_79B1_85EB_CA87);
        hash ^= value.len() as u64;
        for byte in value.as_ref() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01B3);
        }
    }
    hash
}

fn insert_direct_semijoin_i32_utf8_pair_value(
    number: i32,
    dictionary_id: i32,
    selected_string_ids: &[Option<u32>],
    pair_values: &mut Vec<(i32, u32)>,
    numeric_values: &mut SemijoinNumericValues,
) {
    let Ok(dictionary_id) = usize::try_from(dictionary_id) else {
        return;
    };
    let Some(Some(string_id)) = selected_string_ids.get(dictionary_id) else {
        return;
    };
    numeric_values.insert_i32(number);
    pair_values.push((number, *string_id));
}

fn semijoin_like_prefix_filter(filter: Option<&FilterExpr>) -> Option<(&str, &str)> {
    let Expr::Like {
        column,
        pattern,
        negated: false,
        escape: None,
        case_insensitive: false,
    } = filter?.expr()
    else {
        return None;
    };
    let prefix = pattern.strip_suffix('%')?;
    if prefix.is_empty()
        || !prefix.is_ascii()
        || prefix
            .as_bytes()
            .iter()
            .any(|byte| matches!(byte, b'%' | b'_'))
    {
        return None;
    }
    Some((column.as_str(), prefix))
}

fn semijoin_filter_prefers_mixed_tuple(
    filter: Option<&FilterExpr>,
    left_column: &str,
    right_column: &str,
) -> bool {
    semijoin_like_prefix_filter(filter)
        .map(|(column, _)| column == left_column || column == right_column)
        .unwrap_or(false)
}

enum SemijoinI64Array<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
}

fn semijoin_i64_array(array: &ArrayRef) -> Option<SemijoinI64Array<'_>> {
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        Some(SemijoinI64Array::I32(array))
    } else {
        array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(SemijoinI64Array::I64)
    }
}

fn semijoin_i64_array_value(array: &SemijoinI64Array<'_>, row: usize) -> i64 {
    match array {
        SemijoinI64Array::I32(array) => i64::from(array.value(row)),
        SemijoinI64Array::I64(array) => array.value(row),
    }
}

fn semijoin_i64_array_is_valid(array: &SemijoinI64Array<'_>, row: usize) -> bool {
    match array {
        SemijoinI64Array::I32(array) => array.is_valid(row),
        SemijoinI64Array::I64(array) => array.is_valid(row),
    }
}

fn insert_semijoin_i64_utf8_pairs(
    values: &mut SemijoinI64Utf8PairValues,
    numeric_values: &mut SemijoinNumericValues,
    string_ids: &mut FastHashMap<Vec<u8>, u32>,
    left: &ArrayRef,
    right: &ArrayRef,
    numeric_left: bool,
) -> Result<()> {
    let (numbers, string_array, dictionary) = if numeric_left {
        let Some(numbers) = semijoin_i64_array(left) else {
            return Ok(());
        };
        let strings = right.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(right);
        if strings.is_none() && dictionary.is_none() {
            return Ok(());
        }
        (numbers, strings, dictionary)
    } else {
        let Some(numbers) = semijoin_i64_array(right) else {
            return Ok(());
        };
        let strings = left.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(left);
        if strings.is_none() && dictionary.is_none() {
            return Ok(());
        }
        (numbers, strings, dictionary)
    };
    if let Some(strings) = string_array {
        values.reserve(strings.len());
        numeric_values.reserve(strings.len());
        for row in 0..strings.len() {
            if semijoin_i64_array_is_valid(&numbers, row) && strings.is_valid(row) {
                let number = semijoin_i64_array_value(&numbers, row);
                let string_id =
                    semijoin_intern_string_id(string_ids, strings.value(row).as_bytes())?;
                numeric_values.insert_i64(number);
                values.insert(number, string_id);
            }
        }
        return Ok(());
    }
    let Some(dictionary) = dictionary else {
        return Ok(());
    };
    let Some(local_string_ids) = semijoin_dictionary_global_string_ids(dictionary, string_ids)?
    else {
        return Ok(());
    };
    let keys = dictionary.keys();
    values.reserve(keys.len());
    numeric_values.reserve(keys.len());
    for row in 0..keys.len() {
        if semijoin_i64_array_is_valid(&numbers, row) && !dictionary.is_null(row) {
            let Ok(local_id) = usize::try_from(keys[row]) else {
                continue;
            };
            if let Some(Some(string_id)) = local_string_ids.get(local_id) {
                let number = semijoin_i64_array_value(&numbers, row);
                numeric_values.insert_i64(number);
                values.insert(number, *string_id);
            }
        }
    }
    Ok(())
}

fn semijoin_dictionary_i32_view(array: &ArrayRef) -> Option<DictionaryI32View<'_>> {
    array
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .map(DictionaryI32View::Arrow)
}

fn semijoin_intern_string_id(
    string_ids: &mut FastHashMap<Vec<u8>, u32>,
    value: &[u8],
) -> Result<u32> {
    if let Some(id) = string_ids.get(value) {
        return Ok(*id);
    }
    let id = u32::try_from(string_ids.len()).map_err(|_| {
        DodamError::UnsupportedSql("too many distinct string semijoin keys".to_string())
    })?;
    string_ids.insert(value.to_vec(), id);
    Ok(id)
}

fn semijoin_dictionary_global_string_ids(
    dictionary: DictionaryI32View<'_>,
    string_ids: &mut FastHashMap<Vec<u8>, u32>,
) -> Result<Option<Vec<Option<u32>>>> {
    let Some(values) = dictionary.string_values() else {
        return Ok(None);
    };
    let mut ids = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        ids.push(Some(semijoin_intern_string_id(
            string_ids,
            values.value_bytes(index),
        )?));
    }
    Ok(Some(ids))
}

fn semijoin_dictionary_existing_string_ids(
    dictionary: DictionaryI32View<'_>,
    string_ids: &FastHashMap<Vec<u8>, u32>,
) -> Option<Vec<Option<u32>>> {
    let values = dictionary.string_values()?;
    let mut ids = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        ids.push(string_ids.get(values.value_bytes(index)).copied());
    }
    Some(ids)
}

fn semijoin_i64_utf8_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinI64Utf8PairKeys,
    negated: bool,
) -> Result<BooleanArray> {
    let left_index = batch_column_index(batch, left_column)?;
    let right_index = batch_column_index(batch, right_column)?;
    let left = batch.column(left_index);
    let right = batch.column(right_index);
    let (numbers, string_array, dictionary) = if keys.numeric_left {
        let Some(numbers) = semijoin_i64_array(left) else {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        };
        let strings = right.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(right);
        if strings.is_none() && dictionary.is_none() {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        }
        (numbers, strings, dictionary)
    } else {
        let Some(numbers) = semijoin_i64_array(right) else {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        };
        let strings = left.as_any().downcast_ref::<StringArray>();
        let dictionary = semijoin_dictionary_i32_view(left);
        if strings.is_none() && dictionary.is_none() {
            return Err(DodamError::UnsupportedSql(
                "integer/string semijoin key type changed across batches".to_string(),
            ));
        }
        (numbers, strings, dictionary)
    };
    if let Some(dictionary) = dictionary {
        let local_string_ids =
            semijoin_dictionary_existing_string_ids(dictionary, &keys.string_ids);
        if let Some(local_string_ids) = local_string_ids {
            let dictionary_keys = dictionary.keys();
            return Ok(BooleanArray::from(
                (0..batch.num_rows())
                    .map(|row| {
                        if !semijoin_i64_array_is_valid(&numbers, row) || dictionary.is_null(row) {
                            return Some(false);
                        }
                        let number = semijoin_i64_array_value(&numbers, row);
                        if !keys.numeric_values.contains_i64(number) {
                            return Some(negated);
                        }
                        let matched = usize::try_from(dictionary_keys[row])
                            .ok()
                            .and_then(|local_id| local_string_ids.get(local_id).copied().flatten())
                            .is_some_and(|string_id| keys.values.contains(number, string_id));
                        Some(if negated { !matched } else { matched })
                    })
                    .collect::<Vec<_>>(),
            ));
        }
    }
    let Some(strings) = string_array else {
        return Err(DodamError::UnsupportedSql(
            "integer/string semijoin dictionary values must be Utf8".to_string(),
        ));
    };
    Ok(BooleanArray::from(
        (0..batch.num_rows())
            .map(|row| {
                if !semijoin_i64_array_is_valid(&numbers, row) || strings.is_null(row) {
                    return Some(false);
                }
                let number = semijoin_i64_array_value(&numbers, row);
                if !keys.numeric_values.contains_i64(number) {
                    return Some(negated);
                }
                let matched = keys
                    .string_ids
                    .get(strings.value(row).as_bytes())
                    .is_some_and(|string_id| keys.values.contains(number, *string_id));
                Some(if negated { !matched } else { matched })
            })
            .collect::<Vec<_>>(),
    ))
}

fn semijoin_mixed_early_empty_probe_accepts(
    engine: &DodamEngine,
    path: &Path,
    keys: &SemijoinI64Utf8PairKeys,
) -> Result<bool> {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_EARLY_EMPTY").is_some() {
        return Ok(false);
    }
    let total_rows = engine.parquet_total_row_count(path)?;
    if total_rows == 0 {
        return Ok(true);
    }
    let ratio = keys.numeric_values.len() as f64 / total_rows as f64;
    let accepted = ratio <= mixed_tuple_early_empty_max_numeric_ratio();
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-early-empty numeric_values={} total_rows={} ratio={:.6} accepted={}",
            keys.numeric_values.len(),
            total_rows,
            ratio,
            accepted
        );
    }
    Ok(accepted)
}

fn semijoin_mixed_prefix_precheck_accepts(
    engine: &DodamEngine,
    path: &Path,
    keys: &SemijoinMixedPrefixCandidateKeys,
) -> Result<bool> {
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_PREFIX_PRECHECK").is_some() {
        return Ok(false);
    }
    let total_rows = engine.parquet_total_row_count(path)?;
    if total_rows == 0 {
        return Ok(true);
    }
    let ratio = keys.numeric_values.len() as f64 / total_rows as f64;
    let accepted = ratio <= mixed_tuple_early_empty_max_numeric_ratio();
    if semijoin_profile_enabled() {
        eprintln!(
            "[dodam:semijoin-profile] mixed-prefix-precheck numeric_values={} total_rows={} ratio={:.6} accepted={}",
            keys.numeric_values.len(),
            total_rows,
            ratio,
            accepted
        );
    }
    Ok(accepted)
}

fn mixed_tuple_early_empty_max_numeric_ratio() -> f64 {
    std::env::var("DODAM_MIXED_TUPLE_EARLY_EMPTY_MAX_NUMERIC_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.25)
}

fn direct_mixed_tuple_semijoin_outer_has_match(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinI64Utf8PairKeys,
    outer_filter: Option<&FilterExpr>,
) -> Result<Option<bool>> {
    let (numeric_column, string_column) = if keys.numeric_left {
        (left_column, right_column)
    } else {
        (right_column, left_column)
    };
    if !semijoin_direct_outer_filter_compatible(outer_filter, string_column) {
        if semijoin_profile_enabled() {
            eprintln!(
                "[dodam:semijoin-profile] mixed-early-empty skip=incompatible_outer_filter filter={:?} string_column={}",
                outer_filter.map(FilterExpr::expr),
                string_column
            );
        }
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(false));
    }
    let mut found = false;
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_NUMERIC_SELECTED_OUTER").is_none() {
        let metrics = engine.scan_parquet_i32_byte_array_selected_by_i32(
            path,
            &row_groups,
            [numeric_column, string_column],
            |number| keys.numeric_values.contains_i32(number),
            |numbers, dictionary_ids, dictionary| {
                let selected_string_ids =
                    string_id_cache.refresh_existing(dictionary, &keys.string_ids);
                if numbers.len() != dictionary_ids.len() {
                    return Ok(None);
                }
                for (number, dictionary_id) in
                    numbers.iter().copied().zip(dictionary_ids.iter().copied())
                {
                    if direct_mixed_tuple_semijoin_dictionary_row_matches(
                        number,
                        dictionary_id,
                        selected_string_ids,
                        keys,
                    ) {
                        found = true;
                        return Ok(Some(()));
                    }
                }
                Ok(Some(()))
            },
        )?;
        if metrics.is_some() {
            if semijoin_profile_enabled() {
                eprintln!("[dodam:semijoin-profile] mixed-early-empty found={found}");
            }
            return Ok(Some(found));
        }
    }
    let metrics = engine.scan_parquet_i32_dictionary_id_columns(
        path,
        batch_size,
        &row_groups,
        [numeric_column, string_column],
        |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_existing(dictionary, &keys.string_ids);
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        if direct_mixed_tuple_semijoin_dictionary_row_matches(
                            numbers[row],
                            *dictionary_id,
                            &selected_string_ids,
                            keys,
                        ) {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        if direct_mixed_tuple_semijoin_dictionary_row_matches(
                            numbers[row],
                            dictionary_id,
                            &selected_string_ids,
                            keys,
                        ) {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        let metrics = engine.scan_parquet_i32_byte_array_columns(
            path,
            batch_size,
            &row_groups,
            [numeric_column, string_column],
            |numbers, string_def_levels, strings| {
                let mut string_value_offset = 0usize;
                for (row, level) in string_def_levels.iter().copied().enumerate() {
                    if level == 0 {
                        continue;
                    }
                    let Some(value) = strings.get(string_value_offset) else {
                        return Ok(None);
                    };
                    string_value_offset += 1;
                    let number = i64::from(numbers[row]);
                    if !keys.numeric_values.contains_i64(number) {
                        continue;
                    }
                    if keys
                        .string_ids
                        .get(value.as_ref())
                        .is_some_and(|string_id| keys.values.contains(number, *string_id))
                    {
                        found = true;
                        return Ok(Some(()));
                    }
                }
                if string_value_offset != strings.len() {
                    return Ok(None);
                }
                Ok(Some(()))
            },
        )?;
        if metrics.is_none() {
            if semijoin_profile_enabled() {
                eprintln!(
                    "[dodam:semijoin-profile] mixed-early-empty skip=direct_reader_unsupported"
                );
            }
            return Ok(None);
        }
    }
    if semijoin_profile_enabled() {
        eprintln!("[dodam:semijoin-profile] mixed-early-empty found={found}");
    }
    Ok(Some(found))
}

fn direct_mixed_tuple_semijoin_outer_has_prefix_candidate(
    engine: &DodamEngine,
    path: &Path,
    batch_size: usize,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinMixedPrefixCandidateKeys,
    outer_filter: Option<&FilterExpr>,
) -> Result<Option<bool>> {
    let (numeric_column, string_column) = if keys.numeric_left {
        (left_column, right_column)
    } else {
        (right_column, left_column)
    };
    if keys.numeric_values.len() == 0 || keys.string_ids.is_empty() {
        return Ok(Some(false));
    }
    if !semijoin_direct_outer_filter_compatible(outer_filter, string_column) {
        return Ok(None);
    }
    let row_groups = (0..engine.parquet_row_group_count(path)?).collect::<Vec<_>>();
    if row_groups.is_empty() {
        return Ok(Some(false));
    }
    let mut found = false;
    let mut string_id_cache = SemijoinDictionaryStringIdCache::default();
    if std::env::var_os("DODAM_DISABLE_MIXED_TUPLE_NUMERIC_SELECTED_OUTER").is_none() {
        let metrics = engine.scan_parquet_i32_byte_array_selected_by_i32(
            path,
            &row_groups,
            [numeric_column, string_column],
            |number| keys.numeric_values.contains_i32(number),
            |numbers, dictionary_ids, dictionary| {
                let selected_string_ids =
                    string_id_cache.refresh_existing(dictionary, &keys.string_ids);
                if numbers.len() != dictionary_ids.len() {
                    return Ok(None);
                }
                for dictionary_id in dictionary_ids.iter().copied() {
                    if usize::try_from(dictionary_id)
                        .ok()
                        .and_then(|index| selected_string_ids.get(index))
                        .and_then(|id| *id)
                        .is_some()
                    {
                        found = true;
                        return Ok(Some(()));
                    }
                }
                Ok(Some(()))
            },
        )?;
        if metrics.is_some() {
            if semijoin_profile_enabled() {
                eprintln!("[dodam:semijoin-profile] mixed-prefix-precheck outer_found={found}");
            }
            return Ok(Some(found));
        }
    }
    let metrics = engine.scan_parquet_i32_dictionary_id_columns(
        path,
        batch_size,
        &row_groups,
        [numeric_column, string_column],
        |numbers, dictionary_def_levels, dictionary_ids, dictionary| {
            let selected_string_ids =
                string_id_cache.refresh_existing(dictionary, &keys.string_ids);
            let mut dictionary_value_offset = 0usize;
            match dictionary_def_levels {
                Some(levels) => {
                    for (row, level) in levels.iter().copied().enumerate() {
                        if level == 0 {
                            continue;
                        }
                        let Some(dictionary_id) = dictionary_ids.get(dictionary_value_offset)
                        else {
                            return Ok(None);
                        };
                        dictionary_value_offset += 1;
                        if !keys.numeric_values.contains_i32(numbers[row]) {
                            continue;
                        }
                        if usize::try_from(*dictionary_id)
                            .ok()
                            .and_then(|index| selected_string_ids.get(index))
                            .and_then(|id| *id)
                            .is_some()
                        {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                }
                None => {
                    for (row, dictionary_id) in dictionary_ids.iter().copied().enumerate() {
                        if !keys.numeric_values.contains_i32(numbers[row]) {
                            continue;
                        }
                        if usize::try_from(dictionary_id)
                            .ok()
                            .and_then(|index| selected_string_ids.get(index))
                            .and_then(|id| *id)
                            .is_some()
                        {
                            found = true;
                            return Ok(Some(()));
                        }
                    }
                    dictionary_value_offset = dictionary_ids.len();
                }
            }
            if dictionary_value_offset != dictionary_ids.len() {
                return Ok(None);
            }
            Ok(Some(()))
        },
    )?;
    if metrics.is_none() {
        return Ok(None);
    }
    if semijoin_profile_enabled() {
        eprintln!("[dodam:semijoin-profile] mixed-prefix-precheck outer_found={found}");
    }
    Ok(Some(found))
}

fn direct_mixed_tuple_semijoin_dictionary_row_matches(
    number: i32,
    dictionary_id: i32,
    selected_string_ids: &[Option<u32>],
    keys: &SemijoinI64Utf8PairKeys,
) -> bool {
    if !keys.numeric_values.contains_i32(number) {
        return false;
    }
    let number = i64::from(number);
    usize::try_from(dictionary_id)
        .ok()
        .and_then(|dictionary_id| selected_string_ids.get(dictionary_id).copied().flatten())
        .is_some_and(|string_id| keys.values.contains(number, string_id))
}

fn semijoin_direct_outer_filter_compatible(
    filter: Option<&FilterExpr>,
    string_column: &str,
) -> bool {
    match filter.map(FilterExpr::expr) {
        None => true,
        Some(Expr::IsNull {
            column,
            negated: true,
        }) => column == string_column,
        _ => false,
    }
}

fn semijoin_literal_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinLiteralPairKeys,
    negated: bool,
) -> Result<BooleanArray> {
    let left_index = batch_column_index(batch, left_column)?;
    let right_index = batch_column_index(batch, right_column)?;
    let left = batch.column(left_index);
    let right = batch.column(right_index);
    let values = (0..batch.num_rows())
        .map(|row| {
            let (Some(left), Some(right)) =
                (semijoin_key_at(left, row)?, semijoin_key_at(right, row)?)
            else {
                return Ok(Some(false));
            };
            let matched = keys.values.contains(&(left, right));
            Ok(Some(if negated { !matched } else { matched }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanArray::from(values))
}

fn semijoin_i64_pair_mask(
    batch: &RecordBatch,
    left_column: &str,
    right_column: &str,
    keys: &SemijoinI64PairKeys,
    negated: bool,
) -> Result<BooleanArray> {
    let left_index = batch_column_index(batch, left_column)?;
    let right_index = batch_column_index(batch, right_column)?;
    let left = batch.column(left_index);
    let right = batch.column(right_index);
    if negated {
        semijoin_i64_pair_anti_membership_mask_for_arrays(left, right, &keys.values)
    } else {
        semijoin_i64_pair_membership_mask_for_arrays(left, right, &keys.values)
    }
}

struct SemijoinI64PairKeys {
    values: SemijoinI64PairSet,
    has_null: bool,
}

enum SemijoinI64PairSet {
    Pair(FastHashSet<(i64, i64)>),
    PackedI32(FastHashSet<u64>),
    DenseI32ToI32 {
        min: i32,
        values: Vec<i32>,
        present: Vec<u8>,
    },
}

impl SemijoinI64PairSet {
    fn for_arrays(left: &ArrayRef, right: &ArrayRef) -> Self {
        if left.as_any().is::<Int32Array>() && right.as_any().is::<Int32Array>() {
            Self::PackedI32(FastHashSet::default())
        } else {
            Self::empty_pair()
        }
    }

    fn empty_pair() -> Self {
        Self::Pair(FastHashSet::default())
    }

    fn insert_from_arrays(&mut self, left: &ArrayRef, right: &ArrayRef) -> Result<()> {
        match self {
            Self::DenseI32ToI32 { .. } => Err(DodamError::UnsupportedSql(
                "dense integer semijoin set cannot be extended from Arrow arrays".to_string(),
            )),
            Self::PackedI32(values) => {
                if let (Some(left), Some(right)) = (
                    left.as_any().downcast_ref::<Int32Array>(),
                    right.as_any().downcast_ref::<Int32Array>(),
                ) {
                    insert_semijoin_packed_i32_pairs(values, left, right);
                    return Ok(());
                }
                let mut pairs = FastHashSet::default();
                insert_semijoin_i64_pairs_from_arrays(&mut pairs, left, right)?;
                *self = Self::Pair(pairs);
                Ok(())
            }
            Self::Pair(values) => insert_semijoin_i64_pairs_from_arrays(values, left, right),
        }
    }
}

#[inline]
fn pack_i32_pair(left: i32, right: i32) -> u64 {
    (u64::from(left as u32) << 32) | u64::from(right as u32)
}

fn insert_semijoin_packed_i32_pairs(
    values: &mut FastHashSet<u64>,
    left: &Int32Array,
    right: &Int32Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert(pack_i32_pair(left.value(row), right.value(row)));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert(pack_i32_pair(left.value(row), right.value(row)));
        }
    }
}

fn insert_semijoin_i64_pairs_from_arrays(
    values: &mut FastHashSet<(i64, i64)>,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<()> {
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        insert_semijoin_i32_i32_pairs(values, left, right);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        insert_semijoin_i32_i64_pairs(values, left, right);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        insert_semijoin_i64_i32_pairs(values, left, right);
        return Ok(());
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        insert_semijoin_i64_i64_pairs(values, left, right);
        return Ok(());
    }
    for row in 0..left.len() {
        let Some(left) = semijoin_i64_key_at(left, row)? else {
            continue;
        };
        let Some(right) = semijoin_i64_key_at(right, row)? else {
            continue;
        };
        values.insert((left, right));
    }
    Ok(())
}

fn insert_semijoin_i32_i32_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int32Array,
    right: &Int32Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((i64::from(left.value(row)), i64::from(right.value(row))));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((i64::from(left.value(row)), i64::from(right.value(row))));
        }
    }
}

fn insert_semijoin_i32_i64_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int32Array,
    right: &Int64Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((i64::from(left.value(row)), right.value(row)));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((i64::from(left.value(row)), right.value(row)));
        }
    }
}

fn insert_semijoin_i64_i32_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int64Array,
    right: &Int32Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((left.value(row), i64::from(right.value(row))));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((left.value(row), i64::from(right.value(row))));
        }
    }
}

fn insert_semijoin_i64_i64_pairs(
    values: &mut FastHashSet<(i64, i64)>,
    left: &Int64Array,
    right: &Int64Array,
) {
    if left.null_count() == 0 && right.null_count() == 0 {
        values.reserve(left.len());
        for row in 0..left.len() {
            values.insert((left.value(row), right.value(row)));
        }
        return;
    }
    for row in 0..left.len() {
        if left.is_valid(row) && right.is_valid(row) {
            values.insert((left.value(row), right.value(row)));
        }
    }
}

fn semijoin_i64_pair_membership_mask_for_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    keys: &SemijoinI64PairSet,
) -> Result<BooleanArray> {
    if let SemijoinI64PairSet::DenseI32ToI32 {
        min,
        values,
        present,
    } = keys
        && let (Some(left), Some(right)) = (
            left.as_any().downcast_ref::<Int32Array>(),
            right.as_any().downcast_ref::<Int32Array>(),
        )
    {
        return Ok(semijoin_dense_i32_to_i32_membership_mask(
            left, right, *min, values, present,
        ));
    }
    if let SemijoinI64PairSet::PackedI32(keys) = keys
        && let (Some(left), Some(right)) = (
            left.as_any().downcast_ref::<Int32Array>(),
            right.as_any().downcast_ref::<Int32Array>(),
        )
    {
        return Ok(semijoin_packed_i32_pair_membership_mask(left, right, keys));
    }
    let SemijoinI64PairSet::Pair(keys) = keys else {
        return Err(DodamError::UnsupportedSql(
            "integer semijoin key type changed across batches".to_string(),
        ));
    };
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        return Ok(semijoin_i32_i32_pair_membership_mask(left, right, keys));
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int32Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        return Ok(semijoin_i32_i64_pair_membership_mask(left, right, keys));
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int32Array>(),
    ) {
        return Ok(semijoin_i64_i32_pair_membership_mask(left, right, keys));
    }
    if let (Some(left), Some(right)) = (
        left.as_any().downcast_ref::<Int64Array>(),
        right.as_any().downcast_ref::<Int64Array>(),
    ) {
        return Ok(semijoin_i64_i64_pair_membership_mask(left, right, keys));
    }
    let values = (0..left.len())
        .map(|row| {
            let Some(left) = semijoin_i64_key_at(left, row)? else {
                return Ok(Some(false));
            };
            let Some(right) = semijoin_i64_key_at(right, row)? else {
                return Ok(Some(false));
            };
            Ok(Some(keys.contains(&(left, right))))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanArray::from(values))
}

fn semijoin_dense_i32_to_i32_membership_mask(
    left: &Int32Array,
    right: &Int32Array,
    min: i32,
    values: &[i32],
    present: &[u8],
) -> BooleanArray {
    let mut output = Vec::with_capacity(left.len());
    for row in 0..left.len() {
        if !left.is_valid(row) || !right.is_valid(row) {
            output.push(Some(false));
            continue;
        }
        let matched = i64::from(left.value(row))
            .checked_sub(i64::from(min))
            .and_then(|offset| usize::try_from(offset).ok())
            .is_some_and(|index| {
                present.get(index).is_some_and(|byte| *byte != 0)
                    && values
                        .get(index)
                        .is_some_and(|value| *value == right.value(row))
            });
        output.push(Some(matched));
    }
    BooleanArray::from(output)
}

fn semijoin_i64_pair_anti_membership_mask_for_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    keys: &SemijoinI64PairSet,
) -> Result<BooleanArray> {
    let membership = semijoin_i64_pair_membership_mask_for_arrays(left, right, keys)?;
    Ok(BooleanArray::from(
        membership
            .iter()
            .enumerate()
            .map(|(row, value)| {
                if left.is_null(row) || right.is_null(row) {
                    Some(false)
                } else {
                    value.map(|matched| !matched)
                }
            })
            .collect::<Vec<_>>(),
    ))
}

fn semijoin_packed_i32_pair_membership_mask(
    left: &Int32Array,
    right: &Int32Array,
    keys: &FastHashSet<u64>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&pack_i32_pair(left.value(row), right.value(row)))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&pack_i32_pair(left.value(row), right.value(row))))
            })
            .collect::<Vec<_>>(),
    )
}

fn semijoin_i32_i32_pair_membership_mask(
    left: &Int32Array,
    right: &Int32Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(i64::from(left.value(row)), i64::from(right.value(row))))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row)).then(|| {
                    keys.contains(&(i64::from(left.value(row)), i64::from(right.value(row))))
                })
            })
            .collect::<Vec<_>>(),
    )
}

fn semijoin_i32_i64_pair_membership_mask(
    left: &Int32Array,
    right: &Int64Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(i64::from(left.value(row)), right.value(row)))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&(i64::from(left.value(row)), right.value(row))))
            })
            .collect::<Vec<_>>(),
    )
}

fn semijoin_i64_i32_pair_membership_mask(
    left: &Int64Array,
    right: &Int32Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(left.value(row), i64::from(right.value(row))))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&(left.value(row), i64::from(right.value(row)))))
            })
            .collect::<Vec<_>>(),
    )
}

fn semijoin_i64_i64_pair_membership_mask(
    left: &Int64Array,
    right: &Int64Array,
    keys: &FastHashSet<(i64, i64)>,
) -> BooleanArray {
    if left.null_count() == 0 && right.null_count() == 0 {
        return boolean_array_no_nulls_from_len(left.len(), |row| {
            keys.contains(&(left.value(row), right.value(row)))
        });
    }
    BooleanArray::from(
        (0..left.len())
            .map(|row| {
                (left.is_valid(row) && right.is_valid(row))
                    .then(|| keys.contains(&(left.value(row), right.value(row))))
            })
            .collect::<Vec<_>>(),
    )
}

fn invert_boolean_array(array: &BooleanArray) -> BooleanArray {
    BooleanArray::from(
        array
            .iter()
            .map(|value| value.map(|value| !value))
            .collect::<Vec<_>>(),
    )
}

fn semijoin_outer_projection(
    projection: &ParsedProjection,
    group_by: &[String],
    order_by: Option<&SortKey>,
    key_column: &str,
    filter: Option<&FilterExpr>,
) -> Projection {
    let mut columns = vec![key_column.to_string()];
    for column in group_by {
        add_column_once(&mut columns, column.clone());
    }
    match &projection.projection {
        Projection::All => return Projection::All,
        Projection::Columns(projected) => {
            for column in projected {
                add_column_once(&mut columns, column.clone());
            }
        }
    }
    for aggregate in &projection.aggregates {
        if let Some(column) = aggregate.referenced_column() {
            add_column_once(&mut columns, column.to_string());
        }
    }
    for expression in &projection.aggregate_expressions {
        for column in scalar_expression_columns(&expression.expr) {
            add_column_once(&mut columns, column);
        }
    }
    for expression in &projection.expressions {
        for column in scalar_expression_columns(&expression.expr) {
            add_column_once(&mut columns, column);
        }
    }
    if let Some(order_by) = order_by {
        for sort in &order_by.expressions {
            add_column_once(&mut columns, sort.column.clone());
        }
    }
    if let Some(filter) = filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut columns, column);
        }
    }
    Projection::Columns(columns)
}

pub(super) fn semijoin_key_at(column: &ArrayRef, row: usize) -> Result<Option<String>> {
    if column.is_null(row) {
        return Ok(None);
    }
    Ok(Some(sql_literal(&literal_value_from_array(column, row)?)))
}

enum SemijoinKeySet {
    Empty {
        has_null: bool,
    },
    Int64 {
        values: HashSet<i64>,
        has_null: bool,
    },
    Utf8 {
        values: HashSet<String>,
        has_null: bool,
    },
    Literal {
        values: HashSet<String>,
        has_null: bool,
    },
}

impl SemijoinKeySet {
    fn for_data_type(data_type: &DataType) -> Self {
        match data_type {
            DataType::Int32 | DataType::Int64 | DataType::Date32 | DataType::Date64 => {
                Self::Int64 {
                    values: HashSet::new(),
                    has_null: false,
                }
            }
            DataType::Utf8 => Self::Utf8 {
                values: HashSet::new(),
                has_null: false,
            },
            _ => Self::Literal {
                values: HashSet::new(),
                has_null: false,
            },
        }
    }

    fn rhs_empty() -> Self {
        Self::Empty { has_null: false }
    }

    fn has_null(&self) -> bool {
        match self {
            Self::Empty { has_null }
            | Self::Int64 { has_null, .. }
            | Self::Utf8 { has_null, .. }
            | Self::Literal { has_null, .. } => *has_null,
        }
    }

    fn note_nulls(&mut self, null_count: usize) {
        if null_count == 0 {
            return;
        }
        match self {
            Self::Empty { has_null }
            | Self::Int64 { has_null, .. }
            | Self::Utf8 { has_null, .. }
            | Self::Literal { has_null, .. } => *has_null = true,
        }
    }

    fn insert_from_array(&mut self, column: &ArrayRef) -> Result<()> {
        self.note_nulls(column.null_count());
        match self {
            Self::Empty { .. } => Ok(()),
            Self::Int64 { values, .. } => {
                if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
                    if array.null_count() == 0 {
                        values.extend(array.values().iter().map(|value| i64::from(*value)));
                    } else {
                        values.extend(array.iter().flatten().map(i64::from));
                    }
                    return Ok(());
                }
                if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                    if array.null_count() == 0 {
                        values.extend(array.values().iter().copied());
                    } else {
                        values.extend(array.iter().flatten());
                    }
                    return Ok(());
                }
                for row in 0..column.len() {
                    self.insert_from_column(column, row)?;
                }
                Ok(())
            }
            Self::Utf8 { values, .. } => {
                if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
                    if array.null_count() == 0 {
                        values.extend(array.iter().flatten().map(str::to_string));
                    } else {
                        values.extend(array.iter().flatten().map(str::to_string));
                    }
                    return Ok(());
                }
                for row in 0..column.len() {
                    self.insert_from_column(column, row)?;
                }
                Ok(())
            }
            Self::Literal { .. } => {
                for row in 0..column.len() {
                    self.insert_from_column(column, row)?;
                }
                Ok(())
            }
        }
    }

    fn insert_from_column(&mut self, column: &ArrayRef, row: usize) -> Result<()> {
        if column.is_null(row) {
            self.note_nulls(1);
            return Ok(());
        }
        match self {
            Self::Empty { .. } => {}
            Self::Int64 { values, .. } => {
                if let Some(value) = semijoin_i64_key_at(column, row)? {
                    values.insert(value);
                }
            }
            Self::Utf8 { values, .. } => {
                if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                    values.insert(strings.value(row).to_string());
                } else if let Some(value) = semijoin_key_at(column, row)? {
                    values.insert(value);
                }
            }
            Self::Literal { values, .. } => {
                if let Some(value) = semijoin_key_at(column, row)? {
                    values.insert(value);
                }
            }
        }
        Ok(())
    }

    fn membership_mask(&self, column: &ArrayRef) -> Result<BooleanArray> {
        match self {
            Self::Empty { .. } => return Ok(BooleanArray::from(vec![Some(false); column.len()])),
            Self::Int64 { values, .. } => {
                if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
                    return Ok(BooleanArray::from(
                        array
                            .iter()
                            .map(|value| value.map(|value| values.contains(&i64::from(value))))
                            .collect::<Vec<_>>(),
                    ));
                }
                if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                    return Ok(BooleanArray::from(
                        array
                            .iter()
                            .map(|value| value.map(|value| values.contains(&value)))
                            .collect::<Vec<_>>(),
                    ));
                }
            }
            Self::Utf8 { values, .. } => {
                if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
                    return Ok(BooleanArray::from(
                        array
                            .iter()
                            .map(|value| value.map(|value| values.contains(value)))
                            .collect::<Vec<_>>(),
                    ));
                }
            }
            Self::Literal { .. } => {}
        }
        let values = (0..column.len())
            .map(|row| self.contains_column_value(column, row))
            .collect::<Result<Vec<_>>>()?;
        Ok(BooleanArray::from(values))
    }

    fn anti_membership_mask(&self, column: &ArrayRef) -> Result<BooleanArray> {
        if self.has_null() {
            return Ok(BooleanArray::from(vec![Some(false); column.len()]));
        }
        let membership = self.membership_mask(column)?;
        Ok(BooleanArray::from(
            membership
                .iter()
                .map(|value| value.map(|matched| !matched))
                .collect::<Vec<_>>(),
        ))
    }

    fn contains_column_value(&self, column: &ArrayRef, row: usize) -> Result<Option<bool>> {
        if column.is_null(row) {
            return Ok(Some(false));
        }
        match self {
            Self::Empty { .. } => Ok(Some(false)),
            Self::Int64 { values, .. } => Ok(Some(
                semijoin_i64_key_at(column, row)?.is_some_and(|value| values.contains(&value)),
            )),
            Self::Utf8 { values, .. } => {
                if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                    Ok(Some(values.contains(strings.value(row))))
                } else {
                    Ok(Some(
                        semijoin_key_at(column, row)?.is_some_and(|value| values.contains(&value)),
                    ))
                }
            }
            Self::Literal { values, .. } => Ok(Some(
                semijoin_key_at(column, row)?.is_some_and(|value| values.contains(&value)),
            )),
        }
    }

    fn to_small_int64_in_list_expr(
        &self,
        column: &str,
        negated: bool,
        max_values: usize,
    ) -> Option<Expr> {
        match self {
            Self::Empty { has_null } => Some(Expr::InList {
                column: column.to_string(),
                values: Vec::new(),
                negated,
                has_null: *has_null,
            }),
            Self::Int64 { values, has_null } if values.len() <= max_values => {
                let mut values = values.iter().copied().collect::<Vec<_>>();
                values.sort_unstable();
                Some(Expr::InList {
                    column: column.to_string(),
                    values: values.into_iter().map(LiteralValue::Int64).collect(),
                    negated,
                    has_null: *has_null,
                })
            }
            _ => None,
        }
    }
}

fn semijoin_i64_key_at(column: &ArrayRef, row: usize) -> Result<Option<i64>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 semijoin key")
                .value(row),
        ))),
        DataType::Int64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 semijoin key")
                .value(row),
        )),
        DataType::Date32 => Ok(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 semijoin key")
                .value(row),
        ))),
        DataType::Date64 => Ok(Some(
            column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64 semijoin key")
                .value(row),
        )),
        data_type => Err(DodamError::UnsupportedSql(format!(
            "cannot use {data_type} as integer semijoin key"
        ))),
    }
}

pub(super) async fn try_execute_correlated_subquery_filter_sql(
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
    if !expr_contains_materializable_subquery(selection) {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 1 {
        return Ok(None);
    }
    let path = parse_from(select)?;
    let selection_sql = selection.to_string();
    if let Some(outer_alias) = path.alias.as_deref()
        && !subquery_references_outer_alias(&selection_sql, outer_alias)
        && !tpch_alias_prefix(outer_alias)
            .is_some_and(|prefix| selection_sql.contains(&format!("{prefix}_")))
    {
        return Ok(None);
    }
    if path.alias.is_none() {
        let inferred_alias = table_ref_alias_or_name(&path);
        let Some(prefix) = inferred_alias.chars().next() else {
            return Ok(None);
        };
        if !selection_sql.contains(&format!("{prefix}_")) {
            return Ok(None);
        }
    }

    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if parse_distinct(select)? {
        return Err(DodamError::UnsupportedSql(
            "correlated subquery filters with DISTINCT are not supported yet".to_string(),
        ));
    }
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;

    let stream = engine
        .scan_parquet_batches(path.path, batch_size, None, Projection::All, None)
        .await?;
    let outer_batches = collect_batches(stream)?;
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            &selection_sql,
            path.alias.as_deref(),
            &batch,
            path.alias.as_deref(),
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, 0)?;
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

pub(super) async fn evaluate_correlated_subquery_filter_mask(
    engine: &DodamEngine,
    selection_sql: &str,
    outer_alias: Option<&str>,
    batch: &RecordBatch,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<BooleanArray> {
    let selection_expr = parse_sql_expr_fragment(selection_sql)?;
    let selection_expr = Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
        engine,
        selection_expr,
        batch_size,
    ))
    .await?;
    let selection_sql = selection_expr.to_string();
    let mut values = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let bound_sql = bind_outer_row_references(&selection_sql, outer_alias, batch, row)?;
        let bound_expr = parse_sql_expr_fragment(&bound_sql)?;
        let row_batch = batch.slice(row, 1);
        let matches = evaluate_bound_correlated_filter(
            engine,
            &bound_expr,
            &row_batch,
            table_alias,
            batch_size,
        )
        .await?;
        values.push(Some(matches));
    }
    Ok(BooleanArray::from(values))
}

pub(super) async fn apply_correlated_subquery_filter_batches(
    engine: &DodamEngine,
    batches: Vec<RecordBatch>,
    selection_sql: &str,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let mut filtered = Vec::new();
    for batch in batches {
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            selection_sql,
            None,
            &batch,
            None,
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    Ok(filtered)
}

pub(super) async fn rewrite_uncorrelated_scalar_subqueries_to_literals(
    engine: &DodamEngine,
    expr: SqlExpr,
    batch_size: usize,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Exists { subquery, negated } => {
            match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                Ok(output) => {
                    let exists = query_output_batches(output)?
                        .iter()
                        .any(|batch| batch.num_rows() > 0);
                    Ok(SqlExpr::Value(
                        Value::Boolean(if negated { !exists } else { exists }).with_empty_span(),
                    ))
                }
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
                    Ok(SqlExpr::Exists { subquery, negated })
                }
                Err(error) => Err(error),
            }
        }
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
            Ok(output) => {
                let values =
                    literal_values_from_single_column_batches(query_output_batches(output)?)?;
                if values.is_empty() {
                    return Ok(SqlExpr::Value(Value::Boolean(negated).with_empty_span()));
                }
                Ok(SqlExpr::InList {
                    expr,
                    list: values.into_iter().map(literal_value_to_sql_expr).collect(),
                    negated,
                })
            }
            Err(DodamError::UnsupportedSql(_))
            | Err(DodamError::UnknownColumn(_))
            | Err(DodamError::UnknownTableQualifier(_)) => Ok(SqlExpr::InSubquery {
                expr,
                subquery,
                negated,
            }),
            Err(error) => Err(error),
        },
        SqlExpr::Subquery(subquery) => {
            match Box::pin(execute_sql(engine, &subquery.to_string(), batch_size)).await {
                Ok(output) => scalar_literal_value_from_batches(query_output_batches(output)?)
                    .map(literal_value_to_sql_expr),
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => Ok(SqlExpr::Subquery(subquery)),
                Err(error) => Err(error),
            }
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *left, batch_size,
                ))
                .await?,
            ),
            op,
            right: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *right, batch_size,
                ))
                .await?,
            ),
        }),
        SqlExpr::Nested(expr) => Ok(SqlExpr::Nested(Box::new(
            Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                engine, *expr, batch_size,
            ))
            .await?,
        ))),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op,
            expr: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            ),
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let expr = Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            );
            let mut rewritten = Vec::with_capacity(list.len());
            for item in list {
                rewritten.push(
                    Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                        engine, item, batch_size,
                    ))
                    .await?,
                );
            }
            Ok(SqlExpr::InList {
                expr,
                list: rewritten,
                negated,
            })
        }
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(SqlExpr::Between {
            expr: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            ),
            negated,
            low: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *low, batch_size,
                ))
                .await?,
            ),
            high: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *high, batch_size,
                ))
                .await?,
            ),
        }),
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            special,
            shorthand,
        } => Ok(SqlExpr::Substring {
            expr: Box::new(
                Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                    engine, *expr, batch_size,
                ))
                .await?,
            ),
            substring_from: match substring_from {
                Some(expr) => Some(Box::new(
                    Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                        engine, *expr, batch_size,
                    ))
                    .await?,
                )),
                None => None,
            },
            substring_for: match substring_for {
                Some(expr) => Some(Box::new(
                    Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                        engine, *expr, batch_size,
                    ))
                    .await?,
                )),
                None => None,
            },
            special,
            shorthand,
        }),
        expr => Ok(expr),
    }
}

async fn evaluate_bound_correlated_filter(
    engine: &DodamEngine,
    expr: &SqlExpr,
    row_batch: &RecordBatch,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<bool> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(expr, &mut conjuncts);
    for conjunct in conjuncts {
        let conjunct = if expr_contains_materializable_subquery(&conjunct) {
            Box::pin(rewrite_uncorrelated_scalar_subqueries_to_literals(
                engine, conjunct, batch_size,
            ))
            .await?
        } else {
            conjunct
        };
        let matches = if let Ok(mask) = evaluate_scalar_predicate(row_batch, &conjunct, table_alias)
        {
            mask.is_valid(0) && mask.value(0)
        } else if let Some(expr) = Box::pin(parse_filter_with_subqueries(
            engine,
            &conjunct,
            &[],
            table_alias,
            false,
            batch_size,
        ))
        .await?
        {
            filter_batch(row_batch.clone(), &FilterExpr::new(expr))?.num_rows() > 0
        } else {
            true
        };
        if !matches {
            return Ok(false);
        }
    }
    Ok(true)
}

fn parse_sql_expr_fragment(expr: &str) -> Result<SqlExpr> {
    let dialect = GenericDialect {};
    let sql = format!("SELECT * FROM dummy WHERE {expr}");
    let statements = Parser::parse_sql(&dialect, &sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "failed to parse SQL expression".to_string(),
        ));
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(DodamError::UnsupportedSql(
            "failed to parse SQL expression".to_string(),
        ));
    };
    select
        .selection
        .clone()
        .ok_or_else(|| DodamError::UnsupportedSql("failed to parse SQL expression".to_string()))
}

pub(super) async fn try_execute_correlated_exists_subquery_sql(
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
    let path = parse_from(select)?;
    let subquery_sql = subquery.to_string();
    let Some(outer_alias) = path.alias.as_deref() else {
        return Ok(None);
    };
    if !subquery_references_outer_alias(&subquery_sql, outer_alias) {
        return Ok(None);
    }

    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    if parse_distinct(select)?
        || !parsed_projection.aggregates.is_empty()
        || !group_by.is_empty()
        || select.having.is_some()
    {
        return Err(DodamError::UnsupportedSql(
            "correlated EXISTS currently supports only non-aggregate SELECT queries".to_string(),
        ));
    }
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;

    let stream = engine
        .scan_parquet_batches(path.path, batch_size, None, Projection::All, None)
        .await?;
    let outer_batches = collect_batches(stream)?;
    let mut filtered = Vec::new();
    for batch in outer_batches {
        let mask = evaluate_correlated_exists_mask(
            engine,
            &subquery_sql,
            outer_alias,
            &batch,
            negated,
            batch_size,
        )
        .await?;
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit, 0)?;
    batches = apply_output_projection(batches, &parsed_projection.projection)?;
    batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

fn subquery_references_outer_alias(subquery_sql: &str, outer_alias: &str) -> bool {
    subquery_sql.contains(&format!("{outer_alias}."))
        || subquery_sql.contains(&format!("{outer_alias} ."))
}

async fn evaluate_correlated_exists_mask(
    engine: &DodamEngine,
    subquery_sql: &str,
    outer_alias: &str,
    batch: &RecordBatch,
    negated: bool,
    batch_size: usize,
) -> Result<BooleanArray> {
    let mut values = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let bound_sql = bind_outer_row_references(subquery_sql, Some(outer_alias), batch, row)?;
        let output = Box::pin(execute_sql(engine, &bound_sql, batch_size)).await?;
        let exists = query_output_batches(output)?
            .iter()
            .any(|batch| batch.num_rows() > 0);
        values.push(Some(if negated { !exists } else { exists }));
    }
    Ok(BooleanArray::from(values))
}

fn bind_outer_row_references(
    subquery_sql: &str,
    outer_alias: Option<&str>,
    batch: &RecordBatch,
    row: usize,
) -> Result<String> {
    let mut sql = subquery_sql.to_string();
    for (column_index, field) in batch.schema().fields().iter().enumerate() {
        let literal = if batch.column(column_index).is_null(row) {
            LiteralValue::Null
        } else {
            literal_value_from_array(batch.column(column_index), row)?
        };
        let literal = sql_literal(&literal);
        if let Some(outer_alias) = outer_alias {
            sql = sql.replace(&format!("{outer_alias}.{}", field.name()), &literal);
            sql = sql.replace(&format!("{outer_alias} . {}", field.name()), &literal);
            if unqualified_column_matches_table_alias(field.name(), Some(outer_alias)) {
                sql = replace_identifier_tokens_preserving_local_subqueries(
                    &sql,
                    field.name(),
                    &literal,
                );
            }
        } else {
            sql =
                replace_identifier_tokens_preserving_local_subqueries(&sql, field.name(), &literal);
            if let Some((_, unqualified)) = field.name().rsplit_once('.') {
                sql = replace_identifier_tokens_preserving_local_subqueries(
                    &sql,
                    unqualified,
                    &literal,
                );
            }
        }
    }
    Ok(sql)
}

fn replace_identifier_tokens_preserving_local_subqueries(
    input: &str,
    identifier: &str,
    replacement: &str,
) -> String {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_single_quote = false;
    while index < input.len() {
        let rest = &input[index..];
        if !in_single_quote
            && rest.starts_with('(')
            && rest
                .get(1..)
                .is_some_and(|tail| tail.trim_start().to_ascii_lowercase().starts_with("select"))
            && let Some(end) = matching_parenthesis_end(input, index)
        {
            let segment = &input[index..end];
            if subquery_protects_identifier_prefix(segment, identifier) {
                output.push_str(segment);
            } else {
                output.push_str(&replace_identifier_tokens(segment, identifier, replacement));
            }
            index = end;
            continue;
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if ch == '\'' {
            output.push(ch);
            index += ch.len_utf8();
            in_single_quote = !in_single_quote;
            continue;
        }
        if !in_single_quote
            && rest.starts_with(identifier)
            && input[..index]
                .chars()
                .next_back()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
            && input[index + identifier.len()..]
                .chars()
                .next()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
        {
            output.push_str(replacement);
            index += identifier.len();
            continue;
        }
        output.push(ch);
        index += ch.len_utf8();
    }
    output
}

fn matching_parenthesis_end(input: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_single_quote = false;
    for (offset, ch) in input[start..].char_indices() {
        if ch == '\'' {
            in_single_quote = !in_single_quote;
        } else if !in_single_quote && ch == '(' {
            depth += 1;
        } else if !in_single_quote && ch == ')' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(start + offset + ch.len_utf8());
            }
        }
    }
    None
}

fn subquery_protects_identifier_prefix(subquery_sql: &str, identifier: &str) -> bool {
    let Some((prefix, _)) = identifier.split_once('_') else {
        return false;
    };
    subquery_local_prefixes(subquery_sql)
        .iter()
        .any(|local_prefix| local_prefix.eq_ignore_ascii_case(prefix))
}

fn subquery_local_prefixes(subquery_sql: &str) -> Vec<String> {
    let lowercase = subquery_sql.to_ascii_lowercase();
    let Some(from_index) = lowercase.find(" from ") else {
        return Vec::new();
    };
    let mut prefixes = Vec::new();
    let after_from = &subquery_sql[from_index + " from ".len()..];
    let end = after_from
        .to_ascii_lowercase()
        .find(" where ")
        .or_else(|| after_from.to_ascii_lowercase().find(" group "))
        .or_else(|| after_from.to_ascii_lowercase().find(" order "))
        .unwrap_or(after_from.len());
    for relation in after_from[..end].split(',') {
        let Some(table_token) = relation.split_whitespace().next() else {
            continue;
        };
        let table_name = table_token
            .trim_matches('\'')
            .trim_matches('"')
            .rsplit('/')
            .next()
            .unwrap_or(table_token);
        let table_name = table_name.strip_suffix(".parquet").unwrap_or(table_name);
        if let Some(prefix) = tpch_alias_prefix(table_name) {
            add_column_once(&mut prefixes, prefix.to_string());
        } else if let Some(prefix) = table_name.split('_').next()
            && let Some(initial) = prefix.chars().next()
        {
            add_column_once(&mut prefixes, initial.to_string());
        }
    }
    prefixes
}

fn replace_identifier_tokens(input: &str, identifier: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_single_quote = false;
    while index < input.len() {
        let rest = &input[index..];
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if ch == '\'' {
            output.push(ch);
            index += ch.len_utf8();
            in_single_quote = !in_single_quote;
            continue;
        }
        if !in_single_quote
            && rest.starts_with(identifier)
            && input[..index]
                .chars()
                .next_back()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
            && input[index + identifier.len()..]
                .chars()
                .next()
                .is_none_or(|ch| !is_sql_identifier_char(ch))
        {
            output.push_str(replacement);
            index += identifier.len();
            continue;
        }
        output.push(ch);
        index += ch.len_utf8();
    }
    output
}

fn is_sql_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn sql_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::Null => "NULL".to_string(),
        LiteralValue::Boolean(value) => value.to_string(),
        LiteralValue::Int64(value) => value.to_string(),
        LiteralValue::Float64(value) => value.to_string(),
        LiteralValue::Utf8(value) => format!("'{}'", value.replace('\'', "''")),
    }
}

pub(super) async fn try_execute_in_subquery_sql(
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
    if !select
        .selection
        .as_ref()
        .is_some_and(expr_contains_materializable_subquery)
    {
        return Ok(None);
    }
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
            "IN subquery filters over JOIN inputs are not supported yet".to_string(),
        ));
    }

    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    let distinct = parse_distinct(select)?;
    let selection = select.selection.as_ref().expect("selection checked");
    let rewritten_selection = rewrite_same_source_distinct_in_subquery(selection, &path)?
        .unwrap_or_else(|| selection.clone());
    let selection = &rewritten_selection;
    if let Some(output) = try_execute_uncorrelated_i64_pair_in_semijoin_sql(
        engine, query, select, &path, selection, batch_size,
    )
    .await?
    {
        return Ok(Some(output));
    }
    if let Some(output) = try_execute_uncorrelated_in_key_semijoin_sql(
        engine, query, select, &path, selection, batch_size,
    )
    .await?
    {
        return Ok(Some(output));
    }
    let filter_requires_expression = predicate_requires_expression_path(selection);
    let (filter, expression_filters) = if filter_requires_expression {
        split_subquery_and_expression_filters(engine, selection, path.alias.as_deref(), batch_size)
            .await?
    } else {
        (
            Box::pin(parse_filter_with_subqueries(
                engine,
                selection,
                &[],
                path.alias.as_deref(),
                false,
                batch_size,
            ))
            .await?
            .map(FilterExpr::new),
            Vec::new(),
        )
    };
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

    let mut scan_projection = parsed_projection.projection.clone();
    if !parsed_projection.aggregates.is_empty() {
        for aggregate in &parsed_projection.aggregates {
            if let Some(column) = aggregate.referenced_column() {
                add_projection_columns(&mut scan_projection, vec![column.to_string()]);
            }
        }
        for expression in &parsed_projection.aggregate_expressions {
            for column in scalar_expression_columns(&expression.expr) {
                add_projection_columns(&mut scan_projection, vec![column]);
            }
        }
    }
    if filter_requires_expression {
        add_projection_columns(
            &mut scan_projection,
            predicate_expression_columns(selection, path.alias.as_deref())?,
        );
    }

    if !distinct
        && expression_filters.is_empty()
        && parsed_projection.aggregates.is_empty()
        && group_by.is_empty()
        && select.having.is_none()
        && monotonic_order_limit_scan_enabled()
        && limit.is_some()
        && Path::new(&path.path).exists()
        && let Some(column) = monotonic_stream_limit_column(order_by.as_ref())
        && engine
            .parquet_row_groups_monotonic_by_column(path.path.clone(), &column)
            .await?
    {
        let stream = engine
            .scan_parquet_filtered_batches_preserve_order(
                path.path.clone(),
                batch_size,
                scan_projection.clone(),
                filter.clone(),
            )
            .await?;
        if let Some(mut batches) =
            collect_verified_monotonic_order_limit_batches(stream, &column, limit, 0)?
        {
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
            return Ok(Some(QueryOutput::Scan { batches }));
        }
    }

    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        let stream = engine
            .scan_parquet_batches(path.path, batch_size, None, scan_projection, filter)
            .await?;
        let mut batches = collect_batches(stream)?;
        for expression_filter in expression_filters {
            batches =
                apply_output_expression_filter(batches, &expression_filter, path.alias.as_deref())?;
        }
        let stream = Box::new(MemoryExec::new(batches)).execute()?;
        let metrics = collect_aggregates_with_optional_expression_views(
            stream,
            1,
            &group_by,
            &parsed_projection.aggregates,
            &parsed_projection.filtered_aggregates,
            &parsed_projection.aggregate_expressions,
        )?;
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        let having = select
            .having
            .as_ref()
            .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
            .transpose()?;
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

    let stream = if distinct {
        engine
            .scan_parquet_distinct_batches(
                path.path,
                batch_size,
                limit,
                scan_projection,
                filter,
                order_by,
            )
            .await?
    } else if let Some(order_by) = order_by {
        engine
            .scan_parquet_ordered_batches_by(
                path.path,
                batch_size,
                limit,
                scan_projection,
                filter,
                order_by,
            )
            .await?
    } else {
        engine
            .scan_parquet_batches(path.path, batch_size, limit, scan_projection, filter)
            .await?
    };
    let mut batches = collect_batches(stream)?;
    for expression_filter in expression_filters {
        batches =
            apply_output_expression_filter(batches, &expression_filter, path.alias.as_deref())?;
    }
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

async fn try_execute_uncorrelated_i64_pair_in_semijoin_sql(
    engine: &DodamEngine,
    query: &Query,
    select: &Select,
    outer_path: &SqlTableRef,
    selection: &SqlExpr,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((in_index, outer_expr, subquery, negated)) =
        conjuncts.iter().enumerate().find_map(|(index, expr)| {
            let (outer_expr, subquery, negated) = sql_in_subquery_parts(expr)?;
            Some((index, outer_expr, subquery, negated))
        })
    else {
        return Ok(None);
    };
    let outer_alias = table_ref_alias_or_name(outer_path);
    let Some((outer_left, outer_right)) =
        tuple_semijoin_columns(outer_expr, "", &outer_alias, SemijoinColumnOwner::Outer)?
    else {
        return Ok(None);
    };

    let SetExpr::Select(inner_select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    reject_query_features(subquery)?;
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
    let Some((inner_left, inner_right)) = projection_pair_semijoin_columns(
        inner_select,
        &inner_alias,
        &outer_alias,
        SemijoinColumnOwner::Inner,
    )?
    else {
        return Ok(None);
    };
    let inner_filter = inner_select
        .selection
        .as_ref()
        .map(|expr| {
            if predicate_requires_expression_path(expr)
                || expr_contains_materializable_subquery(expr)
            {
                return Ok(None);
            }
            parse_filter(expr, &[], inner_path.alias.as_deref(), false).map(Some)
        })
        .transpose()?
        .flatten();

    let outer_residual = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != in_index).then_some(conjunct))
        .collect::<Vec<_>>();
    if outer_residual.iter().any(|expr| {
        predicate_requires_expression_path(expr) || expr_contains_materializable_subquery(expr)
    }) {
        return Ok(None);
    }
    let outer_filter = combine_sql_and_conjuncts(outer_residual)
        .as_ref()
        .map(|expr| parse_filter(expr, &[], outer_path.alias.as_deref(), false))
        .transpose()?;

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
    let offset = parse_offset(query)?;
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
        &outer_left,
        outer_filter.as_ref(),
    );
    if let Projection::Columns(columns) = &mut outer_projection {
        add_column_once(columns, outer_right.clone());
    }

    let profile = semijoin_profile_enabled();
    let total_started = profile.then(Instant::now);

    if !negated
        && let Some(transferred_filter) = same_source_tuple_semijoin_transferred_filter(
            engine,
            outer_path,
            &inner_path,
            &outer_left,
            &outer_right,
            &inner_left,
            &inner_right,
            outer_filter.clone(),
            inner_filter.clone(),
        )?
        && !distinct
        && parsed_projection.aggregates.is_empty()
        && group_by.is_empty()
        && select.having.is_none()
        && offset == 0
        && monotonic_order_limit_scan_enabled()
        && limit.is_some()
        && Path::new(&outer_path.path).exists()
        && let Some(order_column) = monotonic_stream_limit_column(order_by.as_ref())
        && engine
            .parquet_row_groups_monotonic_by_column(outer_path.path.clone(), &order_column)
            .await?
    {
        let outer_started = profile.then(Instant::now);
        let stream = engine
            .scan_parquet_filtered_batches_preserve_order(
                outer_path.path.clone(),
                batch_size,
                outer_projection.clone(),
                Some(transferred_filter),
            )
            .await?;
        let mut batches = collect_ordered_stream_limit_batches(stream, limit, 0)?;
        let outer_elapsed = outer_started.map(|started| started.elapsed());
        let projection_started = profile.then(Instant::now);
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
        if let (true, Some(total_started)) = (profile, total_started) {
            let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            eprintln!(
                "[dodam:semijoin-profile] kind=tuple-in total={}ms inner=0ms outer={}ms projection={}ms output_rows={} negated={} branch=predicate-transfer",
                total_started.elapsed().as_millis(),
                outer_elapsed
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0),
                projection_started
                    .map(|started| started.elapsed().as_millis())
                    .unwrap_or(0),
                rows,
                negated,
            );
        }
        return Ok(Some(QueryOutput::Scan { batches }));
    }

    if std::env::var_os("DODAM_MIXED_TUPLE_PREFIX_PRECHECK").is_some()
        && !negated
        && !distinct
        && parsed_projection.aggregates.is_empty()
        && group_by.is_empty()
        && select.having.is_none()
        && !projection_requires_expression_path(&parsed_projection.expressions)
    {
        let precheck_started = profile.then(Instant::now);
        if let Some(candidate_keys) = collect_semijoin_mixed_prefix_candidate_keys_direct(
            engine,
            &inner_path.path,
            &inner_left,
            &inner_right,
            inner_filter.as_ref(),
            batch_size,
        )? && semijoin_mixed_prefix_precheck_accepts(engine, &outer_path.path, &candidate_keys)?
            && let Some(false) = direct_mixed_tuple_semijoin_outer_has_prefix_candidate(
                engine,
                &outer_path.path,
                batch_size,
                &outer_left,
                &outer_right,
                &candidate_keys,
                outer_filter.as_ref(),
            )?
        {
            if let (true, Some(total_started)) = (profile, total_started) {
                eprintln!(
                    "[dodam:semijoin-profile] kind=tuple-in total={}ms inner={}ms outer=0ms projection=0ms output_rows=0 negated={} branch=mixed-prefix-empty",
                    total_started.elapsed().as_millis(),
                    precheck_started
                        .map(|started| started.elapsed().as_millis())
                        .unwrap_or(0),
                    negated,
                );
            }
            return Ok(Some(QueryOutput::Scan {
                batches: Vec::new(),
            }));
        }
    }

    let inner_started = profile.then(Instant::now);
    let pair_set =
        if semijoin_filter_prefers_mixed_tuple(inner_filter.as_ref(), &inner_left, &inner_right)
            && let Some(values) = collect_semijoin_i64_utf8_pair_set(
                engine,
                inner_path.path.clone(),
                &inner_left,
                &inner_right,
                inner_filter.clone(),
                batch_size,
            )
            .await?
        {
            TupleSemijoinPairSet::I64Utf8(values)
        } else {
            match collect_semijoin_i64_pair_set(
                engine,
                inner_path.path.clone(),
                &inner_left,
                &inner_right,
                inner_filter.clone(),
                batch_size,
            )
            .await
            {
                Ok(values) => TupleSemijoinPairSet::I64(values),
                Err(DodamError::UnsupportedSql(message))
                    if message.contains("integer semijoin key") =>
                {
                    if let Some(values) = collect_semijoin_i64_utf8_pair_set(
                        engine,
                        inner_path.path.clone(),
                        &inner_left,
                        &inner_right,
                        inner_filter.clone(),
                        batch_size,
                    )
                    .await?
                    {
                        TupleSemijoinPairSet::I64Utf8(values)
                    } else {
                        TupleSemijoinPairSet::Literal(
                            collect_semijoin_literal_pair_set(
                                engine,
                                inner_path.path,
                                &inner_left,
                                &inner_right,
                                inner_filter,
                                batch_size,
                            )
                            .await?,
                        )
                    }
                }
                Err(error) => return Err(error),
            }
        };
    let inner_elapsed = inner_started.map(|started| started.elapsed());
    if negated && pair_set.has_null() {
        return Ok(None);
    }

    if !negated
        && !distinct
        && parsed_projection.aggregates.is_empty()
        && group_by.is_empty()
        && select.having.is_none()
        && !projection_requires_expression_path(&parsed_projection.expressions)
        && let TupleSemijoinPairSet::I64Utf8(keys) = &pair_set
        && semijoin_mixed_early_empty_probe_accepts(engine, &outer_path.path, keys)?
    {
        let outer_started = profile.then(Instant::now);
        if let Some(false) = direct_mixed_tuple_semijoin_outer_has_match(
            engine,
            &outer_path.path,
            batch_size,
            &outer_left,
            &outer_right,
            keys,
            outer_filter.as_ref(),
        )? {
            let outer_elapsed = outer_started.map(|started| started.elapsed());
            if let (true, Some(total_started)) = (profile, total_started) {
                eprintln!(
                    "[dodam:semijoin-profile] kind=tuple-in total={}ms inner={}ms outer={}ms projection=0ms output_rows=0 negated={} branch=mixed-direct-empty",
                    total_started.elapsed().as_millis(),
                    inner_elapsed
                        .map(|elapsed| elapsed.as_millis())
                        .unwrap_or(0),
                    outer_elapsed
                        .map(|elapsed| elapsed.as_millis())
                        .unwrap_or(0),
                    negated,
                );
            }
            return Ok(Some(QueryOutput::Scan {
                batches: Vec::new(),
            }));
        }
    }

    if !distinct
        && parsed_projection.aggregates.is_empty()
        && group_by.is_empty()
        && select.having.is_none()
        && offset == 0
        && monotonic_order_limit_scan_enabled()
        && limit.is_some()
        && Path::new(&outer_path.path).exists()
        && let Some(order_column) = monotonic_stream_limit_column(order_by.as_ref())
        && engine
            .parquet_row_groups_monotonic_by_column(outer_path.path.clone(), &order_column)
            .await?
    {
        let outer_started = profile.then(Instant::now);
        let (stream, pre_filter) = if let TupleSemijoinPairSet::I64Utf8(keys) = &pair_set {
            let dictionary_column = if keys.numeric_left {
                outer_right.clone()
            } else {
                outer_left.clone()
            };
            (
                engine
                    .scan_parquet_batches_dictionary_columns(
                        outer_path.path.clone(),
                        batch_size,
                        outer_projection.clone(),
                        vec![dictionary_column],
                    )
                    .await?,
                outer_filter.as_ref(),
            )
        } else {
            (
                engine
                    .scan_parquet_filtered_batches_preserve_order(
                        outer_path.path.clone(),
                        batch_size,
                        outer_projection.clone(),
                        outer_filter.clone(),
                    )
                    .await?,
                None,
            )
        };
        let mut batches = collect_semijoin_i64_pair_filtered_limit_batches(
            stream,
            &outer_left,
            &outer_right,
            &pair_set,
            pre_filter,
            negated,
            limit.unwrap_or(0),
        )?;
        let outer_elapsed = outer_started.map(|started| started.elapsed());
        let projection_started = profile.then(Instant::now);
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
        if let (true, Some(total_started)) = (profile, total_started) {
            let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            eprintln!(
                "[dodam:semijoin-profile] kind=tuple-in total={}ms inner={}ms outer={}ms projection={}ms output_rows={} negated={} branch=pair-set-limit",
                total_started.elapsed().as_millis(),
                inner_elapsed
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0),
                outer_elapsed
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0),
                projection_started
                    .map(|started| started.elapsed().as_millis())
                    .unwrap_or(0),
                rows,
                negated,
            );
        }
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    let outer_started = profile.then(Instant::now);
    let (stream, pre_filter) = if let TupleSemijoinPairSet::I64Utf8(keys) = &pair_set {
        let dictionary_column = if keys.numeric_left {
            outer_right.clone()
        } else {
            outer_left.clone()
        };
        (
            engine
                .scan_parquet_batches_dictionary_columns(
                    outer_path.path.clone(),
                    batch_size,
                    outer_projection,
                    vec![dictionary_column],
                )
                .await?,
            outer_filter.as_ref(),
        )
    } else {
        (
            engine
                .scan_parquet_batches(
                    outer_path.path.clone(),
                    batch_size,
                    None,
                    outer_projection,
                    outer_filter.clone(),
                )
                .await?,
            None,
        )
    };
    let mut filtered = Vec::new();
    for batch in collect_batches(stream)? {
        let batch = if let Some(pre_filter) = pre_filter {
            let mask = evaluate_filter_mask(&batch, pre_filter)?;
            filter_record_batch(&batch, &mask)?
        } else {
            batch
        };
        if batch.num_rows() == 0 {
            continue;
        }
        let mask =
            match tuple_semijoin_pair_mask(&batch, &outer_left, &outer_right, &pair_set, negated) {
                Ok(mask) => mask,
                Err(DodamError::UnsupportedSql(message))
                    if message.contains("integer semijoin key") =>
                {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    let outer_elapsed = outer_started.map(|started| started.elapsed());

    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        let stream = Box::new(MemoryExec::new(filtered)).execute()?;
        let metrics = collect_aggregates_with_optional_expression_views(
            stream,
            1,
            &group_by,
            &parsed_projection.aggregates,
            &parsed_projection.filtered_aggregates,
            &parsed_projection.aggregate_expressions,
        )?;
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions =
            projection_requires_expression_path(&parsed_projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &parsed_projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit, offset)?;
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
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    if let (true, Some(total_started)) = (profile, total_started) {
        let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        eprintln!(
            "[dodam:semijoin-profile] kind=tuple-in total={}ms inner={}ms outer={}ms output_rows={} negated={} branch=materialized",
            total_started.elapsed().as_millis(),
            inner_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            outer_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            rows,
            negated,
        );
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn try_execute_uncorrelated_in_key_semijoin_sql(
    engine: &DodamEngine,
    query: &Query,
    select: &Select,
    outer_path: &SqlTableRef,
    selection: &SqlExpr,
    batch_size: usize,
) -> Result<Option<QueryOutput>> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let Some((in_index, outer_expr, subquery, negated)) =
        conjuncts.iter().enumerate().find_map(|(index, expr)| {
            let (outer_expr, subquery, negated) = sql_in_subquery_parts(expr)?;
            Some((index, outer_expr, subquery, negated))
        })
    else {
        return Ok(None);
    };
    let Some(outer_column) = semijoin_column_name(outer_expr)? else {
        return Ok(None);
    };
    let outer_alias = table_ref_alias_or_name(outer_path);
    if semijoin_column_owner_in_scope(&outer_column, "", &outer_alias, SemijoinColumnOwner::Outer)
        != Some(SemijoinColumnOwner::Outer)
    {
        return Ok(None);
    }
    let outer_column = unqualified_semijoin_column(&outer_column);

    let SetExpr::Select(inner_select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    reject_query_features(subquery)?;
    reject_select_features(inner_select)?;
    if inner_select.from.len() != 1
        || inner_select
            .from
            .first()
            .is_some_and(|table| !table.joins.is_empty())
        || !matches!(
            inner_select.from.first().map(|table| &table.relation),
            Some(TableFactor::Table { .. })
        )
        || inner_select.having.is_some()
        || !parse_group_by(inner_select, None)?.is_empty()
    {
        return Ok(None);
    }
    let [SelectItem::UnnamedExpr(inner_expr)] = inner_select.projection.as_slice() else {
        return Ok(None);
    };
    let Some(inner_column) = semijoin_column_name(inner_expr)? else {
        return Ok(None);
    };
    let inner_path = match parse_from(inner_select) {
        Ok(path) => path,
        Err(DodamError::UnsupportedSql(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let inner_alias = table_ref_alias_or_name(&inner_path);
    if semijoin_column_owner_in_scope(
        &inner_column,
        &inner_alias,
        &outer_alias,
        SemijoinColumnOwner::Inner,
    ) != Some(SemijoinColumnOwner::Inner)
    {
        return Ok(None);
    }
    let inner_column = unqualified_semijoin_column(&inner_column);
    let inner_filter = inner_select
        .selection
        .as_ref()
        .map(|expr| {
            if predicate_requires_expression_path(expr)
                || expr_contains_materializable_subquery(expr)
            {
                return Ok(None);
            }
            parse_filter(expr, &[], inner_path.alias.as_deref(), false).map(Some)
        })
        .transpose()?
        .flatten();
    let profile = semijoin_profile_enabled();
    let total_started = profile.then(Instant::now);
    let inner_started = profile.then(Instant::now);
    let key_set = collect_semijoin_key_set(
        engine,
        inner_path.path,
        &inner_column,
        inner_filter,
        batch_size,
    )
    .await?;
    let inner_elapsed = inner_started.map(|started| started.elapsed());

    let outer_residual = conjuncts
        .into_iter()
        .enumerate()
        .filter_map(|(index, conjunct)| (index != in_index).then_some(conjunct))
        .collect::<Vec<_>>();
    if outer_residual.iter().any(|expr| {
        predicate_requires_expression_path(expr) || expr_contains_materializable_subquery(expr)
    }) {
        return Ok(None);
    }
    let outer_filter = combine_sql_and_conjuncts(outer_residual)
        .as_ref()
        .map(|expr| parse_filter(expr, &[], outer_path.alias.as_deref(), false))
        .transpose()?;

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
    let offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    let outer_projection = semijoin_outer_projection(
        &parsed_projection,
        &group_by,
        order_by.as_ref(),
        &outer_column,
        outer_filter.as_ref(),
    );
    if !distinct
        && parsed_projection.aggregates.is_empty()
        && group_by.is_empty()
        && select.having.is_none()
        && offset == 0
        && monotonic_order_limit_scan_enabled()
        && limit.is_some()
        && Path::new(&outer_path.path).exists()
        && let Some(order_column) = monotonic_stream_limit_column(order_by.as_ref())
        && engine
            .parquet_row_groups_monotonic_by_column(outer_path.path.clone(), &order_column)
            .await?
    {
        let outer_started = profile.then(Instant::now);
        let mut branch = "block-semijoin";
        let mut batches = if outer_filter.is_none()
            && let Some(predicate) = key_set.to_small_int64_in_list_expr(
                &outer_column,
                negated,
                small_dynamic_in_list_row_filter_limit(),
            ) {
            branch = "row-filter";
            let stream = engine
                .scan_parquet_batches_row_filtered_preserve_order(
                    outer_path.path.clone(),
                    batch_size,
                    outer_projection.clone(),
                    vec![predicate],
                )
                .await?;
            collect_ordered_stream_limit_batches(stream, limit, 0)?
        } else {
            let stream = engine
                .scan_parquet_filtered_batches_preserve_order(
                    outer_path.path.clone(),
                    batch_size,
                    outer_projection.clone(),
                    outer_filter.clone(),
                )
                .await?;
            collect_semijoin_filtered_limit_batches(
                stream,
                &outer_column,
                &key_set,
                negated,
                limit.unwrap_or(0),
            )?
        };
        let outer_elapsed = outer_started.map(|started| started.elapsed());
        let projection_started = profile.then(Instant::now);
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
        if let (true, Some(total_started)) = (profile, total_started) {
            let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            eprintln!(
                "[dodam:semijoin-profile] kind=uncorrelated-in total={}ms inner={}ms outer={}ms projection={}ms output_rows={} negated={} branch={}",
                total_started.elapsed().as_millis(),
                inner_elapsed
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0),
                outer_elapsed
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0),
                projection_started
                    .map(|started| started.elapsed().as_millis())
                    .unwrap_or(0),
                rows,
                negated,
                branch,
            );
        }
        return Ok(Some(QueryOutput::Scan { batches }));
    }
    let outer_started = profile.then(Instant::now);
    let stream = engine
        .scan_parquet_batches(
            outer_path.path.clone(),
            batch_size,
            None,
            outer_projection,
            outer_filter,
        )
        .await?;
    let mut filtered = Vec::new();
    for batch in collect_batches(stream)? {
        let mask = if negated {
            semijoin_anti_membership_mask(&batch, &outer_column, &key_set)?
        } else {
            semijoin_membership_mask(&batch, &outer_column, &key_set)?
        };
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() > 0 {
            filtered.push(batch);
        }
    }
    let outer_elapsed = outer_started.map(|started| started.elapsed());

    if !parsed_projection.aggregates.is_empty() || !group_by.is_empty() || select.having.is_some() {
        let stream = Box::new(MemoryExec::new(filtered)).execute()?;
        let metrics = collect_aggregates_with_optional_expression_views(
            stream,
            1,
            &group_by,
            &parsed_projection.aggregates,
            &parsed_projection.filtered_aggregates,
            &parsed_projection.aggregate_expressions,
        )?;
        let mut batches =
            aggregate_metrics_to_batches(&metrics, &group_by, &parsed_projection.aggregates)?;
        batches = apply_output_filter(batches, having.as_ref())?;
        let has_output_expressions =
            projection_requires_expression_path(&parsed_projection.expressions);
        if has_output_expressions {
            batches = apply_output_expression_projection(batches, &parsed_projection.expressions)?;
        }
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &parsed_projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit, offset)?;
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
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &parsed_projection.aliases)?;
    }
    if let (true, Some(total_started)) = (profile, total_started) {
        let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        eprintln!(
            "[dodam:semijoin-profile] kind=uncorrelated-in total={}ms inner={}ms outer={}ms output_rows={} negated={} branch=materialized",
            total_started.elapsed().as_millis(),
            inner_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            outer_elapsed
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            rows,
            negated,
        );
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

fn sql_in_subquery_parts(expr: &SqlExpr) -> Option<(&SqlExpr, &Query, bool)> {
    match expr {
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            let (expr, subquery, negated) = sql_in_subquery_parts(expr)?;
            Some((expr, subquery, !negated))
        }
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => Some((expr.as_ref(), subquery.as_ref(), *negated)),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let [SqlExpr::Subquery(subquery)] = list.as_slice() else {
                return None;
            };
            Some((expr.as_ref(), subquery.as_ref(), *negated))
        }
        SqlExpr::Nested(expr) => sql_in_subquery_parts(expr),
        _ => None,
    }
}

fn rewrite_same_source_distinct_in_subquery(
    expr: &SqlExpr,
    outer_path: &SqlTableRef,
) -> Result<Option<SqlExpr>> {
    match expr {
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated: false,
        } => {
            if !matches!(
                expr.as_ref(),
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)
            ) {
                return Ok(None);
            }
            let outer_column = sql_column_name(expr, outer_path.alias.as_deref())?;
            if same_source_distinct_non_null_key_subquery(subquery, outer_path, &outer_column)? {
                return Ok(Some(SqlExpr::IsNotNull(Box::new(sql_column_expr(
                    &outer_column,
                )))));
            }
            Ok(None)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let rewritten_left = rewrite_same_source_distinct_in_subquery(left, outer_path)?;
            let rewritten_right = rewrite_same_source_distinct_in_subquery(right, outer_path)?;
            match (rewritten_left, rewritten_right) {
                (Some(rewritten_left), Some(rewritten_right)) => Ok(Some(SqlExpr::BinaryOp {
                    left: Box::new(rewritten_left),
                    op: BinaryOperator::And,
                    right: Box::new(rewritten_right),
                })),
                (Some(rewritten_left), None) => Ok(Some(SqlExpr::BinaryOp {
                    left: Box::new(rewritten_left),
                    op: BinaryOperator::And,
                    right: right.clone(),
                })),
                (None, Some(rewritten_right)) => Ok(Some(SqlExpr::BinaryOp {
                    left: left.clone(),
                    op: BinaryOperator::And,
                    right: Box::new(rewritten_right),
                })),
                (None, None) => Ok(None),
            }
        }
        SqlExpr::Nested(expr) => rewrite_same_source_distinct_in_subquery(expr, outer_path)
            .map(|expr| expr.map(|expr| SqlExpr::Nested(Box::new(expr)))),
        _ => Ok(None),
    }
}

fn same_source_distinct_non_null_key_subquery(
    query: &Query,
    outer_path: &SqlTableRef,
    outer_column: &str,
) -> Result<bool> {
    if query.with.is_some() || query.order_by.is_some() || query.limit_clause.is_some() {
        return Ok(false);
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    let Some(projected_column) = single_projected_column(select, None)? else {
        return Ok(false);
    };
    if projected_column != outer_column {
        return Ok(false);
    }
    if select.from.len() != 1 || select.from[0].joins.len() != 0 {
        return Ok(false);
    }
    match &select.from[0].relation {
        TableFactor::Table { .. } => {
            if !parse_distinct(select)? {
                return Ok(false);
            }
            let table = parse_table_factor(&select.from[0].relation)?;
            Ok(table.path == outer_path.path
                && selection_is_column_is_not_null(
                    select.selection.as_ref(),
                    outer_column,
                    table.alias.as_deref(),
                )?)
        }
        TableFactor::Derived {
            lateral,
            subquery,
            alias,
            ..
        } => {
            if *lateral {
                return Ok(false);
            }
            let Some(alias) = alias else {
                return Ok(false);
            };
            if !alias.columns.is_empty() || alias.at.is_some() {
                return Ok(false);
            }
            let SetExpr::Select(inner) = subquery.body.as_ref() else {
                return Ok(false);
            };
            if subquery.with.is_some()
                || subquery.order_by.is_some()
                || subquery.limit_clause.is_some()
            {
                return Ok(false);
            }
            if !parse_distinct(inner)? {
                return Ok(false);
            }
            let Some(inner_column) = single_projected_column(inner, None)? else {
                return Ok(false);
            };
            if inner_column != outer_column {
                return Ok(false);
            }
            let table = parse_from(inner)?;
            Ok(table.path == outer_path.path
                && selection_is_column_is_not_null(
                    inner.selection.as_ref(),
                    outer_column,
                    table.alias.as_deref(),
                )?)
        }
        _ => Ok(false),
    }
}

fn single_projected_column(select: &Select, table_alias: Option<&str>) -> Result<Option<String>> {
    let [SelectItem::UnnamedExpr(expr)] = select.projection.as_slice() else {
        return Ok(None);
    };
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            sql_column_name(expr, table_alias).map(Some)
        }
        _ => Ok(None),
    }
}

fn selection_is_column_is_not_null(
    selection: Option<&SqlExpr>,
    column: &str,
    table_alias: Option<&str>,
) -> Result<bool> {
    let Some(SqlExpr::IsNotNull(expr)) = selection else {
        return Ok(false);
    };
    Ok(sql_column_name(expr, table_alias)? == column)
}
