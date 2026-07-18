use super::*;

pub(super) async fn try_execute_with_cte_sql(
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
    let Some(with) = query.with.as_ref() else {
        return Ok(None);
    };
    if with.recursive || with.cte_tables.len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "only single non-recursive WITH queries are supported".to_string(),
        ));
    }
    let cte = &with.cte_tables[0];
    if !cte.alias.columns.is_empty() || cte.alias.at.is_some() || cte.from.is_some() {
        return Err(DodamError::UnsupportedSql(
            "WITH column aliases, AT aliases, and FROM identifiers are not supported".to_string(),
        ));
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    reject_select_features(select)?;
    let cte_alias = cte.alias.name.value.clone();
    let cte_output = Box::pin(execute_sql_with_options(
        engine,
        &cte.query.to_string(),
        batch_size,
        options,
    ))
    .await?;
    let cte_batches = query_output_batches(cte_output)?;
    if let Some(output) = try_execute_single_cte_select(&cte_alias, &cte_batches, select, query)? {
        return Ok(Some(output));
    }
    let (relations, left_keys, right_keys, residual, right_filter, join_type) = if select.from.len()
        == 2
        && select.from.iter().all(|table| table.joins.is_empty())
    {
        let mut relations = Vec::new();
        for table in &select.from {
            relations.push(
                materialize_cte_join_relation(
                    engine,
                    &table.relation,
                    &cte_alias,
                    &cte_batches,
                    batch_size,
                    options,
                )
                .await?,
            );
        }
        let [left, right] = relations.as_slice() else {
            return Ok(None);
        };
        let aliases = vec![left.alias.clone(), right.alias.clone()];
        let alias_refs = aliases.iter().map(String::as_str).collect::<Vec<_>>();
        let selection = select.selection.as_ref().ok_or_else(|| {
            DodamError::UnsupportedSql(
                "comma join requires an equality predicate in WHERE".to_string(),
            )
        })?;
        let mut rewritten_selection = selection.clone();
        rewrite_cte_scalar_subqueries_to_literals(
            &mut rewritten_selection,
            &cte_alias,
            &cte_batches,
        )?;
        let (left_keys, right_keys, residual) = split_comma_join_selection(
            Some(&rewritten_selection),
            &left.alias,
            &right.alias,
            &alias_refs,
        )?;
        (
            relations,
            left_keys,
            right_keys,
            residual,
            None,
            JoinType::Inner,
        )
    } else if let [table] = select.from.as_slice()
        && let [join] = table.joins.as_slice()
    {
        let left = materialize_cte_join_relation(
            engine,
            &table.relation,
            &cte_alias,
            &cte_batches,
            batch_size,
            options,
        )
        .await?;
        let right = materialize_cte_join_relation(
            engine,
            &join.relation,
            &cte_alias,
            &cte_batches,
            batch_size,
            options,
        )
        .await?;
        let mut rewritten_join = join.clone();
        if let JoinOperator::Join(JoinConstraint::On(expr))
        | JoinOperator::Inner(JoinConstraint::On(expr))
        | JoinOperator::Left(JoinConstraint::On(expr))
        | JoinOperator::LeftOuter(JoinConstraint::On(expr))
        | JoinOperator::Right(JoinConstraint::On(expr))
        | JoinOperator::RightOuter(JoinConstraint::On(expr))
        | JoinOperator::FullOuter(JoinConstraint::On(expr))
        | JoinOperator::Semi(JoinConstraint::On(expr))
        | JoinOperator::LeftSemi(JoinConstraint::On(expr)) = &mut rewritten_join.join_operator
        {
            rewrite_cte_scalar_subqueries_to_literals(expr, &cte_alias, &cte_batches)?;
        }
        let (join_type, left_keys, right_keys, right_filter) =
            parse_join_condition(&rewritten_join, &left.alias, &right.alias)?;
        let residual = if let Some(selection) = select.selection.as_ref() {
            let mut rewritten_selection = selection.clone();
            rewrite_cte_scalar_subqueries_to_literals(
                &mut rewritten_selection,
                &cte_alias,
                &cte_batches,
            )?;
            Some(rewritten_selection)
        } else {
            None
        };
        (
            vec![left, right],
            left_keys,
            right_keys,
            residual,
            right_filter,
            join_type,
        )
    } else {
        return Err(DodamError::UnsupportedSql(
            "WITH currently supports two-table comma joins and single explicit JOINs".to_string(),
        ));
    };
    let [left, right] = relations.as_slice() else {
        return Ok(None);
    };
    let aliases = vec![left.alias.clone(), right.alias.clone()];
    let alias_refs = aliases.iter().map(String::as_str).collect::<Vec<_>>();
    let output_aliases = if join_type == JoinType::Semi {
        vec![left.alias.as_str()]
    } else {
        alias_refs.clone()
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let distinct = parse_distinct(select)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        None,
    )?;
    let (filter, expression_filter) = parse_join_filter_plan(
        residual.as_ref(),
        &projection.aliases,
        &output_aliases,
        false,
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
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
    let limit = parse_limit(query)?;

    let stream = Box::new(HashJoinExec::new(
        Box::new(MemoryExec::new(left.batches.clone())),
        Box::new(MemoryExec::new(apply_output_filter(
            right.batches.clone(),
            right_filter.as_ref(),
        )?)),
        left_keys,
        right_keys,
        left.alias.clone(),
        right.alias.clone(),
        JoinBuildSide::Right,
        join_type,
        Projection::All,
    ))
    .execute()?;
    let mut batches = apply_output_filter(collect_batches(stream)?, filter.as_ref())?;
    if let Some(expression_filter) = expression_filter.as_ref() {
        batches = apply_output_join_expression_filter(batches, expression_filter, &output_aliases)?;
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

fn try_execute_single_cte_select(
    cte_alias: &str,
    cte_batches: &[RecordBatch],
    select: &Select,
    query: &Query,
) -> Result<Option<QueryOutput>> {
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table.joins.is_empty() {
        return Ok(None);
    }
    let TableFactor::Table { name, alias, .. } = &table.relation else {
        return Ok(None);
    };
    if !object_name_to_string(name)?.eq_ignore_ascii_case(cte_alias) {
        return Ok(None);
    }
    let effective_alias = alias
        .as_ref()
        .map_or_else(|| cte_alias.to_string(), |alias| alias.name.value.clone());
    let group_by = parse_group_by(select, Some(&effective_alias))?;
    let projection = parse_projection(select, &group_by, Some(&effective_alias))?;
    let distinct = parse_distinct(select)?;
    let filter = select
        .selection
        .as_ref()
        .map(|expr| parse_filter(expr, &[], Some(&effective_alias), false))
        .transpose()?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        Some(&effective_alias),
    )?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        order_by.as_ref(),
    )?;

    let mut batches = apply_output_filter(cte_batches.to_vec(), filter.as_ref())?;
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
        batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
        if !has_output_expressions {
            batches = rename_output_batches(batches, &projection.aliases)?;
        }
        return Ok(Some(QueryOutput::Aggregate { metrics, batches }));
    }

    let projection_requires_expression =
        projection_requires_expression_path(&projection.expressions);
    batches = if projection_requires_expression {
        apply_output_expression_projection(batches, &projection.expressions)?
    } else {
        apply_output_projection(batches, &projection.projection)?
    };
    if distinct {
        batches = collect_batches(
            Box::new(DistinctExec::new(Box::new(MemoryExec::new(batches)))).execute()?,
        )?;
    }
    batches = apply_output_order_limit(batches, order_by.as_ref(), limit, offset)?;
    if !projection_requires_expression {
        batches = rename_output_batches(batches, &projection.aliases)?;
    }
    Ok(Some(QueryOutput::Scan { batches }))
}

async fn materialize_cte_join_relation(
    engine: &DodamEngine,
    relation: &TableFactor,
    cte_alias: &str,
    cte_batches: &[RecordBatch],
    batch_size: usize,
    options: SqlExecutionOptions,
) -> Result<MaterializedJoinRelation> {
    match relation {
        TableFactor::Table { name, alias, .. }
            if object_name_to_string(name)?.eq_ignore_ascii_case(cte_alias) =>
        {
            Ok(MaterializedJoinRelation {
                alias: alias
                    .as_ref()
                    .map_or_else(|| cte_alias.to_string(), |alias| alias.name.value.clone()),
                batches: cte_batches.to_vec(),
            })
        }
        TableFactor::Table { .. } => {
            materialize_join_relation(engine, relation, batch_size, options).await
        }
        _ => Err(DodamError::UnsupportedSql(
            "WITH joins currently support direct table references".to_string(),
        )),
    }
}

fn rewrite_cte_scalar_subqueries_to_literals(
    expr: &mut SqlExpr,
    cte_alias: &str,
    cte_batches: &[RecordBatch],
) -> Result<()> {
    match expr {
        SqlExpr::Subquery(query) => {
            if let Some(value) = evaluate_cte_scalar_subquery(query, cte_alias, cte_batches)? {
                *expr = literal_value_to_sql_expr(value);
            }
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(left, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(right, cte_alias, cte_batches)?;
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
        }
        SqlExpr::InList { expr, list, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
            for item in list {
                rewrite_cte_scalar_subqueries_to_literals(item, cte_alias, cte_batches)?;
            }
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(low, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(high, cte_alias, cte_batches)?;
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            rewrite_cte_scalar_subqueries_to_literals(expr, cte_alias, cte_batches)?;
            rewrite_cte_scalar_subqueries_to_literals(pattern, cte_alias, cte_batches)?;
        }
        _ => {}
    }
    Ok(())
}

fn evaluate_cte_scalar_subquery(
    query: &Query,
    cte_alias: &str,
    cte_batches: &[RecordBatch],
) -> Result<Option<LiteralValue>> {
    if query.with.is_some() || query.order_by.is_some() || query.limit_clause.is_some() {
        return Ok(None);
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };
    let [table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !table.joins.is_empty() || select.selection.is_some() || select.having.is_some() {
        return Ok(None);
    }
    let TableFactor::Table { name, .. } = &table.relation else {
        return Ok(None);
    };
    if !object_name_to_string(name)?.eq_ignore_ascii_case(cte_alias) {
        return Ok(None);
    }
    let [SelectItem::UnnamedExpr(SqlExpr::Function(function))] = select.projection.as_slice()
    else {
        return Ok(None);
    };
    let aggregate = parse_aggregate(function, None)?;
    match aggregate {
        AggregateExpr::Max(column) => max_literal_from_batches(cte_batches, &column).map(Some),
        _ => Ok(None),
    }
}

fn max_literal_from_batches(batches: &[RecordBatch], column: &str) -> Result<LiteralValue> {
    let mut max_value: Option<LiteralValue> = None;
    for batch in batches {
        let index = batch_column_index(batch, column)?;
        let array = batch.column(index);
        for row in 0..batch.num_rows() {
            if array.is_null(row) {
                continue;
            }
            let value = literal_value_from_array(array, row)?;
            let replace = match &max_value {
                None => true,
                Some(current) => matches!(
                    compare_literal_values(&value, &BinaryOperator::Gt, current)?,
                    Some(true)
                ),
            };
            if replace {
                max_value = Some(value);
            }
        }
    }
    Ok(max_value.unwrap_or(LiteralValue::Null))
}
