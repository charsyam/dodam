use super::*;

pub(super) async fn try_execute_correlated_join_subquery_filter_sql(
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
    let Some(selection) = select.selection.as_ref() else {
        return Ok(None);
    };
    reject_query_features(query)?;
    reject_select_features(select)?;
    if select.from.len() != 2 || !expr_contains_scalar_subquery(selection) {
        return Ok(None);
    }
    let [left_table, right_table] = select.from.as_slice() else {
        return Ok(None);
    };
    if !left_table.joins.is_empty() || !right_table.joins.is_empty() {
        return Ok(None);
    }

    let left = parse_table_factor(&left_table.relation)?;
    let right = parse_table_factor(&right_table.relation)?;
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let output_aliases = vec![left_alias.as_str(), right_alias.as_str()];
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let _distinct = parse_distinct(select)?;
    let (left_keys, right_keys, residual) =
        split_comma_join_selection(Some(selection), &left_alias, &right_alias, &output_aliases)?;
    let Some(residual) = residual else {
        return Ok(None);
    };
    if !expr_contains_scalar_subquery(&residual) {
        return Ok(None);
    }
    let having = select
        .having
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

    let join_input_projection = &Projection::All;
    let join_plan = plan_join_inputs(
        join_input_projection,
        None,
        order_by.as_ref(),
        &left_alias,
        &left_keys,
        &right_alias,
        &right_keys,
    );
    let output_projection = if !projection.aggregate_expressions.is_empty()
        || !projection.aggregates.is_empty()
        || order_by.is_some()
    {
        Projection::All
    } else {
        projection.projection.clone()
    };
    let output_projection_pushed = !matches!(output_projection, Projection::All);
    let stream = engine
        .join_parquet_batches(JoinParquetRequest {
            left_path: left.path,
            right_path: right.path,
            batch_size,
            left_keys,
            right_keys,
            left_prefix: left_alias,
            right_prefix: right_alias,
            left_projection: join_plan.left_projection,
            right_projection: join_plan.right_projection,
            left_filter: join_plan.left_filter,
            right_filter: join_plan.right_filter,
            output_projection,
            join_memory_limit_bytes: join_memory_limit_bytes(options),
            join_algorithm: JoinAlgorithm::Auto,
            join_type: JoinType::Inner,
        })
        .await?;
    let residual_sql = residual.to_string();
    let mut filtered = Vec::new();
    for batch in collect_batches(stream)? {
        let mask = evaluate_correlated_subquery_filter_mask(
            engine,
            &residual_sql,
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

    if !projection.aggregates.is_empty() {
        let stream: SendableBatchStream = if projection.aggregate_expressions.is_empty() {
            Box::new(MemoryExec::new(filtered)).execute()?
        } else {
            let batches =
                append_aggregate_expression_columns(filtered, &projection.aggregate_expressions)?;
            Box::new(MemoryExec::new(batches)).execute()?
        };
        let metrics = if group_by.is_empty() {
            collect_aggregates(stream, 2, &projection.aggregates)?
        } else {
            collect_grouped_aggregates(stream, 2, &group_by, &projection.aggregates)?
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

    let mut batches = apply_output_order_limit(filtered, order_by.as_ref(), limit, 0)?;
    if !output_projection_pushed {
        batches = apply_output_projection(batches, &projection.projection)?;
    }
    batches = rename_output_batches(batches, &projection.aliases)?;
    Ok(Some(QueryOutput::Scan { batches }))
}

pub(super) async fn try_execute_materialized_join_subquery_sql(
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
    let has_materializable_subquery = select
        .selection
        .as_ref()
        .is_some_and(expr_contains_materializable_subquery)
        || select
            .having
            .as_ref()
            .is_some_and(expr_contains_materializable_subquery);
    if select.from.len() <= 1 || !has_materializable_subquery {
        return Ok(None);
    }
    reject_query_features(query)?;
    reject_select_features(select)?;

    let mut rewritten_query = query.as_ref().clone();
    let SetExpr::Select(rewritten_select) = rewritten_query.body.as_mut() else {
        return Ok(None);
    };
    let mut changed = false;
    if let Some(rewritten_selection) = rewritten_select.selection.take() {
        let Some(rewritten_selection) = Box::pin(rewrite_materializable_subqueries_to_literals(
            engine,
            rewritten_selection,
            batch_size,
            options,
            &mut changed,
        ))
        .await?
        else {
            return Ok(None);
        };
        rewritten_select.selection = Some(rewritten_selection);
    }
    if let Some(rewritten_having) = rewritten_select.having.take() {
        let Some(rewritten_having) = Box::pin(rewrite_materializable_subqueries_to_literals(
            engine,
            rewritten_having,
            batch_size,
            options,
            &mut changed,
        ))
        .await?
        else {
            return Ok(None);
        };
        rewritten_select.having = Some(rewritten_having);
    }
    if !changed {
        return Ok(None);
    }
    Box::pin(execute_sql_with_options(
        engine,
        &rewritten_query.to_string(),
        batch_size,
        options,
    ))
    .await
    .map(Some)
}

async fn rewrite_materializable_subqueries_to_literals(
    engine: &DodamEngine,
    expr: SqlExpr,
    batch_size: usize,
    options: SqlExecutionOptions,
    changed: &mut bool,
) -> Result<Option<SqlExpr>> {
    match expr {
        SqlExpr::Exists { subquery, negated } => {
            let output = match Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await
            {
                Ok(output) => output,
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            let exists = query_output_batches(output)?
                .iter()
                .any(|batch| batch.num_rows() > 0);
            *changed = true;
            Ok(Some(SqlExpr::Value(
                Value::Boolean(if negated { !exists } else { exists }).with_empty_span(),
            )))
        }
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            let output = match Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await
            {
                Ok(output) => output,
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            let values = literal_values_from_single_column_batches(query_output_batches(output)?)?;
            *changed = true;
            if values.is_empty() {
                return Ok(Some(SqlExpr::Value(
                    Value::Boolean(negated).with_empty_span(),
                )));
            }
            Ok(Some(SqlExpr::InList {
                expr,
                list: values.into_iter().map(literal_value_to_sql_expr).collect(),
                negated,
            }))
        }
        SqlExpr::Subquery(subquery) => {
            let output = match Box::pin(execute_sql_with_options(
                engine,
                &subquery.to_string(),
                batch_size,
                options,
            ))
            .await
            {
                Ok(output) => output,
                Err(DodamError::UnsupportedSql(_))
                | Err(DodamError::UnknownColumn(_))
                | Err(DodamError::UnknownTableQualifier(_)) => {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            let value = scalar_literal_value_from_batches(query_output_batches(output)?)?;
            *changed = true;
            Ok(Some(literal_value_to_sql_expr(value)))
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let Some(left) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *left, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(right) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *right, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }))
        }
        SqlExpr::Nested(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Nested(Box::new(expr))))
        }
        SqlExpr::UnaryOp { op, expr } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::UnaryOp {
                op,
                expr: Box::new(expr),
            }))
        }
        SqlExpr::IsNull(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::IsNull(Box::new(expr))))
        }
        SqlExpr::IsNotNull(expr) => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::IsNotNull(Box::new(expr))))
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let mut rewritten_list = Vec::with_capacity(list.len());
            for item in list {
                let Some(item) = Box::pin(rewrite_materializable_subqueries_to_literals(
                    engine, item, batch_size, options, changed,
                ))
                .await?
                else {
                    return Ok(None);
                };
                rewritten_list.push(item);
            }
            Ok(Some(SqlExpr::InList {
                expr: Box::new(expr),
                list: rewritten_list,
                negated,
            }))
        }
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(low) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *low, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(high) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *high, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Between {
                expr: Box::new(expr),
                negated,
                low: Box::new(low),
                high: Box::new(high),
            }))
        }
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            let Some(expr) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *expr, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            let Some(pattern) = Box::pin(rewrite_materializable_subqueries_to_literals(
                engine, *pattern, batch_size, options, changed,
            ))
            .await?
            else {
                return Ok(None);
            };
            Ok(Some(SqlExpr::Like {
                negated,
                any,
                expr: Box::new(expr),
                pattern: Box::new(pattern),
                escape_char,
            }))
        }
        expr => Ok(Some(expr)),
    }
}
