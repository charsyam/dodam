use super::*;

pub(super) fn parse_join_projection(
    select: &Select,
    table_aliases: &[&str],
    group_by: &[String],
) -> Result<ParsedProjection> {
    let mut columns = Vec::new();
    let mut aggregates = Vec::new();
    let mut filtered_aggregates = Vec::new();
    let mut aggregate_expressions = Vec::new();
    let mut aggregate_expression_columns = Vec::new();
    let mut aliases = Vec::new();
    let mut expressions = Vec::new();
    let mut qualified_wildcards = Vec::new();
    let mut wildcard = false;
    let group_expression_bindings = join_group_expression_bindings(select, table_aliases)?;

    for item in &select.projection {
        match item {
            SelectItem::Wildcard(_) => wildcard = true,
            SelectItem::UnnamedExpr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            ) => {
                if let Some((column, field)) = parse_join_struct_field_access(expr, table_aliases)?
                {
                    let output_name = column_output_name(expr);
                    add_column_once(&mut columns, column.clone());
                    aliases.push((expr.to_string(), output_name.clone()));
                    expressions.push(ProjectionExpression {
                        output_name,
                        expr: ScalarSqlExpression::StructField { column, field },
                    });
                } else {
                    let column = join_column_name(expr, table_aliases)?;
                    let output_name = column_output_name(expr);
                    columns.push(column.clone());
                    expressions.push(ProjectionExpression {
                        output_name,
                        expr: ScalarSqlExpression::Column(column),
                    });
                }
            }
            SelectItem::UnnamedExpr(SqlExpr::Function(function)) => {
                let function_expr = SqlExpr::Function(function.clone());
                if let Some((column, _)) =
                    projected_group_expression(&function_expr, &group_expression_bindings)
                {
                    add_column_once(&mut columns, column.clone());
                    aliases.push((function.to_string(), column.clone()));
                    expressions.push(ProjectionExpression {
                        output_name: function.to_string(),
                        expr: ScalarSqlExpression::Column(column),
                    });
                } else if let Some(expr) =
                    parse_join_scalar_function_projection(function, None, table_aliases)?
                {
                    for column in join_scalar_expression_columns(&expr.expr, table_aliases)? {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(expr);
                } else {
                    let (aggregate, expression) = parse_join_aggregate_with_input_expression(
                        function,
                        table_aliases,
                        aggregate_expressions.len(),
                    )?;
                    if let Some(spec) = filtered_join_aggregate_spec_from_function(
                        function,
                        table_aliases,
                        &aggregate,
                        &expression,
                    )? {
                        filtered_aggregates.push(spec);
                    }
                    if let Some(expression) = expression {
                        for column in
                            join_scalar_expression_columns(&expression.expr, table_aliases)?
                        {
                            add_column_once(&mut aggregate_expression_columns, column);
                        }
                        for column in join_sql_expression_columns(
                            &SqlExpr::Function(function.clone()),
                            table_aliases,
                        )? {
                            add_column_once(&mut aggregate_expression_columns, column);
                        }
                        aggregate_expressions.push(expression);
                    }
                    aliases.push((function.to_string(), aggregate.to_string()));
                    aggregates.push(aggregate);
                }
            }
            SelectItem::UnnamedExpr(expr) => {
                let mut expression_columns = Vec::new();
                let mut found_aggregate = false;
                if let Some((column, _)) =
                    projected_group_expression(expr, &group_expression_bindings)
                {
                    add_column_once(&mut columns, column.clone());
                    aliases.push((expr.to_string(), column.clone()));
                    expressions.push(ProjectionExpression {
                        output_name: expr.to_string(),
                        expr: ScalarSqlExpression::Column(column),
                    });
                } else if let Ok(parsed) = parse_join_aggregate_output_expression(
                    expr,
                    table_aliases,
                    &mut aggregates,
                    &mut filtered_aggregates,
                    &mut aggregate_expressions,
                    &mut aggregate_expression_columns,
                    &mut expression_columns,
                    &mut found_aggregate,
                ) && found_aggregate
                {
                    for column in expression_columns {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(ProjectionExpression {
                        output_name: expr.to_string(),
                        expr: parsed,
                    });
                } else {
                    let parsed = parse_join_scalar_sql_expression(expr, table_aliases)?;
                    for column in join_scalar_expression_columns(&parsed, table_aliases)? {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(ProjectionExpression {
                        output_name: expr.to_string(),
                        expr: parsed,
                    });
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => match expr {
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                    if let Some((column, field)) =
                        parse_join_struct_field_access(expr, table_aliases)?
                    {
                        add_column_once(&mut columns, column.clone());
                        aliases.push((alias.value.clone(), alias.value.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: ScalarSqlExpression::StructField { column, field },
                        });
                    } else {
                        let column = join_column_name(expr, table_aliases)?;
                        aliases.push((alias.value.clone(), column.clone()));
                        columns.push(column);
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: ScalarSqlExpression::Column(
                                aliases.last().expect("alias just pushed").1.clone(),
                            ),
                        });
                    }
                }
                SqlExpr::Function(function) => {
                    let function_expr = SqlExpr::Function(function.clone());
                    if let Some((column, _)) =
                        projected_group_expression(&function_expr, &group_expression_bindings)
                    {
                        add_column_once(&mut columns, column.clone());
                        aliases.push((alias.value.clone(), column.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: ScalarSqlExpression::Column(column),
                        });
                    } else if let Some(expr) = parse_join_scalar_function_projection(
                        function,
                        Some(&alias.value),
                        table_aliases,
                    )? {
                        for column in join_scalar_expression_columns(&expr.expr, table_aliases)? {
                            add_column_once(&mut columns, column);
                        }
                        aliases.push((function.to_string(), alias.value.clone()));
                        aliases.push((alias.value.clone(), alias.value.clone()));
                        expressions.push(expr);
                    } else {
                        let (aggregate, expression) = parse_join_aggregate_with_input_expression(
                            function,
                            table_aliases,
                            aggregate_expressions.len(),
                        )?;
                        if let Some(spec) = filtered_join_aggregate_spec_from_function(
                            function,
                            table_aliases,
                            &aggregate,
                            &expression,
                        )? {
                            filtered_aggregates.push(spec);
                        }
                        if let Some(expression) = expression {
                            for column in
                                join_scalar_expression_columns(&expression.expr, table_aliases)?
                            {
                                add_column_once(&mut aggregate_expression_columns, column);
                            }
                            for column in join_sql_expression_columns(expr, table_aliases)? {
                                add_column_once(&mut aggregate_expression_columns, column);
                            }
                            aggregate_expressions.push(expression);
                        }
                        aliases.push((function.to_string(), aggregate.to_string()));
                        aliases.push((alias.value.clone(), aggregate.to_string()));
                        aggregates.push(aggregate);
                    }
                }
                _ => {
                    let mut expression_columns = Vec::new();
                    let mut found_aggregate = false;
                    if let Some((column, _)) =
                        projected_group_expression(expr, &group_expression_bindings)
                    {
                        add_column_once(&mut columns, column.clone());
                        aliases.push((alias.value.clone(), column.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: ScalarSqlExpression::Column(column),
                        });
                    } else if let Ok(parsed) = parse_join_aggregate_output_expression(
                        expr,
                        table_aliases,
                        &mut aggregates,
                        &mut filtered_aggregates,
                        &mut aggregate_expressions,
                        &mut aggregate_expression_columns,
                        &mut expression_columns,
                        &mut found_aggregate,
                    ) && found_aggregate
                    {
                        for column in expression_columns {
                            add_column_once(&mut columns, column);
                        }
                        aliases.push((alias.value.clone(), alias.value.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: parsed,
                        });
                    } else {
                        let parsed = parse_join_scalar_sql_expression(expr, table_aliases)?;
                        for column in join_scalar_expression_columns(&parsed, table_aliases)? {
                            add_column_once(&mut columns, column);
                        }
                        aliases.push((expr.to_string(), alias.value.clone()));
                        aliases.push((alias.value.clone(), alias.value.clone()));
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: parsed,
                        });
                    }
                }
            },
            SelectItem::QualifiedWildcard(name, _) => {
                let qualifier = qualified_wildcard_name(name)?;
                if !table_aliases.iter().any(|alias| *alias == qualifier) {
                    return Err(DodamError::UnknownColumn(format!("{qualifier}.*")));
                }
                qualified_wildcards.push(qualifier);
            }
            _ => {
                return Err(DodamError::UnsupportedSql(format!(
                    "JOIN SELECT items must be qualified columns, got {item}"
                )));
            }
        }
    }

    if wildcard && select.projection.len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "SELECT * cannot be mixed with other items".to_string(),
        ));
    }

    if wildcard && !aggregates.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "SELECT * cannot be used with aggregate JOIN SELECT items".to_string(),
        ));
    }

    if !aggregates.is_empty() {
        for column in &columns {
            if !group_by.iter().any(|group_column| group_column == column) {
                return Err(DodamError::UnsupportedSql(format!(
                    "non-aggregate JOIN SELECT column {column} must appear in GROUP BY"
                )));
            }
        }
        let mut projected_columns = physical_projection_columns(&columns);
        for aggregate in &aggregates {
            if let Some(column) = aggregate.referenced_column() {
                if !column.starts_with("__dodam_join_agg_expr_") {
                    add_column_once(&mut projected_columns, column.to_string());
                }
            }
        }
        add_filtered_aggregate_projection_columns(&mut projected_columns, &filtered_aggregates)?;
        for column in aggregate_expression_columns {
            add_column_once(&mut projected_columns, column);
        }
        for binding in &group_expression_bindings {
            for column in join_scalar_expression_columns(&binding.expression.expr, table_aliases)? {
                add_column_once(&mut projected_columns, column);
            }
            aggregate_expressions.push(binding.expression.clone());
        }
        let ordinal_targets = projection_ordinal_targets(&columns, &aggregates, &expressions);
        return Ok(ParsedProjection {
            projection: Projection::Columns(projected_columns),
            aggregates,
            filtered_aggregates,
            aggregate_expressions,
            aliases,
            expressions,
            ordinal_targets,
            qualified_wildcards,
        });
    }

    let ordinal_targets = projection_ordinal_targets(&columns, &aggregates, &expressions);
    Ok(ParsedProjection {
        projection: if wildcard {
            Projection::All
        } else {
            Projection::Columns(columns)
        },
        aggregates,
        filtered_aggregates,
        aggregate_expressions,
        aliases,
        expressions,
        ordinal_targets,
        qualified_wildcards,
    })
}

pub(super) fn parse_join_group_by(select: &Select, table_aliases: &[&str]) -> Result<Vec<String>> {
    let expressions = group_by_expressions(select)?;
    expressions
        .iter()
        .enumerate()
        .map(|(index, expr)| {
            join_column_name(expr, table_aliases).or_else(|_| {
                parse_join_scalar_sql_expression(expr, table_aliases)
                    .map(|_| group_by_synthetic_column(index))
            })
        })
        .collect::<Result<Vec<_>>>()
}

pub(super) fn parse_join_order_by(
    query: &Query,
    aliases: &[(String, String)],
    ordinal_targets: &[String],
    table_aliases: &[&str],
) -> Result<Option<SortKey>> {
    let Some(order_by) = &query.order_by else {
        return Ok(None);
    };
    if order_by.interpolate.is_some() {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY INTERPOLATE is not supported".to_string(),
        ));
    }
    let OrderByKind::Expressions(expressions) = &order_by.kind else {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY ALL is not supported".to_string(),
        ));
    };
    let expressions = expressions
        .iter()
        .map(|order| {
            if order.with_fill.is_some() {
                return Err(DodamError::UnsupportedSql(
                    "ORDER BY WITH FILL is not supported".to_string(),
                ));
            }
            let column = match &order.expr {
                SqlExpr::Value(value) => resolve_order_by_ordinal(value, ordinal_targets)?,
                SqlExpr::Identifier(ident) => {
                    alias_target(&ident.value, aliases)
                        .cloned()
                        .map(Ok)
                        .unwrap_or_else(|| join_column_name(&order.expr, table_aliases))?
                }
                SqlExpr::CompoundIdentifier(_) => alias_target(&order.expr.to_string(), aliases)
                    .cloned()
                    .map(Ok)
                    .unwrap_or_else(|| join_column_name(&order.expr, table_aliases))?,
                SqlExpr::Function(function) => {
                    if let Some(column) = alias_target(&function.to_string(), aliases).cloned() {
                        column
                    } else {
                        resolve_alias(
                            &parse_join_aggregate(function, table_aliases)?.to_string(),
                            aliases,
                        )
                    }
                }
                expr => {
                    let Some(column) = alias_target(&expr.to_string(), aliases).cloned() else {
                        return Err(DodamError::UnsupportedSql(format!(
                            "expected JOIN ORDER BY alias, qualified column, aggregate expression, or projected scalar expression, got {expr}"
                        )));
                    };
                    column
                }
            };
            Ok(SortExpr {
                column,
                descending: order.options.asc == Some(false),
                nulls_first: order.options.nulls_first.unwrap_or(false),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    SortKey::new(expressions).map(Some)
}
