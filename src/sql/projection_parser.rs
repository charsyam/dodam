use super::*;

pub(super) fn column_output_name(expr: &SqlExpr) -> String {
    match expr {
        SqlExpr::Identifier(ident) => ident.value.clone(),
        SqlExpr::CompoundIdentifier(parts) => parts
            .last()
            .map(|ident| ident.value.clone())
            .unwrap_or_else(|| expr.to_string()),
        _ => expr.to_string(),
    }
}

pub(super) fn tpch_alias_prefix(alias: &str) -> Option<&'static str> {
    match alias.to_ascii_lowercase().as_str() {
        "customer" => Some("c"),
        "orders" => Some("o"),
        "lineitem" => Some("l"),
        "part" => Some("p"),
        "partsupp" => Some("ps"),
        "supplier" => Some("s"),
        "nation" => Some("n"),
        "region" => Some("r"),
        _ => None,
    }
}

pub(super) fn add_filtered_aggregate_projection_columns(
    projected_columns: &mut Vec<String>,
    specs: &[NativeFilteredAggregateSpec],
) -> Result<()> {
    for spec in specs {
        for column in scalar_expression_columns(&spec.input) {
            add_column_once(projected_columns, column);
        }
        collect_predicate_expression_columns(&spec.condition, None, projected_columns)?;
    }
    Ok(())
}

pub(super) fn parse_projection(
    select: &Select,
    group_by: &[String],
    table_alias: Option<&str>,
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
    let group_expression_bindings = group_expression_bindings(select, table_alias)?;

    for item in &select.projection {
        match item {
            SelectItem::Wildcard(_) => wildcard = true,
            SelectItem::QualifiedWildcard(name, _) => {
                let qualifier = qualified_wildcard_name(name)?;
                if let Some(table_alias) = table_alias
                    && table_alias != qualifier
                {
                    return Err(DodamError::UnknownColumn(format!("{qualifier}.*")));
                }
                qualified_wildcards.push(qualifier);
                wildcard = true;
            }
            SelectItem::UnnamedExpr(
                expr @ (SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)),
            ) => {
                if let Some((column, field)) = parse_struct_field_access(expr, table_alias)? {
                    add_column_once(&mut columns, column.clone());
                    expressions.push(ProjectionExpression {
                        output_name: field.clone(),
                        expr: ScalarSqlExpression::StructField { column, field },
                    });
                } else {
                    let column = sql_column_name(expr, table_alias)?;
                    add_column_once(&mut columns, column.clone());
                    expressions.push(ProjectionExpression {
                        output_name: column.clone(),
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
                    parse_scalar_function_projection(function, None, table_alias)?
                {
                    for column in scalar_expression_columns(&expr.expr) {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(expr);
                } else {
                    let (aggregate, expression) = parse_aggregate_with_input_expression(
                        function,
                        table_alias,
                        aggregate_expressions.len(),
                    )?;
                    if let Some(spec) = filtered_aggregate_spec_from_function(
                        function,
                        table_alias,
                        &aggregate,
                        &expression,
                    )? {
                        filtered_aggregates.push(spec);
                    }
                    if let Some(expression) = expression {
                        for column in scalar_expression_columns(&expression.expr) {
                            add_column_once(&mut aggregate_expression_columns, column);
                        }
                        aggregate_expressions.push(expression);
                    }
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
                } else if let Ok(parsed) = parse_aggregate_output_expression(
                    expr,
                    table_alias,
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
                    let expression = parse_scalar_projection(expr, None, table_alias)?;
                    for column in scalar_expression_columns(&expression.expr) {
                        add_column_once(&mut columns, column);
                    }
                    expressions.push(expression);
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => match expr {
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                    if let Some((column, field)) = parse_struct_field_access(expr, table_alias)? {
                        add_column_once(&mut columns, column.clone());
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: ScalarSqlExpression::StructField { column, field },
                        });
                        aliases.push((alias.value.clone(), alias.value.clone()));
                    } else {
                        let column = sql_column_name(expr, table_alias)?;
                        add_column_once(&mut columns, column.clone());
                        expressions.push(ProjectionExpression {
                            output_name: alias.value.clone(),
                            expr: ScalarSqlExpression::Column(column.clone()),
                        });
                        aliases.push((alias.value.clone(), column));
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
                    } else if let Some(expr) =
                        parse_scalar_function_projection(function, Some(&alias.value), table_alias)?
                    {
                        for column in scalar_expression_columns(&expr.expr) {
                            add_column_once(&mut columns, column);
                        }
                        expressions.push(expr);
                    } else {
                        let (aggregate, expression) = parse_aggregate_with_input_expression(
                            function,
                            table_alias,
                            aggregate_expressions.len(),
                        )?;
                        if let Some(spec) = filtered_aggregate_spec_from_function(
                            function,
                            table_alias,
                            &aggregate,
                            &expression,
                        )? {
                            filtered_aggregates.push(spec);
                        }
                        if let Some(expression) = expression {
                            for column in scalar_expression_columns(&expression.expr) {
                                add_column_once(&mut aggregate_expression_columns, column);
                            }
                            aggregate_expressions.push(expression);
                        }
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
                    } else if let Ok(parsed) = parse_aggregate_output_expression(
                        expr,
                        table_alias,
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
                        let expression =
                            parse_scalar_projection(expr, Some(&alias.value), table_alias)?;
                        for column in scalar_expression_columns(&expression.expr) {
                            add_column_once(&mut columns, column);
                        }
                        expressions.push(expression);
                    }
                }
            },
            SelectItem::ExprWithAliases { .. } => {
                return Err(DodamError::UnsupportedSql(
                    "multi-alias SELECT items are not supported".to_string(),
                ));
            }
        }
    }

    if wildcard && select.projection.len() != 1 {
        return Err(DodamError::UnsupportedSql(
            "SELECT * cannot be mixed with other items".to_string(),
        ));
    }

    if aggregates.is_empty() {
        let ordinal_targets = projection_ordinal_targets(&columns, &aggregates, &expressions);
        return Ok(ParsedProjection {
            projection: if wildcard {
                Projection::All
            } else {
                Projection::Columns(columns)
            },
            aggregates,
            filtered_aggregates,
            aggregate_expressions,
            aliases,
            expressions: if wildcard { Vec::new() } else { expressions },
            ordinal_targets,
            qualified_wildcards,
        });
    }

    if !aggregates.is_empty()
        && expressions.iter().any(|expr| {
            !matches!(expr.expr, ScalarSqlExpression::Column(_))
                && !scalar_expression_references_aggregate(&expr.expr, &aggregates)
        })
    {
        return Err(DodamError::UnsupportedSql(
            "aggregate SELECT queries do not support scalar projection expressions yet".to_string(),
        ));
    }

    for column in &columns {
        if !group_by.iter().any(|group_column| group_column == column) {
            return Err(DodamError::UnsupportedSql(format!(
                "non-aggregate SELECT column {column} must appear in GROUP BY"
            )));
        }
    }
    let mut projected_columns = physical_projection_columns(&columns);
    for aggregate in &aggregates {
        if let Some(column) = aggregate.referenced_column() {
            if !column.starts_with("__dodam_agg_expr_") {
                add_column_once(&mut projected_columns, column.to_string());
            }
        }
    }
    add_filtered_aggregate_projection_columns(&mut projected_columns, &filtered_aggregates)?;
    for column in aggregate_expression_columns {
        add_column_once(&mut projected_columns, column);
    }
    for binding in &group_expression_bindings {
        for column in scalar_expression_columns(&binding.expression.expr) {
            add_column_once(&mut projected_columns, column);
        }
        aggregate_expressions.push(binding.expression.clone());
    }
    let ordinal_targets = projection_ordinal_targets(&columns, &aggregates, &expressions);
    Ok(ParsedProjection {
        projection: Projection::Columns(projected_columns),
        aggregates,
        filtered_aggregates,
        aggregate_expressions,
        aliases,
        expressions,
        ordinal_targets,
        qualified_wildcards,
    })
}

fn parse_scalar_projection(
    expr: &SqlExpr,
    alias: Option<&str>,
    table_alias: Option<&str>,
) -> Result<ProjectionExpression> {
    let parsed = parse_scalar_sql_expression(expr, table_alias)?;
    Ok(ProjectionExpression {
        output_name: alias.map_or_else(|| expr.to_string(), ToString::to_string),
        expr: parsed,
    })
}
