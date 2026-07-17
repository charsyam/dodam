use super::*;

pub(super) fn group_by_synthetic_column(index: usize) -> String {
    format!("__dodam_group_expr_{index}")
}

fn is_group_by_synthetic_column(column: &str) -> bool {
    column.starts_with("__dodam_group_expr_")
}

fn resolve_group_by_projection_references(
    expressions: &[SqlExpr],
    select: &Select,
) -> Result<Vec<SqlExpr>> {
    expressions
        .iter()
        .map(|expr| resolve_group_by_projection_reference(expr, select))
        .collect()
}

fn resolve_group_by_projection_reference(expr: &SqlExpr, select: &Select) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Value(value) => {
            let Value::Number(number, _) = &value.value else {
                return Ok(expr.clone());
            };
            let ordinal = number.parse::<usize>().map_err(|_| {
                DodamError::UnsupportedSql(format!("invalid GROUP BY ordinal: {number}"))
            })?;
            if ordinal == 0 {
                return Err(DodamError::UnsupportedSql(
                    "GROUP BY ordinal must be greater than zero".to_string(),
                ));
            }
            projection_expr_at(select, ordinal).ok_or_else(|| {
                DodamError::UnsupportedSql(format!("GROUP BY position {ordinal} is out of range"))
            })
        }
        SqlExpr::Identifier(ident) => projection_alias_expr(select, &ident.value)
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| Ok(expr.clone())),
        _ => Ok(expr.clone()),
    }
}

fn projection_expr_at(select: &Select, ordinal: usize) -> Option<SqlExpr> {
    select
        .projection
        .get(ordinal - 1)
        .and_then(select_item_expr)
}

fn projection_alias_expr<'a>(select: &'a Select, alias: &str) -> Option<&'a SqlExpr> {
    select.projection.iter().find_map(|item| match item {
        SelectItem::ExprWithAlias {
            expr,
            alias: item_alias,
        } if item_alias.value.eq_ignore_ascii_case(alias) => Some(expr),
        _ => None,
    })
}

fn select_item_expr(item: &SelectItem) -> Option<SqlExpr> {
    match item {
        SelectItem::UnnamedExpr(expr) => Some(expr.clone()),
        SelectItem::ExprWithAlias { expr, .. } => Some(expr.clone()),
        _ => None,
    }
}

fn sql_expr_contains_aggregate(expr: &SqlExpr) -> Result<bool> {
    match expr {
        SqlExpr::Function(function) => {
            let name = object_name_to_string(&function.name)?;
            if matches!(
                name.to_ascii_lowercase().as_str(),
                "count" | "sum" | "avg" | "min" | "max"
            ) {
                return Ok(true);
            }
            function_arguments_contain_aggregate(&function.args)
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            Ok(sql_expr_contains_aggregate(left)? || sql_expr_contains_aggregate(right)?)
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::Cast { expr, .. }
        | SqlExpr::Extract { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr) => sql_expr_contains_aggregate(expr),
        SqlExpr::Between {
            expr, low, high, ..
        } => Ok(sql_expr_contains_aggregate(expr)?
            || sql_expr_contains_aggregate(low)?
            || sql_expr_contains_aggregate(high)?),
        SqlExpr::InList { expr, list, .. } => {
            if sql_expr_contains_aggregate(expr)? {
                return Ok(true);
            }
            list.iter().try_fold(false, |found, expr| {
                Ok(found || sql_expr_contains_aggregate(expr)?)
            })
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            Ok(sql_expr_contains_aggregate(expr)? || sql_expr_contains_aggregate(pattern)?)
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(operand) = operand.as_deref()
                && sql_expr_contains_aggregate(operand)?
            {
                return Ok(true);
            }
            for when in conditions {
                if sql_expr_contains_aggregate(&when.condition)?
                    || sql_expr_contains_aggregate(&when.result)?
                {
                    return Ok(true);
                }
            }
            if let Some(else_result) = else_result.as_deref()
                && sql_expr_contains_aggregate(else_result)?
            {
                return Ok(true);
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn function_arguments_contain_aggregate(arguments: &FunctionArguments) -> Result<bool> {
    let FunctionArguments::List(arguments) = arguments else {
        return Ok(false);
    };
    arguments.args.iter().try_fold(false, |found, arg| {
        Ok(found || function_arg_contains_aggregate(arg)?)
    })
}

fn function_arg_contains_aggregate(arg: &FunctionArg) -> Result<bool> {
    match arg {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => sql_expr_contains_aggregate(expr),
        FunctionArg::Named { arg, .. } | FunctionArg::ExprNamed { arg, .. } => {
            function_arg_expr_contains_aggregate(arg)
        }
        _ => Ok(false),
    }
}

fn function_arg_expr_contains_aggregate(arg: &FunctionArgExpr) -> Result<bool> {
    match arg {
        FunctionArgExpr::Expr(expr) => sql_expr_contains_aggregate(expr),
        _ => Ok(false),
    }
}

pub(super) fn qualified_wildcard_name(kind: &SelectItemQualifiedWildcardKind) -> Result<String> {
    match kind {
        SelectItemQualifiedWildcardKind::ObjectName(name) => object_name_to_string(name),
        SelectItemQualifiedWildcardKind::Expr(expr) => Err(DodamError::UnsupportedSql(format!(
            "expression qualified wildcard is not supported: {expr}.*"
        ))),
    }
}

pub(super) fn physical_projection_columns(columns: &[String]) -> Vec<String> {
    columns
        .iter()
        .filter(|column| !is_group_by_synthetic_column(column))
        .cloned()
        .collect()
}

pub(super) fn projected_group_expression(
    expr: &SqlExpr,
    bindings: &[GroupExpressionBinding],
) -> Option<(String, ScalarSqlExpression)> {
    bindings
        .iter()
        .find(|binding| binding.source == expr.to_string())
        .map(|binding| {
            (
                binding.expression.output_name.clone(),
                binding.expression.expr.clone(),
            )
        })
}

pub(super) fn group_expression_bindings(
    select: &Select,
    table_alias: Option<&str>,
) -> Result<Vec<GroupExpressionBinding>> {
    let expressions = group_by_expressions(select)?;
    expressions
        .iter()
        .enumerate()
        .filter_map(|(index, expr)| {
            if sql_column_name(expr, table_alias).is_ok() {
                return None;
            }
            Some(
                parse_scalar_sql_expression(expr, table_alias).map(|parsed| {
                    GroupExpressionBinding {
                        source: expr.to_string(),
                        expression: ProjectionExpression {
                            output_name: group_by_synthetic_column(index),
                            expr: parsed,
                        },
                    }
                }),
            )
        })
        .collect()
}

pub(super) fn join_group_expression_bindings(
    select: &Select,
    table_aliases: &[&str],
) -> Result<Vec<GroupExpressionBinding>> {
    let expressions = group_by_expressions(select)?;
    expressions
        .iter()
        .enumerate()
        .filter_map(|(index, expr)| {
            if join_column_name(expr, table_aliases).is_ok() {
                return None;
            }
            Some(
                parse_join_scalar_sql_expression(expr, table_aliases).map(|parsed| {
                    GroupExpressionBinding {
                        source: expr.to_string(),
                        expression: ProjectionExpression {
                            output_name: group_by_synthetic_column(index),
                            expr: parsed,
                        },
                    }
                }),
            )
        })
        .collect()
}

pub(super) fn parse_group_by(select: &Select, table_alias: Option<&str>) -> Result<Vec<String>> {
    let expressions = group_by_expressions(select)?;
    expressions
        .iter()
        .enumerate()
        .map(|(index, expr)| {
            sql_column_name(expr, table_alias).or_else(|_| {
                parse_scalar_sql_expression(expr, table_alias)
                    .map(|_| group_by_synthetic_column(index))
            })
        })
        .collect::<Result<Vec<_>>>()
}

pub(super) fn group_by_expressions(select: &Select) -> Result<Vec<SqlExpr>> {
    match &select.group_by {
        GroupByExpr::Expressions(expressions, modifiers) if modifiers.is_empty() => {
            resolve_group_by_projection_references(expressions, select)
        }
        GroupByExpr::All(modifiers) if modifiers.is_empty() => {
            group_by_all_projection_expressions(select)
        }
        GroupByExpr::Expressions(_, _) | GroupByExpr::All(_) => Err(DodamError::UnsupportedSql(
            "GROUP BY modifiers are not supported".to_string(),
        )),
    }
}

fn group_by_all_projection_expressions(select: &Select) -> Result<Vec<SqlExpr>> {
    let mut expressions = Vec::new();
    for item in &select.projection {
        let Some(expr) = select_item_expr(item) else {
            return Err(DodamError::UnsupportedSql(
                "GROUP BY ALL with wildcard projections is not supported".to_string(),
            ));
        };
        if sql_expr_contains_aggregate(&expr)? {
            continue;
        }
        expressions.push(expr);
    }
    Ok(expressions)
}

pub(super) fn projection_ordinal_targets(
    columns: &[String],
    aggregates: &[AggregateExpr],
    expressions: &[ProjectionExpression],
) -> Vec<String> {
    if !aggregates.is_empty() {
        columns
            .iter()
            .cloned()
            .chain(aggregates.iter().map(ToString::to_string))
            .collect::<Vec<_>>()
    } else if !expressions.is_empty() {
        expressions
            .iter()
            .map(|expression| expression.output_name.clone())
            .collect::<Vec<_>>()
    } else {
        columns.to_vec()
    }
}
