use super::*;

pub(super) fn parse_join_filter(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<FilterExpr> {
    Ok(FilterExpr::new(join_expr_to_filter_expr(
        expr,
        aliases,
        table_aliases,
        allow_aggregates,
    )?))
}

pub(super) fn parse_join_filter_plan(
    expr: Option<&SqlExpr>,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<(Option<FilterExpr>, Option<SqlExpr>)> {
    let Some(expr) = expr else {
        return Ok((None, None));
    };
    let simplified_expr =
        simplify_simple_case_literal_equality(expr).unwrap_or_else(|| expr.clone());
    let expr = &simplified_expr;
    if join_predicate_requires_expression_path(expr, table_aliases)? {
        return Ok((
            safe_expression_pushdown_filter(expr, None, PredicateParserKind::Join(table_aliases))?,
            Some(expr.clone()),
        ));
    }
    Ok((
        Some(parse_join_filter(
            expr,
            aliases,
            table_aliases,
            allow_aggregates,
        )?),
        None,
    ))
}

fn simplify_simple_case_literal_equality(expr: &SqlExpr) -> Option<SqlExpr> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::And => {
            let left = simplify_simple_case_literal_equality(left).unwrap_or_else(|| *left.clone());
            let right =
                simplify_simple_case_literal_equality(right).unwrap_or_else(|| *right.clone());
            Some(SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            })
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::Eq => {
            simplify_simple_case_literal_equality_side(left, right)
                .or_else(|| simplify_simple_case_literal_equality_side(right, left))
        }
        SqlExpr::Nested(expr) => {
            simplify_simple_case_literal_equality(expr).map(|expr| SqlExpr::Nested(Box::new(expr)))
        }
        _ => None,
    }
}

fn simplify_simple_case_literal_equality_side(
    case_expr: &SqlExpr,
    target_expr: &SqlExpr,
) -> Option<SqlExpr> {
    let target = sql_literal_value(target_expr).ok()?;
    let SqlExpr::Case {
        operand,
        conditions,
        else_result,
        ..
    } = case_expr
    else {
        return None;
    };
    let else_result = sql_literal_value(else_result.as_ref()?).ok()?;
    if else_result == target {
        return None;
    }
    let rewritten_conditions = case_conditions_from_operand(operand.as_deref(), conditions);
    let selected = conditions
        .iter()
        .zip(rewritten_conditions)
        .filter_map(|(when, condition)| {
            (sql_literal_value(&when.result).ok()? == target).then_some(condition)
        })
        .collect::<Vec<_>>();
    combine_sql_and_disjuncts(selected)
}

fn join_predicate_requires_expression_path(expr: &SqlExpr, table_aliases: &[&str]) -> Result<bool> {
    match expr {
        SqlExpr::BinaryOp { left, op, right }
            if matches!(op, BinaryOperator::And | BinaryOperator::Or) =>
        {
            Ok(
                join_predicate_requires_expression_path(left, table_aliases)?
                    || join_predicate_requires_expression_path(right, table_aliases)?,
            )
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            join_predicate_requires_expression_path(expr, table_aliases)
        }
        SqlExpr::Nested(expr) => join_predicate_requires_expression_path(expr, table_aliases),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
            ) =>
        {
            Ok(
                join_scalar_predicate_side_requires_expression(left, table_aliases)?
                    || join_scalar_predicate_side_requires_expression(right, table_aliases)?,
            )
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            join_scalar_predicate_side_requires_expression(expr, table_aliases)
        }
        SqlExpr::InList { expr, list, .. } => {
            if join_scalar_predicate_side_requires_expression(expr, table_aliases)? {
                return Ok(true);
            }
            list.iter().try_fold(false, |found, expr| {
                Ok(found || join_scalar_predicate_side_requires_expression(expr, table_aliases)?)
            })
        }
        SqlExpr::Like { expr, pattern, .. } => Ok(join_scalar_predicate_side_requires_expression(
            expr,
            table_aliases,
        )?
            || join_scalar_predicate_side_requires_expression(pattern, table_aliases)?),
        SqlExpr::Between {
            expr, low, high, ..
        } => Ok(
            join_scalar_predicate_side_requires_expression(expr, table_aliases)?
                || join_scalar_predicate_side_requires_expression(low, table_aliases)?
                || join_scalar_predicate_side_requires_expression(high, table_aliases)?,
        ),
        _ => Ok(false),
    }
}

fn join_scalar_predicate_side_requires_expression(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<bool> {
    let expression = parse_join_scalar_sql_expression(expr, table_aliases)?;
    Ok(!matches!(
        expression,
        ScalarSqlExpression::Column(_) | ScalarSqlExpression::Literal(_)
    ))
}

pub(super) fn join_expr_to_filter_expr(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<Expr> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(Expr::And(
                Box::new(join_expr_to_filter_expr(
                    left,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
                Box::new(join_expr_to_filter_expr(
                    right,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Or => Ok(Expr::Or(
                Box::new(join_expr_to_filter_expr(
                    left,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
                Box::new(join_expr_to_filter_expr(
                    right,
                    aliases,
                    table_aliases,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq => {
                let left_column =
                    join_filter_column(left, aliases, table_aliases, allow_aggregates);
                if left_column.is_err()
                    && let Some(right_column) =
                        maybe_join_filter_column(right, aliases, table_aliases, allow_aggregates)?
                {
                    return Ok(Expr::Comparison(ComparisonExpr {
                        column: right_column,
                        op: reverse_comparison_op(sql_comparison_op(op)),
                        value: sql_literal_value(left)?,
                    }));
                }
                let left = left_column?;
                let op = sql_comparison_op(op);
                if let Some(right) =
                    maybe_join_filter_column(right, aliases, table_aliases, allow_aggregates)?
                {
                    Ok(Expr::ColumnComparison { left, op, right })
                } else {
                    Ok(Expr::Comparison(ComparisonExpr {
                        column: left,
                        op,
                        value: sql_literal_value(right)?,
                    }))
                }
            }
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported JOIN WHERE operator: {op}"
            ))),
        },
        SqlExpr::Nested(expr) => {
            join_expr_to_filter_expr(expr, aliases, table_aliases, allow_aggregates)
        }
        SqlExpr::Value(value) => match &value.value {
            Value::Boolean(value) => Ok(Expr::Boolean(Some(*value))),
            Value::Null => Ok(Expr::Boolean(None)),
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported JOIN WHERE expression: {expr}"
            ))),
        },
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let column = join_filter_column(expr, aliases, table_aliases, allow_aggregates)?;
            let low = sql_literal_value(low)?;
            let high = sql_literal_value(high)?;
            if *negated {
                Ok(Expr::Or(
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column: column.clone(),
                        op: ComparisonOp::Lt,
                        value: low,
                    })),
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column,
                        op: ComparisonOp::Gt,
                        value: high,
                    })),
                ))
            } else {
                Ok(Expr::And(
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column: column.clone(),
                        op: ComparisonOp::GtEq,
                        value: low,
                    })),
                    Box::new(Expr::Comparison(ComparisonExpr {
                        column,
                        op: ComparisonOp::LtEq,
                        value: high,
                    })),
                ))
            }
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => Ok(Expr::Not(Box::new(
            join_expr_to_filter_expr(expr, aliases, table_aliases, allow_aggregates)?,
        ))),
        SqlExpr::IsNull(expr) => Ok(Expr::IsNull {
            column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
            negated: false,
        }),
        SqlExpr::IsNotNull(expr) => Ok(Expr::IsNull {
            column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
            negated: true,
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => Ok(Expr::InList {
            column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
            negated: *negated,
            has_null: literal_list_contains_null(list)?,
            values: non_null_literal_values(list)?,
        }),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(DodamError::UnsupportedSql(
                    "LIKE ANY is not supported".to_string(),
                ));
            }
            Ok(Expr::Like {
                column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
                pattern: sql_like_pattern(pattern)?,
                negated: *negated,
                escape: sql_like_escape(escape_char)?,
                case_insensitive: false,
            })
        }
        SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(DodamError::UnsupportedSql(
                    "ILIKE ANY is not supported".to_string(),
                ));
            }
            Ok(Expr::Like {
                column: join_filter_column(expr, aliases, table_aliases, allow_aggregates)?,
                pattern: sql_like_pattern(pattern)?,
                negated: *negated,
                escape: sql_like_escape(escape_char)?,
                case_insensitive: true,
            })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported JOIN WHERE expression: {expr}"
        ))),
    }
}

fn reverse_comparison_op(op: ComparisonOp) -> ComparisonOp {
    match op {
        ComparisonOp::Eq => ComparisonOp::Eq,
        ComparisonOp::NotEq => ComparisonOp::NotEq,
        ComparisonOp::Gt => ComparisonOp::Lt,
        ComparisonOp::GtEq => ComparisonOp::LtEq,
        ComparisonOp::Lt => ComparisonOp::Gt,
        ComparisonOp::LtEq => ComparisonOp::GtEq,
    }
}

fn join_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<String> {
    match expr {
        SqlExpr::Identifier(ident) => alias_target(&ident.value, aliases)
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| join_column_name(expr, table_aliases)),
        SqlExpr::CompoundIdentifier(_) => join_column_name(expr, table_aliases),
        SqlExpr::Function(function) if allow_aggregates => {
            let function_name = function.to_string();
            let resolved = resolve_alias(&function_name, aliases);
            if resolved == function_name {
                Ok(parse_join_aggregate(function, table_aliases)?.to_string())
            } else {
                Ok(resolved)
            }
        }
        SqlExpr::Nested(expr) => join_filter_column(expr, aliases, table_aliases, allow_aggregates),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected JOIN column or aggregate expression, got {expr}"
        ))),
    }
}

fn maybe_join_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_aliases: &[&str],
    allow_aggregates: bool,
) -> Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) | SqlExpr::Nested(_) => {
            join_filter_column(expr, aliases, table_aliases, allow_aggregates).map(Some)
        }
        _ => Ok(None),
    }
}
