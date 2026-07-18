use super::*;

pub(super) fn parse_filter(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<FilterExpr> {
    Ok(FilterExpr::new(sql_expr_to_filter_expr(
        expr,
        aliases,
        table_alias,
        allow_aggregates,
    )?))
}

pub(super) fn sql_expr_to_filter_expr(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<Expr> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(Expr::And(
                Box::new(sql_expr_to_filter_expr(
                    left,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
                Box::new(sql_expr_to_filter_expr(
                    right,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Or => Ok(Expr::Or(
                Box::new(sql_expr_to_filter_expr(
                    left,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
                Box::new(sql_expr_to_filter_expr(
                    right,
                    aliases,
                    table_alias,
                    allow_aggregates,
                )?),
            )),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq => {
                if let (Ok(left), Ok(right)) = (sql_literal_value(left), sql_literal_value(right)) {
                    return Ok(Expr::Boolean(compare_literal_values(&left, op, &right)?));
                }
                let left = sql_filter_column(left, aliases, table_alias, allow_aggregates)?;
                let op = sql_comparison_op(op);
                if let Some(right) =
                    maybe_sql_filter_column(right, aliases, table_alias, allow_aggregates)?
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
                "unsupported WHERE operator: {op}"
            ))),
        },
        SqlExpr::Nested(expr) => {
            sql_expr_to_filter_expr(expr, aliases, table_alias, allow_aggregates)
        }
        SqlExpr::Value(value) => match &value.value {
            Value::Boolean(value) => Ok(Expr::Boolean(Some(*value))),
            Value::Null => Ok(Expr::Boolean(None)),
            _ => Err(DodamError::UnsupportedSql(format!(
                "unsupported WHERE expression: {expr}"
            ))),
        },
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let column = sql_filter_column(expr, aliases, table_alias, allow_aggregates)?;
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
            sql_expr_to_filter_expr(expr, aliases, table_alias, allow_aggregates)?,
        ))),
        SqlExpr::IsNull(expr) => Ok(Expr::IsNull {
            column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
            negated: false,
        }),
        SqlExpr::IsNotNull(expr) => Ok(Expr::IsNull {
            column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
            negated: true,
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => match sql_filter_column(expr, aliases, table_alias, allow_aggregates) {
            Ok(column) => Ok(Expr::InList {
                column,
                negated: *negated,
                has_null: literal_list_contains_null(list)?,
                values: non_null_literal_values(list)?,
            }),
            Err(error) => {
                let value = sql_literal_value(expr).map_err(|_| error)?;
                Ok(Expr::Boolean(evaluate_literal_in_list(
                    &value, list, *negated,
                )?))
            }
        },
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
                column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
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
                column: sql_filter_column(expr, aliases, table_alias, allow_aggregates)?,
                pattern: sql_like_pattern(pattern)?,
                negated: *negated,
                escape: sql_like_escape(escape_char)?,
                case_insensitive: true,
            })
        }
        _ => Err(DodamError::UnsupportedSql(format!(
            "unsupported WHERE expression: {expr}"
        ))),
    }
}

pub(super) fn sql_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<String> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            Ok(resolve_alias(&sql_column_name(expr, table_alias)?, aliases))
        }
        SqlExpr::Function(function) if allow_aggregates => {
            Ok(parse_aggregate(function, table_alias)?.to_string())
        }
        SqlExpr::Nested(expr) => sql_filter_column(expr, aliases, table_alias, allow_aggregates),
        _ => Err(DodamError::UnsupportedSql(format!(
            "expected column or aggregate expression, got {expr}"
        ))),
    }
}

fn maybe_sql_filter_column(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    table_alias: Option<&str>,
    allow_aggregates: bool,
) -> Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) | SqlExpr::Nested(_) => {
            sql_filter_column(expr, aliases, table_alias, allow_aggregates).map(Some)
        }
        _ => Ok(None),
    }
}
