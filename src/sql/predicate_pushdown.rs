use super::*;

pub(super) async fn split_subquery_and_expression_filters(
    engine: &DodamEngine,
    selection: &SqlExpr,
    table_alias: Option<&str>,
    batch_size: usize,
) -> Result<(Option<FilterExpr>, Vec<SqlExpr>)> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let mut filters = Vec::new();
    let mut expression_filters = Vec::new();
    for conjunct in conjuncts {
        if predicate_requires_expression_path(&conjunct)
            && !expr_contains_materializable_subquery(&conjunct)
        {
            if let Some(filter) = safe_expression_pushdown_filter(
                &conjunct,
                table_alias,
                PredicateParserKind::Single,
            )? {
                filters.push(filter.expr().clone());
            }
            expression_filters.push(conjunct);
            continue;
        }
        if let Some(filter) = Box::pin(parse_filter_with_subqueries(
            engine,
            &conjunct,
            &[],
            table_alias,
            false,
            batch_size,
        ))
        .await?
        {
            filters.push(filter);
        }
    }
    Ok((combine_expr_filters(filters), expression_filters))
}

#[derive(Clone, Copy)]
pub(super) enum PredicateParserKind<'a> {
    Single,
    Join(&'a [&'a str]),
}

pub(super) fn safe_expression_pushdown_filter(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let filter = match expr {
        SqlExpr::Nested(expr) => safe_expression_pushdown_filter(expr, table_alias, parser_kind)?,
        SqlExpr::UnaryOp { op, .. } if *op == UnaryOperator::Not => None,
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::And => {
            combine_filter_options(
                safe_expression_pushdown_filter(left, table_alias, parser_kind)?,
                safe_expression_pushdown_filter(right, table_alias, parser_kind)?,
            )
        }
        SqlExpr::BinaryOp { left, op, right } if *op == BinaryOperator::Or => {
            let Some(left) = safe_expression_pushdown_filter(left, table_alias, parser_kind)?
            else {
                return Ok(None);
            };
            let Some(right) = safe_expression_pushdown_filter(right, table_alias, parser_kind)?
            else {
                return Ok(None);
            };
            Some(FilterExpr::new(Expr::Or(
                Box::new(left.expr().clone()),
                Box::new(right.expr().clone()),
            )))
        }
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
            safe_expression_comparison_pushdown(left, op, right, table_alias, parser_kind)?
        }
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => safe_expression_in_list_pushdown(expr, list, *negated, table_alias, parser_kind)?,
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => safe_expression_between_pushdown(expr, *negated, low, high, table_alias, parser_kind)?,
        SqlExpr::IsNull(expr) => {
            safe_expression_null_pushdown(expr, false, table_alias, parser_kind)?
        }
        SqlExpr::IsNotNull(expr) => {
            safe_expression_null_pushdown(expr, true, table_alias, parser_kind)?
        }
        SqlExpr::Like {
            expr,
            pattern,
            negated,
            any,
            escape_char,
        } => safe_expression_like_pushdown(
            expr,
            pattern,
            *negated,
            *any,
            escape_char,
            table_alias,
            parser_kind,
            false,
        )?,
        SqlExpr::ILike {
            expr,
            pattern,
            negated,
            any,
            escape_char,
        } => safe_expression_like_pushdown(
            expr,
            pattern,
            *negated,
            *any,
            escape_char,
            table_alias,
            parser_kind,
            true,
        )?,
        _ => None,
    };
    Ok(filter)
}

fn safe_expression_between_pushdown(
    expr: &SqlExpr,
    negated: bool,
    low: &SqlExpr,
    high: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    let low = sql_literal_value(low)?;
    let high = sql_literal_value(high)?;
    let lower = Expr::Comparison(ComparisonExpr {
        column: column.clone(),
        op: if negated {
            ComparisonOp::Lt
        } else {
            ComparisonOp::GtEq
        },
        value: low,
    });
    let upper = Expr::Comparison(ComparisonExpr {
        column,
        op: if negated {
            ComparisonOp::Gt
        } else {
            ComparisonOp::LtEq
        },
        value: high,
    });
    Ok(Some(FilterExpr::new(if negated {
        Expr::Or(Box::new(lower), Box::new(upper))
    } else {
        Expr::And(Box::new(lower), Box::new(upper))
    })))
}

fn safe_expression_comparison_pushdown(
    left: &SqlExpr,
    op: &BinaryOperator,
    right: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    if let Some(filter) =
        safe_column_wrapper_comparison_pushdown(left, op, right, table_alias, parser_kind)?
    {
        return Ok(Some(filter));
    }
    safe_column_wrapper_comparison_pushdown(
        right,
        &reverse_binary_operator(op),
        left,
        table_alias,
        parser_kind,
    )
}

fn safe_expression_in_list_pushdown(
    expr: &SqlExpr,
    list: &[SqlExpr],
    negated: bool,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    if negated {
        return Ok(None);
    }
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    let values = non_null_literal_values(list)?;
    if values.is_empty() {
        return Ok(None);
    }
    Ok(Some(FilterExpr::new(Expr::InList {
        column,
        values,
        negated: false,
        has_null: literal_list_contains_null(list)?,
    })))
}

fn safe_expression_null_pushdown(
    expr: &SqlExpr,
    negated: bool,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    Ok(Some(FilterExpr::new(Expr::IsNull { column, negated })))
}

fn safe_expression_like_pushdown(
    expr: &SqlExpr,
    pattern: &SqlExpr,
    negated: bool,
    any: bool,
    escape_char: &Option<sqlparser::ast::ValueWithSpan>,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
    case_insensitive: bool,
) -> Result<Option<FilterExpr>> {
    if any {
        return Ok(None);
    }
    let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? else {
        return Ok(None);
    };
    Ok(Some(FilterExpr::new(Expr::Like {
        column,
        pattern: sql_like_pattern(pattern)?,
        negated,
        escape: sql_like_escape(escape_char)?,
        case_insensitive,
    })))
}

fn safe_column_wrapper_comparison_pushdown(
    expr: &SqlExpr,
    op: &BinaryOperator,
    literal_expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let Ok(literal) = sql_literal_value(literal_expr) else {
        return Ok(None);
    };
    if let Some(column) = simple_column_wrapped_expression(expr, table_alias, parser_kind)? {
        return Ok(Some(FilterExpr::new(Expr::Comparison(ComparisonExpr {
            column,
            op: sql_comparison_op(op),
            value: literal,
        }))));
    }
    coalesce_comparison_pushdown(expr, op, &literal, table_alias, parser_kind)
}

fn simple_column_wrapped_expression(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<String>> {
    match parse_predicate_scalar_expr(expr, table_alias, parser_kind)? {
        ScalarSqlExpression::Column(column) => Ok(Some(column)),
        _ => Ok(None),
    }
}

fn coalesce_comparison_pushdown(
    expr: &SqlExpr,
    op: &BinaryOperator,
    literal: &LiteralValue,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<Option<FilterExpr>> {
    let ScalarSqlExpression::Coalesce(values) =
        parse_predicate_scalar_expr(expr, table_alias, parser_kind)?
    else {
        return Ok(None);
    };
    let [
        ScalarSqlExpression::Column(column),
        ScalarSqlExpression::Literal(fallback),
    ] = values.as_slice()
    else {
        return Ok(None);
    };
    let column_filter = Expr::Comparison(ComparisonExpr {
        column: column.clone(),
        op: sql_comparison_op(op),
        value: literal.clone(),
    });
    if compare_literal_values(fallback, op, literal)? == Some(true) {
        Ok(Some(FilterExpr::new(Expr::Or(
            Box::new(column_filter),
            Box::new(Expr::IsNull {
                column: column.clone(),
                negated: false,
            }),
        ))))
    } else {
        Ok(Some(FilterExpr::new(column_filter)))
    }
}

fn parse_predicate_scalar_expr(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    parser_kind: PredicateParserKind,
) -> Result<ScalarSqlExpression> {
    match parser_kind {
        PredicateParserKind::Single => parse_scalar_sql_expression(expr, table_alias),
        PredicateParserKind::Join(table_aliases) => {
            parse_join_scalar_sql_expression(expr, table_aliases)
        }
    }
}
