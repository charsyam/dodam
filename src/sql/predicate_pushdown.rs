use super::*;

pub(super) fn predicate_requires_expression_path(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::BinaryOp { left, op, right }
            if matches!(op, BinaryOperator::And | BinaryOperator::Or) =>
        {
            predicate_requires_expression_path(left) || predicate_requires_expression_path(right)
        }
        SqlExpr::UnaryOp { op, expr } if *op == UnaryOperator::Not => {
            predicate_requires_expression_path(expr)
        }
        SqlExpr::Nested(expr) => predicate_requires_expression_path(expr),
        SqlExpr::BinaryOp { left, right, .. } => {
            scalar_predicate_side_requires_expression(left)
                || scalar_predicate_side_requires_expression(right)
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => {
            scalar_predicate_side_requires_expression(expr)
        }
        SqlExpr::InList { expr, list, .. } => {
            scalar_predicate_side_requires_expression(expr)
                || list.iter().any(scalar_predicate_side_requires_expression)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            scalar_predicate_side_requires_expression(expr)
                || scalar_predicate_side_requires_expression(low)
                || scalar_predicate_side_requires_expression(high)
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            scalar_predicate_side_requires_expression(expr)
                || scalar_predicate_side_requires_expression(pattern)
        }
        _ => false,
    }
}

pub(super) fn scalar_predicate_side_requires_expression(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Identifier(_) => false,
        SqlExpr::CompoundIdentifier(parts) => parts.len() > 1,
        SqlExpr::Function(_)
        | SqlExpr::Substring { .. }
        | SqlExpr::Cast { .. }
        | SqlExpr::Case { .. }
        | SqlExpr::CompoundFieldAccess { .. } => true,
        _ => sql_literal_value(expr).is_err(),
    }
}

pub(super) fn predicate_expression_columns(
    expr: &SqlExpr,
    table_alias: Option<&str>,
) -> Result<Vec<String>> {
    let mut columns = Vec::new();
    collect_predicate_expression_columns(expr, table_alias, &mut columns)?;
    Ok(columns)
}

pub(super) fn collect_predicate_expression_columns(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_predicate_expression_columns(left, table_alias, columns)?;
            collect_predicate_expression_columns(right, table_alias, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr) => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
        }
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            if let Some((column, _)) = parse_struct_field_access(expr, table_alias)? {
                add_column_once(columns, column);
            } else {
                add_column_once(columns, sql_column_name(expr, table_alias)?);
            }
        }
        SqlExpr::CompoundFieldAccess { .. } => {
            for column in
                scalar_expression_columns(&parse_scalar_sql_expression(expr, table_alias)?)
            {
                add_column_once(columns, column);
            }
        }
        SqlExpr::Function(function) => {
            if let Some(expression) = parse_scalar_function_projection(function, None, table_alias)?
            {
                for column in scalar_expression_columns(&expression.expr) {
                    add_column_once(columns, column);
                }
            }
        }
        SqlExpr::Substring { .. } => {
            for column in
                scalar_expression_columns(&parse_scalar_sql_expression(expr, table_alias)?)
            {
                add_column_once(columns, column);
            }
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
            for item in list {
                collect_predicate_expression_columns(item, table_alias, columns)?;
            }
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
            collect_predicate_expression_columns(pattern, table_alias, columns)?;
        }
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            collect_subquery_outer_columns(subquery, table_alias, columns)?;
        }
        SqlExpr::Cast { expr, .. } => {
            collect_predicate_expression_columns(expr, table_alias, columns)?;
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
}

fn collect_subquery_outer_columns(
    query: &Query,
    table_alias: Option<&str>,
    columns: &mut Vec<String>,
) -> Result<()> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(());
    };
    if let Some(selection) = select.selection.as_ref() {
        collect_outer_column_candidates(selection, table_alias, columns)?;
    }
    Ok(())
}

fn collect_outer_column_candidates(
    expr: &SqlExpr,
    table_alias: Option<&str>,
    columns: &mut Vec<String>,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_outer_column_candidates(left, table_alias, columns)?;
            collect_outer_column_candidates(right, table_alias, columns)?;
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => {
            collect_outer_column_candidates(expr, table_alias, columns)?;
        }
        SqlExpr::Identifier(ident) => {
            if unqualified_column_matches_table_alias(&ident.value, table_alias) {
                add_column_once(columns, ident.value.clone());
            }
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if let [qualifier, column] = parts.as_slice()
                && table_alias.is_some_and(|alias| qualifier.value.eq_ignore_ascii_case(alias))
            {
                add_column_once(columns, column.value.clone());
            }
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_outer_column_candidates(expr, table_alias, columns)?;
            for item in list {
                collect_outer_column_candidates(item, table_alias, columns)?;
            }
        }
        SqlExpr::Exists { subquery, .. }
        | SqlExpr::InSubquery { subquery, .. }
        | SqlExpr::Subquery(subquery) => {
            collect_subquery_outer_columns(subquery, table_alias, columns)?;
        }
        SqlExpr::Function(function) => {
            for arg in function_arg_exprs(function) {
                collect_outer_column_candidates(arg, table_alias, columns)?;
            }
        }
        SqlExpr::Value(_) => {}
        _ => {}
    }
    Ok(())
}

pub(super) fn unqualified_column_matches_table_alias(
    column: &str,
    table_alias: Option<&str>,
) -> bool {
    let Some(table_alias) = table_alias else {
        return false;
    };
    let Some((prefix, _)) = column.split_once('_') else {
        return false;
    };
    infer_tpch_table_alias(prefix, &[table_alias]).is_some_and(|alias| alias == table_alias)
}

pub(super) fn function_arg_exprs(function: &sqlparser::ast::Function) -> Vec<&SqlExpr> {
    let FunctionArguments::List(args) = &function.args else {
        return Vec::new();
    };
    args.args
        .iter()
        .filter_map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Some(expr),
            _ => None,
        })
        .collect()
}

pub(super) fn expr_contains_scalar_subquery(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Subquery(_) => true,
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_contains_scalar_subquery(left) || expr_contains_scalar_subquery(right)
        }
        SqlExpr::Nested(expr) | SqlExpr::UnaryOp { expr, .. } => {
            expr_contains_scalar_subquery(expr)
        }
        SqlExpr::IsNull(expr) | SqlExpr::IsNotNull(expr) => expr_contains_scalar_subquery(expr),
        SqlExpr::InList { expr, list, .. } => {
            expr_contains_scalar_subquery(expr) || list.iter().any(expr_contains_scalar_subquery)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            expr_contains_scalar_subquery(expr)
                || expr_contains_scalar_subquery(low)
                || expr_contains_scalar_subquery(high)
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            expr_contains_scalar_subquery(expr) || expr_contains_scalar_subquery(pattern)
        }
        _ => false,
    }
}

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
