use super::*;

pub(super) fn collect_sql_and_conjuncts(expr: &SqlExpr, conjuncts: &mut Vec<SqlExpr>) {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_sql_and_conjuncts(left, conjuncts);
            collect_sql_and_conjuncts(right, conjuncts);
        }
        SqlExpr::Nested(expr) => collect_sql_and_conjuncts(expr, conjuncts),
        expr => conjuncts.push(expr.clone()),
    }
}

pub(super) fn collect_sql_or_disjuncts(expr: &SqlExpr, disjuncts: &mut Vec<SqlExpr>) {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Or,
            right,
        } => {
            collect_sql_or_disjuncts(left, disjuncts);
            collect_sql_or_disjuncts(right, disjuncts);
        }
        SqlExpr::Nested(expr) => collect_sql_or_disjuncts(expr, disjuncts),
        expr => disjuncts.push(expr.clone()),
    }
}

pub(super) fn combine_sql_and_conjuncts(mut conjuncts: Vec<SqlExpr>) -> Option<SqlExpr> {
    let first = conjuncts.pop()?;
    Some(
        conjuncts
            .into_iter()
            .fold(first, |right, left| SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            }),
    )
}

pub(super) fn combine_sql_and_disjuncts(mut disjuncts: Vec<SqlExpr>) -> Option<SqlExpr> {
    let first = disjuncts.pop()?;
    Some(
        disjuncts
            .into_iter()
            .fold(first, |right, left| SqlExpr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            }),
    )
}

pub(super) fn comma_join_equality_keys(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<Option<(String, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let left_prefix = format!("{left_alias}.");
    let right_prefix = format!("{right_alias}.");
    if left_column.starts_with(&left_prefix) && right_column.starts_with(&right_prefix) {
        return Ok(Some((
            left_column
                .strip_prefix(&left_prefix)
                .expect("left prefix")
                .to_string(),
            right_column
                .strip_prefix(&right_prefix)
                .expect("right prefix")
                .to_string(),
        )));
    }
    if left_column.starts_with(&right_prefix) && right_column.starts_with(&left_prefix) {
        return Ok(Some((
            right_column
                .strip_prefix(&left_prefix)
                .expect("left prefix")
                .to_string(),
            left_column
                .strip_prefix(&right_prefix)
                .expect("right prefix")
                .to_string(),
        )));
    }
    Ok(None)
}

pub(super) fn comma_join_keys_for_next(
    expr: &SqlExpr,
    joined_aliases: &[String],
    next_alias: &str,
    table_aliases: &[&str],
) -> Result<Option<(String, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let Some(left_owner) = join_column_owner(&left_column, table_aliases) else {
        return Ok(None);
    };
    let Some(right_owner) = join_column_owner(&right_column, table_aliases) else {
        return Ok(None);
    };
    if left_owner == next_alias && joined_aliases.iter().any(|alias| alias == right_owner) {
        return Ok(Some((
            joined_comma_join_key(&right_column, right_owner, joined_aliases),
            unqualified_join_column(&left_column, next_alias),
        )));
    }
    if right_owner == next_alias && joined_aliases.iter().any(|alias| alias == left_owner) {
        return Ok(Some((
            joined_comma_join_key(&left_column, left_owner, joined_aliases),
            unqualified_join_column(&right_column, next_alias),
        )));
    }
    Ok(None)
}

pub(super) fn comma_join_base_edge<'a>(
    expr: &SqlExpr,
    table_aliases: &'a [&'a str],
) -> Result<Option<(&'a str, String, &'a str, String)>> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return Ok(None);
    };
    let Some(left_column) = maybe_join_column_name(left, table_aliases)? else {
        return Ok(None);
    };
    let Some(right_column) = maybe_join_column_name(right, table_aliases)? else {
        return Ok(None);
    };
    let Some(left_owner) = join_column_owner(&left_column, table_aliases) else {
        return Ok(None);
    };
    let Some(right_owner) = join_column_owner(&right_column, table_aliases) else {
        return Ok(None);
    };
    if left_owner == right_owner {
        return Ok(None);
    }
    Ok(Some((
        left_owner,
        unqualified_join_column(&left_column, left_owner),
        right_owner,
        unqualified_join_column(&right_column, right_owner),
    )))
}

pub(super) fn join_column_owner<'a>(column: &str, table_aliases: &'a [&str]) -> Option<&'a str> {
    table_aliases
        .iter()
        .copied()
        .find(|alias| column.starts_with(&format!("{alias}.")))
}

pub(super) fn joined_comma_join_key(
    column: &str,
    owner: &str,
    joined_aliases: &[String],
) -> String {
    if joined_aliases.len() == 1 && joined_aliases[0] == owner {
        unqualified_join_column(column, owner)
    } else {
        column.to_string()
    }
}

pub(super) fn unqualified_join_column(column: &str, alias: &str) -> String {
    column
        .strip_prefix(&format!("{alias}."))
        .expect("qualified join column")
        .to_string()
}

pub(super) fn maybe_join_column_name(
    expr: &SqlExpr,
    table_aliases: &[&str],
) -> Result<Option<String>> {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            join_column_name(expr, table_aliases).map(Some)
        }
        _ => Ok(None),
    }
}

pub(super) fn parse_join_condition(
    join: &sqlparser::ast::Join,
    left_alias: &str,
    right_alias: &str,
) -> Result<(JoinType, Vec<String>, Vec<String>, Option<FilterExpr>)> {
    let (join_type, constraint) = match &join.join_operator {
        JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => {
            (JoinType::Inner, constraint)
        }
        JoinOperator::Semi(constraint) | JoinOperator::LeftSemi(constraint) => {
            (JoinType::Semi, constraint)
        }
        JoinOperator::Left(constraint) | JoinOperator::LeftOuter(constraint) => {
            (JoinType::Left, constraint)
        }
        JoinOperator::Right(constraint) | JoinOperator::RightOuter(constraint) => {
            (JoinType::Right, constraint)
        }
        JoinOperator::FullOuter(constraint) => (JoinType::Full, constraint),
        operator => {
            return Err(DodamError::UnsupportedSql(format!(
                "only INNER, LEFT, RIGHT, FULL, and LEFT SEMI JOIN are supported, got {operator:?}"
            )));
        }
    };
    let JoinConstraint::On(expr) = constraint else {
        return Err(DodamError::UnsupportedSql(
            "JOIN requires equality ON conditions".to_string(),
        ));
    };

    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut right_filters = Vec::new();
    collect_join_equalities(
        expr,
        left_alias,
        right_alias,
        &mut left_keys,
        &mut right_keys,
        &mut right_filters,
        join_type,
    )?;
    if left_keys.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "JOIN requires at least one equality ON condition".to_string(),
        ));
    }
    Ok((
        join_type,
        left_keys,
        right_keys,
        combine_expr_filters(right_filters),
    ))
}

pub(super) fn collect_join_equalities(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    left_keys: &mut Vec<String>,
    right_keys: &mut Vec<String>,
    right_filters: &mut Vec<Expr>,
    join_type: JoinType,
) -> Result<()> {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_join_equalities(
                left,
                left_alias,
                right_alias,
                left_keys,
                right_keys,
                right_filters,
                join_type,
            )?;
            collect_join_equalities(
                right,
                left_alias,
                right_alias,
                left_keys,
                right_keys,
                right_filters,
                join_type,
            )
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            if let (Some(left_column), Some(right_column)) = (
                unqualified_column_identifier(left),
                unqualified_column_identifier(right),
            ) {
                left_keys.push(left_column);
                right_keys.push(right_column);
                return Ok(());
            }
            let left_column = maybe_join_column_name(left, &[left_alias, right_alias])?;
            let right_column = maybe_join_column_name(right, &[left_alias, right_alias])?;
            if let (Some(left_column), Some(right_column)) = (left_column, right_column) {
                let (left_column, right_column) = if left_column
                    .starts_with(&format!("{left_alias}."))
                    && right_column.starts_with(&format!("{right_alias}."))
                {
                    (left_column, right_column)
                } else if left_column.starts_with(&format!("{right_alias}."))
                    && right_column.starts_with(&format!("{left_alias}."))
                {
                    (right_column, left_column)
                } else {
                    return Err(DodamError::UnsupportedSql(
                        "JOIN condition must compare one column from each side".to_string(),
                    ));
                };
                left_keys.push(
                    left_column
                        .strip_prefix(&format!("{left_alias}."))
                        .expect("left prefix")
                        .to_string(),
                );
                right_keys.push(
                    right_column
                        .strip_prefix(&format!("{right_alias}."))
                        .expect("right prefix")
                        .to_string(),
                );
                return Ok(());
            }
            push_join_on_residual_filter(expr, left_alias, right_alias, right_filters, join_type)
        }
        _ => push_join_on_residual_filter(expr, left_alias, right_alias, right_filters, join_type),
    }
}

pub(super) fn push_join_on_residual_filter(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    right_filters: &mut Vec<Expr>,
    join_type: JoinType,
) -> Result<()> {
    let filter = join_expr_to_filter_expr(expr, &[], &[left_alias, right_alias], false)?;
    let filter = normalize_right_join_on_filter(filter, left_alias, right_alias, join_type)?;
    right_filters.push(filter);
    Ok(())
}

pub(super) fn unqualified_column_identifier(expr: &SqlExpr) -> Option<String> {
    match expr {
        SqlExpr::Identifier(ident) => Some(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => {
            let [ident] = parts.as_slice() else {
                return None;
            };
            Some(ident.value.clone())
        }
        _ => None,
    }
}

pub(super) fn normalize_right_join_on_filter(
    expr: Expr,
    left_alias: &str,
    right_alias: &str,
    join_type: JoinType,
) -> Result<Expr> {
    if !matches!(join_type, JoinType::Inner | JoinType::Left | JoinType::Semi) {
        return Err(DodamError::UnsupportedSql(
            "JOIN ON residual filters are only supported for INNER, LEFT, and SEMI joins"
                .to_string(),
        ));
    }
    let mut columns = Vec::new();
    collect_filter_columns(&expr, &mut columns);
    if columns
        .iter()
        .any(|column| column == left_alias || column.starts_with(&format!("{left_alias}.")))
    {
        return Err(DodamError::UnsupportedSql(
            "JOIN ON residual filters may only reference the right input".to_string(),
        ));
    }
    Ok(strip_filter_prefix(expr, right_alias))
}

pub(super) fn combine_expr_filters(mut filters: Vec<Expr>) -> Option<FilterExpr> {
    let first = filters.pop()?;
    Some(FilterExpr::new(
        filters.into_iter().fold(first, |right, left| {
            Expr::And(Box::new(left), Box::new(right))
        }),
    ))
}

pub(super) fn combine_filter_options(
    left: Option<FilterExpr>,
    right: Option<FilterExpr>,
) -> Option<FilterExpr> {
    match (left, right) {
        (None, None) => None,
        (Some(filter), None) | (None, Some(filter)) => Some(filter),
        (Some(left), Some(right)) => Some(FilterExpr::new(Expr::And(
            Box::new(left.expr().clone()),
            Box::new(right.expr().clone()),
        ))),
    }
}

pub(super) fn collect_filter_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Boolean(_) => {}
        Expr::Comparison(comparison) => add_filter_column(columns, &comparison.column),
        Expr::ColumnComparison { left, right, .. } => {
            add_filter_column(columns, left);
            add_filter_column(columns, right);
        }
        Expr::InList { column, .. } | Expr::Like { column, .. } | Expr::IsNull { column, .. } => {
            add_filter_column(columns, column);
        }
        Expr::Not(expr) => collect_filter_columns(expr, columns),
        Expr::And(left, right) | Expr::Or(left, right) => {
            collect_filter_columns(left, columns);
            collect_filter_columns(right, columns);
        }
    }
}

fn add_filter_column(columns: &mut Vec<String>, column: &str) {
    if !columns.iter().any(|existing| existing == column) {
        columns.push(column.to_string());
    }
}

pub(super) fn strip_filter_prefix(expr: Expr, prefix: &str) -> Expr {
    match expr {
        Expr::Boolean(value) => Expr::Boolean(value),
        Expr::Comparison(mut comparison) => {
            comparison.column = strip_column_prefix(&comparison.column, prefix);
            Expr::Comparison(comparison)
        }
        Expr::ColumnComparison { left, op, right } => Expr::ColumnComparison {
            left: strip_column_prefix(&left, prefix),
            op,
            right: strip_column_prefix(&right, prefix),
        },
        Expr::InList {
            column,
            values,
            negated,
            has_null,
        } => Expr::InList {
            column: strip_column_prefix(&column, prefix),
            values,
            negated,
            has_null,
        },
        Expr::Like {
            column,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => Expr::Like {
            column: strip_column_prefix(&column, prefix),
            pattern,
            negated,
            escape,
            case_insensitive,
        },
        Expr::IsNull { column, negated } => Expr::IsNull {
            column: strip_column_prefix(&column, prefix),
            negated,
        },
        Expr::Not(expr) => Expr::Not(Box::new(strip_filter_prefix(*expr, prefix))),
        Expr::And(left, right) => Expr::And(
            Box::new(strip_filter_prefix(*left, prefix)),
            Box::new(strip_filter_prefix(*right, prefix)),
        ),
        Expr::Or(left, right) => Expr::Or(
            Box::new(strip_filter_prefix(*left, prefix)),
            Box::new(strip_filter_prefix(*right, prefix)),
        ),
    }
}

pub(super) fn strip_column_prefix(column: &str, prefix: &str) -> String {
    column
        .strip_prefix(&format!("{prefix}."))
        .unwrap_or(column)
        .to_string()
}

pub(super) fn common_or_comma_join_equality_keys(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<Vec<(String, String)>> {
    match expr {
        SqlExpr::Nested(expr) => {
            common_or_comma_join_equality_keys(expr, left_alias, right_alias, table_aliases)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Or,
            right,
        } => {
            let left_keys =
                common_or_comma_join_equality_keys(left, left_alias, right_alias, table_aliases)?;
            let right_keys =
                common_or_comma_join_equality_keys(right, left_alias, right_alias, table_aliases)?;
            Ok(left_keys
                .into_iter()
                .filter(|key| right_keys.iter().any(|right_key| right_key == key))
                .collect())
        }
        expr => branch_comma_join_equality_keys(expr, left_alias, right_alias, table_aliases),
    }
}

fn branch_comma_join_equality_keys(
    expr: &SqlExpr,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<Vec<(String, String)>> {
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(expr, &mut conjuncts);
    let mut keys = Vec::new();
    for conjunct in conjuncts {
        if let Some(key) =
            comma_join_equality_keys(&conjunct, left_alias, right_alias, table_aliases)?
            && !keys.iter().any(|existing| existing == &key)
        {
            keys.push(key);
        }
    }
    Ok(keys)
}
