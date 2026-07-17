use super::*;

pub(super) fn parse_order_by(
    query: &Query,
    aliases: &[(String, String)],
    ordinal_targets: &[String],
    table_alias: Option<&str>,
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
            Ok(SortExpr {
                column: parse_order_expr(&order.expr, aliases, ordinal_targets, table_alias)?,
                descending: order.options.asc == Some(false),
                nulls_first: order.options.nulls_first.unwrap_or(false),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    SortKey::new(expressions).map(Some)
}

fn parse_order_expr(
    expr: &SqlExpr,
    aliases: &[(String, String)],
    ordinal_targets: &[String],
    table_alias: Option<&str>,
) -> Result<String> {
    let column = match expr {
        SqlExpr::Value(value) => resolve_order_by_ordinal(value, ordinal_targets)?,
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            sql_column_name(expr, table_alias)?
        }
        SqlExpr::Function(function) => parse_aggregate(function, table_alias)?.to_string(),
        _ => {
            return Err(DodamError::UnsupportedSql(format!(
                "expected ORDER BY column or aggregate expression, got {expr}"
            )));
        }
    };
    Ok(resolve_alias(&column, aliases))
}

pub(super) fn resolve_alias(name: &str, aliases: &[(String, String)]) -> String {
    alias_target(name, aliases)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

pub(super) fn resolve_order_by_ordinal(
    value: &sqlparser::ast::ValueWithSpan,
    ordinal_targets: &[String],
) -> Result<String> {
    let Value::Number(number, _) = &value.value else {
        return Err(DodamError::UnsupportedSql(format!(
            "expected ORDER BY column, ordinal, or aggregate expression, got {}",
            value.value
        )));
    };
    let ordinal = number
        .parse::<usize>()
        .map_err(|_| DodamError::UnsupportedSql(format!("invalid ORDER BY ordinal: {number}")))?;
    if ordinal == 0 {
        return Err(DodamError::UnsupportedSql(
            "ORDER BY ordinal must be greater than zero".to_string(),
        ));
    }
    ordinal_targets.get(ordinal - 1).cloned().ok_or_else(|| {
        DodamError::UnsupportedSql(format!("ORDER BY position {ordinal} is out of range"))
    })
}

pub(super) fn alias_target<'a>(name: &str, aliases: &'a [(String, String)]) -> Option<&'a String> {
    aliases
        .iter()
        .find(|(alias, _)| alias == name)
        .map(|(_, target)| target)
}

pub(super) fn parse_limit(query: &Query) -> Result<Option<usize>> {
    let Some(limit_clause) = &query.limit_clause else {
        return Ok(None);
    };
    match limit_clause {
        LimitClause::LimitOffset {
            limit: Some(limit),
            offset: _,
            limit_by,
        } if limit_by.is_empty() => parse_usize_literal(limit).map(Some),
        LimitClause::LimitOffset {
            limit: None,
            offset: _,
            limit_by,
        } if limit_by.is_empty() => Ok(None),
        _ => Err(DodamError::UnsupportedSql(
            "only LIMIT <integer> with optional OFFSET is supported".to_string(),
        )),
    }
}

pub(super) fn parse_offset(query: &Query) -> Result<usize> {
    let Some(limit_clause) = &query.limit_clause else {
        return Ok(0);
    };
    match limit_clause {
        LimitClause::LimitOffset {
            offset: Some(offset),
            limit_by,
            ..
        } if limit_by.is_empty() => parse_usize_literal(&offset.value),
        LimitClause::LimitOffset {
            offset: None,
            limit_by,
            ..
        } if limit_by.is_empty() => Ok(0),
        _ => Err(DodamError::UnsupportedSql(
            "only LIMIT/OFFSET without LIMIT BY is supported".to_string(),
        )),
    }
}

pub(super) fn scan_limit_with_offset(limit: Option<usize>, offset: usize) -> Result<Option<usize>> {
    limit
        .map(|limit| {
            limit
                .checked_add(offset)
                .ok_or_else(|| DodamError::UnsupportedSql("LIMIT + OFFSET overflow".to_string()))
        })
        .transpose()
}
