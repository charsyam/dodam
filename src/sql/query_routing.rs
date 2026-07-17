use super::*;

pub(super) fn sql_uses_materialized_subquery(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    Ok(parse_derived_from(select)?.is_some()
        || select.selection.as_ref().is_some_and(|expr| {
            top_level_exists_subquery(Some(expr)).is_some()
                || expr_contains_materializable_subquery(expr)
        })
        || select
            .having
            .as_ref()
            .is_some_and(expr_contains_materializable_subquery))
}

pub(super) fn sql_uses_multi_comma_join(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    Ok(parse_multi_input_join_table_refs_and_conjuncts(select)?
        .is_some_and(|(tables, _)| tables.len() > 2))
}

pub(super) fn sql_select_has_explicit_join(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    Ok(select.from.iter().any(|table| !table.joins.is_empty()))
}

pub(super) fn sql_uses_set_operation(sql: &str) -> Result<bool> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(false);
    };
    Ok(query_contains_set_operation(query.body.as_ref()))
}

pub(super) fn plan_direct_join_sink_request_relaxed(
    sql: &str,
    batch_size: usize,
) -> Result<Option<JoinParquetRequest>> {
    match plan_direct_join_sink_request(sql, batch_size) {
        Ok(request) => Ok(request),
        Err(DodamError::UnsupportedSql(message)) if sql_rule_shape_mismatch_error(&message) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
