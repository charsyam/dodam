use super::*;

pub(super) fn reject_query_features(query: &Query) -> Result<()> {
    if query.with.is_some()
        || query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Err(DodamError::UnsupportedSql(
            "WITH/FETCH/locks/settings/format/pipe clauses are not supported".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn reject_select_features(select: &Select) -> Result<()> {
    if select.select_modifiers.is_some()
        || select.top.is_some()
        || select.into.is_some()
        || select.prewhere.is_some()
        || !select.connect_by.is_empty()
        || !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
        || !select.sort_by.is_empty()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.value_table_mode.is_some()
    {
        return Err(DodamError::UnsupportedSql(
            "TOP/window/qualify/select modifiers are not supported".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn parse_distinct(select: &Select) -> Result<bool> {
    match &select.distinct {
        None | Some(Distinct::All) => Ok(false),
        Some(Distinct::Distinct) => Ok(true),
        Some(Distinct::On(_)) => Err(DodamError::UnsupportedSql(
            "DISTINCT ON is not supported".to_string(),
        )),
    }
}

pub(super) fn validate_distinct(
    distinct: bool,
    projection: &Projection,
    aggregates: &[AggregateExpr],
    order_by: Option<&SortKey>,
) -> Result<()> {
    if !distinct {
        return Ok(());
    }
    if !aggregates.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "DISTINCT with aggregate SELECT items is not supported".to_string(),
        ));
    }

    if let (Projection::Columns(columns), Some(order_by)) = (projection, order_by) {
        for sort in &order_by.expressions {
            if !columns.iter().any(|column| column == &sort.column) {
                return Err(DodamError::UnsupportedSql(format!(
                    "DISTINCT ORDER BY column {} must appear in SELECT list",
                    sort.column
                )));
            }
        }
    }

    Ok(())
}
