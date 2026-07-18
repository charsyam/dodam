use super::*;

pub fn parse_sql(input: &str) -> Result<SqlQuery> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, input)
        .map_err(|error| DodamError::UnsupportedSql(error.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one statement".to_string(),
        ));
    };

    let Statement::Query(query) = statement else {
        return Err(DodamError::UnsupportedSql(
            "only SELECT queries are supported".to_string(),
        ));
    };
    parse_query(query)
}

pub(super) fn parse_query(query: &Query) -> Result<SqlQuery> {
    reject_query_features(query)?;
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(DodamError::UnsupportedSql(
            "only simple SELECT queries are supported".to_string(),
        ));
    };
    parse_select(query, select)
}

fn parse_select(query: &Query, select: &Select) -> Result<SqlQuery> {
    reject_select_features(select)?;
    if select.from.len() > 1 {
        return parse_comma_join_select(query, select);
    }
    if select
        .from
        .first()
        .is_some_and(|table| !table.joins.is_empty())
    {
        return parse_join_select(query, select);
    }
    let path = parse_from(select)?;
    let group_by = parse_group_by(select, path.alias.as_deref())?;
    let parsed_projection = parse_projection(select, &group_by, path.alias.as_deref())?;
    let distinct = parse_distinct(select)?;
    let (filter, expression_filter) = if let Some(selection) = select.selection.as_ref() {
        if predicate_requires_expression_path(selection)
            && !expr_contains_materializable_subquery(selection)
        {
            (None, Some(selection.clone()))
        } else {
            (
                Some(parse_filter(selection, &[], path.alias.as_deref(), false)?),
                None,
            )
        }
    } else {
        (None, None)
    };
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_filter(expr, &parsed_projection.aliases, None, true))
        .transpose()?;
    let order_by = parse_order_by(
        query,
        &parsed_projection.aliases,
        &parsed_projection.ordinal_targets,
        path.alias.as_deref(),
    )?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &parsed_projection.projection,
        &parsed_projection.aggregates,
        order_by.as_ref(),
    )?;

    Ok(SqlQuery {
        path: path.path,
        join: None,
        projection: parsed_projection.projection,
        filter,
        expression_filter,
        having,
        order_by,
        limit,
        offset,
        distinct,
        aggregates: parsed_projection.aggregates,
        filtered_aggregates: parsed_projection.filtered_aggregates,
        aggregate_expressions: parsed_projection.aggregate_expressions,
        expressions: parsed_projection.expressions,
        group_by,
        aliases: parsed_projection.aliases,
        qualified_wildcards: parsed_projection.qualified_wildcards,
    })
}

fn parse_comma_join_select(query: &Query, select: &Select) -> Result<SqlQuery> {
    let Some(tables) = parse_comma_join_table_refs(select)? else {
        return Err(DodamError::UnsupportedSql(
            "comma joins currently support exactly two FROM tables".to_string(),
        ));
    };
    let [left, right] = tables.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "comma joins currently support exactly two FROM tables".to_string(),
        ));
    };
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let output_aliases = vec![left_alias.as_str(), right_alias.as_str()];
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let distinct = parse_distinct(select)?;
    let (left_keys, right_keys, residual) = split_comma_join_selection(
        select.selection.as_ref(),
        &left_alias,
        &right_alias,
        &output_aliases,
    )?;
    let (filter, expression_filter) = parse_join_filter_plan(
        residual.as_ref(),
        &projection.aliases,
        &output_aliases,
        false,
    )?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        order_by.as_ref(),
    )?;

    Ok(SqlQuery {
        path: left.path.clone(),
        join: Some(SqlJoin {
            right: right.clone(),
            left_alias,
            right_alias,
            left_keys,
            right_keys,
            right_filter: None,
            join_type: JoinType::Inner,
        }),
        projection: projection.projection,
        filter,
        expression_filter,
        having,
        order_by,
        limit,
        offset,
        distinct,
        aggregates: projection.aggregates,
        filtered_aggregates: projection.filtered_aggregates,
        aggregate_expressions: projection.aggregate_expressions,
        expressions: projection.expressions,
        group_by,
        aliases: projection.aliases,
        qualified_wildcards: projection.qualified_wildcards,
    })
}

fn parse_join_select(query: &Query, select: &Select) -> Result<SqlQuery> {
    let [table] = select.from.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one FROM table".to_string(),
        ));
    };
    let [join] = table.joins.as_slice() else {
        return Err(DodamError::UnsupportedSql(
            "expected exactly one JOIN".to_string(),
        ));
    };
    let left = parse_table_factor(&table.relation)?;
    let right = parse_table_factor(&join.relation)?;
    let left_alias = table_ref_alias_or_name(&left);
    let right_alias = table_ref_alias_or_name(&right);
    let (join_type, left_keys, right_keys, right_filter) =
        parse_join_condition(join, &left_alias, &right_alias)?;
    let join_aliases = [left_alias.as_str(), right_alias.as_str()];
    let output_aliases = if join_type == JoinType::Semi {
        vec![left_alias.as_str()]
    } else {
        join_aliases.to_vec()
    };
    let group_by = parse_join_group_by(select, &output_aliases)?;
    let projection = parse_join_projection(select, &output_aliases, &group_by)?;
    let distinct = parse_distinct(select)?;
    let (filter, expression_filter) = parse_join_filter_plan(
        select.selection.as_ref(),
        &projection.aliases,
        &output_aliases,
        false,
    )?;
    let having = select
        .having
        .as_ref()
        .map(|expr| parse_join_filter(expr, &projection.aliases, &output_aliases, true))
        .transpose()?;
    let order_by = parse_join_order_by(
        query,
        &projection.aliases,
        &projection.ordinal_targets,
        &output_aliases,
    )?;
    let limit = parse_limit(query)?;
    let offset = parse_offset(query)?;
    validate_distinct(
        distinct,
        &projection.projection,
        &projection.aggregates,
        order_by.as_ref(),
    )?;

    Ok(SqlQuery {
        path: left.path,
        join: Some(SqlJoin {
            right,
            left_alias,
            right_alias,
            left_keys,
            right_keys,
            right_filter,
            join_type,
        }),
        projection: projection.projection,
        filter,
        expression_filter,
        having,
        order_by,
        limit,
        offset,
        distinct,
        aggregates: projection.aggregates,
        filtered_aggregates: projection.filtered_aggregates,
        aggregate_expressions: projection.aggregate_expressions,
        expressions: projection.expressions,
        group_by,
        aliases: projection.aliases,
        qualified_wildcards: projection.qualified_wildcards,
    })
}

pub(super) fn split_comma_join_selection(
    selection: Option<&SqlExpr>,
    left_alias: &str,
    right_alias: &str,
    table_aliases: &[&str],
) -> Result<(Vec<String>, Vec<String>, Option<SqlExpr>)> {
    let Some(selection) = selection else {
        return Err(DodamError::UnsupportedSql(
            "comma join requires an equality predicate in WHERE".to_string(),
        ));
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(selection, &mut conjuncts);
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut residuals = Vec::new();
    for conjunct in conjuncts {
        match comma_join_equality_keys(&conjunct, left_alias, right_alias, table_aliases)? {
            Some((left_key, right_key)) => {
                left_keys.push(left_key);
                right_keys.push(right_key);
            }
            None => residuals.push(conjunct),
        }
    }
    if left_keys.is_empty() {
        for (left_key, right_key) in
            common_or_comma_join_equality_keys(selection, left_alias, right_alias, table_aliases)?
        {
            left_keys.push(left_key);
            right_keys.push(right_key);
        }
    }
    if left_keys.is_empty() {
        return Err(DodamError::UnsupportedSql(
            "comma join requires at least one equality predicate between the two tables"
                .to_string(),
        ));
    }
    Ok((left_keys, right_keys, combine_sql_and_conjuncts(residuals)))
}

pub(super) fn split_subquery_residual(
    residual: Option<SqlExpr>,
) -> (Option<SqlExpr>, Option<SqlExpr>) {
    let Some(residual) = residual else {
        return (None, None);
    };
    let mut conjuncts = Vec::new();
    collect_sql_and_conjuncts(&residual, &mut conjuncts);
    let mut filter_conjuncts = Vec::new();
    let mut subquery_conjuncts = Vec::new();
    for conjunct in conjuncts {
        if expr_contains_materializable_subquery(&conjunct) {
            subquery_conjuncts.push(conjunct);
        } else {
            filter_conjuncts.push(conjunct);
        }
    }
    (
        combine_sql_and_conjuncts(filter_conjuncts),
        combine_sql_and_conjuncts(subquery_conjuncts),
    )
}
