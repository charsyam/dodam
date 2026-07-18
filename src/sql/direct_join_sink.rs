use super::*;

pub(super) fn plan_direct_join_sink_request(
    sql: &str,
    batch_size: usize,
) -> Result<Option<JoinParquetRequest>> {
    if sql_uses_set_operation(sql)? {
        return Ok(None);
    }
    if sql_uses_materialized_subquery(sql)? || sql_uses_multi_comma_join(sql)? {
        return Ok(None);
    }
    if !sql_select_has_explicit_join(sql)? {
        return Ok(None);
    }
    let query = parse_sql(sql)?;
    direct_join_sink_request(query, batch_size)
}

fn direct_join_sink_request(
    query: SqlQuery,
    batch_size: usize,
) -> Result<Option<JoinParquetRequest>> {
    let Some(join) = query.join.clone() else {
        return Ok(None);
    };
    if !query_allows_direct_join_sink(&query) {
        return Ok(None);
    }

    let join_plan = plan_join_inputs(
        &query.projection,
        query.filter.as_ref(),
        None,
        &join.left_alias,
        &join.left_keys,
        &join.right_alias,
        &join.right_keys,
    );
    let output_projection = pushed_join_output_projection(&query)?;
    if matches!(output_projection, Projection::All) && !matches!(query.projection, Projection::All)
    {
        return Ok(None);
    }

    Ok(Some(JoinParquetRequest {
        left_path: query.path,
        right_path: join.right.path,
        batch_size,
        left_keys: join.left_keys,
        right_keys: join.right_keys,
        left_prefix: join.left_alias,
        right_prefix: join.right_alias,
        left_projection: join_plan.left_projection,
        right_projection: join_plan.right_projection,
        left_filter: join_plan.left_filter,
        right_filter: combine_filter_options(join_plan.right_filter, join.right_filter.clone()),
        output_projection,
        join_memory_limit_bytes: default_join_memory_limit_bytes(),
        join_algorithm: JoinAlgorithm::Auto,
        join_type: join.join_type,
    }))
}

fn query_allows_direct_join_sink(query: &SqlQuery) -> bool {
    query.join.is_some()
        && !query.is_aggregate()
        && query.having.is_none()
        && !query.distinct
        && query.filter.as_ref().is_none_or(|filter| {
            query.join.as_ref().is_some_and(|join| {
                filter_fully_pushable_to_join_inputs(
                    filter.expr(),
                    &join.left_alias,
                    &join.right_alias,
                )
            })
        })
        && query.expression_filter.is_none()
        && query.order_by.is_none()
        && join_aliases_are_implicit_output_names(&query.aliases)
}

fn filter_fully_pushable_to_join_inputs(expr: &Expr, left_alias: &str, right_alias: &str) -> bool {
    match expr {
        Expr::Boolean(value) => matches!(value, Some(true)),
        Expr::Comparison(_)
        | Expr::ColumnComparison { .. }
        | Expr::InList { .. }
        | Expr::Like { .. }
        | Expr::IsNull { .. } => {
            expr_references_only_join_side(expr, left_alias)
                || expr_references_only_join_side(expr, right_alias)
        }
        Expr::Not(expr) => {
            expr_references_only_join_side(expr, left_alias)
                || expr_references_only_join_side(expr, right_alias)
        }
        Expr::And(left, right) => {
            filter_fully_pushable_to_join_inputs(left, left_alias, right_alias)
                && filter_fully_pushable_to_join_inputs(right, left_alias, right_alias)
        }
        Expr::Or(_, _) => {
            expr_references_only_join_side(expr, left_alias)
                || expr_references_only_join_side(expr, right_alias)
        }
    }
}

fn expr_references_only_join_side(expr: &Expr, alias: &str) -> bool {
    let mut columns = Vec::new();
    collect_filter_columns(expr, &mut columns);
    !columns.is_empty()
        && columns.iter().all(|column| {
            column
                .strip_prefix(alias)
                .is_some_and(|rest| rest.starts_with('.'))
        })
}

fn join_aliases_are_implicit_output_names(aliases: &[(String, String)]) -> bool {
    aliases.iter().all(|(source, target)| {
        source
            .rsplit_once('.')
            .is_some_and(|(_, column)| column == target)
    })
}
