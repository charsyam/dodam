use super::*;

pub(super) fn pushed_join_output_projection(query: &SqlQuery) -> Result<Projection> {
    let Some(join) = &query.join else {
        return Ok(Projection::All);
    };
    if query.is_aggregate() {
        return aggregate_join_output_projection(query);
    }
    if query.distinct {
        return Ok(Projection::All);
    }
    if !matches!(join.join_type, JoinType::Inner | JoinType::Semi) {
        return Ok(Projection::All);
    }
    if !query.qualified_wildcards.is_empty() {
        return Ok(Projection::All);
    }
    join_working_projection(query)
}

fn join_working_projection(query: &SqlQuery) -> Result<Projection> {
    let Some(join) = &query.join else {
        return Ok(query.projection.clone());
    };
    let mut projection = if query.is_aggregate() {
        aggregate_join_output_projection(query)?
    } else {
        query.projection.clone()
    };
    if let Some(filter) = &query.filter {
        add_projection_columns(&mut projection, filter.referenced_columns());
    }
    if let Some(order_by) = &query.order_by {
        add_projection_columns(
            &mut projection,
            order_by
                .expressions
                .iter()
                .map(|sort| sort.column.clone())
                .collect(),
        );
    }
    if let Some(expression_filter) = query.expression_filter.as_ref() {
        let aliases = [join.left_alias.as_str(), join.right_alias.as_str()];
        add_projection_columns(
            &mut projection,
            join_sql_expression_columns(expression_filter, &aliases)?,
        );
    }
    Ok(projection)
}

pub(super) fn join_input_projection_with_expression_filter(query: &SqlQuery) -> Result<Projection> {
    if query.join.is_none() {
        return Ok(query.projection.clone());
    }
    join_working_projection(query)
}

fn aggregate_join_output_projection(query: &SqlQuery) -> Result<Projection> {
    let Projection::Columns(columns) = &query.projection else {
        return Ok(Projection::All);
    };
    let Some(join) = &query.join else {
        return Ok(Projection::All);
    };
    let mut columns = columns.clone();
    if let Some(filter) = &query.filter {
        for column in filter.referenced_columns() {
            add_column_once(&mut columns, column);
        }
    }
    let aliases = [join.left_alias.as_str(), join.right_alias.as_str()];
    if let Some(expression_filter) = query.expression_filter.as_ref() {
        for column in join_sql_expression_columns(expression_filter, &aliases)? {
            add_column_once(&mut columns, column);
        }
    }
    for expression in &query.aggregate_expressions {
        for column in join_scalar_expression_columns(&expression.expr, &aliases)
            .unwrap_or_else(|_| scalar_expression_columns(&expression.expr))
        {
            add_column_once(&mut columns, column);
        }
    }
    Ok(Projection::Columns(columns))
}
