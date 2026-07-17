use super::*;

pub(super) fn add_column_once(columns: &mut Vec<String>, column: String) {
    if !columns.iter().any(|existing| existing == &column) {
        columns.push(column);
    }
}

pub(super) fn projection_expressions_are_plain_columns(
    expressions: &[ProjectionExpression],
) -> bool {
    expressions
        .iter()
        .all(|expression| matches!(expression.expr, ScalarSqlExpression::Column(_)))
}

pub(super) fn projection_requires_expression_path(expressions: &[ProjectionExpression]) -> bool {
    expressions
        .iter()
        .any(|expression| !matches!(expression.expr, ScalarSqlExpression::Column(_)))
}

pub(super) fn add_projection_columns(projection: &mut Projection, columns: Vec<String>) {
    let Projection::Columns(existing) = projection else {
        return;
    };
    for column in columns {
        add_column_once(existing, column);
    }
}
