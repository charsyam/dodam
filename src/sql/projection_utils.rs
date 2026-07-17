use super::*;

pub(super) fn add_column_once(columns: &mut Vec<String>, column: String) {
    if !columns.iter().any(|existing| existing == &column) {
        columns.push(column);
    }
}

pub(super) fn add_projection_column_once(projection: &mut Projection, column: String) {
    if let Projection::Columns(columns) = projection
        && !columns.iter().any(|existing| existing == &column)
    {
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

pub(super) fn apply_output_projection(
    batches: Vec<RecordBatch>,
    projection: &Projection,
) -> Result<Vec<RecordBatch>> {
    let Projection::Columns(columns) = projection else {
        return Ok(batches);
    };

    batches
        .into_iter()
        .map(|batch| {
            if projection_matches_batch_schema(&batch, columns) {
                return Ok(batch);
            }
            let indices = columns
                .iter()
                .map(|column| output_batch_column_index(&batch, column))
                .collect::<Result<Vec<_>>>()?;
            Ok(batch.project(&indices)?)
        })
        .collect()
}

pub(super) fn projection_matches_batch_schema(batch: &RecordBatch, columns: &[String]) -> bool {
    batch.num_columns() == columns.len()
        && batch
            .schema()
            .fields()
            .iter()
            .zip(columns)
            .all(|(field, column)| field.name() == column)
}

pub(super) fn apply_qualified_wildcard_projection(
    batches: Vec<RecordBatch>,
    qualified_wildcards: &[String],
    projection: &Projection,
) -> Result<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(batches);
    }
    batches
        .into_iter()
        .map(|batch| {
            let mut columns = Vec::new();
            for qualifier in qualified_wildcards {
                let prefix = format!("{qualifier}.");
                for field in batch.schema().fields() {
                    if field.name().starts_with(&prefix) {
                        add_column_once(&mut columns, field.name().clone());
                    }
                }
            }
            if let Projection::Columns(projected) = projection {
                for column in projected {
                    add_column_once(&mut columns, column.clone());
                }
            }
            if columns.is_empty() {
                return Err(DodamError::UnsupportedSql(
                    "qualified wildcard did not match any output columns".to_string(),
                ));
            }
            let indices = columns
                .iter()
                .map(|column| output_batch_column_index(&batch, column))
                .collect::<Result<Vec<_>>>()?;
            Ok(batch.project(&indices)?)
        })
        .collect()
}

pub(super) fn output_batch_column_index(batch: &RecordBatch, column: &str) -> Result<usize> {
    if let Some(bound) = resolve_batch_column(batch, column)? {
        return batch_column_index(batch, &bound.physical_name);
    }
    Err(DodamError::UnknownColumn(column.to_string()))
}
