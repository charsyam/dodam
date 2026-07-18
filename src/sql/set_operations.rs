use super::*;

pub(super) fn append_unique_literal_values(
    output: &mut Vec<LiteralValue>,
    values: Vec<LiteralValue>,
) {
    for value in values {
        if !output.iter().any(|existing| existing == &value) {
            output.push(value);
        }
    }
}

pub(super) fn literal_values_to_unique_i64(
    values: Vec<LiteralValue>,
    column: &str,
) -> Result<Vec<i64>> {
    let mut output = Vec::new();
    for value in values {
        let value = value.as_i64(column)?;
        if !output.iter().any(|existing| *existing == value) {
            output.push(value);
        }
    }
    Ok(output)
}

pub(super) fn intersect_literal_values(
    left_values: Vec<LiteralValue>,
    right_values: &[LiteralValue],
) -> Vec<LiteralValue> {
    let mut output = Vec::new();
    for value in left_values {
        if right_values.iter().any(|right| right == &value)
            && !output.iter().any(|existing| existing == &value)
        {
            output.push(value);
        }
    }
    output
}

pub(super) fn except_literal_values(
    left_values: Vec<LiteralValue>,
    right_values: &[LiteralValue],
) -> Vec<LiteralValue> {
    let mut output = Vec::new();
    for value in left_values {
        if !right_values.iter().any(|right| right == &value)
            && !output.iter().any(|existing| existing == &value)
        {
            output.push(value);
        }
    }
    output
}

pub(super) fn append_disjoint_literal_values(
    output: &mut Vec<LiteralValue>,
    values: Vec<LiteralValue>,
) -> Option<()> {
    for value in values {
        if output.iter().any(|existing| existing == &value) {
            return None;
        }
        output.push(value);
    }
    Some(())
}

pub(super) fn union_quantifier_is_distinct(set_quantifier: SetQuantifier) -> bool {
    matches!(
        set_quantifier,
        SetQuantifier::None | SetQuantifier::Distinct
    )
}

pub(super) fn union_child_topk_for_quantifier<'a>(
    set_quantifier: SetQuantifier,
    child_topk: Option<(&'a SortKey, usize)>,
) -> Option<(&'a SortKey, usize)> {
    if set_quantifier == SetQuantifier::All {
        child_topk
    } else {
        None
    }
}

pub(super) fn union_all_operand_sql_with_child_topk(
    sql: &str,
    child_topk: Option<(&SortKey, usize)>,
) -> String {
    let Some((order_by, limit)) = child_topk else {
        return sql.to_string();
    };
    format!("{sql} ORDER BY {} LIMIT {limit}", sort_key_to_sql(order_by))
}

fn sort_key_to_sql(order_by: &SortKey) -> String {
    order_by
        .expressions
        .iter()
        .map(|sort| {
            let mut sql = sort.column.clone();
            if sort.descending {
                sql.push_str(" DESC");
            } else {
                sql.push_str(" ASC");
            }
            if sort.nulls_first {
                sql.push_str(" NULLS FIRST");
            }
            sql
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn apply_distinct_row_set_operation(
    left_batches: Vec<RecordBatch>,
    right_batches: Vec<RecordBatch>,
    op: SetOperator,
) -> Result<Vec<RecordBatch>> {
    let Some(left_schema) = set_operation_schema(&left_batches).cloned() else {
        return Ok(Vec::new());
    };
    validate_union_all_batches(&left_schema, &left_batches)?;
    validate_union_all_batches(&left_schema, &right_batches)?;

    let left_batch = concat_or_empty_set_batches(&left_schema, &left_batches)?;
    if left_batch.num_rows() == 0 {
        return Ok(vec![left_batch]);
    }

    let right_batch = concat_or_empty_set_batches(&left_schema, &right_batches)?;
    if right_batch.num_rows() == 0 {
        return match op {
            SetOperator::Except => apply_output_distinct(vec![left_batch], true),
            SetOperator::Intersect => Ok(vec![RecordBatch::new_empty(left_schema)]),
            _ => Err(DodamError::UnsupportedSql(format!(
                "{op} is not supported by DISTINCT row set operation"
            ))),
        };
    }

    let converter = row_set_converter(left_schema.as_ref())?;
    let right_rows = converter.convert_columns(right_batch.columns())?;
    let right_set = right_rows
        .iter()
        .map(|row| row.owned())
        .collect::<HashSet<OwnedRow>>();

    let left_rows = converter.convert_columns(left_batch.columns())?;
    let mut seen_left = HashSet::<OwnedRow>::new();
    let mut indices = Vec::new();
    for (index, row) in left_rows.iter().enumerate() {
        let owned = row.owned();
        let matches = right_set.contains(&owned);
        let keep = match op {
            SetOperator::Intersect => matches,
            SetOperator::Except => !matches,
            _ => false,
        };
        if keep && seen_left.insert(owned) {
            let index = u32::try_from(index).map_err(|_| {
                DodamError::UnsupportedSql(
                    "set operation currently supports up to u32::MAX rows".to_string(),
                )
            })?;
            indices.push(index);
        }
    }

    if indices.is_empty() {
        return Ok(vec![RecordBatch::new_empty(left_schema)]);
    }
    let indices = UInt32Array::from(indices);
    Ok(vec![take_record_batch(&left_batch, &indices)?])
}

pub(super) fn apply_all_row_set_operation(
    left_batches: Vec<RecordBatch>,
    right_batches: Vec<RecordBatch>,
    op: SetOperator,
) -> Result<Vec<RecordBatch>> {
    let Some(left_schema) = set_operation_schema(&left_batches).cloned() else {
        return Ok(Vec::new());
    };
    validate_union_all_batches(&left_schema, &left_batches)?;
    validate_union_all_batches(&left_schema, &right_batches)?;

    let left_batch = concat_or_empty_set_batches(&left_schema, &left_batches)?;
    if left_batch.num_rows() == 0 {
        return Ok(vec![left_batch]);
    }

    let right_batch = concat_or_empty_set_batches(&left_schema, &right_batches)?;
    if right_batch.num_rows() == 0 {
        return match op {
            SetOperator::Except => Ok(vec![left_batch]),
            SetOperator::Intersect => Ok(vec![RecordBatch::new_empty(left_schema)]),
            _ => Err(DodamError::UnsupportedSql(format!(
                "{op} is not supported by ALL row set operation"
            ))),
        };
    }

    let converter = row_set_converter(left_schema.as_ref())?;
    let right_rows = converter.convert_columns(right_batch.columns())?;
    let mut right_counts = HashMap::<OwnedRow, usize>::new();
    for row in right_rows.iter() {
        *right_counts.entry(row.owned()).or_insert(0) += 1;
    }

    let left_rows = converter.convert_columns(left_batch.columns())?;
    let mut indices = Vec::new();
    for (index, row) in left_rows.iter().enumerate() {
        let owned = row.owned();
        let keep = match op {
            SetOperator::Intersect => {
                if let Some(count) = right_counts.get_mut(&owned) {
                    if *count > 0 {
                        *count -= 1;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            SetOperator::Except => {
                if let Some(count) = right_counts.get_mut(&owned) {
                    if *count > 0 {
                        *count -= 1;
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            _ => false,
        };
        if keep {
            let index = u32::try_from(index).map_err(|_| {
                DodamError::UnsupportedSql(
                    "set operation currently supports up to u32::MAX rows".to_string(),
                )
            })?;
            indices.push(index);
        }
    }

    if indices.is_empty() {
        return Ok(vec![RecordBatch::new_empty(left_schema)]);
    }
    let indices = UInt32Array::from(indices);
    Ok(vec![take_record_batch(&left_batch, &indices)?])
}

fn set_operation_schema(batches: &[RecordBatch]) -> Option<&Arc<Schema>> {
    batches
        .iter()
        .find(|batch| batch.num_columns() > 0 || batch.num_rows() > 0)
        .map(RecordBatch::schema_ref)
        .or_else(|| batches.first().map(RecordBatch::schema_ref))
}

fn concat_or_empty_set_batches(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
) -> Result<RecordBatch> {
    let non_empty = batches
        .iter()
        .filter(|batch| batch.num_rows() > 0)
        .collect::<Vec<_>>();
    if non_empty.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    Ok(concat_batches(schema, non_empty)?)
}

fn row_set_converter(schema: &Schema) -> Result<RowConverter> {
    let sort_fields = schema
        .fields()
        .iter()
        .map(|field| SortField::new(field.data_type().clone()))
        .collect::<Vec<_>>();
    Ok(RowConverter::new(sort_fields)?)
}

pub(super) fn append_union_all_batches(
    output: &mut Vec<RecordBatch>,
    mut batches: Vec<RecordBatch>,
) -> Result<()> {
    let Some(output_schema) = output
        .iter()
        .find(|batch| batch.num_columns() > 0 || batch.num_rows() > 0)
        .map(RecordBatch::schema)
        .or_else(|| batches.first().map(RecordBatch::schema))
    else {
        output.append(&mut batches);
        return Ok(());
    };
    validate_union_all_batches(&output_schema, output)?;
    validate_union_all_batches(&output_schema, &batches)?;
    output.extend(
        batches
            .into_iter()
            .map(|batch| align_union_all_batch_schema(batch, output_schema.clone()))
            .collect::<Result<Vec<_>>>()?,
    );
    Ok(())
}

pub(super) fn validate_union_all_batches(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
) -> Result<()> {
    for batch in batches {
        if batch.num_columns() != schema.fields().len() {
            return Err(DodamError::UnsupportedSql(format!(
                "UNION ALL column count mismatch: expected {}, got {}",
                schema.fields().len(),
                batch.num_columns()
            )));
        }
        for (index, (expected, actual)) in schema
            .fields()
            .iter()
            .zip(batch.schema().fields())
            .enumerate()
        {
            if expected.data_type() != actual.data_type() {
                return Err(DodamError::UnsupportedSql(format!(
                    "UNION ALL column {} type mismatch: expected {}, got {}",
                    index + 1,
                    expected.data_type(),
                    actual.data_type()
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn align_union_all_batch_schema(
    batch: RecordBatch,
    schema: Arc<Schema>,
) -> Result<RecordBatch> {
    if batch.schema().fields() == schema.fields() {
        return Ok(batch);
    }
    Ok(RecordBatch::try_new(schema, batch.columns().to_vec())?)
}
