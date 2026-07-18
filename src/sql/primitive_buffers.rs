use super::*;

pub(super) fn primitive_topk_key(column: NullFreePrimitiveColumn<'_>, row: usize) -> Option<i128> {
    match column {
        NullFreePrimitiveColumn::I32(values) => Some(i128::from(values[row])),
        NullFreePrimitiveColumn::I64(values) => Some(i128::from(values[row])),
    }
}

pub(super) fn primitive_column_values_key(
    column: &PrimitiveColumnValues,
    row: usize,
) -> Option<i128> {
    match column {
        PrimitiveColumnValues::I32(values) => values.get(row).map(|value| (*value).into()),
        PrimitiveColumnValues::I64(values) => values.get(row).map(|value| (*value).into()),
    }
}

pub(super) fn primitive_output_len(output: &PrimitiveColumnOutput) -> usize {
    match output {
        PrimitiveColumnOutput::I32(values) => values.len(),
        PrimitiveColumnOutput::I64(values) => values.len(),
    }
}

pub(super) fn push_primitive_batch_value(
    source: &PrimitiveColumnValues,
    column_type: &DirectPrimitiveColumnType,
    row: usize,
    target: &mut PrimitiveColumnOutput,
) -> Result<()> {
    match (source, column_type, target) {
        (
            PrimitiveColumnValues::I32(values),
            DirectPrimitiveColumnType::I32,
            PrimitiveColumnOutput::I32(target),
        ) => {
            let Some(value) = values.get(row).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            target.push(value);
            Ok(())
        }
        (
            PrimitiveColumnValues::I64(values),
            DirectPrimitiveColumnType::I64,
            PrimitiveColumnOutput::I64(target),
        ) => {
            let Some(value) = values.get(row).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            target.push(value);
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive selected top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn push_direct_selected_page_value(
    source: &DirectSelectedPrimitiveColumnPageView<'_>,
    column_type: &DirectPrimitiveColumnType,
    row: usize,
    target: &mut PrimitiveColumnOutput,
) -> Result<()> {
    match (source, column_type, target) {
        (
            DirectSelectedPrimitiveColumnPageView::I32Plain { .. }
            | DirectSelectedPrimitiveColumnPageView::I32Dictionary { .. },
            DirectPrimitiveColumnType::I32,
            PrimitiveColumnOutput::I32(target),
        ) => {
            let Some(value) = source.value_i32(row) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            target.push(value);
            Ok(())
        }
        (
            DirectSelectedPrimitiveColumnPageView::I64Plain { .. },
            DirectPrimitiveColumnType::I64,
            PrimitiveColumnOutput::I64(target),
        ) => {
            let Some(value) = source.value_i64(row) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            target.push(value);
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive selected top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn overwrite_direct_selected_page_value(
    source: &DirectSelectedPrimitiveColumnPageView<'_>,
    column_type: &DirectPrimitiveColumnType,
    row: usize,
    target: &mut PrimitiveColumnOutput,
    slot: usize,
) -> Result<()> {
    match (source, column_type, target) {
        (
            DirectSelectedPrimitiveColumnPageView::I32Plain { .. }
            | DirectSelectedPrimitiveColumnPageView::I32Dictionary { .. },
            DirectPrimitiveColumnType::I32,
            PrimitiveColumnOutput::I32(target),
        ) => {
            let Some(value) = source.value_i32(row) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            let Some(target) = target.get_mut(slot) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k slot out of range".to_string(),
                ));
            };
            *target = value;
            Ok(())
        }
        (
            DirectSelectedPrimitiveColumnPageView::I64Plain { .. },
            DirectPrimitiveColumnType::I64,
            PrimitiveColumnOutput::I64(target),
        ) => {
            let Some(value) = source.value_i64(row) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            let Some(target) = target.get_mut(slot) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k slot out of range".to_string(),
                ));
            };
            *target = value;
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive selected top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn overwrite_primitive_batch_value(
    source: &PrimitiveColumnValues,
    column_type: &DirectPrimitiveColumnType,
    row: usize,
    target: &mut PrimitiveColumnOutput,
    slot: usize,
) -> Result<()> {
    match (source, column_type, target) {
        (
            PrimitiveColumnValues::I32(values),
            DirectPrimitiveColumnType::I32,
            PrimitiveColumnOutput::I32(target),
        ) => {
            let Some(value) = values.get(row).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            let Some(target) = target.get_mut(slot) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k slot out of range".to_string(),
                ));
            };
            *target = value;
            Ok(())
        }
        (
            PrimitiveColumnValues::I64(values),
            DirectPrimitiveColumnType::I64,
            PrimitiveColumnOutput::I64(target),
        ) => {
            let Some(value) = values.get(row).copied() else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k row out of range".to_string(),
                ));
            };
            let Some(target) = target.get_mut(slot) else {
                return Err(DodamError::UnsupportedSql(
                    "primitive selected top-k slot out of range".to_string(),
                ));
            };
            *target = value;
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive selected top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn push_null_free_primitive_value(
    source: &NullFreePrimitiveColumn<'_>,
    row: usize,
    target: &mut PrimitiveColumnOutput,
) -> Result<()> {
    match (source, target) {
        (NullFreePrimitiveColumn::I32(values), PrimitiveColumnOutput::I32(target)) => {
            target.push(values[row]);
            Ok(())
        }
        (NullFreePrimitiveColumn::I64(values), PrimitiveColumnOutput::I64(target)) => {
            target.push(values[row]);
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn overwrite_null_free_primitive_value(
    source: &NullFreePrimitiveColumn<'_>,
    row: usize,
    target: &mut PrimitiveColumnOutput,
    slot: usize,
) -> Result<()> {
    match (source, target) {
        (NullFreePrimitiveColumn::I32(values), PrimitiveColumnOutput::I32(target)) => {
            target[slot] = values[row];
            Ok(())
        }
        (NullFreePrimitiveColumn::I64(values), PrimitiveColumnOutput::I64(target)) => {
            target[slot] = values[row];
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn push_primitive_output_slot(
    source: &PrimitiveColumnOutput,
    source_slot: usize,
    target: &mut PrimitiveColumnOutput,
) -> Result<()> {
    match (source, target) {
        (PrimitiveColumnOutput::I32(source), PrimitiveColumnOutput::I32(target)) => {
            target.push(source[source_slot]);
            Ok(())
        }
        (PrimitiveColumnOutput::I64(source), PrimitiveColumnOutput::I64(target)) => {
            target.push(source[source_slot]);
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn overwrite_primitive_output_slot(
    source: &PrimitiveColumnOutput,
    source_slot: usize,
    target: &mut PrimitiveColumnOutput,
    target_slot: usize,
) -> Result<()> {
    match (source, target) {
        (PrimitiveColumnOutput::I32(source), PrimitiveColumnOutput::I32(target)) => {
            target[target_slot] = source[source_slot];
            Ok(())
        }
        (PrimitiveColumnOutput::I64(source), PrimitiveColumnOutput::I64(target)) => {
            target[target_slot] = source[source_slot];
            Ok(())
        }
        _ => Err(DodamError::UnsupportedSql(
            "primitive top-k column type mismatch".to_string(),
        )),
    }
}

pub(super) fn primitive_output_batch_from_columns(
    output: Vec<PrimitiveColumnOutput>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    let columns = output
        .into_iter()
        .zip(column_names.iter())
        .zip(column_types.iter())
        .map(|((values, name), column_type)| {
            let values = match values {
                PrimitiveColumnOutput::I32(values) => PrimitiveColumnValues::I32(values),
                PrimitiveColumnOutput::I64(values) => PrimitiveColumnValues::I64(values),
            };
            Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PrimitiveBatch { columns })
}

pub(super) fn primitive_column_matches_direct_type(
    column: &PrimitiveColumn,
    column_type: &DirectPrimitiveColumnType,
) -> bool {
    matches!(
        (&column.values, column_type),
        (
            PrimitiveColumnValues::I32(_),
            DirectPrimitiveColumnType::I32
        ) | (
            PrimitiveColumnValues::I64(_),
            DirectPrimitiveColumnType::I64
        )
    )
}

pub(super) enum PrimitiveFilterValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

pub(super) fn direct_ordered_primitive_batch_to_primitive_batch(
    batch: DirectOrderedPrimitiveBatch,
    names: &[String],
    column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    if batch.columns.len() != names.len() || names.len() != column_types.len() {
        return Err(DodamError::UnsupportedSql(
            "primitive batch schema mismatch".to_string(),
        ));
    }
    let columns = batch
        .columns
        .into_iter()
        .zip(names.iter())
        .zip(column_types.iter())
        .map(|((column, name), column_type)| {
            let values = match (column, column_type) {
                (
                    DirectOrderedPrimitiveColumnValues::I32(values),
                    DirectPrimitiveColumnType::I32,
                ) => PrimitiveColumnValues::I32(values),
                (
                    DirectOrderedPrimitiveColumnValues::I64(values),
                    DirectPrimitiveColumnType::I64,
                ) => PrimitiveColumnValues::I64(values),
                _ => {
                    return Err(DodamError::UnsupportedSql(
                        "primitive batch value type mismatch".to_string(),
                    ));
                }
            };
            Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PrimitiveBatch { columns })
}

pub(super) fn primitive_output_data_type(
    column_type: &DirectPrimitiveColumnType,
) -> Result<DataType> {
    match column_type {
        DirectPrimitiveColumnType::I32 => Ok(DataType::Int32),
        DirectPrimitiveColumnType::I64 => Ok(DataType::Int64),
        DirectPrimitiveColumnType::Date32 => Ok(DataType::Date32),
        DirectPrimitiveColumnType::Decimal128Int64 { precision, scale }
        | DirectPrimitiveColumnType::Decimal128Int64Raw { precision, scale } => {
            Ok(DataType::Decimal128(*precision, *scale))
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum NullFreePrimitiveColumn<'a> {
    I32(&'a [i32]),
    I64(&'a [i64]),
}

pub(super) enum PrimitiveColumnOutput {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

pub(super) fn primitive_empty_batch(
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
) -> Result<PrimitiveBatch> {
    let columns = column_names
        .iter()
        .zip(column_types.iter())
        .map(|(name, column_type)| {
            let values = match column_type {
                DirectPrimitiveColumnType::I32 | DirectPrimitiveColumnType::Date32 => {
                    PrimitiveColumnValues::I32(Vec::new())
                }
                DirectPrimitiveColumnType::I64
                | DirectPrimitiveColumnType::Decimal128Int64 { .. }
                | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => {
                    PrimitiveColumnValues::I64(Vec::new())
                }
            };
            Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PrimitiveBatch { columns })
}

pub(super) fn push_null_free_primitive_row(
    columns: &[NullFreePrimitiveColumn<'_>],
    output: &mut [PrimitiveColumnOutput],
    row: usize,
) -> Result<()> {
    for (column, output) in columns.iter().zip(output.iter_mut()) {
        match (column, output) {
            (NullFreePrimitiveColumn::I32(values), PrimitiveColumnOutput::I32(output)) => {
                output.push(values[row]);
            }
            (NullFreePrimitiveColumn::I64(values), PrimitiveColumnOutput::I64(output)) => {
                output.push(values[row]);
            }
            _ => {
                return Err(DodamError::UnsupportedSql(
                    "primitive ordered fast path column type mismatch".to_string(),
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn gather_null_free_primitive_column(
    column: &NullFreePrimitiveColumn<'_>,
    positions: &[usize],
    output: &mut PrimitiveColumnOutput,
) -> Result<()> {
    match (column, output) {
        (NullFreePrimitiveColumn::I32(values), PrimitiveColumnOutput::I32(output)) => {
            output.reserve(positions.len());
            extend_i32_desc_positions(values, positions, output);
        }
        (NullFreePrimitiveColumn::I64(values), PrimitiveColumnOutput::I64(output)) => {
            output.reserve(positions.len());
            extend_i64_desc_positions(values, positions, output);
        }
        _ => {
            return Err(DodamError::UnsupportedSql(
                "primitive ordered fast path column type mismatch".to_string(),
            ));
        }
    }
    Ok(())
}

pub(super) fn extend_i32_desc_positions(
    values: &[i32],
    positions: &[usize],
    output: &mut Vec<i32>,
) {
    let mut index = 0usize;
    while index < positions.len() {
        let end = positions[index];
        let mut len = 1usize;
        while index + len < positions.len()
            && positions[index + len - 1] == positions[index + len] + 1
        {
            len += 1;
        }
        let start = end + 1 - len;
        output.extend(values[start..=end].iter().rev().copied());
        index += len;
    }
}

pub(super) fn extend_i64_desc_positions(
    values: &[i64],
    positions: &[usize],
    output: &mut Vec<i64>,
) {
    let mut index = 0usize;
    while index < positions.len() {
        let end = positions[index];
        let mut len = 1usize;
        while index + len < positions.len()
            && positions[index + len - 1] == positions[index + len] + 1
        {
            len += 1;
        }
        let start = end + 1 - len;
        output.extend(values[start..=end].iter().rev().copied());
        index += len;
    }
}
