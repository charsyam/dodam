use arrow::array::{Array, ArrayRef, Date32Array, Int64Array, StringArray};

use crate::error::Result;

pub fn try_for_each_i64_date32_str<Visit>(
    int_values: &ArrayRef,
    date_values: &ArrayRef,
    string_values: &StringArray,
    mut visit: Visit,
) -> Result<bool>
where
    Visit: FnMut(i64, i32, &str) -> Result<()>,
{
    let (Some(int_values), Some(date_values)) = (
        int_values.as_any().downcast_ref::<Int64Array>(),
        date_values.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    for row in 0..int_values.len() {
        if int_values.is_null(row) || date_values.is_null(row) || string_values.is_null(row) {
            continue;
        }
        visit(
            int_values.value(row),
            date_values.value(row),
            string_values.value(row),
        )?;
    }
    Ok(true)
}

pub fn try_for_each_i64_i64_date32<Visit>(
    left_values: &ArrayRef,
    right_values: &ArrayRef,
    date_values: &ArrayRef,
    mut visit: Visit,
) -> Result<bool>
where
    Visit: FnMut(i64, i64, i32) -> Result<()>,
{
    let (Some(left_values), Some(right_values), Some(date_values)) = (
        left_values.as_any().downcast_ref::<Int64Array>(),
        right_values.as_any().downcast_ref::<Int64Array>(),
        date_values.as_any().downcast_ref::<Date32Array>(),
    ) else {
        return Ok(false);
    };
    for row in 0..left_values.len() {
        if left_values.is_null(row) || right_values.is_null(row) || date_values.is_null(row) {
            continue;
        }
        visit(
            left_values.value(row),
            right_values.value(row),
            date_values.value(row),
        )?;
    }
    Ok(true)
}
