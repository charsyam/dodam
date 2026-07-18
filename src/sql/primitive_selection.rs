use super::*;

pub(super) fn primitive_ordered_selected_batch(
    view: BatchView<'_>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
    filter_index: usize,
    filter_values: &PrimitiveFilterValues,
) -> Result<Option<PrimitiveBatch>> {
    if let Some(batch) = primitive_ordered_selected_batch_null_free_fast(
        view,
        column_names,
        column_types,
        filter_index,
        filter_values,
    )? {
        return Ok(Some(batch));
    }
    if !row_at_time_fallback_enabled() {
        return Ok(None);
    }
    let mut selected = Vec::new();
    match filter_values {
        PrimitiveFilterValues::I32(values) => {
            if !matches!(column_types[filter_index], DirectPrimitiveColumnType::I32) {
                return Ok(None);
            }
            let Some(column) = view.i32_vector(filter_index) else {
                return Ok(None);
            };
            for row in (0..view.num_rows()).rev() {
                if !column.is_null(row) && values.iter().any(|value| *value == column.value(row)) {
                    selected.push(row as u32);
                }
            }
        }
        PrimitiveFilterValues::I64(values) => {
            if !matches!(column_types[filter_index], DirectPrimitiveColumnType::I64) {
                return Ok(None);
            }
            let Some(column) = view.i64_vector(filter_index) else {
                return Ok(None);
            };
            for row in (0..view.num_rows()).rev() {
                if !column.is_null(row) && values.iter().any(|value| *value == column.value(row)) {
                    selected.push(row as u32);
                }
            }
        }
    }
    if selected.is_empty() {
        return primitive_empty_batch(column_names, column_types).map(Some);
    }
    let mut columns = Vec::with_capacity(column_types.len());
    for (index, column_type) in column_types.iter().enumerate() {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(column) = view.i32_vector(index) else {
                    return Ok(None);
                };
                let mut output = Vec::with_capacity(selected.len());
                for row in &selected {
                    let row = *row as usize;
                    if column.is_null(row) {
                        return Ok(None);
                    }
                    output.push(column.value(row));
                }
                columns.push(PrimitiveColumn {
                    name: column_names[index].clone(),
                    data_type: primitive_output_data_type(column_type)?,
                    nullable: false,
                    values: PrimitiveColumnValues::I32(output),
                });
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(column) = view.i64_vector(index) else {
                    return Ok(None);
                };
                let mut output = Vec::with_capacity(selected.len());
                for row in &selected {
                    let row = *row as usize;
                    if column.is_null(row) {
                        return Ok(None);
                    }
                    output.push(column.value(row));
                }
                columns.push(PrimitiveColumn {
                    name: column_names[index].clone(),
                    data_type: primitive_output_data_type(column_type)?,
                    nullable: false,
                    values: PrimitiveColumnValues::I64(output),
                });
            }
            DirectPrimitiveColumnType::Date32
            | DirectPrimitiveColumnType::Decimal128Int64 { .. }
            | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(None),
        }
    }
    Ok(Some(PrimitiveBatch { columns }))
}

pub(super) fn primitive_topk_filter_positions_into(
    column: NullFreePrimitiveColumn<'_>,
    filter_values: &PrimitiveFilterValues,
    selected: &mut Vec<usize>,
) {
    selected.clear();
    match (column, filter_values) {
        (NullFreePrimitiveColumn::I32(values), PrimitiveFilterValues::I32(filter_values)) => {
            reserve_selected_positions(selected, values.len());
            primitive_topk_filter_i32_positions(values, filter_values, selected);
        }
        (NullFreePrimitiveColumn::I64(values), PrimitiveFilterValues::I64(filter_values)) => {
            reserve_selected_positions(selected, values.len());
            primitive_topk_filter_i64_positions(values, filter_values, selected);
        }
        _ => {}
    }
}

pub(super) fn primitive_topk_filter_positions_with_min_key_into(
    filter_column: NullFreePrimitiveColumn<'_>,
    filter_values: &PrimitiveFilterValues,
    sort_column: NullFreePrimitiveColumn<'_>,
    min_key: i128,
    selected: &mut Vec<usize>,
) {
    selected.clear();
    match (filter_column, filter_values, sort_column) {
        (
            NullFreePrimitiveColumn::I32(filter_values_slice),
            PrimitiveFilterValues::I32(filter_values),
            NullFreePrimitiveColumn::I32(sort_values),
        ) => {
            let min_key = match i32::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i32::MIN) => {
                    primitive_topk_filter_i32_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i32_positions_with_i32_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        (
            NullFreePrimitiveColumn::I32(filter_values_slice),
            PrimitiveFilterValues::I32(filter_values),
            NullFreePrimitiveColumn::I64(sort_values),
        ) => {
            let min_key = match i64::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i64::MIN) => {
                    primitive_topk_filter_i32_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i32_positions_with_i64_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        (
            NullFreePrimitiveColumn::I64(filter_values_slice),
            PrimitiveFilterValues::I64(filter_values),
            NullFreePrimitiveColumn::I64(sort_values),
        ) => {
            let min_key = match i64::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i64::MIN) => {
                    primitive_topk_filter_i64_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i64_positions_with_i64_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        (
            NullFreePrimitiveColumn::I64(filter_values_slice),
            PrimitiveFilterValues::I64(filter_values),
            NullFreePrimitiveColumn::I32(sort_values),
        ) => {
            let min_key = match i32::try_from(min_key) {
                Ok(min_key) => min_key,
                Err(_) if min_key < i128::from(i32::MIN) => {
                    primitive_topk_filter_i64_positions(
                        filter_values_slice,
                        filter_values,
                        selected,
                    );
                    return;
                }
                Err(_) => {
                    selected.clear();
                    return;
                }
            };
            reserve_selected_positions(selected, filter_values_slice.len());
            primitive_topk_filter_i64_positions_with_i32_min_key(
                filter_values_slice,
                filter_values,
                sort_values,
                min_key,
                selected,
            );
        }
        _ => primitive_topk_filter_positions_into(filter_column, filter_values, selected),
    }
}

pub(super) fn primitive_topk_filter_i32_positions(
    values: &[i32],
    filter_values: &[i32],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i32_eq_positions(values, *a, selected),
        [a, b] => push_i32_eq2_positions(values, *a, *b, selected),
        [a, b, c] => push_i32_eq3_positions(values, *a, *b, *c, selected),
        [a, b, c, d] => push_i32_eq4_positions(values, *a, *b, *c, *d, selected),
        _ => {
            for (row, value) in values.iter().copied().enumerate() {
                if filter_values.contains(&value) {
                    selected.push(row);
                }
            }
        }
    }
}

pub(super) fn primitive_topk_filter_i64_positions(
    values: &[i64],
    filter_values: &[i64],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i64_eq_positions(values, *a, selected),
        [a, b] => push_i64_eq2_positions(values, *a, *b, selected),
        [a, b, c] => push_i64_eq3_positions(values, *a, *b, *c, selected),
        [a, b, c, d] => push_i64_eq4_positions(values, *a, *b, *c, *d, selected),
        _ => {
            for (row, value) in values.iter().copied().enumerate() {
                if filter_values.contains(&value) {
                    selected.push(row);
                }
            }
        }
    }
}

fn primitive_topk_filter_i32_positions_with_i32_min_key(
    values: &[i32],
    filter_values: &[i32],
    sort_values: &[i32],
    min_key: i32,
    selected: &mut Vec<usize>,
) {
    push_primitive_position_pairs_unrolled_with_offset(
        values,
        sort_values,
        0,
        selected,
        |value, key| key >= min_key && small_i32_filter_values_contains(filter_values, value),
    );
}

fn primitive_topk_filter_i32_positions_with_i64_min_key(
    values: &[i32],
    filter_values: &[i32],
    sort_values: &[i64],
    min_key: i64,
    selected: &mut Vec<usize>,
) {
    if let [a, b] = filter_values {
        if primitive_topk_block_max_skip_enabled() {
            push_i32_eq2_positions_with_i64_min_key_blocked(
                values,
                *a,
                *b,
                sort_values,
                min_key,
                selected,
            );
        } else {
            push_primitive_position_pairs_unrolled_with_offset(
                values,
                sort_values,
                0,
                selected,
                |value, key| key >= min_key && (value == *a || value == *b),
            );
        }
    } else {
        push_primitive_position_pairs_unrolled_with_offset(
            values,
            sort_values,
            0,
            selected,
            |value, key| key >= min_key && small_i32_filter_values_contains(filter_values, value),
        );
    }
}

fn primitive_topk_filter_i64_positions_with_i64_min_key(
    values: &[i64],
    filter_values: &[i64],
    sort_values: &[i64],
    min_key: i64,
    selected: &mut Vec<usize>,
) {
    push_primitive_position_pairs_unrolled_with_offset(
        values,
        sort_values,
        0,
        selected,
        |value, key| key >= min_key && small_i64_filter_values_contains(filter_values, value),
    );
}

fn primitive_topk_filter_i64_positions_with_i32_min_key(
    values: &[i64],
    filter_values: &[i64],
    sort_values: &[i32],
    min_key: i32,
    selected: &mut Vec<usize>,
) {
    push_primitive_position_pairs_unrolled_with_offset(
        values,
        sort_values,
        0,
        selected,
        |value, key| key >= min_key && small_i64_filter_values_contains(filter_values, value),
    );
}

fn push_i32_eq2_positions_with_i64_min_key_blocked(
    values: &[i32],
    a: i32,
    b: i32,
    keys: &[i64],
    min_key: i64,
    selected: &mut Vec<usize>,
) {
    let len = values.len().min(keys.len());
    let mut row = 0usize;
    const BLOCK: usize = 64;
    while row + BLOCK <= len {
        let block_keys = &keys[row..row + BLOCK];
        let mut max_key = block_keys[0];
        for key in block_keys.iter().copied().skip(1) {
            if key > max_key {
                max_key = key;
            }
        }
        if max_key >= min_key {
            push_primitive_position_pairs_unrolled_with_offset(
                &values[row..row + BLOCK],
                block_keys,
                row,
                selected,
                |value, key| key >= min_key && (value == a || value == b),
            );
        }
        row += BLOCK;
    }
    while row < len {
        let value = values[row];
        if keys[row] >= min_key && (value == a || value == b) {
            selected.push(row);
        }
        row += 1;
    }
}

fn small_i32_filter_values_contains(values: &[i32], value: i32) -> bool {
    match values {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => values.contains(&value),
    }
}

fn small_i64_filter_values_contains(values: &[i64], value: i64) -> bool {
    match values {
        [] => false,
        [a] => value == *a,
        [a, b] => value == *a || value == *b,
        [a, b, c] => value == *a || value == *b || value == *c,
        [a, b, c, d] => value == *a || value == *b || value == *c || value == *d,
        _ => values.contains(&value),
    }
}

fn push_i32_eq_positions(values: &[i32], a: i32, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a);
}

fn push_i32_eq2_positions(values: &[i32], a: i32, b: i32, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a || value == b);
}

fn push_i32_eq3_positions(values: &[i32], a: i32, b: i32, c: i32, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c
    });
}

fn push_i32_eq4_positions(
    values: &[i32],
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    selected: &mut Vec<usize>,
) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c || value == d
    });
}

fn push_i64_eq_positions(values: &[i64], a: i64, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a);
}

fn push_i64_eq2_positions(values: &[i64], a: i64, b: i64, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| value == a || value == b);
}

fn push_i64_eq3_positions(values: &[i64], a: i64, b: i64, c: i64, selected: &mut Vec<usize>) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c
    });
}

fn push_i64_eq4_positions(
    values: &[i64],
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    selected: &mut Vec<usize>,
) {
    push_primitive_positions_unrolled(values, selected, |value| {
        value == a || value == b || value == c || value == d
    });
}

fn push_primitive_positions_unrolled<T, F>(values: &[T], selected: &mut Vec<usize>, mut matches: F)
where
    T: Copy,
    F: FnMut(T) -> bool,
{
    let mut row = 0usize;
    let chunks = values.len() / 8 * 8;
    while row < chunks {
        let v0 = values[row];
        let v1 = values[row + 1];
        let v2 = values[row + 2];
        let v3 = values[row + 3];
        let v4 = values[row + 4];
        let v5 = values[row + 5];
        let v6 = values[row + 6];
        let v7 = values[row + 7];
        if matches(v0) {
            selected.push(row);
        }
        if matches(v1) {
            selected.push(row + 1);
        }
        if matches(v2) {
            selected.push(row + 2);
        }
        if matches(v3) {
            selected.push(row + 3);
        }
        if matches(v4) {
            selected.push(row + 4);
        }
        if matches(v5) {
            selected.push(row + 5);
        }
        if matches(v6) {
            selected.push(row + 6);
        }
        if matches(v7) {
            selected.push(row + 7);
        }
        row += 8;
    }
    while row < values.len() {
        if matches(values[row]) {
            selected.push(row);
        }
        row += 1;
    }
}

fn push_primitive_position_pairs_unrolled_with_offset<T, U, F>(
    values: &[T],
    keys: &[U],
    offset: usize,
    selected: &mut Vec<usize>,
    mut matches: F,
) where
    T: Copy,
    U: Copy,
    F: FnMut(T, U) -> bool,
{
    let len = values.len().min(keys.len());
    let mut row = 0usize;
    let chunks = len / 8 * 8;
    while row < chunks {
        let v0 = values[row];
        let v1 = values[row + 1];
        let v2 = values[row + 2];
        let v3 = values[row + 3];
        let v4 = values[row + 4];
        let v5 = values[row + 5];
        let v6 = values[row + 6];
        let v7 = values[row + 7];
        let k0 = keys[row];
        let k1 = keys[row + 1];
        let k2 = keys[row + 2];
        let k3 = keys[row + 3];
        let k4 = keys[row + 4];
        let k5 = keys[row + 5];
        let k6 = keys[row + 6];
        let k7 = keys[row + 7];
        if matches(v0, k0) {
            selected.push(offset + row);
        }
        if matches(v1, k1) {
            selected.push(offset + row + 1);
        }
        if matches(v2, k2) {
            selected.push(offset + row + 2);
        }
        if matches(v3, k3) {
            selected.push(offset + row + 3);
        }
        if matches(v4, k4) {
            selected.push(offset + row + 4);
        }
        if matches(v5, k5) {
            selected.push(offset + row + 5);
        }
        if matches(v6, k6) {
            selected.push(offset + row + 6);
        }
        if matches(v7, k7) {
            selected.push(offset + row + 7);
        }
        row += 8;
    }
    while row < len {
        if matches(values[row], keys[row]) {
            selected.push(offset + row);
        }
        row += 1;
    }
}

fn primitive_filter_i32_positions_desc(
    values: &[i32],
    filter_values: &[i32],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i32_eq_positions_desc(values, *a, selected),
        [a, b] => push_i32_eq2_positions_desc(values, *a, *b, selected),
        [a, b, c] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c
        }),
        [a, b, c, d] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c || value == *d
        }),
        _ => {
            let mut row = values.len();
            while row > 0 {
                row -= 1;
                if filter_values.contains(&values[row]) {
                    selected.push(row);
                }
            }
        }
    }
}

fn primitive_filter_i64_positions_desc(
    values: &[i64],
    filter_values: &[i64],
    selected: &mut Vec<usize>,
) {
    match filter_values {
        [] => {}
        [a] => push_i64_eq_positions_desc(values, *a, selected),
        [a, b] => push_i64_eq2_positions_desc(values, *a, *b, selected),
        [a, b, c] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c
        }),
        [a, b, c, d] => push_primitive_positions_desc_unrolled(values, selected, |value| {
            value == *a || value == *b || value == *c || value == *d
        }),
        _ => {
            let mut row = values.len();
            while row > 0 {
                row -= 1;
                if filter_values.contains(&values[row]) {
                    selected.push(row);
                }
            }
        }
    }
}

fn push_i32_eq_positions_desc(values: &[i32], a: i32, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        if values[row + 7] == a {
            selected.push(row + 7);
        }
        if values[row + 6] == a {
            selected.push(row + 6);
        }
        if values[row + 5] == a {
            selected.push(row + 5);
        }
        if values[row + 4] == a {
            selected.push(row + 4);
        }
        if values[row + 3] == a {
            selected.push(row + 3);
        }
        if values[row + 2] == a {
            selected.push(row + 2);
        }
        if values[row + 1] == a {
            selected.push(row + 1);
        }
        if values[row] == a {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        if values[row] == a {
            selected.push(row);
        }
    }
}

fn push_i32_eq2_positions_desc(values: &[i32], a: i32, b: i32, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        let v7 = values[row + 7];
        let v6 = values[row + 6];
        let v5 = values[row + 5];
        let v4 = values[row + 4];
        let v3 = values[row + 3];
        let v2 = values[row + 2];
        let v1 = values[row + 1];
        let v0 = values[row];
        if v7 == a || v7 == b {
            selected.push(row + 7);
        }
        if v6 == a || v6 == b {
            selected.push(row + 6);
        }
        if v5 == a || v5 == b {
            selected.push(row + 5);
        }
        if v4 == a || v4 == b {
            selected.push(row + 4);
        }
        if v3 == a || v3 == b {
            selected.push(row + 3);
        }
        if v2 == a || v2 == b {
            selected.push(row + 2);
        }
        if v1 == a || v1 == b {
            selected.push(row + 1);
        }
        if v0 == a || v0 == b {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        let value = values[row];
        if value == a || value == b {
            selected.push(row);
        }
    }
}

fn push_i64_eq_positions_desc(values: &[i64], a: i64, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        if values[row + 7] == a {
            selected.push(row + 7);
        }
        if values[row + 6] == a {
            selected.push(row + 6);
        }
        if values[row + 5] == a {
            selected.push(row + 5);
        }
        if values[row + 4] == a {
            selected.push(row + 4);
        }
        if values[row + 3] == a {
            selected.push(row + 3);
        }
        if values[row + 2] == a {
            selected.push(row + 2);
        }
        if values[row + 1] == a {
            selected.push(row + 1);
        }
        if values[row] == a {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        if values[row] == a {
            selected.push(row);
        }
    }
}

fn push_i64_eq2_positions_desc(values: &[i64], a: i64, b: i64, selected: &mut Vec<usize>) {
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        let v7 = values[row + 7];
        let v6 = values[row + 6];
        let v5 = values[row + 5];
        let v4 = values[row + 4];
        let v3 = values[row + 3];
        let v2 = values[row + 2];
        let v1 = values[row + 1];
        let v0 = values[row];
        if v7 == a || v7 == b {
            selected.push(row + 7);
        }
        if v6 == a || v6 == b {
            selected.push(row + 6);
        }
        if v5 == a || v5 == b {
            selected.push(row + 5);
        }
        if v4 == a || v4 == b {
            selected.push(row + 4);
        }
        if v3 == a || v3 == b {
            selected.push(row + 3);
        }
        if v2 == a || v2 == b {
            selected.push(row + 2);
        }
        if v1 == a || v1 == b {
            selected.push(row + 1);
        }
        if v0 == a || v0 == b {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        let value = values[row];
        if value == a || value == b {
            selected.push(row);
        }
    }
}

fn push_primitive_positions_desc_unrolled<T, F>(
    values: &[T],
    selected: &mut Vec<usize>,
    mut matches: F,
) where
    T: Copy,
    F: FnMut(T) -> bool,
{
    let mut row = values.len();
    while row >= 8 {
        row -= 8;
        if matches(values[row + 7]) {
            selected.push(row + 7);
        }
        if matches(values[row + 6]) {
            selected.push(row + 6);
        }
        if matches(values[row + 5]) {
            selected.push(row + 5);
        }
        if matches(values[row + 4]) {
            selected.push(row + 4);
        }
        if matches(values[row + 3]) {
            selected.push(row + 3);
        }
        if matches(values[row + 2]) {
            selected.push(row + 2);
        }
        if matches(values[row + 1]) {
            selected.push(row + 1);
        }
        if matches(values[row]) {
            selected.push(row);
        }
    }
    while row > 0 {
        row -= 1;
        if matches(values[row]) {
            selected.push(row);
        }
    }
}

pub(super) fn reserve_selected_positions(selected: &mut Vec<usize>, row_count: usize) {
    let target = row_count / 4 + 1;
    if selected.capacity() < target {
        selected.reserve(target - selected.capacity());
    }
}

pub(super) fn primitive_topk_sequence_base(location: DirectPrimitiveBatchLocation) -> Option<u64> {
    let row_group = u64::try_from(location.row_group).ok()?;
    let row_offset = u64::try_from(location.row_offset).ok()?;
    Some((row_group << 32) | row_offset)
}

fn primitive_ordered_selected_batch_null_free_fast(
    view: BatchView<'_>,
    column_names: &[String],
    column_types: &[DirectPrimitiveColumnType],
    filter_index: usize,
    filter_values: &PrimitiveFilterValues,
) -> Result<Option<PrimitiveBatch>> {
    if filter_index >= column_types.len() {
        return Ok(None);
    }
    let row_count = view.num_rows();
    let mut columns = Vec::with_capacity(column_types.len());
    for (index, column_type) in column_types.iter().enumerate() {
        match column_type {
            DirectPrimitiveColumnType::I32 => {
                let Some(column) = view.i32_vector(index) else {
                    return Ok(None);
                };
                let Some(values) = column.values_if_null_free() else {
                    return Ok(None);
                };
                columns.push(NullFreePrimitiveColumn::I32(values));
            }
            DirectPrimitiveColumnType::I64 => {
                let Some(column) = view.i64_vector(index) else {
                    return Ok(None);
                };
                let Some(values) = column.values_if_null_free() else {
                    return Ok(None);
                };
                columns.push(NullFreePrimitiveColumn::I64(values));
            }
            DirectPrimitiveColumnType::Date32
            | DirectPrimitiveColumnType::Decimal128Int64 { .. }
            | DirectPrimitiveColumnType::Decimal128Int64Raw { .. } => return Ok(None),
        }
    }
    if columns.iter().any(|column| match column {
        NullFreePrimitiveColumn::I32(values) => values.len() != row_count,
        NullFreePrimitiveColumn::I64(values) => values.len() != row_count,
    }) {
        return Ok(None);
    }
    let mut output = column_types
        .iter()
        .map(|column_type| match column_type {
            DirectPrimitiveColumnType::I32 => {
                PrimitiveColumnOutput::I32(Vec::with_capacity(row_count / 4 + 1))
            }
            DirectPrimitiveColumnType::I64 => {
                PrimitiveColumnOutput::I64(Vec::with_capacity(row_count / 4 + 1))
            }
            _ => unreachable!("checked primitive ordered column type"),
        })
        .collect::<Vec<_>>();
    let mut selected_positions = Vec::with_capacity(row_count / 4 + 1);
    match (filter_values, columns[filter_index]) {
        (PrimitiveFilterValues::I32(values), NullFreePrimitiveColumn::I32(filter_column)) => {
            primitive_filter_i32_positions_desc(filter_column, values, &mut selected_positions);
        }
        (PrimitiveFilterValues::I64(values), NullFreePrimitiveColumn::I64(filter_column)) => {
            primitive_filter_i64_positions_desc(filter_column, values, &mut selected_positions);
        }
        _ => return Ok(None),
    }
    if selected_positions.is_empty() {
        return primitive_empty_batch(column_names, column_types).map(Some);
    }
    if selected_positions.len().saturating_mul(2) >= row_count {
        for (column, output) in columns.iter().zip(output.iter_mut()) {
            gather_null_free_primitive_column(column, &selected_positions, output)?;
        }
    } else {
        for &row in &selected_positions {
            push_null_free_primitive_row(&columns, &mut output, row)?;
        }
    }
    let selected_rows = output
        .first()
        .map(|column| match column {
            PrimitiveColumnOutput::I32(values) => values.len(),
            PrimitiveColumnOutput::I64(values) => values.len(),
        })
        .unwrap_or(0);
    if selected_rows == 0 {
        return primitive_empty_batch(column_names, column_types).map(Some);
    }
    let columns = output
        .into_iter()
        .zip(column_names.iter())
        .zip(column_types.iter())
        .map(|column| match column {
            ((PrimitiveColumnOutput::I32(values), name), column_type) => Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values: PrimitiveColumnValues::I32(values),
            }),
            ((PrimitiveColumnOutput::I64(values), name), column_type) => Ok(PrimitiveColumn {
                name: name.clone(),
                data_type: primitive_output_data_type(column_type)?,
                nullable: false,
                values: PrimitiveColumnValues::I64(values),
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(PrimitiveBatch { columns }))
}

pub(super) fn row_at_time_fallback_enabled() -> bool {
    std::env::var("DODAM_ENABLE_ROW_AT_TIME_FALLBACK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(super) fn ordered_sink_profile_enabled() -> bool {
    std::env::var("DODAM_PROFILE_ORDERED_SINK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}
