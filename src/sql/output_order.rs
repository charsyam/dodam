use super::*;

pub(super) fn apply_output_order_limit(
    batches: Vec<RecordBatch>,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<RecordBatch>> {
    let Some(order_by) = order_by else {
        return Ok(limit_batches(batches, limit, offset));
    };
    if batches.is_empty() {
        return Ok(batches);
    }
    if let Some(limit) = limit
        && offset == 0
        && let Some(batches) = try_streaming_primitive_desc_topk_batches(&batches, order_by, limit)?
    {
        return Ok(batches);
    }
    if let Some(limit) = limit
        && offset == 0
        && let Some(batches) = try_streaming_primitive_topk_batches(&batches, order_by, limit)?
    {
        return Ok(batches);
    }
    if topk_batch_prune_enabled()
        && let Some(limit) = limit
        && offset == 0
        && batches.len() > 1
        && batches.iter().any(|batch| batch.num_rows() > limit)
    {
        let candidates = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .map(|batch| sort_output_batch(batch, order_by, Some(limit)))
            .collect::<Result<Vec<_>>>()?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let schema = candidates[0].schema();
        let batch = if candidates.len() == 1 {
            candidates[0].clone()
        } else {
            concat_batches(&schema, candidates.iter())?
        };
        return Ok(vec![sort_output_batch(&batch, order_by, Some(limit))?]);
    }

    let schema = batches[0].schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        concat_batches(&schema, batches.iter())?
    };
    if output_batch_satisfies_order(&batch, order_by)? {
        return Ok(limit_batches(vec![batch], limit, offset));
    }
    let sorted_limit = limit.and_then(|limit| limit.checked_add(offset));
    let sorted = sort_output_batch(&batch, order_by, sorted_limit)?;
    Ok(limit_batches(vec![sorted], limit, offset))
}

type StreamingTopKSelected = (StreamingTopKKey, u64, usize, u32);
type DescTopKHeapItem = (Reverse<i64>, u64, usize, u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StreamingTopKKey {
    I32I64I64(i32, i64, i64),
    NullableI32I64(bool, i32, i64),
}

fn try_streaming_primitive_desc_topk_batches(
    batches: &[RecordBatch],
    order_by: &SortKey,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if limit == 0 || batches.len() < 2 {
        return Ok(None);
    }
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(None);
    };
    if !sort.descending || sort.nulls_first {
        return Ok(None);
    }
    let Some(sort_index) = batches[0]
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == &sort.column)
    else {
        return Ok(None);
    };
    if batches.iter().any(|batch| {
        batch
            .schema()
            .fields()
            .get(sort_index)
            .is_none_or(|field| field.data_type() != &DataType::Int64)
    }) {
        return Ok(None);
    }

    let mut heap = BinaryHeap::<DescTopKHeapItem>::with_capacity(limit.saturating_add(1));
    let mut sequence = 0_u64;
    for (batch_index, batch) in batches.iter().enumerate() {
        if batch.num_rows() == 0 {
            continue;
        }
        let values = batch
            .column(sort_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("checked Int64 sort column");
        if values.null_count() != 0 {
            return Ok(None);
        }
        if heap.len() >= limit
            && let Some(worst) = heap.peek()
            && let Some(max_value) = int64_batch_max(values)
            && max_value <= (worst.0).0
        {
            sequence = sequence.wrapping_add(batch.num_rows() as u64);
            continue;
        }
        for row in 0..batch.num_rows() {
            let item = (
                Reverse(values.value(row)),
                sequence,
                batch_index,
                row as u32,
            );
            if heap.len() < limit {
                heap.push(item);
            } else if heap.peek().is_some_and(|worst| item < *worst) {
                heap.pop();
                heap.push(item);
            }
            sequence = sequence.wrapping_add(1);
        }
    }
    if heap.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut selected = heap
        .into_iter()
        .map(|(Reverse(value), sequence, batch_index, row)| (value, sequence, batch_index, row))
        .collect::<Vec<_>>();
    selected.sort_unstable_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    let selected = selected
        .into_iter()
        .map(|(_, sequence, batch_index, row)| (0_i128, sequence, batch_index, row))
        .collect::<Vec<_>>();
    Ok(Some(vec![materialize_topk_selected_rows(
        batches, &selected,
    )?]))
}

fn int64_batch_max(values: &Int64Array) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    let raw = values.values();
    let mut max_value = raw[0];
    for value in &raw[1..] {
        max_value = max_value.max(*value);
    }
    Some(max_value)
}

fn try_streaming_primitive_topk_batches(
    batches: &[RecordBatch],
    order_by: &SortKey,
    limit: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    if !streaming_primitive_topk_sort_enabled()
        || limit == 0
        || batches.len() < 2
        || order_by.expressions.iter().any(|sort| sort.descending)
    {
        return Ok(None);
    }
    let Some(shape) = streaming_topk_sort_shape(batches, order_by)? else {
        return Ok(None);
    };
    let mut heap = BinaryHeap::<StreamingTopKSelected>::with_capacity(limit.saturating_add(1));
    let mut sequence = 0_u64;
    for (batch_index, batch) in batches.iter().enumerate() {
        if batch.num_rows() == 0 {
            continue;
        }
        match shape {
            StreamingTopKSortShape::I32I64I64(first, second, third) => {
                let first = batch
                    .column(first)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("streaming top-k i32 sort key");
                let second = batch
                    .column(second)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("streaming top-k i64 sort key");
                let third = batch
                    .column(third)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("streaming top-k i64 sort key");
                if heap.len() >= limit
                    && let Some(worst) = heap.peek()
                    && let Some(min_key) =
                        streaming_topk_batch_min_i32_i64_i64(first, second, third)
                    && (min_key, 0_u64, batch_index, 0_u32) >= *worst
                {
                    sequence = sequence.wrapping_add(batch.num_rows() as u64);
                    continue;
                }
                for row in 0..batch.num_rows() {
                    let item = (
                        StreamingTopKKey::I32I64I64(
                            first.value(row),
                            second.value(row),
                            third.value(row),
                        ),
                        sequence,
                        batch_index,
                        row as u32,
                    );
                    streaming_topk_push(limit, &mut heap, item);
                    sequence = sequence.wrapping_add(1);
                }
            }
            StreamingTopKSortShape::NullableI32I64(first, second) => {
                let first = batch
                    .column(first)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("streaming top-k i32 sort key");
                let second = batch
                    .column(second)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("streaming top-k i64 sort key");
                if heap.len() >= limit
                    && let Some(worst) = heap.peek()
                    && let Some(min_key) = streaming_topk_batch_min_nullable_i32_i64(first, second)
                    && (min_key, 0_u64, batch_index, 0_u32) >= *worst
                {
                    sequence = sequence.wrapping_add(batch.num_rows() as u64);
                    continue;
                }
                for row in 0..batch.num_rows() {
                    let present = !first.is_null(row);
                    let value = if present { first.value(row) } else { 0 };
                    let item = (
                        StreamingTopKKey::NullableI32I64(present, value, second.value(row)),
                        sequence,
                        batch_index,
                        row as u32,
                    );
                    streaming_topk_push(limit, &mut heap, item);
                    sequence = sequence.wrapping_add(1);
                }
            }
        }
    }
    if heap.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut selected = heap.into_vec();
    selected.sort_unstable();
    let materialize_selected = selected
        .into_iter()
        .map(|(_, sequence, batch_index, row)| (0_i128, sequence, batch_index, row))
        .collect::<Vec<_>>();
    Ok(Some(vec![materialize_topk_selected_rows(
        batches,
        &materialize_selected,
    )?]))
}

fn streaming_topk_batch_min_i32_i64_i64(
    first: &Int32Array,
    second: &Int64Array,
    third: &Int64Array,
) -> Option<StreamingTopKKey> {
    let mut min_key = None;
    for row in 0..first.len() {
        let key =
            StreamingTopKKey::I32I64I64(first.value(row), second.value(row), third.value(row));
        min_key = Some(min_key.map_or(key, |min: StreamingTopKKey| min.min(key)));
    }
    min_key
}

fn streaming_topk_batch_min_nullable_i32_i64(
    first: &Int32Array,
    second: &Int64Array,
) -> Option<StreamingTopKKey> {
    let mut min_key = None;
    for row in 0..first.len() {
        let present = !first.is_null(row);
        let value = if present { first.value(row) } else { 0 };
        let key = StreamingTopKKey::NullableI32I64(present, value, second.value(row));
        min_key = Some(min_key.map_or(key, |min: StreamingTopKKey| min.min(key)));
    }
    min_key
}

fn streaming_topk_push(
    limit: usize,
    heap: &mut BinaryHeap<StreamingTopKSelected>,
    item: StreamingTopKSelected,
) {
    if heap.len() < limit {
        heap.push(item);
        return;
    }
    if let Some(mut worst) = heap.peek_mut()
        && item < *worst
    {
        *worst = item;
    }
}

#[derive(Clone, Copy)]
enum StreamingTopKSortShape {
    I32I64I64(usize, usize, usize),
    NullableI32I64(usize, usize),
}

fn streaming_topk_sort_shape(
    batches: &[RecordBatch],
    order_by: &SortKey,
) -> Result<Option<StreamingTopKSortShape>> {
    let Some(first_batch) = batches.iter().find(|batch| batch.num_rows() > 0) else {
        return Ok(None);
    };
    let expressions = order_by.expressions.as_slice();
    if let [first, second, third] = expressions
        && !first.nulls_first
        && !second.nulls_first
        && !third.nulls_first
    {
        let first_index = output_batch_column_index(first_batch, &first.column)?;
        let second_index = output_batch_column_index(first_batch, &second.column)?;
        let third_index = output_batch_column_index(first_batch, &third.column)?;
        if streaming_topk_column_is_non_null_i32(batches, first_index)
            && streaming_topk_column_is_non_null_i64(batches, second_index)
            && streaming_topk_column_is_non_null_i64(batches, third_index)
        {
            return Ok(Some(StreamingTopKSortShape::I32I64I64(
                first_index,
                second_index,
                third_index,
            )));
        }
    }
    if let [first, second] = expressions
        && first.nulls_first
        && !second.nulls_first
    {
        let first_index = output_batch_column_index(first_batch, &first.column)?;
        let second_index = output_batch_column_index(first_batch, &second.column)?;
        if streaming_topk_column_is_i32(batches, first_index)
            && streaming_topk_column_is_non_null_i64(batches, second_index)
        {
            return Ok(Some(StreamingTopKSortShape::NullableI32I64(
                first_index,
                second_index,
            )));
        }
    }
    Ok(None)
}

fn streaming_topk_column_is_i32(batches: &[RecordBatch], index: usize) -> bool {
    batches.iter().all(|batch| {
        batch
            .column(index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .is_some()
    })
}

fn streaming_topk_column_is_non_null_i32(batches: &[RecordBatch], index: usize) -> bool {
    batches.iter().all(|batch| {
        batch
            .column(index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .is_some_and(|array| array.null_count() == 0)
    })
}

fn streaming_topk_column_is_non_null_i64(batches: &[RecordBatch], index: usize) -> bool {
    batches.iter().all(|batch| {
        batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .is_some_and(|array| array.null_count() == 0)
    })
}

fn streaming_primitive_topk_sort_enabled() -> bool {
    std::env::var("DODAM_ENABLE_STREAMING_PRIMITIVE_TOPK_SORT")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn output_batch_satisfies_order(batch: &RecordBatch, order_by: &SortKey) -> Result<bool> {
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(false);
    };
    if sort.descending || sort.nulls_first {
        return Ok(false);
    }
    let index = output_batch_column_index(batch, &sort.column)?;
    let column = batch.column(index);
    match column.data_type() {
        DataType::Int32 => {
            let values = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            for row in 1..values.len() {
                if values.is_null(row - 1) || values.is_null(row) {
                    return Ok(false);
                }
                if values.value(row - 1) > values.value(row) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        DataType::Date32 => {
            let values = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            for row in 1..values.len() {
                if values.is_null(row - 1) || values.is_null(row) {
                    return Ok(false);
                }
                if values.value(row - 1) > values.value(row) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        DataType::Int64 => {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            for row in 1..values.len() {
                if values.is_null(row - 1) || values.is_null(row) {
                    return Ok(false);
                }
                if values.value(row - 1) > values.value(row) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        DataType::Utf8 => {
            let values = column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 data type");
            for row in 1..values.len() {
                if values.is_null(row - 1) || values.is_null(row) {
                    return Ok(false);
                }
                if values.value(row - 1) > values.value(row) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(super) fn output_batches_satisfy_order(
    batches: &[RecordBatch],
    order_by: &SortKey,
) -> Result<bool> {
    if batches.is_empty() {
        return Ok(true);
    }
    for batch in batches {
        if !output_batch_satisfies_order(batch, order_by)? {
            return Ok(false);
        }
    }
    let [sort] = order_by.expressions.as_slice() else {
        return Ok(false);
    };
    if sort.descending || sort.nulls_first {
        return Ok(false);
    }
    let mut previous = None::<ScalarOrderValue>;
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let index = output_batch_column_index(batch, &sort.column)?;
        let column = batch.column(index);
        let first = scalar_order_value(column, 0)?;
        let last = scalar_order_value(column, batch.num_rows() - 1)?;
        if let Some(previous) = previous
            && previous > first
        {
            return Ok(false);
        }
        previous = Some(last);
    }
    Ok(true)
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ScalarOrderValue {
    Int32(i32),
    Date32(i32),
    Int64(i64),
    Utf8(String),
}

fn scalar_order_value(column: &ArrayRef, row: usize) -> Result<ScalarOrderValue> {
    match column.data_type() {
        DataType::Int32 => {
            let values = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            if values.is_null(row) {
                return Err(DodamError::UnsupportedSql(
                    "ordered batch boundary contains NULL".to_string(),
                ));
            }
            Ok(ScalarOrderValue::Int32(values.value(row)))
        }
        DataType::Date32 => {
            let values = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 data type");
            if values.is_null(row) {
                return Err(DodamError::UnsupportedSql(
                    "ordered batch boundary contains NULL".to_string(),
                ));
            }
            Ok(ScalarOrderValue::Date32(values.value(row)))
        }
        DataType::Int64 => {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            if values.is_null(row) {
                return Err(DodamError::UnsupportedSql(
                    "ordered batch boundary contains NULL".to_string(),
                ));
            }
            Ok(ScalarOrderValue::Int64(values.value(row)))
        }
        DataType::Utf8 => {
            let values = column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 data type");
            if values.is_null(row) {
                return Err(DodamError::UnsupportedSql(
                    "ordered batch boundary contains NULL".to_string(),
                ));
            }
            Ok(ScalarOrderValue::Utf8(values.value(row).to_string()))
        }
        _ => Err(DodamError::UnsupportedSql(
            "ordered batch boundary type is unsupported".to_string(),
        )),
    }
}

pub(super) fn apply_aggregate_output_order_limit(
    batches: Vec<RecordBatch>,
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
    metrics: &AggregateMetrics,
    group_by: &[String],
) -> Result<Vec<RecordBatch>> {
    if aggregate_output_already_ordered(metrics, group_by, order_by) {
        return Ok(limit_batches(batches, limit, offset));
    }
    apply_output_order_limit(batches, order_by, limit, offset)
}

fn aggregate_output_already_ordered(
    metrics: &AggregateMetrics,
    group_by: &[String],
    order_by: Option<&SortKey>,
) -> bool {
    let Some(order_by) = order_by else {
        return false;
    };
    if order_by.expressions.is_empty() || order_by.expressions.len() > group_by.len() {
        return false;
    }
    for (index, expression) in order_by.expressions.iter().enumerate() {
        if expression.descending || expression.column != group_by[index] {
            return false;
        }
        if !expression.nulls_first
            && metrics
                .groups
                .iter()
                .any(|group| group.keys.get(index).is_some_and(group_value_is_null))
        {
            return false;
        }
    }
    true
}

fn group_value_is_null(value: &GroupValue) -> bool {
    match value {
        GroupValue::Int64(value) => value.is_none(),
        GroupValue::UInt64(value) => value.is_none(),
        GroupValue::Decimal128(value, _, _) => value.is_none(),
        GroupValue::Date32(value) => value.is_none(),
        GroupValue::Date64(value) => value.is_none(),
        GroupValue::Utf8(value) => value.is_none(),
    }
}

pub(super) fn apply_output_expression_projection_order_limit(
    mut batches: Vec<RecordBatch>,
    expressions: &[ProjectionExpression],
    order_by: Option<&SortKey>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<RecordBatch>> {
    if sort_key_available_in_batches(&batches, order_by) {
        batches = apply_output_order_limit(batches, order_by, limit, offset)?;
        apply_output_expression_projection(batches, expressions)
    } else {
        batches = apply_output_expression_projection(batches, expressions)?;
        apply_output_order_limit(batches, order_by, limit, offset)
    }
}

fn sort_key_available_in_batches(batches: &[RecordBatch], order_by: Option<&SortKey>) -> bool {
    let Some(order_by) = order_by else {
        return false;
    };
    let Some(batch) = batches.first() else {
        return false;
    };
    order_by
        .expressions
        .iter()
        .all(|sort| output_batch_column_index(batch, &sort.column).is_ok())
}

fn topk_batch_prune_enabled() -> bool {
    std::env::var("DODAM_DISABLE_TOPK_BATCH_PRUNE")
        .map(|value| value != "1" && !value.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn sort_output_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    if let Some(batch) = sort_nullable_i32_then_ordered_i64_batch(batch, order_by, limit)? {
        return Ok(batch);
    }
    if let Some(batch) = sort_fixed_primitive_output_batch(batch, order_by, limit)? {
        return Ok(batch);
    }
    if let Some(batch) = sort_primitive_output_batch(batch, order_by, limit)? {
        return Ok(batch);
    }
    let sort_columns = order_by
        .expressions
        .iter()
        .map(|sort| {
            let column_index = batch
                .schema()
                .fields()
                .iter()
                .position(|field| field.name() == &sort.column)
                .ok_or_else(|| DodamError::UnknownColumn(sort.column.clone()))?;
            Ok(SortColumn {
                values: batch.column(column_index).clone(),
                options: Some(SortOptions {
                    descending: sort.descending,
                    nulls_first: sort.nulls_first,
                }),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let indices = lexsort_to_indices(&sort_columns, limit)?;
    if let Some(batch) = take_primitive_record_batch_by_indices(batch, &indices)? {
        return Ok(batch);
    }
    Ok(take_record_batch(batch, &indices)?)
}

fn sort_nullable_i32_then_ordered_i64_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: Option<usize>,
) -> Result<Option<RecordBatch>> {
    let Some(limit) = limit else {
        return Ok(None);
    };
    if limit == 0 || batch.num_rows() == 0 || limit >= batch.num_rows() {
        return Ok(None);
    }
    let [first_sort, second_sort] = order_by.expressions.as_slice() else {
        return Ok(None);
    };
    if first_sort.descending
        || second_sort.descending
        || !first_sort.nulls_first
        || second_sort.nulls_first
    {
        return Ok(None);
    }
    let first_index = output_batch_column_index(batch, &first_sort.column)?;
    let second_index = output_batch_column_index(batch, &second_sort.column)?;
    let first = batch.column(first_index);
    let second = batch.column(second_index);
    let Some(first) = first.as_any().downcast_ref::<Int32Array>() else {
        return Ok(None);
    };
    if !primitive_integer_array_is_ascending(second) {
        return Ok(None);
    }
    let Some((min, slot_count)) = dense_i32_array_range(first) else {
        return Ok(None);
    };
    let mut null_rows = Vec::new();
    let mut buckets = (0..slot_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for row in 0..first.len() {
        let row = u32::try_from(row).map_err(|_| {
            DodamError::UnsupportedSql("nullable i32/i64 ordered sort row overflow".to_string())
        })?;
        if first.is_null(row as usize) {
            null_rows.push(row);
        } else {
            let slot = (first.value(row as usize) - min) as usize;
            buckets[slot].push(row);
        }
    }
    let mut indices = Vec::with_capacity(limit);
    for row in null_rows {
        indices.push(row);
        if indices.len() == limit {
            return take_primitive_record_batch_by_raw_indices(batch, &indices).map(Some);
        }
    }
    for bucket in buckets {
        for row in bucket {
            indices.push(row);
            if indices.len() == limit {
                return take_primitive_record_batch_by_raw_indices(batch, &indices).map(Some);
            }
        }
    }
    take_primitive_record_batch_by_raw_indices(batch, &indices).map(Some)
}

fn primitive_integer_array_is_ascending(values: &ArrayRef) -> bool {
    match values.data_type() {
        DataType::Int32 => {
            let values = values
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type");
            if values.null_count() != 0 {
                return false;
            }
            for row in 1..values.len() {
                if values.value(row - 1) > values.value(row) {
                    return false;
                }
            }
            true
        }
        DataType::Int64 => {
            let values = values
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type");
            if values.null_count() != 0 {
                return false;
            }
            for row in 1..values.len() {
                if values.value(row - 1) > values.value(row) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn dense_i32_array_range(values: &Int32Array) -> Option<(i32, usize)> {
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut has_value = false;
    for row in 0..values.len() {
        if values.is_null(row) {
            continue;
        }
        let value = values.value(row);
        min = min.min(value);
        max = max.max(value);
        has_value = true;
    }
    if !has_value {
        return Some((0, 0));
    }
    let slot_count = usize::try_from(i64::from(max) - i64::from(min) + 1).ok()?;
    (slot_count <= values.len().saturating_mul(4) && slot_count <= 1_000_000)
        .then_some((min, slot_count))
}

fn sort_fixed_primitive_output_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: Option<usize>,
) -> Result<Option<RecordBatch>> {
    if !fixed_primitive_partial_sort_enabled() {
        return Ok(None);
    }
    let Some(limit) = limit else {
        return Ok(None);
    };
    if batch.num_rows() == 0 || limit == 0 || limit >= batch.num_rows() {
        return Ok(None);
    }
    if order_by.expressions.iter().any(|sort| sort.descending) {
        return Ok(None);
    }
    if let Some(batch) = sort_i32_i64_i64_output_batch(batch, order_by, limit)? {
        return Ok(Some(batch));
    }
    sort_nullable_i32_i64_output_batch(batch, order_by, limit)
}

fn sort_i32_i64_i64_output_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: usize,
) -> Result<Option<RecordBatch>> {
    let [first, second, third] = order_by.expressions.as_slice() else {
        return Ok(None);
    };
    if first.nulls_first || second.nulls_first || third.nulls_first {
        return Ok(None);
    }
    let first = batch.column(output_batch_column_index(batch, &first.column)?);
    let second = batch.column(output_batch_column_index(batch, &second.column)?);
    let third = batch.column(output_batch_column_index(batch, &third.column)?);
    if first.null_count() != 0 || second.null_count() != 0 || third.null_count() != 0 {
        return Ok(None);
    }
    let (Some(first), Some(second), Some(third)) = (
        first.as_any().downcast_ref::<Int32Array>(),
        second.as_any().downcast_ref::<Int64Array>(),
        third.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return Ok(None);
    };
    let mut keys = (0..batch.num_rows())
        .map(|row| {
            (
                first.value(row),
                second.value(row),
                third.value(row),
                row as u32,
            )
        })
        .collect::<Vec<_>>();
    keys.select_nth_unstable_by(limit, |left, right| {
        (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
    });
    keys.truncate(limit);
    keys.sort_unstable_by(|left, right| {
        (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
    });
    let indices = keys
        .into_iter()
        .map(|(_, _, _, row)| row)
        .collect::<Vec<_>>();
    Ok(Some(take_primitive_record_batch_by_raw_indices(
        batch, &indices,
    )?))
}

fn sort_nullable_i32_i64_output_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: usize,
) -> Result<Option<RecordBatch>> {
    let [first, second] = order_by.expressions.as_slice() else {
        return Ok(None);
    };
    if !first.nulls_first || second.nulls_first {
        return Ok(None);
    }
    let first = batch.column(output_batch_column_index(batch, &first.column)?);
    let second = batch.column(output_batch_column_index(batch, &second.column)?);
    if second.null_count() != 0 {
        return Ok(None);
    }
    let (Some(first), Some(second)) = (
        first.as_any().downcast_ref::<Int32Array>(),
        second.as_any().downcast_ref::<Int64Array>(),
    ) else {
        return Ok(None);
    };
    let mut keys = (0..batch.num_rows())
        .map(|row| {
            let present = !first.is_null(row);
            let value = if present { first.value(row) } else { 0 };
            (present, value, second.value(row), row as u32)
        })
        .collect::<Vec<_>>();
    keys.select_nth_unstable_by(limit, |left, right| {
        (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
    });
    keys.truncate(limit);
    keys.sort_unstable_by(|left, right| {
        (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
    });
    let indices = keys
        .into_iter()
        .map(|(_, _, _, row)| row)
        .collect::<Vec<_>>();
    Ok(Some(take_primitive_record_batch_by_raw_indices(
        batch, &indices,
    )?))
}

fn fixed_primitive_partial_sort_enabled() -> bool {
    std::env::var("DODAM_ENABLE_FIXED_PRIMITIVE_PARTIAL_SORT")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

enum PrimitiveSortColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
    Float64(&'a Float64Array),
    Date32(&'a Date32Array),
}

impl PrimitiveSortColumn<'_> {
    fn compare(&self, left: usize, right: usize) -> std::cmp::Ordering {
        match self {
            Self::Int32(values) => values.value(left).cmp(&values.value(right)),
            Self::Int64(values) => values.value(left).cmp(&values.value(right)),
            Self::UInt64(values) => values.value(left).cmp(&values.value(right)),
            Self::Float64(values) => values.value(left).total_cmp(&values.value(right)),
            Self::Date32(values) => values.value(left).cmp(&values.value(right)),
        }
    }
}

fn sort_primitive_output_batch(
    batch: &RecordBatch,
    order_by: &SortKey,
    limit: Option<usize>,
) -> Result<Option<RecordBatch>> {
    if !primitive_partial_sort_enabled() {
        return Ok(None);
    }
    let Some(limit) = limit else {
        return Ok(None);
    };
    if batch.num_rows() == 0 || limit == 0 || limit >= batch.num_rows() {
        return Ok(None);
    }
    if order_by.expressions.is_empty() || order_by.expressions.len() > 4 {
        return Ok(None);
    }
    let mut sort_columns = Vec::with_capacity(order_by.expressions.len());
    let mut descending = Vec::with_capacity(order_by.expressions.len());
    for sort in &order_by.expressions {
        let column_index = output_batch_column_index(batch, &sort.column)?;
        let column = batch.column(column_index);
        if column.null_count() != 0 || sort.nulls_first {
            return Ok(None);
        }
        let Some(sort_column) = primitive_sort_column(column) else {
            return Ok(None);
        };
        sort_columns.push(sort_column);
        descending.push(sort.descending);
    }
    if !batch.columns().iter().all(|column| {
        column.null_count() == 0
            && matches!(
                column.data_type(),
                DataType::Int32
                    | DataType::Int64
                    | DataType::UInt64
                    | DataType::Float64
                    | DataType::Date32
            )
    }) {
        return Ok(None);
    }
    let row_count = u32::try_from(batch.num_rows()).map_err(|_| {
        DodamError::UnsupportedSql("primitive partial sort row count overflow".to_string())
    })?;
    let mut indices = (0..row_count).collect::<Vec<_>>();
    {
        let compare_rows = |left: &u32, right: &u32| {
            let left = *left as usize;
            let right = *right as usize;
            for (column, descending) in sort_columns.iter().zip(&descending) {
                let ordering = column.compare(left, right);
                if ordering != std::cmp::Ordering::Equal {
                    return if *descending {
                        ordering.reverse()
                    } else {
                        ordering
                    };
                }
            }
            left.cmp(&right)
        };
        indices.select_nth_unstable_by(limit, compare_rows);
        indices.truncate(limit);
        indices.sort_unstable_by(compare_rows);
    }
    take_primitive_record_batch_by_raw_indices(batch, &indices).map(Some)
}

fn primitive_sort_column(column: &ArrayRef) -> Option<PrimitiveSortColumn<'_>> {
    match column.data_type() {
        DataType::Int32 => column
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(PrimitiveSortColumn::Int32),
        DataType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(PrimitiveSortColumn::Int64),
        DataType::UInt64 => column
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(PrimitiveSortColumn::UInt64),
        DataType::Float64 => column
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(PrimitiveSortColumn::Float64),
        DataType::Date32 => column
            .as_any()
            .downcast_ref::<Date32Array>()
            .map(PrimitiveSortColumn::Date32),
        _ => None,
    }
}

fn take_primitive_record_batch_by_indices(
    batch: &RecordBatch,
    indices: &UInt32Array,
) -> Result<Option<RecordBatch>> {
    if indices.null_count() != 0 || batch.num_rows() == 0 {
        return Ok(None);
    }
    let mut raw_indices = Vec::with_capacity(indices.len());
    for row in 0..indices.len() {
        raw_indices.push(indices.value(row));
    }
    take_primitive_record_batch_by_raw_indices(batch, &raw_indices).map(Some)
}

fn take_primitive_record_batch_by_raw_indices(
    batch: &RecordBatch,
    raw_indices: &[u32],
) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        let Some(array) = gather_primitive_array(column, &raw_indices) else {
            return Ok(take_record_batch(
                batch,
                &UInt32Array::from(raw_indices.to_vec()),
            )?);
        };
        columns.push(array);
    }
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

fn primitive_partial_sort_enabled() -> bool {
    std::env::var("DODAM_ENABLE_PRIMITIVE_PARTIAL_SORT")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
