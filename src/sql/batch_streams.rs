use super::*;

pub(super) struct OrderedLimitCollector {
    to_skip: usize,
    remaining: usize,
}

impl OrderedLimitCollector {
    pub(super) fn new(limit: Option<usize>, offset: usize) -> Self {
        Self {
            to_skip: offset,
            remaining: limit.unwrap_or(usize::MAX),
        }
    }

    pub(super) fn push_batch(&mut self, batch: RecordBatch, output: &mut Vec<RecordBatch>) {
        if self.remaining == 0 {
            return;
        }
        if self.to_skip >= batch.num_rows() {
            self.to_skip -= batch.num_rows();
            return;
        }
        let batch = if self.to_skip > 0 {
            let sliced = batch.slice(self.to_skip, batch.num_rows() - self.to_skip);
            self.to_skip = 0;
            sliced
        } else {
            batch
        };
        let rows = self.remaining.min(batch.num_rows());
        self.remaining -= rows;
        if rows > 0 {
            output.push(batch.slice(0, rows));
        }
    }

    pub(super) fn is_complete(&self) -> bool {
        self.remaining == 0
    }
}

#[derive(Default)]
pub(super) enum MonotonicOrderState {
    #[default]
    Empty,
    Int32(i32),
    Int64(i64),
    Utf8(String),
}

impl MonotonicOrderState {
    pub(super) fn consume_batch(&mut self, batch: &RecordBatch, column: &str) -> Result<bool> {
        let index = output_batch_column_index(batch, column)?;
        let values = batch.column(index);
        match values.data_type() {
            DataType::Int32 | DataType::Date32 => {
                let values = values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32/Date32 data type");
                for row in 0..values.len() {
                    if values.is_null(row) {
                        return Ok(false);
                    }
                    let value = values.value(row);
                    if let Self::Int32(previous) = self
                        && value < *previous
                    {
                        return Ok(false);
                    }
                    *self = Self::Int32(value);
                }
                Ok(true)
            }
            DataType::Int64 => {
                let values = values
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 data type");
                for row in 0..values.len() {
                    if values.is_null(row) {
                        return Ok(false);
                    }
                    let value = values.value(row);
                    if let Self::Int64(previous) = self
                        && value < *previous
                    {
                        return Ok(false);
                    }
                    *self = Self::Int64(value);
                }
                Ok(true)
            }
            DataType::Utf8 => {
                let values = values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 data type");
                for row in 0..values.len() {
                    if values.is_null(row) {
                        return Ok(false);
                    }
                    let value = values.value(row);
                    if let Self::Utf8(previous) = self
                        && value < previous.as_str()
                    {
                        return Ok(false);
                    }
                    *self = Self::Utf8(value.to_string());
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

pub(super) fn collect_batches(mut stream: SendableBatchStream) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }
    Ok(batches)
}

pub(super) fn collect_ordered_stream_limit_batches(
    mut stream: SendableBatchStream,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<RecordBatch>> {
    let mut to_skip = offset;
    let mut remaining = limit.unwrap_or(usize::MAX);
    let mut batches = Vec::new();
    if remaining == 0 {
        return Ok(batches);
    }
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        if to_skip >= batch.num_rows() {
            to_skip -= batch.num_rows();
            continue;
        }
        let batch = if to_skip > 0 {
            let sliced = batch.slice(to_skip, batch.num_rows() - to_skip);
            to_skip = 0;
            sliced
        } else {
            batch
        };
        let rows = remaining.min(batch.num_rows());
        remaining -= rows;
        if rows > 0 {
            batches.push(batch.slice(0, rows));
        }
        if remaining == 0 {
            break;
        }
    }
    Ok(batches)
}

pub(super) fn collect_verified_monotonic_order_limit_batches(
    mut stream: SendableBatchStream,
    order_column: &str,
    limit: Option<usize>,
    offset: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let mut order_state = MonotonicOrderState::default();
    let mut limiter = OrderedLimitCollector::new(limit, offset);
    let mut output = Vec::new();
    if limiter.is_complete() {
        return Ok(Some(output));
    }
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        if !order_state.consume_batch(&batch, order_column)? {
            return Ok(None);
        }
        limiter.push_batch(batch, &mut output);
        if limiter.is_complete() {
            break;
        }
    }
    Ok(Some(output))
}

pub(super) fn collect_expression_filtered_limit_batches(
    mut stream: SendableBatchStream,
    predicate: &SqlExpr,
    table_alias: Option<&str>,
    limit: usize,
) -> Result<Vec<RecordBatch>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut batches = Vec::new();
    let mut remaining = limit;
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let mut filtered = apply_output_expression_filter(vec![batch], predicate, table_alias)?;
        for batch in filtered.drain(..) {
            if remaining == 0 {
                return Ok(batches);
            }
            if batch.num_rows() > remaining {
                batches.push(batch.slice(0, remaining));
                return Ok(batches);
            }
            remaining -= batch.num_rows();
            batches.push(batch);
        }
    }
    Ok(batches)
}

pub(super) fn apply_output_filter_stream(
    stream: SendableBatchStream,
    filter: Option<FilterExpr>,
) -> SendableBatchStream {
    let Some(filter) = filter else {
        return stream;
    };
    let (input, metrics) = stream.into_parts();
    SendableBatchStream::new(Box::new(OutputFilterStream { input, filter }), metrics)
}

struct OutputFilterStream {
    input: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    filter: FilterExpr,
}

impl Iterator for OutputFilterStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        for batch in &mut self.input {
            match batch {
                Ok(batch) => match filter_batch(batch, &self.filter) {
                    Ok(batch) if batch.num_rows() == 0 => continue,
                    result => return Some(result),
                },
                Err(error) => return Some(Err(error)),
            }
        }
        None
    }
}
