use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{BuildHasherDefault, Hasher};

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMillisecondArray, UInt64Array,
};
use arrow::compute::kernels::aggregate::{max, max_string, min, min_string, sum};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_row::{OwnedRow, RowConverter, SortField};

use crate::error::{DodamError, Result};
use crate::execution::logical::{
    AggregateExpr, AggregateMetrics, AggregateResult, AggregateValue, GroupAggregateResult,
    GroupValue,
};
use crate::execution::metrics::SendableBatchStream;
use crate::execution::physical::column_index;

type AggregateHashMap<K, V> = HashMap<K, V, BuildHasherDefault<AggregateHasher>>;

#[derive(Default)]
struct AggregateHasher {
    hash: u64,
}

impl Hasher for AggregateHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.hash ^ 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        self.hash = hash;
    }

    fn write_i32(&mut self, value: i32) {
        self.write_u64(value as u32 as u64);
    }

    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    fn write_u64(&mut self, value: u64) {
        let mut hash = self.hash ^ value;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^= hash >> 33;
        self.hash = hash;
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

pub fn collect_aggregates(
    mut stream: SendableBatchStream,
    fragments: usize,
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let shared_sum_avg_columns = shared_sum_avg_columns(aggregates);
    let mut shared_numeric_states = shared_sum_avg_columns
        .iter()
        .map(|column| (column.clone(), NumericState::default()))
        .collect::<Vec<_>>();
    let mut accumulators = aggregates
        .iter()
        .cloned()
        .map(|aggregate| GlobalAggregateSlot::new(aggregate, &shared_sum_avg_columns))
        .collect::<Vec<_>>();
    let mut metrics = AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    };

    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }

        metrics.batches += 1;
        metrics.rows += batch.num_rows();
        for (column, state) in &mut shared_numeric_states {
            state.update(
                batch.column(column_index(&batch, column)?),
                &AggregateExpr::Sum(column.clone()),
            )?;
        }
        for accumulator in &mut accumulators {
            accumulator.update(&batch)?;
        }
    }

    metrics.values = accumulators
        .into_iter()
        .map(|accumulator| accumulator.finish(&shared_numeric_states))
        .collect::<Result<Vec<_>>>()?;
    Ok(metrics)
}

fn shared_sum_avg_columns(aggregates: &[AggregateExpr]) -> Vec<String> {
    let mut columns = Vec::new();
    for aggregate in aggregates {
        let AggregateExpr::Sum(column) = aggregate else {
            continue;
        };
        if aggregates.iter().any(
            |candidate| matches!(candidate, AggregateExpr::Avg(avg_column) if avg_column == column),
        ) && !columns.iter().any(|existing| existing == column)
        {
            columns.push(column.clone());
        }
    }
    columns
}

enum GlobalAggregateSlot {
    Accumulator(AggregateAccumulator),
    SharedSum {
        expr: AggregateExpr,
        state_index: usize,
    },
    SharedAvg {
        expr: AggregateExpr,
        state_index: usize,
    },
}

impl GlobalAggregateSlot {
    fn new(expr: AggregateExpr, shared_sum_avg_columns: &[String]) -> Self {
        match &expr {
            AggregateExpr::Sum(column) => {
                if let Some(state_index) = shared_sum_avg_columns
                    .iter()
                    .position(|shared| shared == column)
                {
                    Self::SharedSum { expr, state_index }
                } else {
                    Self::Accumulator(AggregateAccumulator::new(expr))
                }
            }
            AggregateExpr::Avg(column) => {
                if let Some(state_index) = shared_sum_avg_columns
                    .iter()
                    .position(|shared| shared == column)
                {
                    Self::SharedAvg { expr, state_index }
                } else {
                    Self::Accumulator(AggregateAccumulator::new(expr))
                }
            }
            _ => Self::Accumulator(AggregateAccumulator::new(expr)),
        }
    }

    fn update(&mut self, batch: &RecordBatch) -> Result<()> {
        match self {
            Self::Accumulator(accumulator) => accumulator.update(batch),
            Self::SharedSum { .. } | Self::SharedAvg { .. } => Ok(()),
        }
    }

    fn finish(self, shared_numeric_states: &[(String, NumericState)]) -> Result<AggregateResult> {
        Ok(match self {
            Self::Accumulator(accumulator) => accumulator.finish()?,
            Self::SharedSum { expr, state_index } => AggregateResult {
                expr,
                value: shared_numeric_states
                    .get(state_index)
                    .expect("shared numeric state")
                    .1
                    .sum_value(),
            },
            Self::SharedAvg { expr, state_index } => AggregateResult {
                expr,
                value: shared_numeric_states
                    .get(state_index)
                    .expect("shared numeric state")
                    .1
                    .avg_value(),
            },
        })
    }
}

pub fn collect_grouped_aggregates(
    stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    if can_use_single_key_count_sum_path(group_by, aggregates) {
        return collect_single_key_count_sum_groups(stream, fragments, group_by, aggregates);
    }
    if can_use_single_key_fast_path(group_by, aggregates) {
        return collect_single_key_groups(stream, fragments, group_by, aggregates);
    }

    collect_grouped_aggregates_generic(stream, fragments, group_by, aggregates, None, None)
}

fn can_use_single_key_count_sum_path(group_by: &[String], aggregates: &[AggregateExpr]) -> bool {
    group_by.len() == 1
        && matches!(
            aggregates,
            [AggregateExpr::CountStar, AggregateExpr::Sum(_)]
        )
}

fn collect_single_key_count_sum_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let AggregateExpr::Sum(sum_column) = &aggregates[1] else {
        unreachable!("count/sum fast path precondition");
    };
    let sum_expr = aggregates[1].clone();
    let mut group_index = CountSumGroupIndex::Unset;
    let mut groups = Vec::<CountSumGroup>::new();
    let mut metrics = AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    };

    while let Some(batch) = stream.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        metrics.batches += 1;
        metrics.rows += batch.num_rows();

        let key_column = batch.column(column_index(&batch, &group_by[0])?);
        if !matches!(
            key_column.data_type(),
            DataType::Utf8 | DataType::Int32 | DataType::Int64 | DataType::UInt64
        ) {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        }
        group_index.ensure_type(key_column.data_type());
        let sum_column = batch.column(column_index(&batch, sum_column)?);
        let sum_input = CountSumValueInput::new(sum_column, &sum_expr)?;

        for row in 0..batch.num_rows() {
            let group_id = group_index.group_id(key_column, row, &mut groups, &sum_input)?;
            groups[group_id].update(&sum_input, row);
        }
    }

    let mut group_results = groups
        .into_iter()
        .map(|group| group.finish(sum_expr.clone()))
        .collect::<Vec<_>>();
    group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.groups = group_results;
    Ok(metrics)
}

enum CountSumValueInput<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float64(&'a Float64Array),
}

impl<'a> CountSumValueInput<'a> {
    fn new(column: &'a ArrayRef, expr: &AggregateExpr) -> Result<Self> {
        match column.data_type() {
            DataType::Int32 => Ok(Self::Int32(
                column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 sum input"),
            )),
            DataType::Int64 => Ok(Self::Int64(
                column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 sum input"),
            )),
            DataType::Float64 => Ok(Self::Float64(
                column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 sum input"),
            )),
            data_type => Err(DodamError::UnsupportedAggregateType {
                function: "sum".to_string(),
                column: expr.referenced_column().unwrap_or("*").to_string(),
                data_type: data_type.clone(),
            }),
        }
    }
}

enum CountSumGroupIndex {
    Unset,
    Utf8 {
        groups: AggregateHashMap<String, usize>,
        null_group: Option<usize>,
    },
    Int32 {
        groups: AggregateHashMap<i32, usize>,
        null_group: Option<usize>,
    },
    Int64 {
        groups: AggregateHashMap<i64, usize>,
        null_group: Option<usize>,
    },
    UInt64 {
        groups: AggregateHashMap<u64, usize>,
        null_group: Option<usize>,
    },
}

impl CountSumGroupIndex {
    fn ensure_type(&mut self, data_type: &DataType) {
        if !matches!(self, Self::Unset) {
            return;
        }
        *self = match data_type {
            DataType::Utf8 => Self::Utf8 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::Int32 => Self::Int32 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::Int64 => Self::Int64 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::UInt64 => Self::UInt64 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            _ => unreachable!("count/sum fast path key type precondition"),
        };
    }

    fn group_id(
        &mut self,
        key_column: &ArrayRef,
        row: usize,
        groups_out: &mut Vec<CountSumGroup>,
        sum_input: &CountSumValueInput<'_>,
    ) -> Result<usize> {
        match self {
            Self::Utf8 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 group key");
                if values.is_null(row) {
                    return Ok(count_sum_null_group_id(
                        null_group,
                        groups_out,
                        GroupValue::Utf8(None),
                        sum_input,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key.to_string(), group_id);
                groups_out.push(CountSumGroup::new(
                    GroupValue::Utf8(Some(key.to_string())),
                    sum_input,
                ));
                Ok(group_id)
            }
            Self::Int32 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 group key");
                if values.is_null(row) {
                    return Ok(count_sum_null_group_id(
                        null_group,
                        groups_out,
                        GroupValue::Int64(None),
                        sum_input,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(&key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(CountSumGroup::new(
                    GroupValue::Int64(Some(i64::from(key))),
                    sum_input,
                ));
                Ok(group_id)
            }
            Self::Int64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 group key");
                if values.is_null(row) {
                    return Ok(count_sum_null_group_id(
                        null_group,
                        groups_out,
                        GroupValue::Int64(None),
                        sum_input,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(&key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(CountSumGroup::new(GroupValue::Int64(Some(key)), sum_input));
                Ok(group_id)
            }
            Self::UInt64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 group key");
                if values.is_null(row) {
                    return Ok(count_sum_null_group_id(
                        null_group,
                        groups_out,
                        GroupValue::UInt64(None),
                        sum_input,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(&key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(CountSumGroup::new(GroupValue::UInt64(Some(key)), sum_input));
                Ok(group_id)
            }
            Self::Unset => unreachable!("group index type should be initialized"),
        }
    }
}

fn count_sum_null_group_id(
    null_group: &mut Option<usize>,
    groups: &mut Vec<CountSumGroup>,
    key: GroupValue,
    sum_input: &CountSumValueInput<'_>,
) -> usize {
    if let Some(group_id) = *null_group {
        return group_id;
    }
    let group_id = groups.len();
    groups.push(CountSumGroup::new(key, sum_input));
    *null_group = Some(group_id);
    group_id
}

struct CountSumGroup {
    key: GroupValue,
    count: u64,
    sum_i64: i64,
    sum_f64: f64,
    sum_is_float: bool,
    sum_count: u64,
}

impl CountSumGroup {
    fn new(key: GroupValue, sum_input: &CountSumValueInput<'_>) -> Self {
        Self {
            key,
            count: 0,
            sum_i64: 0,
            sum_f64: 0.0,
            sum_is_float: matches!(sum_input, CountSumValueInput::Float64(_)),
            sum_count: 0,
        }
    }

    fn update(&mut self, sum_input: &CountSumValueInput<'_>, row: usize) {
        self.count += 1;
        match sum_input {
            CountSumValueInput::Int32(values) if values.is_valid(row) => {
                self.sum_i64 += i64::from(values.value(row));
                self.sum_count += 1;
            }
            CountSumValueInput::Int64(values) if values.is_valid(row) => {
                self.sum_i64 += values.value(row);
                self.sum_count += 1;
            }
            CountSumValueInput::Float64(values) if values.is_valid(row) => {
                self.sum_f64 += values.value(row);
                self.sum_count += 1;
            }
            _ => {}
        }
    }

    fn finish(self, sum_expr: AggregateExpr) -> GroupAggregateResult {
        let sum_value = if self.sum_count == 0 {
            if self.sum_is_float {
                AggregateValue::Float64(None)
            } else {
                AggregateValue::Int64(None)
            }
        } else if self.sum_is_float {
            AggregateValue::Float64(Some(self.sum_f64))
        } else {
            AggregateValue::Int64(Some(self.sum_i64))
        };
        GroupAggregateResult {
            keys: vec![self.key],
            values: vec![
                AggregateResult {
                    expr: AggregateExpr::CountStar,
                    value: AggregateValue::Count(self.count),
                },
                AggregateResult {
                    expr: sum_expr,
                    value: sum_value,
                },
            ],
        }
    }
}

fn collect_grouped_aggregates_generic(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
    first_batch: Option<RecordBatch>,
    initial_metrics: Option<AggregateMetrics>,
) -> Result<AggregateMetrics> {
    let mut groups: AggregateHashMap<OwnedRow, GroupState> = AggregateHashMap::default();
    let mut metrics = initial_metrics.unwrap_or(AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    });
    let mut pending = first_batch;

    loop {
        let batch = if let Some(batch) = pending.take() {
            batch
        } else {
            let Some(batch) = stream.next() else {
                break;
            };
            batch?
        };
        if batch.num_rows() == 0 {
            continue;
        }

        metrics.batches += 1;
        metrics.rows += batch.num_rows();
        let group_columns = group_by
            .iter()
            .map(|column| Ok((column.as_str(), batch.column(column_index(&batch, column)?))))
            .collect::<Result<Vec<_>>>()?;

        let group_arrays = group_columns
            .iter()
            .map(|(_, column)| (*column).clone())
            .collect::<Vec<_>>();
        let converter = RowConverter::new(
            group_arrays
                .iter()
                .map(|column| SortField::new(column.data_type().clone()))
                .collect(),
        )?;
        let encoded_rows = converter.convert_columns(&group_arrays)?;
        let aggregate_columns = aggregates
            .iter()
            .map(|aggregate| {
                aggregate
                    .referenced_column()
                    .map(|column| Ok(batch.column(column_index(&batch, column)?).clone()))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;

        for (row, encoded_row) in encoded_rows.iter().enumerate() {
            let key = encoded_row.owned();
            let group = match groups.entry(key) {
                Entry::Vacant(entry) => entry.insert(GroupState {
                    keys: group_key(&group_columns, row)?,
                    accumulators: aggregates
                        .iter()
                        .cloned()
                        .map(AggregateAccumulator::new)
                        .collect(),
                }),
                Entry::Occupied(entry) => entry.into_mut(),
            };
            for (accumulator, column) in group.accumulators.iter_mut().zip(&aggregate_columns) {
                accumulator.update_row(column.as_ref(), row)?;
            }
        }
    }

    let mut group_results = groups
        .into_values()
        .map(|group| {
            Ok(GroupAggregateResult {
                keys: group.keys,
                values: group
                    .accumulators
                    .into_iter()
                    .map(AggregateAccumulator::finish)
                    .collect::<Result<Vec<_>>>()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.groups = group_results;
    Ok(metrics)
}

fn can_use_single_key_fast_path(group_by: &[String], aggregates: &[AggregateExpr]) -> bool {
    group_by.len() == 1
        && !aggregates.is_empty()
        && aggregates.iter().all(|aggregate| {
            matches!(
                aggregate,
                AggregateExpr::CountStar
                    | AggregateExpr::Count(_)
                    | AggregateExpr::Sum(_)
                    | AggregateExpr::Avg(_)
                    | AggregateExpr::Min(_)
                    | AggregateExpr::Max(_)
            )
        })
}

fn collect_single_key_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let mut group_index = SingleKeyGroupIndex::Unset;
    let mut groups = Vec::<SingleKeyGroup>::new();
    let mut metrics = AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    };

    while let Some(batch) = stream.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        metrics.batches += 1;
        metrics.rows += batch.num_rows();

        let key_column = batch.column(column_index(&batch, &group_by[0])?);
        if !matches!(
            key_column.data_type(),
            DataType::Utf8 | DataType::Int32 | DataType::Int64 | DataType::UInt64
        ) {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        }
        group_index.ensure_type(key_column.data_type());
        let aggregate_inputs = typed_fast_inputs(&batch, aggregates)?;

        for row in 0..batch.num_rows() {
            let group_id = group_index.group_id(key_column, row, &mut groups, &aggregate_inputs)?;
            let group = &mut groups[group_id];
            for (state, input) in group.states.iter_mut().zip(&aggregate_inputs) {
                state.update(input, row);
            }
        }
    }

    let mut group_results = groups
        .into_iter()
        .map(SingleKeyGroup::finish)
        .collect::<Vec<_>>();
    group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.groups = group_results;
    Ok(metrics)
}

enum SingleKeyGroupIndex {
    Unset,
    Utf8 {
        groups: AggregateHashMap<String, usize>,
        null_group: Option<usize>,
    },
    Int32 {
        groups: AggregateHashMap<i32, usize>,
        null_group: Option<usize>,
    },
    Int64 {
        groups: AggregateHashMap<i64, usize>,
        null_group: Option<usize>,
    },
    UInt64 {
        groups: AggregateHashMap<u64, usize>,
        null_group: Option<usize>,
    },
}

impl SingleKeyGroupIndex {
    fn ensure_type(&mut self, data_type: &DataType) {
        if !matches!(self, Self::Unset) {
            return;
        }
        *self = match data_type {
            DataType::Utf8 => Self::Utf8 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::Int32 => Self::Int32 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::Int64 => Self::Int64 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            DataType::UInt64 => Self::UInt64 {
                groups: AggregateHashMap::default(),
                null_group: None,
            },
            _ => unreachable!("fast path key type precondition"),
        };
    }

    fn group_id(
        &mut self,
        key_column: &ArrayRef,
        row: usize,
        groups_out: &mut Vec<SingleKeyGroup>,
        inputs: &[FastAggregateInput<'_>],
    ) -> Result<usize> {
        match self {
            Self::Utf8 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 group key");
                if values.is_null(row) {
                    return Ok(group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::Utf8(None),
                        inputs,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key.to_string(), group_id);
                groups_out.push(SingleKeyGroup::new(
                    GroupValue::Utf8(Some(key.to_string())),
                    inputs,
                ));
                Ok(group_id)
            }
            Self::Int32 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 group key");
                if values.is_null(row) {
                    return Ok(group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::Int64(None),
                        inputs,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(&key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(SingleKeyGroup::new(
                    GroupValue::Int64(Some(i64::from(key))),
                    inputs,
                ));
                Ok(group_id)
            }
            Self::Int64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 group key");
                if values.is_null(row) {
                    return Ok(group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::Int64(None),
                        inputs,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(&key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(SingleKeyGroup::new(GroupValue::Int64(Some(key)), inputs));
                Ok(group_id)
            }
            Self::UInt64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 group key");
                if values.is_null(row) {
                    return Ok(group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::UInt64(None),
                        inputs,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(&key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(SingleKeyGroup::new(GroupValue::UInt64(Some(key)), inputs));
                Ok(group_id)
            }
            Self::Unset => unreachable!("group index type should be initialized"),
        }
    }
}

fn group_id_for_null(
    null_group: &mut Option<usize>,
    groups: &mut Vec<SingleKeyGroup>,
    key: GroupValue,
    inputs: &[FastAggregateInput<'_>],
) -> usize {
    if let Some(group_id) = *null_group {
        return group_id;
    }
    let group_id = groups.len();
    groups.push(SingleKeyGroup::new(key, inputs));
    *null_group = Some(group_id);
    group_id
}

struct SingleKeyGroup {
    key: GroupValue,
    states: Vec<FastAggregateState>,
}

impl SingleKeyGroup {
    fn new(key: GroupValue, inputs: &[FastAggregateInput<'_>]) -> Self {
        Self {
            key,
            states: inputs.iter().map(FastAggregateState::new).collect(),
        }
    }

    fn finish(self) -> GroupAggregateResult {
        GroupAggregateResult {
            keys: vec![self.key],
            values: self
                .states
                .into_iter()
                .map(FastAggregateState::finish)
                .collect::<Vec<_>>(),
        }
    }
}

enum FastAggregateInput<'a> {
    CountStar,
    Count {
        expr: AggregateExpr,
        values: &'a ArrayRef,
    },
    NumericInt32 {
        expr: AggregateExpr,
        values: &'a Int32Array,
    },
    NumericInt64 {
        expr: AggregateExpr,
        values: &'a Int64Array,
    },
    NumericFloat64 {
        expr: AggregateExpr,
        values: &'a Float64Array,
    },
    Date32 {
        expr: AggregateExpr,
        values: &'a Date32Array,
    },
    Date64 {
        expr: AggregateExpr,
        values: &'a Date64Array,
    },
    TimestampMillisecond {
        expr: AggregateExpr,
        values: &'a TimestampMillisecondArray,
        timezone: Option<String>,
    },
    Utf8 {
        expr: AggregateExpr,
        values: &'a StringArray,
    },
}

fn typed_fast_inputs<'a>(
    batch: &'a RecordBatch,
    aggregates: &[AggregateExpr],
) -> Result<Vec<FastAggregateInput<'a>>> {
    aggregates
        .iter()
        .map(|aggregate| match aggregate {
            AggregateExpr::CountStar => Ok(FastAggregateInput::CountStar),
            AggregateExpr::Count(column) => {
                let values = batch.column(column_index(batch, column)?);
                Ok(FastAggregateInput::Count {
                    expr: aggregate.clone(),
                    values,
                })
            }
            AggregateExpr::Sum(column) | AggregateExpr::Avg(column) => {
                let values = batch.column(column_index(batch, column)?);
                let expr = aggregate.clone();
                match values.data_type() {
                    DataType::Int32 => Ok(FastAggregateInput::NumericInt32 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .expect("Int32 numeric input"),
                    }),
                    DataType::Int64 => Ok(FastAggregateInput::NumericInt64 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .expect("Int64 numeric input"),
                    }),
                    DataType::Float64 => Ok(FastAggregateInput::NumericFloat64 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .expect("Float64 numeric input"),
                    }),
                    _ => Err(DodamError::UnsupportedAggregateType {
                        function: aggregate_function_name(aggregate).to_string(),
                        column: column.clone(),
                        data_type: values.data_type().clone(),
                    }),
                }
            }
            AggregateExpr::Min(column) | AggregateExpr::Max(column) => {
                let values = batch.column(column_index(batch, column)?);
                let expr = aggregate.clone();
                match values.data_type() {
                    DataType::Int32 => Ok(FastAggregateInput::NumericInt32 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .expect("Int32 min/max input"),
                    }),
                    DataType::Int64 => Ok(FastAggregateInput::NumericInt64 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .expect("Int64 min/max input"),
                    }),
                    DataType::Float64 => Ok(FastAggregateInput::NumericFloat64 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .expect("Float64 min/max input"),
                    }),
                    DataType::Date32 => Ok(FastAggregateInput::Date32 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Date32Array>()
                            .expect("Date32 min/max input"),
                    }),
                    DataType::Date64 => Ok(FastAggregateInput::Date64 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Date64Array>()
                            .expect("Date64 min/max input"),
                    }),
                    DataType::Timestamp(TimeUnit::Millisecond, timezone) => {
                        Ok(FastAggregateInput::TimestampMillisecond {
                            expr,
                            values: values
                                .as_any()
                                .downcast_ref::<TimestampMillisecondArray>()
                                .expect("TimestampMillisecond min/max input"),
                            timezone: timezone.as_ref().map(ToString::to_string),
                        })
                    }
                    DataType::Utf8 => Ok(FastAggregateInput::Utf8 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .expect("Utf8 min/max input"),
                    }),
                    _ => Err(DodamError::UnsupportedAggregateType {
                        function: aggregate_function_name(aggregate).to_string(),
                        column: column.clone(),
                        data_type: values.data_type().clone(),
                    }),
                }
            }
        })
        .collect()
}

enum FastAggregateState {
    Count {
        expr: AggregateExpr,
        count: u64,
    },
    SumInt {
        expr: AggregateExpr,
        sum: i64,
        count: u64,
    },
    SumFloat {
        expr: AggregateExpr,
        sum: f64,
        count: u64,
    },
    AvgInt {
        expr: AggregateExpr,
        sum: i64,
        count: u64,
    },
    AvgFloat {
        expr: AggregateExpr,
        sum: f64,
        count: u64,
    },
    MinMax {
        expr: AggregateExpr,
        value: Option<AggregateValue>,
        null_value: AggregateValue,
    },
}

impl FastAggregateState {
    fn new(input: &FastAggregateInput<'_>) -> Self {
        match input {
            FastAggregateInput::CountStar => Self::Count {
                expr: AggregateExpr::CountStar,
                count: 0,
            },
            FastAggregateInput::Count { expr, .. } => Self::Count {
                expr: expr.clone(),
                count: 0,
            },
            FastAggregateInput::NumericInt32 { expr, .. }
            | FastAggregateInput::NumericInt64 { expr, .. } => numeric_fast_state(expr, false),
            FastAggregateInput::NumericFloat64 { expr, .. } => numeric_fast_state(expr, true),
            FastAggregateInput::Date32 { expr, .. } => {
                min_max_fast_state(expr, AggregateValue::Date32(None))
            }
            FastAggregateInput::Date64 { expr, .. } => {
                min_max_fast_state(expr, AggregateValue::Date64(None))
            }
            FastAggregateInput::TimestampMillisecond { expr, timezone, .. } => min_max_fast_state(
                expr,
                AggregateValue::TimestampMillisecond(None, timezone.clone()),
            ),
            FastAggregateInput::Utf8 { expr, .. } => {
                min_max_fast_state(expr, AggregateValue::Utf8(None))
            }
        }
    }

    fn update(&mut self, input: &FastAggregateInput<'_>, row: usize) {
        match input {
            FastAggregateInput::CountStar => {
                if let Self::Count { count, .. } = self {
                    *count += 1;
                }
            }
            FastAggregateInput::Count { values, .. } => {
                if let Self::Count { count, .. } = self
                    && values.is_valid(row)
                {
                    *count += 1;
                }
            }
            FastAggregateInput::NumericInt32 { values, .. } if values.is_valid(row) => {
                let value = i64::from(values.value(row));
                match self {
                    Self::SumInt { sum, count, .. } | Self::AvgInt { sum, count, .. } => {
                        *sum += value;
                        *count += 1;
                    }
                    Self::MinMax {
                        expr, value: state, ..
                    } => {
                        update_min_max_value(expr, state, AggregateValue::Int64(Some(value)));
                    }
                    _ => {}
                }
            }
            FastAggregateInput::NumericInt64 { values, .. } if values.is_valid(row) => {
                let value = values.value(row);
                match self {
                    Self::SumInt { sum, count, .. } | Self::AvgInt { sum, count, .. } => {
                        *sum += value;
                        *count += 1;
                    }
                    Self::MinMax {
                        expr, value: state, ..
                    } => {
                        update_min_max_value(expr, state, AggregateValue::Int64(Some(value)));
                    }
                    _ => {}
                }
            }
            FastAggregateInput::NumericFloat64 { values, .. } if values.is_valid(row) => {
                let value = values.value(row);
                match self {
                    Self::SumFloat { sum, count, .. } | Self::AvgFloat { sum, count, .. } => {
                        *sum += value;
                        *count += 1;
                    }
                    Self::MinMax {
                        expr, value: state, ..
                    } => {
                        update_min_max_value(expr, state, AggregateValue::Float64(Some(value)));
                    }
                    _ => {}
                }
            }
            FastAggregateInput::Date32 { values, .. } if values.is_valid(row) => {
                if let Self::MinMax {
                    expr, value: state, ..
                } = self
                {
                    update_min_max_value(
                        expr,
                        state,
                        AggregateValue::Date32(Some(values.value(row))),
                    );
                }
            }
            FastAggregateInput::Date64 { values, .. } if values.is_valid(row) => {
                if let Self::MinMax {
                    expr, value: state, ..
                } = self
                {
                    update_min_max_value(
                        expr,
                        state,
                        AggregateValue::Date64(Some(values.value(row))),
                    );
                }
            }
            FastAggregateInput::TimestampMillisecond {
                values, timezone, ..
            } if values.is_valid(row) => {
                if let Self::MinMax {
                    expr, value: state, ..
                } = self
                {
                    update_min_max_value(
                        expr,
                        state,
                        AggregateValue::TimestampMillisecond(
                            Some(values.value(row)),
                            timezone.clone(),
                        ),
                    );
                }
            }
            FastAggregateInput::Utf8 { values, .. } if values.is_valid(row) => {
                if let Self::MinMax {
                    expr, value: state, ..
                } = self
                {
                    update_min_max_value(
                        expr,
                        state,
                        AggregateValue::Utf8(Some(values.value(row).to_string())),
                    );
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> AggregateResult {
        match self {
            Self::Count { expr, count } => AggregateResult {
                expr,
                value: AggregateValue::Count(count),
            },
            Self::SumInt { expr, sum, count } => AggregateResult {
                expr,
                value: if count > 0 {
                    AggregateValue::Int64(Some(sum))
                } else {
                    AggregateValue::Int64(None)
                },
            },
            Self::SumFloat { expr, sum, count } => AggregateResult {
                expr,
                value: if count > 0 {
                    AggregateValue::Float64(Some(sum))
                } else {
                    AggregateValue::Float64(None)
                },
            },
            Self::AvgInt { expr, sum, count } => AggregateResult {
                expr,
                value: if count > 0 {
                    AggregateValue::Float64(Some(sum as f64 / count as f64))
                } else {
                    AggregateValue::Float64(None)
                },
            },
            Self::AvgFloat { expr, sum, count } => AggregateResult {
                expr,
                value: if count > 0 {
                    AggregateValue::Float64(Some(sum / count as f64))
                } else {
                    AggregateValue::Float64(None)
                },
            },
            Self::MinMax {
                expr,
                value,
                null_value,
            } => AggregateResult {
                expr,
                value: value.unwrap_or(null_value),
            },
        }
    }
}

fn numeric_fast_state(expr: &AggregateExpr, float: bool) -> FastAggregateState {
    match (expr, float) {
        (AggregateExpr::Sum(_), false) => FastAggregateState::SumInt {
            expr: expr.clone(),
            sum: 0,
            count: 0,
        },
        (AggregateExpr::Sum(_), true) => FastAggregateState::SumFloat {
            expr: expr.clone(),
            sum: 0.0,
            count: 0,
        },
        (AggregateExpr::Avg(_), false) => FastAggregateState::AvgInt {
            expr: expr.clone(),
            sum: 0,
            count: 0,
        },
        (AggregateExpr::Avg(_), true) => FastAggregateState::AvgFloat {
            expr: expr.clone(),
            sum: 0.0,
            count: 0,
        },
        (AggregateExpr::Min(_) | AggregateExpr::Max(_), false) => {
            min_max_fast_state(expr, AggregateValue::Int64(None))
        }
        (AggregateExpr::Min(_) | AggregateExpr::Max(_), true) => {
            min_max_fast_state(expr, AggregateValue::Float64(None))
        }
        _ => unreachable!("numeric fast input only supports sum/avg/min/max"),
    }
}

fn min_max_fast_state(expr: &AggregateExpr, null_value: AggregateValue) -> FastAggregateState {
    FastAggregateState::MinMax {
        expr: expr.clone(),
        value: None,
        null_value,
    }
}

fn update_min_max_value(
    expr: &AggregateExpr,
    state: &mut Option<AggregateValue>,
    candidate: AggregateValue,
) {
    let replace = match (&expr, state.as_ref(), &candidate) {
        (_, None, _) => true,
        (
            AggregateExpr::Min(_),
            Some(AggregateValue::Int64(Some(current))),
            AggregateValue::Int64(Some(candidate)),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::Int64(Some(current))),
            AggregateValue::Int64(Some(candidate)),
        ) => candidate > current,
        (
            AggregateExpr::Min(_),
            Some(AggregateValue::Float64(Some(current))),
            AggregateValue::Float64(Some(candidate)),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::Float64(Some(current))),
            AggregateValue::Float64(Some(candidate)),
        ) => candidate > current,
        (
            AggregateExpr::Min(_),
            Some(AggregateValue::Date32(Some(current))),
            AggregateValue::Date32(Some(candidate)),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::Date32(Some(current))),
            AggregateValue::Date32(Some(candidate)),
        ) => candidate > current,
        (
            AggregateExpr::Min(_),
            Some(AggregateValue::Date64(Some(current))),
            AggregateValue::Date64(Some(candidate)),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::Date64(Some(current))),
            AggregateValue::Date64(Some(candidate)),
        ) => candidate > current,
        (
            AggregateExpr::Min(_),
            Some(AggregateValue::TimestampMillisecond(Some(current), _)),
            AggregateValue::TimestampMillisecond(Some(candidate), _),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::TimestampMillisecond(Some(current), _)),
            AggregateValue::TimestampMillisecond(Some(candidate), _),
        ) => candidate > current,
        (
            AggregateExpr::Min(_),
            Some(AggregateValue::Utf8(Some(current))),
            AggregateValue::Utf8(Some(candidate)),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::Utf8(Some(current))),
            AggregateValue::Utf8(Some(candidate)),
        ) => candidate > current,
        _ => false,
    };
    if replace {
        *state = Some(candidate);
    }
}

struct GroupState {
    keys: Vec<GroupValue>,
    accumulators: Vec<AggregateAccumulator>,
}

fn group_key(columns: &[(&str, &ArrayRef)], row: usize) -> Result<Vec<GroupValue>> {
    columns
        .iter()
        .map(|(name, column)| match column.data_type() {
            DataType::Int32 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 data type");
                Ok(GroupValue::Int64(
                    values.is_valid(row).then(|| i64::from(values.value(row))),
                ))
            }
            DataType::Int64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 data type");
                Ok(GroupValue::Int64(
                    values.is_valid(row).then(|| values.value(row)),
                ))
            }
            DataType::UInt64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 data type");
                Ok(GroupValue::UInt64(
                    values.is_valid(row).then(|| values.value(row)),
                ))
            }
            DataType::Utf8 => {
                let values = column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 data type");
                Ok(GroupValue::Utf8(
                    values.is_valid(row).then(|| values.value(row).to_string()),
                ))
            }
            data_type => Err(DodamError::UnsupportedGroupByType {
                column: (*name).to_string(),
                data_type: data_type.clone(),
            }),
        })
        .collect()
}

fn compare_group_keys(left: &[GroupValue], right: &[GroupValue]) -> std::cmp::Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| match (left, right) {
            (GroupValue::Int64(left), GroupValue::Int64(right)) => left.cmp(right),
            (GroupValue::UInt64(left), GroupValue::UInt64(right)) => left.cmp(right),
            (GroupValue::Utf8(left), GroupValue::Utf8(right)) => left.cmp(right),
            (GroupValue::Int64(_), GroupValue::UInt64(_) | GroupValue::Utf8(_)) => {
                std::cmp::Ordering::Less
            }
            (GroupValue::UInt64(_), GroupValue::Int64(_)) => std::cmp::Ordering::Greater,
            (GroupValue::UInt64(_), GroupValue::Utf8(_)) => std::cmp::Ordering::Less,
            (GroupValue::Utf8(_), GroupValue::Int64(_) | GroupValue::UInt64(_)) => {
                std::cmp::Ordering::Greater
            }
        })
        .find(|ordering| *ordering != std::cmp::Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

enum AggregateAccumulator {
    CountStar {
        count: u64,
    },
    Count {
        expr: AggregateExpr,
        count: u64,
    },
    Sum {
        expr: AggregateExpr,
        state: NumericState,
    },
    Avg {
        expr: AggregateExpr,
        state: NumericState,
    },
    Min {
        expr: AggregateExpr,
        state: MinMaxState,
    },
    Max {
        expr: AggregateExpr,
        state: MinMaxState,
    },
}

impl AggregateAccumulator {
    fn new(expr: AggregateExpr) -> Self {
        match expr {
            AggregateExpr::CountStar => Self::CountStar { count: 0 },
            AggregateExpr::Count(_) => Self::Count { expr, count: 0 },
            AggregateExpr::Sum(_) => Self::Sum {
                expr,
                state: NumericState::default(),
            },
            AggregateExpr::Avg(_) => Self::Avg {
                expr,
                state: NumericState::default(),
            },
            AggregateExpr::Min(_) => Self::Min {
                expr,
                state: MinMaxState::default(),
            },
            AggregateExpr::Max(_) => Self::Max {
                expr,
                state: MinMaxState::default(),
            },
        }
    }

    fn update(&mut self, batch: &RecordBatch) -> Result<()> {
        match self {
            Self::CountStar { count } => {
                *count += batch.num_rows() as u64;
                Ok(())
            }
            Self::Count { expr, count } => {
                let column = aggregate_column(batch, expr)?;
                *count += (column.len() - column.null_count()) as u64;
                Ok(())
            }
            Self::Sum { expr, state } | Self::Avg { expr, state } => {
                state.update(aggregate_column(batch, expr)?, expr)
            }
            Self::Min { expr, state } => state.update_min(aggregate_column(batch, expr)?, expr),
            Self::Max { expr, state } => state.update_max(aggregate_column(batch, expr)?, expr),
        }
    }

    fn update_row(&mut self, column: Option<&ArrayRef>, row: usize) -> Result<()> {
        match self {
            Self::CountStar { count } => {
                *count += 1;
                Ok(())
            }
            Self::Count { expr, count } => {
                let column =
                    column.ok_or_else(|| DodamError::InvalidAggregate(expr.to_string()))?;
                if column.is_valid(row) {
                    *count += 1;
                }
                Ok(())
            }
            Self::Sum { expr, state } | Self::Avg { expr, state } => {
                let column =
                    column.ok_or_else(|| DodamError::InvalidAggregate(expr.to_string()))?;
                state.update_row(column, row, expr)
            }
            Self::Min { expr, state } => {
                let column =
                    column.ok_or_else(|| DodamError::InvalidAggregate(expr.to_string()))?;
                state.update_min_row(column, row, expr)
            }
            Self::Max { expr, state } => {
                let column =
                    column.ok_or_else(|| DodamError::InvalidAggregate(expr.to_string()))?;
                state.update_max_row(column, row, expr)
            }
        }
    }

    fn finish(self) -> Result<AggregateResult> {
        Ok(match self {
            Self::CountStar { count } => AggregateResult {
                expr: AggregateExpr::CountStar,
                value: AggregateValue::Count(count),
            },
            Self::Count { expr, count } => AggregateResult {
                expr,
                value: AggregateValue::Count(count),
            },
            Self::Sum { expr, state } => AggregateResult {
                expr,
                value: state.sum_value(),
            },
            Self::Avg { expr, state } => AggregateResult {
                expr,
                value: state.avg_value(),
            },
            Self::Min { expr, state } => AggregateResult {
                expr,
                value: state.value(),
            },
            Self::Max { expr, state } => AggregateResult {
                expr,
                value: state.value(),
            },
        })
    }
}

#[derive(Default)]
struct NumericState {
    sum_i64: i64,
    sum_f64: f64,
    count: u64,
    output: Option<NumericOutput>,
}

#[derive(Clone, Copy)]
enum NumericOutput {
    Int64,
    Float64,
}

impl NumericState {
    fn update(&mut self, column: &ArrayRef, expr: &AggregateExpr) -> Result<()> {
        match column.data_type() {
            DataType::Int32 => {
                self.output.get_or_insert(NumericOutput::Int64);
                let values = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 data type");
                if let Some(value) = sum(values) {
                    self.sum_i64 += i64::from(value);
                }
                self.count += (values.len() - values.null_count()) as u64;
                Ok(())
            }
            DataType::Int64 => {
                self.output.get_or_insert(NumericOutput::Int64);
                let values = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 data type");
                if let Some(value) = sum(values) {
                    self.sum_i64 += value;
                }
                self.count += (values.len() - values.null_count()) as u64;
                Ok(())
            }
            DataType::Float64 => {
                self.output.get_or_insert(NumericOutput::Float64);
                let values = column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 data type");
                if let Some(value) = sum(values) {
                    self.sum_f64 += value;
                }
                self.count += (values.len() - values.null_count()) as u64;
                Ok(())
            }
            data_type => unsupported_aggregate_type(expr, data_type),
        }
    }

    fn update_row(&mut self, column: &ArrayRef, row: usize, expr: &AggregateExpr) -> Result<()> {
        match column.data_type() {
            DataType::Int32 => {
                self.output.get_or_insert(NumericOutput::Int64);
                let values = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 data type");
                if values.is_valid(row) {
                    self.sum_i64 += i64::from(values.value(row));
                    self.count += 1;
                }
                Ok(())
            }
            DataType::Int64 => {
                self.output.get_or_insert(NumericOutput::Int64);
                let values = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 data type");
                if values.is_valid(row) {
                    self.sum_i64 += values.value(row);
                    self.count += 1;
                }
                Ok(())
            }
            DataType::Float64 => {
                self.output.get_or_insert(NumericOutput::Float64);
                let values = column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 data type");
                if values.is_valid(row) {
                    self.sum_f64 += values.value(row);
                    self.count += 1;
                }
                Ok(())
            }
            data_type => unsupported_aggregate_type(expr, data_type),
        }
    }

    fn sum_value(&self) -> AggregateValue {
        match self.output {
            Some(NumericOutput::Int64) if self.count > 0 => {
                AggregateValue::Int64(Some(self.sum_i64))
            }
            Some(NumericOutput::Float64) if self.count > 0 => {
                AggregateValue::Float64(Some(self.sum_f64))
            }
            Some(NumericOutput::Int64) | None => AggregateValue::Int64(None),
            Some(NumericOutput::Float64) => AggregateValue::Float64(None),
        }
    }

    fn avg_value(&self) -> AggregateValue {
        match self.output {
            Some(NumericOutput::Int64) if self.count > 0 => {
                AggregateValue::Float64(Some(self.sum_i64 as f64 / self.count as f64))
            }
            Some(NumericOutput::Float64) if self.count > 0 => {
                AggregateValue::Float64(Some(self.sum_f64 / self.count as f64))
            }
            _ => AggregateValue::Float64(None),
        }
    }
}

#[derive(Default)]
struct MinMaxState {
    value: Option<AggregateValue>,
}

impl MinMaxState {
    fn update_min(&mut self, column: &ArrayRef, expr: &AggregateExpr) -> Result<()> {
        self.update(column, expr, |candidate, current| candidate < current)
    }

    fn update_max(&mut self, column: &ArrayRef, expr: &AggregateExpr) -> Result<()> {
        self.update(column, expr, |candidate, current| candidate > current)
    }

    fn update_min_row(
        &mut self,
        column: &ArrayRef,
        row: usize,
        expr: &AggregateExpr,
    ) -> Result<()> {
        self.update_row(column, row, expr, |candidate, current| candidate < current)
    }

    fn update_max_row(
        &mut self,
        column: &ArrayRef,
        row: usize,
        expr: &AggregateExpr,
    ) -> Result<()> {
        self.update_row(column, row, expr, |candidate, current| candidate > current)
    }

    fn update(
        &mut self,
        column: &ArrayRef,
        expr: &AggregateExpr,
        replace: impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) -> Result<()> {
        match column.data_type() {
            DataType::Int32 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min(values)
                } else {
                    max(values)
                } {
                    self.update_int64(i64::from(value), &replace);
                }
                Ok(())
            }
            DataType::Int64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min(values)
                } else {
                    max(values)
                } {
                    self.update_int64(value, &replace);
                }
                Ok(())
            }
            DataType::Float64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min(values)
                } else {
                    max(values)
                } {
                    self.update_f64(value, &replace);
                }
                Ok(())
            }
            DataType::Date32 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Date32 data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min(values)
                } else {
                    max(values)
                } {
                    self.update_date32(value, &replace);
                }
                Ok(())
            }
            DataType::Date64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Date64Array>()
                    .expect("Date64 data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min(values)
                } else {
                    max(values)
                } {
                    self.update_date64(value, &replace);
                }
                Ok(())
            }
            DataType::Timestamp(TimeUnit::Millisecond, timezone) => {
                let values = column
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .expect("TimestampMillisecond data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min(values)
                } else {
                    max(values)
                } {
                    self.update_timestamp_millisecond(
                        value,
                        timezone.as_ref().map(ToString::to_string),
                        &replace,
                    );
                }
                Ok(())
            }
            DataType::Utf8 => {
                let values = column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 data type");
                if let Some(value) = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal)
                {
                    min_string(values)
                } else {
                    max_string(values)
                } {
                    self.update_utf8(value, &replace);
                }
                Ok(())
            }
            data_type => unsupported_aggregate_type(expr, data_type),
        }
    }

    fn update_row(
        &mut self,
        column: &ArrayRef,
        row: usize,
        expr: &AggregateExpr,
        replace: impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) -> Result<()> {
        if column.is_null(row) {
            return Ok(());
        }
        match column.data_type() {
            DataType::Int32 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 data type");
                self.update_int64(i64::from(values.value(row)), &replace);
                Ok(())
            }
            DataType::Int64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 data type");
                self.update_int64(values.value(row), &replace);
                Ok(())
            }
            DataType::Float64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 data type");
                self.update_f64(values.value(row), &replace);
                Ok(())
            }
            DataType::Date32 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Date32 data type");
                self.update_date32(values.value(row), &replace);
                Ok(())
            }
            DataType::Date64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Date64Array>()
                    .expect("Date64 data type");
                self.update_date64(values.value(row), &replace);
                Ok(())
            }
            DataType::Timestamp(TimeUnit::Millisecond, timezone) => {
                let values = column
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .expect("TimestampMillisecond data type");
                self.update_timestamp_millisecond(
                    values.value(row),
                    timezone.as_ref().map(ToString::to_string),
                    &replace,
                );
                Ok(())
            }
            DataType::Utf8 => {
                let values = column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 data type");
                self.update_utf8(values.value(row), &replace);
                Ok(())
            }
            data_type => unsupported_aggregate_type(expr, data_type),
        }
    }

    fn update_int64(
        &mut self,
        candidate: i64,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        match &self.value {
            Some(AggregateValue::Int64(Some(current)))
                if !replace(candidate.cmp(current), std::cmp::Ordering::Equal) => {}
            _ => self.value = Some(AggregateValue::Int64(Some(candidate))),
        }
    }

    fn update_f64(
        &mut self,
        candidate: f64,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        let Some(ordering) = candidate.partial_cmp(&candidate) else {
            return;
        };
        match &self.value {
            Some(AggregateValue::Float64(Some(current))) => {
                if let Some(candidate_ordering) = candidate.partial_cmp(current)
                    && replace(candidate_ordering, ordering)
                {
                    self.value = Some(AggregateValue::Float64(Some(candidate)));
                }
            }
            _ => self.value = Some(AggregateValue::Float64(Some(candidate))),
        }
    }

    fn update_date32(
        &mut self,
        candidate: i32,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        match &self.value {
            Some(AggregateValue::Date32(Some(current)))
                if !replace(candidate.cmp(current), std::cmp::Ordering::Equal) => {}
            _ => self.value = Some(AggregateValue::Date32(Some(candidate))),
        }
    }

    fn update_date64(
        &mut self,
        candidate: i64,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        match &self.value {
            Some(AggregateValue::Date64(Some(current)))
                if !replace(candidate.cmp(current), std::cmp::Ordering::Equal) => {}
            _ => self.value = Some(AggregateValue::Date64(Some(candidate))),
        }
    }

    fn update_timestamp_millisecond(
        &mut self,
        candidate: i64,
        timezone: Option<String>,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        match &self.value {
            Some(AggregateValue::TimestampMillisecond(Some(current), _))
                if !replace(candidate.cmp(current), std::cmp::Ordering::Equal) => {}
            _ => {
                self.value = Some(AggregateValue::TimestampMillisecond(
                    Some(candidate),
                    timezone,
                ))
            }
        }
    }

    fn update_utf8(
        &mut self,
        candidate: &str,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        match &self.value {
            Some(AggregateValue::Utf8(Some(current)))
                if !replace(candidate.cmp(current), std::cmp::Ordering::Equal) => {}
            _ => self.value = Some(AggregateValue::Utf8(Some(candidate.to_string()))),
        }
    }

    fn value(self) -> AggregateValue {
        self.value.unwrap_or(AggregateValue::Int64(None))
    }
}

fn aggregate_column<'a>(batch: &'a RecordBatch, expr: &AggregateExpr) -> Result<&'a ArrayRef> {
    let Some(column) = expr.referenced_column() else {
        return Err(DodamError::InvalidAggregate(expr.to_string()));
    };
    Ok(batch.column(column_index(batch, column)?))
}

fn unsupported_aggregate_type<T>(expr: &AggregateExpr, data_type: &DataType) -> Result<T> {
    let column = expr.referenced_column().unwrap_or("*").to_string();
    Err(DodamError::UnsupportedAggregateType {
        function: aggregate_function_name(expr).to_string(),
        column,
        data_type: data_type.clone(),
    })
}

fn aggregate_function_name(expr: &AggregateExpr) -> &'static str {
    match expr {
        AggregateExpr::CountStar | AggregateExpr::Count(_) => "count",
        AggregateExpr::Sum(_) => "sum",
        AggregateExpr::Avg(_) => "avg",
        AggregateExpr::Min(_) => "min",
        AggregateExpr::Max(_) => "max",
    }
}
