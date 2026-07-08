use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::sync::{Arc, mpsc};
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, StringArray, TimestampMillisecondArray, UInt64Array,
};
use arrow::compute::kernels::aggregate::{max, max_string, min, min_string, sum};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow_row::{OwnedRow, RowConverter, SortField};

use crate::error::{DodamError, Result};
use crate::execution::logical::{
    AggregateExpr, AggregateMetrics, AggregateResult, AggregateValue, GroupAggregateResult,
    GroupValue,
};
use crate::execution::metrics::SendableBatchStream;
use crate::execution::physical::column_index;
use crate::hash::FastHashMap as AggregateHashMap;

const SMALL_GROUP_LINEAR_LIMIT: usize = 8;
const TWO_UTF8_SMALL_GROUP_LIMIT: usize = 8;

fn small_group_linear_limit() -> usize {
    std::env::var("DODAM_SMALL_GROUP_LINEAR_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(SMALL_GROUP_LINEAR_LIMIT)
}

fn two_utf8_small_group_limit() -> usize {
    std::env::var("DODAM_TWO_UTF8_SMALL_GROUP_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(TWO_UTF8_SMALL_GROUP_LIMIT)
}

pub fn collect_aggregates(
    mut stream: SendableBatchStream,
    fragments: usize,
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let aggregate_started = Instant::now();
    let mut state = GlobalAggregateState::new(aggregates);
    let mut metrics = AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    };

    let (sender, receiver) = mpsc::channel();
    let mut pending_batches = 0_usize;
    for batch in stream.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }

        metrics.batches += 1;
        metrics.rows += batch.num_rows();
        let sender = sender.clone();
        let aggregates = aggregates.to_vec();
        pending_batches += 1;
        rayon::spawn(move || {
            let _ = sender.send(GlobalAggregateState::from_batch(batch, &aggregates));
        });
    }
    drop(sender);
    let merge_started = Instant::now();
    for _ in 0..pending_batches {
        let partial = receiver
            .recv()
            .map_err(|_| DodamError::InvalidAggregate("aggregate worker stopped".to_string()))??;
        state.merge(partial);
    }
    metrics.aggregate_merge_nanos = elapsed_nanos(merge_started);

    metrics.values = state.finish()?;
    metrics.aggregate_nanos = elapsed_nanos(aggregate_started);
    Ok(metrics)
}

struct GlobalAggregateState {
    shared_numeric_states: Vec<(String, NumericState)>,
    accumulators: Vec<GlobalAggregateSlot>,
}

impl GlobalAggregateState {
    fn new(aggregates: &[AggregateExpr]) -> Self {
        let shared_sum_avg_columns = shared_sum_avg_columns(aggregates);
        let shared_numeric_states = shared_sum_avg_columns
            .iter()
            .map(|column| (column.clone(), NumericState::default()))
            .collect::<Vec<_>>();
        let accumulators = aggregates
            .iter()
            .cloned()
            .map(|aggregate| GlobalAggregateSlot::new(aggregate, &shared_sum_avg_columns))
            .collect::<Vec<_>>();
        Self {
            shared_numeric_states,
            accumulators,
        }
    }

    fn from_batch(batch: RecordBatch, aggregates: &[AggregateExpr]) -> Result<Self> {
        let mut state = Self::new(aggregates);
        for (column, numeric_state) in &mut state.shared_numeric_states {
            numeric_state.update(
                batch.column(column_index(&batch, column)?),
                &AggregateExpr::Sum(column.clone()),
            )?;
        }
        for accumulator in &mut state.accumulators {
            accumulator.update(&batch)?;
        }
        Ok(state)
    }

    fn merge(&mut self, other: Self) {
        for ((_, state), (_, other_state)) in self
            .shared_numeric_states
            .iter_mut()
            .zip(other.shared_numeric_states)
        {
            state.merge(other_state);
        }
        for (accumulator, other_accumulator) in self.accumulators.iter_mut().zip(other.accumulators)
        {
            accumulator.merge(other_accumulator);
        }
    }

    fn finish(self) -> Result<Vec<AggregateResult>> {
        self.accumulators
            .into_iter()
            .map(|accumulator| accumulator.finish(&self.shared_numeric_states))
            .collect::<Result<Vec<_>>>()
    }
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

    fn merge(&mut self, other: Self) {
        match (self, other) {
            (Self::Accumulator(accumulator), Self::Accumulator(other)) => accumulator.merge(other),
            (Self::SharedSum { .. } | Self::SharedAvg { .. }, _) => {}
            _ => {}
        }
    }
}

pub fn collect_grouped_aggregates(
    stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let aggregate_started = Instant::now();
    let mut metrics = if can_use_single_key_count_sum_min_max_path(group_by, aggregates) {
        collect_single_key_count_sum_min_max_groups(stream, fragments, group_by, aggregates)?
    } else if can_use_single_key_count_sum_path(group_by, aggregates) {
        collect_single_key_count_sum_groups(stream, fragments, group_by, aggregates)?
    } else if can_use_single_key_fast_path(group_by, aggregates) {
        collect_single_key_groups(stream, fragments, group_by, aggregates)?
    } else if can_use_two_utf8_key_fast_path(group_by, aggregates) {
        collect_two_utf8_key_groups(stream, fragments, group_by, aggregates)?
    } else if can_use_two_key_count_sum_path(group_by, aggregates) {
        collect_two_key_count_sum_groups(stream, fragments, group_by, aggregates)?
    } else if can_use_three_key_count_sum_path(group_by, aggregates) {
        collect_three_key_count_sum_groups(stream, fragments, group_by, aggregates)?
    } else if can_use_two_key_sum_path(group_by, aggregates) {
        collect_two_key_sum_groups(stream, fragments, group_by, aggregates)?
    } else {
        collect_grouped_aggregates_generic(stream, fragments, group_by, aggregates, None, None)?
    };
    metrics.aggregate_nanos = elapsed_nanos(aggregate_started);
    Ok(metrics)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupKeyExpr {
    Column(String),
    CoalesceLiteral {
        column: String,
        fallback: GroupKeyLiteral,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupKeyLiteral {
    Null,
    Boolean(bool),
    Int64(i64),
    Float64(u64),
    Utf8(String),
}

pub fn collect_grouped_aggregates_with_key_exprs(
    stream: SendableBatchStream,
    fragments: usize,
    group_keys: &[GroupKeyExpr],
    aggregates: &[AggregateExpr],
) -> Result<Option<AggregateMetrics>> {
    if let Some(plan) = CoalesceKeyCountSumPlan::new(group_keys, aggregates) {
        return collect_coalesce_key_count_sum_groups(stream, fragments, aggregates, plan)
            .map(Some);
    }
    Ok(None)
}

struct CoalesceKeyCountSumPlan {
    leading_keys: Vec<String>,
    coalesce_column: String,
    fallback: String,
    sum_column: String,
}

impl CoalesceKeyCountSumPlan {
    fn new(group_keys: &[GroupKeyExpr], aggregates: &[AggregateExpr]) -> Option<Self> {
        if !(group_keys.len() == 2 || group_keys.len() == 3) {
            return None;
        }
        let [AggregateExpr::CountStar, AggregateExpr::Sum(sum_column)] = aggregates else {
            return None;
        };
        let GroupKeyExpr::CoalesceLiteral { column, fallback } = group_keys.last()? else {
            return None;
        };
        let GroupKeyLiteral::Utf8(fallback) = fallback else {
            return None;
        };
        let mut leading_keys = Vec::with_capacity(group_keys.len() - 1);
        for key in &group_keys[..group_keys.len() - 1] {
            let GroupKeyExpr::Column(column) = key else {
                return None;
            };
            leading_keys.push(column.clone());
        }
        Some(Self {
            leading_keys,
            coalesce_column: column.clone(),
            fallback: fallback.clone(),
            sum_column: sum_column.clone(),
        })
    }
}

fn collect_coalesce_key_count_sum_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    aggregates: &[AggregateExpr],
    plan: CoalesceKeyCountSumPlan,
) -> Result<AggregateMetrics> {
    let mut bound_plan = None;
    let mut groups = None;
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

        let bound = match &bound_plan {
            Some(bound) => bound,
            None => {
                bound_plan = Some(plan.bind(&batch)?);
                bound_plan.as_ref().expect("bound coalesce aggregate plan")
            }
        };
        let first = batch.column(bound.first);
        let second = bound.second.map(|index| batch.column(index));
        let coalesce = batch.column(bound.coalesce);
        let sum = batch.column(bound.sum);
        let Some(reader) = CoalesceKeyCountSumReader::new(first, second, coalesce, sum) else {
            return Err(DodamError::UnsupportedSql(
                "group expression view aggregate requires integer/date leading keys, coalesce(utf8,literal), and integer sum inputs"
                    .to_string(),
            ));
        };
        let groups = groups.get_or_insert_with(|| {
            CoalesceKeyCountSumGroups::new(
                plan.leading_keys.len(),
                reader.dense_leading_range(batch.num_rows()),
            )
        });
        reader.update_groups(groups, batch.num_rows(), plan.fallback.as_str());
    }
    metrics.groups = groups
        .map(|groups| groups.finish(aggregates))
        .unwrap_or_default();
    Ok(metrics)
}

struct BoundCoalesceKeyCountSumPlan {
    first: usize,
    second: Option<usize>,
    coalesce: usize,
    sum: usize,
}

impl CoalesceKeyCountSumPlan {
    fn bind(&self, batch: &RecordBatch) -> Result<BoundCoalesceKeyCountSumPlan> {
        Ok(BoundCoalesceKeyCountSumPlan {
            first: column_index(batch, &self.leading_keys[0])?,
            second: if self.leading_keys.len() == 2 {
                Some(column_index(batch, &self.leading_keys[1])?)
            } else {
                None
            },
            coalesce: column_index(batch, &self.coalesce_column)?,
            sum: column_index(batch, &self.sum_column)?,
        })
    }
}

enum CoalesceKeyCountSumReader<'a> {
    TwoInt32 {
        first: &'a Int32Array,
        coalesce: &'a StringArray,
        sum: Int64LikeArray<'a>,
    },
    TwoInt64 {
        first: &'a Int64Array,
        coalesce: &'a StringArray,
        sum: Int64LikeArray<'a>,
    },
    ThreeInt32Date {
        first: &'a Int32Array,
        second: &'a Date32Array,
        coalesce: &'a StringArray,
        sum: Int64LikeArray<'a>,
    },
    ThreeInt64Date {
        first: &'a Int64Array,
        second: &'a Date32Array,
        coalesce: &'a StringArray,
        sum: Int64LikeArray<'a>,
    },
}

impl<'a> CoalesceKeyCountSumReader<'a> {
    fn new(
        first: &'a ArrayRef,
        second: Option<&'a ArrayRef>,
        coalesce: &'a ArrayRef,
        sum: &'a ArrayRef,
    ) -> Option<Self> {
        let coalesce = coalesce.as_any().downcast_ref::<StringArray>()?;
        let sum = Int64LikeArray::new(sum)?;
        match (first.data_type(), second.map(|array| array.data_type())) {
            (DataType::Int32, None) => Some(Self::TwoInt32 {
                first: first.as_any().downcast_ref::<Int32Array>()?,
                coalesce,
                sum,
            }),
            (DataType::Int64, None) => Some(Self::TwoInt64 {
                first: first.as_any().downcast_ref::<Int64Array>()?,
                coalesce,
                sum,
            }),
            (DataType::Int32, Some(DataType::Date32)) => Some(Self::ThreeInt32Date {
                first: first.as_any().downcast_ref::<Int32Array>()?,
                second: second?.as_any().downcast_ref::<Date32Array>()?,
                coalesce,
                sum,
            }),
            (DataType::Int64, Some(DataType::Date32)) => Some(Self::ThreeInt64Date {
                first: first.as_any().downcast_ref::<Int64Array>()?,
                second: second?.as_any().downcast_ref::<Date32Array>()?,
                coalesce,
                sum,
            }),
            _ => None,
        }
    }

    fn key(&self, row: usize, fallback: &'a str) -> CoalesceKeyBorrowed<'a> {
        match self {
            Self::TwoInt32 {
                first, coalesce, ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| i64::from(first.value(row))),
                second: None,
                third: coalesce_key(coalesce, row, fallback),
            },
            Self::TwoInt64 {
                first, coalesce, ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| first.value(row)),
                second: None,
                third: coalesce_key(coalesce, row, fallback),
            },
            Self::ThreeInt32Date {
                first,
                second,
                coalesce,
                ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| i64::from(first.value(row))),
                second: second.is_valid(row).then(|| second.value(row)),
                third: coalesce_key(coalesce, row, fallback),
            },
            Self::ThreeInt64Date {
                first,
                second,
                coalesce,
                ..
            } => CoalesceKeyBorrowed {
                first: first.is_valid(row).then(|| first.value(row)),
                second: second.is_valid(row).then(|| second.value(row)),
                third: coalesce_key(coalesce, row, fallback),
            },
        }
    }

    fn sum_value(&self, row: usize) -> Option<i64> {
        match self {
            Self::TwoInt32 { sum, .. }
            | Self::TwoInt64 { sum, .. }
            | Self::ThreeInt32Date { sum, .. }
            | Self::ThreeInt64Date { sum, .. } => sum.value(row),
        }
    }

    fn update_groups(
        &self,
        groups: &mut CoalesceKeyCountSumGroups,
        row_count: usize,
        fallback: &'a str,
    ) {
        match self {
            Self::ThreeInt32Date {
                first,
                second,
                coalesce,
                sum,
            } if first.null_count() == 0 && second.null_count() == 0 => {
                for row in 0..row_count {
                    groups.update_three_non_null(
                        i64::from(first.value(row)),
                        second.value(row),
                        coalesce_key(coalesce, row, fallback).expect("coalesce key is non-null"),
                        sum.value(row),
                    );
                }
            }
            Self::ThreeInt64Date {
                first,
                second,
                coalesce,
                sum,
            } if first.null_count() == 0 && second.null_count() == 0 => {
                for row in 0..row_count {
                    groups.update_three_non_null(
                        first.value(row),
                        second.value(row),
                        coalesce_key(coalesce, row, fallback).expect("coalesce key is non-null"),
                        sum.value(row),
                    );
                }
            }
            _ => {
                for row in 0..row_count {
                    groups.update(self.key(row, fallback), self.sum_value(row));
                }
            }
        }
    }

    fn dense_leading_range(&self, row_count: usize) -> Option<DenseLeadingRange> {
        const MAX_DENSE_LEADING_SLOTS: usize = 4096;
        let (first_min, first_max, second_min, second_max) = match self {
            Self::ThreeInt32Date { first, second, .. } => {
                let (first_min, first_max) = int32_non_null_min_max_as_i64(first, row_count)?;
                let (second_min, second_max) = date32_non_null_min_max(second, row_count)?;
                (first_min, first_max, second_min, second_max)
            }
            Self::ThreeInt64Date { first, second, .. } => {
                let (first_min, first_max) = int64_non_null_min_max(first, row_count)?;
                let (second_min, second_max) = date32_non_null_min_max(second, row_count)?;
                (first_min, first_max, second_min, second_max)
            }
            _ => return None,
        };
        let first_len = usize::try_from(first_max.checked_sub(first_min)? + 1).ok()?;
        let second_len = usize::try_from(second_max.checked_sub(second_min)? + 1).ok()?;
        if first_len == 0
            || second_len == 0
            || first_len.checked_mul(second_len)? > MAX_DENSE_LEADING_SLOTS
        {
            return None;
        }
        Some(DenseLeadingRange {
            first_min,
            first_len,
            second_min,
            second_len,
        })
    }
}

fn int32_non_null_min_max_as_i64(values: &Int32Array, row_count: usize) -> Option<(i64, i64)> {
    int32_non_null_min_max(values, row_count).map(|(min, max)| (i64::from(min), i64::from(max)))
}

fn int32_non_null_min_max(values: &Int32Array, row_count: usize) -> Option<(i32, i32)> {
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for row in 0..row_count {
        if values.is_null(row) {
            return None;
        }
        let value = values.value(row);
        min = min.min(value);
        max = max.max(value);
    }
    Some((min, max))
}

fn int64_non_null_min_max(values: &Int64Array, row_count: usize) -> Option<(i64, i64)> {
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for row in 0..row_count {
        if values.is_null(row) {
            return None;
        }
        let value = values.value(row);
        min = min.min(value);
        max = max.max(value);
    }
    Some((min, max))
}

fn date32_non_null_min_max(values: &Date32Array, row_count: usize) -> Option<(i32, i32)> {
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for row in 0..row_count {
        if values.is_null(row) {
            return None;
        }
        let value = values.value(row);
        min = min.min(value);
        max = max.max(value);
    }
    Some((min, max))
}

fn coalesce_key<'a>(values: &'a StringArray, row: usize, fallback: &'a str) -> Option<&'a str> {
    if values.is_valid(row) {
        Some(values.value(row))
    } else {
        Some(fallback)
    }
}

struct CoalesceKeyBorrowed<'a> {
    first: Option<i64>,
    second: Option<i32>,
    third: Option<&'a str>,
}

struct CoalesceKeyCountSumGroups {
    key_len: usize,
    index: CoalesceLeadingIndex,
    third_string_ids: AggregateHashMap<String, u32>,
    third_strings: Vec<String>,
    groups: Vec<CoalesceKeyCountSumGroup>,
}

impl CoalesceKeyCountSumGroups {
    fn new(leading_key_count: usize, dense_range: Option<DenseLeadingRange>) -> Self {
        Self {
            key_len: leading_key_count + 1,
            index: CoalesceLeadingIndex::new(leading_key_count, dense_range),
            third_string_ids: AggregateHashMap::default(),
            third_strings: Vec::new(),
            groups: Vec::new(),
        }
    }

    fn update(&mut self, key: CoalesceKeyBorrowed<'_>, sum: Option<i64>) {
        let string_id = match key.third {
            Some(value) => Some(
                if let Some(string_id) = self.third_string_ids.get(value).copied() {
                    string_id
                } else {
                    let string_id =
                        u32::try_from(self.third_strings.len()).expect("too many string groups");
                    self.third_string_ids.insert(value.to_string(), string_id);
                    self.third_strings.push(value.to_string());
                    string_id
                },
            ),
            None => None,
        };
        let third_index = self.index.third_groups(key.first, key.second);
        let group_id = match string_id {
            Some(string_id) => {
                if let Some(group_id) = third_index.non_null.get(string_id) {
                    group_id
                } else {
                    let group_id = self.groups.len();
                    third_index.non_null.insert(string_id, group_id);
                    self.groups
                        .push(CoalesceKeyCountSumGroup::new(coalesce_group_values(
                            self.key_len,
                            key.first,
                            key.second,
                            Some(self.third_strings[string_id as usize].clone()),
                        )));
                    group_id
                }
            }
            None => {
                if let Some(group_id) = third_index.null_group {
                    group_id
                } else {
                    let group_id = self.groups.len();
                    third_index.null_group = Some(group_id);
                    self.groups
                        .push(CoalesceKeyCountSumGroup::new(coalesce_group_values(
                            self.key_len,
                            key.first,
                            key.second,
                            None,
                        )));
                    group_id
                }
            }
        };
        self.groups[group_id].update(sum);
    }

    fn update_three_non_null(&mut self, first: i64, second: i32, third: &str, sum: Option<i64>) {
        let string_id = if let Some(string_id) = self.third_string_ids.get(third).copied() {
            string_id
        } else {
            let string_id =
                u32::try_from(self.third_strings.len()).expect("too many string groups");
            self.third_string_ids.insert(third.to_string(), string_id);
            self.third_strings.push(third.to_string());
            string_id
        };
        let third_index = self.index.third_groups_three_non_null(first, second);
        let group_id = if let Some(group_id) = third_index.non_null.get(string_id) {
            group_id
        } else {
            let group_id = self.groups.len();
            third_index.non_null.insert(string_id, group_id);
            self.groups
                .push(CoalesceKeyCountSumGroup::new(coalesce_group_values(
                    self.key_len,
                    Some(first),
                    Some(second),
                    Some(self.third_strings[string_id as usize].clone()),
                )));
            group_id
        };
        self.groups[group_id].update(sum);
    }

    fn finish(self, aggregates: &[AggregateExpr]) -> Vec<GroupAggregateResult> {
        let Self {
            key_len,
            index,
            third_string_ids,
            third_strings,
            groups,
        } = self;
        let can_finish_ordered = key_len == 3
            && matches!(
                &index,
                CoalesceLeadingIndex::DenseThree { fallback, .. } if fallback.is_empty()
            );
        if can_finish_ordered && let CoalesceLeadingIndex::DenseThree { range, slots, .. } = index {
            return finish_dense_three_ordered_groups(
                range,
                slots,
                third_strings,
                groups,
                aggregates,
            );
        }
        let _ = third_string_ids;
        let _ = third_strings;
        let _ = index;
        let mut groups = groups
            .into_iter()
            .map(|group| group.finish(aggregates))
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
        groups
    }
}

fn finish_dense_three_ordered_groups(
    range: DenseLeadingRange,
    slots: Vec<CoalesceThirdGroups>,
    third_strings: Vec<String>,
    groups: Vec<CoalesceKeyCountSumGroup>,
    aggregates: &[AggregateExpr],
) -> Vec<GroupAggregateResult> {
    let mut groups = groups.into_iter().map(Some).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(groups.len());
    for first_offset in 0..range.first_len {
        for second_offset in 0..range.second_len {
            let slot = first_offset * range.second_len + second_offset;
            let third_groups = &slots[slot];
            if let Some(group_id) = third_groups.null_group
                && let Some(group) = groups[group_id].take()
            {
                output.push(group.finish(aggregates));
            }
            let mut non_null = third_groups.non_null.iter().collect::<Vec<_>>();
            non_null.sort_by(|(left, _), (right, _)| {
                third_strings[*left as usize].cmp(&third_strings[*right as usize])
            });
            for (_, group_id) in non_null {
                if let Some(group) = groups[group_id].take() {
                    output.push(group.finish(aggregates));
                }
            }
        }
    }
    output
}

#[derive(Clone, Copy)]
struct DenseLeadingRange {
    first_min: i64,
    first_len: usize,
    second_min: i32,
    second_len: usize,
}

enum CoalesceLeadingIndex {
    Hash(AggregateHashMap<Option<i64>, CoalesceSecondGroups>),
    DenseThree {
        range: DenseLeadingRange,
        slots: Vec<CoalesceThirdGroups>,
        fallback: AggregateHashMap<Option<i64>, CoalesceSecondGroups>,
    },
}

impl CoalesceLeadingIndex {
    fn new(leading_key_count: usize, dense_range: Option<DenseLeadingRange>) -> Self {
        if leading_key_count == 2
            && let Some(range) = dense_range
        {
            return Self::DenseThree {
                range,
                slots: (0..range.first_len * range.second_len)
                    .map(|_| CoalesceThirdGroups::default())
                    .collect(),
                fallback: AggregateHashMap::default(),
            };
        }
        Self::Hash(AggregateHashMap::default())
    }

    fn third_groups(
        &mut self,
        first: Option<i64>,
        second: Option<i32>,
    ) -> &mut CoalesceThirdGroups {
        match self {
            Self::Hash(index) => index
                .entry(first)
                .or_default()
                .index
                .entry(second)
                .or_default(),
            Self::DenseThree {
                range,
                slots,
                fallback,
            } => {
                if let (Some(first), Some(second)) = (first, second)
                    && let Some(slot) = dense_leading_slot(*range, first, second)
                {
                    return &mut slots[slot];
                }
                fallback
                    .entry(first)
                    .or_default()
                    .index
                    .entry(second)
                    .or_default()
            }
        }
    }

    fn third_groups_three_non_null(&mut self, first: i64, second: i32) -> &mut CoalesceThirdGroups {
        match self {
            Self::Hash(index) => index
                .entry(Some(first))
                .or_default()
                .index
                .entry(Some(second))
                .or_default(),
            Self::DenseThree {
                range,
                slots,
                fallback,
            } => {
                if let Some(slot) = dense_leading_slot(*range, first, second) {
                    return &mut slots[slot];
                }
                fallback
                    .entry(Some(first))
                    .or_default()
                    .index
                    .entry(Some(second))
                    .or_default()
            }
        }
    }
}

fn dense_leading_slot(range: DenseLeadingRange, first: i64, second: i32) -> Option<usize> {
    let first_offset = usize::try_from(first.checked_sub(range.first_min)?).ok()?;
    let second_offset = usize::try_from(second.checked_sub(range.second_min)?).ok()?;
    if first_offset < range.first_len && second_offset < range.second_len {
        Some(first_offset * range.second_len + second_offset)
    } else {
        None
    }
}

fn coalesce_group_values(
    key_len: usize,
    first: Option<i64>,
    second: Option<i32>,
    third: Option<String>,
) -> Vec<GroupValue> {
    if key_len == 2 {
        vec![GroupValue::Int64(first), GroupValue::Utf8(third)]
    } else {
        vec![
            GroupValue::Int64(first),
            GroupValue::Date32(second),
            GroupValue::Utf8(third),
        ]
    }
}

#[derive(Default)]
struct CoalesceSecondGroups {
    index: AggregateHashMap<Option<i32>, CoalesceThirdGroups>,
}

#[derive(Default)]
struct CoalesceThirdGroups {
    non_null: AdaptiveCopyGroupIndex<u32>,
    null_group: Option<usize>,
}

struct CoalesceKeyCountSumGroup {
    keys: Vec<GroupValue>,
    count: u64,
    sum: i64,
    sum_count: u64,
}

impl CoalesceKeyCountSumGroup {
    fn new(keys: Vec<GroupValue>) -> Self {
        Self {
            keys,
            count: 0,
            sum: 0,
            sum_count: 0,
        }
    }

    fn update(&mut self, sum: Option<i64>) {
        self.count += 1;
        if let Some(sum) = sum {
            self.sum += sum;
            self.sum_count += 1;
        }
    }

    fn finish(self, aggregates: &[AggregateExpr]) -> GroupAggregateResult {
        GroupAggregateResult {
            keys: self.keys,
            values: vec![
                AggregateResult {
                    expr: aggregates[0].clone(),
                    value: AggregateValue::Count(self.count),
                },
                AggregateResult {
                    expr: aggregates[1].clone(),
                    value: if self.sum_count == 0 {
                        AggregateValue::Int64(None)
                    } else {
                        AggregateValue::Int64(Some(self.sum))
                    },
                },
            ],
        }
    }
}

pub fn can_merge_partial_aggregates(aggregates: &[AggregateExpr]) -> bool {
    aggregates.iter().all(|aggregate| {
        !matches!(
            aggregate,
            AggregateExpr::Avg(_) | AggregateExpr::CountDistinct(_)
        )
    })
}

pub fn merge_partial_aggregate_metrics(
    partials: Vec<AggregateMetrics>,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    if !can_merge_partial_aggregates(aggregates) {
        return Err(DodamError::UnsupportedSql(
            "partial aggregate merge currently supports count/sum/min/max only".to_string(),
        ));
    }
    let mut metrics = AggregateMetrics {
        fragments,
        ..AggregateMetrics::default()
    };
    for partial in &partials {
        metrics.batches += partial.batches;
        metrics.rows += partial.rows;
    }
    if group_by.is_empty() {
        let merge_started = Instant::now();
        let mut values: Option<Vec<AggregateResult>> = None;
        for partial in partials {
            metrics.aggregate_nanos = metrics
                .aggregate_nanos
                .saturating_add(partial.aggregate_nanos);
            if partial.values.is_empty() {
                continue;
            }
            match &mut values {
                Some(values) => merge_aggregate_results(values, partial.values)?,
                None => values = Some(partial.values),
            }
        }
        metrics.aggregate_merge_nanos = elapsed_nanos(merge_started);
        metrics.values = values.unwrap_or_default();
        return Ok(metrics);
    }

    let merge_started = Instant::now();
    if can_use_single_key_count_sum_min_max_path(group_by, aggregates)
        && let Some(groups) = merge_single_key_count_sum_min_max_partials(&partials, aggregates)?
    {
        metrics.aggregate_nanos = partials.iter().fold(0_u64, |nanos, partial| {
            nanos.saturating_add(partial.aggregate_nanos)
        });
        metrics.aggregate_merge_nanos = elapsed_nanos(merge_started);
        metrics.groups = groups;
        return Ok(metrics);
    }
    let mut groups = AggregateHashMap::<Vec<GroupValue>, Vec<AggregateResult>>::default();
    for partial in partials {
        metrics.aggregate_nanos = metrics
            .aggregate_nanos
            .saturating_add(partial.aggregate_nanos);
        for group in partial.groups {
            match groups.entry(group.keys) {
                Entry::Occupied(mut entry) => {
                    merge_aggregate_results(entry.get_mut(), group.values)?
                }
                Entry::Vacant(entry) => {
                    entry.insert(group.values);
                }
            }
        }
    }
    let mut groups = groups
        .into_iter()
        .map(|(keys, values)| GroupAggregateResult { keys, values })
        .collect::<Vec<_>>();
    groups.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.aggregate_merge_nanos = elapsed_nanos(merge_started);
    metrics.groups = groups;
    Ok(metrics)
}

pub fn collect_partial_aggregate_batch(
    batch: RecordBatch,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Option<AggregateMetrics>> {
    if batch.num_rows() == 0 {
        return Ok(None);
    }
    let rows = batch.num_rows();
    let mut metrics = AggregateMetrics {
        fragments,
        batches: 1,
        rows,
        ..AggregateMetrics::default()
    };
    let aggregate_started = Instant::now();
    if group_by.is_empty() {
        metrics.values = GlobalAggregateState::from_batch(batch, aggregates)?.finish()?;
    } else {
        let groups = collect_grouped_aggregates_batch(batch, group_by, aggregates)?;
        metrics.groups = finish_group_map(groups)?;
    }
    metrics.aggregate_nanos = elapsed_nanos(aggregate_started);
    Ok(Some(metrics))
}

fn merge_single_key_count_sum_min_max_partials(
    partials: &[AggregateMetrics],
    aggregates: &[AggregateExpr],
) -> Result<Option<Vec<GroupAggregateResult>>> {
    let mut index = SingleKeyCountSumMinMaxIndex::Unset;
    let mut groups = Vec::<SingleKeyCountSumMinMaxGroup>::new();
    for partial in partials {
        for group in &partial.groups {
            let Some(key) = group.keys.first() else {
                return Ok(None);
            };
            let Some((precision, scale)) = decimal_min_result_type(group.values.get(2)) else {
                return Ok(None);
            };
            let group_id = match key {
                GroupValue::Int64(Some(key)) => {
                    index.ensure_type(&DataType::Int64);
                    let SingleKeyCountSumMinMaxIndex::Int64 { groups: index, .. } = &mut index
                    else {
                        return Ok(None);
                    };
                    count_sum_min_max_group_id_for_i64(index, *key, &mut groups, precision, scale)
                }
                GroupValue::Int64(None) => {
                    index.ensure_type(&DataType::Int64);
                    let SingleKeyCountSumMinMaxIndex::Int64 { null_group, .. } = &mut index else {
                        return Ok(None);
                    };
                    count_sum_min_max_group_id_for_null(
                        null_group,
                        &mut groups,
                        GroupValue::Int64(None),
                        precision,
                        scale,
                    )
                }
                GroupValue::UInt64(Some(key)) => {
                    index.ensure_type(&DataType::UInt64);
                    let SingleKeyCountSumMinMaxIndex::UInt64 { groups: index, .. } = &mut index
                    else {
                        return Ok(None);
                    };
                    count_sum_min_max_group_id_for_u64(index, *key, &mut groups, precision, scale)
                }
                GroupValue::UInt64(None) => {
                    index.ensure_type(&DataType::UInt64);
                    let SingleKeyCountSumMinMaxIndex::UInt64 { null_group, .. } = &mut index else {
                        return Ok(None);
                    };
                    count_sum_min_max_group_id_for_null(
                        null_group,
                        &mut groups,
                        GroupValue::UInt64(None),
                        precision,
                        scale,
                    )
                }
                _ => return Ok(None),
            };
            groups[group_id].merge_partial_values(&group.values)?;
        }
    }
    Ok(Some(finish_single_key_count_sum_min_max_groups(
        index, groups, aggregates,
    )))
}

fn decimal_min_result_type(value: Option<&AggregateResult>) -> Option<(u8, i8)> {
    let AggregateValue::Decimal128(_, precision, scale) = value?.value else {
        return None;
    };
    Some((precision, scale))
}

pub fn aggregate_metrics_to_batches(
    metrics: &AggregateMetrics,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<Vec<RecordBatch>> {
    if group_by.is_empty() {
        return aggregate_values_to_batch(&metrics.values).map(|batch| vec![batch]);
    }

    let mut fields = Vec::new();
    let mut columns = Vec::new();

    for (index, column) in group_by.iter().enumerate() {
        let values = metrics
            .groups
            .iter()
            .map(|group| group.keys.get(index))
            .collect::<Vec<_>>();
        let (field, array) = group_values_to_column(column, &values);
        fields.push(field);
        columns.push(array);
    }

    for (index, aggregate) in aggregates.iter().enumerate() {
        let values = metrics
            .groups
            .iter()
            .filter_map(|group| group.values.get(index))
            .map(|result| &result.value)
            .collect::<Vec<_>>();
        let (field, array) = aggregate_values_to_column(&aggregate.to_string(), &values);
        fields.push(field);
        columns.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(vec![RecordBatch::try_new(schema, columns)?])
}

fn aggregate_values_to_batch(values: &[AggregateResult]) -> Result<RecordBatch> {
    let mut fields = Vec::new();
    let mut columns = Vec::new();

    for value in values {
        let (field, array) = aggregate_values_to_column(&value.expr.to_string(), &[&value.value]);
        fields.push(field);
        columns.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn group_values_to_column(name: &str, values: &[Option<&GroupValue>]) -> (Field, ArrayRef) {
    let data_type = values
        .iter()
        .find_map(|value| match value {
            Some(GroupValue::Utf8(_)) => Some(DataType::Utf8),
            Some(GroupValue::Date64(_)) => Some(DataType::Date64),
            Some(GroupValue::Date32(_)) => Some(DataType::Date32),
            Some(GroupValue::Decimal128(_, precision, scale)) => {
                Some(DataType::Decimal128(*precision, *scale))
            }
            Some(GroupValue::UInt64(_)) => Some(DataType::UInt64),
            Some(GroupValue::Int64(_)) => Some(DataType::Int64),
            None => None,
        })
        .unwrap_or(DataType::Int64);

    match data_type {
        DataType::Utf8 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(GroupValue::Utf8(value)) => value.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Utf8, true),
                Arc::new(StringArray::from(values)),
            )
        }
        DataType::UInt64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(GroupValue::UInt64(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::UInt64, true),
                Arc::new(UInt64Array::from(values)),
            )
        }
        DataType::Decimal128(precision, scale) => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(GroupValue::Decimal128(value, _, _)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Decimal128(precision, scale), true),
                Arc::new(
                    Decimal128Array::from(values)
                        .with_precision_and_scale(precision, scale)
                        .expect("valid Decimal128 group type"),
                ),
            )
        }
        DataType::Date32 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(GroupValue::Date32(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date32, true),
                Arc::new(Date32Array::from(values)),
            )
        }
        DataType::Date64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(GroupValue::Date64(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date64, true),
                Arc::new(Date64Array::from(values)),
            )
        }
        _ => {
            let values = values
                .iter()
                .map(|value| match value {
                    Some(GroupValue::Int64(value)) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Int64, true),
                Arc::new(Int64Array::from(values)),
            )
        }
    }
}

fn aggregate_values_to_column(name: &str, values: &[&AggregateValue]) -> (Field, ArrayRef) {
    let data_type = values
        .iter()
        .map(|value| match value {
            AggregateValue::Count(_) => DataType::UInt64,
            AggregateValue::Int64(_) => DataType::Int64,
            AggregateValue::Float64(_) => DataType::Float64,
            AggregateValue::Date32(_) => DataType::Date32,
            AggregateValue::Date64(_) => DataType::Date64,
            AggregateValue::TimestampMillisecond(_, timezone) => {
                DataType::Timestamp(TimeUnit::Millisecond, timezone.clone().map(Into::into))
            }
            AggregateValue::Decimal128(_, precision, scale) => {
                DataType::Decimal128(*precision, *scale)
            }
            AggregateValue::Utf8(_) => DataType::Utf8,
        })
        .next()
        .unwrap_or(DataType::Int64);

    match data_type {
        DataType::UInt64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Count(value) => Some(*value),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::UInt64, true),
                Arc::new(UInt64Array::from(values)),
            )
        }
        DataType::Float64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Float64(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Float64, true),
                Arc::new(Float64Array::from(values)),
            )
        }
        DataType::Utf8 => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Utf8(value) => value.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Utf8, true),
                Arc::new(StringArray::from(values)),
            )
        }
        DataType::Date32 => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Date32(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date32, true),
                Arc::new(Date32Array::from(values)),
            )
        }
        DataType::Date64 => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Date64(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Date64, true),
                Arc::new(Date64Array::from(values)),
            )
        }
        DataType::Decimal128(precision, scale) => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Decimal128(value, _, _) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Decimal128(precision, scale), true),
                Arc::new(
                    Decimal128Array::from(values)
                        .with_precision_and_scale(precision, scale)
                        .expect("valid Decimal128 aggregate type"),
                ),
            )
        }
        DataType::Timestamp(TimeUnit::Millisecond, timezone) => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::TimestampMillisecond(value, _) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            let array = TimestampMillisecondArray::from(values);
            let array = if let Some(timezone) = timezone.as_ref() {
                array.with_timezone(timezone.clone())
            } else {
                array
            };
            (
                Field::new(
                    name,
                    DataType::Timestamp(TimeUnit::Millisecond, timezone),
                    true,
                ),
                Arc::new(array),
            )
        }
        _ => {
            let values = values
                .iter()
                .map(|value| match value {
                    AggregateValue::Int64(value) => *value,
                    _ => None,
                })
                .collect::<Vec<_>>();
            (
                Field::new(name, DataType::Int64, true),
                Arc::new(Int64Array::from(values)),
            )
        }
    }
}

fn elapsed_nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn merge_aggregate_results(
    values: &mut [AggregateResult],
    other: Vec<AggregateResult>,
) -> Result<()> {
    if values.len() != other.len() {
        return Err(DodamError::UnsupportedSql(
            "partial aggregate result shape mismatch".to_string(),
        ));
    }
    for (value, other) in values.iter_mut().zip(other) {
        if value.expr != other.expr {
            return Err(DodamError::UnsupportedSql(
                "partial aggregate expression mismatch".to_string(),
            ));
        }
        merge_aggregate_value(&value.expr, &mut value.value, other.value)?;
    }
    Ok(())
}

fn merge_aggregate_value(
    expr: &AggregateExpr,
    value: &mut AggregateValue,
    other: AggregateValue,
) -> Result<()> {
    match expr {
        AggregateExpr::CountStar | AggregateExpr::Count(_) => {
            let (AggregateValue::Count(value), AggregateValue::Count(other)) = (value, other)
            else {
                return Err(DodamError::UnsupportedSql(
                    "partial count aggregate type mismatch".to_string(),
                ));
            };
            *value += other;
        }
        AggregateExpr::Sum(_) => merge_sum_value(value, other)?,
        AggregateExpr::Min(_) | AggregateExpr::Max(_) => merge_min_max_value(expr, value, other),
        AggregateExpr::Avg(_) | AggregateExpr::CountDistinct(_) => {
            return Err(DodamError::UnsupportedSql(
                "partial aggregate merge currently supports count/sum/min/max only".to_string(),
            ));
        }
    }
    Ok(())
}

fn merge_min_max_value(expr: &AggregateExpr, value: &mut AggregateValue, other: AggregateValue) {
    if aggregate_value_is_null(&other) {
        return;
    }
    let mut state = if aggregate_value_is_null(value) {
        None
    } else {
        Some(value.clone())
    };
    update_min_max_value(expr, &mut state, other);
    if let Some(state) = state {
        *value = state;
    }
}

fn merge_sum_value(value: &mut AggregateValue, other: AggregateValue) -> Result<()> {
    match (value, other) {
        (AggregateValue::Int64(value), AggregateValue::Int64(other)) => {
            if let Some(other) = other {
                *value = Some(value.unwrap_or_default() + other);
            }
        }
        (AggregateValue::Float64(value), AggregateValue::Float64(other)) => {
            if let Some(other) = other {
                *value = Some(value.unwrap_or_default() + other);
            }
        }
        (
            AggregateValue::Decimal128(value, precision, scale),
            AggregateValue::Decimal128(other, _, _),
        ) => {
            if let Some(other) = other {
                *value = Some(value.unwrap_or_default() + other);
            }
            let _ = (precision, scale);
        }
        (_, other) => {
            if !aggregate_value_is_null(&other) {
                return Err(DodamError::UnsupportedSql(
                    "partial sum aggregate type mismatch".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn aggregate_value_is_null(value: &AggregateValue) -> bool {
    matches!(
        value,
        AggregateValue::Int64(None)
            | AggregateValue::Float64(None)
            | AggregateValue::Date32(None)
            | AggregateValue::Date64(None)
            | AggregateValue::TimestampMillisecond(None, _)
            | AggregateValue::Decimal128(None, _, _)
            | AggregateValue::Utf8(None)
    )
}

fn can_use_two_utf8_key_fast_path(group_by: &[String], aggregates: &[AggregateExpr]) -> bool {
    group_by.len() == 2
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

fn collect_two_utf8_key_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let mut group_index = TwoUtf8KeyGroupIndex::default();
    let mut groups = Vec::<TwoKeyGroup>::new();
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

        let first_key = batch.column(column_index(&batch, &group_by[0])?);
        let second_key = batch.column(column_index(&batch, &group_by[1])?);
        let (Some(first_key), Some(second_key)) = (
            first_key.as_any().downcast_ref::<StringArray>(),
            second_key.as_any().downcast_ref::<StringArray>(),
        ) else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        let aggregate_inputs = typed_fast_inputs(&batch, aggregates)?;

        for row in 0..batch.num_rows() {
            let first = first_key.is_valid(row).then(|| first_key.value(row));
            let second = second_key.is_valid(row).then(|| second_key.value(row));
            let group_id = group_index.group_id(first, second, &mut groups, &aggregate_inputs);
            let group = &mut groups[group_id];
            for (state, input) in group.states.iter_mut().zip(&aggregate_inputs) {
                state.update(input, row);
            }
        }
    }

    let mut group_results = groups
        .into_iter()
        .map(TwoKeyGroup::finish)
        .collect::<Vec<_>>();
    group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.groups = group_results;
    Ok(metrics)
}

enum TwoUtf8KeyGroupIndex {
    Small(Vec<TwoUtf8SmallGroup>),
    Hash(TwoUtf8HashGroupIndex),
}

impl Default for TwoUtf8KeyGroupIndex {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

struct TwoUtf8SmallGroup {
    first: Option<String>,
    second: Option<String>,
    group_id: usize,
}

impl TwoUtf8KeyGroupIndex {
    fn group_id(
        &mut self,
        first: Option<&str>,
        second: Option<&str>,
        groups_out: &mut Vec<TwoKeyGroup>,
        inputs: &[FastAggregateInput<'_>],
    ) -> usize {
        match self {
            Self::Small(groups) => {
                if let Some(group_id) = groups.iter().find_map(|group| {
                    (group.first.as_deref() == first && group.second.as_deref() == second)
                        .then_some(group.group_id)
                }) {
                    return group_id;
                }
                if groups.len() < two_utf8_small_group_limit() {
                    let group_id = push_two_utf8_group(
                        groups_out,
                        first.map(str::to_string),
                        second.map(str::to_string),
                        inputs,
                    );
                    groups.push(TwoUtf8SmallGroup {
                        first: first.map(str::to_string),
                        second: second.map(str::to_string),
                        group_id,
                    });
                    return group_id;
                }
                let mut hash = TwoUtf8HashGroupIndex::default();
                for group in groups.drain(..) {
                    hash.insert_existing(group.first, group.second, group.group_id);
                }
                let group_id = hash.group_id(first, second, groups_out, inputs);
                *self = Self::Hash(hash);
                group_id
            }
            Self::Hash(hash) => hash.group_id(first, second, groups_out, inputs),
        }
    }
}

#[derive(Default)]
struct TwoUtf8HashGroupIndex {
    groups: AggregateHashMap<String, AggregateHashMap<String, usize>>,
    null_first: AggregateHashMap<String, usize>,
    null_second: AggregateHashMap<String, usize>,
    null_both: Option<usize>,
}

impl TwoUtf8HashGroupIndex {
    fn insert_existing(&mut self, first: Option<String>, second: Option<String>, group_id: usize) {
        match (first, second) {
            (Some(first), Some(second)) => {
                self.groups
                    .entry(first)
                    .or_default()
                    .insert(second, group_id);
            }
            (None, Some(second)) => {
                self.null_first.insert(second, group_id);
            }
            (Some(first), None) => {
                self.null_second.insert(first, group_id);
            }
            (None, None) => {
                self.null_both = Some(group_id);
            }
        }
    }

    fn group_id(
        &mut self,
        first: Option<&str>,
        second: Option<&str>,
        groups_out: &mut Vec<TwoKeyGroup>,
        inputs: &[FastAggregateInput<'_>],
    ) -> usize {
        match (first, second) {
            (Some(first), Some(second)) => {
                if let Some(second_groups) = self.groups.get(first)
                    && let Some(group_id) = second_groups.get(second)
                {
                    return *group_id;
                }
                let group_id = groups_out.len();
                self.groups
                    .entry(first.to_string())
                    .or_default()
                    .insert(second.to_string(), group_id);
                groups_out.push(TwoKeyGroup::new(
                    vec![
                        GroupValue::Utf8(Some(first.to_string())),
                        GroupValue::Utf8(Some(second.to_string())),
                    ],
                    inputs,
                ));
                group_id
            }
            (None, Some(second)) => {
                if let Some(group_id) = self.null_first.get(second) {
                    return *group_id;
                }
                let group_id = groups_out.len();
                self.null_first.insert(second.to_string(), group_id);
                groups_out.push(TwoKeyGroup::new(
                    vec![
                        GroupValue::Utf8(None),
                        GroupValue::Utf8(Some(second.to_string())),
                    ],
                    inputs,
                ));
                group_id
            }
            (Some(first), None) => {
                if let Some(group_id) = self.null_second.get(first) {
                    return *group_id;
                }
                let group_id = groups_out.len();
                self.null_second.insert(first.to_string(), group_id);
                groups_out.push(TwoKeyGroup::new(
                    vec![
                        GroupValue::Utf8(Some(first.to_string())),
                        GroupValue::Utf8(None),
                    ],
                    inputs,
                ));
                group_id
            }
            (None, None) => {
                if let Some(group_id) = self.null_both {
                    return group_id;
                }
                let group_id = groups_out.len();
                self.null_both = Some(group_id);
                groups_out.push(TwoKeyGroup::new(
                    vec![GroupValue::Utf8(None), GroupValue::Utf8(None)],
                    inputs,
                ));
                group_id
            }
        }
    }
}

fn push_two_utf8_group(
    groups_out: &mut Vec<TwoKeyGroup>,
    first: Option<String>,
    second: Option<String>,
    inputs: &[FastAggregateInput<'_>],
) -> usize {
    let group_id = groups_out.len();
    groups_out.push(TwoKeyGroup::new(
        vec![GroupValue::Utf8(first), GroupValue::Utf8(second)],
        inputs,
    ));
    group_id
}

struct TwoKeyGroup {
    keys: Vec<GroupValue>,
    states: Vec<FastAggregateState>,
}

impl TwoKeyGroup {
    fn new(keys: Vec<GroupValue>, inputs: &[FastAggregateInput<'_>]) -> Self {
        Self {
            keys,
            states: inputs.iter().map(FastAggregateState::new).collect(),
        }
    }

    fn finish(self) -> GroupAggregateResult {
        GroupAggregateResult {
            keys: self.keys,
            values: self
                .states
                .into_iter()
                .map(FastAggregateState::finish)
                .collect::<Vec<_>>(),
        }
    }
}

fn can_use_two_key_sum_path(group_by: &[String], aggregates: &[AggregateExpr]) -> bool {
    group_by.len() == 2 && matches!(aggregates, [AggregateExpr::Sum(_)])
}

fn can_use_two_key_count_sum_path(group_by: &[String], aggregates: &[AggregateExpr]) -> bool {
    group_by.len() == 2
        && matches!(
            aggregates,
            [AggregateExpr::CountStar, AggregateExpr::Sum(_)]
        )
}

fn can_use_three_key_count_sum_path(group_by: &[String], aggregates: &[AggregateExpr]) -> bool {
    group_by.len() == 3
        && matches!(
            aggregates,
            [AggregateExpr::CountStar, AggregateExpr::Sum(_)]
        )
}

fn collect_three_key_count_sum_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let AggregateExpr::Sum(sum_column) = &aggregates[1] else {
        unreachable!("three-key count/sum fast path precondition");
    };
    let mut groups = Vec::new();
    let mut group_index = ThreeKeyCountSumIndex::default();
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

        let first = batch.column(column_index(&batch, &group_by[0])?);
        let second = batch.column(column_index(&batch, &group_by[1])?);
        let third = batch.column(column_index(&batch, &group_by[2])?);
        let Some(keys) = ThreeKeyCountSumReader::new(first, second, third) else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        let sum_values = batch.column(column_index(&batch, sum_column)?);
        let Some(sum_values) = Int64LikeArray::new(sum_values) else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };

        for row in 0..batch.num_rows() {
            let key = keys.key(row);
            let group_id = group_index.group_id(key, &mut groups);
            groups[group_id].update(&sum_values, row);
        }
    }

    let mut group_results = groups
        .into_iter()
        .map(|group| group.finish(aggregates))
        .collect::<Vec<_>>();
    group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.groups = group_results;
    Ok(metrics)
}

enum ThreeKeyCountSumReader<'a> {
    Int32DateUtf8(&'a Int32Array, &'a Date32Array, &'a StringArray),
    Int64DateUtf8(&'a Int64Array, &'a Date32Array, &'a StringArray),
}

impl<'a> ThreeKeyCountSumReader<'a> {
    fn new(first: &'a ArrayRef, second: &'a ArrayRef, third: &'a ArrayRef) -> Option<Self> {
        match (first.data_type(), second.data_type(), third.data_type()) {
            (DataType::Int32, DataType::Date32, DataType::Utf8) => Some(Self::Int32DateUtf8(
                first.as_any().downcast_ref::<Int32Array>()?,
                second.as_any().downcast_ref::<Date32Array>()?,
                third.as_any().downcast_ref::<StringArray>()?,
            )),
            (DataType::Int64, DataType::Date32, DataType::Utf8) => Some(Self::Int64DateUtf8(
                first.as_any().downcast_ref::<Int64Array>()?,
                second.as_any().downcast_ref::<Date32Array>()?,
                third.as_any().downcast_ref::<StringArray>()?,
            )),
            _ => None,
        }
    }

    fn key(&self, row: usize) -> ThreeKeyCountSumBorrowedKey<'_> {
        match self {
            Self::Int32DateUtf8(first, second, third) => ThreeKeyCountSumBorrowedKey {
                first: first.is_valid(row).then(|| i64::from(first.value(row))),
                second: second.is_valid(row).then(|| second.value(row)),
                third: third.is_valid(row).then(|| third.value(row)),
            },
            Self::Int64DateUtf8(first, second, third) => ThreeKeyCountSumBorrowedKey {
                first: first.is_valid(row).then(|| first.value(row)),
                second: second.is_valid(row).then(|| second.value(row)),
                third: third.is_valid(row).then(|| third.value(row)),
            },
        }
    }
}

struct ThreeKeyCountSumBorrowedKey<'a> {
    first: Option<i64>,
    second: Option<i32>,
    third: Option<&'a str>,
}

#[derive(Default)]
struct ThreeKeyCountSumIndex {
    first: AggregateHashMap<Option<i64>, ThreeKeySecondIndex>,
}

#[derive(Default)]
struct ThreeKeySecondIndex {
    second: AggregateHashMap<Option<i32>, Utf8SecondGroupIndex>,
}

impl ThreeKeyCountSumIndex {
    fn group_id(
        &mut self,
        key: ThreeKeyCountSumBorrowedKey<'_>,
        groups: &mut Vec<ThreeKeyCountSumGroup>,
    ) -> usize {
        let third = self
            .first
            .entry(key.first)
            .or_default()
            .second
            .entry(key.second)
            .or_default();
        if let Some(group_id) = third.lookup(key.third) {
            return group_id;
        }
        let group_id = groups.len();
        third.insert(key.third, group_id);
        groups.push(ThreeKeyCountSumGroup::new(vec![
            GroupValue::Int64(key.first),
            GroupValue::Date32(key.second),
            GroupValue::Utf8(key.third.map(str::to_string)),
        ]));
        group_id
    }
}

struct ThreeKeyCountSumGroup {
    keys: Vec<GroupValue>,
    count: u64,
    sum: i64,
    sum_count: u64,
}

impl ThreeKeyCountSumGroup {
    fn new(keys: Vec<GroupValue>) -> Self {
        Self {
            keys,
            count: 0,
            sum: 0,
            sum_count: 0,
        }
    }

    fn update(&mut self, sum_values: &Int64LikeArray<'_>, row: usize) {
        self.count += 1;
        if let Some(value) = sum_values.value(row) {
            self.sum += value;
            self.sum_count += 1;
        }
    }

    fn finish(self, aggregates: &[AggregateExpr]) -> GroupAggregateResult {
        GroupAggregateResult {
            keys: self.keys,
            values: vec![
                AggregateResult {
                    expr: aggregates[0].clone(),
                    value: AggregateValue::Count(self.count),
                },
                AggregateResult {
                    expr: aggregates[1].clone(),
                    value: if self.sum_count == 0 {
                        AggregateValue::Int64(None)
                    } else {
                        AggregateValue::Int64(Some(self.sum))
                    },
                },
            ],
        }
    }
}

fn collect_two_key_count_sum_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let AggregateExpr::Sum(sum_column) = &aggregates[1] else {
        unreachable!("two-key count/sum fast path precondition");
    };
    let sum_expr = aggregates[1].clone();
    let mut groups = Vec::new();
    let mut group_index = TwoKeyCountSumIndex::Uninitialized;
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

        let first_key = batch.column(column_index(&batch, &group_by[0])?);
        let second_key = batch.column(column_index(&batch, &group_by[1])?);
        let Some(key_reader) = TwoKeyCountSumReader::new(first_key, second_key) else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        let sum_column = batch.column(column_index(&batch, sum_column)?);
        let sum_input = CountSumValueInput::new(sum_column, &sum_expr)?;

        group_index.ensure_shape(&key_reader)?;
        for row in 0..batch.num_rows() {
            let key = key_reader.key(row);
            let group_id = group_index.group_id(key, &mut groups, &sum_input)?;
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

enum TwoKeyCountSumReader<'a> {
    Int32Utf8(&'a Int32Array, &'a StringArray),
    Int64Utf8(&'a Int64Array, &'a StringArray),
    Utf8Int32(&'a StringArray, &'a Int32Array),
    Utf8Int64(&'a StringArray, &'a Int64Array),
}

impl<'a> TwoKeyCountSumReader<'a> {
    fn new(first: &'a ArrayRef, second: &'a ArrayRef) -> Option<Self> {
        match (first.data_type(), second.data_type()) {
            (DataType::Int32, DataType::Utf8) => Some(Self::Int32Utf8(
                first.as_any().downcast_ref::<Int32Array>()?,
                second.as_any().downcast_ref::<StringArray>()?,
            )),
            (DataType::Int64, DataType::Utf8) => Some(Self::Int64Utf8(
                first.as_any().downcast_ref::<Int64Array>()?,
                second.as_any().downcast_ref::<StringArray>()?,
            )),
            (DataType::Utf8, DataType::Int32) => Some(Self::Utf8Int32(
                first.as_any().downcast_ref::<StringArray>()?,
                second.as_any().downcast_ref::<Int32Array>()?,
            )),
            (DataType::Utf8, DataType::Int64) => Some(Self::Utf8Int64(
                first.as_any().downcast_ref::<StringArray>()?,
                second.as_any().downcast_ref::<Int64Array>()?,
            )),
            _ => None,
        }
    }

    fn shape(&self) -> TwoKeyCountSumShape {
        match self {
            Self::Int32Utf8(_, _) | Self::Int64Utf8(_, _) => TwoKeyCountSumShape::IntUtf8,
            Self::Utf8Int32(_, _) | Self::Utf8Int64(_, _) => TwoKeyCountSumShape::Utf8Int,
        }
    }

    fn key(&self, row: usize) -> TwoKeyCountSumBorrowedKey<'_> {
        match self {
            Self::Int32Utf8(first, second) => TwoKeyCountSumBorrowedKey::IntUtf8(
                first.is_valid(row).then(|| i64::from(first.value(row))),
                second.is_valid(row).then(|| second.value(row)),
            ),
            Self::Int64Utf8(first, second) => TwoKeyCountSumBorrowedKey::IntUtf8(
                first.is_valid(row).then(|| first.value(row)),
                second.is_valid(row).then(|| second.value(row)),
            ),
            Self::Utf8Int32(first, second) => TwoKeyCountSumBorrowedKey::Utf8Int(
                first.is_valid(row).then(|| first.value(row)),
                second.is_valid(row).then(|| i64::from(second.value(row))),
            ),
            Self::Utf8Int64(first, second) => TwoKeyCountSumBorrowedKey::Utf8Int(
                first.is_valid(row).then(|| first.value(row)),
                second.is_valid(row).then(|| second.value(row)),
            ),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TwoKeyCountSumShape {
    IntUtf8,
    Utf8Int,
}

enum TwoKeyCountSumBorrowedKey<'a> {
    IntUtf8(Option<i64>, Option<&'a str>),
    Utf8Int(Option<&'a str>, Option<i64>),
}

enum TwoKeyCountSumIndex {
    Uninitialized,
    IntUtf8(AggregateHashMap<Option<i64>, Utf8SecondGroupIndex>),
    Utf8Int {
        first_groups: AggregateHashMap<String, IntSecondGroupIndex>,
        null_first: IntSecondGroupIndex,
    },
}

#[derive(Default)]
struct Utf8SecondGroupIndex {
    non_null: AggregateHashMap<String, usize>,
    null_group: Option<usize>,
}

#[derive(Default)]
struct IntSecondGroupIndex {
    groups: AggregateHashMap<Option<i64>, usize>,
}

impl TwoKeyCountSumIndex {
    fn ensure_shape(&mut self, reader: &TwoKeyCountSumReader<'_>) -> Result<()> {
        let shape = reader.shape();
        match (&self, shape) {
            (Self::Uninitialized, TwoKeyCountSumShape::IntUtf8) => {
                *self = Self::IntUtf8(AggregateHashMap::default());
            }
            (Self::Uninitialized, TwoKeyCountSumShape::Utf8Int) => {
                *self = Self::Utf8Int {
                    first_groups: AggregateHashMap::default(),
                    null_first: IntSecondGroupIndex::default(),
                };
            }
            (Self::IntUtf8(_), TwoKeyCountSumShape::IntUtf8)
            | (Self::Utf8Int { .. }, TwoKeyCountSumShape::Utf8Int) => {}
            _ => {
                return Err(DodamError::TypeMismatch(
                    "mixed two-key aggregate batches changed key type shape".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn group_id(
        &mut self,
        key: TwoKeyCountSumBorrowedKey<'_>,
        groups: &mut Vec<TwoKeyCountSumGroup>,
        sum_input: &CountSumValueInput<'_>,
    ) -> Result<usize> {
        match (self, key) {
            (Self::IntUtf8(index), TwoKeyCountSumBorrowedKey::IntUtf8(first, second)) => {
                let second_index = index.entry(first).or_default();
                if let Some(group_id) = second_index.lookup(second) {
                    return Ok(group_id);
                }
                let group_id = Self::push_group(
                    groups,
                    vec![
                        GroupValue::Int64(first),
                        GroupValue::Utf8(second.map(str::to_string)),
                    ],
                    sum_input,
                );
                second_index.insert(second, group_id);
                Ok(group_id)
            }
            (
                Self::Utf8Int {
                    first_groups,
                    null_first,
                },
                TwoKeyCountSumBorrowedKey::Utf8Int(first, second),
            ) => {
                let second_index = match first {
                    Some(first) => {
                        if !first_groups.contains_key(first) {
                            first_groups.insert(first.to_string(), IntSecondGroupIndex::default());
                        }
                        first_groups
                            .get_mut(first)
                            .expect("inserted utf8 first-key group")
                    }
                    None => null_first,
                };
                if let Some(group_id) = second_index.lookup(second) {
                    return Ok(group_id);
                }
                let group_id = Self::push_group(
                    groups,
                    vec![
                        GroupValue::Utf8(first.map(str::to_string)),
                        GroupValue::Int64(second),
                    ],
                    sum_input,
                );
                second_index.insert(second, group_id);
                Ok(group_id)
            }
            _ => Err(DodamError::TypeMismatch(
                "mixed two-key aggregate key shape mismatch".to_string(),
            )),
        }
    }

    fn push_group(
        groups: &mut Vec<TwoKeyCountSumGroup>,
        keys: Vec<GroupValue>,
        sum_input: &CountSumValueInput<'_>,
    ) -> usize {
        let group_id = groups.len();
        groups.push(TwoKeyCountSumGroup::new(keys, sum_input));
        group_id
    }
}

impl Utf8SecondGroupIndex {
    fn lookup(&self, value: Option<&str>) -> Option<usize> {
        match value {
            Some(value) => self.non_null.get(value).copied(),
            None => self.null_group,
        }
    }

    fn insert(&mut self, value: Option<&str>, group_id: usize) {
        match value {
            Some(value) => {
                self.non_null.insert(value.to_string(), group_id);
            }
            None => {
                self.null_group = Some(group_id);
            }
        }
    }
}

impl IntSecondGroupIndex {
    fn lookup(&self, value: Option<i64>) -> Option<usize> {
        self.groups.get(&value).copied()
    }

    fn insert(&mut self, value: Option<i64>, group_id: usize) {
        self.groups.insert(value, group_id);
    }
}

struct TwoKeyCountSumGroup {
    keys: Vec<GroupValue>,
    count: u64,
    sum_i64: i64,
    sum_f64: f64,
    sum_is_float: bool,
    sum_count: u64,
}

impl TwoKeyCountSumGroup {
    fn new(keys: Vec<GroupValue>, sum_input: &CountSumValueInput<'_>) -> Self {
        Self {
            keys,
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
            keys: self.keys,
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

fn collect_two_key_sum_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let AggregateExpr::Sum(sum_column) = &aggregates[0] else {
        unreachable!("two-key sum fast path precondition");
    };
    let sum_expr = aggregates[0].clone();
    let mut groups: AggregateHashMap<(Option<String>, Option<i64>), NumericState> =
        AggregateHashMap::default();
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

        let first_key = batch.column(column_index(&batch, &group_by[0])?);
        let second_key = batch.column(column_index(&batch, &group_by[1])?);
        let sum_values = batch.column(column_index(&batch, sum_column)?);
        let Some(first_key) = first_key.as_any().downcast_ref::<StringArray>() else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        if !matches!(second_key.data_type(), DataType::Int32 | DataType::Int64) {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        }
        let second_key = two_key_sum_key_column(second_key);
        let Some(sum_values) = two_key_sum_column(sum_values) else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };

        for row in 0..batch.num_rows() {
            let key = (
                first_key
                    .is_valid(row)
                    .then(|| first_key.value(row).to_string()),
                second_key.value(row),
            );
            sum_values.update_state(groups.entry(key).or_default(), row);
        }
    }

    let mut group_results = groups
        .into_iter()
        .map(|((first, second), state)| GroupAggregateResult {
            keys: vec![GroupValue::Utf8(first), GroupValue::Int64(second)],
            values: vec![AggregateResult {
                expr: sum_expr.clone(),
                value: state.sum_value(),
            }],
        })
        .collect::<Vec<_>>();
    group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
    metrics.groups = group_results;
    Ok(metrics)
}

enum TwoKeySumKeyColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
}

impl TwoKeySumKeyColumn<'_> {
    fn value(&self, row: usize) -> Option<i64> {
        match self {
            Self::Int32(values) => values.is_valid(row).then(|| i64::from(values.value(row))),
            Self::Int64(values) => values.is_valid(row).then(|| values.value(row)),
        }
    }
}

fn two_key_sum_key_column(column: &ArrayRef) -> TwoKeySumKeyColumn<'_> {
    match column.data_type() {
        DataType::Int32 => TwoKeySumKeyColumn::Int32(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type"),
        ),
        DataType::Int64 => TwoKeySumKeyColumn::Int64(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type"),
        ),
        _ => unreachable!("validated two-key sum key type"),
    }
}

enum TwoKeySumColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float64(&'a Float64Array),
    Decimal128(&'a Decimal128Array, f64),
}

impl TwoKeySumColumn<'_> {
    fn update_state(&self, state: &mut NumericState, row: usize) {
        match self {
            Self::Int32(values) => {
                if values.is_valid(row) {
                    state.add_i64(i64::from(values.value(row)));
                }
            }
            Self::Int64(values) => {
                if values.is_valid(row) {
                    state.add_i64(values.value(row));
                }
            }
            Self::Float64(values) => {
                if values.is_valid(row) {
                    state.add_f64(values.value(row));
                }
            }
            Self::Decimal128(values, scale) => {
                if values.is_valid(row) {
                    state.add_f64(values.value(row) as f64 / scale);
                }
            }
        }
    }
}

fn two_key_sum_column(column: &ArrayRef) -> Option<TwoKeySumColumn<'_>> {
    match column.data_type() {
        DataType::Int32 => Some(TwoKeySumColumn::Int32(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 data type"),
        )),
        DataType::Int64 => Some(TwoKeySumColumn::Int64(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 data type"),
        )),
        DataType::Float64 => Some(TwoKeySumColumn::Float64(
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 data type"),
        )),
        DataType::Decimal128(_, scale) => Some(TwoKeySumColumn::Decimal128(
            column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("Decimal128 data type"),
            decimal_scale_factor(*scale),
        )),
        _ => None,
    }
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
    let (sender, receiver) = mpsc::channel();
    let mut pending_batches = 0_usize;

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
        let sender = sender.clone();
        let group_by = group_by.to_vec();
        let aggregates = aggregates.to_vec();
        pending_batches += 1;
        rayon::spawn(move || {
            let _ = sender.send(collect_grouped_aggregates_batch(
                batch,
                &group_by,
                &aggregates,
            ));
        });
    }
    drop(sender);
    for _ in 0..pending_batches {
        let partial = receiver.recv().map_err(|_| {
            DodamError::InvalidAggregate("grouped aggregate worker stopped".to_string())
        })??;
        merge_group_maps(&mut groups, partial);
    }

    metrics.groups = finish_group_map(groups)?;
    Ok(metrics)
}

fn finish_group_map(
    groups: AggregateHashMap<OwnedRow, GroupState>,
) -> Result<Vec<GroupAggregateResult>> {
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
    Ok(group_results)
}

fn collect_grouped_aggregates_batch(
    batch: RecordBatch,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateHashMap<OwnedRow, GroupState>> {
    let mut groups: AggregateHashMap<OwnedRow, GroupState> = AggregateHashMap::default();
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
    Ok(groups)
}

fn merge_group_maps(
    groups: &mut AggregateHashMap<OwnedRow, GroupState>,
    partial: AggregateHashMap<OwnedRow, GroupState>,
) {
    for (key, partial_group) in partial {
        match groups.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(partial_group);
            }
            Entry::Occupied(mut entry) => {
                let group = entry.get_mut();
                for (accumulator, partial_accumulator) in group
                    .accumulators
                    .iter_mut()
                    .zip(partial_group.accumulators)
                {
                    accumulator.merge(partial_accumulator);
                }
            }
        }
    }
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

fn can_use_single_key_count_sum_min_max_path(
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> bool {
    group_by.len() == 1
        && matches!(
            aggregates,
            [
                AggregateExpr::CountStar,
                AggregateExpr::Sum(_),
                AggregateExpr::Min(_),
                AggregateExpr::Max(_)
            ]
        )
}

fn collect_single_key_count_sum_min_max_groups(
    mut stream: SendableBatchStream,
    fragments: usize,
    group_by: &[String],
    aggregates: &[AggregateExpr],
) -> Result<AggregateMetrics> {
    let (
        AggregateExpr::CountStar,
        AggregateExpr::Sum(sum_column),
        AggregateExpr::Min(min_column),
        AggregateExpr::Max(max_column),
    ) = (
        &aggregates[0],
        &aggregates[1],
        &aggregates[2],
        &aggregates[3],
    )
    else {
        unreachable!("single-key count/sum/min/max fast path precondition");
    };
    let mut group_index = SingleKeyCountSumMinMaxIndex::Unset;
    let mut groups = Vec::<SingleKeyCountSumMinMaxGroup>::new();
    let mut bound_plan = None;
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

        let bound = match &bound_plan {
            Some(bound) => bound,
            None => {
                bound_plan = Some(BoundSingleKeyCountSumMinMaxPlan::bind(
                    &batch,
                    &group_by[0],
                    sum_column,
                    min_column,
                    max_column,
                )?);
                bound_plan
                    .as_ref()
                    .expect("bound single-key count/sum/min/max plan")
            }
        };
        let key_column = batch.column(bound.key);
        let sum_column_ref = batch.column(bound.sum);
        let min_column_ref = batch.column(bound.min);
        let max_column_ref = batch.column(bound.max);
        let Some(sum_values) = Int64LikeArray::new(sum_column_ref) else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        let DataType::Decimal128(decimal_precision, decimal_scale) = min_column_ref.data_type()
        else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        let Some(min_values) = min_column_ref.as_any().downcast_ref::<Decimal128Array>() else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
        let Some(max_values) = max_column_ref.as_any().downcast_ref::<Date32Array>() else {
            return collect_grouped_aggregates_generic(
                stream,
                fragments,
                group_by,
                aggregates,
                Some(batch),
                Some(metrics),
            );
        };
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

        match &mut group_index {
            SingleKeyCountSumMinMaxIndex::Int32 {
                groups: index,
                null_group,
            } => {
                let key_values = key_column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 group key");
                if key_values.null_count() == 0
                    && sum_values.null_count() == 0
                    && min_values.null_count() == 0
                    && max_values.null_count() == 0
                {
                    for row in 0..batch.num_rows() {
                        let group_id = count_sum_min_max_group_id_for_i32(
                            index,
                            key_values.value(row),
                            &mut groups,
                            *decimal_precision,
                            *decimal_scale,
                        );
                        groups[group_id].update_non_null(&sum_values, min_values, max_values, row);
                    }
                } else {
                    for row in 0..batch.num_rows() {
                        let group_id = if key_values.is_null(row) {
                            count_sum_min_max_group_id_for_null(
                                null_group,
                                &mut groups,
                                GroupValue::Int64(None),
                                *decimal_precision,
                                *decimal_scale,
                            )
                        } else {
                            count_sum_min_max_group_id_for_i32(
                                index,
                                key_values.value(row),
                                &mut groups,
                                *decimal_precision,
                                *decimal_scale,
                            )
                        };
                        groups[group_id].update(&sum_values, min_values, max_values, row);
                    }
                }
            }
            SingleKeyCountSumMinMaxIndex::Int64 {
                groups: index,
                null_group,
            } => {
                let key_values = key_column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 group key");
                if key_values.null_count() == 0
                    && sum_values.null_count() == 0
                    && min_values.null_count() == 0
                    && max_values.null_count() == 0
                {
                    for row in 0..batch.num_rows() {
                        let group_id = count_sum_min_max_group_id_for_i64(
                            index,
                            key_values.value(row),
                            &mut groups,
                            *decimal_precision,
                            *decimal_scale,
                        );
                        groups[group_id].update_non_null(&sum_values, min_values, max_values, row);
                    }
                } else {
                    for row in 0..batch.num_rows() {
                        let group_id = if key_values.is_null(row) {
                            count_sum_min_max_group_id_for_null(
                                null_group,
                                &mut groups,
                                GroupValue::Int64(None),
                                *decimal_precision,
                                *decimal_scale,
                            )
                        } else {
                            count_sum_min_max_group_id_for_i64(
                                index,
                                key_values.value(row),
                                &mut groups,
                                *decimal_precision,
                                *decimal_scale,
                            )
                        };
                        groups[group_id].update(&sum_values, min_values, max_values, row);
                    }
                }
            }
            SingleKeyCountSumMinMaxIndex::UInt64 {
                groups: index,
                null_group,
            } => {
                let key_values = key_column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 group key");
                if key_values.null_count() == 0
                    && sum_values.null_count() == 0
                    && min_values.null_count() == 0
                    && max_values.null_count() == 0
                {
                    for row in 0..batch.num_rows() {
                        let group_id = count_sum_min_max_group_id_for_u64(
                            index,
                            key_values.value(row),
                            &mut groups,
                            *decimal_precision,
                            *decimal_scale,
                        );
                        groups[group_id].update_non_null(&sum_values, min_values, max_values, row);
                    }
                } else {
                    for row in 0..batch.num_rows() {
                        let group_id = if key_values.is_null(row) {
                            count_sum_min_max_group_id_for_null(
                                null_group,
                                &mut groups,
                                GroupValue::UInt64(None),
                                *decimal_precision,
                                *decimal_scale,
                            )
                        } else {
                            count_sum_min_max_group_id_for_u64(
                                index,
                                key_values.value(row),
                                &mut groups,
                                *decimal_precision,
                                *decimal_scale,
                            )
                        };
                        groups[group_id].update(&sum_values, min_values, max_values, row);
                    }
                }
            }
            SingleKeyCountSumMinMaxIndex::Utf8 { .. } => {
                for row in 0..batch.num_rows() {
                    let group_id = group_index.group_id(
                        key_column,
                        row,
                        &mut groups,
                        *decimal_precision,
                        *decimal_scale,
                    )?;
                    groups[group_id].update(&sum_values, min_values, max_values, row);
                }
            }
            SingleKeyCountSumMinMaxIndex::Unset => {
                unreachable!("group index type should be initialized")
            }
        }
    }

    metrics.groups = finish_single_key_count_sum_min_max_groups(group_index, groups, aggregates);
    Ok(metrics)
}

fn finish_single_key_count_sum_min_max_groups(
    group_index: SingleKeyCountSumMinMaxIndex,
    groups: Vec<SingleKeyCountSumMinMaxGroup>,
    aggregates: &[AggregateExpr],
) -> Vec<GroupAggregateResult> {
    match group_index {
        SingleKeyCountSumMinMaxIndex::Int32 {
            groups: index,
            null_group,
        } => finish_ordered_copy_key_groups(index.iter(), null_group, groups, aggregates),
        SingleKeyCountSumMinMaxIndex::Int64 {
            groups: index,
            null_group,
        } => finish_ordered_copy_key_groups(index.iter(), null_group, groups, aggregates),
        SingleKeyCountSumMinMaxIndex::UInt64 {
            groups: index,
            null_group,
        } => finish_ordered_copy_key_groups(index.iter(), null_group, groups, aggregates),
        _ => {
            let mut group_results = groups
                .into_iter()
                .map(|group| group.finish(aggregates))
                .collect::<Vec<_>>();
            group_results.sort_by(|left, right| compare_group_keys(&left.keys, &right.keys));
            group_results
        }
    }
}

fn finish_ordered_copy_key_groups<K, I>(
    index: I,
    null_group: Option<usize>,
    groups: Vec<SingleKeyCountSumMinMaxGroup>,
    aggregates: &[AggregateExpr],
) -> Vec<GroupAggregateResult>
where
    K: Copy + Ord,
    I: Iterator<Item = (K, usize)>,
{
    let mut groups = groups.into_iter().map(Some).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(groups.len());
    if let Some(group_id) = null_group
        && let Some(group) = groups[group_id].take()
    {
        output.push(group.finish(aggregates));
    }
    let mut keyed = index.collect::<Vec<_>>();
    keyed.sort_by_key(|(key, _)| *key);
    for (_, group_id) in keyed {
        if let Some(group) = groups[group_id].take() {
            output.push(group.finish(aggregates));
        }
    }
    output
}

struct BoundSingleKeyCountSumMinMaxPlan {
    key: usize,
    sum: usize,
    min: usize,
    max: usize,
}

impl BoundSingleKeyCountSumMinMaxPlan {
    fn bind(
        batch: &RecordBatch,
        key_column: &str,
        sum_column: &str,
        min_column: &str,
        max_column: &str,
    ) -> Result<Self> {
        Ok(Self {
            key: column_index(batch, key_column)?,
            sum: column_index(batch, sum_column)?,
            min: column_index(batch, min_column)?,
            max: column_index(batch, max_column)?,
        })
    }
}

enum Int64LikeArray<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
}

impl<'a> Int64LikeArray<'a> {
    fn new(array: &'a ArrayRef) -> Option<Self> {
        match array.data_type() {
            DataType::Int32 => Some(Self::Int32(array.as_any().downcast_ref::<Int32Array>()?)),
            DataType::Int64 => Some(Self::Int64(array.as_any().downcast_ref::<Int64Array>()?)),
            _ => None,
        }
    }

    fn value(&self, row: usize) -> Option<i64> {
        match self {
            Self::Int32(values) => values.is_valid(row).then(|| i64::from(values.value(row))),
            Self::Int64(values) => values.is_valid(row).then(|| values.value(row)),
        }
    }

    fn value_non_null(&self, row: usize) -> i64 {
        match self {
            Self::Int32(values) => i64::from(values.value(row)),
            Self::Int64(values) => values.value(row),
        }
    }

    fn null_count(&self) -> usize {
        match self {
            Self::Int32(values) => values.null_count(),
            Self::Int64(values) => values.null_count(),
        }
    }
}

enum SingleKeyCountSumMinMaxIndex {
    Unset,
    Utf8 {
        groups: AggregateHashMap<String, usize>,
        null_group: Option<usize>,
    },
    Int32 {
        groups: AdaptiveCopyGroupIndex<i32>,
        null_group: Option<usize>,
    },
    Int64 {
        groups: AdaptiveCopyGroupIndex<i64>,
        null_group: Option<usize>,
    },
    UInt64 {
        groups: AdaptiveCopyGroupIndex<u64>,
        null_group: Option<usize>,
    },
}

impl SingleKeyCountSumMinMaxIndex {
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
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            DataType::Int64 => Self::Int64 {
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            DataType::UInt64 => Self::UInt64 {
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            _ => unreachable!("fast path key type precondition"),
        };
    }

    fn group_id(
        &mut self,
        key_column: &ArrayRef,
        row: usize,
        groups_out: &mut Vec<SingleKeyCountSumMinMaxGroup>,
        decimal_precision: u8,
        decimal_scale: i8,
    ) -> Result<usize> {
        match self {
            Self::Utf8 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 group key");
                if values.is_null(row) {
                    return Ok(count_sum_min_max_group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::Utf8(None),
                        decimal_precision,
                        decimal_scale,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(key).copied() {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key.to_string(), group_id);
                groups_out.push(SingleKeyCountSumMinMaxGroup::new(
                    GroupValue::Utf8(Some(key.to_string())),
                    decimal_precision,
                    decimal_scale,
                ));
                Ok(group_id)
            }
            Self::Int32 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 group key");
                if values.is_null(row) {
                    return Ok(count_sum_min_max_group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::Int64(None),
                        decimal_precision,
                        decimal_scale,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(key) {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(SingleKeyCountSumMinMaxGroup::new(
                    GroupValue::Int64(Some(i64::from(key))),
                    decimal_precision,
                    decimal_scale,
                ));
                Ok(group_id)
            }
            Self::Int64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 group key");
                if values.is_null(row) {
                    return Ok(count_sum_min_max_group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::Int64(None),
                        decimal_precision,
                        decimal_scale,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(key) {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(SingleKeyCountSumMinMaxGroup::new(
                    GroupValue::Int64(Some(key)),
                    decimal_precision,
                    decimal_scale,
                ));
                Ok(group_id)
            }
            Self::UInt64 { groups, null_group } => {
                let values = key_column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("UInt64 group key");
                if values.is_null(row) {
                    return Ok(count_sum_min_max_group_id_for_null(
                        null_group,
                        groups_out,
                        GroupValue::UInt64(None),
                        decimal_precision,
                        decimal_scale,
                    ));
                }
                let key = values.value(row);
                if let Some(group_id) = groups.get(key) {
                    return Ok(group_id);
                }
                let group_id = groups_out.len();
                groups.insert(key, group_id);
                groups_out.push(SingleKeyCountSumMinMaxGroup::new(
                    GroupValue::UInt64(Some(key)),
                    decimal_precision,
                    decimal_scale,
                ));
                Ok(group_id)
            }
            Self::Unset => unreachable!("group index type should be initialized"),
        }
    }
}

fn count_sum_min_max_group_id_for_null(
    null_group: &mut Option<usize>,
    groups: &mut Vec<SingleKeyCountSumMinMaxGroup>,
    key: GroupValue,
    decimal_precision: u8,
    decimal_scale: i8,
) -> usize {
    if let Some(group_id) = *null_group {
        return group_id;
    }
    let group_id = groups.len();
    groups.push(SingleKeyCountSumMinMaxGroup::new(
        key,
        decimal_precision,
        decimal_scale,
    ));
    *null_group = Some(group_id);
    group_id
}

fn count_sum_min_max_group_id_for_i32(
    index: &mut AdaptiveCopyGroupIndex<i32>,
    key: i32,
    groups: &mut Vec<SingleKeyCountSumMinMaxGroup>,
    decimal_precision: u8,
    decimal_scale: i8,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(SingleKeyCountSumMinMaxGroup::new(
        GroupValue::Int64(Some(i64::from(key))),
        decimal_precision,
        decimal_scale,
    ));
    group_id
}

fn count_sum_min_max_group_id_for_i64(
    index: &mut AdaptiveCopyGroupIndex<i64>,
    key: i64,
    groups: &mut Vec<SingleKeyCountSumMinMaxGroup>,
    decimal_precision: u8,
    decimal_scale: i8,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(SingleKeyCountSumMinMaxGroup::new(
        GroupValue::Int64(Some(key)),
        decimal_precision,
        decimal_scale,
    ));
    group_id
}

fn count_sum_min_max_group_id_for_u64(
    index: &mut AdaptiveCopyGroupIndex<u64>,
    key: u64,
    groups: &mut Vec<SingleKeyCountSumMinMaxGroup>,
    decimal_precision: u8,
    decimal_scale: i8,
) -> usize {
    if let Some(group_id) = index.get(key) {
        return group_id;
    }
    let group_id = groups.len();
    index.insert(key, group_id);
    groups.push(SingleKeyCountSumMinMaxGroup::new(
        GroupValue::UInt64(Some(key)),
        decimal_precision,
        decimal_scale,
    ));
    group_id
}

struct SingleKeyCountSumMinMaxGroup {
    key: GroupValue,
    count: u64,
    sum: i64,
    sum_count: u64,
    min_decimal: Option<i128>,
    decimal_precision: u8,
    decimal_scale: i8,
    max_date32: Option<i32>,
}

impl SingleKeyCountSumMinMaxGroup {
    fn new(key: GroupValue, decimal_precision: u8, decimal_scale: i8) -> Self {
        Self {
            key,
            count: 0,
            sum: 0,
            sum_count: 0,
            min_decimal: None,
            decimal_precision,
            decimal_scale,
            max_date32: None,
        }
    }

    fn update(
        &mut self,
        sum_values: &Int64LikeArray<'_>,
        min_values: &Decimal128Array,
        max_values: &Date32Array,
        row: usize,
    ) {
        self.count += 1;
        if let Some(value) = sum_values.value(row) {
            self.sum += value;
            self.sum_count += 1;
        }
        if min_values.is_valid(row) {
            let value = min_values.value(row);
            self.min_decimal = Some(match self.min_decimal {
                Some(current) => current.min(value),
                None => value,
            });
        }
        if max_values.is_valid(row) {
            let value = max_values.value(row);
            self.max_date32 = Some(match self.max_date32 {
                Some(current) => current.max(value),
                None => value,
            });
        }
    }

    fn update_non_null(
        &mut self,
        sum_values: &Int64LikeArray<'_>,
        min_values: &Decimal128Array,
        max_values: &Date32Array,
        row: usize,
    ) {
        self.count += 1;
        self.sum += sum_values.value_non_null(row);
        self.sum_count += 1;

        let min_value = min_values.value(row);
        self.min_decimal = Some(match self.min_decimal {
            Some(current) => current.min(min_value),
            None => min_value,
        });

        let max_value = max_values.value(row);
        self.max_date32 = Some(match self.max_date32 {
            Some(current) => current.max(max_value),
            None => max_value,
        });
    }

    fn merge_partial_values(&mut self, values: &[AggregateResult]) -> Result<()> {
        let [
            AggregateResult {
                value: AggregateValue::Count(count),
                ..
            },
            AggregateResult {
                value: sum_value, ..
            },
            AggregateResult {
                value: min_value, ..
            },
            AggregateResult {
                value: max_value, ..
            },
        ] = values
        else {
            return Err(DodamError::InvalidAggregate(
                "partial count/sum/min/max group shape mismatch".to_string(),
            ));
        };
        self.count = self.count.saturating_add(*count);
        match sum_value {
            AggregateValue::Int64(Some(sum)) => {
                self.sum += *sum;
                self.sum_count += 1;
            }
            AggregateValue::Int64(None) => {}
            _ => {
                return Err(DodamError::InvalidAggregate(
                    "partial count/sum/min/max sum type mismatch".to_string(),
                ));
            }
        }
        match min_value {
            AggregateValue::Decimal128(Some(value), _, _) => {
                self.min_decimal = Some(match self.min_decimal {
                    Some(current) => current.min(*value),
                    None => *value,
                });
            }
            AggregateValue::Decimal128(None, _, _) => {}
            _ => {
                return Err(DodamError::InvalidAggregate(
                    "partial count/sum/min/max min type mismatch".to_string(),
                ));
            }
        }
        match max_value {
            AggregateValue::Date32(Some(value)) => {
                self.max_date32 = Some(match self.max_date32 {
                    Some(current) => current.max(*value),
                    None => *value,
                });
            }
            AggregateValue::Date32(None) => {}
            _ => {
                return Err(DodamError::InvalidAggregate(
                    "partial count/sum/min/max max type mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn finish(self, aggregates: &[AggregateExpr]) -> GroupAggregateResult {
        GroupAggregateResult {
            keys: vec![self.key],
            values: vec![
                AggregateResult {
                    expr: aggregates[0].clone(),
                    value: AggregateValue::Count(self.count),
                },
                AggregateResult {
                    expr: aggregates[1].clone(),
                    value: if self.sum_count == 0 {
                        AggregateValue::Int64(None)
                    } else {
                        AggregateValue::Int64(Some(self.sum))
                    },
                },
                AggregateResult {
                    expr: aggregates[2].clone(),
                    value: AggregateValue::Decimal128(
                        self.min_decimal,
                        self.decimal_precision,
                        self.decimal_scale,
                    ),
                },
                AggregateResult {
                    expr: aggregates[3].clone(),
                    value: AggregateValue::Date32(self.max_date32),
                },
            ],
        }
    }
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
        groups: AdaptiveCopyGroupIndex<i32>,
        null_group: Option<usize>,
    },
    Int64 {
        groups: AdaptiveCopyGroupIndex<i64>,
        null_group: Option<usize>,
    },
    UInt64 {
        groups: AdaptiveCopyGroupIndex<u64>,
        null_group: Option<usize>,
    },
}

enum AdaptiveCopyGroupIndex<K> {
    Small(Vec<(K, usize)>),
    Hash(AggregateHashMap<K, usize>),
}

impl<K> Default for AdaptiveCopyGroupIndex<K> {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

impl<K> AdaptiveCopyGroupIndex<K>
where
    K: Copy + Eq + std::hash::Hash,
{
    fn get(&self, key: K) -> Option<usize> {
        match self {
            Self::Small(groups) => groups
                .iter()
                .find_map(|(candidate, group_id)| (*candidate == key).then_some(*group_id)),
            Self::Hash(groups) => groups.get(&key).copied(),
        }
    }

    fn insert(&mut self, key: K, group_id: usize) {
        match self {
            Self::Small(groups) if groups.len() < small_group_linear_limit() => {
                groups.push((key, group_id));
            }
            Self::Small(groups) => {
                let mut hash = groups.drain(..).collect::<AggregateHashMap<_, _>>();
                hash.insert(key, group_id);
                *self = Self::Hash(hash);
            }
            Self::Hash(groups) => {
                groups.insert(key, group_id);
            }
        }
    }

    fn iter(&self) -> AdaptiveCopyGroupIndexIter<'_, K> {
        match self {
            Self::Small(groups) => AdaptiveCopyGroupIndexIter::Small(groups.iter()),
            Self::Hash(groups) => AdaptiveCopyGroupIndexIter::Hash(groups.iter()),
        }
    }
}

enum AdaptiveCopyGroupIndexIter<'a, K> {
    Small(std::slice::Iter<'a, (K, usize)>),
    Hash(std::collections::hash_map::Iter<'a, K, usize>),
}

impl<K> Iterator for AdaptiveCopyGroupIndexIter<'_, K>
where
    K: Copy,
{
    type Item = (K, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Small(iter) => iter.next().map(|(key, group_id)| (*key, *group_id)),
            Self::Hash(iter) => iter.next().map(|(key, group_id)| (*key, *group_id)),
        }
    }
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
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            DataType::Int64 => Self::Int64 {
                groups: AdaptiveCopyGroupIndex::default(),
                null_group: None,
            },
            DataType::UInt64 => Self::UInt64 {
                groups: AdaptiveCopyGroupIndex::default(),
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
                if let Some(group_id) = groups.get(key) {
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
                if let Some(group_id) = groups.get(key) {
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
                if let Some(group_id) = groups.get(key) {
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
    Decimal128 {
        expr: AggregateExpr,
        values: &'a Decimal128Array,
        precision: u8,
        scale: i8,
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
            AggregateExpr::CountDistinct(_) => {
                Err(DodamError::InvalidAggregate(aggregate.to_string()))
            }
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
                    DataType::Decimal128(precision, scale) => Ok(FastAggregateInput::Decimal128 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .expect("Decimal128 numeric input"),
                        precision: *precision,
                        scale: *scale,
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
                    DataType::Decimal128(precision, scale) => Ok(FastAggregateInput::Decimal128 {
                        expr,
                        values: values
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .expect("Decimal128 min/max input"),
                        precision: *precision,
                        scale: *scale,
                    }),
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
            FastAggregateInput::Decimal128 { expr, .. }
                if matches!(expr, AggregateExpr::Sum(_) | AggregateExpr::Avg(_)) =>
            {
                numeric_fast_state(expr, true)
            }
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
            FastAggregateInput::Decimal128 {
                expr,
                precision,
                scale,
                ..
            } => min_max_fast_state(expr, AggregateValue::Decimal128(None, *precision, *scale)),
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
            FastAggregateInput::Decimal128 {
                values,
                precision,
                scale,
                ..
            } if values.is_valid(row) => match self {
                Self::SumFloat { sum, count, .. } | Self::AvgFloat { sum, count, .. } => {
                    *sum += values.value(row) as f64 / decimal_scale_factor(*scale);
                    *count += 1;
                }
                Self::MinMax {
                    expr, value: state, ..
                } => {
                    update_min_max_value(
                        expr,
                        state,
                        AggregateValue::Decimal128(Some(values.value(row)), *precision, *scale),
                    );
                }
                _ => {}
            },
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
            Some(AggregateValue::Decimal128(Some(current), _, _)),
            AggregateValue::Decimal128(Some(candidate), _, _),
        ) => candidate < current,
        (
            AggregateExpr::Max(_),
            Some(AggregateValue::Decimal128(Some(current), _, _)),
            AggregateValue::Decimal128(Some(candidate), _, _),
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
            DataType::Decimal128(precision, scale) => {
                let values = column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128 data type");
                Ok(GroupValue::Decimal128(
                    values.is_valid(row).then(|| values.value(row)),
                    *precision,
                    *scale,
                ))
            }
            DataType::Date32 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Date32 data type");
                Ok(GroupValue::Date32(
                    values.is_valid(row).then(|| values.value(row)),
                ))
            }
            DataType::Date64 => {
                let values = column
                    .as_any()
                    .downcast_ref::<Date64Array>()
                    .expect("Date64 data type");
                Ok(GroupValue::Date64(
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
            (
                GroupValue::Decimal128(left, left_precision, left_scale),
                GroupValue::Decimal128(right, right_precision, right_scale),
            ) => (left_scale, left_precision, left).cmp(&(right_scale, right_precision, right)),
            (GroupValue::Date32(left), GroupValue::Date32(right)) => left.cmp(right),
            (GroupValue::Date64(left), GroupValue::Date64(right)) => left.cmp(right),
            (GroupValue::Utf8(left), GroupValue::Utf8(right)) => left.cmp(right),
            (GroupValue::Int64(_), _) => std::cmp::Ordering::Less,
            (GroupValue::UInt64(_), GroupValue::Int64(_)) => std::cmp::Ordering::Greater,
            (GroupValue::UInt64(_), _) => std::cmp::Ordering::Less,
            (GroupValue::Decimal128(_, _, _), GroupValue::Int64(_) | GroupValue::UInt64(_)) => {
                std::cmp::Ordering::Greater
            }
            (GroupValue::Decimal128(_, _, _), _) => std::cmp::Ordering::Less,
            (
                GroupValue::Date32(_),
                GroupValue::Int64(_) | GroupValue::UInt64(_) | GroupValue::Decimal128(_, _, _),
            ) => std::cmp::Ordering::Greater,
            (GroupValue::Date32(_), _) => std::cmp::Ordering::Less,
            (
                GroupValue::Date64(_),
                GroupValue::Int64(_)
                | GroupValue::UInt64(_)
                | GroupValue::Decimal128(_, _, _)
                | GroupValue::Date32(_),
            ) => std::cmp::Ordering::Greater,
            (GroupValue::Date64(_), _) => std::cmp::Ordering::Less,
            (GroupValue::Utf8(_), _) => std::cmp::Ordering::Greater,
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
    CountDistinct {
        expr: AggregateExpr,
        values: HashSet<GroupValue>,
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
            AggregateExpr::CountDistinct(_) => Self::CountDistinct {
                expr,
                values: HashSet::new(),
            },
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
            Self::CountDistinct { expr, values } => {
                let column = aggregate_column(batch, expr)?;
                for row in 0..column.len() {
                    if let Some(value) = distinct_group_value(column, row)? {
                        values.insert(value);
                    }
                }
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
            Self::CountDistinct { expr, values } => {
                let column =
                    column.ok_or_else(|| DodamError::InvalidAggregate(expr.to_string()))?;
                if let Some(value) = distinct_group_value(column, row)? {
                    values.insert(value);
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
            Self::CountDistinct { expr, values } => AggregateResult {
                expr,
                value: AggregateValue::Count(values.len() as u64),
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

    fn merge(&mut self, other: Self) {
        match (self, other) {
            (Self::CountStar { count }, Self::CountStar { count: other }) => {
                *count += other;
            }
            (Self::Count { count, .. }, Self::Count { count: other, .. }) => {
                *count += other;
            }
            (Self::CountDistinct { values, .. }, Self::CountDistinct { values: other, .. }) => {
                values.extend(other);
            }
            (Self::Sum { state, .. }, Self::Sum { state: other, .. })
            | (Self::Avg { state, .. }, Self::Avg { state: other, .. }) => {
                state.merge(other);
            }
            (Self::Min { expr, state }, Self::Min { state: other, .. })
            | (Self::Max { expr, state }, Self::Max { state: other, .. }) => {
                state.merge(other, expr);
            }
            _ => {}
        }
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
    fn add_i64(&mut self, value: i64) {
        self.output.get_or_insert(NumericOutput::Int64);
        self.sum_i64 += value;
        self.count += 1;
    }

    fn add_f64(&mut self, value: f64) {
        self.output.get_or_insert(NumericOutput::Float64);
        self.sum_f64 += value;
        self.count += 1;
    }

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
            DataType::Decimal128(_, scale) => {
                self.output.get_or_insert(NumericOutput::Float64);
                let values = column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128 data type");
                let scale = decimal_scale_factor(*scale);
                for value in values.iter().flatten() {
                    self.sum_f64 += value as f64 / scale;
                    self.count += 1;
                }
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
            DataType::Decimal128(_, scale) => {
                self.output.get_or_insert(NumericOutput::Float64);
                let values = column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128 data type");
                if values.is_valid(row) {
                    self.sum_f64 += values.value(row) as f64 / decimal_scale_factor(*scale);
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

    fn merge(&mut self, other: Self) {
        match other.output {
            Some(NumericOutput::Int64) => {
                self.output.get_or_insert(NumericOutput::Int64);
                self.sum_i64 += other.sum_i64;
            }
            Some(NumericOutput::Float64) => {
                self.output.get_or_insert(NumericOutput::Float64);
                self.sum_f64 += other.sum_f64;
            }
            None => {}
        }
        self.count += other.count;
    }
}

fn decimal_scale_factor(scale: i8) -> f64 {
    10_f64.powi(i32::from(scale))
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
            DataType::Decimal128(precision, scale) => {
                let values = column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128 data type");
                let value = if replace(std::cmp::Ordering::Less, std::cmp::Ordering::Equal) {
                    values.iter().flatten().min()
                } else {
                    values.iter().flatten().max()
                };
                if let Some(value) = value {
                    self.update_decimal128(value, *precision, *scale, &replace);
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
            DataType::Decimal128(precision, scale) => {
                let values = column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128 data type");
                self.update_decimal128(values.value(row), *precision, *scale, &replace);
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

    fn update_decimal128(
        &mut self,
        candidate: i128,
        precision: u8,
        scale: i8,
        replace: &impl Fn(std::cmp::Ordering, std::cmp::Ordering) -> bool,
    ) {
        match &self.value {
            Some(AggregateValue::Decimal128(Some(current), _, _))
                if !replace(candidate.cmp(current), std::cmp::Ordering::Equal) => {}
            _ => {
                self.value = Some(AggregateValue::Decimal128(
                    Some(candidate),
                    precision,
                    scale,
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

    fn merge(&mut self, other: Self, expr: &AggregateExpr) {
        if let Some(candidate) = other.value {
            update_min_max_value(expr, &mut self.value, candidate);
        }
    }
}

fn aggregate_column<'a>(batch: &'a RecordBatch, expr: &AggregateExpr) -> Result<&'a ArrayRef> {
    let Some(column) = expr.referenced_column() else {
        return Err(DodamError::InvalidAggregate(expr.to_string()));
    };
    Ok(batch.column(column_index(batch, column)?))
}

fn distinct_group_value(column: &ArrayRef, row: usize) -> Result<Option<GroupValue>> {
    if column.is_null(row) {
        return Ok(None);
    }
    match column.data_type() {
        DataType::Int32 => Ok(Some(GroupValue::Int64(Some(i64::from(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 distinct input")
                .value(row),
        ))))),
        DataType::Int64 => Ok(Some(GroupValue::Int64(Some(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 distinct input")
                .value(row),
        )))),
        DataType::UInt64 => Ok(Some(GroupValue::UInt64(Some(
            column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 distinct input")
                .value(row),
        )))),
        DataType::Decimal128(precision, scale) => Ok(Some(GroupValue::Decimal128(
            Some(
                column
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("Decimal128 distinct input")
                    .value(row),
            ),
            *precision,
            *scale,
        ))),
        DataType::Date32 => Ok(Some(GroupValue::Date32(Some(
            column
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Date32 distinct input")
                .value(row),
        )))),
        DataType::Date64 => Ok(Some(GroupValue::Date64(Some(
            column
                .as_any()
                .downcast_ref::<Date64Array>()
                .expect("Date64 distinct input")
                .value(row),
        )))),
        DataType::Utf8 => Ok(Some(GroupValue::Utf8(Some(
            column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 distinct input")
                .value(row)
                .to_string(),
        )))),
        data_type => {
            unsupported_aggregate_type(&AggregateExpr::CountDistinct("*".to_string()), data_type)
        }
    }
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
        AggregateExpr::CountStar | AggregateExpr::Count(_) | AggregateExpr::CountDistinct(_) => {
            "count"
        }
        AggregateExpr::Sum(_) => "sum",
        AggregateExpr::Avg(_) => "avg",
        AggregateExpr::Min(_) => "min",
        AggregateExpr::Max(_) => "max",
    }
}
